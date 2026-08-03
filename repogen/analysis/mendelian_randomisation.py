"""Cis-eQTL Mendelian Randomisation (Branch C).

Implements native cis-MR using eQTL instruments, including:
- Wald ratio (k=1), IVW fixed/random-effects (k≥2)
- MR-Egger and weighted-median sensitivity analyses (k≥3)
- Steiger directionality test
- Colocalisation via native coloc.abf reimplementation
- Directional drug matching with confidence tiering
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
from statsmodels.stats.multitest import multipletests

from repogen.config.schema import (
    EQTLSourceConfig,
    MRConfig,
    MRDrugMatchConfig,
    ReferenceConfig,
)
from repogen.data.mhc_annotation import add_mhc_flag, load_mhc_gene_annotation
from repogen.data.schemas import GWASMetadata
from repogen.utils.constants import mhc_interval
from repogen.utils.logging import setup_logging
from repogen.utils.provenance import plink_provenance

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Direction lookup constants
# ---------------------------------------------------------------------------

DOWNREGULATING_TYPES = frozenset({"inhibitor", "antagonist", "blocker", "negative_modulator"})
UPREGULATING_TYPES = frozenset({"agonist", "activator", "positive_modulator"})
AMBIGUOUS_TYPES = frozenset({"partial_agonist", "modulator", "other"})
# Direction-inferable = the MR sign can be interpreted against the interaction sign.
# This is distinct from "has a curated mechanism_of_action text".
INFERABLE_TYPES = DOWNREGULATING_TYPES | UPREGULATING_TYPES

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


# ---------------------------------------------------------------------------
# MAF resolution helpers (Branch-C only, in-memory)
# ---------------------------------------------------------------------------


def _normalize_to_maf(af: pd.Series) -> pd.Series:
    """Coerce allele frequencies to minor-allele scale (0, 0.5]."""
    af = pd.to_numeric(af, errors="coerce")
    af = af.where((af > 0) & (af < 1))
    return af.where(af <= 0.5, 1.0 - af)


def _resolve_maf(
    gwas_df: pd.DataFrame,
    ref_freq_path: Path | None = None,
) -> tuple[pd.Series, dict[str, int]]:
    """Derive per-SNP MAF from available GWAS columns.

    Precedence (strict; later steps only fill still-missing rows):

    1. Existing ``MAF`` column (normalized to minor-allele scale).
    2. ``FCON`` - control allele frequency (best population proxy
       for case-control studies).
    3. Weighted case/control AF from ``FCAS``, ``FCON``, ``N_CAS``,
       ``N_CON``.
    4. Reference ``.frq`` lookup by SNP (for still-unresolved rows).
    5. Remaining NaN left unresolved (formulas apply 0.3 per-entry).

    Returns ``(maf_series, counters)`` aligned to *gwas_df* index.
    """
    maf = pd.Series(np.nan, index=gwas_df.index, dtype=float)
    counters: dict[str, int] = {}

    if "MAF" in gwas_df.columns:
        existing = _normalize_to_maf(gwas_df["MAF"])
        filled = existing.notna() & maf.isna()
        maf = maf.where(~filled, existing)
        counters["input_maf"] = int(filled.sum())

    if "FCON" in gwas_df.columns:
        fcon = _normalize_to_maf(gwas_df["FCON"])
        filled = fcon.notna() & maf.isna()
        maf = maf.where(~filled, fcon)
        counters["fcon"] = int(filled.sum())

    has_weighted = all(
        c in gwas_df.columns for c in ("FCAS", "FCON", "N_CAS", "N_CON")
    )
    if has_weighted:
        n_cas = pd.to_numeric(gwas_df["N_CAS"], errors="coerce")
        n_con = pd.to_numeric(gwas_df["N_CON"], errors="coerce")
        fcas = pd.to_numeric(gwas_df["FCAS"], errors="coerce")
        fcon_raw = pd.to_numeric(gwas_df["FCON"], errors="coerce")
        total_n = n_cas + n_con
        weighted = (fcas * n_cas + fcon_raw * n_con) / total_n
        weighted_maf = _normalize_to_maf(weighted)
        filled = weighted_maf.notna() & maf.isna()
        maf = maf.where(~filled, weighted_maf)
        counters["weighted_af"] = int(filled.sum())

    if ref_freq_path is not None and ref_freq_path.exists() and "SNP" in gwas_df.columns:
        unresolved_mask = maf.isna()
        n_unresolved = int(unresolved_mask.sum())
        if n_unresolved > 0:
            try:
                freq_df = pd.read_csv(
                    ref_freq_path, sep=r"\s+", dtype={"SNP": str},
                    usecols=["SNP", "MAF"],
                )
                freq_df = freq_df.drop_duplicates(subset="SNP")
                unresolved_snps = gwas_df.loc[unresolved_mask, "SNP"]
                lookup = unresolved_snps.to_frame().merge(
                    freq_df, on="SNP", how="left",
                )
                ref_maf = _normalize_to_maf(lookup.set_index(unresolved_snps.index)["MAF"])
                filled = ref_maf.notna()
                maf.loc[unresolved_mask] = maf.loc[unresolved_mask].where(
                    ~filled, ref_maf,
                )
                counters["ref_frq"] = int(filled.sum())
            except Exception as exc:
                logger.warning("Reference .frq MAF lookup failed: %s", exc)
                counters["ref_frq"] = 0

    counters["unresolved"] = int(maf.isna().sum())
    return maf, counters


# ---------------------------------------------------------------------------
# Reference frequency file
# ---------------------------------------------------------------------------


def _detect_plink_binary() -> Path:
    """Find PLINK 1.9 binary on PATH."""
    plink = shutil.which("plink")
    if plink is None:
        plink = shutil.which("plink1.9")
    if plink is None:
        raise FileNotFoundError(
            "PLINK 1.9 not found on PATH. Install via conda: conda install -c bioconda plink=1.9"
        )
    return Path(plink)


def ensure_ref_freq(bfile_full_path: Path, plink_binary: Path) -> Path:
    """Ensure reference .frq file exists; generate via PLINK if missing.

    CRITICAL: Do NOT use bfile_full_path.with_suffix(".frq") - Python's
    with_suffix() replaces the last dot-suffix, which silently corrupts
    dotted prefixes (e.g. ``data_maf0.01`` -> ``data_maf0.frq``).
    """
    frq_path = Path(str(bfile_full_path) + ".frq")
    if not frq_path.exists():
        logger.info("Generating reference frequency file: %s", frq_path)
        result = subprocess.run(
            [str(plink_binary), "--bfile", str(bfile_full_path),
             "--freq", "--out", str(bfile_full_path)],
            check=True, capture_output=True, text=True,
        )
        logger.debug("PLINK --freq stdout: %s", result.stdout)
        if not frq_path.exists():
            raise FileNotFoundError(
                f"PLINK --freq did not produce expected output: {frq_path}"
            )
    return frq_path


# ---------------------------------------------------------------------------
# eQTL loading
# ---------------------------------------------------------------------------


def load_eqtl_source(
    source_config: EQTLSourceConfig,
    ref_freq_path: Path | None = None,
) -> pd.DataFrame:
    """Load and standardise eQTL summary statistics from a given source.

    Returns DataFrame with columns: SNP, gene, chr, pos, a1, a2, beta, se,
    pval, n
    """
    source = source_config.source.lower()
    eqtl_dir = source_config.path

    if source == "eqtlgen":
        return _load_eqtlgen(eqtl_dir, ref_freq_path)
    elif source.startswith("metabrain"):
        return _load_metabrain(eqtl_dir)
    else:
        raise ValueError(f"Unknown eQTL source: {source_config.source}")


def _load_eqtlgen(eqtl_dir: Path, ref_freq_path: Path | None) -> pd.DataFrame:
    """Load eQTLGen cis-eQTL data with Z-to-beta conversion via MAF lookup."""
    eqtl_files = list(eqtl_dir.glob("*.txt*")) + list(eqtl_dir.glob("*.tsv*"))
    if not eqtl_files:
        raise FileNotFoundError(f"No eQTLGen files found in {eqtl_dir}")

    eqtl_path = eqtl_files[0]
    logger.info("Loading eQTLGen from %s", eqtl_path)

    df = pd.read_csv(eqtl_path, sep="\t", dtype={"SNP": str, "Gene": str})

    col_map = {
        "Pvalue": "pval", "SNP": "SNP", "SNPChr": "chr", "SNPPos": "pos",
        "AssessedAllele": "a1", "OtherAllele": "a2", "Zscore": "zscore",
        "Gene": "gene", "NrSamples": "n",
        "GeneChr": "gene_chr", "GenePos": "gene_pos",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=available)

    required = {"SNP", "chr", "pos", "a1", "a2", "zscore", "gene", "n"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"eQTLGen missing required columns: {missing}")

    if ref_freq_path is None:
        raise ValueError(
            "ref_freq_path is required for eQTLGen Z-to-beta conversion. "
            "Generate via ensure_ref_freq()."
        )

    freq_df = pd.read_csv(ref_freq_path, sep=r"\s+", dtype={"SNP": str})
    freq_df = freq_df[["SNP", "MAF"]].drop_duplicates(subset="SNP")

    n_before = len(df)
    df = df.merge(freq_df, on="SNP", how="inner")
    n_excluded = n_before - len(df)
    if n_excluded > 0:
        logger.warning(
            "Excluded %d eQTLGen SNPs not in reference .frq file", n_excluded
        )

    maf = df["MAF"].values
    z = df["zscore"].values
    n = df["n"].values.astype(float)

    denom = np.sqrt(2.0 * maf * (1.0 - maf) * (n + z**2))
    mask = denom > 0
    df = df.loc[mask].copy()
    denom = denom[mask]
    z = z[mask]

    df["beta"] = z / denom
    df["se"] = 1.0 / denom
    df["pval"] = df.get("pval", 2 * scipy.stats.norm.sf(np.abs(z)))

    df["chr"] = df["chr"].astype(int)
    df["pos"] = df["pos"].astype(int)
    df["a1"] = df["a1"].str.upper()
    df["a2"] = df["a2"].str.upper()

    for col in ("gene_chr", "gene_pos"):
        if col in df.columns:
            df[col] = df[col].astype(int)

    logger.info(
        "Loaded %d eQTLGen cis-eQTL records for %d genes",
        len(df), df["gene"].nunique(),
    )

    out_cols = ["SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n"]
    for extra in ("gene_chr", "gene_pos"):
        if extra in df.columns:
            out_cols.append(extra)
    return df[out_cols]


def _load_metabrain(eqtl_dir: Path) -> pd.DataFrame:
    """Load MetaBrain cortex eQTL data from the normalized file.

    Expects ``metabrain_cortex_normalized.tsv.gz`` produced by
    ``setup-resources`` (derived from raw per-chromosome files).
    """
    normalized = eqtl_dir / "metabrain_cortex_normalized.tsv.gz"
    if not normalized.exists():
        raise FileNotFoundError(
            f"MetaBrain normalized file not found at {normalized}. "
            "Run 'repogen setup-resources' to generate it from the "
            "raw per-chromosome files in this directory."
        )

    logger.info("Loading MetaBrain from %s", normalized)
    df = pd.read_csv(normalized, sep="\t", dtype={"SNP": str, "gene": str})

    required = {"SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MetaBrain normalized file missing columns: {missing}")

    if "n" not in df.columns:
        df["n"] = 2970

    df["chr"] = pd.to_numeric(df["chr"], errors="coerce").astype("Int64")
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df["a1"] = df["a1"].str.upper()
    df["a2"] = df["a2"].str.upper()

    for col in ("gene_chr", "gene_pos"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # The normalized MetaBrain file stores composite ``chr:pos:rsid:alleles``
    # SNP tokens. Parse them to bare rsIDs and drop exact duplicates so this legacy
    # loader is consistent with the chunked MR path (multiallelic collisions are kept
    # for harmonisation to resolve, mirroring _load_metabrain_chunked).
    df, _norm_counts = _normalise_metabrain_snp_ids(df)
    df, _dedup_counts = _dedup_metabrain_instruments(df, collapse_multiallelic=False)

    logger.info(
        "Loaded %d MetaBrain cis-eQTL records for %d genes",
        len(df), df["gene"].nunique(),
    )

    out_cols = ["SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n"]
    for extra in ("gene_chr", "gene_pos"):
        if extra in df.columns:
            out_cols.append(extra)
    return df[out_cols]


# ---------------------------------------------------------------------------
# Memory-safe chunked eQTL loaders
# ---------------------------------------------------------------------------

_EQTLGEN_USECOLS = [
    "Pvalue", "SNP", "SNPChr", "SNPPos", "AssessedAllele",
    "OtherAllele", "Zscore", "Gene", "NrSamples", "GeneChr", "GenePos",
]

_EQTLGEN_DTYPES = {
    "Pvalue": float, "SNP": str, "SNPChr": "Int64", "SNPPos": "Int64",
    "AssessedAllele": str, "OtherAllele": str, "Zscore": float,
    "Gene": str, "NrSamples": "Int64", "GeneChr": "Int64", "GenePos": "Int64",
}

_METABRAIN_USECOLS = [
    "gene", "SNP", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n",
    "gene_chr", "gene_pos",
]


def _get_peak_rss_mb() -> float:
    """Return peak RSS in MB (Linux only; returns 0.0 on other platforms)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


def _load_eqtlgen_chunked(
    eqtl_dir: Path,
    ref_freq_path: Path,
    instrument_pval: float,
    chunksize: int = 2_000_000,
) -> tuple[pd.DataFrame, dict[str, dict], int]:
    """Memory-safe chunked eQTLGen loader for MR instrument extraction.

    Returns
    -------
    instruments_df : DataFrame
        Rows with pval < instrument_pval, post-.frq merge and Z-to-beta.
        Columns: SNP, gene, chr, pos, a1, a2, beta, se, pval, n, gene_chr, gene_pos
    gene_metadata : dict[str, dict]
        Per-gene anchor metadata {gene_id: {"chr": int, "start": int}}.
        Uses gene_pos (TSS) when available; falls back to min(SNP pos).
    n_genes_valid : int
        Total unique genes after loader validity transforms (Bonferroni denominator).
    """
    eqtl_files = list(eqtl_dir.glob("*.txt*")) + list(eqtl_dir.glob("*.tsv*"))
    if not eqtl_files:
        raise FileNotFoundError(f"No eQTLGen files found in {eqtl_dir}")
    eqtl_path = eqtl_files[0]

    if ref_freq_path is None:
        raise ValueError(
            "ref_freq_path is required for eQTLGen Z-to-beta conversion."
        )

    logger.info(
        "Loading eQTLGen (chunked) from %s [instrument_pval=%.1e, chunksize=%d]",
        eqtl_path, instrument_pval, chunksize,
    )

    freq_df = pd.read_csv(
        ref_freq_path, sep=r"\s+", dtype={"SNP": str}, usecols=["SNP", "MAF"],
    )
    freq_df = freq_df.drop_duplicates(subset="SNP")
    maf_index = freq_df.set_index("SNP")["MAF"]
    logger.info("Reference .frq loaded: %d SNPs (%.2f GB)",
                len(freq_df), freq_df.memory_usage(deep=True).sum() / 1e9)

    valid_genes: set[str] = set()
    gene_metadata: dict[str, dict] = {}
    min_snp_pos: dict[str, int] = {}
    instrument_chunks: list[pd.DataFrame] = []
    total_rows_read = 0
    total_rows_valid = 0
    total_rows_retained = 0

    col_map = {
        "Pvalue": "pval", "SNP": "SNP", "SNPChr": "chr", "SNPPos": "pos",
        "AssessedAllele": "a1", "OtherAllele": "a2", "Zscore": "zscore",
        "Gene": "gene", "NrSamples": "n", "GeneChr": "gene_chr", "GenePos": "gene_pos",
    }

    reader = pd.read_csv(
        eqtl_path, sep="\t", usecols=_EQTLGEN_USECOLS,
        dtype=_EQTLGEN_DTYPES, chunksize=chunksize,
    )

    t0 = time.time()
    for chunk_idx, chunk in enumerate(reader):
        total_rows_read += len(chunk)

        available = {k: v for k, v in col_map.items() if k in chunk.columns}
        chunk = chunk.rename(columns=available)

        # Indexed MAF lookup (equivalent to inner-join: rows without match -> NaN -> dropped)
        chunk["MAF"] = chunk["SNP"].map(maf_index)
        chunk = chunk.dropna(subset=["MAF"])
        if chunk.empty:
            continue

        maf = chunk["MAF"].values
        z = chunk["zscore"].values
        n = chunk["n"].values.astype(float)
        denom = np.sqrt(2.0 * maf * (1.0 - maf) * (n + z**2))
        mask = denom > 0
        chunk = chunk.loc[mask].copy()
        if chunk.empty:
            continue
        denom = denom[mask]
        z = z[mask]

        chunk["beta"] = z / denom
        chunk["se"] = 1.0 / denom
        chunk["pval"] = chunk.get("pval", 2 * scipy.stats.norm.sf(np.abs(z)))
        chunk["chr"] = chunk["chr"].astype(int)
        chunk["pos"] = chunk["pos"].astype(int)

        for col in ("gene_chr", "gene_pos"):
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

        total_rows_valid += len(chunk)
        chunk_genes = chunk["gene"].unique()
        valid_genes.update(chunk_genes)

        # Vectorized gene metadata extraction via grouped pass
        has_gene_pos = "gene_pos" in chunk.columns and "gene_chr" in chunk.columns
        new_genes = [g for g in chunk_genes if g not in gene_metadata]
        continuing_fallback_genes = [g for g in chunk_genes if g in min_snp_pos]

        if new_genes:
            new_gene_chunk = chunk.loc[chunk["gene"].isin(new_genes)]
            grp = new_gene_chunk.groupby("gene", sort=False)
            first_rows = grp.nth(0).set_index("gene")
            pos_mins = grp["pos"].min()

            for g in new_genes:
                if g not in first_rows.index:
                    continue
                row0 = first_rows.loc[g]
                if (
                    has_gene_pos
                    and pd.notna(row0.get("gene_chr"))
                    and pd.notna(row0.get("gene_pos"))
                ):
                    gene_metadata[g] = {
                        "chr": int(row0["gene_chr"]),
                        "start": int(row0["gene_pos"]),
                    }
                else:
                    pos_min = int(pos_mins[g])
                    gene_metadata[g] = {"chr": int(row0["chr"]), "start": pos_min}
                    min_snp_pos[g] = pos_min

        if continuing_fallback_genes:
            fallback_chunk = chunk.loc[chunk["gene"].isin(continuing_fallback_genes)]
            chunk_mins = fallback_chunk.groupby("gene", sort=False)["pos"].min()
            for g in continuing_fallback_genes:
                if g in chunk_mins.index:
                    chunk_min = int(chunk_mins[g])
                    if chunk_min < min_snp_pos[g]:
                        min_snp_pos[g] = chunk_min
                        gene_metadata[g]["start"] = chunk_min

        # Extract instruments (apply a1/a2 uppercasing only to retained rows)
        sig_mask = chunk["pval"] < instrument_pval
        if sig_mask.any():
            instruments = chunk.loc[sig_mask].copy()
            instruments["a1"] = instruments["a1"].str.upper()
            instruments["a2"] = instruments["a2"].str.upper()
            out_cols = ["SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n"]
            for extra in ("gene_chr", "gene_pos"):
                if extra in instruments.columns:
                    out_cols.append(extra)
            instrument_chunks.append(instruments[out_cols])
            total_rows_retained += len(instruments)

        if (chunk_idx + 1) % 5 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  eQTLGen chunk %d: %d rows read, %d valid, %d instruments, "
                "%d genes | %.1fs elapsed",
                chunk_idx + 1, total_rows_read, total_rows_valid,
                total_rows_retained, len(valid_genes), elapsed,
            )

    elapsed = time.time() - t0
    n_genes_valid = len(valid_genes)

    if instrument_chunks:
        instruments_df = pd.concat(instrument_chunks, ignore_index=True)
    else:
        out_cols = ["SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n",
                    "gene_chr", "gene_pos"]
        instruments_df = pd.DataFrame(columns=out_cols)

    logger.info(
        "eQTLGen chunked load complete: %d rows read, %d valid, "
        "%d instruments retained, %d genes (Bonf denom) | %.1fs",
        total_rows_read, total_rows_valid, len(instruments_df),
        n_genes_valid, elapsed,
    )

    return instruments_df, gene_metadata, n_genes_valid


# ---------------------------------------------------------------------------
# MetaBrain SNP normalisation & deduplication
# ---------------------------------------------------------------------------

# Exact bare-rsID pattern (case-insensitive, whole token). MetaBrain distributes a
# composite SNP identifier ``chr:pos:rsid:alleles`` (e.g. "10:100000012:rs12345:A_G");
# the GWAS and eQTLGen sources use bare rsIDs ("rs12345"). We parse *field-aware*
# (bare token, or the 3rd colon-field of the composite) rather than extracting any
# ``rs\d+`` substring, so malformed tokens are dropped rather than silently accepted.
_METABRAIN_BARE_RSID_RE = r"rs\d+"


def _normalise_metabrain_snp_ids(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse MetaBrain composite SNP tokens to bare, lowercase rsIDs (field-aware).

    MetaBrain's ``SNP`` column carries a composite ``chr:pos:rsid:alleles`` token
    (e.g. ``10:100000012:rs12345:A_G``), whereas the GWAS and eQTLGen sources use
    bare rsIDs (``rs12345``). Left unparsed, MetaBrain instruments never join the
    GWAS on ``SNP`` and the source silently collapses to a handful of positionally
    recovered rows. This helper resolves the rsID *field-aware*:

    - a whole token matching ``^rs\\d+$`` (case-insensitive) is treated as a bare rsID;
    - otherwise the token is split on ``:`` and the third field (index 2, the
      ``rsid`` slot of ``chr:pos:rsid:alleles``) is used, but only if it itself
      matches ``^rs\\d+$``;
    - any other token (no rsID in the expected slot, e.g. positional-only
      ``chr:pos:ref:alt``) is dropped and counted.

    This is stricter than substring extraction: it will not lift an ``rs\\d+`` out of
    an unexpected field, so parse-recovery counters are not overstated. The helper is
    *idempotent* (bare rsIDs pass through), trims whitespace, canonicalises ``RS``->``rs``,
    and preserves the original token in an additive internal ``variant_id`` column.

    Returns ``(normalised_df, counts)`` where ``counts`` has keys:
    ``metabrain_snp_bare_rsid``, ``metabrain_snp_composite_parsed``,
    ``metabrain_snp_dropped_non_rsid``, ``metabrain_rows_after_snp_normalisation``.
    """
    counts = {
        "metabrain_snp_bare_rsid": 0,
        "metabrain_snp_composite_parsed": 0,
        "metabrain_snp_dropped_non_rsid": 0,
        "metabrain_rows_after_snp_normalisation": 0,
    }

    out = df.copy()
    if out.empty or "SNP" not in out.columns:
        if "variant_id" not in out.columns:
            out["variant_id"] = pd.Series(dtype=object)
        counts["metabrain_rows_after_snp_normalisation"] = len(out)
        return out, counts

    snp_raw = out["SNP"].astype("string")
    snp_trimmed = snp_raw.str.strip()
    # Provenance: keep the original (untrimmed) token.
    out["variant_id"] = snp_raw

    # Bare rsID (whole token).
    bare_mask = snp_trimmed.str.fullmatch(f"(?i){_METABRAIN_BARE_RSID_RE}").fillna(False)

    # Composite: split on ':' (cap at 4 fields), take field index 2 (the rsid slot).
    parts = snp_trimmed.str.split(":", n=3, expand=True)
    if parts.shape[1] >= 3:
        field2 = parts[2]
        field2_mask = field2.str.fullmatch(
            f"(?i){_METABRAIN_BARE_RSID_RE}"
        ).fillna(False)
    else:
        field2 = pd.Series(pd.NA, index=snp_trimmed.index, dtype="string")
        field2_mask = pd.Series(False, index=snp_trimmed.index)

    composite_mask = (~bare_mask) & field2_mask

    resolved = pd.Series(pd.NA, index=snp_trimmed.index, dtype="string")
    resolved[bare_mask] = snp_trimmed[bare_mask].str.lower()
    resolved[composite_mask] = field2[composite_mask].str.lower()
    matched = resolved.notna()

    counts["metabrain_snp_bare_rsid"] = int(bare_mask.sum())
    counts["metabrain_snp_composite_parsed"] = int(composite_mask.sum())
    counts["metabrain_snp_dropped_non_rsid"] = int((~matched).sum())

    out["SNP"] = resolved
    out = out.loc[matched].reset_index(drop=True)
    # Restore plain-object str dtype to match GWAS/eQTLGen 'SNP' convention.
    out["SNP"] = out["SNP"].astype(str)
    out["variant_id"] = out["variant_id"].astype(str)

    counts["metabrain_rows_after_snp_normalisation"] = len(out)
    return out, counts


def _dedup_metabrain_instruments(
    df: pd.DataFrame,
    collapse_multiallelic: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """De-duplicate MetaBrain rows sharing an rsID within a gene.

    After composite->rsID normalisation, MetaBrain rows can share a single rsID
    within a gene in two ways:

    - Exact duplicates - identical ``(gene, SNP, a1, a2)``. These are always
      collapsed to the lowest-p row: ``harmonise_gwas_eqtl`` would otherwise keep
      *both* (both allele-compatible and identical), over-weighting ``k``.
    - Multiallelic collisions - same ``(gene, SNP)`` but different allele pairs.

    Handling of multiallelic collisions is controlled by ``collapse_multiallelic``:

    - ``True`` (colocalisation path): collapse to one row per ``(gene, SNP)``,
      keeping the lowest p-value with a deterministic allele tiebreak. This is
      correct for ``coloc_abf`` (which is orientation-invariant - it uses squared
      z-scores - and must not double-count a SNP).
    - ``False`` (MR-instrument path): keep multiallelic rows and let
      ``harmonise_gwas_eqtl`` select the single GWAS-allele-compatible pair
      downstream. Collapsing by p-value alone here is unsafe: if the lowest-p pair
      is allele-incompatible/palindromic while a higher-p pair would harmonise, the
      valid instrument would be lost. ``clump_instruments`` de-duplicates its PLINK
      input, so the surviving multiallelic rows do not destabilise clumping.

    Returns ``(deduped_df, counts)`` where ``counts`` has keys:
    ``metabrain_exact_duplicate_rows``, ``metabrain_multiallelic_collisions``,
    ``metabrain_rows_after_dedup``.
    """
    counts = {
        "metabrain_exact_duplicate_rows": 0,
        "metabrain_multiallelic_collisions": 0,
        "metabrain_rows_after_dedup": 0,
    }
    key = ["gene", "SNP"]
    if df.empty or not set(key).issubset(df.columns):
        counts["metabrain_rows_after_dedup"] = len(df)
        return df, counts

    n_before = len(df)

    exact_subset = [c for c in ("gene", "SNP", "a1", "a2") if c in df.columns]
    counts["metabrain_exact_duplicate_rows"] = int(
        df.duplicated(subset=exact_subset).sum()
    )

    if {"a1", "a2"}.issubset(df.columns):
        allele_pair = df["a1"].astype(str) + "/" + df["a2"].astype(str)
        allele_nunique = allele_pair.groupby([df["gene"], df["SNP"]]).nunique()
        counts["metabrain_multiallelic_collisions"] = int((allele_nunique > 1).sum())

    sort_cols = list(key)
    ascending = [True, True]
    if "pval" in df.columns:
        sort_cols.append("pval")
        ascending.append(True)
    for tb in ("a1", "a2"):
        if tb in df.columns:
            sort_cols.append(tb)
            ascending.append(True)

    # Dedup key: full (gene, SNP) when collapsing multiallelic; else exact-only.
    dedup_subset = key if collapse_multiallelic else exact_subset
    out = (
        df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .drop_duplicates(subset=dedup_subset, keep="first")
        .reset_index(drop=True)
    )
    counts["metabrain_rows_after_dedup"] = len(out)

    if counts["metabrain_multiallelic_collisions"] > 0 or counts[
        "metabrain_exact_duplicate_rows"
    ] > 0:
        logger.warning(
            "MetaBrain dedup (collapse_multiallelic=%s): %d multiallelic "
            "(gene,SNP) collisions, %d exact duplicate rows (%d -> %d rows)",
            collapse_multiallelic,
            counts["metabrain_multiallelic_collisions"],
            counts["metabrain_exact_duplicate_rows"],
            n_before,
            len(out),
        )
    return out, counts


def _load_metabrain_chunked(
    eqtl_dir: Path,
    instrument_pval: float,
    chunksize: int = 2_000_000,
    stats_out: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict], int]:
    """Memory-safe chunked MetaBrain loader for MR instrument extraction.

    Returns same tuple structure as _load_eqtlgen_chunked.
    """
    normalized = eqtl_dir / "metabrain_cortex_normalized.tsv.gz"
    if not normalized.exists():
        raise FileNotFoundError(
            f"MetaBrain normalized file not found at {normalized}. "
            "Run 'repogen setup-resources' to generate it."
        )

    logger.info(
        "Loading MetaBrain (chunked) from %s [instrument_pval=%.1e, chunksize=%d]",
        normalized, instrument_pval, chunksize,
    )

    usecols = [c for c in _METABRAIN_USECOLS]
    dtypes = {"SNP": str, "gene": str}

    valid_genes: set[str] = set()
    gene_metadata: dict[str, dict] = {}
    min_snp_pos: dict[str, int] = {}
    instrument_chunks: list[pd.DataFrame] = []
    total_rows_read = 0
    total_rows_retained = 0

    reader = pd.read_csv(
        normalized, sep="\t", dtype=dtypes, chunksize=chunksize,
        usecols=lambda c: c in usecols,
    )

    t0 = time.time()
    for chunk_idx, chunk in enumerate(reader):
        total_rows_read += len(chunk)

        if "n" not in chunk.columns:
            chunk["n"] = 2970

        chunk["chr"] = pd.to_numeric(chunk["chr"], errors="coerce").astype("Int64")
        chunk["pos"] = pd.to_numeric(chunk["pos"], errors="coerce").astype("Int64")

        for col in ("gene_chr", "gene_pos"):
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

        chunk_genes = chunk["gene"].unique()
        valid_genes.update(chunk_genes)

        # Vectorized gene metadata extraction via grouped pass
        has_gene_pos = "gene_pos" in chunk.columns and "gene_chr" in chunk.columns
        new_genes = [g for g in chunk_genes if g not in gene_metadata]
        continuing_fallback_genes = [g for g in chunk_genes if g in min_snp_pos]

        if new_genes:
            new_gene_chunk = chunk.loc[chunk["gene"].isin(new_genes)]
            grp = new_gene_chunk.groupby("gene", sort=False)
            first_rows = grp.nth(0).set_index("gene")
            pos_mins = grp["pos"].min()

            for g in new_genes:
                if g not in first_rows.index:
                    continue
                row0 = first_rows.loc[g]
                if (
                    has_gene_pos
                    and pd.notna(row0.get("gene_chr"))
                    and pd.notna(row0.get("gene_pos"))
                ):
                    gene_metadata[g] = {
                        "chr": int(row0["gene_chr"]),
                        "start": int(row0["gene_pos"]),
                    }
                else:
                    pos_val = pos_mins.get(g)
                    if pd.notna(pos_val):
                        pos_min = int(pos_val)
                        gene_metadata[g] = {"chr": int(row0["chr"]), "start": pos_min}
                        min_snp_pos[g] = pos_min

        if continuing_fallback_genes:
            fallback_chunk = chunk.loc[chunk["gene"].isin(continuing_fallback_genes)]
            chunk_mins = fallback_chunk.groupby("gene", sort=False)["pos"].min()
            for g in continuing_fallback_genes:
                if g in chunk_mins.index:
                    pos_val = chunk_mins[g]
                    if pd.notna(pos_val):
                        chunk_min = int(pos_val)
                        if chunk_min < min_snp_pos[g]:
                            min_snp_pos[g] = chunk_min
                            gene_metadata[g]["start"] = chunk_min

        # Extract instruments (apply a1/a2 uppercasing only to retained rows)
        sig_mask = chunk["pval"] < instrument_pval
        if sig_mask.any():
            instruments = chunk.loc[sig_mask].copy()
            instruments["a1"] = instruments["a1"].str.upper()
            instruments["a2"] = instruments["a2"].str.upper()
            out_cols = ["SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n"]
            for extra in ("gene_chr", "gene_pos"):
                if extra in instruments.columns:
                    out_cols.append(extra)
            instrument_chunks.append(instruments[out_cols])
            total_rows_retained += len(instruments)

        if (chunk_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  MetaBrain chunk %d: %d rows read, %d instruments, "
                "%d genes | %.1fs elapsed",
                chunk_idx + 1, total_rows_read,
                total_rows_retained, len(valid_genes), elapsed,
            )

    elapsed = time.time() - t0
    n_genes_valid = len(valid_genes)

    if instrument_chunks:
        instruments_df = pd.concat(instrument_chunks, ignore_index=True)
    else:
        out_cols = ["SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n",
                    "gene_chr", "gene_pos"]
        instruments_df = pd.DataFrame(columns=out_cols)

    # Parse composite SNP tokens to bare rsIDs, then remove *exact*
    # duplicate rows only. Multiallelic (gene,SNP) collisions are deliberately kept
    # so harmonise_gwas_eqtl can select the GWAS-allele-compatible pair downstream
    # (collapsing by p-value alone could discard the compatible instrument).
    instruments_df, norm_counts = _normalise_metabrain_snp_ids(instruments_df)
    instruments_df, dedup_counts = _dedup_metabrain_instruments(
        instruments_df, collapse_multiallelic=False,
    )
    if stats_out is not None:
        stats_out.update(norm_counts)
        stats_out.update(dedup_counts)

    logger.info(
        "MetaBrain SNP normalisation: %d bare, %d composite parsed, %d dropped "
        "(non-rsID); %d rows after normalisation, %d after dedup "
        "(%d exact dups, %d multiallelic collisions)",
        norm_counts["metabrain_snp_bare_rsid"],
        norm_counts["metabrain_snp_composite_parsed"],
        norm_counts["metabrain_snp_dropped_non_rsid"],
        norm_counts["metabrain_rows_after_snp_normalisation"],
        dedup_counts["metabrain_rows_after_dedup"],
        dedup_counts["metabrain_exact_duplicate_rows"],
        dedup_counts["metabrain_multiallelic_collisions"],
    )

    logger.info(
        "MetaBrain chunked load complete: %d rows read, %d instruments retained, "
        "%d genes (Bonf denom) | %.1fs",
        total_rows_read, len(instruments_df), n_genes_valid, elapsed,
    )

    return instruments_df, gene_metadata, n_genes_valid


def _load_eqtl_coloc_genes(
    eqtl_dir: Path,
    ref_freq_path: Path | None,
    gene_set: set[str],
    source: str,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Reload full locus data for specific genes (coloc Phase 3).

    Re-streams the source file, retaining ALL rows for genes in gene_set.
    Applies same validity transforms as the respective full loader.
    """
    if not gene_set:
        return pd.DataFrame()

    if source == "eqtlgen":
        return _reload_eqtlgen_for_coloc(eqtl_dir, ref_freq_path, gene_set, chunksize)
    elif source.startswith("metabrain"):
        return _reload_metabrain_for_coloc(eqtl_dir, gene_set, chunksize)
    else:
        raise ValueError(f"Unknown eQTL source for coloc reload: {source}")


def _reload_eqtlgen_for_coloc(
    eqtl_dir: Path,
    ref_freq_path: Path,
    gene_set: set[str],
    chunksize: int,
) -> pd.DataFrame:
    """Re-stream eQTLGen retaining all rows for specified genes."""
    eqtl_files = list(eqtl_dir.glob("*.txt*")) + list(eqtl_dir.glob("*.tsv*"))
    eqtl_path = eqtl_files[0]

    freq_df = pd.read_csv(
        ref_freq_path, sep=r"\s+", dtype={"SNP": str}, usecols=["SNP", "MAF"],
    )
    freq_df = freq_df.drop_duplicates(subset="SNP")

    col_map = {
        "Pvalue": "pval", "SNP": "SNP", "SNPChr": "chr", "SNPPos": "pos",
        "AssessedAllele": "a1", "OtherAllele": "a2", "Zscore": "zscore",
        "Gene": "gene", "NrSamples": "n", "GeneChr": "gene_chr", "GenePos": "gene_pos",
    }

    logger.info("Coloc reload: streaming eQTLGen for %d genes", len(gene_set))
    retained_chunks: list[pd.DataFrame] = []
    t0 = time.time()

    reader = pd.read_csv(
        eqtl_path, sep="\t", usecols=_EQTLGEN_USECOLS,
        dtype=_EQTLGEN_DTYPES, chunksize=chunksize,
    )

    for chunk in reader:
        available = {k: v for k, v in col_map.items() if k in chunk.columns}
        chunk = chunk.rename(columns=available)

        chunk = chunk.loc[chunk["gene"].isin(gene_set)]
        if chunk.empty:
            continue

        chunk = chunk.merge(freq_df, on="SNP", how="inner")
        if chunk.empty:
            continue

        maf = chunk["MAF"].values
        z = chunk["zscore"].values
        n = chunk["n"].values.astype(float)
        denom = np.sqrt(2.0 * maf * (1.0 - maf) * (n + z**2))
        mask = denom > 0
        chunk = chunk.loc[mask].copy()
        if chunk.empty:
            continue
        denom = denom[mask]
        z = z[mask]

        chunk["beta"] = z / denom
        chunk["se"] = 1.0 / denom
        chunk["pval"] = chunk.get("pval", 2 * scipy.stats.norm.sf(np.abs(z)))
        chunk["chr"] = chunk["chr"].astype(int)
        chunk["pos"] = chunk["pos"].astype(int)
        chunk["a1"] = chunk["a1"].str.upper()
        chunk["a2"] = chunk["a2"].str.upper()

        for col in ("gene_chr", "gene_pos"):
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

        out_cols = ["SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n"]
        for extra in ("gene_chr", "gene_pos"):
            if extra in chunk.columns:
                out_cols.append(extra)
        retained_chunks.append(chunk[out_cols])

    elapsed = time.time() - t0
    if retained_chunks:
        result = pd.concat(retained_chunks, ignore_index=True)
    else:
        result = pd.DataFrame()

    logger.info(
        "Coloc reload complete: %d rows for %d genes | %.1fs",
        len(result), result["gene"].nunique() if not result.empty else 0, elapsed,
    )
    return result


def _reload_metabrain_for_coloc(
    eqtl_dir: Path,
    gene_set: set[str],
    chunksize: int,
) -> pd.DataFrame:
    """Re-stream MetaBrain retaining all rows for specified genes."""
    normalized = eqtl_dir / "metabrain_cortex_normalized.tsv.gz"

    usecols = [c for c in _METABRAIN_USECOLS]
    dtypes = {"SNP": str, "gene": str}

    logger.info("Coloc reload: streaming MetaBrain for %d genes", len(gene_set))
    retained_chunks: list[pd.DataFrame] = []
    t0 = time.time()

    reader = pd.read_csv(
        normalized, sep="\t", dtype=dtypes, chunksize=chunksize,
        usecols=lambda c: c in usecols,
    )

    for chunk in reader:
        chunk = chunk.loc[chunk["gene"].isin(gene_set)]
        if chunk.empty:
            continue

        if "n" not in chunk.columns:
            chunk["n"] = 2970

        chunk["chr"] = pd.to_numeric(chunk["chr"], errors="coerce").astype("Int64")
        chunk["pos"] = pd.to_numeric(chunk["pos"], errors="coerce").astype("Int64")
        chunk["a1"] = chunk["a1"].str.upper()
        chunk["a2"] = chunk["a2"].str.upper()

        for col in ("gene_chr", "gene_pos"):
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

        out_cols = ["SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n"]
        for extra in ("gene_chr", "gene_pos"):
            if extra in chunk.columns:
                out_cols.append(extra)
        retained_chunks.append(chunk[out_cols])

    elapsed = time.time() - t0
    if retained_chunks:
        result = pd.concat(retained_chunks, ignore_index=True)
    else:
        result = pd.DataFrame()

    # Apply the same SNP normalisation used in Phase 1 so the coloc merge
    # keys (SNP) line up with the GWAS rsIDs. Collapse multiallelic collisions to
    # one row per (gene, SNP) here - coloc_abf is orientation-invariant (uses z²) and
    # must not double-count a SNP, so lowest-p selection is correct for coloc.
    result, norm_counts = _normalise_metabrain_snp_ids(result)
    result, dedup_counts = _dedup_metabrain_instruments(
        result, collapse_multiallelic=True,
    )

    logger.info(
        "Coloc reload complete: %d rows for %d genes | %.1fs "
        "(%d composite parsed, %d dropped non-rsID, %d multiallelic collisions)",
        len(result), result["gene"].nunique() if not result.empty else 0, elapsed,
        norm_counts["metabrain_snp_composite_parsed"],
        norm_counts["metabrain_snp_dropped_non_rsid"],
        dedup_counts["metabrain_multiallelic_collisions"],
    )
    return result


# ---------------------------------------------------------------------------
# Instrument selection & clumping
# ---------------------------------------------------------------------------


def select_instruments(
    eqtl_df: pd.DataFrame,
    gene: str,
    cis_window_kb: int,
    gene_start: int,
    gene_chr: int,
    pval_threshold: float,
    f_stat_threshold: float,
) -> pd.DataFrame:
    """Select cis-eQTL instruments for a single gene."""
    gene_eqtl = eqtl_df.loc[eqtl_df["gene"] == gene].copy()
    if gene_eqtl.empty:
        return gene_eqtl

    window = cis_window_kb * 1000
    gene_eqtl = gene_eqtl.loc[
        (gene_eqtl["chr"] == gene_chr)
        & (gene_eqtl["pos"] >= gene_start - window)
        & (gene_eqtl["pos"] <= gene_start + window)
    ]

    gene_eqtl = gene_eqtl.loc[gene_eqtl["pval"] < pval_threshold]

    if gene_eqtl.empty:
        return gene_eqtl

    gene_eqtl["f_stat"] = f_statistic(
        gene_eqtl["beta"].values, gene_eqtl["se"].values
    )
    gene_eqtl = gene_eqtl.loc[gene_eqtl["f_stat"] >= f_stat_threshold]

    return gene_eqtl


def clump_instruments(
    instruments: pd.DataFrame,
    bfile_full_path: Path,
    clump_r2: float,
    clump_kb: int,
    plink_binary: Path,
    gene_chr: int | None = None,
) -> pd.DataFrame:
    """LD-clump instruments using PLINK 1.9 subprocess.

    Parameters
    ----------
    gene_chr : int | None
        If provided (≥1), restricts PLINK to this chromosome via --chr,
        materially reducing reference panel scan time.
    """
    if len(instruments) <= 1:
        return instruments

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="repogen_clump_")
        clump_file = Path(tmpdir) / "clump_input.txt"
        extract_file = Path(tmpdir) / "extract_snps.txt"
        out_prefix = Path(tmpdir) / "clumped"

        clump_data = instruments[["SNP", "pval"]].rename(columns={"pval": "P"})
        if clump_data["SNP"].duplicated().any():
            # Multiallelic rsIDs (e.g. MetaBrain) would otherwise appear
            # multiple times in the PLINK --clump input; keep one row per SNP
            # (lowest P) so clumping ranks cleanly and does not choke on duplicate
            # IDs. The returned frame still keeps all rows for retained rsIDs, and
            # harmonise_gwas_eqtl then selects the single allele-compatible pair.
            clump_data = (
                clump_data.sort_values("P", kind="mergesort")
                .drop_duplicates(subset="SNP", keep="first")
            )
        clump_data.to_csv(clump_file, sep="\t", index=False)

        extract_snps = instruments["SNP"].dropna().unique()
        extract_file.write_text("\n".join(extract_snps) + "\n")

        cmd = [
            str(plink_binary),
            "--bfile", str(bfile_full_path),
            "--extract", str(extract_file),
            "--clump", str(clump_file),
            "--clump-p1", "1",
            "--clump-r2", str(clump_r2),
            "--clump-kb", str(clump_kb),
            "--out", str(out_prefix),
        ]

        if gene_chr is not None and isinstance(gene_chr, int) and gene_chr >= 1:
            cmd.extend(["--chr", str(gene_chr)])
        elif gene_chr is not None:
            logger.warning(
                "Invalid gene_chr=%r for PLINK --chr; running without chromosome scope",
                gene_chr,
            )
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            combined = (e.stdout or "") + (e.stderr or "")
            log_path = out_prefix.with_suffix(".log")
            if log_path.exists():
                combined += log_path.read_text(errors="replace")
            combined_lower = combined.lower()
            if "no variants remaining" in combined_lower or "0 variants remaining" in combined_lower:
                logger.debug(
                    "PLINK clumping: no variants remaining after filters for "
                    "gene_chr=%s. Returning empty instruments.", gene_chr,
                )
                return instruments.iloc[0:0]
            raise RuntimeError(
                f"PLINK clumping failed (returncode={e.returncode}): "
                f"{(e.stderr or '')[:500]}"
            ) from e

        clumped_path = Path(str(out_prefix) + ".clumped")
        if not clumped_path.exists():
            logger.info(
                "PLINK produced no .clumped file - no variants matched "
                "reference panel. Returning empty instruments."
            )
            return instruments.iloc[0:0]

        clumped_df = pd.read_csv(clumped_path, sep=r"\s+")
        if "SNP" not in clumped_df.columns:
            raise ValueError(
                f"Malformed PLINK .clumped output: 'SNP' column missing. "
                f"Columns found: {list(clumped_df.columns)}"
            )

        retained_snps = set(clumped_df["SNP"].dropna())
        clumped = instruments.loc[instruments["SNP"].isin(retained_snps)]

        logger.debug(
            "Clumped %d -> %d instruments", len(instruments), len(clumped)
        )
        return clumped

    finally:
        if tmpdir is not None:
            import shutil as _shutil
            _shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Variant harmonisation
# ---------------------------------------------------------------------------


def _complement_allele(allele: str) -> str:
    return "".join(COMPLEMENT.get(b, b) for b in allele)


def _merge_eqtl_gwas_two_stage(
    eqtl_df: pd.DataFrame,
    gwas_df: pd.DataFrame,
    gwas_cols: list[str],
    suffixes: tuple[str, str] = ("_eqtl", "_gwas"),
) -> pd.DataFrame:
    """Two-stage merge: rsID primary, chr:pos fallback for null-rsID GWAS rows.

    Shared by harmonisation and coloc to avoid divergent join strategies.
    """
    gwas_snp = gwas_df.loc[gwas_df["SNP"].notna()].copy()
    merged = eqtl_df.merge(
        gwas_snp[["SNP"] + gwas_cols],
        on="SNP", how="inner", suffixes=suffixes,
    )

    n_rsid = len(merged)
    unmatched = eqtl_df.loc[~eqtl_df["SNP"].isin(merged["SNP"])]

    if len(unmatched) > 0:
        eqtl_pos = unmatched.copy()
        eqtl_pos["_chrpos"] = (
            eqtl_pos["chr"].astype(str) + ":" + eqtl_pos["pos"].astype(str)
        )
        gwas_pos = gwas_df.copy()
        gwas_pos["_chrpos"] = (
            gwas_pos["CHR"].astype(str) + ":" + gwas_pos["POS"].astype(str)
        )
        pos_merged = eqtl_pos.merge(
            gwas_pos[["_chrpos"] + gwas_cols],
            on="_chrpos", how="inner", suffixes=suffixes,
        )
        if len(pos_merged) > 0:
            pos_merged = pos_merged.drop(columns=["_chrpos"], errors="ignore")
            merged = pd.concat([merged, pos_merged], ignore_index=True)
            logger.info(
                "Two-stage merge: %d via rsID, %d via positional (chr:pos) join",
                n_rsid, len(pos_merged),
            )

    return merged


def harmonise_gwas_eqtl(
    gwas_df: pd.DataFrame,
    instruments: pd.DataFrame,
) -> pd.DataFrame:
    """Align GWAS and eQTL alleles for MR instruments.

    Uses vectorised operations for allele concordance.
    """
    if instruments.empty:
        return pd.DataFrame()

    gwas_cols = ["A1", "A2", "BETA", "SE", "P", "N", "CHR", "POS", "MAF"]
    gwas_cols = [c for c in gwas_cols if c in gwas_df.columns]
    merged = _merge_eqtl_gwas_two_stage(
        instruments, gwas_df, gwas_cols, suffixes=("_exp", "_out"),
    )

    if merged.empty:
        return pd.DataFrame()

    e_a1 = merged["a1"].str.upper().values
    e_a2 = merged["a2"].str.upper().values
    g_a1 = merged["A1"].str.upper().values
    g_a2 = merged["A2"].str.upper().values

    direct = (e_a1 == g_a1) & (e_a2 == g_a2)
    flipped = (e_a1 == g_a2) & (e_a2 == g_a1)

    comp_a1 = np.array([_complement_allele(a) for a in e_a1])
    comp_a2 = np.array([_complement_allele(a) for a in e_a2])
    comp_direct = (comp_a1 == g_a1) & (comp_a2 == g_a2)
    comp_flipped = (comp_a1 == g_a2) & (comp_a2 == g_a1)

    palindromic = np.array([
        (set(a) == {"A", "T"} or set(a) == {"C", "G"})
        for a in zip(e_a1, e_a2)
    ])

    keep = (direct | flipped | comp_direct | comp_flipped) & ~palindromic
    flip_sign = flipped | comp_flipped

    n_excluded_palindromic = palindromic.sum()
    n_excluded_nomatch = (~(direct | flipped | comp_direct | comp_flipped) & ~palindromic).sum()

    if n_excluded_palindromic > 0:
        logger.debug("Excluded %d palindromic SNPs", n_excluded_palindromic)
    if n_excluded_nomatch > 0:
        logger.warning("Excluded %d SNPs with incompatible alleles", n_excluded_nomatch)

    merged = merged.loc[keep].copy()
    flip_sign = flip_sign[keep]

    gwas_beta = merged["BETA"].values.copy()
    gwas_beta[flip_sign] *= -1

    result = pd.DataFrame({
        "SNP": merged["SNP"].values,
        "beta_exposure": merged["beta"].values,
        "se_exposure": merged["se"].values,
        "beta_outcome": gwas_beta,
        "se_outcome": merged["SE"].values,
        "pval_exposure": merged["pval"].values if "pval" in merged.columns else np.nan,
        "n_exposure": merged["n"].values if "n" in merged.columns else np.nan,
        "n_outcome": merged["N"].values if "N" in merged.columns else np.nan,
        "maf": merged["MAF"].values if "MAF" in merged.columns else np.nan,
    })

    return result


# ---------------------------------------------------------------------------
# MR estimation functions
# ---------------------------------------------------------------------------


def f_statistic(beta_exp: np.ndarray, se_exp: np.ndarray) -> np.ndarray:
    """Per-SNP F-statistic. F = (beta/se)^2."""
    return (beta_exp / se_exp) ** 2


def wald_ratio(
    beta_exp: float, se_exp: float, beta_out: float, se_out: float,
) -> tuple[float, float, float]:
    """Single-instrument MR via Wald ratio with full delta-method SE."""
    beta_mr = beta_out / beta_exp
    var_mr = (se_out**2 / beta_exp**2) + (beta_mr**2 * se_exp**2 / beta_exp**2)
    se_mr = np.sqrt(var_mr)
    z = beta_mr / se_mr
    pval = 2.0 * scipy.stats.norm.sf(abs(z))
    return float(beta_mr), float(se_mr), float(pval)


def ivw_fixed_effects(
    beta_exp: np.ndarray, se_exp: np.ndarray,
    beta_out: np.ndarray, se_out: np.ndarray,
) -> tuple[float, float, float]:
    """IVW fixed-effects (Burgess et al. 2013). k≥2 instruments."""
    w = 1.0 / se_out**2
    beta_mr = np.sum(w * beta_exp * beta_out) / np.sum(w * beta_exp**2)
    se_mr = 1.0 / np.sqrt(np.sum(w * beta_exp**2))
    z = beta_mr / se_mr
    pval = 2.0 * scipy.stats.norm.sf(abs(z))
    return float(beta_mr), float(se_mr), float(pval)


def cochrans_q(
    beta_exp: np.ndarray, se_exp: np.ndarray,
    beta_out: np.ndarray, se_out: np.ndarray,
    beta_ivw: float,
) -> tuple[float, float]:
    """Cochran's Q for heterogeneity. Returns (q_stat, q_pval). k≥2."""
    w = 1.0 / se_out**2
    q = float(np.sum(w * (beta_out - beta_ivw * beta_exp) ** 2))
    df = len(beta_exp) - 1
    q_pval = float(scipy.stats.chi2.sf(q, df=df))
    return q, q_pval


def ivw_random_effects(
    beta_exp: np.ndarray, se_exp: np.ndarray,
    beta_out: np.ndarray, se_out: np.ndarray,
) -> tuple[float, float, float]:
    """IVW random-effects (DerSimonian-Laird)."""
    fe_beta, _, _ = ivw_fixed_effects(beta_exp, se_exp, beta_out, se_out)
    q, _ = cochrans_q(beta_exp, se_exp, beta_out, se_out, fe_beta)

    k = len(beta_exp)
    w = 1.0 / se_out**2

    c = np.sum(w * beta_exp**2) - np.sum((w * beta_exp**2) ** 2) / np.sum(w * beta_exp**2)
    tau2 = max(0.0, (q - (k - 1)) / c)

    w_re = 1.0 / (se_out**2 + tau2)
    beta_mr = np.sum(w_re * beta_exp * beta_out) / np.sum(w_re * beta_exp**2)
    se_mr = 1.0 / np.sqrt(np.sum(w_re * beta_exp**2))
    z = beta_mr / se_mr
    pval = 2.0 * scipy.stats.norm.sf(abs(z))
    return float(beta_mr), float(se_mr), float(pval)


def mr_egger(
    beta_exp: np.ndarray, se_exp: np.ndarray,
    beta_out: np.ndarray, se_out: np.ndarray,
) -> dict:
    """MR-Egger regression. k≥3."""
    sign = np.sign(beta_exp)
    sign[sign == 0] = 1
    bx = beta_exp * sign
    by = beta_out * sign

    w = 1.0 / se_out**2
    X = np.column_stack([np.ones(len(bx)), bx])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ by

    try:
        coef = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return {
            "intercept": np.nan, "intercept_se": np.nan, "intercept_pval": np.nan,
            "slope": np.nan, "slope_se": np.nan, "slope_pval": np.nan,
        }

    resid = by - X @ coef
    k = len(bx)
    sigma2 = float(np.sum(w * resid**2) / (k - 2))

    cov_matrix = sigma2 * np.linalg.inv(XtWX)
    se_coef = np.sqrt(np.diag(cov_matrix))

    intercept, slope = float(coef[0]), float(coef[1])
    se_intercept, se_slope = float(se_coef[0]), float(se_coef[1])

    t_intercept = intercept / se_intercept if se_intercept > 0 else 0.0
    t_slope = slope / se_slope if se_slope > 0 else 0.0

    df = k - 2
    intercept_pval = float(2 * scipy.stats.t.sf(abs(t_intercept), df=df))
    slope_pval = float(2 * scipy.stats.t.sf(abs(t_slope), df=df))

    return {
        "intercept": intercept, "intercept_se": se_intercept,
        "intercept_pval": intercept_pval,
        "slope": slope, "slope_se": se_slope, "slope_pval": slope_pval,
    }


def weighted_median(
    beta_exp: np.ndarray, se_exp: np.ndarray,
    beta_out: np.ndarray, se_out: np.ndarray,
    n_boot: int = 1000, seed: int = 42,
) -> tuple[float, float, float]:
    """Weighted median estimator. k≥3."""
    beta_iv = beta_out / beta_exp
    weights = 1.0 / (se_out**2 / beta_exp**2)

    order = np.argsort(beta_iv)
    beta_sorted = beta_iv[order]
    w_sorted = weights[order]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    idx = np.searchsorted(cum_w, total_w / 2.0)
    idx = min(idx, len(beta_sorted) - 1)
    beta_wm = float(beta_sorted[idx])

    rng = np.random.default_rng(seed)
    boot_betas = np.empty(n_boot)
    for b in range(n_boot):
        bx_boot = beta_exp + rng.normal(0, se_exp)
        by_boot = beta_out + rng.normal(0, se_out)
        biv_boot = by_boot / bx_boot
        w_boot = 1.0 / (se_out**2 / bx_boot**2)
        o = np.argsort(biv_boot)
        cw = np.cumsum(w_boot[o])
        i = np.searchsorted(cw, cw[-1] / 2.0)
        i = min(i, len(biv_boot) - 1)
        boot_betas[b] = biv_boot[o][i]

    se_wm = float(np.std(boot_betas, ddof=1))
    if se_wm > 0:
        z = beta_wm / se_wm
        pval = float(2 * scipy.stats.norm.sf(abs(z)))
    else:
        pval = 1.0

    return beta_wm, se_wm, pval


def steiger_test(
    r2_exp: float, r2_out: float, n_exp: int, n_out: int,
    trait_type: str = "quantitative",
    n_cases: int | None = None, n_controls: int | None = None,
    population_prevalence: float | None = None,
) -> tuple[float, bool]:
    """Steiger directionality test. Returns (p_value, direction_valid).

    For a binary outcome the observed-scale R² is converted to the
    liability scale using the Lee et al. 2011 transform - but ONLY when a true
    *population* prevalence ``K`` is supplied. The sample case fraction
    ``n_cases/(n_cases+n_controls)`` is an ascertainment proportion, NOT the
    population prevalence (for SCZ ≈0.33 vs ≈0.01), so using it here would badly
    mis-scale R². When ``population_prevalence`` is None the test stays on the
    observed/binary-approximation scale (annotate-only ).

        R²_liability = R²_obs · [K(1-K)]² / (z² · P(1-P))
        z = φ(Φ⁻¹(1-K)),  P = sample case fraction,  K = population prevalence
    """
    if (
        trait_type == "case_control"
        and population_prevalence is not None
        and n_cases is not None and n_controls is not None
    ):
        K = float(population_prevalence)
        P = n_cases / (n_cases + n_controls)
        z_thresh = scipy.stats.norm.ppf(1 - K)
        height = scipy.stats.norm.pdf(z_thresh)
        if height > 0 and 0 < P < 1:
            r2_out = r2_out * (K * (1 - K)) ** 2 / (height**2 * P * (1 - P))
            r2_out = min(r2_out, 0.999)

    z_exp = 0.5 * np.log((1 + np.sqrt(r2_exp)) / (1 - np.sqrt(r2_exp)))
    z_out = 0.5 * np.log((1 + np.sqrt(r2_out)) / (1 - np.sqrt(r2_out)))

    diff = z_exp - z_out
    se_diff = np.sqrt(1 / (n_exp - 3) + 1 / (n_out - 3))

    if se_diff > 0:
        z_stat = diff / se_diff
        pval = float(2 * scipy.stats.norm.sf(abs(z_stat)))
    else:
        pval = 1.0

    direction_valid = r2_exp > r2_out

    return pval, direction_valid


# ---------------------------------------------------------------------------
# Colocalisation (native coloc.abf)
# ---------------------------------------------------------------------------


class ColocConfigurationError(RuntimeError):
    """Invalid coloc calibration configuration that should abort the run."""


def _has_valid_case_control_count(value: object) -> bool:
    """Return True for finite, positive case/control sample counts."""
    if value is None:
        return False
    try:
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric > 0 and numeric.is_integer())
    except (TypeError, ValueError):
        return False


def _validate_coloc_calibration_config(
    config: MRConfig,
    gwas_metadata: GWASMetadata,
) -> None:
    """Preflight the coloc calibration so invalid opt-in modes fail once.

    The default reported-SE path needs no case/control metadata. The opt-in
    case/control approximation is only valid for binary traits with valid
    case/control counts; otherwise it must not be downgraded to per-gene coloc
    missingness.
    """
    if getattr(config, "coloc_variance_mode", "reported_se") != "case_control_approx":
        return

    trait_type = getattr(gwas_metadata, "trait_type", "unknown")
    if trait_type != "case_control":
        logger.warning(
            "coloc_variance_mode='case_control_approx' requested for trait_type=%r; "
            "case/control variance is undefined, so coloc will use reported GWAS SE.",
            trait_type,
        )
        return

    n_cases = getattr(gwas_metadata, "n_cases", None)
    n_controls = getattr(gwas_metadata, "n_controls", None)
    if not (
        _has_valid_case_control_count(n_cases)
        and _has_valid_case_control_count(n_controls)
    ):
        raise ColocConfigurationError(
            "coloc_variance_mode='case_control_approx' requires finite positive "
            "n_cases and n_controls in the GWAS metadata. Set "
            "study.n_cases / study.n_controls, or use coloc_variance_mode='reported_se'."
        )


def _resolve_coloc_variance_inputs(
    config: MRConfig,
    trait_type: str,
    n_cases: int | None,
    n_controls: int | None,
    n_out: int,
) -> tuple[float | None, int]:
    """Decide coloc's outcome-variance inputs (s2, n2).

    - ``reported_se`` (default): return ``(None, n_out)`` so ``coloc_abf`` uses
      the GWAS BETA/SE (V2 = se2**2). Byte-stable with previous output.
    - ``case_control_approx``: only for a case/control trait with both
      ``n_cases`` and ``n_controls``; return ``(s2, n_cases+n_controls)`` where
      ``s2`` is the sample case fraction and ``n2`` is the TOTAL sample size -
      never Neff (which ``n_out`` may be via the NEFF->N alias). If the mode is
      requested but the counts are missing, fail loud rather than silently using
      the wrong variance.
    """
    if getattr(config, "coloc_variance_mode", "reported_se") != "case_control_approx":
        return None, n_out
    if trait_type != "case_control":
        # Quantitative trait: the case/control variance is undefined; use SE.
        return None, n_out
    if not (
        _has_valid_case_control_count(n_cases)
        and _has_valid_case_control_count(n_controls)
    ):
        raise ColocConfigurationError(
            "coloc_variance_mode='case_control_approx' requires finite positive "
            "n_cases and n_controls in the GWAS metadata. Set "
            "study.n_cases / study.n_controls, or use coloc_variance_mode='reported_se'."
        )
    cases = int(float(n_cases))
    controls = int(float(n_controls))
    total = cases + controls
    return cases / total, total


def coloc_abf(
    beta1: np.ndarray, se1: np.ndarray,
    beta2: np.ndarray, se2: np.ndarray,
    maf: np.ndarray,
    n1: int, n2: int,
    type1: str = "quant", type2: str = "cc",
    s2: float | None = None,
    p1: float = 1e-4, p2: float = 1e-4, p12: float = 1e-5,
    prior_var1: float = 0.15**2, prior_var2: float = 0.15**2,
) -> dict:
    """Native reimplementation of coloc.abf (Giambartolomei et al. 2014).

    All computation in log space to avoid numerical overflow.
    """
    n_snps = len(beta1)
    if n_snps == 0:
        return {
            "pp_h0": 1.0, "pp_h1": 0.0, "pp_h2": 0.0, "pp_h3": 0.0,
            "pp_h4": 0.0, "n_snps": 0, "snp_pp_h4": np.array([]),
        }

    z1 = beta1 / se1
    z2 = beta2 / se2

    V1 = se1**2
    if type2 == "cc" and s2 is not None:
        V2 = 1.0 / (2 * n2 * maf * (1 - maf) * s2 * (1 - s2))
    else:
        V2 = se2**2

    W1 = prior_var1
    W2 = prior_var2

    r1 = W1 / (W1 + V1)
    r2 = W2 / (W2 + V2)

    labf1 = 0.5 * np.log(1 - r1) + 0.5 * r1 * z1**2
    labf2 = 0.5 * np.log(1 - r2) + 0.5 * r2 * z2**2

    lsum1 = scipy.special.logsumexp(labf1)
    lsum2 = scipy.special.logsumexp(labf2)
    lsum12 = scipy.special.logsumexp(labf1 + labf2)

    lp1 = np.log(p1)
    lp2 = np.log(p2)
    lp12 = np.log(p12)
    log1mp = np.log(1 - p1 - p2 - p12) if (p1 + p2 + p12) < 1.0 else -np.inf

    lh0 = log1mp
    lh1 = lp1 + lsum1
    lh2 = lp2 + lsum2
    lh3 = lp1 + lp2 + lsum1 + lsum2
    lh4 = lp12 + lsum12

    all_lh = np.array([lh0, lh1, lh2, lh3, lh4])
    log_total = scipy.special.logsumexp(all_lh)
    pp = np.exp(all_lh - log_total)

    snp_log_h4 = labf1 + labf2
    snp_pp_h4 = np.exp(snp_log_h4 - scipy.special.logsumexp(snp_log_h4))

    return {
        "pp_h0": float(pp[0]),
        "pp_h1": float(pp[1]),
        "pp_h2": float(pp[2]),
        "pp_h3": float(pp[3]),
        "pp_h4": float(pp[4]),
        "n_snps": n_snps,
        "snp_pp_h4": snp_pp_h4,
    }


# ---------------------------------------------------------------------------
# Cross-source concordance
# ---------------------------------------------------------------------------


def annotate_cross_source(mr_results: pd.DataFrame) -> pd.DataFrame:
    """For each gene, compare MR results across sources."""
    if mr_results.empty:
        return mr_results

    sources = mr_results["eqtl_source"].unique()
    if len(sources) <= 1:
        mr_results["cross_source_status"] = "unavailable"
        return mr_results

    mr_results = mr_results.copy()
    mr_results["cross_source_status"] = "unavailable"

    for gene in mr_results["gene_ensembl_id"].unique():
        gene_mask = mr_results["gene_ensembl_id"] == gene
        gene_rows = mr_results.loc[gene_mask]

        if len(gene_rows) < 2:
            continue

        sig_rows = gene_rows.loc[gene_rows["mr_significant"]]
        if sig_rows.empty:
            mr_results.loc[gene_mask, "cross_source_status"] = "non_significant"
            continue

        primary = sig_rows.iloc[0]
        other_source = [s for s in sources if s != primary["eqtl_source"]]
        if not other_source:
            continue

        other_rows = gene_rows.loc[gene_rows["eqtl_source"] == other_source[0]]
        if other_rows.empty:
            mr_results.loc[gene_mask, "cross_source_status"] = "untestable"
            continue

        other = other_rows.iloc[0]

        if other["mr_pval"] < 0.05:
            if np.sign(primary["mr_beta"]) == np.sign(other["mr_beta"]):
                mr_results.loc[gene_mask, "cross_source_status"] = "concordant"
            else:
                mr_results.loc[gene_mask, "cross_source_status"] = "discordant"
        else:
            mr_results.loc[gene_mask, "cross_source_status"] = "non_significant"

    return mr_results


# ---------------------------------------------------------------------------
# Confidence tiering
# ---------------------------------------------------------------------------


def assign_confidence_tiers(
    mr_results: pd.DataFrame,
    require_coloc: bool = True,
    require_steiger: bool = False,
) -> pd.DataFrame:
    """Assign confidence_tier to each gene based on MR results.

    Must be called AFTER cross_source_status is populated.
    """
    if mr_results.empty:
        return mr_results

    mr_results = mr_results.copy()
    mr_results["confidence_tier"] = "low"

    for idx in mr_results.index:
        row = mr_results.loc[idx]

        if row.get("weak_instrument_excluded", False):
            continue

        if not row.get("mr_significant", False):
            continue

        if require_steiger and row.get("steiger_valid") == False and row.get("steiger_valid") is not None:  # noqa: E712
            mr_results.at[idx, "confidence_tier"] = "steiger_flagged"
            continue

        if not row.get("coloc_supported", False):
            mr_results.at[idx, "confidence_tier"] = "low"
            continue

        css = row.get("cross_source_status", "unavailable")
        if css == "concordant":
            mr_results.at[idx, "confidence_tier"] = "high"
        elif css == "discordant":
            mr_results.at[idx, "confidence_tier"] = "direction_conflict"
        else:
            mr_results.at[idx, "confidence_tier"] = "medium"

    return mr_results


# ---------------------------------------------------------------------------
# Directional drug matching
# ---------------------------------------------------------------------------


def _ensembl_no_version(value: object) -> str:
    """Strip an Ensembl version suffix (``ENSG...3`` -> ``ENSG...``)."""
    s = str(value) if value is not None else ""
    return s.split(".", 1)[0]


def _has_text(value: object) -> bool:
    """Null-safe non-empty-text check.

    Handles ``pd.NA`` (which raises on ``bool()``), ``np.nan``, ``None`` and ``""``.
    The drug loader emits ``pd.NA`` for ``mechanism_of_action`` / ``interaction_type``
    on affinity-only records, so scalar truthiness must never be taken directly.
    """
    try:
        if not bool(pd.notna(value)):
            return False
    except (TypeError, ValueError):
        return False
    return str(value).strip() != ""


def _resolve_drug_matches_by_id(
    gene_result: pd.Series,
    drug_targets: pd.DataFrame,
    config: MRDrugMatchConfig,
) -> tuple[pd.DataFrame, str]:
    """Resolve a gene's drug-target rows using the configured ID-join strategy.

    Returns ``(matches, match_via)``. ``match_via`` is one of
    ``ensembl`` / ``entrez`` / ``uniprot`` / ``symbol`` / ``""`` (no match).

    - ``legacy`` mode reproduces the previous behaviour: exact Ensembl, then a loose
      ``gene_symbol`` fallback (when ``allow_symbol_fallback``).
    - ``strict`` mode uses an exact-ID hierarchy - Ensembl (version-stripped) ->
      Entrez -> UniProt -> unambiguous symbol - and never merges distinct Ensembl
      IDs that merely share a symbol.
    """
    gene_ens = gene_result.get("gene_ensembl_id", "") or ""
    gene_sym = gene_result.get("gene_symbol", "") or ""

    if config.match_mode == "legacy":
        matches = drug_targets.loc[drug_targets["gene_ensembl_id"] == gene_ens]
        if not matches.empty:
            return matches, "ensembl"
        if config.allow_symbol_fallback and gene_sym:
            matches = drug_targets.loc[drug_targets["gene_symbol"] == gene_sym]
            if not matches.empty:
                return matches, "symbol"
        return matches, ""

    # --- strict mode: exact-ID hierarchy ---
    if "_ens_norm" in drug_targets.columns:
        ens_norm = drug_targets["_ens_norm"]
    else:
        ens_norm = drug_targets["gene_ensembl_id"].map(_ensembl_no_version)

    ens_key = _ensembl_no_version(gene_ens)
    if ens_key:
        matches = drug_targets.loc[ens_norm == ens_key]
        if not matches.empty:
            return matches, "ensembl"

    entrez = gene_result.get("gene_entrez_id")
    if entrez is not None and not (isinstance(entrez, float) and np.isnan(entrez)) and "gene_entrez_id" in drug_targets.columns:
        dt_entrez = pd.to_numeric(drug_targets["gene_entrez_id"], errors="coerce")
        matches = drug_targets.loc[dt_entrez == pd.to_numeric(entrez, errors="coerce")]
        if not matches.empty:
            return matches, "entrez"

    uniprot = gene_result.get("gene_uniprot_id")
    if uniprot and "gene_uniprot_id" in drug_targets.columns:
        matches = drug_targets.loc[drug_targets["gene_uniprot_id"].astype(str) == str(uniprot)]
        if not matches.empty:
            return matches, "uniprot"

    if config.allow_symbol_fallback and gene_sym:
        sym_matches = drug_targets.loc[drug_targets["gene_symbol"] == gene_sym]
        if not sym_matches.empty:
            # Accept a symbol match only when it maps to a single Ensembl gene,
            # so distinct paralogs (e.g. C4A/C4B) are never merged by shared symbol.
            sym_ens = ens_norm.loc[sym_matches.index] if "_ens_norm" in drug_targets.columns else sym_matches["gene_ensembl_id"].map(_ensembl_no_version)
            if sym_ens.nunique(dropna=True) <= 1:
                return sym_matches, "symbol"

    return drug_targets.iloc[0:0], ""


def _drug_match_rank_keys(config: MRDrugMatchConfig) -> tuple[list[str], list[bool]]:
    """Sort keys for within-gene drug ranking, per direction_policy."""
    if config.direction_policy == "prefer_inferable":
        return (
            ["direction_inferable", "has_mechanism_text", "max_phase", "pchembl_value"],
            [False, False, False, False],
        )
    # 'all' and 'inferable_only' rank by evidence quality (no direction preference).
    return (
        ["has_mechanism_text", "max_phase", "pchembl_value"],
        [False, False, False],
    )


def _build_drug_match_records(
    gene_result: pd.Series,
    raw_matches: pd.DataFrame,
    match_via: str,
    config: MRDrugMatchConfig,
) -> pd.DataFrame:
    """Apply drug-match filters and build the per-(gene, drug) record frame.

    Adds additive provenance/ranking columns (``match_via``, ``direction_inferable``,
    ``mechanism_of_action``, ``has_mechanism_text``, ``action_type``,
    ``drug_target_source``, ``drug_target_confidence``, ``drug_match_rank``). Legacy
    columns/values are preserved so default config reproduces the previous row set.
    """
    if raw_matches.empty:
        return pd.DataFrame()

    # require_druggable: gate on the gene's druggable_tier annotation.
    # Null-safe: druggable_tier may be pd.NA / NaN / None / "" for non-druggable genes.
    if config.require_druggable and not _has_text(gene_result.get("druggable_tier")):
        return pd.DataFrame()

    m = raw_matches

    # min_phase filter (with scope), mirroring Branch A semantics (max_phase >= x).
    if config.min_phase is not None and "max_phase" in m.columns:
        max_phase_num = pd.to_numeric(m["max_phase"], errors="coerce").fillna(0)
        if config.phase_filter_scope == "chembl_only" and "source" in m.columns:
            # Token/contains check (mirrors Branch A drug_enrichment): the loader
            # merges provenance into compound strings like "chembl,dgidb", so a
            # ChEMBL-sourced drug must still be subject to the phase filter.
            is_chembl = m["source"].astype(str).str.contains("chembl", case=False, na=False)
            keep = (~is_chembl) | (max_phase_num >= config.min_phase)
        else:
            keep = max_phase_num >= config.min_phase
        m = m.loc[keep]

    # min_pchembl filter (null-pchembl rows retained).
    if config.min_pchembl is not None and "pchembl_value" in m.columns:
        pch = pd.to_numeric(m["pchembl_value"], errors="coerce")
        m = m.loc[pch.isna() | (pch >= config.min_pchembl)]

    # direction_policy: drop direction-ambiguous rows when 'inferable_only'.
    if config.direction_policy == "inferable_only" and not m.empty:
        itypes = m["interaction_type"] if "interaction_type" in m.columns else pd.Series("other", index=m.index)
        m = m.loc[itypes.isin(INFERABLE_TYPES)]

    if m.empty:
        return pd.DataFrame()

    mr_beta = gene_result["mr_beta"]

    records = []
    for row in m.itertuples(index=False):
        itype = getattr(row, "interaction_type", "other")
        ambiguous = itype in AMBIGUOUS_TYPES
        inferable = itype in INFERABLE_TYPES

        if ambiguous:
            concordant = None
        elif mr_beta > 0:
            concordant = itype in DOWNREGULATING_TYPES
        elif mr_beta < 0:
            concordant = itype in UPREGULATING_TYPES
        else:
            concordant = None

        moa = getattr(row, "mechanism_of_action", None)
        # Null-safe: the drug loader emits pd.NA for affinity-only records, and
        # bool(pd.NA) raises TypeError - never take scalar truthiness directly.
        has_moa = _has_text(moa)

        records.append({
            "gene_ensembl_id": gene_result.get("gene_ensembl_id", ""),
            "gene_symbol": gene_result.get("gene_symbol", ""),
            "gene_entrez_id": gene_result.get("gene_entrez_id"),
            "gene_uniprot_id": gene_result.get("gene_uniprot_id"),
            "eqtl_source": gene_result["eqtl_source"],
            "mr_beta": mr_beta,
            "mr_pval": gene_result["mr_pval"],
            "pp_h4": gene_result.get("pp_h4"),
            "confidence_tier": gene_result.get("confidence_tier", "low"),
            "drug_chembl_id": getattr(row, "drug_chembl_id", ""),
            "drug_name": getattr(row, "drug_name", ""),
            "drug_inchikey": getattr(row, "drug_inchikey", None),
            "interaction_type": itype,
            "direction_concordant": concordant,
            "interaction_direction_ambiguous": ambiguous,
            "pchembl_value": getattr(row, "pchembl_value", None),
            "max_phase": getattr(row, "max_phase", 0),
            "atc_codes": getattr(row, "atc_codes", None),
            # --- additive provenance / ranking ---
            "action_type": getattr(row, "action_type", None),
            "mechanism_of_action": moa if has_moa else None,
            "has_mechanism_text": has_moa,
            "direction_inferable": inferable,
            "drug_target_source": getattr(row, "source", None),
            "drug_target_confidence": getattr(row, "confidence", None),
            "match_via": match_via or None,
            "drug_match_rank": None,
        })

    df = pd.DataFrame(records)

    # Within-gene rank (1 = best), by direction_policy; does not reorder rows.
    sort_by, ascending = _drug_match_rank_keys(config)
    sort_by = [c for c in sort_by if c in df.columns]
    if sort_by:
        order = df.sort_values(
            by=sort_by, ascending=ascending[: len(sort_by)],
            kind="mergesort", na_position="last",
        ).index
        rank_lookup = {idx: i + 1 for i, idx in enumerate(order)}
        df["drug_match_rank"] = df.index.map(rank_lookup)

    return df


def match_drugs_to_mr_gene(
    gene_result: pd.Series,
    drug_targets: pd.DataFrame,
    config: MRDrugMatchConfig | None = None,
) -> pd.DataFrame:
    """Match a single MR-significant gene to drugs from DrugTargetRecord.

    ``config`` controls the ID-join strategy and phase/potency/direction filters.
    When ``None`` (default), a default :class:`MRDrugMatchConfig` is used, which
    reproduces the previous legacy behaviour (exact Ensembl -> loose symbol fallback,
    no filtering); additive provenance columns are still populated.
    """
    if config is None:
        config = MRDrugMatchConfig()
    raw, via = _resolve_drug_matches_by_id(gene_result, drug_targets, config)
    return _build_drug_match_records(gene_result, raw, via, config)


def _summarise_gene_verdict(
    gene_result: pd.Series,
    raw_count: int,
    matches: pd.DataFrame,
    config: MRDrugMatchConfig,
) -> dict:
    """Build one per-gene drug-actionability verdict row.

    Emitted for every drug-match-eligible gene, including those with no passing
    drug, so absence of an actionable drug is a recorded result.
    """
    n_filtered = 0 if matches is None or matches.empty else len(matches)
    if n_filtered and "direction_inferable" in matches.columns:
        n_inferable = int(matches["direction_inferable"].fillna(False).astype(bool).sum())
    else:
        n_inferable = 0
    if n_filtered and "direction_concordant" in matches.columns:
        n_concordant = int((matches["direction_concordant"] == True).sum())  # noqa: E712
    else:
        n_concordant = 0

    if raw_count == 0:
        status = "no_drug_record"
    elif n_filtered == 0:
        status = "no_filtered_match"
    elif n_inferable > 0:
        status = "actionable"
    else:
        status = "binder_only"

    best = None
    if n_filtered:
        if "drug_match_rank" in matches.columns and matches["drug_match_rank"].notna().any():
            best = matches.sort_values("drug_match_rank", kind="mergesort").iloc[0]
        else:
            best = matches.iloc[0]

    def _best(col: str):
        if best is None or col not in matches.columns:
            return None
        val = best[col]
        if isinstance(val, float) and np.isnan(val):
            return None
        return val

    return {
        "gene_ensembl_id": gene_result.get("gene_ensembl_id", ""),
        "gene_symbol": gene_result.get("gene_symbol", ""),
        "gene_entrez_id": gene_result.get("gene_entrez_id"),
        "gene_uniprot_id": gene_result.get("gene_uniprot_id"),
        "eqtl_source": gene_result["eqtl_source"],
        "mr_beta": gene_result.get("mr_beta"),
        "mr_pval": gene_result.get("mr_pval"),
        "pp_h4": gene_result.get("pp_h4"),
        "confidence_tier": gene_result.get("confidence_tier", "low"),
        "verdict_status": status,
        "n_drug_records_raw": int(raw_count),
        "n_filtered_matches": int(n_filtered),
        "n_direction_inferable": n_inferable,
        "n_direction_concordant": n_concordant,
        "best_max_phase": _best("max_phase"),
        "best_pchembl": _best("pchembl_value"),
        "best_match_via": _best("match_via"),
        "best_drug_chembl_id": _best("drug_chembl_id"),
        "best_drug_name": _best("drug_name"),
        "druggable_tier": gene_result.get("druggable_tier"),
    }


def _bh_fdr_finite(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values over finite p-values only.

    ``multipletests`` propagates NaN into every q-value, so a single
    non-finite p-value would spoil the whole group. We mask non-finite
    values, apply BH to the valid subset, and leave the rest NaN. Mirrors
    ``negative_correlation._bh_fdr_per_group``.
    """
    arr = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return out
    _, q_valid, _, _ = multipletests(arr[valid], method="fdr_bh")
    out.iloc[np.flatnonzero(valid)] = q_valid
    return out


def _annotate_fdr_track(
    mr_results: pd.DataFrame,
    *,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Additive per-source BH-FDR + tested-Bonferroni sensitivity.

    The primary confirmatory call (``mr_significant`` at the source-eligible
    Bonferroni threshold) is never touched. This adds, per ``eqtl_source``
    and over *finite* MR p-values only:

      - ``mr_fdr_bh_q`` / ``mr_significant_fdr_bh`` - discovery-regime view;
      - ``bonferroni_threshold_tested`` / ``mr_significant_bonferroni_tested`` -
        Bonferroni recomputed over genes that actually emitted a p-value
        (a secondary, less-conservative denominator than the source-eligible one).

    Rows with a non-finite ``mr_pval`` get NaN q / False significance.
    """
    if mr_results.empty or "mr_pval" not in mr_results.columns:
        mr_results["mr_fdr_bh_q"] = np.nan
        mr_results["mr_significant_fdr_bh"] = False
        mr_results["bonferroni_threshold_tested"] = np.nan
        mr_results["mr_significant_bonferroni_tested"] = False
        return mr_results

    q = pd.Series(np.nan, index=mr_results.index, dtype=float)
    bonf_tested = pd.Series(np.nan, index=mr_results.index, dtype=float)
    pval_num = pd.to_numeric(mr_results["mr_pval"], errors="coerce")

    for _source, grp in mr_results.groupby("eqtl_source", sort=False):
        q.loc[grp.index] = _bh_fdr_finite(grp["mr_pval"])
        n_tested = int(np.isfinite(pd.to_numeric(grp["mr_pval"], errors="coerce")).sum())
        if n_tested > 0:
            bonf_tested.loc[grp.index] = fdr_alpha / n_tested

    mr_results["mr_fdr_bh_q"] = q
    mr_results["mr_significant_fdr_bh"] = (q < fdr_alpha).fillna(False)
    mr_results["bonferroni_threshold_tested"] = bonf_tested
    mr_results["mr_significant_bonferroni_tested"] = (
        pval_num < bonf_tested
    ).fillna(False)
    return mr_results


# Best-effort eQTL-source -> gene-coordinate build map for the *approximate*,
# opt-in coordinate fallback only. The Ensembl-membership primary path needs
# no build assumption.
_EQTL_SOURCE_BUILD = {"eqtlgen": "GRCh37", "metabrain": "GRCh38", "gtex": "GRCh38"}


def _annotate_mhc_flag(
    mr_results: pd.DataFrame,
    config: MRConfig,
    reference_config: ReferenceConfig,
    gene_id_converter,
) -> tuple[pd.DataFrame, dict]:
    """Additive MHC-region flag on ``mr_results``.

    Primary path: build-invariant Ensembl-ID membership using the shared Branch B
    annotation machinery (``repogen.data.mhc_annotation``). No gene coordinates
    required, so it is immune to the eQTL-loader's unreliable min-SNP-position
    gene anchor. When the annotation cannot be loaded:

      - ``allow_mhc_coordinate_fallback=False`` (default) -> fail loud;
      - ``allow_mhc_coordinate_fallback=True`` -> approximate per-source
        coordinate flag from ``gene_chr``/``gene_start`` (clearly labelled).

    Primary ``mr_results``/``mr_significant``/coloc/drug matching are untouched;
    this only adds ``mhc_flag`` + ``mhc_flag_method``. Returns (df, metadata).
    """
    mhc_cfg = config.mhc_sensitivity
    meta: dict = {"enabled": bool(mhc_cfg.enabled)}
    if not mhc_cfg.enabled:
        return mr_results, meta

    allow_coord = mhc_cfg.allow_mhc_coordinate_fallback
    annotation, ann_meta = load_mhc_gene_annotation(
        reference_config, gene_id_converter,
        require_mhc_annotation=not allow_coord,
    )
    meta.update(ann_meta)

    if annotation is not None and not annotation.empty:
        mr_results = add_mhc_flag(mr_results, annotation, build="GRCh38")
        mr_results["mhc_flag_method"] = "ensembl_membership"
        meta["mhc_flag_method"] = "ensembl_membership"
        meta["n_mhc_flagged"] = int(mr_results["mhc_flag"].sum())
        return mr_results, meta

    # No canonical annotation. With fail-loud default we would already have
    # raised inside load_mhc_gene_annotation; reaching here means the operator
    # opted into the approximate coordinate fallback.
    if not allow_coord:
        raise RuntimeError(
            "mr.mhc_sensitivity.enabled=True but no MHC gene annotation could be "
            "loaded and allow_mhc_coordinate_fallback=False. Provide "
            "reference.gene_loc_file_grch38 (recommended) or opt into the "
            "approximate coordinate fallback."
        )

    logger.warning(
        "MHC Ensembl annotation unavailable; using APPROXIMATE per-source "
        "coordinate fallback. Flags derive from the eQTL-loader gene "
        "anchor (possibly min-SNP-position) and may mis-flag boundary genes.",
    )
    has_coords = {"gene_chr", "gene_start"}.issubset(mr_results.columns)
    flag = pd.Series(False, index=mr_results.index)
    if has_coords:
        chr_num = pd.to_numeric(mr_results["gene_chr"], errors="coerce")
        pos = pd.to_numeric(mr_results["gene_start"], errors="coerce")
        for src, grp in mr_results.groupby("eqtl_source", sort=False):
            build = _EQTL_SOURCE_BUILD.get(str(src).lower(), "GRCh37")
            mhc_chr, mhc_start, mhc_end = mhc_interval(build)
            in_mhc = (
                (chr_num.loc[grp.index] == mhc_chr)
                & (pos.loc[grp.index] >= mhc_start)
                & (pos.loc[grp.index] <= mhc_end)
            )
            flag.loc[grp.index] = in_mhc.fillna(False)
    else:
        logger.warning(
            "Coordinate fallback requested but gene_chr/gene_start missing; "
            "mhc_flag left False for all rows.",
        )
    mr_results = mr_results.copy()
    mr_results["mhc_flag"] = flag.to_numpy()
    mr_results["mhc_flag_method"] = "coordinate_fallback_approximate"
    meta["mhc_flag_method"] = "coordinate_fallback_approximate"
    meta["n_mhc_flagged"] = int(flag.sum())
    return mr_results, meta


def _write_mhc_excluded_sensitivity(
    output_dir: Path,
    mr_results: pd.DataFrame,
    mr_drug_matches: pd.DataFrame,
    mr_target_verdicts: pd.DataFrame,
) -> dict:
    """Emit an MHC-excluded sensitivity view.

    Writes ``mr/sensitivity/mhc_excluded/`` copies of the three MR tables with
    MHC-flagged genes removed, using the primary thresholds (no denominator
    recomputation). Returns a counts summary that is also persisted as JSON.
    """
    sens_dir = output_dir / "sensitivity" / "mhc_excluded"
    sens_dir.mkdir(parents=True, exist_ok=True)

    non_mhc = mr_results.loc[~mr_results["mhc_flag"].fillna(False)].copy()
    mhc_genes = set(
        mr_results.loc[mr_results["mhc_flag"].fillna(False), "gene_ensembl_id"]
    )

    def _drop_mhc(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "gene_ensembl_id" not in df.columns:
            return df if df is not None else pd.DataFrame()
        return df.loc[~df["gene_ensembl_id"].isin(mhc_genes)].copy()

    drugs_ex = _drop_mhc(mr_drug_matches)
    verdicts_ex = _drop_mhc(mr_target_verdicts)

    non_mhc.to_parquet(sens_dir / "mr_results.parquet", engine="pyarrow", index=False)
    (drugs_ex if drugs_ex is not None else pd.DataFrame()).to_parquet(
        sens_dir / "mr_drug_matches.parquet", engine="pyarrow", index=False,
    )
    (verdicts_ex if verdicts_ex is not None else pd.DataFrame()).to_parquet(
        sens_dir / "mr_target_verdicts.parquet", engine="pyarrow", index=False,
    )

    def _count_sig(df: pd.DataFrame) -> int:
        return int(df["mr_significant"].sum()) if "mr_significant" in df.columns else 0

    def _count_coloc(df: pd.DataFrame) -> int:
        return int(df.get("coloc_supported", pd.Series(dtype=bool)).fillna(False).sum())

    def _count_high(df: pd.DataFrame) -> int:
        if "confidence_tier" not in df.columns:
            return 0
        return int((df["confidence_tier"] == "high").sum())

    summary = {
        "n_mhc_flagged_genes": int(len(mhc_genes)),
        "n_genes_total": int(mr_results["gene_ensembl_id"].nunique()),
        "n_genes_non_mhc": int(non_mhc["gene_ensembl_id"].nunique()),
        "primary": {
            "n_significant": _count_sig(mr_results),
            "n_colocalised": _count_coloc(mr_results),
            "n_high_confidence": _count_high(mr_results),
            "n_drug_matched": (
                int(mr_drug_matches["gene_ensembl_id"].nunique())
                if mr_drug_matches is not None and not mr_drug_matches.empty
                and "gene_ensembl_id" in mr_drug_matches.columns else 0
            ),
        },
        "mhc_excluded": {
            "n_significant": _count_sig(non_mhc),
            "n_colocalised": _count_coloc(non_mhc),
            "n_high_confidence": _count_high(non_mhc),
            "n_drug_matched": (
                int(drugs_ex["gene_ensembl_id"].nunique())
                if drugs_ex is not None and not drugs_ex.empty
                and "gene_ensembl_id" in drugs_ex.columns else 0
            ),
        },
        "threshold_policy": "primary_thresholds_reused (no non-MHC denominator recompute)",
    }
    with open(sens_dir / "mhc_excluded_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _annotate_druggable_track(
    mr_results: pd.DataFrame,
    config: MRDrugMatchConfig,
) -> pd.DataFrame:
    """Optional druggable-genome annotation + secondary significance track.

    When ``druggable_genome_path`` is configured, adds (additively):
      - ``druggable_tier`` from the static resource (NA for non-druggable genes),
      - ``bonferroni_threshold_druggable`` / ``mr_significant_druggable`` - Bonferroni
        recomputed over the druggable genes *actually tested* per source.

    The primary ``mr_significant`` flag and the genome-wide denominator are never
    overwritten. When no resource is configured, mr_results is returned unchanged so
    the default output schema is preserved (druggable_tier stays absent).
    """
    path = config.druggable_genome_path
    if path is None:
        return mr_results
    path = Path(path)
    # Fail loud when the resource is configured but unusable. Silently
    # skipping would, under require_druggable=True, gate away every drug match and
    # masquerade as a valid "no actionable drug" result.
    if not path.exists():
        raise FileNotFoundError(
            f"Druggable-genome resource configured at {path} but not found. "
            f"Set mr.drug_match.druggable_genome_path to a valid TSV or unset it."
        )

    dg = pd.read_csv(path, sep="\t")
    if "gene_ensembl_id" not in dg.columns or "druggable_tier" not in dg.columns:
        raise ValueError(
            f"Druggable-genome resource {path} missing required columns "
            f"(gene_ensembl_id, druggable_tier); got {list(dg.columns)}."
        )

    tier_map = dict(
        zip(dg["gene_ensembl_id"].map(_ensembl_no_version), dg["druggable_tier"].astype(str))
    )
    mr_results = mr_results.copy()
    mr_results["druggable_tier"] = (
        mr_results["gene_ensembl_id"].map(_ensembl_no_version).map(tier_map)
    )
    n_annot = int(mr_results["druggable_tier"].notna().sum())
    logger.info(
        "Annotated %d / %d MR rows with druggable_tier from %s",
        n_annot, len(mr_results), path,
    )

    # Secondary track: Bonferroni over druggable genes actually tested per source.
    mr_results["bonferroni_threshold_druggable"] = np.nan
    mr_results["mr_significant_druggable"] = pd.Series(
        pd.NA, index=mr_results.index, dtype="boolean"
    )
    for _source, grp in mr_results.groupby("eqtl_source"):
        druggable_tested = grp["druggable_tier"].notna() & grp["mr_pval"].notna()
        n_dr = int(druggable_tested.sum())
        if n_dr == 0:
            continue
        thr = 0.05 / n_dr
        idx = grp.index[druggable_tested]
        mr_results.loc[idx, "bonferroni_threshold_druggable"] = thr
        mr_results.loc[idx, "mr_significant_druggable"] = (
            mr_results.loc[idx, "mr_pval"] < thr
        ).astype("boolean")
    return mr_results


# ---------------------------------------------------------------------------
# Per-gene MR runner
# ---------------------------------------------------------------------------


def _run_mr_for_gene(
    gene: str,
    gene_info: dict,
    eqtl_df: pd.DataFrame,
    gwas_df: pd.DataFrame,
    gwas_metadata: GWASMetadata,
    config: MRConfig,
    bfile_full_path: Path,
    plink_binary: Path,
    source_name: str,
    bonf_threshold: float,
    skip_coloc: bool = False,
) -> dict | None:
    """Run MR pipeline for a single gene. Returns result dict or None.

    Parameters
    ----------
    skip_coloc : bool
        If True, skip inline coloc (used by chunked orchestrator which
        handles coloc in a separate Phase 3 pass).
    """
    instruments = select_instruments(
        eqtl_df, gene,
        cis_window_kb=config.cis_window_kb,
        gene_start=gene_info["start"],
        gene_chr=gene_info["chr"],
        pval_threshold=config.instrument_pval,
        f_stat_threshold=config.f_stat_threshold,
    )

    if instruments.empty:
        return None

    instruments = clump_instruments(
        instruments, bfile_full_path,
        clump_r2=config.clump_r2,
        clump_kb=config.cis_window_kb,
        plink_binary=plink_binary,
        gene_chr=gene_info.get("chr"),
    )

    if instruments.empty:
        return None

    harmonised = harmonise_gwas_eqtl(gwas_df, instruments)
    if harmonised.empty:
        return None

    k = len(harmonised)
    bx = harmonised["beta_exposure"].values
    sx = harmonised["se_exposure"].values
    by = harmonised["beta_outcome"].values
    sy = harmonised["se_outcome"].values

    f_stats = f_statistic(bx, sx)
    mean_f = float(np.mean(f_stats))

    if np.all(f_stats < config.f_stat_threshold):
        return {
            "gene_ensembl_id": gene,
            "gene_symbol": gene_info.get("symbol", ""),
            "gene_entrez_id": gene_info.get("entrez_id"),
            "gene_uniprot_id": gene_info.get("uniprot_id"),
            "gene_chr": gene_info.get("chr"),
            "gene_start": gene_info.get("start"),
            "eqtl_source": source_name,
            "n_instruments": k,
            "mr_method": "none",
            "mr_beta": np.nan, "mr_se": np.nan, "mr_pval": np.nan,
            "mr_significant": False,
            "bonferroni_threshold": bonf_threshold,
            "mean_f_stat": mean_f,
            "weak_instrument_excluded": True,
            "heterogeneity_warning": False,
            "coloc_supported": False,
            "coloc_status": "not_run",
        }

    result: dict = {
        "gene_ensembl_id": gene,
        "gene_symbol": gene_info.get("symbol", ""),
        "gene_entrez_id": gene_info.get("entrez_id"),
        "gene_uniprot_id": gene_info.get("uniprot_id"),
        "gene_chr": gene_info.get("chr"),
        "gene_start": gene_info.get("start"),
        "eqtl_source": source_name,
        "n_instruments": k,
        "mean_f_stat": mean_f,
        "weak_instrument_excluded": False,
        "bonferroni_threshold": bonf_threshold,
    }

    if k == 1:
        beta_mr, se_mr, pval = wald_ratio(bx[0], sx[0], by[0], sy[0])
        result["mr_method"] = "wald"
        result["mr_beta"] = beta_mr
        result["mr_se"] = se_mr
        result["mr_pval"] = pval
        result["heterogeneity_warning"] = False
    else:
        fe_beta, fe_se, fe_pval = ivw_fixed_effects(bx, sx, by, sy)
        result["ivw_fe_beta"] = fe_beta
        result["ivw_fe_pval"] = fe_pval

        q_stat, q_pval = cochrans_q(bx, sx, by, sy, fe_beta)
        result["q_stat"] = q_stat
        result["q_pval"] = q_pval
        result["heterogeneity_warning"] = q_pval < 0.05

        if q_pval < 0.05:
            re_beta, re_se, re_pval = ivw_random_effects(bx, sx, by, sy)
            result["mr_method"] = "ivw_re"
            result["mr_beta"] = re_beta
            result["mr_se"] = re_se
            result["mr_pval"] = re_pval
            result["ivw_re_beta"] = re_beta
            result["ivw_re_pval"] = re_pval
        else:
            result["mr_method"] = "ivw_fe"
            result["mr_beta"] = fe_beta
            result["mr_se"] = fe_se
            result["mr_pval"] = fe_pval

        if k >= 3:
            egger = mr_egger(bx, sx, by, sy)
            result["egger_intercept_pval"] = egger["intercept_pval"]
            result["egger_slope"] = egger["slope"]
            result["egger_se"] = egger["slope_se"]

            wm_b, wm_s, wm_p = weighted_median(bx, sx, by, sy)
            result["wm_beta"] = wm_b
            result["wm_se"] = wm_s
            result["wm_pval"] = wm_p

    result["mr_significant"] = result["mr_pval"] < bonf_threshold

    # Exposure/outcome sample sizes (used by Steiger and coloc Phase 3)
    n_exp = int(np.median(harmonised["n_exposure"].values)) if "n_exposure" in harmonised.columns else 31684
    n_out = int(np.median(harmonised["n_outcome"].values)) if "n_outcome" in harmonised.columns else 100000
    result["n_exposure_median"] = n_exp
    result["n_outcome_median"] = n_out

    maf_steiger = harmonised["maf"].values if "maf" in harmonised.columns else np.full(k, np.nan)
    steiger_maf_fallback_count = int((~np.isfinite(maf_steiger)).sum())
    maf_steiger = np.where(np.isfinite(maf_steiger), maf_steiger, 0.3)
    geno_var = 2.0 * maf_steiger * (1.0 - maf_steiger)
    r2_exp = float(np.sum(bx**2 * geno_var))
    r2_out = float(np.sum(by**2 * geno_var))

    r2_exp = min(r2_exp, 0.999)
    r2_out = min(r2_out, 0.999)

    trait_type = getattr(gwas_metadata, "trait_type", "quantitative")
    n_cases = None
    n_controls = None
    pop_prevalence = getattr(gwas_metadata, "population_prevalence", None)
    if trait_type == "case_control":
        n_cases = getattr(gwas_metadata, "n_cases", None)
        n_controls = getattr(gwas_metadata, "n_controls", None)

    # Record whether the liability-scale calibration fired.
    if trait_type != "case_control":
        result["steiger_binary_calibration"] = "not_applicable"
    elif pop_prevalence is not None and n_cases and n_controls:
        result["steiger_binary_calibration"] = "applied"
    elif n_cases and n_controls:
        result["steiger_binary_calibration"] = "skipped_no_prevalence"
    else:
        result["steiger_binary_calibration"] = "missing_n"

    try:
        s_pval, s_valid = steiger_test(
            r2_exp, r2_out, n_exp, n_out,
            trait_type=trait_type if trait_type in ("quantitative", "case_control") else "quantitative",
            n_cases=n_cases, n_controls=n_controls,
            population_prevalence=pop_prevalence,
        )
        result["steiger_pval"] = s_pval
        result["steiger_valid"] = s_valid
        result["r2_exposure"] = r2_exp
        result["r2_outcome"] = r2_out
        result["steiger_maf_fallback_count"] = steiger_maf_fallback_count
    except (ValueError, ZeroDivisionError):
        pass

    # Colocalisation (skipped when chunked orchestrator handles it in Phase 3)
    result["coloc_supported"] = False
    result["coloc_status"] = "not_run"

    if not skip_coloc and config.coloc_enabled and result.get("mr_significant", False):
        try:
            eqtl_gene = eqtl_df.loc[eqtl_df["gene"] == gene].copy()
            gwas_cols = [c for c in ["A1", "A2", "BETA", "SE", "P", "N", "CHR", "POS", "MAF"] if c in gwas_df.columns]
            shared = _merge_eqtl_gwas_two_stage(
                eqtl_gene, gwas_df, gwas_cols, suffixes=("_eqtl", "_gwas"),
            )

            if len(shared) < config.min_coloc_snps:
                result["coloc_status"] = "insufficient_data"
                result["n_snps_coloc"] = len(shared)
            else:
                maf_vals = shared["MAF"].values if "MAF" in shared.columns else np.full(len(shared), 0.3)
                coloc_maf_fallback_count = int((~np.isfinite(maf_vals)).sum())
                maf_vals = np.where(np.isfinite(maf_vals), maf_vals, 0.3)
                maf_vals = np.clip(maf_vals, 0.01, 0.49)

                # Only derive V2 from N/MAF/s2 when explicitly
                # opted in. Default 'reported_se' keeps s2=None -> coloc uses the
                # GWAS SE (byte-stable). In 'case_control_approx', n2 must be the
                # TOTAL sample size (cases+controls), never Neff.
                s2_val, coloc_n2 = _resolve_coloc_variance_inputs(
                    config, trait_type, n_cases, n_controls, n_out,
                )

                coloc_result = coloc_abf(
                    beta1=shared["beta"].values if "beta" in shared.columns else shared["beta_eqtl"].values,
                    se1=shared["se"].values if "se" in shared.columns else shared["se_eqtl"].values,
                    beta2=shared["BETA"].values if "BETA" in shared.columns else shared["beta_gwas"].values,
                    se2=shared["SE"].values if "SE" in shared.columns else shared["se_gwas"].values,
                    maf=maf_vals,
                    n1=n_exp, n2=coloc_n2,
                    type1="quant", type2="cc" if trait_type == "case_control" else "quant",
                    s2=s2_val,
                    p1=config.coloc_prior_p1,
                    p2=config.coloc_prior_p2,
                    p12=config.coloc_prior_p12,
                )

                result["pp_h4"] = coloc_result["pp_h4"]
                result["pp_h3"] = coloc_result["pp_h3"]
                result["n_snps_coloc"] = coloc_result["n_snps"]
                result["coloc_supported"] = coloc_result["pp_h4"] >= config.coloc_pp_h4_threshold
                result["coloc_maf_fallback_count"] = coloc_maf_fallback_count

                if coloc_result["pp_h4"] >= config.coloc_pp_h4_threshold:
                    result["coloc_status"] = "colocalised"
                elif coloc_result["pp_h3"] > 0.5:
                    result["coloc_status"] = "distinct_signals"
                else:
                    result["coloc_status"] = "unsupported"

        except ColocConfigurationError:
            raise
        except (KeyError, ValueError) as e:
            logger.warning("Coloc failed for gene %s: %s", gene, e)
            result["coloc_status"] = "not_run"

    return result


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def _run_coloc_for_gene(
    gene: str,
    eqtl_gene_df: pd.DataFrame,
    gwas_df: pd.DataFrame,
    gwas_metadata: GWASMetadata,
    config: MRConfig,
    n_exp: int,
    n_out: int,
) -> dict:
    """Run colocalisation for a single MR-significant gene.

    Requires full-locus eQTL data (not just instruments).
    Returns dict of coloc fields to merge into gene result.
    """
    trait_type = getattr(gwas_metadata, "trait_type", "quantitative")
    n_cases = getattr(gwas_metadata, "n_cases", None)
    n_controls = getattr(gwas_metadata, "n_controls", None)

    gwas_cols = [c for c in ["A1", "A2", "BETA", "SE", "P", "N", "CHR", "POS", "MAF"]
                 if c in gwas_df.columns]
    shared = _merge_eqtl_gwas_two_stage(
        eqtl_gene_df, gwas_df, gwas_cols, suffixes=("_eqtl", "_gwas"),
    )

    if len(shared) < config.min_coloc_snps:
        return {
            "coloc_status": "insufficient_data",
            "coloc_supported": False,
            "n_snps_coloc": len(shared),
        }

    maf_vals = shared["MAF"].values if "MAF" in shared.columns else np.full(len(shared), 0.3)
    coloc_maf_fallback_count = int((~np.isfinite(maf_vals)).sum())
    maf_vals = np.where(np.isfinite(maf_vals), maf_vals, 0.3)
    maf_vals = np.clip(maf_vals, 0.01, 0.49)

    # Gate the case/control N-derived variance on the explicit
    # mode; default 'reported_se' passes s2=None (uses GWAS SE, byte-stable).
    s2_val, coloc_n2 = _resolve_coloc_variance_inputs(
        config, trait_type, n_cases, n_controls, n_out,
    )

    coloc_result = coloc_abf(
        beta1=shared["beta"].values if "beta" in shared.columns else shared["beta_eqtl"].values,
        se1=shared["se"].values if "se" in shared.columns else shared["se_eqtl"].values,
        beta2=shared["BETA"].values if "BETA" in shared.columns else shared["beta_gwas"].values,
        se2=shared["SE"].values if "SE" in shared.columns else shared["se_gwas"].values,
        maf=maf_vals,
        n1=n_exp, n2=coloc_n2,
        type1="quant", type2="cc" if trait_type == "case_control" else "quant",
        s2=s2_val,
        p1=config.coloc_prior_p1,
        p2=config.coloc_prior_p2,
        p12=config.coloc_prior_p12,
    )

    out = {
        "pp_h4": coloc_result["pp_h4"],
        "pp_h3": coloc_result["pp_h3"],
        "n_snps_coloc": coloc_result["n_snps"],
        "coloc_supported": coloc_result["pp_h4"] >= config.coloc_pp_h4_threshold,
        "coloc_maf_fallback_count": coloc_maf_fallback_count,
    }

    if coloc_result["pp_h4"] >= config.coloc_pp_h4_threshold:
        out["coloc_status"] = "colocalised"
    elif coloc_result["pp_h3"] > 0.5:
        out["coloc_status"] = "distinct_signals"
    else:
        out["coloc_status"] = "unsupported"

    return out


def _run_mr_for_gene_timed(**kwargs) -> tuple[dict | None, dict]:
    """Wrapper that times _run_mr_for_gene and returns (result, metrics)."""
    t0 = time.time()
    result = _run_mr_for_gene(**kwargs)
    elapsed = time.time() - t0
    return result, {"task_time_seconds": elapsed}


# Default Phase-2 yield floor applied to sources marked required=True when no
# explicit min_result_fraction is configured.
_DEFAULT_REQUIRED_MIN_RESULT_FRACTION = 0.10
# Below this yield we emit a WARNING for observability even when not failing.
_SOFT_WARN_RESULT_FRACTION = 0.20


def _enforce_source_yield(
    source_config: EQTLSourceConfig,
    result_fraction: float | None,
    genes_processed: int,
) -> None:
    """Guard against silent eQTL-source degradation.

    ``result_fraction`` is the Phase-2 yield: emitted MR rows / genes processed.
    Raises ``RuntimeError`` only when the source is marked ``required`` (using its
    ``min_result_fraction`` or the default floor) or when an explicit
    ``min_result_fraction`` is unmet. Otherwise emits a WARNING for low yield so
    exploratory sources and the eQTLGen-only default remain non-fatal.
    """
    source = source_config.source
    required = bool(getattr(source_config, "required", False))
    min_frac = getattr(source_config, "min_result_fraction", None)

    if genes_processed == 0 or result_fraction is None:
        if required:
            raise RuntimeError(
                f"Required eQTL source '{source}' processed 0 genes in Phase 2 "
                f"(no instruments survived loading/harmonisation). This is a hard "
                f"failure - check the source path, genome build, and SNP formatting."
            )
        logger.warning(
            "eQTL source '%s' processed 0 genes in Phase 2 (result_fraction=None).",
            source,
        )
        return

    threshold = min_frac if min_frac is not None else (
        _DEFAULT_REQUIRED_MIN_RESULT_FRACTION if required else None
    )
    if threshold is not None and result_fraction < threshold:
        reason = (
            "min_result_fraction"
            if min_frac is not None
            else "default floor for required source"
        )
        raise RuntimeError(
            f"eQTL source '{source}' Phase-2 yield {result_fraction:.2%} is below "
            f"the required minimum {threshold:.2%} ({reason}). Failing loud to "
            f"prevent silent source degradation."
        )

    if result_fraction < _SOFT_WARN_RESULT_FRACTION:
        logger.warning(
            "eQTL source '%s' Phase-2 yield is low (%.2f%%); not failing "
            "(source not marked required and no min_result_fraction set).",
            source, result_fraction * 100.0,
        )


def run_mendelian_randomisation(
    gwas_path: Path,
    gwas_metadata: GWASMetadata,
    drug_targets_path: Path,
    gene_id_converter,
    config: MRConfig,
    reference_config: ReferenceConfig,
    output_dir: Path,
    n_workers_cap: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cis-MR pipeline across all configured eQTL sources.

    Uses memory-safe 3-phase architecture:
      Phase 1: Chunked instrument extraction + gene metadata collection
      Phase 2: MR estimation (parallel when n_workers > 1)
      Phase 3: Targeted coloc reload for significant genes
      Phase 4: Assembly, drug matching, output

    Parameters
    ----------
    n_workers_cap : int | None
        Optional ceiling on effective workers (e.g., from Snakemake --threads).
        Effective workers = min(config.n_workers, n_workers_cap, cpu_count, n_tasks).

    Returns (mr_results_df, mr_drug_matches_df).
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    rss_start = _get_peak_rss_mb()

    plink_binary = _detect_plink_binary()

    bfile_full_path = reference_config.genome_dir / reference_config.bfile_prefix
    ref_freq_path = ensure_ref_freq(bfile_full_path, plink_binary)

    logger.info("Loading GWAS summary statistics from %s", gwas_path)
    gwas_df = pd.read_parquet(gwas_path)
    if config.coloc_enabled:
        _validate_coloc_calibration_config(config, gwas_metadata)

    maf_series, maf_counters = _resolve_maf(gwas_df, ref_freq_path=ref_freq_path)
    gwas_df["MAF"] = maf_series
    n_resolved = len(gwas_df) - maf_counters.get("unresolved", 0)
    logger.info(
        "MAF resolution: %d / %d variants resolved (%.1f%%) - sources: %s",
        n_resolved, len(gwas_df),
        100.0 * n_resolved / max(len(gwas_df), 1),
        ", ".join(f"{k}={v}" for k, v in maf_counters.items()),
    )

    all_results: list[dict] = []
    source_metadata: list[dict] = []

    for source_config in config.eqtl_sources:
        source_name = source_config.source
        logger.info("--- Phase 1: Chunked instrument loading for %s ---", source_name)

        # --- Phase 1: chunked instrument extraction ---
        load_stats: dict[str, int] = {}
        needs_freq = source_name.lower() == "eqtlgen"
        if needs_freq:
            instruments_df, gene_metadata, n_genes_valid = _load_eqtlgen_chunked(
                source_config.path, ref_freq_path, config.instrument_pval,
            )
        else:
            instruments_df, gene_metadata, n_genes_valid = _load_metabrain_chunked(
                source_config.path, config.instrument_pval, stats_out=load_stats,
            )

        bonf_threshold = 0.05 / n_genes_valid if n_genes_valid > 0 else 0.05

        # Annotate gene_info_map with gene_id_converter
        gene_info_map: dict[str, dict] = {}
        for g, meta in gene_metadata.items():
            info = {"chr": meta["chr"], "start": meta["start"],
                    "symbol": "", "entrez_id": None, "uniprot_id": None}
            if gene_id_converter is not None:
                try:
                    record = gene_id_converter.get_full_record(str(g), "ensembl")
                    # get_full_record returns keys 'entrez'/'uniprot'
                    # (not 'entrez_id'/'uniprot_id'); it can also return None when the
                    # gene is unknown. Reading the wrong keys left gene_entrez_id /
                    # gene_uniprot_id null across all Branch C outputs.
                    if record is not None:
                        info["symbol"] = record.get("symbol") or ""
                        info["entrez_id"] = record.get("entrez")
                        info["uniprot_id"] = record.get("uniprot")
                except (KeyError, ValueError):
                    pass
            gene_info_map[g] = info

        # --- Phase 2: MR estimation ---
        n_gene_tasks = len(instruments_df["gene"].unique()) if not instruments_df.empty else 0
        effective_workers = min(
            n_workers_cap if n_workers_cap is not None else config.n_workers,
            config.n_workers,
            os.cpu_count() or 4,
            max(n_gene_tasks, 1),
        )

        logger.info(
            "--- Phase 2: MR estimation for %s (%d genes, Bonf=%.2e, workers=%d) ---",
            source_name, n_genes_valid, bonf_threshold, effective_workers,
        )
        logger.info(
            "Phase 2 worker selection: config_n_workers=%d, threads_cap=%s, "
            "cpu_count=%d, n_gene_tasks=%d -> effective_workers=%d",
            config.n_workers, n_workers_cap, os.cpu_count() or 4,
            n_gene_tasks, effective_workers,
        )

        n_tested = 0
        n_significant = 0
        n_gene_tasks = 0
        task_times: list[tuple[str, float]] = []  # (gene_id, seconds)
        tasks_returned_none = 0
        phase2_t0 = time.time()

        if not instruments_df.empty:
            grouped = instruments_df.groupby("gene")
            gene_tasks = sorted(
                [(gene, gene_df) for gene, gene_df in grouped
                 if "chr" in gene_info_map.get(gene, {})
                 and "start" in gene_info_map.get(gene, {})],
                key=lambda x: (-len(x[1]), x[0]),
            )
            n_gene_tasks = len(gene_tasks)

            if effective_workers <= 1 or n_gene_tasks <= 1:
                # Sequential execution - exact baseline parity
                phase2_t0 = time.time()
                last_log_time = phase2_t0
                for task_idx, (gene, gene_instruments) in enumerate(gene_tasks):
                    gene_info = gene_info_map[gene]
                    n_tested += 1
                    result, metrics = _run_mr_for_gene_timed(
                        gene=gene,
                        gene_info=gene_info,
                        eqtl_df=gene_instruments,
                        gwas_df=gwas_df,
                        gwas_metadata=gwas_metadata,
                        config=config,
                        bfile_full_path=bfile_full_path,
                        plink_binary=plink_binary,
                        source_name=source_name,
                        bonf_threshold=bonf_threshold,
                        skip_coloc=True,
                    )
                    task_times.append((gene, metrics["task_time_seconds"]))
                    if result is not None:
                        all_results.append(result)
                        if result.get("mr_significant", False):
                            n_significant += 1
                    else:
                        tasks_returned_none += 1

                    now = time.time()
                    if (task_idx + 1) % 100 == 0 or (now - last_log_time) >= 60:
                        elapsed = now - phase2_t0
                        completed = task_idx + 1
                        throughput = completed / max(elapsed, 0.1) * 60
                        eta_s = (n_gene_tasks - completed) / max(throughput / 60, 0.001)
                        logger.info(
                            "  Phase 2 progress: %d/%d genes (%.1f%%), "
                            "%.0fs elapsed, %.1f genes/min, ETA %.0fs, "
                            "%d significant",
                            completed, n_gene_tasks,
                            100.0 * completed / n_gene_tasks,
                            elapsed, throughput, eta_s, n_significant,
                        )
                        last_log_time = now
            else:
                # Parallel execution via ThreadPoolExecutor (manual lifecycle
                # so fail-fast doesn't block on __exit__ shutdown(wait=True))
                phase2_t0 = time.time()
                last_log_time = phase2_t0
                results_by_idx: dict[int, dict | None] = {}
                completed_count = 0
                parallel_significant = 0

                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=effective_workers
                )
                try:
                    future_to_idx: dict[concurrent.futures.Future, int] = {}
                    for task_idx, (gene, gene_instruments) in enumerate(gene_tasks):
                        gene_info = gene_info_map[gene]
                        future = executor.submit(
                            _run_mr_for_gene_timed,
                            gene=gene,
                            gene_info=gene_info,
                            eqtl_df=gene_instruments,
                            gwas_df=gwas_df,
                            gwas_metadata=gwas_metadata,
                            config=config,
                            bfile_full_path=bfile_full_path,
                            plink_binary=plink_binary,
                            source_name=source_name,
                            bonf_threshold=bonf_threshold,
                            skip_coloc=True,
                        )
                        future_to_idx[future] = task_idx

                    for future in concurrent.futures.as_completed(future_to_idx):
                        task_idx = future_to_idx[future]
                        gene_name = gene_tasks[task_idx][0]
                        try:
                            result, metrics = future.result()
                        except Exception as e:
                            logger.error(
                                "Phase 2 FATAL: gene %s (source=%s) failed: %s",
                                gene_name, source_name, e,
                            )
                            raise RuntimeError(
                                f"MR failed for gene {gene_name} "
                                f"(source={source_name}): {e}"
                            ) from e

                        results_by_idx[task_idx] = result
                        task_times.append((gene_name, metrics["task_time_seconds"]))
                        completed_count += 1
                        if result is not None and result.get("mr_significant", False):
                            parallel_significant += 1
                        elif result is None:
                            tasks_returned_none += 1

                        now = time.time()
                        if completed_count % 100 == 0 or (now - last_log_time) >= 60:
                            elapsed = now - phase2_t0
                            throughput = completed_count / max(elapsed, 0.1) * 60
                            eta_s = (n_gene_tasks - completed_count) / max(
                                throughput / 60, 0.001
                            )
                            logger.info(
                                "  Phase 2 progress: %d/%d genes (%.1f%%), "
                                "%.0fs elapsed, %.1f genes/min, ETA %.0fs, "
                                "%d significant",
                                completed_count, n_gene_tasks,
                                100.0 * completed_count / n_gene_tasks,
                                elapsed, throughput, eta_s, parallel_significant,
                            )
                            last_log_time = now
                except BaseException:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)

                # Reassemble in deterministic order
                for idx in sorted(results_by_idx.keys()):
                    result = results_by_idx[idx]
                    n_tested += 1
                    if result is not None:
                        all_results.append(result)
                        if result.get("mr_significant", False):
                            n_significant += 1

        # Also record genes with valid data but no instruments (counted in Bonferroni)
        genes_with_instruments = set(instruments_df["gene"].unique()) if not instruments_df.empty else set()
        genes_without_instruments = set(gene_metadata.keys()) - genes_with_instruments
        for gene in genes_without_instruments:
            gene_info = gene_info_map.get(gene, {})
            if "chr" not in gene_info or "start" not in gene_info:
                continue
            n_tested += 1

        phase2_duration = time.time() - phase2_t0 if n_gene_tasks > 0 else 0.0
        phase2_genes_per_min = (n_gene_tasks / max(phase2_duration, 0.1)) * 60

        logger.info(
            "Phase 2 complete for %s: %d genes in Bonf denom, %d with instruments, "
            "%d MR-significant (%.1fs, %.1f genes/min)",
            source_name, n_genes_valid, len(genes_with_instruments), n_significant,
            phase2_duration, phase2_genes_per_min,
        )

        # --- Phase 3: Targeted coloc reload ---
        significant_genes = set()
        for r in all_results:
            if (
                r.get("eqtl_source") == source_name
                and r.get("mr_significant", False)
            ):
                significant_genes.add(r["gene_ensembl_id"])

        coloc_reload_triggered = (
            config.coloc_enabled and len(significant_genes) > 0
        )

        if coloc_reload_triggered:
            logger.info(
                "--- Phase 3: Coloc reload for %s (%d significant genes) ---",
                source_name, len(significant_genes),
            )

            coloc_df = _load_eqtl_coloc_genes(
                source_config.path, ref_freq_path if needs_freq else None,
                significant_genes, source_name,
            )

            if not coloc_df.empty:
                coloc_grouped = coloc_df.groupby("gene")
                for r in all_results:
                    if (
                        r.get("eqtl_source") != source_name
                        or not r.get("mr_significant", False)
                    ):
                        continue
                    gene = r["gene_ensembl_id"]
                    if gene not in coloc_grouped.groups:
                        r["coloc_status"] = "insufficient_data"
                        continue

                    eqtl_gene_df = coloc_grouped.get_group(gene)
                    n_exp = r.get("n_exposure_median", 31684)
                    n_out = r.get("n_outcome_median", 100000)

                    try:
                        coloc_out = _run_coloc_for_gene(
                            gene, eqtl_gene_df, gwas_df, gwas_metadata,
                            config, n_exp, n_out,
                        )
                        r.update(coloc_out)
                    except ColocConfigurationError:
                        raise
                    except (KeyError, ValueError) as e:
                        logger.warning("Coloc failed for gene %s: %s", gene, e)
                        r["coloc_status"] = "not_run"
            else:
                logger.warning("Coloc reload returned empty DataFrame for %s", source_name)
                for r in all_results:
                    if (
                        r.get("eqtl_source") == source_name
                        and r.get("mr_significant", False)
                    ):
                        r["coloc_status"] = "insufficient_data"

        task_time_values = [t for _, t in task_times]
        task_time_total = sum(task_time_values)
        if task_time_values:
            p50 = round(float(np.percentile(task_time_values, 50)), 3)
            p90 = round(float(np.percentile(task_time_values, 90)), 3)
            p99 = round(float(np.percentile(task_time_values, 99)), 3)
            sorted_tasks = sorted(task_times, key=lambda x: x[1], reverse=True)
            top_slowest = [
                {"gene": g, "time_seconds": round(t, 3)}
                for g, t in sorted_tasks[:5]
            ]
        else:
            p50 = p90 = p99 = None
            top_slowest = []

        # Phase-2 yield = emitted MR rows / genes processed. Denominator
        # edge case (0 genes) -> None; used by the source-yield guardrail below.
        result_rows_emitted = max(n_gene_tasks - tasks_returned_none, 0)
        phase2_result_fraction = (
            result_rows_emitted / n_gene_tasks if n_gene_tasks > 0 else None
        )

        source_metadata.append({
            "source": source_name,
            "n_genes_bonferroni_denominator": n_genes_valid,
            "bonferroni_threshold": bonf_threshold,
            "n_instruments_retained": len(instruments_df),
            "n_significant": n_significant,
            "coloc_reload_triggered": coloc_reload_triggered,
            "coloc_reload_genes_count": len(significant_genes) if coloc_reload_triggered else 0,
            "phase2_duration_seconds": round(phase2_duration, 2),
            "phase2_genes_processed": n_gene_tasks,
            "phase2_genes_per_minute": round(phase2_genes_per_min, 1),
            "phase2_task_time_total_seconds": round(task_time_total, 2),
            "phase2_task_time_p50_seconds": p50,
            "phase2_task_time_p90_seconds": p90,
            "phase2_task_time_p99_seconds": p99,
            "phase2_top_slowest_genes": top_slowest,
            "phase2_tasks_returned_none": tasks_returned_none,
            "phase2_result_rows_emitted": result_rows_emitted,
            "phase2_result_fraction": (
                round(phase2_result_fraction, 4)
                if phase2_result_fraction is not None else None
            ),
            **load_stats,
        })

        logger.info(
            "Source %s Phase-2 yield: %s (emitted=%d / processed=%d)",
            source_name,
            f"{phase2_result_fraction:.2%}" if phase2_result_fraction is not None else "n/a",
            result_rows_emitted, n_gene_tasks,
        )
        # Fail loud on silent source degradation when configured.
        _enforce_source_yield(source_config, phase2_result_fraction, n_gene_tasks)

    if not all_results:
        raise RuntimeError("MR analysis produced no results across all eQTL sources")

    mr_results = pd.DataFrame(all_results)

    # Ensure coloc columns exist with defaults for non-significant genes
    for col, default in [("coloc_supported", False), ("coloc_status", "not_run")]:
        if col not in mr_results.columns:
            mr_results[col] = default
        else:
            mr_results[col] = mr_results[col].fillna(default)

    # Additive per-source BH-FDR + tested-Bonferroni sensitivity.
    # Primary mr_significant / bonferroni_threshold are untouched.
    mr_results = _annotate_fdr_track(mr_results)

    mr_results = annotate_cross_source(mr_results)
    mr_results = assign_confidence_tiers(
        mr_results,
        require_coloc=config.require_coloc,
        require_steiger=config.require_steiger,
    )

    dm_config = config.drug_match

    # Optional druggable-genome annotation + secondary significance track.
    mr_results = _annotate_druggable_track(mr_results, dm_config)

    # Additive MHC flag (Ensembl membership; fail-loud unless
    # coordinate fallback is opted in). Off by default.
    mr_results, mhc_metadata = _annotate_mhc_flag(
        mr_results, config, reference_config, gene_id_converter,
    )

    # --- Phase 4: Drug matching (deferred load) ---
    logger.info("Loading drug targets from %s", drug_targets_path)
    drug_targets = pd.read_parquet(drug_targets_path)
    # Precompute version-stripped Ensembl once for the strict-mode ID hierarchy.
    if dm_config.match_mode == "strict" and "gene_ensembl_id" in drug_targets.columns:
        drug_targets = drug_targets.copy()
        drug_targets["_ens_norm"] = drug_targets["gene_ensembl_id"].map(_ensembl_no_version)

    drug_match_records: list[pd.DataFrame] = []
    verdict_rows: list[dict] = []

    eligible = mr_results.loc[
        mr_results["mr_significant"]
        & ~mr_results["weak_instrument_excluded"]
        & (mr_results["confidence_tier"] != "direction_conflict")
    ]

    if config.require_coloc:
        eligible = eligible.loc[eligible["coloc_supported"]]

    if config.require_steiger:
        eligible = eligible.loc[eligible.get("steiger_valid", True) == True]  # noqa: E712

    for idx in eligible.index:
        gene_row = eligible.loc[idx]
        raw, via = _resolve_drug_matches_by_id(gene_row, drug_targets, dm_config)
        matches = _build_drug_match_records(gene_row, raw, via, dm_config)
        if not matches.empty:
            drug_match_records.append(matches)
        # One verdict per eligible gene, incl. genes with no passing drug.
        verdict_rows.append(
            _summarise_gene_verdict(gene_row, len(raw), matches, dm_config)
        )

    mr_drug_matches = pd.concat(drug_match_records, ignore_index=True) if drug_match_records else pd.DataFrame()
    mr_target_verdicts = pd.DataFrame(verdict_rows) if verdict_rows else pd.DataFrame()

    # Canonical sort for deterministic output regardless of scheduling order
    mr_results.sort_values(
        ["eqtl_source", "gene_ensembl_id"], kind="mergesort", ignore_index=True, inplace=True,
    )
    if not mr_drug_matches.empty:
        _drug_sort_keys = ["eqtl_source", "gene_ensembl_id", "drug_chembl_id"]
        missing_keys = [k for k in _drug_sort_keys if k not in mr_drug_matches.columns]
        if missing_keys:
            raise ValueError(
                f"mr_drug_matches missing expected sort keys {missing_keys}. "
                f"Columns present: {list(mr_drug_matches.columns)}"
            )
        mr_drug_matches.sort_values(
            _drug_sort_keys, kind="mergesort", ignore_index=True, inplace=True,
        )
    if not mr_target_verdicts.empty:
        mr_target_verdicts.sort_values(
            ["eqtl_source", "gene_ensembl_id"], kind="mergesort",
            ignore_index=True, inplace=True,
        )

    # Save outputs
    mr_results.to_parquet(output_dir / "mr_results.parquet", engine="pyarrow", index=False)
    mr_csv = mr_results.copy()
    for col in mr_csv.columns:
        if mr_csv[col].dtype == object:
            sample = mr_csv[col].dropna().head(1)
            if len(sample) > 0 and isinstance(sample.iloc[0], list):
                mr_csv[col] = mr_csv[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    mr_csv.to_csv(output_dir / "mr_results.csv", index=False)

    if not mr_drug_matches.empty:
        mr_drug_matches.to_parquet(output_dir / "mr_drug_matches.parquet", engine="pyarrow", index=False)
        dm_csv = mr_drug_matches.copy()
        for col in dm_csv.columns:
            if dm_csv[col].dtype == object:
                sample = dm_csv[col].dropna().head(1)
                if len(sample) > 0 and isinstance(sample.iloc[0], list):
                    dm_csv[col] = dm_csv[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        dm_csv.to_csv(output_dir / "mr_drug_matches.csv", index=False)
    else:
        pd.DataFrame().to_parquet(output_dir / "mr_drug_matches.parquet", engine="pyarrow", index=False)

    # Per-gene target verdicts - always written (empty frame if none eligible).
    if not mr_target_verdicts.empty:
        mr_target_verdicts.to_parquet(
            output_dir / "mr_target_verdicts.parquet", engine="pyarrow", index=False,
        )
        mr_target_verdicts.to_csv(output_dir / "mr_target_verdicts.csv", index=False)
    else:
        pd.DataFrame().to_parquet(
            output_dir / "mr_target_verdicts.parquet", engine="pyarrow", index=False,
        )
        pd.DataFrame().to_csv(output_dir / "mr_target_verdicts.csv", index=False)

    # MHC-excluded sensitivity view (only when flagging ran).
    if config.mhc_sensitivity.enabled and "mhc_flag" in mr_results.columns:
        mhc_sensitivity_summary = _write_mhc_excluded_sensitivity(
            output_dir, mr_results, mr_drug_matches, mr_target_verdicts,
        )
        mhc_metadata["sensitivity_summary"] = mhc_sensitivity_summary

    elapsed = time.time() - start_time
    n_sig = int(mr_results["mr_significant"].sum()) if "mr_significant" in mr_results.columns else 0
    rss_peak = _get_peak_rss_mb()

    # Make the multiple-testing regime explicit and auditable.
    n_sig_fdr = (
        int(mr_results["mr_significant_fdr_bh"].sum())
        if "mr_significant_fdr_bh" in mr_results.columns else 0
    )
    multiple_testing = {
        "primary_method": "per_source_bonferroni",
        "denominator_policy": (
            "source_eligible_genes (all genes with >=1 instrument per source; "
            "). mr_significant still uses 0.05/n_genes_valid."
        ),
        "fdr_method": "fdr_bh",
        "fdr_scope": "per_eqtl_source_over_finite_pvalues",
        "fdr_alpha": 0.05,
        "n_significant_bonferroni_primary": n_sig,
        "n_significant_fdr_bh": n_sig_fdr,
        "n_significant_bonferroni_tested": (
            int(mr_results["mr_significant_bonferroni_tested"].sum())
            if "mr_significant_bonferroni_tested" in mr_results.columns else 0
        ),
        "per_source": [
            {
                "source": str(src),
                "n_emitted_rows": int(len(grp)),
                "n_finite_pval": int(
                    np.isfinite(
                        pd.to_numeric(grp["mr_pval"], errors="coerce")
                    ).sum()
                ),
                "n_significant_bonferroni_primary": int(grp["mr_significant"].sum()),
                "n_significant_fdr_bh": (
                    int(grp["mr_significant_fdr_bh"].sum())
                    if "mr_significant_fdr_bh" in grp.columns else 0
                ),
            }
            for src, grp in mr_results.groupby("eqtl_source", sort=False)
        ],
    }

    metadata = {
        # PLINK does the LD clumping that selects instruments, so its build
        # is part of the provenance of every MR estimate here.
        **plink_provenance(plink_binary),
        "n_eqtl_sources": len(config.eqtl_sources),
        "eqtl_sources": [s.source for s in config.eqtl_sources],
        "n_genes_tested": len(mr_results["gene_ensembl_id"].unique()),
        "n_genes_significant": n_sig,
        "multiple_testing": multiple_testing,
        "n_drug_matches": len(mr_drug_matches),
        "n_target_verdicts": len(mr_target_verdicts),
        "verdict_status_counts": (
            {
                str(k): int(v)
                for k, v in mr_target_verdicts["verdict_status"].value_counts().items()
            }
            if not mr_target_verdicts.empty and "verdict_status" in mr_target_verdicts.columns
            else {}
        ),
        "drug_match_config": {
            "match_mode": dm_config.match_mode,
            "min_pchembl": dm_config.min_pchembl,
            "min_phase": dm_config.min_phase,
            "phase_filter_scope": dm_config.phase_filter_scope,
            "direction_policy": dm_config.direction_policy,
            "allow_symbol_fallback": dm_config.allow_symbol_fallback,
            "require_druggable": dm_config.require_druggable,
            "druggable_genome_path": (
                str(dm_config.druggable_genome_path)
                if dm_config.druggable_genome_path is not None else None
            ),
        },
        "instrument_pval": config.instrument_pval,
        "clump_r2": config.clump_r2,
        "f_stat_threshold": config.f_stat_threshold,
        "coloc_enabled": config.coloc_enabled,
        "coloc_pp_h4_threshold": config.coloc_pp_h4_threshold,
        # Binary-outcome calibration provenance.
        "coloc_variance_mode": getattr(config, "coloc_variance_mode", "reported_se"),
        "gwas_trait_type": getattr(gwas_metadata, "trait_type", None),
        "gwas_n_cases": getattr(gwas_metadata, "n_cases", None),
        "gwas_n_controls": getattr(gwas_metadata, "n_controls", None),
        "gwas_population_prevalence": getattr(gwas_metadata, "population_prevalence", None),
        "case_control_n_source": getattr(gwas_metadata, "case_control_n_source", None),
        "steiger_binary_calibration_counts": (
            {
                str(k): int(v)
                for k, v in mr_results["steiger_binary_calibration"].value_counts().items()
            }
            if "steiger_binary_calibration" in mr_results.columns else {}
        ),
        "require_coloc": config.require_coloc,
        "require_steiger": config.require_steiger,
        # MHC annotation + sensitivity provenance.
        "mhc_sensitivity": mhc_metadata,
        "maf_resolution": maf_counters,
        "elapsed_seconds": round(elapsed, 2),
        "peak_rss_mb": round(rss_peak, 1),
        "per_source_metadata": source_metadata,
    }

    with open(output_dir / "mr_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "MR analysis complete: %d genes tested, %d significant, %d drug matches "
        "(%.1fs, peak RSS %.0f MB)",
        metadata["n_genes_tested"], n_sig, len(mr_drug_matches), elapsed, rss_peak,
    )

    return mr_results, mr_drug_matches


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cis-MR drug repurposing analysis")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gwas", type=Path, required=True)
    parser.add_argument(
        "--gwas-metadata", type=Path, default=None,
        help="Path to GWAS metadata JSON sidecar. Default: <gwas_stem>.meta.json",
    )
    parser.add_argument("--drug-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Cap on parallel workers (e.g., from Snakemake {threads}). "
             "Effective workers = min(config.mr.n_workers, --threads, cpu_count, n_tasks).",
    )
    args = parser.parse_args()

    from repogen.config.loader import load_config
    from repogen.data.gene_id_converter import GeneIDConverter

    pipeline_config = load_config(args.config)

    meta_path = args.gwas_metadata or args.gwas.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"GWAS metadata sidecar not found at {meta_path}. "
            f"Run gwas_prep.py first, or specify --gwas-metadata explicitly."
        )
    gwas_meta = GWASMetadata.model_validate_json(meta_path.read_text())

    converter = None
    ref = pipeline_config.reference
    if ref.ensembl_to_name and ref.name_to_ensembl and ref.uniprot_to_ensembl:
        converter = GeneIDConverter(
            biomart_dicts={
                "ensembl_to_name": ref.ensembl_to_name,
                "name_to_ensembl": ref.name_to_ensembl,
                "uniprot_to_ensembl": ref.uniprot_to_ensembl,
            },
            entrez_mapping_file=ref.ncbi_gene_info,
            gene_history_file=ref.ncbi_gene_history,
        )

    run_mendelian_randomisation(
        gwas_path=args.gwas,
        gwas_metadata=gwas_meta,
        drug_targets_path=args.drug_targets,
        gene_id_converter=converter,
        config=pipeline_config.mr,
        reference_config=pipeline_config.reference,
        output_dir=args.output_dir,
        n_workers_cap=args.threads,
    )
