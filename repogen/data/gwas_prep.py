"""Unified GWAS summary statistics preprocessing.

Ingests GWAS data from any major source (PGC, UKB, FinnGen,
GWAS Catalog, OpenGWAS, METAL output, custom), auto-detects format
and columns, standardises to the ``StandardizedGWAS`` schema, and
performs quality control.

Replaces the old ``prepare_gwas.py`` + ``format_gwas_for_spredixcan.py``
pair.  Produces a single rich intermediate that serves MAGMA,
S-PrediXcan, MR, and any future analysis module.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from repogen.data.schemas import GWASMetadata, validate_dataframe
from repogen.utils.constants import GWAS_COLUMN_ALIASES
from repogen.utils.io import check_file_exists, ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Internal detection state (not a shared schema - stays in this module)
# ---------------------------------------------------------------------------


@dataclass
class GWASFormat:
    """Internal detection result - describes the detected input format."""

    source: str = "custom"
    delimiter: str = "\t"
    compression: Optional[str] = None
    column_map: dict[str, str] = field(default_factory=dict)
    has_or: bool = False
    has_beta: bool = True
    genome_build: Optional[str] = None
    trait_type: Optional[str] = None
    has_variant_id: bool = False
    has_sample_size: bool = False
    effect_allele_column: Optional[str] = None
    n_meta_lines: int = 0
    header_starts_with_hash: bool = False


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_REPEATED_WS_RE = re.compile(r"[ \t]{2,}")


def _refine_whitespace_delimiter(
    header_line: str, data_lines_raw: list[str]
) -> str:
    """Choose ``" "`` (literal space) or ``r"\\s+"`` for whitespace files.

    For files that are uniformly single-space-delimited, a literal space
    separator allows ``pandas.read_csv`` to use the faster C engine, which
    is critical for large files (e.g. 27M-row GIANT BMI).

    Falls back to ``r"\\s+"`` (regex, requires slower Python engine) when
    the file has tabs, repeated whitespace, or inconsistent token counts
    across sampled lines.
    """
    data_stripped = [l.rstrip("\n") for l in data_lines_raw if l.strip()]
    sampled = [header_line] + data_stripped

    has_tab = any("\t" in s for s in sampled)
    has_repeated_ws = any(_REPEATED_WS_RE.search(s) for s in sampled)

    if has_tab or has_repeated_ws:
        logger.debug(
            "Whitespace delimiter kept as \\s+ (tabs=%s, repeated_ws=%s)",
            has_tab,
            has_repeated_ws,
        )
        return r"\s+"

    counts = [len(s.strip().split(" ")) for s in sampled]
    if len(set(counts)) == 1 and counts[0] > 1:
        logger.debug(
            "Whitespace delimiter refined to literal ' ' "
            "(uniform token count: %d across %d sampled lines)",
            counts[0],
            len(sampled),
        )
        return " "

    logger.debug(
        "Whitespace delimiter kept as \\s+ (unstable token counts: %s)",
        counts,
    )
    return r"\s+"


def detect_gwas_format(input_path: Path) -> GWASFormat:
    """Auto-detect GWAS summary statistics format.

    Hierarchical strategy:

    1. Check file extension for compression.
    2. Read first lines, detect delimiter.
    3. Map headers against ``GWAS_COLUMN_ALIASES`` -> identify source.
    4. Determine if effect column is OR or BETA.

    Args:
        input_path: Path to the GWAS file.

    Returns:
        A ``GWASFormat`` dataclass with all detected settings.
    """
    fmt = GWASFormat()
    input_path = Path(input_path)

    if input_path.suffix in (".gz", ".bgz"):
        fmt.compression = "gzip"
    opener = gzip.open if fmt.compression else open

    with opener(input_path, "rt") as fh:
        meta_count = 0
        lines: list[str] = []
        for _ in range(10_000):
            raw = fh.readline()
            if not raw:
                break
            if raw.startswith("##"):
                meta_count += 1
                if "genomeReference" in raw:
                    val = raw.split("=", 1)[-1].strip().strip('"')
                    if val:
                        fmt.genome_build = val
                continue
            lines.append(raw)
            if len(lines) >= 5:
                break
    fmt.n_meta_lines = meta_count

    if not lines:
        raise ValueError(
            f"Could not find a data header in {input_path} "
            f"(scanned {meta_count} metadata lines)"
        )

    header_line = lines[0].rstrip("\n")

    if "\t" in header_line:
        fmt.delimiter = "\t"
    elif "," in header_line:
        fmt.delimiter = ","
    else:
        fmt.delimiter = _refine_whitespace_delimiter(header_line, lines[1:])

    if fmt.delimiter == r"\s+":
        raw_cols = header_line.split()
    else:
        raw_cols = header_line.split(fmt.delimiter)

    upper_cols = [c.upper().strip() for c in raw_cols]

    for canonical, aliases in GWAS_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.upper() in upper_cols:
                idx = upper_cols.index(alias.upper())
                fmt.column_map[canonical] = raw_cols[idx]
                break

    fmt.header_starts_with_hash = any(
        v.startswith("#") for v in fmt.column_map.values()
    )

    if "#CHROM" in upper_cols:
        fmt.source = "FinnGen"
    elif "VARIANT_ID" in fmt.column_map and ":" in _peek_value(lines, raw_cols, fmt.column_map.get("VARIANT_ID", ""), fmt.delimiter):
        fmt.source = "UKB"
        fmt.has_variant_id = True
    elif "DIRECTION" in upper_cols:
        fmt.source = "METAL"
    elif fmt.column_map.get("SNP") and fmt.column_map.get("P"):
        fmt.source = "PGC"

    if "OR" in fmt.column_map and "BETA" not in fmt.column_map:
        fmt.has_or = True
        fmt.has_beta = False
    elif "OR" in fmt.column_map and "BETA" in fmt.column_map:
        fmt.has_or = False
        fmt.has_beta = True
    elif "BETA" in fmt.column_map:
        fmt.has_or = False
        fmt.has_beta = True
    else:
        effect_vals = _peek_effect_values(lines, raw_cols, fmt)
        if effect_vals:
            mean_abs = np.nanmean(np.abs(effect_vals))
            fmt.has_or = mean_abs > 0.5
            fmt.has_beta = not fmt.has_or

    fmt.has_sample_size = "N" in fmt.column_map or (
        "N_CAS" in fmt.column_map and "N_CON" in fmt.column_map
    )
    if "A1" in fmt.column_map:
        fmt.effect_allele_column = fmt.column_map["A1"]

    logger.info(
        "Detected GWAS format: source=%s, delimiter=%r, OR=%s, "
        "columns mapped: %d, sample_size_in_file=%s",
        fmt.source,
        fmt.delimiter,
        fmt.has_or,
        len(fmt.column_map),
        fmt.has_sample_size,
    )
    return fmt


def _split_data_line(line: str, delimiter: str) -> list[str]:
    """Split a data line using the detected delimiter."""
    stripped = line.rstrip("\n")
    if delimiter == r"\s+":
        return stripped.split()
    return stripped.split(delimiter)


def _peek_value(lines: list[str], raw_cols: list[str], col_name: str, delimiter: str = r"\s+") -> str:
    """Get the first data value for a column (from line index 1)."""
    if not col_name or len(lines) < 2:
        return ""
    try:
        idx = [c.strip() for c in raw_cols].index(col_name.strip())
    except ValueError:
        return ""
    data_fields = _split_data_line(lines[1], delimiter)
    return data_fields[idx] if idx < len(data_fields) else ""


def _peek_effect_values(
    lines: list[str], raw_cols: list[str], fmt: GWASFormat
) -> list[float]:
    """Try to read a few effect-size values from data lines."""
    candidates = ["BETA", "OR", "EFFECT", "EFFECT_SIZE", "B", "LOGODDS"]
    upper_cols = [c.upper().strip() for c in raw_cols]
    col_idx: Optional[int] = None
    for cand in candidates:
        if cand in upper_cols:
            col_idx = upper_cols.index(cand)
            break
    if col_idx is None:
        return []
    vals: list[float] = []
    for line in lines[1:]:
        parts = _split_data_line(line, fmt.delimiter)
        if col_idx < len(parts):
            try:
                vals.append(float(parts[col_idx]))
            except ValueError:
                pass
    return vals


# ---------------------------------------------------------------------------
# Column mapping and transforms
# ---------------------------------------------------------------------------


def map_columns(df: pd.DataFrame, detected_format: GWASFormat) -> pd.DataFrame:
    """Map source-specific columns to canonical names.

    Additional transforms:

    * OR -> BETA conversion (``BETA = log(OR)``).
    * N computation from N_CAS + N_CON when total N is missing.
    * VARIANT_ID parsing/construction.
    * Allele uppercasing.

    Args:
        df: Raw DataFrame.
        detected_format: Result of :func:`detect_gwas_format`.

    Returns:
        DataFrame with canonical column names.
    """
    rename_map = {v: k for k, v in detected_format.column_map.items() if v in df.columns}
    df = df.rename(columns=rename_map)

    upper_to_col = {c.upper().strip(): c for c in df.columns}
    fallback_map: dict[str, str] = {}
    for canonical, aliases in GWAS_COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias.upper() in upper_to_col:
                fallback_map[upper_to_col[alias.upper()]] = canonical
                break
    if fallback_map:
        logger.info("Fallback alias mapping: %s", fallback_map)
        df = df.rename(columns=fallback_map)

    if detected_format.has_or and "OR" in df.columns:
        logger.info("Converting OR -> BETA (log transform)")
        df["BETA"] = np.log(pd.to_numeric(df["OR"], errors="coerce"))

    for col in ("BETA", "SE", "P", "MAF", "INFO"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("CHR", "POS"):
        if col in df.columns:
            df[col] = (
                pd.to_numeric(
                    df[col].astype(str).str.replace(r"^chr", "", regex=True),
                    errors="coerce",
                )
                .astype("Int64")
            )

    for col in ("N", "N_CAS", "N_CON"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64")

    if "N" not in df.columns or df["N"].isna().all():
        if "N_CAS" in df.columns and "N_CON" in df.columns:
            logger.info("Computing N = N_CAS + N_CON")
            df["N"] = df["N_CAS"].fillna(0) + df["N_CON"].fillna(0)
            df.loc[df["N"] == 0, "N"] = pd.NA

    for col in ("A1", "A2"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()

    if "SNP" in df.columns:
        snp_col = df["SNP"].astype(str)
        has_rs_suffix = snp_col.str.match(r"^rs\d+:.+")
        if has_rs_suffix.any():
            n_norm = has_rs_suffix.sum()
            df["SNP"] = snp_col.where(~has_rs_suffix, snp_col.str.split(":").str[0])
            logger.info("Normalised %d SNP IDs (stripped rs...:allele suffixes)", n_norm)

    if "VARIANT_ID" in df.columns and detected_format.has_variant_id:
        parts = df["VARIANT_ID"].str.split(":", expand=True)
        if parts.shape[1] >= 4:
            if "CHR" not in df.columns or df["CHR"].isna().all():
                df["CHR"] = pd.to_numeric(
                    parts[0].str.replace("chr", "", regex=False), errors="coerce"
                ).astype("Int64")
            if "POS" not in df.columns or df["POS"].isna().all():
                df["POS"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
            if "A1" not in df.columns:
                df["A1"] = parts[2].str.upper()
            if "A2" not in df.columns:
                df["A2"] = parts[3].str.upper()

    if "VARIANT_ID" not in df.columns:
        if all(c in df.columns for c in ("CHR", "POS", "A1", "A2")):
            df["VARIANT_ID"] = (
                df["CHR"].astype(str) + ":" +
                df["POS"].astype(str) + ":" +
                df["A1"].astype(str) + ":" +
                df["A2"].astype(str)
            )

    if "SNP" not in df.columns:
        df["SNP"] = pd.NA

    return df


# ---------------------------------------------------------------------------
# Allele harmonisation
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _resolve_alleles(merged: pd.DataFrame) -> pd.DataFrame:
    """Apply palindromic removal, allele flipping, and strand complement.

    Expects a DataFrame with columns A1, A2, REF, ALT, BETA.
    Returns the filtered DataFrame with REF/ALT dropped.
    """
    is_palindromic = (
        ((merged["A1"] == "A") & (merged["A2"] == "T")) |
        ((merged["A1"] == "T") & (merged["A2"] == "A")) |
        ((merged["A1"] == "C") & (merged["A2"] == "G")) |
        ((merged["A1"] == "G") & (merged["A2"] == "C"))
    )
    n_palindromic = is_palindromic.sum()
    merged = merged[~is_palindromic]
    logger.info("Removed %d palindromic SNPs", n_palindromic)

    needs_flip = (merged["A1"] == merged["REF"]) & (merged["A2"] == merged["ALT"])
    merged.loc[needs_flip, "BETA"] = -merged.loc[needs_flip, "BETA"]
    merged.loc[needs_flip, ["A1", "A2"]] = merged.loc[needs_flip, ["A2", "A1"]].values
    logger.info("Flipped alleles for %d SNPs", needs_flip.sum())

    correct = (merged["A1"] == merged["ALT"]) & (merged["A2"] == merged["REF"])

    complement_a1 = merged["A1"].str.translate(_COMPLEMENT)
    complement_a2 = merged["A2"].str.translate(_COMPLEMENT)
    strand_match = (complement_a1 == merged["ALT"]) & (complement_a2 == merged["REF"])

    keep = correct | strand_match
    n_removed = (~keep).sum()
    if n_removed > 0:
        logger.warning("Removed %d SNPs with unresolvable allele mismatches", n_removed)
    merged = merged[keep].drop(columns=["REF", "ALT"])

    return merged


def harmonize_alleles(
    df: pd.DataFrame, ref_bim: pd.DataFrame
) -> pd.DataFrame:
    """Align effect alleles to a reference panel.

    Primary path: merge on SNP (rsID). If that yields 0 matches and the
    GWAS has CHR + POS, falls back to coordinate-based matching and
    assigns the reference rsID as the SNP column.

    Handles strand flips and logs warnings for unresolvable mismatches.
    Palindromic SNPs (A/T, C/G) are removed.

    Args:
        df: StandardizedGWAS DataFrame (must have SNP, A1, A2, BETA).
        ref_bim: Reference BIM DataFrame with columns
            ``["CHR", "SNP", "CM", "POS", "REF", "ALT"]``.

    Returns:
        Harmonised DataFrame (rows with unresolvable alleles removed).
    """
    n_before = len(df)
    ref = ref_bim.rename(columns={ref_bim.columns[1]: "SNP", ref_bim.columns[4]: "REF", ref_bim.columns[5]: "ALT"})
    ref["REF"] = ref["REF"].str.upper()
    ref["ALT"] = ref["ALT"].str.upper()

    # --- Primary path: merge on SNP (rsID) ---
    ref_snp = ref[["SNP", "REF", "ALT"]].drop_duplicates(subset="SNP")
    merged = df.merge(ref_snp, on="SNP", how="inner")
    n_in_ref = len(merged)
    logger.info("Allele harmonisation (SNP): %d / %d SNPs found in reference", n_in_ref, n_before)

    # --- Fallback: merge on CHR + POS if SNP merge yielded 0 ---
    if n_in_ref == 0 and "CHR" in df.columns and "POS" in df.columns:
        logger.info(
            "SNP-based harmonisation found 0 matches - "
            "attempting coordinate-based fallback (CHR + POS)"
        )
        ref_bim_renamed = ref_bim.rename(
            columns={
                ref_bim.columns[0]: "CHR",
                ref_bim.columns[1]: "_REF_SNP",
                ref_bim.columns[3]: "POS",
                ref_bim.columns[4]: "REF",
                ref_bim.columns[5]: "ALT",
            }
        )
        ref_coord = ref_bim_renamed[["CHR", "POS", "_REF_SNP", "REF", "ALT"]].copy()
        ref_coord["REF"] = ref_coord["REF"].str.upper()
        ref_coord["ALT"] = ref_coord["ALT"].str.upper()
        ref_coord["CHR"] = pd.to_numeric(ref_coord["CHR"], errors="coerce").astype("Int64")
        ref_coord["POS"] = pd.to_numeric(ref_coord["POS"], errors="coerce").astype("Int64")
        ref_coord = ref_coord.drop_duplicates(subset=["CHR", "POS"])

        merged = df.merge(ref_coord, on=["CHR", "POS"], how="inner")
        n_coord = len(merged)
        logger.info(
            "Coordinate fallback: %d / %d variants matched by CHR+POS",
            n_coord, n_before,
        )
        if n_coord > 0:
            merged["SNP"] = merged["_REF_SNP"]
            merged = merged.drop(columns=["_REF_SNP"])
        else:
            logger.warning(
                "Coordinate fallback also found 0 matches. "
                "This may indicate a genome build mismatch - "
                "check whether liftover is needed."
            )
    elif "_REF_SNP" in merged.columns:
        merged = merged.drop(columns=["_REF_SNP"], errors="ignore")

    merged = _resolve_alleles(merged)

    logger.info("After harmonisation: %d variants retained", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Liftover
# ---------------------------------------------------------------------------


_ALLELE_COMPLEMENT: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}


def apply_liftover(
    df: pd.DataFrame,
    chain_file: Path,
    source_build: str,
    target_build: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coordinate liftover via pyliftover with correctness hardening.

    Handles coordinate basis (GWAS 1-based to pyliftover 0-based),
    updates both CHR and POS from liftover output, applies allele
    complementation for minus-strand mappings, and rejects multi-mapped
    variants deterministically.

    Args:
        df: DataFrame with ``CHR`` and ``POS`` columns. ``A1`` and ``A2``
            are recommended for full VARIANT_ID reconstruction
            (``CHR:POS:A1:A2``); without them VARIANT_ID falls back to
            positional only (``CHR:POS``).
        chain_file: Path to UCSC chain file.
        source_build: Original build (e.g. ``"GRCh37"``).
        target_build: Target build (e.g. ``"GRCh38"``).

    Returns:
        Tuple of (DataFrame with updated coordinates, counters dict).
        Counters dict keys: ``unmapped``, ``multimap_dropped``,
        ``minus_strand_complemented``, ``minus_strand_dropped``,
        ``converted``.
    """
    from pyliftover import LiftOver

    logger.info("Applying liftover: %s -> %s", source_build, target_build)
    lo = LiftOver(str(chain_file))

    n_before = len(df)
    has_alleles = "A1" in df.columns and "A2" in df.columns

    chroms = ("chr" + df["CHR"].astype(str)).values
    positions = df["POS"].values.astype(int)

    new_chroms: list[str | None] = []
    new_positions: list[int | None] = []
    new_strands: list[str | None] = []

    n_unmapped = 0
    n_multimap = 0

    for chrom, pos in zip(chroms, positions):
        try:
            result = lo.convert_coordinate(chrom, int(pos) - 1)
        except (TypeError, ValueError, IndexError):
            result = None

        if not result:
            new_chroms.append(None)
            new_positions.append(None)
            new_strands.append(None)
            n_unmapped += 1
        elif len(result) > 1:
            new_chroms.append(None)
            new_positions.append(None)
            new_strands.append(None)
            n_multimap += 1
        else:
            mapped_chr, mapped_pos, mapped_strand, _ = result[0]
            chr_num = mapped_chr.replace("chr", "")
            new_chroms.append(chr_num)
            new_positions.append(int(mapped_pos) + 1)
            new_strands.append(mapped_strand)

    df = df.copy()
    df["_new_chr"] = new_chroms
    df["_new_pos"] = new_positions
    df["_strand"] = new_strands

    mapped_mask = df["_new_pos"].notna()
    if n_unmapped > 0:
        logger.warning("Liftover: %d variants unmapped -- removed.", n_unmapped)
    if n_multimap > 0:
        logger.warning("Liftover: %d variants multi-mapped -- removed.", n_multimap)

    df = df[mapped_mask].copy()

    df["CHR"] = pd.to_numeric(df["_new_chr"], errors="coerce")
    df["POS"] = df["_new_pos"].astype(int)

    n_minus_complemented = 0
    n_minus_dropped = 0
    minus_mask = df["_strand"] == "-"
    if minus_mask.any():
        if has_alleles:
            minus_rows = df.loc[minus_mask]
            comp_a1 = minus_rows["A1"].map(_ALLELE_COMPLEMENT)
            comp_a2 = minus_rows["A2"].map(_ALLELE_COMPLEMENT)
            valid_row = comp_a1.notna() & comp_a2.notna()
            valid_idx = minus_rows.index[valid_row]
            invalid_idx = minus_rows.index[~valid_row]

            df.loc[valid_idx, "A1"] = comp_a1.loc[valid_idx]
            df.loc[valid_idx, "A2"] = comp_a2.loc[valid_idx]

            n_minus_complemented = len(valid_idx)
            n_minus_dropped = len(invalid_idx)
            if n_minus_dropped > 0:
                df = df.drop(index=invalid_idx).copy()
                logger.warning(
                    "Liftover: %d minus-strand variants had non-standard alleles -- dropped.",
                    n_minus_dropped,
                )
        else:
            n_minus_dropped = minus_mask.sum()
            df = df[~minus_mask].copy()
            logger.warning(
                "Liftover: %d minus-strand variants dropped (no allele columns to complement).",
                n_minus_dropped,
            )

    n_non_numeric_chr = 0
    non_numeric_chr = df["CHR"].isna()
    if non_numeric_chr.any():
        n_non_numeric_chr = int(non_numeric_chr.sum())
        df = df[~non_numeric_chr].copy()
        logger.info("Liftover: %d non-numeric CHR variants removed post-lift.", n_non_numeric_chr)
    df["CHR"] = df["CHR"].astype(int)

    df = df.drop(columns=["_new_chr", "_new_pos", "_strand"], errors="ignore")

    if has_alleles:
        df["VARIANT_ID"] = (
            df["CHR"].astype(str) + ":" + df["POS"].astype(str)
            + ":" + df["A1"] + ":" + df["A2"]
        )
    elif "VARIANT_ID" in df.columns:
        df["VARIANT_ID"] = df["CHR"].astype(str) + ":" + df["POS"].astype(str)
        logger.warning("VARIANT_ID rebuilt without alleles (A1/A2 columns not present)")

    expected_remaining = (
        n_before - n_unmapped - n_multimap - n_minus_dropped - n_non_numeric_chr
    )
    if len(df) != expected_remaining:
        logger.warning(
            "Liftover row accounting mismatch: expected %d remaining "
            "(from %d - %d unmapped - %d multimap - %d minus_dropped - %d non_num_chr), "
            "got %d.",
            expected_remaining, n_before, n_unmapped, n_multimap,
            n_minus_dropped, n_non_numeric_chr, len(df),
        )

    counters = {
        "unmapped": n_unmapped,
        "multimap_dropped": n_multimap,
        "minus_strand_complemented": n_minus_complemented,
        "minus_strand_dropped": n_minus_dropped,
        "non_numeric_chr_dropped": n_non_numeric_chr,
        "converted": len(df),
    }

    logger.info(
        "Liftover complete: %d -> %d variants "
        "(unmapped=%d, multimap=%d, minus_strand_complemented=%d, minus_strand_dropped=%d, "
        "non_numeric_chr=%d)",
        n_before, len(df), n_unmapped, n_multimap,
        n_minus_complemented, n_minus_dropped, n_non_numeric_chr,
    )
    return df, counters


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------


def quality_control(
    df: pd.DataFrame,
    info_threshold: float = 0.6,
    maf_threshold: float = 0.01,
    remove_mhc: bool = False,
) -> pd.DataFrame:
    """Apply standard GWAS QC filters.

    Args:
        df: DataFrame with StandardizedGWAS columns.
        info_threshold: Minimum INFO score (if column present).
        maf_threshold: Minimum MAF (if column present).
        remove_mhc: Whether to remove the MHC region
            (chr6:25-35 Mb in GRCh37 coordinates).

    Returns:
        Filtered DataFrame.
    """
    n_start = len(df)
    logger.info("Starting QC on %d variants", n_start)

    if "P" not in df.columns:
        raise ValueError(
            f"Required column 'P' not found after column mapping. "
            f"Available columns: {sorted(df.columns.tolist())}. "
            f"Check that the input file header is correctly detected "
            f"and that GWAS_COLUMN_ALIASES covers your p-value column name."
        )

    if "INFO" in df.columns and df["INFO"].notna().any():
        before = len(df)
        df = df[df["INFO"].isna() | (df["INFO"] >= info_threshold)]
        logger.info("INFO >= %.2f: %d -> %d", info_threshold, before, len(df))

    if "MAF" in df.columns and df["MAF"].notna().any():
        before = len(df)
        df = df[df["MAF"].isna() | (df["MAF"] >= maf_threshold)]
        logger.info("MAF >= %.3f: %d -> %d", maf_threshold, before, len(df))

    before = len(df)
    df = df.dropna(subset=["P"])
    df = df[(df["P"] > 0) & (df["P"] <= 1)]
    logger.info("Valid P-values: %d -> %d", before, len(df))

    _REQUIRED_FIELDS = ("SNP", "CHR", "POS", "A1", "A2", "BETA", "SE", "P", "N")
    check_cols = [c for c in _REQUIRED_FIELDS if c in df.columns]
    if check_cols:
        before = len(df)
        df = df.dropna(subset=check_cols)
        n_dropped = before - len(df)
        if n_dropped > 0:
            logger.info(
                "Required-field completeness (%s): %d -> %d",
                ", ".join(check_cols), before, len(df),
            )

    before = len(df)
    if "VARIANT_ID" in df.columns:
        df = df.drop_duplicates(subset="VARIANT_ID", keep="first")
    elif "SNP" in df.columns:
        df = df.drop_duplicates(subset="SNP", keep="first")
    logger.info("Deduplicated: %d -> %d", before, len(df))

    if remove_mhc and "CHR" in df.columns and "POS" in df.columns:
        before = len(df)
        mhc = (df["CHR"] == 6) & (df["POS"] >= 25_000_000) & (df["POS"] <= 35_000_000)
        df = df[~mhc]
        logger.info("MHC removal (chr6:25-35Mb): %d -> %d", before, len(df))

    logger.info("QC complete: %d -> %d variants (%.1f%% retained)", n_start, len(df), 100 * len(df) / max(n_start, 1))
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _resolve_case_control_n(
    df: pd.DataFrame,
    n_cases: Optional[int],
    n_controls: Optional[int],
) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Resolve scalar case/control N for the metadata sidecar.

    Policy:
      1. If ``n_cases``/``n_controls`` are config-supplied, use them verbatim
         (invariant, defensible) -> source ``"config"``.
      2. Otherwise, if the QC'd frame carries ``N_CAS``/``N_CON`` columns, use
         the per-variant median (rounded) -> source ``"gwas_column_median"``.
         Per-variant N can vary (imputation), so the median is the robust
         scalar; the range is logged for transparency.
      3. Otherwise return ``(None, None, None)`` - no case/control calibration.

    Returns ``(n_cases, n_controls, source_label)``.
    """
    if n_cases is not None and n_controls is not None:
        return int(n_cases), int(n_controls), "config"

    def _median_scalar(col: str) -> Optional[int]:
        if col not in df.columns:
            return None
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            return None
        lo, hi = int(vals.min()), int(vals.max())
        if lo != hi:
            logger.info(
                "%s varies per variant (min=%d, max=%d); using median=%d as the "
                "scalar case/control N.", col, lo, hi, int(round(vals.median())),
            )
        return int(round(vals.median()))

    med_cases = int(n_cases) if n_cases is not None else _median_scalar("N_CAS")
    med_controls = int(n_controls) if n_controls is not None else _median_scalar("N_CON")
    if med_cases is not None and med_controls is not None:
        source = "config" if (n_cases is not None or n_controls is not None) else "gwas_column_median"
        return med_cases, med_controls, source
    return None, None, None


def prepare_gwas(
    input_path: Path,
    reference_bim: Optional[Path] = None,
    genome_build: Optional[str] = None,
    trait_type: Optional[str] = None,
    sample_size: Optional[int] = None,
    n_cases: Optional[int] = None,
    n_controls: Optional[int] = None,
    population_prevalence: Optional[float] = None,
    info_threshold: float = 0.6,
    maf_threshold: float = 0.01,
    liftover_to: Optional[str] = None,
    chain_file: Optional[Path] = None,
    remove_mhc: bool = False,
) -> tuple[pd.DataFrame, GWASMetadata]:
    """Ingest and standardise GWAS summary statistics.

    Pipeline:

    1. ``detect_gwas_format()`` - identify source, delimiter, columns.
    2. Read data.
    3. ``map_columns()`` - canonical names, OR->BETA, N computation.
    4. ``harmonize_alleles()`` - align to reference (if provided).
    5. ``apply_liftover()`` - if requested.
    6. ``quality_control()`` - INFO, MAF, duplicates, MHC.
    7. Validate output against ``StandardizedGWAS`` schema.
    8. Return DataFrame + metadata.

    Args:
        input_path: Path to the raw GWAS file.
        reference_bim: Optional reference BIM file for allele harmonisation.
        genome_build: ``"GRCh37"`` or ``"GRCh38"``; auto-detected if
            ``None``.
        trait_type: ``"case_control"`` or ``"quantitative"``; auto-detected.
        sample_size: Override total N (used when not in file).
        n_cases: Override case count.
        n_controls: Override control count.
        info_threshold: Minimum INFO score.
        maf_threshold: Minimum MAF.
        liftover_to: Target build for liftover; ``None`` = skip.
        chain_file: Liftover chain file path.
        remove_mhc: Remove MHC region.

    Returns:
        Tuple of ``(StandardizedGWAS DataFrame, GWASMetadata)``.

    Raises:
        FileNotFoundError: If input file is missing.
        RuntimeError: If no variants survive QC.
    """
    input_path = check_file_exists(Path(input_path), label="GWAS summary statistics")

    detected = detect_gwas_format(input_path)
    original_n = 0

    is_regex = detected.delimiter == r"\s+"
    read_csv_kwargs: dict[str, object] = {
        "sep": detected.delimiter,
        "compression": "gzip" if detected.compression else None,
        "engine": "python" if is_regex else "c",
    }
    if detected.n_meta_lines > 0:
        read_csv_kwargs["skiprows"] = detected.n_meta_lines
    elif not detected.header_starts_with_hash:
        read_csv_kwargs["comment"] = "#"
    if not is_regex:
        read_csv_kwargs["low_memory"] = False

    df = pd.read_csv(input_path, **read_csv_kwargs)

    if df.columns[0].startswith("#"):
        df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("#")})
    original_n = len(df)
    logger.info("Read %d variants from %s", original_n, input_path.name)

    df = map_columns(df, detected)

    if sample_size is not None and ("N" not in df.columns or df["N"].isna().all()):
        df["N"] = sample_size
    if n_cases is not None:
        if "N_CAS" not in df.columns or df["N_CAS"].isna().all():
            df["N_CAS"] = n_cases
        else:
            df["N_CAS"] = df["N_CAS"].fillna(n_cases)
    if n_controls is not None:
        if "N_CON" not in df.columns or df["N_CON"].isna().all():
            df["N_CON"] = n_controls
        else:
            df["N_CON"] = df["N_CON"].fillna(n_controls)
    if sample_size is not None:
        df["N"] = df["N"].fillna(sample_size)

    if reference_bim is not None:
        _required_for_harmonisation = {"A1", "A2", "BETA"}
        missing = _required_for_harmonisation - set(df.columns)
        if missing:
            raise ValueError(
                f"Cannot harmonise alleles: required column(s) {sorted(missing)} "
                f"not found after column mapping. Available columns: "
                f"{sorted(df.columns.tolist())}. Check GWAS_COLUMN_ALIASES for "
                f"missing effect/non-effect allele aliases."
            )
        bim_path = check_file_exists(Path(reference_bim), label="Reference BIM")
        ref = pd.read_csv(bim_path, sep="\t", header=None, names=["CHR", "SNP", "CM", "POS", "REF", "ALT"])
        df = harmonize_alleles(df, ref)

    resolved_build = genome_build or detected.genome_build or "GRCh37"

    if genome_build is None and detected.genome_build is None:
        logger.warning(
            "Genome build could not be auto-detected and was not specified in config - "
            "defaulting to GRCh37. If your data is GRCh38, set genome_build='GRCh38' "
            "in StudyConfig to avoid incorrect variant positions."
        )

    if liftover_to and liftover_to != resolved_build and chain_file:
        df, _liftover_counters = apply_liftover(
            df, Path(chain_file), resolved_build, liftover_to,
        )
        liftover_applied = True
        liftover_source = resolved_build
        resolved_build = liftover_to
    else:
        liftover_applied = False
        liftover_source = None

    df = quality_control(
        df,
        info_threshold=info_threshold,
        maf_threshold=maf_threshold,
        remove_mhc=remove_mhc,
    )

    if len(df) == 0:
        raise RuntimeError(
            f"No variants survived QC for {input_path.name}. "
            "Check input format, column detection, and filter thresholds."
        )

    resolved_trait = trait_type or detected.trait_type or "unknown"

    # Thread case/control structure into the metadata sidecar so
    # Branch C coloc/Steiger can calibrate for a binary outcome. Scalar-N policy:
    # prefer the config-supplied scalar (invariant, defensible); otherwise fall
    # back to the per-variant median from the QC'd frame and record the range so
    # a reviewer can see how much N varies across variants.
    meta_n_cases, meta_n_controls, cc_n_source = _resolve_case_control_n(
        df, n_cases, n_controls,
    )
    meta_n_total = (
        meta_n_cases + meta_n_controls
        if meta_n_cases is not None and meta_n_controls is not None
        else None
    )

    metadata = GWASMetadata(
        genome_build=resolved_build,
        trait_type=resolved_trait,
        source=detected.source,
        original_n_variants=original_n,
        n_variants_after_qc=len(df),
        liftover_applied=liftover_applied,
        liftover_source=liftover_source,
        effect_allele_column=detected.effect_allele_column,
        n_cases=meta_n_cases,
        n_controls=meta_n_controls,
        n_total_cases_controls=meta_n_total,
        case_control_n_source=cc_n_source,
        population_prevalence=population_prevalence,
    )

    errors = validate_dataframe(df, "StandardizedGWAS")
    if errors:
        logger.warning(
            "Output does not fully conform to StandardizedGWAS schema: %s",
            "; ".join(errors),
        )

    logger.info(
        "GWAS prep complete: %d -> %d variants, build=%s, source=%s",
        original_n,
        len(df),
        resolved_build,
        detected.source,
    )
    return df, metadata


# ---------------------------------------------------------------------------
# eQTL mode (for future MR module)
# ---------------------------------------------------------------------------


def prepare_eqtl(
    input_path: Path,
    source: str,
    tissue: Optional[str] = None,
    **kwargs,
) -> tuple[pd.DataFrame, GWASMetadata]:
    """Same pipeline as :func:`prepare_gwas` with eQTL-specific detection.

    Output schema is ``StandardizedGWAS`` plus ``gene`` and ``tissue``
    columns.

    Args:
        input_path: Path to eQTL summary statistics.
        source: ``"gtex"`` or ``"eqtlgen"``.
        tissue: Tissue name (required for GTEx).
        **kwargs: Passed through to :func:`prepare_gwas`.

    Returns:
        Tuple of ``(DataFrame, GWASMetadata)``.
    """
    df, metadata = prepare_gwas(input_path, **kwargs)
    metadata.source = source

    if tissue:
        df["tissue"] = tissue

    if "gene" not in df.columns:
        logger.warning("No 'gene' column found in eQTL data - downstream MR will need to add it")

    return df, metadata


# ---------------------------------------------------------------------------
# Branch-B GWAS preparation (isolated liftover for S-PrediXcan)
# ---------------------------------------------------------------------------


def prepare_gwas_for_spredixcan(
    raw_gwas_path: Path,
    output_parquet: Path,
    output_meta: Path,
    chain_file: Path | None = None,
    target_build: str = "GRCh38",
    genome_build: str | None = None,
    trait_type: str | None = None,
    sample_size: int | None = None,
    n_cases: int | None = None,
    n_controls: int | None = None,
    info_threshold: float = 0.6,
    maf_threshold: float = 0.01,
) -> None:
    """Prepare a Branch-B-only GWAS artifact for S-PrediXcan.

    Reads raw GWAS summary statistics directly (not the shared
    ``gwas_standardized.parquet``) and applies column detection,
    mapping, QC, and liftover to *target_build* - but **no reference-BIM
    harmonization**.  This avoids the palindromic-SNP and allele-mismatch
    losses that BIM harmonization applies for Branch A/C, preserving
    maximum variant coverage for S-PrediXcan model matching.

    Branch A and Branch C inputs (``gwas_standardized.parquet``) are
    completely unaffected by this function.

    Args:
        raw_gwas_path: Path to raw GWAS summary statistics file.
        output_parquet: Output path for the Branch-B GWAS artifact.
        output_meta: Output path for the Branch-B metadata sidecar.
        chain_file: Path to UCSC liftover chain file (e.g. hg19ToHg38).
        target_build: Target genome build (default ``"GRCh38"``).
        genome_build: Source genome build (auto-detected if ``None``).
        trait_type: ``"case_control"`` or ``"quantitative"``.
        sample_size: Override total N.
        n_cases: Override case count.
        n_controls: Override control count.
        info_threshold: Minimum INFO score for QC.
        maf_threshold: Minimum MAF for QC.
    """
    logger.info(
        "Preparing Branch-B GWAS for S-PrediXcan from raw input: %s",
        raw_gwas_path,
    )

    df, gwas_meta = prepare_gwas(
        input_path=raw_gwas_path,
        reference_bim=None,
        genome_build=genome_build,
        trait_type=trait_type,
        sample_size=sample_size,
        n_cases=n_cases,
        n_controls=n_controls,
        info_threshold=info_threshold,
        maf_threshold=maf_threshold,
        liftover_to=target_build,
        chain_file=chain_file,
        remove_mhc=False,
    )

    meta = gwas_meta.model_dump()
    meta["spredixcan_artifact"] = True
    meta["branch_b_source"] = "raw_gwas"
    meta["bim_harmonization_applied"] = False
    meta["n_variants_spredixcan"] = len(df)

    ensure_directory(output_parquet.parent)
    df.to_parquet(output_parquet, engine="pyarrow", index=False)
    logger.info("Wrote Branch-B GWAS: %s (%d variants)", output_parquet, len(df))

    with open(output_meta, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Wrote Branch-B metadata: %s", output_meta)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GWAS summary statistics")
    parser.add_argument("--input", type=Path, help="Input GWAS file")
    parser.add_argument("--output", required=True, type=Path, help="Output Parquet file")
    parser.add_argument("--reference-bim", type=Path, default=None, help="Reference BIM file")
    parser.add_argument("--genome-build", default=None, help="GRCh37 or GRCh38")
    parser.add_argument("--trait-type", default=None, help="case_control or quantitative")
    parser.add_argument("--sample-size", type=int, default=None, help="Override total N")
    parser.add_argument("--n-cases", type=int, default=None)
    parser.add_argument("--n-controls", type=int, default=None)
    parser.add_argument(
        "--population-prevalence", type=float, default=None,
        help="Population disease prevalence (case/control traits) for the "
             "Branch C Steiger liability transform. Never the sample fraction.",
    )
    parser.add_argument("--info-threshold", type=float, default=0.6)
    parser.add_argument("--maf-threshold", type=float, default=0.01)
    parser.add_argument("--liftover-to", default=None, help="Target build for liftover")
    parser.add_argument("--chain-file", type=Path, default=None)
    parser.add_argument("--remove-mhc", action="store_true")
    parser.add_argument("--metadata-out", type=Path, default=None, help="Output metadata JSON file")
    parser.add_argument(
        "--spredixcan-prep", action="store_true",
        help="Branch-B mode: raw GWAS -> QC (no BIM) -> liftover -> S-PrediXcan artifact",
    )
    parser.add_argument("--target-build", default="GRCh38")
    args = parser.parse_args()

    if args.spredixcan_prep:
        if args.input is None:
            parser.error("--input is required for --spredixcan-prep mode")
        output_meta = args.metadata_out or args.output.with_suffix(".meta.json")
        prepare_gwas_for_spredixcan(
            raw_gwas_path=args.input,
            output_parquet=args.output,
            output_meta=output_meta,
            chain_file=args.chain_file,
            target_build=args.target_build,
            genome_build=args.genome_build,
            trait_type=args.trait_type,
            sample_size=args.sample_size,
            n_cases=args.n_cases,
            n_controls=args.n_controls,
            info_threshold=args.info_threshold,
            maf_threshold=args.maf_threshold,
        )
    else:
        if args.input is None:
            parser.error("--input is required")

        df, meta = prepare_gwas(
            input_path=args.input,
            reference_bim=args.reference_bim,
            genome_build=args.genome_build,
            trait_type=args.trait_type,
            sample_size=args.sample_size,
            n_cases=args.n_cases,
            n_controls=args.n_controls,
            population_prevalence=args.population_prevalence,
            info_threshold=args.info_threshold,
            maf_threshold=args.maf_threshold,
            liftover_to=args.liftover_to,
            chain_file=args.chain_file,
            remove_mhc=args.remove_mhc,
        )

        ensure_directory(args.output.parent)
        df.to_parquet(args.output, engine="pyarrow", index=False)
        logger.info("Wrote %d variants to %s", len(df), args.output)

        meta_path = args.metadata_out or args.output.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta.model_dump(), indent=2, default=str))
        logger.info("Wrote metadata to %s", meta_path)
