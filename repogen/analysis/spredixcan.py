"""S-PrediXcan disease gene expression signature.

Native reimplementation of the S-PrediXcan algorithm (Barbeira et al. 2018)
for computing gene-level association Z-scores from GWAS summary statistics
and pre-trained expression prediction models. Does not depend on MetaXcan
software - reads PredictDB model files directly.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from repogen.data.mhc_annotation import add_mhc_flag, load_mhc_gene_annotation
from repogen.data.schemas import validate_dataframe
from repogen.utils.constants import BRAIN_TISSUES, mhc_interval
from repogen.utils.logging import setup_logging

# Backward-compatible alias: the MHC annotation loader lives in the shared
# ``repogen.data.mhc_annotation`` module so Branch B and Branch C
# call the same machinery. The private name is retained for callers/tests.
_load_mhc_gene_annotation = load_mhc_gene_annotation

logger = setup_logging(__name__)

COMPLEMENT: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

_EXTRA_COL_MAP: dict[str, str] = {
    "n.snps.in.model": "n_snps_in_model",
    "pred.perf.R2": "pred_perf_r2",
    "pred.perf.pval": "pred_perf_pval",
    "pred.perf.qval": "pred_perf_qval",
}

TISSUE_PRESETS: dict[str, list[str]] = {
    "brain_13": list(BRAIN_TISSUES),
    "brain_extended": list(BRAIN_TISSUES) + [
        "Pituitary", "Adrenal_Gland", "Thyroid", "Whole_Blood",
    ],
}


# --- Model Loading ---------------------------------------------------


def load_prediction_model(model_db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SNP weights and gene metadata from a PredictDB .db file.

    Args:
        model_db_path: Path to the .db SQLite file.

    Returns:
        Tuple of (weights_df, gene_extra_df).
        weights_df columns: gene, rsid, weight, ref_allele, eff_allele
        gene_extra_df columns: gene, genename, n_snps_in_model,
            pred_perf_r2, pred_perf_pval

    Raises:
        FileNotFoundError: If the .db file does not exist.
    """
    if not model_db_path.exists():
        raise FileNotFoundError(f"PredictDB model file not found: {model_db_path}")

    conn = sqlite3.connect(str(model_db_path))
    try:
        weights_df = pd.read_sql_query("SELECT * FROM weights", conn)
        extra_df = pd.read_sql_query("SELECT * FROM extra", conn)
    finally:
        conn.close()

    actual_w_cols = set(weights_df.columns)
    expected_w = {"gene", "rsid", "weight", "ref_allele", "eff_allele"}
    if not expected_w.issubset(actual_w_cols):
        missing = expected_w - actual_w_cols
        raise ValueError(
            f"PredictDB weights table missing columns: {missing}. "
            f"Found: {sorted(actual_w_cols)}"
        )

    extra_df = extra_df.rename(columns=_EXTRA_COL_MAP)

    logger.debug(
        "Loaded model %s: %d weights, %d genes%s",
        model_db_path.name, len(weights_df), extra_df["gene"].nunique(),
        ", has varID" if "varID" in actual_w_cols else "",
    )
    return weights_df, extra_df


def load_covariance(cov_path: Path) -> dict[str, list[tuple[str, str, float]]]:
    """Load gene covariance data from a .txt.gz covariance file.

    Covariance entries are keyed by variant identifiers whose format
    depends on the model type.  For mashr models the keys are varIDs
    (``chr_pos_ref_alt_b38``); for elastic-net they may be rsIDs.

    Args:
        cov_path: Path to the gzipped covariance file.

    Returns:
        Dict mapping gene_id -> list of (key1, key2, covariance_value).

    Raises:
        FileNotFoundError: If the covariance file does not exist.
    """
    if not cov_path.exists():
        raise FileNotFoundError(f"Covariance file not found: {cov_path}")

    cov_data: dict[str, list[tuple[str, str, float]]] = {}
    with gzip.open(cov_path, "rt") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            gene, rsid1, rsid2, value = parts[0], parts[1], parts[2], parts[3]
            cov_data.setdefault(gene, []).append((rsid1, rsid2, float(value)))

    logger.debug("Loaded covariance for %d genes from %s", len(cov_data), cov_path.name)
    return cov_data


# --- Variant Matching ------------------------------------------------


def align_alleles(
    gwas_a1: np.ndarray,
    gwas_a2: np.ndarray,
    model_eff: np.ndarray,
    model_ref: np.ndarray,
    exclude_palindromic: bool = True,
) -> np.ndarray:
    """Vectorised allele alignment. Returns array of +1, -1, or 0.

    +1: alleles match (same strand, same direction)
    -1: alleles flipped (effect allele is GWAS non-effect allele)
     0: unresolvable (or palindromic when exclude_palindromic=True)

    When exclude_palindromic=False, palindromic SNPs (A/T, C/G) are
    aligned via normal direct/flip/complement logic. This accepts strand
    ambiguity risk - the alignment may be wrong for palindromic SNPs
    whose strand cannot be determined from allele identity alone.

    Args:
        gwas_a1: GWAS effect alleles (array of str).
        gwas_a2: GWAS non-effect alleles (array of str).
        model_eff: Model effect alleles (array of str).
        model_ref: Model reference alleles (array of str).
        exclude_palindromic: If True, return 0 for palindromic SNPs.

    Returns:
        Integer array of alignment codes (+1, -1, 0).
    """
    result = np.zeros(len(gwas_a1), dtype=np.int8)

    palindromic = (
        ((gwas_a1 == "A") & (gwas_a2 == "T"))
        | ((gwas_a1 == "T") & (gwas_a2 == "A"))
        | ((gwas_a1 == "C") & (gwas_a2 == "G"))
        | ((gwas_a1 == "G") & (gwas_a2 == "C"))
    )

    mask = ~palindromic if exclude_palindromic else np.ones(len(gwas_a1), dtype=bool)

    direct_match = (gwas_a1 == model_eff) & (gwas_a2 == model_ref)
    direct_flip = (gwas_a1 == model_ref) & (gwas_a2 == model_eff)

    comp_a1 = np.array([COMPLEMENT.get(a, "") for a in gwas_a1])
    comp_a2 = np.array([COMPLEMENT.get(a, "") for a in gwas_a2])
    comp_match = (comp_a1 == model_eff) & (comp_a2 == model_ref)
    comp_flip = (comp_a1 == model_ref) & (comp_a2 == model_eff)

    result[direct_match & mask] = 1
    result[direct_flip & mask] = -1
    result[(comp_match & ~direct_match & ~direct_flip) & mask] = 1
    result[(comp_flip & ~direct_match & ~direct_flip) & mask] = -1

    return result


def _precompute_gwas_lookups(
    gwas_df: pd.DataFrame,
) -> dict[str, pd.DataFrame | None]:
    """Precompute GWAS lookup tables for variant matching.

    Builds deduplicated, indexed DataFrames for SNP and (optionally)
    VARIANT_ID matching.  Intended to be called once per run and
    shared across all tissue calls, avoiding redundant O(n) GWAS
    preprocessing per tissue.

    Args:
        gwas_df: StandardizedGWAS DataFrame.

    Returns:
        Dict with keys ``"by_rsid"`` (DataFrame indexed by SNP) and
        ``"by_varid"`` (DataFrame indexed by VARIANT_ID, or ``None``).
    """
    gwas_z = gwas_df["BETA"] / gwas_df["SE"]

    gwas_lookup = gwas_df[["SNP", "A1", "A2"]].copy()
    gwas_lookup["Z"] = gwas_z
    gwas_by_rsid = gwas_lookup.drop_duplicates(subset="SNP").set_index("SNP")

    gwas_by_varid: pd.DataFrame | None = None
    if "VARIANT_ID" in gwas_df.columns:
        gv = gwas_df[["VARIANT_ID", "A1", "A2"]].copy()
        gv["Z"] = gwas_z
        gwas_by_varid = gv.drop_duplicates(subset="VARIANT_ID").set_index("VARIANT_ID")

    logger.info(
        "GWAS lookup tables: %d unique SNPs%s",
        len(gwas_by_rsid),
        f", {len(gwas_by_varid)} unique VARIANT_IDs" if gwas_by_varid is not None else "",
    )

    return {"by_rsid": gwas_by_rsid, "by_varid": gwas_by_varid}


def match_variants_to_gwas(
    model_weights: pd.DataFrame,
    gwas_df: pd.DataFrame,
    exclude_palindromic: bool = True,
    *,
    gwas_lookups: dict[str, pd.DataFrame | None] | None = None,
) -> pd.DataFrame:
    """Match model SNPs to GWAS variants and align alleles.

    Primary match: rsid (against GWAS SNP column).
    Fallback: rsid against GWAS VARIANT_ID (for chr:pos-style IDs).

    Uses a single vectorized merge instead of per-gene membership
    scans, reducing complexity from O(n_genes * n_gwas) to
    O(n_model_weights + n_gwas).

    Args:
        model_weights: DataFrame with columns
            ``[gene, rsid, weight, ref_allele, eff_allele]``.
        gwas_df: StandardizedGWAS DataFrame (used only when
            *gwas_lookups* is ``None``).
        exclude_palindromic: If True, skip palindromic (A/T, C/G) SNPs.
        gwas_lookups: Precomputed GWAS lookup tables from
            :func:`_precompute_gwas_lookups`.  If ``None``, lookups
            are computed from *gwas_df* (backward-compatible).

    Returns:
        DataFrame with columns ``[gene, rsid, weight, zscore, alignment]``
        where ``zscore = (BETA/SE) * alignment`` and rows with
        ``alignment == 0`` are dropped.
    """
    if gwas_lookups is None:
        gwas_lookups = _precompute_gwas_lookups(gwas_df)

    gwas_by_rsid = gwas_lookups["by_rsid"]
    gwas_by_varid = gwas_lookups["by_varid"]

    # --- Primary match: model rsid -> GWAS SNP ---
    merged = model_weights.merge(
        gwas_by_rsid[["A1", "A2", "Z"]],
        left_on="rsid",
        right_index=True,
        how="left",
    )

    rsid_hit = int(merged["Z"].notna().sum())

    # --- Fallback: unmatched rsids -> GWAS VARIANT_ID ---
    varid_hit = 0
    if gwas_by_varid is not None:
        unmatched_mask = merged["Z"].isna()
        if unmatched_mask.any():
            unmatched = merged.loc[unmatched_mask, ["gene", "rsid", "weight", "ref_allele", "eff_allele"]]
            fallback = unmatched.merge(
                gwas_by_varid[["A1", "A2", "Z"]],
                left_on="rsid",
                right_index=True,
                how="inner",
            )
            varid_hit = len(fallback)
            if varid_hit > 0:
                merged.loc[fallback.index, "A1"] = fallback["A1"]
                merged.loc[fallback.index, "A2"] = fallback["A2"]
                merged.loc[fallback.index, "Z"] = fallback["Z"]

    # --- Drop unmatched rows ---
    found_mask = merged["Z"].notna()
    miss = int((~found_mask).sum())
    matched = merged[found_mask].copy()

    logger.debug(
        "Variant matching: %d rsid hits, %d varid hits, %d misses",
        rsid_hit, varid_hit, miss,
    )

    if matched.empty:
        return pd.DataFrame(columns=["gene", "rsid", "weight", "zscore", "alignment"])

    # --- Vectorized allele alignment ---
    alignment = align_alleles(
        matched["A1"].values.astype(str),
        matched["A2"].values.astype(str),
        matched["eff_allele"].values.astype(str),
        matched["ref_allele"].values.astype(str),
        exclude_palindromic=exclude_palindromic,
    )

    valid_mask = alignment != 0
    result = matched.loc[valid_mask, ["gene", "rsid", "weight"]].copy()
    result["zscore"] = matched.loc[valid_mask, "Z"].values * alignment[valid_mask]
    result["alignment"] = alignment[valid_mask].astype(int)
    result = result.reset_index(drop=True)

    return result


# --- Core Computation ------------------------------------------------


def build_covariance_matrix(
    snp_ids: list[str],
    cov_entries: list[tuple[str, str, float]],
) -> np.ndarray:
    """Build symmetric covariance matrix for matched SNPs.

    Args:
        snp_ids: Ordered list of variant keys (varIDs or rsIDs) matching
            the key-space used in *cov_entries*.
        cov_entries: List of (key1, key2, covariance) for this gene.

    Returns:
        k x k numpy array. Diagonal = variance. Off-diagonal = covariance.
    """
    k = len(snp_ids)
    snp_to_idx = {snp: i for i, snp in enumerate(snp_ids)}
    cov = np.zeros((k, k), dtype=np.float64)

    for rsid1, rsid2, value in cov_entries:
        i = snp_to_idx.get(rsid1)
        j = snp_to_idx.get(rsid2)
        if i is not None and j is not None:
            cov[i, j] = value
            cov[j, i] = value

    return cov


def compute_gene_zscore(
    weights: np.ndarray,
    zscores: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute S-PrediXcan Z-score for a single gene.

    Args:
        weights: Model SNP weights (k,).
        zscores: Aligned GWAS Z-scores (k,).
        cov_matrix: SNP covariance matrix (k, k).

    Returns:
        Tuple of (zscore, pvalue, effect_size, se).
        Returns (NaN, NaN, NaN, NaN) if variance <= 0.
    """
    numerator = float(np.dot(weights, zscores))
    sigma = float(weights @ cov_matrix @ weights)

    if sigma <= 0:
        return (np.nan, np.nan, np.nan, np.nan)

    denom = np.sqrt(sigma)
    z = numerator / denom
    p = 2.0 * stats.norm.sf(abs(z))
    effect = numerator / sigma
    se = 1.0 / denom

    return (z, p, effect, se)


def run_spredixcan_tissue(
    gwas_df: pd.DataFrame,
    model_db_path: Path,
    cov_path: Path,
    min_snps_fraction: float = 0.1,
    exclude_palindromic: bool = True,
    *,
    gwas_lookups: dict[str, pd.DataFrame | None] | None = None,
) -> pd.DataFrame:
    """Run S-PrediXcan for a single tissue.

    Args:
        gwas_df: StandardizedGWAS DataFrame.
        model_db_path: Path to tissue .db model file.
        cov_path: Path to tissue .txt.gz covariance file.
        min_snps_fraction: Minimum fraction of model SNPs found in GWAS.
        exclude_palindromic: Skip palindromic SNPs in alignment.
        gwas_lookups: Precomputed GWAS lookup tables from
            :func:`_precompute_gwas_lookups`.  If ``None``, lookups
            are computed from *gwas_df* internally.

    Returns:
        DataFrame with per-tissue schema columns (without gene_entrez_id or mhc_flag).
    """
    t_model = time.perf_counter()
    weights_df, extra_df = load_prediction_model(model_db_path)
    t_model = time.perf_counter() - t_model

    t_cov = time.perf_counter()
    cov_data = load_covariance(cov_path)
    t_cov = time.perf_counter() - t_cov

    weights_df["gene_clean"] = weights_df["gene"].str.split(".").str[0]

    # Build rsid -> varID mapping for covariance key lookup
    has_varid = "varID" in weights_df.columns
    varid_map: dict[str, str] = {}
    if has_varid:
        _vmap = weights_df[["rsid", "varID"]].drop_duplicates(subset="rsid")
        _vmap = _vmap.dropna(subset=["varID"])
        varid_map = dict(zip(_vmap["rsid"], _vmap["varID"]))
        logger.info(
            "  varID mapping: %d/%d unique rsids have varID for covariance lookup",
            len(varid_map), weights_df["rsid"].nunique(),
        )

    extra_df = extra_df.copy()
    extra_df["gene_clean"] = extra_df["gene"].astype(str).str.split(".").str[0]
    extra_indexed = extra_df.set_index("gene_clean")
    extra_lookup: dict[str, dict[str, Any]] = {
        gene_clean: {
            "genename": vals.get("genename", ""),
            "n_snps_in_model": int(vals.get("n_snps_in_model", 0)),
            "pred_perf_r2": vals.get("pred_perf_r2"),
            "pred_perf_pval": vals.get("pred_perf_pval"),
        }
        for gene_clean, vals in extra_indexed.to_dict(orient="index").items()
    }

    t_match = time.perf_counter()
    matched_df = match_variants_to_gwas(
        weights_df, gwas_df,
        exclude_palindromic=exclude_palindromic,
        gwas_lookups=gwas_lookups,
    )
    t_match = time.perf_counter() - t_match

    n_matched_snps = len(matched_df)
    logger.info(
        "  model=%.2fs, cov=%.2fs, match=%.2fs (%d SNPs matched)",
        t_model, t_cov, t_match, n_matched_snps,
    )

    if matched_df.empty:
        return pd.DataFrame()

    t_score = time.perf_counter()
    results: list[dict[str, Any]] = []
    n_genes_total = matched_df["gene"].nunique()
    n_genes_done = 0
    n_skip_snpfrac = 0
    n_skip_variance = 0
    n_cov_identity_fallback = 0
    n_cov_overlap_total = 0
    for gene_id, gene_grp in matched_df.groupby("gene"):
        gene_clean = str(gene_id).split(".")[0]
        gene_extra = extra_lookup.get(gene_clean, extra_lookup.get(str(gene_id), {}))
        n_snps_in_model = gene_extra.get("n_snps_in_model", 0)

        n_matched = len(gene_grp)
        if n_snps_in_model > 0 and n_matched / n_snps_in_model < min_snps_fraction:
            logger.debug(
                "Gene %s: %d/%d SNPs matched (%.1f%%), below threshold %.0f%%, skipping",
                gene_clean, n_matched, n_snps_in_model,
                100 * n_matched / n_snps_in_model, 100 * min_snps_fraction,
            )
            n_genes_done += 1
            n_skip_snpfrac += 1
            continue

        rsids = gene_grp["rsid"].tolist()
        if varid_map:
            cov_keys = [varid_map.get(r, r) for r in rsids]
        else:
            cov_keys = rsids
        w = gene_grp["weight"].values.astype(np.float64)
        z = gene_grp["zscore"].values.astype(np.float64)

        gene_cov_entries = cov_data.get(str(gene_id), [])
        if not gene_cov_entries:
            gene_cov_entries = cov_data.get(gene_clean, [])

        if not gene_cov_entries:
            logger.debug("Gene %s: no covariance data, using identity", gene_clean)
            cov_matrix = np.eye(len(cov_keys), dtype=np.float64)
            n_cov_identity_fallback += 1
        else:
            cov_matrix = build_covariance_matrix(cov_keys, gene_cov_entries)
            cov_entry_keys = {e[0] for e in gene_cov_entries} | {e[1] for e in gene_cov_entries}
            n_cov_overlap_total += len(set(cov_keys) & cov_entry_keys)

        zscore, pvalue, effect_size, se = compute_gene_zscore(w, z, cov_matrix)
        if np.isnan(zscore):
            logger.debug("Gene %s: degenerate variance, skipping", gene_clean)
            n_genes_done += 1
            n_skip_variance += 1
            continue

        results.append({
            "gene_ensembl_id": gene_clean,
            "gene_symbol": gene_extra.get("genename", gene_clean),
            "zscore": zscore,
            "pvalue": pvalue,
            "effect_size": effect_size,
            "se": se,
            "n_snps_used": n_matched,
            "n_snps_in_model": n_snps_in_model if n_snps_in_model > 0 else n_matched,
            "pred_perf_r2": gene_extra.get("pred_perf_r2"),
            "pred_perf_pval": gene_extra.get("pred_perf_pval"),
        })

        n_genes_done += 1
        if n_genes_done % 1000 == 0:
            logger.info("  gene scoring: %d/%d genes", n_genes_done, n_genes_total)

    t_score = time.perf_counter() - t_score
    logger.info(
        "  scoring=%.2fs, %d/%d genes passed QC "
        "(skipped: %d snp_fraction, %d degenerate_variance, %d identity_cov_fallback; "
        "cov_overlap_snps=%d)",
        t_score, len(results), n_genes_total,
        n_skip_snpfrac, n_skip_variance, n_cov_identity_fallback,
        n_cov_overlap_total,
    )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# --- Meta-Analysis ---------------------------------------------------


def ivw_meta_analysis(
    per_tissue_df: pd.DataFrame,
    brain_tissues: list[str],
) -> tuple[pd.DataFrame, str]:
    """IVW fixed-effects meta-analysis across tissues.

    Applies deterministic tissue subsetting:
    1. All brain -> IVW across all
    2. Mixed brain + non-brain -> IVW across brain only
    3. No brain -> IVW across all configured

    Args:
        per_tissue_df: Stacked per-tissue results.
        brain_tissues: List of brain tissue names (from BRAIN_TISSUES constant).

    Returns:
        Tuple of (meta_df, ivw_tissue_subset_label).
    """
    configured_tissues = set(per_tissue_df["tissue"].unique())
    brain_set = set(brain_tissues)
    brain_present = configured_tissues & brain_set
    non_brain_present = configured_tissues - brain_set

    if non_brain_present and brain_present:
        subset_label = "brain_only"
        analysis_df = per_tissue_df[per_tissue_df["tissue"].isin(brain_present)].copy()
    elif not brain_present:
        subset_label = "all_configured"
        analysis_df = per_tissue_df.copy()
    else:
        subset_label = "all"
        analysis_df = per_tissue_df.copy()

    logger.info("IVW meta-analysis: subset=%s, %d tissues", subset_label, analysis_df["tissue"].nunique())

    meta_rows: list[dict[str, Any]] = []
    for gene_id, grp in analysis_df.groupby("gene_ensembl_id"):
        if len(grp) < 2:
            row = grp.iloc[0]
            meta_rows.append({
                "gene_ensembl_id": gene_id,
                "gene_symbol": row["gene_symbol"],
                "meta_zscore": row["zscore"],
                "meta_pvalue": row["pvalue"],
                "meta_beta": row["effect_size"],
                "meta_se": row["se"],
                "n_tissues": 1,
                "i_squared": 0.0,
                "q_statistic": 0.0,
                "q_pvalue": 1.0,
                "best_tissue": row["tissue"],
                "best_tissue_zscore": row["zscore"],
                "mhc_flag": row.get("mhc_flag", False),
            })
            if "gene_entrez_id" in grp.columns:
                meta_rows[-1]["gene_entrez_id"] = row.get("gene_entrez_id")
            continue

        betas = grp["effect_size"].values.astype(np.float64)
        ses = grp["se"].values.astype(np.float64)

        valid = (ses > 0) & np.isfinite(betas) & np.isfinite(ses)
        if valid.sum() < 1:
            continue

        betas = betas[valid]
        ses = ses[valid]
        k = len(betas)

        w = 1.0 / (ses ** 2)
        w_sum = w.sum()
        beta_pooled = np.dot(w, betas) / w_sum
        se_pooled = 1.0 / np.sqrt(w_sum)
        z_pooled = beta_pooled / se_pooled
        p_pooled = 2.0 * stats.norm.sf(abs(z_pooled))

        q_stat = float(np.dot(w, (betas - beta_pooled) ** 2))
        df = k - 1
        i_sq = max(0.0, (q_stat - df) / q_stat * 100) if q_stat > 0 else 0.0
        q_pval = float(1.0 - stats.chi2.cdf(q_stat, df)) if df > 0 else 1.0

        valid_grp = grp[valid.tolist()] if isinstance(valid, np.ndarray) else grp
        best_idx = valid_grp["pvalue"].idxmin()
        best_row = valid_grp.loc[best_idx]

        meta_entry: dict[str, Any] = {
            "gene_ensembl_id": gene_id,
            "gene_symbol": grp["gene_symbol"].iloc[0],
            "meta_zscore": float(z_pooled),
            "meta_pvalue": float(p_pooled),
            "meta_beta": float(beta_pooled),
            "meta_se": float(se_pooled),
            "n_tissues": k,
            "i_squared": float(i_sq),
            "q_statistic": float(q_stat),
            "q_pvalue": float(q_pval),
            "best_tissue": best_row["tissue"],
            "best_tissue_zscore": float(best_row["zscore"]),
            "mhc_flag": grp["mhc_flag"].any() if "mhc_flag" in grp.columns else False,
        }
        if "gene_entrez_id" in grp.columns:
            meta_entry["gene_entrez_id"] = grp["gene_entrez_id"].iloc[0]
        meta_rows.append(meta_entry)

    if not meta_rows:
        return pd.DataFrame(), subset_label

    meta_df = pd.DataFrame(meta_rows)
    return meta_df, subset_label


# --- Utilities --------------------------------------------------------


def resolve_model_paths(
    model_type: str,
    tissues: list[str],
    model_dir: Path,
    cov_dir: Path | None = None,
) -> list[tuple[str, Path, Path]]:
    """Resolve tissue names to (tissue_name, db_path, cov_path) tuples.

    Args:
        model_type: "mashr", "jti", or "elastic_net".
        tissues: List of GTEx tissue names.
        model_dir: Base directory containing model files.
        cov_dir: Covariance directory. If None, same as model_dir.

    Returns:
        List of (tissue_name, db_path, cov_path).

    Raises:
        FileNotFoundError: If any model file is missing.
    """
    effective_cov_dir = cov_dir if cov_dir is not None else model_dir
    result: list[tuple[str, Path, Path]] = []

    for tissue in tissues:
        db_name = f"{model_type}_{tissue}.db"
        cov_name = f"{model_type}_{tissue}.txt.gz"

        db_path = model_dir / db_name
        cp = effective_cov_dir / cov_name

        if not db_path.exists():
            raise FileNotFoundError(
                f"Model file missing for tissue '{tissue}': {db_path}"
            )
        if not cp.exists():
            raise FileNotFoundError(
                f"Covariance file missing for tissue '{tissue}': {cp}"
            )

        result.append((tissue, db_path, cp))

    return result


def add_entrez_ids(
    df: pd.DataFrame,
    gene_id_converter: Any,
) -> pd.DataFrame:
    """Add gene_entrez_id column via GeneIDConverter.

    Args:
        df: DataFrame with gene_ensembl_id column.
        gene_id_converter: GeneIDConverter instance.

    Returns:
        DataFrame with gene_entrez_id column.
    """
    df = df.copy()
    unique_ensembl = df["gene_ensembl_id"].unique()

    entrez_map: dict[str, int | None] = {}
    n_failed = 0
    for ens_id in unique_ensembl:
        try:
            record = gene_id_converter.get_full_record(str(ens_id), "ensembl")
            entrez_map[ens_id] = record.get("entrez") if record else None
        except (KeyError, ValueError):
            entrez_map[ens_id] = None
            n_failed += 1

    n_mapped = sum(1 for v in entrez_map.values() if v is not None)
    logger.info(
        "Entrez ID mapping: %d/%d genes mapped, %d lookup failures",
        n_mapped, len(unique_ensembl), n_failed,
    )

    df["gene_entrez_id"] = df["gene_ensembl_id"].map(entrez_map)
    df["gene_entrez_id"] = pd.array(df["gene_entrez_id"].values, dtype="Int64")
    return df


def _resolve_model_dir(
    reference: Any,
    config_nc: Any | None = None,
) -> Path:
    """Resolve PredictDB model directory with deprecation fallback.

    Args:
        reference: ReferenceConfig instance.
        config_nc: Optional NegativeCorrelationConfig for deprecated fallback.

    Returns:
        Resolved model directory path.

    Raises:
        FileNotFoundError: If no model directory is configured.
    """
    model_dir = getattr(reference, "predixcan_model_dir", None)

    if model_dir is None and config_nc is not None:
        nc_dir = getattr(config_nc, "predixcan_models_dir", None)
        if nc_dir is not None:
            logger.warning(
                "predixcan_models_dir in [negative_correlation] config is deprecated. "
                "Move to [reference].predixcan_model_dir. "
                "This fallback will be removed in the 8.7 module update."
            )
            model_dir = nc_dir

    if model_dir is None:
        raise FileNotFoundError(
            "No PredictDB model directory configured. "
            "Set reference.predixcan_model_dir in reference.yaml."
        )

    return Path(model_dir)


def _resolve_cov_dir(
    reference: Any,
    model_dir: Path,
    config_nc: Any | None = None,
) -> Path:
    """Resolve covariance directory with deprecation fallback.

    Args:
        reference: ReferenceConfig instance.
        model_dir: Already-resolved model directory (fallback).
        config_nc: Optional NegativeCorrelationConfig for deprecated fallback.

    Returns:
        Resolved covariance directory path.
    """
    cov_dir = getattr(reference, "predixcan_covariance_dir", None)

    if cov_dir is None and config_nc is not None:
        nc_cov = getattr(config_nc, "predixcan_covariances_dir", None)
        if nc_cov is not None:
            logger.warning(
                "predixcan_covariances_dir in [negative_correlation] config is deprecated. "
                "Move to [reference].predixcan_covariance_dir. "
                "This fallback will be removed in the 8.7 module update."
            )
            return Path(nc_cov)

    if cov_dir is None:
        logger.info(
            "No predixcan_covariance_dir set; assuming covariance files are in model_dir: %s",
            model_dir,
        )
        return model_dir

    return Path(cov_dir)


def _resolve_tissue_list(config: Any, model_dir: Path) -> list[str]:
    """Resolve the tissue list from config, handling presets.

    Args:
        config: SpredixcanConfig instance.
        model_dir: Resolved model directory (used for all_gtex discovery).

    Returns:
        List of tissue names.

    Raises:
        ValueError: If preset is 'custom' but no tissues are specified,
            or if 'all_gtex' finds no model files.
    """
    if config.tissues:
        return config.tissues

    preset = config.tissue_preset

    if preset in TISSUE_PRESETS:
        return TISSUE_PRESETS[preset]

    if preset == "all_gtex":
        model_type = config.model_type
        suffix = ".db"
        prefix = f"{model_type}_"
        discovered = sorted(
            p.stem[len(prefix):]
            for p in model_dir.glob(f"{prefix}*{suffix}")
        )
        if not discovered:
            raise ValueError(
                f"tissue_preset='all_gtex' but no {prefix}*{suffix} files found in {model_dir}"
            )
        logger.info("all_gtex preset: discovered %d tissues from %s", len(discovered), model_dir)
        return discovered

    if preset == "custom":
        raise ValueError(
            "tissue_preset='custom' requires a non-empty 'tissues' list in config"
        )

    raise ValueError(f"Unknown tissue_preset '{preset}'")


# --- Orchestrator -----------------------------------------------------


def run_spredixcan(
    gwas_path: Path,
    config: Any,
    reference: Any,
    output_dir: Path,
    gene_annotation: pd.DataFrame | None = None,
    gene_id_converter: Any | None = None,
    negative_correlation_config: Any | None = None,
    mhc_build: str = "GRCh38",
    mhc_annotation_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Run the full S-PrediXcan pipeline.

    Args:
        gwas_path: Path to StandardizedGWAS Parquet file.
        config: SpredixcanConfig from pipeline config.
        reference: ReferenceConfig with model directory paths.
        output_dir: Directory for output files.
        gene_annotation: Optional ``GeneAnnotationRecord``-shaped
            DataFrame for MHC flagging.  Coordinates must match
            *mhc_build*.  When this is non-empty AND zero spx rows
            land in the MHC interval, ``run_spredixcan`` always
            raises ``RuntimeError`` - this is the
            fail-loud guardrail that makes the previous silent inert-MHC
            bug structurally impossible.
        gene_id_converter: Optional GeneIDConverter for Entrez ID mapping.
        negative_correlation_config: Optional NegativeCorrelationConfig for deprecated fallback.
        mhc_build: Genome build of *gene_annotation*.
            Default ``"GRCh38"`` - matches PredictDB MASHR output
            coordinates.  Set to ``"GRCh37"`` only when the caller
            passes a GRCh37 annotation frame.
        mhc_annotation_metadata: Optional dict describing the MHC
            annotation provenance, merged into the output
            ``spredixcan_metadata.json``.  Produced by
            ``_load_mhc_gene_annotation`` for the production CLI path.

    Returns:
        Dict mapping output names to file paths.

    Raises:
        RuntimeError: If *gene_annotation* is non-empty but zero spx
            rows are flagged in_mhc (guardrail; tightened
 to always raise rather than only when
            MHC exclusion is downstream-required).
    """
    if config.gwas_imputation:
        raise NotImplementedError(
            "GWAS summary imputation is deferred to v2.0. "
            "Set gwas_imputation=false to use built-in variant harmonisation."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GWAS data from %s", gwas_path)
    gwas_df = pd.read_parquet(gwas_path)

    gwas_meta: dict[str, Any] = {}
    canonical_meta = gwas_path.with_suffix(".meta.json")
    legacy_meta = gwas_path.with_suffix(".json")
    if canonical_meta.exists():
        with open(canonical_meta) as f:
            gwas_meta = json.load(f)
        logger.info("Loaded GWAS sidecar: %s", canonical_meta)
    elif legacy_meta.exists():
        with open(legacy_meta) as f:
            gwas_meta = json.load(f)
        logger.info("Loaded GWAS sidecar (legacy path): %s", legacy_meta)

    genome_build = gwas_meta.get("genome_build", "")
    if genome_build and genome_build != "GRCh38":
        logger.warning(
            "GWAS genome build is %s, but PredictDB MASHR models use GRCh38. "
            "Variant matching may be impaired. Consider applying liftover.",
            genome_build,
        )

    model_dir = _resolve_model_dir(reference, negative_correlation_config)
    cov_dir = _resolve_cov_dir(reference, model_dir, negative_correlation_config)

    tissues = _resolve_tissue_list(config, model_dir)

    tissue_paths = resolve_model_paths(config.model_type, tissues, model_dir, cov_dir)

    logger.info("Precomputing GWAS lookup tables (%d rows)...", len(gwas_df))
    t_gwas_prep = time.perf_counter()
    gwas_lookups = _precompute_gwas_lookups(gwas_df)
    t_gwas_prep = time.perf_counter() - t_gwas_prep
    logger.info("GWAS preprocessing: %.2fs", t_gwas_prep)

    all_tissue_results: list[pd.DataFrame] = []
    n_genes_per_tissue: dict[str, int] = {}

    for tissue_name, db_path, cp in tissue_paths:
        t_tissue = time.perf_counter()
        logger.info("Processing tissue: %s", tissue_name)
        tissue_df = run_spredixcan_tissue(
            gwas_df, db_path, cp,
            min_snps_fraction=config.min_snps_used_fraction,
            exclude_palindromic=config.exclude_palindromic,
            gwas_lookups=gwas_lookups,
        )

        t_tissue = time.perf_counter() - t_tissue

        if tissue_df.empty:
            logger.warning("Tissue %s: 0 genes passed QC (%.2fs), skipping", tissue_name, t_tissue)
            n_genes_per_tissue[tissue_name] = 0
            continue

        tissue_df["tissue"] = tissue_name
        n_genes_per_tissue[tissue_name] = len(tissue_df)
        logger.info("Tissue %s: %d genes (%.2fs total)", tissue_name, len(tissue_df), t_tissue)
        all_tissue_results.append(tissue_df)

    if not all_tissue_results:
        raise ValueError(
            "S-PrediXcan produced no results across any tissue. "
            "Check GWAS-model variant overlap and genome build compatibility."
        )

    per_tissue_df = pd.concat(all_tissue_results, ignore_index=True)

    per_tissue_df = add_mhc_flag(per_tissue_df, gene_annotation, build=mhc_build)

    # If a
    # non-empty annotation was supplied (i.e. MHC flagging was intended)
    # but zero spx rows landed in the MHC interval, that always signals
    # a build mismatch or a broken annotation pipeline - never legitimate
    # in a production run. The previous inert MHC flag passed silently for
    # years; we must not regress, regardless of whether MHC exclusion is
    # requested downstream.
    n_mhc_flagged = int(per_tissue_df["mhc_flag"].sum())
    n_unique_mhc_in_spx = int(
        per_tissue_df.loc[per_tissue_df["mhc_flag"], "gene_ensembl_id"].nunique()
    )
    if (
        gene_annotation is not None
        and not gene_annotation.empty
        and n_mhc_flagged == 0
    ):
        raise RuntimeError(
            f"MHC annotation supplied ({len(gene_annotation)} rows, "
            f"build={mhc_build}) but zero S-PrediXcan rows were flagged "
            f"in_mhc. This always indicates a build mismatch, a broken "
            f"GeneIDConverter mapping, or an annotation containing no "
            f"MHC-region genes - never a legitimate production state. "
            f"Re-run `repogen setup-resources` to refresh "
            f"resources/reference/NCBI38.gene.loc, verify "
            f"reference.gene_loc_file_grch38 points at it, and check that "
            f"GeneIDConverter resources (BioMart dicts + NCBI gene_info) "
            f"are present."
        )

    if gene_id_converter is not None:
        per_tissue_df = add_entrez_ids(per_tissue_df, gene_id_converter)
    else:
        per_tissue_df["gene_entrez_id"] = pd.array(
            [pd.NA] * len(per_tissue_df), dtype="Int64"
        )

    schema_errors = validate_dataframe(per_tissue_df, "DiseaseSignaturePerTissue")
    if schema_errors:
        logger.warning("Per-tissue schema validation: %d issues", len(schema_errors))

    meta_df, ivw_subset = ivw_meta_analysis(per_tissue_df, list(BRAIN_TISSUES))

    if not meta_df.empty:
        if "gene_entrez_id" not in meta_df.columns:
            meta_df["gene_entrez_id"] = pd.array(
                [pd.NA] * len(meta_df), dtype="Int64"
            )
        meta_errors = validate_dataframe(meta_df, "DiseaseSignatureMeta")
        if meta_errors:
            logger.warning("Meta schema validation: %d issues", len(meta_errors))

    per_tissue_path = output_dir / "spredixcan_per_tissue.parquet"
    meta_path = output_dir / "spredixcan_meta_analysis.parquet"
    metadata_path = output_dir / "spredixcan_metadata.json"

    per_tissue_df.to_parquet(per_tissue_path, index=False)
    logger.info("Per-tissue results saved: %s (%d rows)", per_tissue_path, len(per_tissue_df))

    if not meta_df.empty:
        meta_df.to_parquet(meta_path, index=False)
        logger.info("Meta-analysis results saved: %s (%d genes)", meta_path, len(meta_df))
    else:
        meta_df.to_parquet(meta_path, index=False)

    try:
        from importlib.metadata import version as pkg_version
        impl_version = pkg_version("repogen")
    except Exception:
        impl_version = "unknown"

    n_total_unique_genes = int(per_tissue_df["gene_ensembl_id"].nunique())
    n_with_entrez = int(
        per_tissue_df.loc[per_tissue_df["gene_entrez_id"].notna(), "gene_ensembl_id"]
        .nunique()
    )

    # MHC / Entrez observability block. Always present so
    # downstream tooling can rely on a stable schema; values default to
    # the "no MHC annotation" case when the caller did not pass one.
    # coordinate-mode + source-interval fields make the
    # membership-only semantics explicit and distinguish source vs flag
    # bounds for the GRCh37 fallback path.
    mhc_meta_input = mhc_annotation_metadata or {
        "mhc_annotation_source": None,
        "mhc_annotation_build": None,
        "mhc_annotation_strategy": "no_annotation",
        "mhc_annotation_coordinate_mode": "membership_interval_placeholder",
        "mhc_source_interval": None,
        "mhc_flag_interval": list(mhc_interval(mhc_build)),
        "n_unique_mhc_entrez_ids_in_gene_loc": 0,
        "n_unique_mhc_ensembl_ids_after_conversion": 0,
    }

    metadata: dict[str, Any] = {
        "trait": gwas_meta.get("trait", ""),
        "model_type": config.model_type,
        "tissues": tissues,
        "extra_models": [str(p) for p in config.extra_models],
        "gwas_imputation": config.gwas_imputation,
        "genome_build": genome_build,
        "n_genes_per_tissue": n_genes_per_tissue,
        "n_genes_meta": len(meta_df),
        "mhc_genes_count": n_mhc_flagged if "mhc_flag" in per_tissue_df.columns else 0,
        # MHC annotation provenance + coverage
        "mhc_annotation_source": mhc_meta_input.get("mhc_annotation_source"),
        "mhc_annotation_build": mhc_meta_input.get("mhc_annotation_build"),
        "mhc_annotation_strategy": mhc_meta_input.get("mhc_annotation_strategy"),
        "mhc_annotation_coordinate_mode": mhc_meta_input.get(
            "mhc_annotation_coordinate_mode",
            "membership_interval_placeholder",
        ),
        "mhc_source_interval": mhc_meta_input.get("mhc_source_interval"),
        "mhc_flag_interval": mhc_meta_input.get("mhc_flag_interval"),
        "n_unique_mhc_entrez_ids_in_gene_loc": mhc_meta_input.get(
            "n_unique_mhc_entrez_ids_in_gene_loc", 0
        ),
        "n_unique_mhc_ensembl_ids_after_conversion": mhc_meta_input.get(
            "n_unique_mhc_ensembl_ids_after_conversion", 0
        ),
        "n_unique_mhc_ensembl_ids_in_spx": n_unique_mhc_in_spx,
        "n_mhc_flagged_rows": n_mhc_flagged,
        "n_spx_unique_genes_total": n_total_unique_genes,
        "n_spx_unique_genes_with_entrez": n_with_entrez,
        # End block
        "implementation": "repogen_native",
        "implementation_version": impl_version,
        "model_source": "https://predictdb.org/",
        "ivw_tissue_subset": ivw_subset,
        "meta_analysis_note": (
            "IVW meta-analysis assumes tissue independence; p-values are approximate "
            "due to cross-tissue expression correlation (median r ≈ 0.56). "
            "Use per-tissue results as primary output for drug repurposing."
        ),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: %s", metadata_path)

    total_genes = per_tissue_df["gene_ensembl_id"].nunique()
    logger.info(
        "S-PrediXcan complete: %d unique genes across %d tissues, %d meta-analysis genes",
        total_genes, len(n_genes_per_tissue), len(meta_df),
    )

    return {
        "per_tissue": per_tissue_path,
        "meta_analysis": meta_path,
        "metadata": metadata_path,
    }


if __name__ == "__main__":
    import argparse
    from repogen.config.loader import load_config

    parser = argparse.ArgumentParser(description="Run S-PrediXcan analysis")
    parser.add_argument("--gwas-path", required=True, type=Path, help="Path to GWAS Parquet")
    parser.add_argument("--config-path", required=True, type=Path, help="Path to config YAML")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    pipeline_config = load_config(args.config_path)
    ref = pipeline_config.reference
    nc_config = getattr(pipeline_config, "negative_correlation", None)

    # Defensively construct GeneIDConverter (mirrors the
    # pattern in repogen.analysis.negative_correlation.run_negative_correlation).
    # Without this, gene_entrez_id stays null in spx output and MHC
    # flagging is inert - the two previous bugs.
    converter = None
    try:
        from repogen.data.gene_id_converter import GeneIDConverter
        if ref.ensembl_to_name and ref.name_to_ensembl:
            biomart: dict[str, Path] = {
                "ensembl_to_name": ref.ensembl_to_name,
                "name_to_ensembl": ref.name_to_ensembl,
            }
            if ref.uniprot_to_ensembl is not None:
                biomart["uniprot_to_ensembl"] = ref.uniprot_to_ensembl
            converter = GeneIDConverter(
                biomart_dicts=biomart,
                entrez_mapping_file=ref.ncbi_gene_info,
                gene_history_file=ref.ncbi_gene_history,
            )
            logger.info("GeneIDConverter constructed for Entrez/MHC enrichment")
    except (ImportError, FileNotFoundError) as exc:
        logger.warning(
            "GeneIDConverter unavailable (%s); spx output will have null "
            "gene_entrez_id and degraded MHC flagging", exc,
        )

    # Load build-aware MHC annotation (GRCh38 primary, GRCh37 fallback).
    # When negative_correlation.exclude_mhc is True we require the
    # annotation - a silent "no annotation" path was the original bug.
    require_mhc = bool(
        nc_config is not None and getattr(nc_config, "exclude_mhc", False)
    )
    mhc_annotation, mhc_metadata = _load_mhc_gene_annotation(
        ref, converter, require_mhc_annotation=require_mhc,
    )

    result = run_spredixcan(
        gwas_path=args.gwas_path,
        config=pipeline_config.spredixcan,
        reference=ref,
        output_dir=args.output_dir,
        gene_annotation=mhc_annotation,
        gene_id_converter=converter,
        negative_correlation_config=nc_config,
        mhc_build="GRCh38",
        mhc_annotation_metadata=mhc_metadata,
    )
    for name, path in result.items():
        logger.info("Output %s: %s", name, path)
