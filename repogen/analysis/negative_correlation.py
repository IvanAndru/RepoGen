"""Negative correlation analysis between disease and drug expression signatures.

Computes Spearman rank correlation between S-PrediXcan disease gene
expression signatures (per tissue) and LINCS L1000 drug expression
signatures.  A strongly negative correlation suggests the drug may
counteract disease-associated gene expression changes.

Primary inference: Spearman rho -> two-sided p-value -> global BH-FDR q-value.
Supplementary: XSum (eXtreme Sum) as a descriptive concordance metric.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2, spearmanr
from statsmodels.stats.multitest import multipletests

from repogen.config.schema import PipelineConfig
from repogen.data.schemas import validate_dataframe
from repogen.utils.io import ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

MATCH_CONFIDENCE_RANK = {"inchikey": 3, "pubchem_cid": 2, "name": 1}

# Theoretical median of chi²_1 = qchisq(0.5, df=1) ≈ 0.4549.
# Used as the reference in the theoretical genomic-control lambda.
# Cached at import time so the calibration helper does not repeatedly
# call ``scipy.stats.chi2.isf(0.5, 1)`` (deterministic constant).
CHI2_1_MEDIAN: float = float(chi2.isf(0.5, df=1))

# Defensive clip bounds for p-value -> chi² conversion.
# Mirrors the existing in-pipeline precedent at ``repogen/plotting/qq.py``
# to prevent ``chi2.isf(0, df=1) = inf`` from corrupting median-based
# lambda estimates.  ``1e-300`` stays well above IEEE-754 subnormals.
PVAL_CLIP_LOW: float = 1e-300
PVAL_CLIP_HIGH: float = 1.0


# ---------------------------------------------------------------------------
# Phase A: Data loading and preparation
# ---------------------------------------------------------------------------


def load_lincs_gene_info(
    path: Path,
    gene_set_mode: str,
) -> set[int]:
    """Load lincs_gene_info.tsv and return allowed Entrez IDs for the given gene set mode.

    Args:
        path: Path to lincs_gene_info.tsv
        gene_set_mode: One of "landmark", "landmark_bing", "all"

    Returns:
        Set of allowed Entrez Gene IDs
    """
    df = pd.read_csv(path, sep="\t")

    if gene_set_mode == "landmark":
        mask = df["is_landmark"] == True  # noqa: E712
    elif gene_set_mode == "landmark_bing":
        mask = (df["is_landmark"] == True) | (df["is_bing"] == True)  # noqa: E712
    elif gene_set_mode == "all":
        mask = pd.Series(True, index=df.index)
    else:
        raise ValueError(f"Unknown gene_set_mode: {gene_set_mode}")

    allowed = set(df.loc[mask, "entrez_id"].dropna().astype(int))
    logger.info("Loaded %d genes for gene_set_mode='%s' from %s", len(allowed), gene_set_mode, path)
    return allowed


def filter_drug_signatures(
    drug_sigs: pd.DataFrame,
    allowed_gene_set: set[int],
    match_confidence_threshold: str,
    min_profiles: int,
) -> tuple[dict[str, dict[int, float]], pd.DataFrame]:
    """Filter drug signatures by quality, match confidence, and gene set.

    Args:
        drug_sigs: DrugSignatureRecord DataFrame
        allowed_gene_set: Set of Entrez IDs to keep
        match_confidence_threshold: Minimum match confidence tier
        min_profiles: Minimum n_profiles_aggregated

    Returns:
        Tuple of:
        - Dict mapping lincs_pert_id -> {entrez_id: z_score} for filtered genes
        - Filtered DataFrame retaining all metadata columns for the surviving drugs
    """
    min_rank = MATCH_CONFIDENCE_RANK[match_confidence_threshold]

    n_total = len(drug_sigs)
    n_dropped_profiles = 0
    n_dropped_confidence = 0

    profile_mask = drug_sigs["n_profiles_aggregated"] >= min_profiles
    n_dropped_profiles = int((~profile_mask).sum())
    df = drug_sigs[profile_mask].copy()

    conf_ranks = df["match_confidence"].map(MATCH_CONFIDENCE_RANK)
    conf_mask = conf_ranks >= min_rank
    n_dropped_confidence = int((~conf_mask).sum())
    df = df[conf_mask].copy()

    logger.info(
        "Drug signature filtering: %d total, %d dropped (low profiles), "
        "%d dropped (match confidence), %d surviving",
        n_total, n_dropped_profiles, n_dropped_confidence, len(df),
    )

    drug_zscore_dicts: dict[str, dict[int, float]] = {}
    for tup in df.itertuples():
        pert_id = tup.lincs_pert_id
        gene_ids = tup.gene_ids if isinstance(tup.gene_ids, list) else list(tup.gene_ids)
        z_scores = tup.z_scores if isinstance(tup.z_scores, list) else list(tup.z_scores)

        filtered = {
            int(gid): float(z)
            for gid, z in zip(gene_ids, z_scores)
            if int(gid) in allowed_gene_set
        }
        if filtered:
            drug_zscore_dicts[pert_id] = filtered

    df_out = df.set_index("lincs_pert_id")
    df_out = df_out.loc[df_out.index.isin(drug_zscore_dicts.keys())]

    logger.info("After gene set filtering: %d drugs with non-empty signatures", len(drug_zscore_dicts))
    return drug_zscore_dicts, df_out


def prepare_disease_signatures(
    disease_sigs: pd.DataFrame,
    exclude_mhc: bool,
    gene_id_converter=None,
) -> dict[str, dict[int, float]]:
    """Prepare per-tissue disease Z-score dicts from DiseaseSignaturePerTissue.

    Args:
        disease_sigs: DiseaseSignaturePerTissue DataFrame
        exclude_mhc: Whether to exclude MHC region genes
        gene_id_converter: Optional GeneIDConverter for Entrez fallback

    Returns:
        Dict mapping tissue_name -> {entrez_id: zscore}
    """
    df = disease_sigs.copy()

    null_entrez_mask = df["gene_entrez_id"].isna()
    n_null_before = int(null_entrez_mask.sum())

    if n_null_before > 0 and gene_id_converter is not None:
        null_rows = df.loc[null_entrez_mask]
        ensembl_ids = null_rows["gene_ensembl_id"].unique().tolist()
        mapping = gene_id_converter.convert(ensembl_ids, "ensembl", "entrez")
        n_recovered = 0
        for ens_id, entrez_str in mapping.items():
            if entrez_str is not None:
                mask = (df["gene_ensembl_id"] == ens_id) & df["gene_entrez_id"].isna()
                df.loc[mask, "gene_entrez_id"] = int(entrez_str)
                n_recovered += int(mask.sum())
        logger.info("Recovered %d Entrez IDs via GeneIDConverter fallback", n_recovered)

    null_after = df["gene_entrez_id"].isna()
    n_dropped = int(null_after.sum())
    if n_dropped > 0:
        logger.info("Dropped %d genes with no Entrez mapping after fallback", n_dropped)
    df = df[~null_after].copy()
    df["gene_entrez_id"] = df["gene_entrez_id"].astype(int)

    if exclude_mhc and "mhc_flag" in df.columns:
        n_mhc = int(df["mhc_flag"].sum())
        df = df[~df["mhc_flag"]].copy()
        if n_mhc > 0:
            logger.info("Excluded %d MHC genes from disease signatures", n_mhc)

    tissue_dicts: dict[str, dict[int, float]] = {}
    for tissue, group in df.groupby("tissue"):
        zdict = dict(zip(group["gene_entrez_id"].values, group["zscore"].values))
        if len(zdict) == 0:
            logger.warning("Tissue %s has 0 genes with Entrez IDs - skipping", tissue)
            continue
        tissue_dicts[str(tissue)] = zdict

    logger.info("Prepared disease signatures for %d tissues", len(tissue_dicts))
    return tissue_dicts


# ---------------------------------------------------------------------------
# Phase B: Correlation computation
# ---------------------------------------------------------------------------


def compute_spearman(
    disease_zscores: dict[int, float],
    drug_zscores: dict[int, float],
    min_overlap: int,
) -> Optional[tuple[float, float, int, list[int]]]:
    """Compute Spearman correlation between disease and drug Z-score vectors.

    Args:
        disease_zscores: {entrez_id: zscore} for disease
        drug_zscores: {entrez_id: zscore} for drug
        min_overlap: Minimum overlapping genes required

    Returns:
        (rho, pvalue, n_overlapping, overlapping_gene_ids) or None if insufficient overlap
    """
    overlapping = sorted(set(disease_zscores.keys()) & set(drug_zscores.keys()))
    if len(overlapping) < min_overlap:
        return None

    disease_vec = np.array([disease_zscores[g] for g in overlapping])
    drug_vec = np.array([drug_zscores[g] for g in overlapping])

    rho, pvalue = spearmanr(disease_vec, drug_vec)
    return float(rho), float(pvalue), len(overlapping), overlapping


def compute_xsum(
    disease_zscores: dict[int, float],
    drug_zscores: dict[int, float],
    overlapping_genes: list[int],
    top_n: int,
    n_permutations: int = 0,
    xsum_seed: int = 42,
) -> tuple[Optional[float], Optional[float]]:
    """Compute eXtreme Sum score between disease and drug signatures.

    Args:
        disease_zscores: {entrez_id: zscore} for disease
        drug_zscores: {entrez_id: zscore} for drug
        overlapping_genes: List of shared Entrez IDs
        top_n: Number of top extreme genes to use
        n_permutations: Number of permutations for p-value (0 = no permutation)
        xsum_seed: RNG seed for permutation.  Default ``42`` preserves the
            byte-for-byte behaviour of the previous hard-coded ``default_rng(42)``.
            Only consulted when ``n_permutations > 0``.

    Returns:
        (xsum_score, xsum_pvalue) - both None if len(overlapping_genes) < top_n
    """
    if len(overlapping_genes) < top_n:
        return None, None

    drug_abs = np.array([abs(drug_zscores[g]) for g in overlapping_genes])
    top_indices = np.argsort(drug_abs)[-top_n:]
    top_genes = [overlapping_genes[i] for i in top_indices]

    disease_vals = np.array([disease_zscores[g] for g in top_genes])
    drug_vals = np.array([drug_zscores[g] for g in top_genes])

    up_mask = drug_vals > 0
    down_mask = drug_vals < 0
    xsum = float(np.sum(disease_vals[up_mask]) + np.sum(-disease_vals[down_mask]))

    if n_permutations <= 0:
        return xsum, None

    rng = np.random.default_rng(xsum_seed)
    all_disease_vals = np.array([disease_zscores[g] for g in overlapping_genes])
    n_extreme = 0
    for _ in range(n_permutations):
        perm_vals = rng.permutation(all_disease_vals)[:top_n]
        perm_xsum = float(np.sum(perm_vals[up_mask]) + np.sum(-perm_vals[down_mask]))
        if perm_xsum <= xsum:
            n_extreme += 1
    pvalue = (n_extreme + 1) / (n_permutations + 1)
    return xsum, float(pvalue)


def compute_top_contributing_genes(
    disease_zscores: dict[int, float],
    drug_zscores: dict[int, float],
    overlapping_genes: list[int],
    disease_sigs: pd.DataFrame,
    tissue: str,
    n_top: int = 10,
) -> list[str]:
    """Identify genes contributing most to the correlation signal.

    Args:
        disease_zscores, drug_zscores: Z-score dicts
        overlapping_genes: Shared gene IDs
        disease_sigs: Full disease signature DF (for gene_symbol lookup)
        tissue: Current tissue name
        n_top: Number of genes to return

    Returns:
        List of gene symbols sorted by abs(disease_Z * drug_Z) descending
    """
    contributions = []
    for g in overlapping_genes:
        score = abs(disease_zscores[g] * drug_zscores[g])
        contributions.append((g, score))
    contributions.sort(key=lambda x: x[1], reverse=True)
    top_gene_ids = [g for g, _ in contributions[:n_top]]

    tissue_df = disease_sigs[disease_sigs["tissue"] == tissue]
    entrez_to_symbol = dict(zip(
        tissue_df["gene_entrez_id"].dropna().astype(int),
        tissue_df["gene_symbol"],
    ))

    return [entrez_to_symbol.get(g, str(g)) for g in top_gene_ids]


# ---------------------------------------------------------------------------
# Phase E: Drug metadata enrichment
# ---------------------------------------------------------------------------


def _collect_string_values(series_values) -> list[str]:
    """Collect string elements from a pandas Series of containers or scalars.

    Handles ``numpy.ndarray``, ``list``, ``tuple``, ``set``, and bare
    ``str`` values as they appear after a Parquet/PyArrow round-trip.

    Returns:
        Flat, deduplicated, sorted ``list[str]``.
    """
    result: list[str] = []
    for val in series_values:
        if isinstance(val, str):
            result.append(val)
        elif hasattr(val, "__iter__"):
            result.extend(str(x) for x in val)
    return sorted(set(result))


def aggregate_drug_metadata(
    drug_targets: pd.DataFrame,
) -> dict[str, dict]:
    """Pre-aggregate DrugTargetRecord into per-drug metadata dicts.

    Builds two lookup indices (drug_inchikey and drug_chembl_id) so that
    the enrichment step can join by whichever identifier is available on the
    DrugSignatureRecord. Both indices point to the same aggregated dicts.

    Args:
        drug_targets: DrugTargetRecord DataFrame

    Returns:
        Dict with two sub-dicts:
        - "by_inchikey": drug_inchikey -> aggregated metadata dict
        - "by_chembl_id": drug_chembl_id -> aggregated metadata dict
    """
    by_inchikey: dict[str, dict] = {}
    by_chembl_id: dict[str, dict] = {}

    if drug_targets is None or len(drug_targets) == 0:
        return {"by_inchikey": by_inchikey, "by_chembl_id": by_chembl_id}

    for drug_id, group in drug_targets.groupby("drug_chembl_id"):
        phases = group["max_phase"].dropna()
        clinical_phase = int(phases.max()) if len(phases) > 0 else None

        moa_vals = group["mechanism_of_action"].dropna()
        mechanism_of_action = str(moa_vals.iloc[0]) if len(moa_vals) > 0 else None

        known_targets = sorted(set(group["gene_symbol"].dropna()))

        known_indications = (
            _collect_string_values(group["indication_mesh"].dropna())
            if "indication_mesh" in group.columns
            else []
        )

        atc_codes = (
            _collect_string_values(group["atc_codes"].dropna())
            if "atc_codes" in group.columns
            else []
        )

        pubchem_vals = group.get("drug_pubchem_cid", pd.Series(dtype=object)).dropna()
        drug_pubchem_cid = str(pubchem_vals.iloc[0]) if len(pubchem_vals) > 0 else None

        meta = {
            "drug_pubchem_cid": drug_pubchem_cid,
            "drug_chembl_id": str(drug_id),
            "clinical_phase": clinical_phase,
            "mechanism_of_action": mechanism_of_action,
            "known_targets": known_targets,
            "known_indications": known_indications,
            "atc_codes": atc_codes,
        }

        by_chembl_id[str(drug_id)] = meta

        inchikey_vals = group.get("drug_inchikey", pd.Series(dtype=object)).dropna()
        if len(inchikey_vals) > 0:
            by_inchikey[str(inchikey_vals.iloc[0])] = meta

    return {"by_inchikey": by_inchikey, "by_chembl_id": by_chembl_id}


def _enrich_row_metadata(
    pert_id: str,
    drug_meta_df: pd.DataFrame,
    agg_metadata: dict[str, dict],
) -> dict:
    """Look up DrugTargetRecord metadata for a single drug signature."""
    row = drug_meta_df.loc[pert_id] if pert_id in drug_meta_df.index else None
    if row is None:
        return {
            "drug_pubchem_cid": None, "drug_chembl_id": None,
            "atc_codes": [], "clinical_phase": None,
            "mechanism_of_action": None, "known_targets": [],
            "known_indications": [],
        }

    inchikey = getattr(row, "drug_inchikey", None)
    chembl_id = getattr(row, "drug_chembl_id", None)

    by_ik = agg_metadata["by_inchikey"]
    by_cid = agg_metadata["by_chembl_id"]

    meta = None
    if inchikey and str(inchikey) != "nan" and str(inchikey) in by_ik:
        meta = by_ik[str(inchikey)]
    elif chembl_id and str(chembl_id) != "nan" and str(chembl_id) in by_cid:
        meta = by_cid[str(chembl_id)]

    if meta is None:
        return {
            "drug_pubchem_cid": None,
            "drug_chembl_id": str(chembl_id) if chembl_id and str(chembl_id) != "nan" else None,
            "atc_codes": [], "clinical_phase": None,
            "mechanism_of_action": None, "known_targets": [],
            "known_indications": [],
        }

    result = dict(meta)
    if result["drug_chembl_id"] is None and chembl_id and str(chembl_id) != "nan":
        result["drug_chembl_id"] = str(chembl_id)
    return result


# ---------------------------------------------------------------------------
# directional (one-sided reversal) inference helper
# ---------------------------------------------------------------------------


def _directional_pvalue(rho: float, two_sided_p: float) -> float:
    """Convert a two-sided Spearman p-value into a one-sided reversal p-value.

    The one-sided lower-tail p-value tests ``H0: rho >= 0`` vs
    ``H1: rho < 0`` (reversal / negative correlation).  For a
    symmetric-null two-sided test:

    * ``p_one_sided = p_two_sided / 2``       when ``rho < 0``
    * ``p_one_sided = 1 - p_two_sided / 2``   when ``rho > 0``
    * ``p_one_sided = 0.5``                    when ``rho == 0``

    NaN in either input propagates to NaN out.  Output is clipped to
    ``[0, 1]`` to guard against floating-point drift.

    Args:
        rho: Observed Spearman correlation.
        two_sided_p: Two-sided p-value from ``scipy.stats.spearmanr``.

    Returns:
        One-sided lower-tail p-value in ``[0, 1]`` (or NaN if either
        input is NaN).
    """
    if np.isnan(rho) or np.isnan(two_sided_p):
        return float("nan")
    if rho == 0.0:
        return 0.5
    if rho < 0.0:
        return float(np.clip(two_sided_p / 2.0, 0.0, 1.0))
    return float(np.clip(1.0 - two_sided_p / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Phase B continued: Core correlation loop
# ---------------------------------------------------------------------------


def _run_correlation_loop(
    tissue_dicts: dict[str, dict[int, float]],
    drug_zscore_dicts: dict[str, dict[int, float]],
    drug_meta_df: pd.DataFrame,
    disease_sigs: pd.DataFrame,
    agg_metadata: dict[str, dict],
    min_overlap: int,
    xsum_top_n: int,
    xsum_permutations: int,
    xsum_seed: int = 42,
) -> tuple[list[dict], dict]:
    """Run the Spearman correlation loop over all tissue × drug pairs.

    Returns:
        Tuple of (results_rows, skip_counts)
    """
    results: list[dict] = []
    n_skipped_overlap = 0

    for tissue, disease_z in tissue_dicts.items():
        n_disease_genes = len(disease_z)
        for pert_id, drug_z in drug_zscore_dicts.items():
            result = compute_spearman(disease_z, drug_z, min_overlap)
            if result is None:
                n_skipped_overlap += 1
                logger.debug("Insufficient overlap for %s in %s", pert_id, tissue)
                continue

            rho, pvalue, n_overlap, overlapping_genes = result
            n_drug_genes = len(drug_z)

            xsum_score, xsum_pvalue = compute_xsum(
                disease_z, drug_z, overlapping_genes,
                xsum_top_n, xsum_permutations,
                xsum_seed=xsum_seed,
            )

            directional_p = _directional_pvalue(rho, pvalue)

            top_genes = compute_top_contributing_genes(
                disease_z, drug_z, overlapping_genes,
                disease_sigs, tissue,
            )

            drug_row = drug_meta_df.loc[pert_id] if pert_id in drug_meta_df.index else None
            drug_name = str(getattr(drug_row, "drug_name", pert_id)) if drug_row is not None else pert_id
            drug_inchikey = getattr(drug_row, "drug_inchikey", None) if drug_row is not None else None
            if drug_inchikey is not None and str(drug_inchikey) == "nan":
                drug_inchikey = None
            match_conf = str(getattr(drug_row, "match_confidence", "unknown")) if drug_row is not None else "unknown"
            n_profiles = int(getattr(drug_row, "n_profiles_aggregated", 0)) if drug_row is not None else 0
            cell_lines_val = getattr(drug_row, "cell_lines", []) if drug_row is not None else []
            if not isinstance(cell_lines_val, list):
                cell_lines_val = list(cell_lines_val) if cell_lines_val is not None else []

            # Composition columns propagated from drug_signatures
            # (upstream extract_drug_signatures writes them; drug_meta_df carries
            # them through the join).  All Optional-nullable; missing -> None
            # (archived previous parquets still produce valid NC output).
            def _drug_attr(name):
                if drug_row is None:
                    return None
                v = getattr(drug_row, name, None)
                if v is None:
                    return None
                try:
                    if isinstance(v, float) and np.isnan(v):
                        return None
                except (TypeError, ValueError):
                    pass
                return v

            n_profiles_total_r4 = _drug_attr("n_profiles_total")
            n_profiles_neural_r4 = _drug_attr("n_profiles_neural")
            neural_fraction_r4 = _drug_attr("neural_fraction")
            neural_weight_fraction_r4 = _drug_attr("neural_weight_fraction")
            cell_line_weighting_mode_r4 = _drug_attr("cell_line_weighting_mode")

            meta = _enrich_row_metadata(pert_id, drug_meta_df, agg_metadata)

            results.append({
                "drug_name": drug_name,
                "drug_pubchem_cid": meta["drug_pubchem_cid"],
                "drug_chembl_id": meta["drug_chembl_id"],
                "drug_inchikey": str(drug_inchikey) if drug_inchikey else None,
                "atc_codes": meta["atc_codes"],
                "tissue": tissue,
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
                # One-sided reversal p; always populated.
                "directional_pvalue": directional_p,
                "xsum_score": xsum_score,
                "xsum_pvalue": xsum_pvalue,
                "fdr_global": np.nan,
                # FDR variants populated in
                # _apply_fdr_and_aggregate.  fdr_global stays byte-identical.
                "directional_fdr_global": np.nan,
                "spearman_fdr_per_tissue": np.nan,
                "directional_fdr_per_tissue": np.nan,
                "n_tissues_nominal": 0,
                "n_overlapping_genes": n_overlap,
                "overlap_fraction_disease": n_overlap / n_disease_genes if n_disease_genes > 0 else 0.0,
                "overlap_fraction_drug": n_overlap / n_drug_genes if n_drug_genes > 0 else 0.0,
                "match_confidence": match_conf,
                "lincs_pert_id": pert_id,
                "n_profiles_aggregated": n_profiles,
                "cell_lines": cell_lines_val,
                "clinical_phase": meta["clinical_phase"],
                "mechanism_of_action": meta["mechanism_of_action"],
                "known_targets": meta["known_targets"],
                "known_indications": meta["known_indications"],
                "direction": "reversal" if rho < 0 else "mimicry",
                "top_contributing_genes": top_genes,
                # Cell-line composition propagation.
                "n_profiles_total": n_profiles_total_r4,
                "n_profiles_neural": n_profiles_neural_r4,
                "neural_fraction": neural_fraction_r4,
                "neural_weight_fraction": neural_weight_fraction_r4,
                "cell_line_weighting_mode": cell_line_weighting_mode_r4,
            })

    skip_counts = {"n_drugs_skipped_low_overlap": n_skipped_overlap}
    return results, skip_counts


# ---------------------------------------------------------------------------
# Phase C–D: FDR correction and cross-tissue aggregation
# ---------------------------------------------------------------------------


def _bh_fdr_per_group(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR within a groupby group, robust to size == 1
    and to NaN inputs.

    ``statsmodels.multipletests`` needs at least one input; groups with
    zero valid p-values return an empty Series that ``groupby.transform``
    re-aligns.  Callers should pass a Series of p-values already grouped
    by tissue (or any other stratum).

    NaN handling: ``multipletests``
    propagates NaN into every q-value in the input, so a single invalid
    p-value would spoil the per-tissue FDR column for the entire tissue.
    We mask NaNs, apply BH only to valid rows, and leave invalid rows as
    NaN in the output - matching the way ``_apply_fdr_and_aggregate``
    already handles NaN for the global directional column.

    helper.
    """
    arr = pvals.values
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    if arr.size == 0:
        return out
    valid_mask = ~np.isnan(arr)
    if not valid_mask.any():
        return out
    _, q_valid, _, _ = multipletests(arr[valid_mask], method="fdr_bh")
    out.iloc[np.flatnonzero(valid_mask)] = q_valid
    return out


def _apply_fdr_and_aggregate(
    results_df: pd.DataFrame,
    fdr_threshold: float,
) -> pd.DataFrame:
    """Apply BH-FDR (global two-sided, global one-sided, per-tissue two-sided,
    per-tissue one-sided) and compute cross-tissue nominal counts.

    Populates the following columns in place:

    * ``fdr_global`` - global BH on ``spearman_pvalue`` (byte-identical to the previous behaviour; primary downstream contract per plots, exports,
      reporting, and schema).
    * ``directional_fdr_global`` - global BH on ``directional_pvalue``
.
    * ``spearman_fdr_per_tissue`` - per-tissue BH on ``spearman_pvalue``
.
    * ``directional_fdr_per_tissue`` - per-tissue BH on
      ``directional_pvalue``.
    * ``n_tissues_nominal`` - number of tissues where
      ``spearman_pvalue < 0.05`` for each drug (previous semantics).
    """
    if len(results_df) == 0:
        return results_df

    # Previous primary FDR column, byte-identical.
    pvals = results_df["spearman_pvalue"].values
    _, fdr_q, _, _ = multipletests(pvals, method="fdr_bh")
    results_df["fdr_global"] = fdr_q

    # One-sided global BH. Handled the same way (all rows).
    # NaN directional p (from NaN rho / NaN spearman_pvalue) would break
    # multipletests; guard by masking.
    dir_pvals = results_df["directional_pvalue"].values
    dir_mask = ~np.isnan(dir_pvals)
    if dir_mask.any():
        _, dir_q, _, _ = multipletests(dir_pvals[dir_mask], method="fdr_bh")
        results_df["directional_fdr_global"] = np.nan
        results_df.loc[dir_mask, "directional_fdr_global"] = dir_q
    else:
        results_df["directional_fdr_global"] = np.nan

    # Per-tissue BH (both two-sided and one-sided).
    # Uses groupby.transform to align back to the original row order.
    results_df["spearman_fdr_per_tissue"] = (
        results_df.groupby("tissue")["spearman_pvalue"]
        .transform(_bh_fdr_per_group)
    )
    results_df["directional_fdr_per_tissue"] = (
        results_df.groupby("tissue")["directional_pvalue"]
        .transform(_bh_fdr_per_group)
    )

    nominal_counts = (
        results_df[results_df["spearman_pvalue"] < 0.05]
        .groupby("lincs_pert_id")
        .size()
    )
    results_df["n_tissues_nominal"] = results_df["lincs_pert_id"].map(nominal_counts).fillna(0).astype(int)

    return results_df


def _build_summary(
    results_df: pd.DataFrame,
    fdr_threshold: float,
) -> pd.DataFrame:
    """Build per-drug summary DataFrame from per-tissue results.

    Selects the tissue with the most negative Spearman rho for each drug
    (``idxmin(spearman_rho)``) and reports per-drug counts of significant
    tissues alongside the additive descriptive columns.

    Important semantic note:

    * ``best_spearman_rho`` / ``best_tissue`` - selected-tissue statistic
      (unchanged from previous).
    * ``n_tissues_fdr_significant`` - count of tissues per drug where
      ``fdr_global < fdr_threshold`` (unchanged).
    * ``n_tissues_directional_nominal`` - count of tissues where
      ``directional_pvalue < 0.05`` (additive).
    * ``n_tissues_directional_fdr_significant`` - count of tissues where
      ``directional_fdr_global < fdr_threshold`` (additive).
    * ``best_directional_pvalue`` / ``best_directional_fdr_per_tissue``
      - descriptive selected-tissue statistics (additive).

    ``best_directional_*`` fields are NOT drug-level calibrated
    FDR inference** - they compound cross-tissue selection with a
    per-tissue-calibrated statistic and are appropriate only as
    ranking/descriptive aids.  The principled drug-level combined
    statistic (Cauchy / harmonic-mean p / Fisher) is deferred to a future revision;
    per-drug correction is intentionally not applied here.
    """
    if len(results_df) == 0:
        return pd.DataFrame()

    best_idx = results_df.groupby("lincs_pert_id")["spearman_rho"].idxmin()
    summary = results_df.loc[best_idx].copy()

    fdr_sig_counts = (
        results_df[results_df["fdr_global"] < fdr_threshold]
        .groupby("lincs_pert_id")
        .size()
    )

    summary = summary.rename(columns={
        "spearman_rho": "best_spearman_rho",
        "tissue": "best_tissue",
    })
    summary["n_tissues_fdr_significant"] = summary["lincs_pert_id"].map(fdr_sig_counts).fillna(0).astype(int)

    # additive counts (descriptive; see class docstring).
    dir_nominal_counts = (
        results_df[results_df["directional_pvalue"] < 0.05]
        .groupby("lincs_pert_id")
        .size()
    )
    dir_fdr_sig_counts = (
        results_df[results_df["directional_fdr_global"] < fdr_threshold]
        .groupby("lincs_pert_id")
        .size()
    )
    summary["n_tissues_directional_nominal"] = (
        summary["lincs_pert_id"].map(dir_nominal_counts).fillna(0).astype(int)
    )
    summary["n_tissues_directional_fdr_significant"] = (
        summary["lincs_pert_id"].map(dir_fdr_sig_counts).fillna(0).astype(int)
    )

    # descriptive minima (selection artifact - NOT drug-level
    # inference; see docstring warning).
    best_dir_p = (
        results_df.groupby("lincs_pert_id")["directional_pvalue"].min()
    )
    best_dir_fdr_pt = (
        results_df.groupby("lincs_pert_id")["directional_fdr_per_tissue"].min()
    )
    summary["best_directional_pvalue"] = summary["lincs_pert_id"].map(best_dir_p)
    summary["best_directional_fdr_per_tissue"] = summary["lincs_pert_id"].map(best_dir_fdr_pt)

    keep_cols = [
        "drug_name", "best_spearman_rho", "best_tissue",
        "n_tissues_fdr_significant", "n_tissues_nominal",
        # additive summary columns (position-stable so
        # downstream diffs stay clean).
        "n_tissues_directional_nominal", "n_tissues_directional_fdr_significant",
        "best_directional_pvalue", "best_directional_fdr_per_tissue",
        "drug_pubchem_cid", "drug_chembl_id", "drug_inchikey",
        "atc_codes", "match_confidence", "lincs_pert_id",
        "n_profiles_aggregated", "cell_lines", "clinical_phase",
        "mechanism_of_action", "known_targets", "known_indications",
        # additive composition columns propagated from
        # per_tissue_results via the selected-best-tissue row.
        "n_profiles_total", "n_profiles_neural", "neural_fraction",
        "neural_weight_fraction", "cell_line_weighting_mode",
    ]
    existing = [c for c in keep_cols if c in summary.columns]
    summary = summary[existing].sort_values("best_spearman_rho").reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Permutation calibration diagnostic + atomic sidecar write
# ---------------------------------------------------------------------------


def _lambda_gc(
    obs_pvals: np.ndarray,
    null_pvals: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Estimate genomic-control-style lambda from a p-value distribution.

    Follows the standard Devlin & Roeder (1999) definition using
    :math:`\\chi^2_1` inflation:

    .. math::

        \\lambda_{GC} = \\frac{\\text{median}(\\chi^2_{obs})}{\\text{median}(\\chi^2_{null})}

    where :math:`\\chi^2 = \\text{qchisq}(1 - p, df=1)` and the null
    median is either the theoretical value ``0.4549`` (``chi2.isf(0.5, 1)``)
    or, when *null_pvals* is provided, the empirical median of the
    permutation-derived null.

    p-values are clipped to
    ``[PVAL_CLIP_LOW, PVAL_CLIP_HIGH]`` and NaNs are dropped before
    ``chi2.isf`` to prevent one ``p=0`` from making the median ``inf``.
    This mirrors the existing in-pipeline precedent at
    ``repogen/plotting/qq.py``.

    Also reports a secondary ``neglog10_ratio_heuristic`` (ratio of
    medians of ``-log10(p)``).  This is NOT equivalent to
    :math:`\\lambda_{GC}` under inflation - provided as a diagnostic
    only.

    Args:
        obs_pvals: Observed p-values (any array-like).
        null_pvals: Optional permutation-derived null p-values.  When
            provided, ``lambda_gc_empirical`` and
            ``neglog10_ratio_heuristic`` are computed alongside the
            theoretical estimate; otherwise both are ``None``.

    Returns:
        Dict with keys:
          * ``lambda_gc_theoretical`` (float)
          * ``lambda_gc_empirical`` (float or None)
          * ``neglog10_ratio_heuristic`` (float or None)
          * ``n_pvals_used`` (int)
          * ``n_pvals_dropped_nan`` (int)
    """
    obs_arr = np.asarray(obs_pvals, dtype=float)
    n_input = int(obs_arr.size)
    obs_valid = obs_arr[~np.isnan(obs_arr)]
    n_dropped = int(n_input - obs_valid.size)

    if obs_valid.size == 0:
        return {
            "lambda_gc_theoretical": float("nan"),
            "lambda_gc_empirical": None,
            "neglog10_ratio_heuristic": None,
            "n_pvals_used": 0,
            "n_pvals_dropped_nan": n_dropped,
        }

    obs_clipped = np.clip(obs_valid, PVAL_CLIP_LOW, PVAL_CLIP_HIGH)
    chi2_obs = chi2.isf(obs_clipped, df=1)
    median_obs = float(np.median(chi2_obs))
    lambda_theoretical = median_obs / CHI2_1_MEDIAN

    lambda_empirical: Optional[float] = None
    neglog10_ratio: Optional[float] = None
    if null_pvals is not None:
        null_arr = np.asarray(null_pvals, dtype=float)
        null_valid = null_arr[~np.isnan(null_arr)]
        if null_valid.size > 0:
            null_clipped = np.clip(null_valid, PVAL_CLIP_LOW, PVAL_CLIP_HIGH)
            chi2_null = chi2.isf(null_clipped, df=1)
            med_null = float(np.median(chi2_null))
            if med_null > 0:
                lambda_empirical = median_obs / med_null
            obs_neglog = -np.log10(obs_clipped)
            null_neglog = -np.log10(null_clipped)
            null_neglog_med = float(np.median(null_neglog))
            if null_neglog_med > 0:
                neglog10_ratio = float(np.median(obs_neglog) / null_neglog_med)

    return {
        "lambda_gc_theoretical": float(lambda_theoretical),
        "lambda_gc_empirical": lambda_empirical,
        "neglog10_ratio_heuristic": neglog10_ratio,
        "n_pvals_used": int(obs_valid.size),
        "n_pvals_dropped_nan": n_dropped,
    }


def _run_calibration(
    tissue_dicts: dict[str, dict[int, float]],
    drug_zscore_dicts: dict[str, dict[int, float]],
    obs_results_df: pd.DataFrame,
    n_permutations: int,
    seed: int,
    min_overlap: int,
) -> dict[str, Any]:
    """Run disease-vector permutation calibration and return a JSON payload.

    For each of *n_permutations*, the disease Z-vector is shuffled
    within each tissue's gene set (dependency structure across genes
    is broken; drug vectors and gene-overlap counts are unchanged).
    All drug × tissue Spearman correlations are recomputed under the
    permuted disease vector, producing a pooled null distribution of
    p-values.

    From the observed and null p-value pools this helper reports
    genomic-control lambda estimates (theoretical + empirical) and a
    secondary -log10(p) ratio heuristic - see :func:`_lambda_gc`.

    contract:

    * Purely diagnostic.  Does not modify *obs_results_df*.
    * Raises on any internal error so the caller can abort BEFORE any
      primary parquet is written (clean-failure
      discipline).  All disk I/O is the caller's responsibility.
    * ``n_permutations`` at the default of 100 is a *diagnostic* scale
      - sufficient for a single scalar lambda estimate, NOT for
      per-test empirical p-values.  Do not use the pooled null
      distribution to derive per-test FDR without validating R first.

    Args:
        tissue_dicts: Mapping ``tissue_name -> {entrez_id: disease_z}``.
        drug_zscore_dicts: Mapping ``pert_id -> {entrez_id: drug_z}``.
        obs_results_df: The observed per-tissue results, needed for the
            observed p-value pool by tissue and globally.
        n_permutations: Number of disease-vector shuffles (per tissue).
        seed: RNG seed (deterministic by default).
        min_overlap: Minimum shared-gene count for a Spearman call.
            Matches the primary analysis threshold so the null and
            observed distributions come from comparable tests.

    Returns:
        JSON-ready dict (see the ``calibration.json`` schema in ).
    """
    t0 = time.monotonic()
    rng = np.random.default_rng(seed)

    per_tissue_null: dict[str, list[float]] = {t: [] for t in tissue_dicts}
    for tissue, disease_z in tissue_dicts.items():
        genes = sorted(disease_z.keys())
        if not genes:
            continue
        base_vals = np.array([disease_z[g] for g in genes], dtype=float)
        for _ in range(n_permutations):
            perm_vals = rng.permutation(base_vals)
            perm_disease = dict(zip(genes, perm_vals.tolist()))
            for _pert_id, drug_z in drug_zscore_dicts.items():
                res = compute_spearman(perm_disease, drug_z, min_overlap)
                if res is None:
                    continue
                _rho, pval, _n, _shared = res
                if not np.isnan(pval):
                    per_tissue_null[tissue].append(float(pval))

    global_null = np.concatenate(
        [np.asarray(v, dtype=float) for v in per_tissue_null.values() if v]
    ) if any(per_tissue_null.values()) else np.array([], dtype=float)

    global_obs = obs_results_df["spearman_pvalue"].to_numpy(dtype=float, copy=True)
    global_lambda = _lambda_gc(global_obs, global_null if global_null.size else None)

    per_tissue_payload: dict[str, dict[str, Any]] = {}
    for tissue, null_pvals in per_tissue_null.items():
        obs_mask = obs_results_df["tissue"] == tissue
        obs_p = obs_results_df.loc[obs_mask, "spearman_pvalue"].to_numpy(
            dtype=float, copy=True,
        )
        null_arr = np.asarray(null_pvals, dtype=float) if null_pvals else None
        per_tissue_payload[tissue] = _lambda_gc(obs_p, null_arr)

    elapsed = time.monotonic() - t0

    return {
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "runtime_seconds": float(elapsed),
        "n_tests_global": int(global_obs.size),
        "n_tests_global_null_pool": int(global_null.size),
        "lambda_gc_theoretical": global_lambda["lambda_gc_theoretical"],
        "lambda_gc_empirical": global_lambda["lambda_gc_empirical"],
        "neglog10_ratio_heuristic": global_lambda["neglog10_ratio_heuristic"],
        "n_pvals_used": global_lambda["n_pvals_used"],
        "n_pvals_dropped_nan": global_lambda["n_pvals_dropped_nan"],
        "lambda_gc_per_tissue": per_tissue_payload,
        "notes": (
            "Diagnostic sidecar only. Does not modify "
            "per_tissue_results.parquet primary p-values. "
            "lambda_gc_theoretical uses the theoretical chi2_1 null "
            "median (0.4549). lambda_gc_empirical uses the median of "
            "the permutation-derived null chi2_1 distribution and is "
            "the recommended primary interpretation. "
            "neglog10_ratio_heuristic is a coarser secondary diagnostic "
            "- NOT equivalent to lambda_gc_* under inflation. "
            "n_permutations=100 (default) is a diagnostic scale; NOT "
            "sufficient for per-test empirical p-values."
        ),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* atomically.

    Writes to a temporary file in the destination's parent directory,
    ``fsync``s, then ``os.replace``s to the final path (POSIX + Windows
    atomic).  On any exception the temp file is cleaned up before
    re-raising so a partially-written sidecar can never leak.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=path.parent, suffix=".json.tmp",
        encoding="utf-8",
    )
    tmp_path = Path(tmp.name)
    try:
        json.dump(payload, tmp, indent=2, sort_keys=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp.close()
        except OSError:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


_LIST_COLS = ["atc_codes", "cell_lines", "known_targets", "known_indications", "top_contributing_genes"]


def _save_csv_with_stringified_lists(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV, JSON-encoding list columns for flat-file compatibility."""
    csv_df = df.copy()
    for col in _LIST_COLS:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
    csv_df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_negative_correlation(
    config: PipelineConfig,
    disease_signature_path: Path,
    drug_signatures_path: Path,
    drug_targets_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Orchestrator: run full negative correlation analysis.

    Args:
        config: Validated PipelineConfig
        disease_signature_path: Path to DiseaseSignaturePerTissue Parquet
        drug_signatures_path: Path to DrugSignatureRecord Parquet
        drug_targets_path: Path to DrugTargetRecord Parquet
        output_dir: Directory for output files

    Returns:
        NegativeCorrelationResult DataFrame (per tissue-drug pair)
    """
    t0 = time.monotonic()
    nc_config = config.negative_correlation
    ensure_directory(output_dir)

    # --- Phase A: Load data ---
    logger.info("Loading disease signatures from %s", disease_signature_path)
    disease_sigs = pd.read_parquet(disease_signature_path)

    logger.info("Loading drug signatures from %s", drug_signatures_path)
    drug_sigs = pd.read_parquet(drug_signatures_path)

    logger.info("Loading drug targets from %s", drug_targets_path)
    drug_targets = pd.read_parquet(drug_targets_path)

    lincs_path = nc_config.lincs_gene_info_path
    if lincs_path is None:
        lincs_path = Path("resources") / "drug_signatures" / "lincs_gene_info.tsv"
    if not lincs_path.exists():
        raise FileNotFoundError(
            f"lincs_gene_info.tsv not found at {lincs_path}. "
            "Set negative_correlation.lincs_gene_info_path or run 'repogen setup-resources'."
        )
    allowed_gene_set = load_lincs_gene_info(lincs_path, nc_config.gene_set_mode)

    drug_zscore_dicts, drug_meta_df = filter_drug_signatures(
        drug_sigs, allowed_gene_set,
        nc_config.match_confidence_threshold,
        nc_config.min_profiles_aggregated,
    )

    n_drugs_skipped_profiles = int(
        (drug_sigs["n_profiles_aggregated"] < nc_config.min_profiles_aggregated).sum()
    )
    conf_ranks = drug_sigs["match_confidence"].map(MATCH_CONFIDENCE_RANK)
    n_drugs_skipped_confidence = int(
        (conf_ranks < MATCH_CONFIDENCE_RANK[nc_config.match_confidence_threshold]).sum()
    )

    converter = None
    try:
        from repogen.data.gene_id_converter import GeneIDConverter
        ref = config.reference
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
    except (ImportError, FileNotFoundError):
        logger.debug("GeneIDConverter not available - null Entrez fallback disabled")

    tissue_dicts = prepare_disease_signatures(
        disease_sigs, nc_config.exclude_mhc, converter,
    )

    agg_metadata = aggregate_drug_metadata(drug_targets)

    # --- Phase B: Correlation ---
    logger.info(
        "Computing correlations: %d tissues × %d drugs",
        len(tissue_dicts), len(drug_zscore_dicts),
    )
    results_rows, skip_counts = _run_correlation_loop(
        tissue_dicts, drug_zscore_dicts, drug_meta_df,
        disease_sigs, agg_metadata,
        nc_config.min_overlapping_genes,
        nc_config.xsum_top_n, nc_config.xsum_permutations,
        xsum_seed=nc_config.xsum_seed,
    )

    if len(results_rows) == 0:
        raise ValueError("No valid drug-tissue pairs met quality thresholds.")

    results_df = pd.DataFrame(results_rows)

    # --- Phase C-D: FDR + cross-tissue aggregation ---
    results_df = _apply_fdr_and_aggregate(results_df, nc_config.fdr_threshold)
    summary_df = _build_summary(results_df, nc_config.fdr_threshold)

    # --- Phase E: Optional permutation calibration ---
    # compute the calibration
    # payload BEFORE any parquet is written to disk.  If _run_calibration
    # raises, ``run_negative_correlation`` aborts with no partial disk
    # state - the primary parquets, metadata, and calibration sidecar
    # are all missing rather than the calibration silently skipped.
    calibration_payload: Optional[dict[str, Any]] = None
    if nc_config.permutation_calibration.enabled:
        cal_cfg = nc_config.permutation_calibration
        n_primary_tests = len(results_df)
        # Warm-up-based runtime estimate: use the
        # observed correlation-loop wall-clock so far as a scaling
        # anchor.  No hard-coded percentage claims.
        est_seconds = (time.monotonic() - t0) * cal_cfg.n_permutations
        logger.info(
            "Calibration enabled (R=%d, seed=%d, n_tests=%d). "
            "Rough runtime estimate: %.0f seconds "
            "(anchor: correlation-loop time so far × R). "
            "Actual cost is data-dependent; benchmark on your study.",
            cal_cfg.n_permutations, cal_cfg.seed, n_primary_tests, est_seconds,
        )
        calibration_payload = _run_calibration(
            tissue_dicts, drug_zscore_dicts, results_df,
            n_permutations=cal_cfg.n_permutations,
            seed=cal_cfg.seed,
            min_overlap=nc_config.min_overlapping_genes,
        )

    # --- Phase F: MHC sensitivity ---
    sensitivity_results = None
    sensitivity_summary = None
    if not nc_config.exclude_mhc:
        logger.info("Running MHC-excluded sensitivity analysis")
        tissue_dicts_no_mhc = prepare_disease_signatures(
            disease_sigs, exclude_mhc=True, gene_id_converter=converter,
        )
        sens_rows, _ = _run_correlation_loop(
            tissue_dicts_no_mhc, drug_zscore_dicts, drug_meta_df,
            disease_sigs, agg_metadata,
            nc_config.min_overlapping_genes,
            nc_config.xsum_top_n, nc_config.xsum_permutations,
            xsum_seed=nc_config.xsum_seed,
        )
        if len(sens_rows) > 0:
            sensitivity_results = pd.DataFrame(sens_rows)
            sensitivity_results = _apply_fdr_and_aggregate(sensitivity_results, nc_config.fdr_threshold)
            sensitivity_summary = _build_summary(sensitivity_results, nc_config.fdr_threshold)
    else:
        logger.info("MHC sensitivity skipped - primary analysis already excludes MHC")

    # --- Phase G: Validate and save ---
    validate_dataframe(results_df, "NegativeCorrelationResult")
    validate_dataframe(summary_df, "NegativeCorrelationSummary")

    # Track every path written by
    # THIS invocation so we can roll back cleanly if the calibration
    # sidecar write fails after primary parquets are already on disk.
    # Only files created in this run are tracked - pre-existing archived
    # outputs from prior runs are never touched.
    written_paths_this_run: list[Path] = []

    def _write(path: Path, writer) -> None:
        writer(path)
        written_paths_this_run.append(path)

    _write(output_dir / "per_tissue_results.parquet",
           lambda p: results_df.to_parquet(p, engine="pyarrow", index=False))
    _write(output_dir / "drug_summary.parquet",
           lambda p: summary_df.to_parquet(p, engine="pyarrow", index=False))
    _write(output_dir / "per_tissue_results.csv",
           lambda p: _save_csv_with_stringified_lists(results_df, p))
    _write(output_dir / "drug_summary.csv",
           lambda p: _save_csv_with_stringified_lists(summary_df, p))

    if sensitivity_results is not None:
        sens_dir = output_dir / "sensitivity" / "mhc_excluded"
        ensure_directory(sens_dir)
        _write(
            sens_dir / "per_tissue_results.parquet",
            lambda p: sensitivity_results.to_parquet(p, engine="pyarrow", index=False),
        )
        if sensitivity_summary is not None:
            _write(
                sens_dir / "drug_summary.parquet",
                lambda p: sensitivity_summary.to_parquet(p, engine="pyarrow", index=False),
            )

    n_sig = int((results_df["fdr_global"] < nc_config.fdr_threshold).sum())
    n_total_tests = len(results_df)
    median_overlap = int(results_df["n_overlapping_genes"].median()) if len(results_df) > 0 else 0

    metadata = {
        "module": "negative_correlation",
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "correlation_method": nc_config.correlation_method,
            "gene_set_mode": nc_config.gene_set_mode,
            "min_overlapping_genes": nc_config.min_overlapping_genes,
            "xsum_top_n": nc_config.xsum_top_n,
            "fdr_threshold": nc_config.fdr_threshold,
            "exclude_mhc": nc_config.exclude_mhc,
            "match_confidence_threshold": nc_config.match_confidence_threshold,
            "aggregation_mode": nc_config.aggregation_mode,
            "min_profiles_aggregated": nc_config.min_profiles_aggregated,
        },
        "summary": {
            "n_drugs_tested": len(drug_zscore_dicts),
            "n_tissues": len(tissue_dicts),
            "n_total_tests": n_total_tests,
            "n_significant_fdr": n_sig,
            "n_drugs_skipped_low_overlap": skip_counts["n_drugs_skipped_low_overlap"],
            "n_drugs_skipped_low_profiles": n_drugs_skipped_profiles,
            "n_drugs_skipped_match_confidence": n_drugs_skipped_confidence,
            "median_gene_overlap": median_overlap,
        },
    }

    if not nc_config.exclude_mhc and sensitivity_results is not None:
        n_sig_sens = int((sensitivity_results["fdr_global"] < nc_config.fdr_threshold).sum())
        metadata["sensitivity"] = {
            "mhc_excluded": {"n_significant_fdr": n_sig_sens},
        }

    # Extend the existing metadata.json
    # with an ``r3_provenance`` subkey rather than creating a second
    # metadata file.  Keeps a single declared Snakemake metadata output
    # and mirrors the ``sensitivity`` subkey pattern.
    metadata["r3_provenance"] = {
        "xsum_permutations": int(nc_config.xsum_permutations),
        "xsum_seed": int(nc_config.xsum_seed),
        "permutation_calibration": {
            "enabled": bool(nc_config.permutation_calibration.enabled),
            "n_permutations": int(nc_config.permutation_calibration.n_permutations),
            "seed": int(nc_config.permutation_calibration.seed),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "notes": (
            "xsum_seed=42 preserves previous byte-for-byte "
            "behaviour (compute_xsum previously hard-coded default_rng(42)). "
            "permutation_calibration.enabled=False by default; when true "
            "produces negative_correlation/calibration.json diagnostic sidecar "
            "that does NOT modify primary p-values in per_tissue_results.parquet."
        ),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    written_paths_this_run.append(metadata_path)

    # Write calibration
    # sidecar atomically AFTER primary parquets + metadata.  The calibration
    # payload was computed in memory before any disk state was written
    # (Phase E) so *compute* failure aborts the run without leaving partial
    # outputs.  Round-4 additionally closes the *write*-failure hole: if
    # ``_atomic_write_json`` raises here, we roll back every file this
    # invocation wrote before re-raising, so the user never sees a
    # "complete-looking" primary output for a run whose requested
    # calibration sidecar could not be produced.
    if calibration_payload is not None:
        try:
            _atomic_write_json(output_dir / "calibration.json", calibration_payload)
        except Exception:
            for p in written_paths_this_run:
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    logger.warning(
                        "Rollback failed to unlink %s after calibration write failure; "
                        "leaving partial state on disk.", p,
                    )
            raise
        logger.info(
            "Calibration sidecar written: %s (lambda_gc_theoretical=%.3f, "
            "lambda_gc_empirical=%s)",
            output_dir / "calibration.json",
            calibration_payload["lambda_gc_theoretical"],
            (
                f"{calibration_payload['lambda_gc_empirical']:.3f}"
                if calibration_payload["lambda_gc_empirical"] is not None
                else "None"
            ),
        )

    elapsed = time.monotonic() - t0
    logger.info(
        "Negative correlation analysis completed in %.1fs. "
        "%d tests, %d significant at FDR < %.2f.",
        elapsed, n_total_tests, n_sig, nc_config.fdr_threshold,
    )

    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Negative correlation analysis")
    parser.add_argument("--disease-signature", type=Path, required=True,
                        help="Path to DiseaseSignaturePerTissue Parquet")
    parser.add_argument("--drug-signatures", type=Path, required=True,
                        help="Path to DrugSignatureRecord Parquet")
    parser.add_argument("--drug-targets", type=Path, required=True,
                        help="Path to DrugTargetRecord Parquet")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to pipeline config YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    from repogen.config.loader import load_config

    pipeline_config = load_config(args.config)

    run_negative_correlation(
        config=pipeline_config,
        disease_signature_path=args.disease_signature,
        drug_signatures_path=args.drug_signatures,
        drug_targets_path=args.drug_targets,
        output_dir=args.output_dir,
    )
