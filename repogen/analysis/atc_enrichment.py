"""ATC (Anatomical Therapeutic Chemical) class enrichment analysis.

Primary test: GLS (Generalised Least Squares) regression with a drug-drug
covariance matrix derived from MAGMA's LD-aware gene-gene correlations -
a Python reimplementation of DRUGSETS (Bell et al. 2022).

Secondary test: Wilcoxon rank-sum as non-parametric concordance check.

Optional: drug-label permutation test.

References:
    Bell et al. 2022, DRUGSETS (medRxiv 10.1101/2022.09.06.22279660)
    Gaspar & Breen 2017, Scientific Reports 7:12460
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from repogen.config.schema import ATCEnrichmentConfig, CustomATCClass
from repogen.utils.constants import ATC_DESCRIPTIONS
from repogen.utils.io import ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

ATC_LEVEL_PREFIX = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}


# ---------------------------------------------------------------------------
# ATC-code helpers
# ---------------------------------------------------------------------------


def _deserialize_atc_codes(x) -> list:
    """Normalize an atc_codes field value to a canonical list of strings.

    Handles: Python list/tuple/ndarray (passthrough), JSON array strings,
    plain ATC code strings, JSON scalar strings, None/NaN/malformed.
    Always returns a list (possibly empty). Never raises.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float):
        return []
    if isinstance(x, str):
        x_stripped = x.strip()
        if not x_stripped:
            return []
        try:
            parsed = json.loads(x_stripped)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, str):
                return [parsed]
            return []
        except (json.JSONDecodeError, ValueError):
            return [x_stripped]
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    return []


def _has_atc_codes(x) -> bool:
    """Check whether a deserialized atc_codes list contains at least one valid ATC code.

    Expects canonical list input (from _deserialize_atc_codes), but handles
    all types defensively for safety.
    """
    if x is None:
        return False
    if isinstance(x, float):
        return False
    if isinstance(x, str):
        return False
    try:
        if pd.isna(x):
            return False
    except (TypeError, ValueError):
        pass
    if hasattr(x, "__iter__"):
        return any(isinstance(e, str) and len(e.strip()) > 0 for e in x)
    return False


# ---------------------------------------------------------------------------
# .genes.raw parser
# ---------------------------------------------------------------------------


def _detect_raw_field_layout(
    first_data_parts: list[str],
    covar_names: list[str],
) -> tuple[int, int, int, int | None, int | None]:
    """Determine field indices for a ``.genes.raw`` data line.

    MAGMA ``.genes.raw`` field order varies by version:

    * Legacy (no ``# COVAR`` header, 7 fields):
      ``GENE CHR NSNPS NPARAM N MAC ZSTAT``
    * v1.10+ with ``# COVAR = NSAMP MAC`` (9 fields):
      ``GENE CHR START STOP NSNPS NPARAM NSAMP MAC ZSTAT``

    Returns:
        ``(nsnps_idx, nparam_idx, zstat_idx, nsamp_idx, mac_idx)``
        where ``nsamp_idx`` / ``mac_idx`` may be ``None`` if absent.
    """
    info_fields = len(first_data_parts)
    n_covars = len(covar_names)

    if covar_names:
        # Modern format: GENE CHR [START STOP] NSNPS NPARAM [COVARS...] ZSTAT
        base_no_startop = 2 + 2 + n_covars + 1  # GENE+CHR + NSNPS+NPARAM + covars + ZSTAT
        has_start_stop = info_fields > base_no_startop
        offset = 4 if has_start_stop else 2
        nsnps_idx = offset
        nparam_idx = offset + 1
        zstat_idx = info_fields - 1
        covar_start = offset + 2
        covar_map = {name.upper(): covar_start + ci for ci, name in enumerate(covar_names)}
        nsamp_idx = covar_map.get("NSAMP")
        mac_idx = covar_map.get("MAC")
    else:
        # Legacy: GENE CHR NSNPS NPARAM N MAC ZSTAT
        nsnps_idx = 2
        nparam_idx = 3
        nsamp_idx = 4 if info_fields >= 7 else None
        mac_idx = 5 if info_fields >= 7 else None
        zstat_idx = info_fields - 1

    return nsnps_idx, nparam_idx, zstat_idx, nsamp_idx, mac_idx


def parse_genes_raw(
    genes_raw_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Parse MAGMA ``.genes.raw`` file for gene info and per-chromosome LD correlations.

    Auto-detects the field layout from the ``# COVAR`` header and the first
    data line (which has zero correlation entries), so both legacy 7-field
    and MAGMA v1.10+ 9-field formats are supported.

    Args:
        genes_raw_path: Path to MAGMA ``.genes.raw`` file.

    Returns:
        Tuple of (gene_info DataFrame, per-chromosome correlation matrices).
        ``gene_info`` has columns: gene, chr, nsnps, nparam, nsamp, mac, zstat.
        ``chr_corr_matrices`` maps chromosome name (str) to a symmetric
        correlation matrix with 1.0 on the diagonal.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no gene records.
    """
    if not genes_raw_path.is_file():
        raise FileNotFoundError(f"MAGMA .genes.raw file not found: {genes_raw_path}")

    comments: list[str] = []
    data_lines: list[str] = []
    with open(genes_raw_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line.strip())
            elif line.strip():
                data_lines.append(line.strip())

    if not data_lines:
        raise ValueError(f"MAGMA .genes.raw file contains no gene records: {genes_raw_path}")

    covar_names: list[str] = []
    for c in comments:
        if "COVAR" in c and "=" in c:
            covar_names = c.split("=", 1)[1].strip().split()
            break

    first_parts = data_lines[0].split()
    info_fields = len(first_parts)
    nsnps_idx, nparam_idx, zstat_idx, nsamp_idx, mac_idx = _detect_raw_field_layout(
        first_parts, covar_names,
    )

    logger.info(
        "Detected genes.raw layout: %d info fields, covariates=%s",
        info_fields,
        ", ".join(covar_names) if covar_names else "none (legacy)",
    )

    gene_records: list[dict] = []
    raw_corrs: list[list[float]] = []

    for line_str in data_lines:
        parts = line_str.split()
        if len(parts) < info_fields:
            continue
        info = parts[:info_fields]
        corr_entries = [float(x) for x in parts[info_fields:]]

        gene_records.append({
            "gene": int(info[0]),
            "chr": info[1],
            "nsnps": int(info[nsnps_idx]),
            "nparam": int(info[nparam_idx]),
            "nsamp": int(float(info[nsamp_idx])) if nsamp_idx is not None else 0,
            "mac": float(info[mac_idx]) if mac_idx is not None else 0.0,
            "zstat": float(info[zstat_idx]),
        })
        raw_corrs.append(corr_entries)

    if not gene_records:
        raise ValueError(f"No valid gene records parsed from: {genes_raw_path}")

    gene_info = pd.DataFrame(gene_records)

    chr_corr_matrices: dict[str, np.ndarray] = {}
    for chrom in gene_info["chr"].unique():
        mask = gene_info["chr"] == chrom
        chr_indices = np.where(mask.values)[0]
        n = len(chr_indices)
        corr = np.zeros((n, n))
        for local_i, global_i in enumerate(chr_indices):
            entries = raw_corrs[global_i]
            if entries:
                n_entries = len(entries)
                start = local_i - n_entries
                corr[local_i, start:local_i] = entries
        corr = corr + corr.T
        np.fill_diagonal(corr, 1.0)
        chr_corr_matrices[chrom] = corr

    logger.info(
        "Parsed .genes.raw: %d genes across %d chromosomes",
        len(gene_info),
        len(chr_corr_matrices),
    )
    return gene_info, chr_corr_matrices


# ---------------------------------------------------------------------------
# Chromosome projections
# ---------------------------------------------------------------------------


def build_chromosome_projections(
    gene_info: pd.DataFrame,
    chr_corr_matrices: dict[str, np.ndarray],
    eigenvalue_threshold: float = 0.1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build whitening projection matrices per chromosome via eigendecomposition.

    For each chromosome's gene-gene LD correlation matrix, retains eigenvalues
    above ``eigenvalue_threshold`` and constructs a whitening projection:
    ``V_kept @ diag(1 / sqrt(lambda_kept))``.

    Args:
        gene_info: Gene info DataFrame from :func:`parse_genes_raw`.
        chr_corr_matrices: Per-chromosome correlation matrices.
        eigenvalue_threshold: Minimum eigenvalue to retain (default 0.1).

    Returns:
        Tuple of (projections dict, proj_indices dict).
        ``projections[chrom]`` has shape ``(n_genes_chr, n_components_retained)``.
        ``proj_indices[chrom]`` is a 1D int array of column indices in the
        concatenated projected space.
    """
    projections: dict[str, np.ndarray] = {}
    proj_indices: dict[str, np.ndarray] = {}
    offset = 0

    for chrom in gene_info["chr"].unique():
        corr = chr_corr_matrices[chrom]
        corr = (corr + corr.T) / 2.0

        eigvals, eigvecs = np.linalg.eigh(corr)

        keep = eigvals >= eigenvalue_threshold
        if not keep.any():
            logger.warning("Chromosome %s: no eigenvalues above %.3f", chrom, eigenvalue_threshold)
            keep[np.argmax(eigvals)] = True

        kept_vals = eigvals[keep]
        kept_vecs = eigvecs[:, keep]

        projection = kept_vecs @ np.diag(1.0 / np.sqrt(kept_vals))
        projections[chrom] = projection

        n_components = kept_vals.shape[0]
        proj_indices[chrom] = np.arange(offset, offset + n_components)
        offset += n_components

    logger.info("Total projected dimensions: %d (from %d genes)", offset, len(gene_info))
    return projections, proj_indices


# ---------------------------------------------------------------------------
# Drug-drug set correlations (Stage A)
# ---------------------------------------------------------------------------


def compute_drug_set_correlations(
    gene_info: pd.DataFrame,
    projections: dict[str, np.ndarray],
    proj_indices: dict[str, np.ndarray],
    drug_results: pd.DataFrame,
    drug_gene_sets: dict[str, list[int]],
    eigenvalue_threshold: float = 0.1,
    condition_sets: dict[str, list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute DRUGSETS-style drug-drug set correlation matrix from MAGMA .genes.raw.

    Replicates DRUGSETS ``compute_corrs.r``:

    1. Project gene-level data into whitened space per chromosome.
    2. Build and project residualization covariates (nsnps, nparam, nsamp, 1/mac + logs).
    3. Project binary drug membership vectors into same space.
    4. Partial out residualization covariates (and optional conditioning sets).
    5. Compute set correlations from the partialled membership vectors.
    6. Regularize and invert the correlation matrix via eigenvalue clipping.

    Args:
        gene_info: Gene info from :func:`parse_genes_raw`.
        projections: From :func:`build_chromosome_projections`.
        proj_indices: From :func:`build_chromosome_projections`.
        drug_results: Drug enrichment results DataFrame (must contain
            ``drug_chembl_id``, ``magma_z``, ``n_target_genes``).
        drug_gene_sets: Dict mapping drug_chembl_id to lists of Entrez gene IDs.
        eigenvalue_threshold: For regularizing the drug-drug correlation inverse.
        condition_sets: Forward-compatible hook for v2.0 druggable-genome
            conditioning (unused in v1.0).

    Returns:
        Tuple of (drug_z_ordered, set_corrs, set_corrs_inv,
        correlation_matrix_condition_number).
        ``drug_z_ordered``: shape ``(n_drugs,)``.
        ``set_corrs``: shape ``(n_drugs, n_drugs)``.
        ``set_corrs_inv``: regularized pseudo-inverse, shape ``(n_drugs, n_drugs)``.
        ``correlation_matrix_condition_number``: condition number of the
        regularized (eigenvalue-clipped) drug-drug set correlation matrix.

    Raises:
        ValueError: If the precision matrix is not positive semi-definite.
    """
    total_proj_dim = max(idx.max() for idx in proj_indices.values()) + 1
    chromosomes = gene_info["chr"].unique()

    def project_data(data_matrix: np.ndarray) -> np.ndarray:
        """Project gene-level data (n_genes x k) into whitened space."""
        out = np.full((total_proj_dim, data_matrix.shape[1]), 0.0)
        for chrom in chromosomes:
            chr_mask = (gene_info["chr"] == chrom).values
            proj = projections[chrom]
            idx = proj_indices[chrom]
            out[idx, :] = proj.T @ data_matrix[chr_mask, :]
        return out

    # Residualization covariates
    nsnps = gene_info["nsnps"].values.astype(float)
    nparam = gene_info["nparam"].values.astype(float)
    nsamp = gene_info["nsamp"].values.astype(float)
    mac_vals = gene_info["mac"].values.astype(float)
    mac_vals = np.where(mac_vals == 0, 1e-10, mac_vals)
    inv_mac = 1.0 / mac_vals

    residualize = np.column_stack([nsnps, nparam, nsamp, inv_mac])
    col_var = residualize.var(axis=0)
    residualize = residualize[:, col_var > 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_resid = np.log(np.abs(residualize))
        log_resid = np.nan_to_num(log_resid, nan=0.0, posinf=0.0, neginf=0.0)
    residualize = np.column_stack([residualize, log_resid])

    resid_mean = residualize.mean(axis=0)
    resid_std = residualize.std(axis=0)
    resid_std[resid_std == 0] = 1.0
    residualize = (residualize - resid_mean) / resid_std
    residualize = np.column_stack([np.ones(len(gene_info)), residualize])

    residualize_proj = project_data(residualize)

    # Drug membership vectors
    gene_id_to_idx = {int(g): i for i, g in enumerate(gene_info["gene"].values)}
    drug_ids_ordered = drug_results["drug_chembl_id"].tolist()
    n_drugs = len(drug_ids_ordered)

    membership = np.zeros((len(gene_info), n_drugs))
    for j, drug_id in enumerate(drug_ids_ordered):
        gene_ids = drug_gene_sets.get(drug_id, [])
        for gid in gene_ids:
            idx = gene_id_to_idx.get(gid)
            if idx is not None:
                membership[idx, j] = 1.0

    sets_proj = project_data(membership)

    # Covariate matrix for partialling
    covar = residualize_proj[:, 0:1]

    if condition_sets is not None:
        for set_name, set_genes in condition_sets.items():
            cond_vec = np.zeros((len(gene_info), 1))
            for gid in set_genes:
                idx = gene_id_to_idx.get(gid)
                if idx is not None:
                    cond_vec[idx, 0] = 1.0
            cond_proj = project_data(cond_vec)
            covar = np.column_stack([covar, cond_proj])
            logger.info("Conditioning on set '%s' (%d genes)", set_name, len(set_genes))

    # Partial out covariates from projected drug sets
    ctc = covar.T @ covar
    ctc_reg = ctc + np.eye(ctc.shape[0]) * 1e-10
    ctc_inv = np.linalg.solve(ctc_reg, np.eye(covar.shape[1]))
    cts = covar.T @ sets_proj
    det = np.sum(sets_proj ** 2, axis=0) - np.sum(cts * (ctc_inv @ cts), axis=0)
    det = np.maximum(det, 1e-10)
    V = (sets_proj - covar @ ctc_inv @ cts) / det[np.newaxis, :]

    # Drug-drug set correlations
    raw_cov = V.T @ V
    diag_sqrt = np.sqrt(np.diag(raw_cov))
    diag_sqrt[diag_sqrt == 0] = 1.0
    set_corrs = raw_cov / np.outer(diag_sqrt, diag_sqrt)
    set_corrs = (set_corrs + set_corrs.T) / 2.0
    np.fill_diagonal(set_corrs, 1.0)

    # Regularized inverse via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(set_corrs)
    keep = eigvals >= eigenvalue_threshold
    if not keep.any():
        logger.warning("No eigenvalues above threshold in drug correlation matrix; keeping largest")
        keep[np.argmax(eigvals)] = True

    kept_vals = eigvals[keep]
    kept_vecs = eigvecs[:, keep]
    set_corrs_inv = kept_vecs @ np.diag(1.0 / kept_vals) @ kept_vecs.T

    min_eigval = np.min(np.linalg.eigvalsh(set_corrs_inv))
    if min_eigval < -1e-8:
        raise ValueError(f"Precision matrix not PSD: min eigenvalue = {min_eigval}")

    correlation_matrix_condition_number = float(kept_vals[-1] / kept_vals[0]) if kept_vals[0] > 0 else float("inf")
    logger.info(
        "Drug correlation matrix: %d drugs, correlation_matrix_condition_number = %.1f",
        n_drugs,
        correlation_matrix_condition_number,
    )
    if correlation_matrix_condition_number > 1e6:
        logger.warning(
            "High condition number (%.1f) - results may be numerically unstable",
            correlation_matrix_condition_number,
        )

    drug_z_ordered = drug_results.set_index("drug_chembl_id").loc[drug_ids_ordered, "magma_z"].values.astype(float)

    return drug_z_ordered, set_corrs, set_corrs_inv, correlation_matrix_condition_number


# ---------------------------------------------------------------------------
# ATC class extraction
# ---------------------------------------------------------------------------


def extract_atc_classes(
    drug_results: pd.DataFrame,
    atc_levels: list[int],
    min_drugs_per_class: int,
) -> dict[str, list[str]]:
    """Extract ATC class memberships by string-slicing Level 5 ATC codes.

    A drug with multiple ATC codes contributes to all corresponding classes
    at each level.  A drug is counted once per class even if it has multiple
    Level 5 codes mapping to the same higher-level prefix.

    Args:
        drug_results: Drug enrichment results DataFrame (must contain
            ``drug_chembl_id`` and ``atc_codes`` columns).
        atc_levels: List of ATC hierarchy levels to extract (1-4).
        min_drugs_per_class: Minimum drug count for a class to be included.

    Returns:
        Dict mapping ATC code (at tested level) to list of drug_chembl_id
        values in that class.
    """
    atc_classes: dict[str, list[str]] = defaultdict(list)

    for level in atc_levels:
        prefix_len = ATC_LEVEL_PREFIX[level]
        exploded = drug_results[["drug_chembl_id", "atc_codes"]].explode("atc_codes")
        exploded = exploded.dropna(subset=["atc_codes"])
        exploded = exploded[exploded["atc_codes"].str.len() >= prefix_len]
        exploded["atc_class"] = exploded["atc_codes"].str[:prefix_len]

        for atc_class, group in exploded.groupby("atc_class"):
            unique_drugs = group["drug_chembl_id"].unique().tolist()
            if len(unique_drugs) >= min_drugs_per_class:
                if atc_class not in atc_classes:
                    atc_classes[atc_class] = unique_drugs

    logger.info(
        "Extracted %d ATC classes across levels %s (min %d drugs/class)",
        len(atc_classes),
        atc_levels,
        min_drugs_per_class,
    )
    return dict(atc_classes)


def build_custom_atc_classes(
    drug_results: pd.DataFrame,
    custom_classes: list[CustomATCClass],
    min_drugs_per_class: int,
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Resolve curated drug-sets (by ATC-code membership) over the current universe.

    A drug joins a curated class if any of its ``atc_codes`` matches one of the
    class's ``atc_members``.  The same ``min_drugs_per_class`` floor as for
    standard classes is applied; classes below the floor are skipped (logged),
    mirroring standard small-class behaviour.

    Args:
        drug_results: Drug enrichment results DataFrame with ``drug_chembl_id``
            and (already-deserialised) list-valued ``atc_codes`` columns.
        custom_classes: List of CustomATCClass definitions.
        min_drugs_per_class: Minimum drug count for a class to be tested.

    Returns:
        Tuple of:
            - dict mapping custom class code to list of ``drug_chembl_id``.
            - dict mapping custom class code to ``{'level': int, 'description': str}``
              metadata (used to populate the result instead of inferring from code).
    """
    classes: dict[str, list[str]] = {}
    meta: dict[str, dict] = {}
    if not custom_classes or len(drug_results) == 0:
        return classes, meta

    exploded = drug_results[["drug_chembl_id", "atc_codes"]].explode("atc_codes")
    exploded = exploded.dropna(subset=["atc_codes"])

    for cc in custom_classes:
        member_set = set(cc.atc_members)
        hit = exploded[exploded["atc_codes"].isin(member_set)]
        unique_drugs = hit["drug_chembl_id"].unique().tolist()
        if len(unique_drugs) < min_drugs_per_class:
            logger.warning(
                "Custom ATC class '%s' has %d members (< min_drugs_per_class=%d) - skipped",
                cc.code, len(unique_drugs), min_drugs_per_class,
            )
            continue
        classes[cc.code] = unique_drugs
        meta[cc.code] = {"level": int(cc.level), "description": cc.description}
        logger.info("Custom ATC class '%s': %d members", cc.code, len(unique_drugs))

    return classes, meta


# ---------------------------------------------------------------------------
# GLS regression
# ---------------------------------------------------------------------------


def run_gls_regression(
    y: np.ndarray,
    atc_indicator: np.ndarray,
    n_genes: np.ndarray,
    set_corrs_inv: np.ndarray,
    two_sided: bool = False,
) -> dict[str, float]:
    """GLS regression of drug Z-scores on ATC indicator with size covariates.

    Design matrix: ``[intercept, atc_indicator, n_genes, log(n_genes)]``.
    Uses the precision matrix (Sigma^-1) directly, matching DRUGSETS ``lnreg_dep``.

    Args:
        y: Drug MAGMA Z-statistics, shape ``(N,)``.
        atc_indicator: Binary vector (1 = drug in class), shape ``(N,)``.
        n_genes: Gene set sizes per drug, shape ``(N,)``.
        set_corrs_inv: Precision matrix, shape ``(N, N)``.
        two_sided: If True, compute two-sided p-value.

    Returns:
        Dict with keys: ``beta``, ``se``, ``t_stat``, ``p_value``.

    References:
        Bell et al. 2022, DRUGSETS (medRxiv 10.1101/2022.09.06.22279660)
    """
    N = len(y)
    n_genes_f = n_genes.astype(float)
    n_genes_f = np.maximum(n_genes_f, 1.0)
    log_n_genes = np.log(n_genes_f)
    X = np.column_stack([np.ones(N), atc_indicator, n_genes_f, log_n_genes])
    K = X.shape[1]

    Xt = X.T
    XtSiX = Xt @ set_corrs_inv @ X
    XtSiX_reg = XtSiX + np.eye(K) * 1e-12
    W = np.linalg.solve(XtSiX_reg, np.eye(K))
    B = W @ Xt @ set_corrs_inv @ y
    residuals = y - X @ B
    denom = max(N - K, 1)
    sigma_sq = (residuals @ set_corrs_inv @ residuals) / denom

    beta = float(B[1])
    se = float(np.sqrt(max(sigma_sq * W[1, 1], 0.0)))
    t_stat = beta / se if se > 0 else 0.0

    if two_sided:
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(N - K, 1)))
    else:
        p_value = float(stats.t.sf(t_stat, df=max(N - K, 1)))

    return {"beta": beta, "se": se, "t_stat": t_stat, "p_value": p_value}


# ---------------------------------------------------------------------------
# Wilcoxon rank-sum
# ---------------------------------------------------------------------------


def run_wilcoxon_test(
    y: np.ndarray,
    atc_indicator: np.ndarray,
    two_sided: bool = False,
) -> dict[str, float]:
    """Wilcoxon rank-sum (Mann-Whitney U) for drugs inside vs outside an ATC class.

    Non-parametric secondary test.  Does NOT account for inter-drug
    correlation - p-values may be anti-conservative for classes with
    many overlapping drug targets.

    Args:
        y: Drug MAGMA Z-statistics, shape ``(N,)``.
        atc_indicator: Binary vector (1 = drug in class), shape ``(N,)``.
        two_sided: If True, use ``alternative='two-sided'``.

    Returns:
        Dict with keys: ``u_stat``, ``p_value``.
    """
    inside = y[atc_indicator == 1]
    outside = y[atc_indicator == 0]

    if len(inside) < 2 or len(outside) < 2:
        return {"u_stat": np.nan, "p_value": np.nan}

    alternative = "two-sided" if two_sided else "greater"
    u_stat, p_value = stats.mannwhitneyu(inside, outside, alternative=alternative)

    return {"u_stat": float(u_stat), "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def run_permutation_test(
    y: np.ndarray,
    atc_classes: dict[str, list[str]],
    drug_id_to_idx: dict[str, int],
    n_genes: np.ndarray,
    set_corrs_inv: np.ndarray,
    n_permutations: int,
    seed: int | None,
    two_sided: bool,
) -> dict[str, float]:
    """Drug-label permutation test for all ATC classes.

    For each class, shuffles the ATC indicator (preserving class size)
    and re-runs :func:`run_gls_regression` to build a null distribution
    of t-statistics.

    Args:
        y: Drug MAGMA Z-statistics, shape ``(N,)``.
        atc_classes: Dict mapping ATC code to list of drug_chembl_id.
        drug_id_to_idx: Dict mapping drug_chembl_id to index in ``y``.
        n_genes: Gene set sizes per drug, shape ``(N,)``.
        set_corrs_inv: Precision matrix, shape ``(N, N)``.
        n_permutations: Number of permutations.
        seed: Random seed (None for non-deterministic).
        two_sided: Whether to use absolute t-stat for comparison.

    Returns:
        Dict mapping ATC code to empirical permutation p-value.
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    perm_results: dict[str, float] = {}

    for atc_code, drug_ids in atc_classes.items():
        class_indices = np.array([drug_id_to_idx[d] for d in drug_ids if d in drug_id_to_idx])
        k = len(class_indices)
        if k < 2:
            perm_results[atc_code] = np.nan
            continue

        real_indicator = np.zeros(N)
        real_indicator[class_indices] = 1.0
        obs = run_gls_regression(y, real_indicator, n_genes, set_corrs_inv, two_sided)
        t_obs = obs["t_stat"]

        count = 0
        for _ in range(n_permutations):
            perm_idx = rng.choice(N, size=k, replace=False)
            perm_indicator = np.zeros(N)
            perm_indicator[perm_idx] = 1.0
            # n_genes is a per-drug property (annotation density) and stays fixed;
            # only the class indicator is permuted.  This matches DRUGSETS' design
            # where size covariates control for each drug's target count regardless
            # of class assignment.
            perm_res = run_gls_regression(y, perm_indicator, n_genes, set_corrs_inv, two_sided)
            t_perm = perm_res["t_stat"]

            if two_sided:
                if abs(t_perm) >= abs(t_obs):
                    count += 1
            else:
                if t_perm >= t_obs:
                    count += 1

        perm_results[atc_code] = (count + 1) / (n_permutations + 1)

    return perm_results


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def assemble_atc_results(
    class_results: list[dict],
    fdr_method: str,
    fdr_threshold: float,
    perm_results: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Assemble per-class results into the ATCEnrichmentResult DataFrame.

    Applies BH-FDR correction globally across all tested classes for both
    the GLS and Wilcoxon p-values.

    Args:
        class_results: List of dicts, one per tested ATC class.
        fdr_method: statsmodels multipletests method name.
        fdr_threshold: For reporting, not filtering.
        perm_results: Optional dict mapping ATC code to permutation p-value.

    Returns:
        ATCEnrichmentResult DataFrame sorted by ``gls_p``.
    """
    if not class_results:
        logger.warning("No ATC classes tested - returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(class_results)

    # GLS FDR
    gls_p = df["gls_p"].values
    _, gls_fdr, _, _ = multipletests(gls_p, method=fdr_method)
    df["gls_fdr"] = gls_fdr

    # Wilcoxon FDR
    wilcoxon_p = df["wilcoxon_p"].values
    valid_wilcoxon = ~np.isnan(wilcoxon_p)
    wilcoxon_fdr = np.full(len(df), np.nan)
    if valid_wilcoxon.any():
        _, wfdr, _, _ = multipletests(wilcoxon_p[valid_wilcoxon], method=fdr_method)
        wilcoxon_fdr[valid_wilcoxon] = wfdr
    df["wilcoxon_fdr"] = wilcoxon_fdr

    # Permutation p-values
    if perm_results is not None:
        df["perm_p"] = df["atc_code"].map(perm_results)
    else:
        df["perm_p"] = np.nan

    df = df.sort_values("gls_p").reset_index(drop=True)

    output_cols = [
        "atc_code", "atc_level", "atc_description", "n_drugs",
        "n_genes_total", "mean_targets_per_drug", "median_targets_per_drug",
        "annotation_bias_risk", "borderline_power",
        "gls_beta", "gls_se", "gls_t", "gls_p", "gls_fdr",
        "wilcoxon_u", "wilcoxon_p", "wilcoxon_auc", "wilcoxon_fdr",
        "perm_p", "contributing_drugs", "contributing_drug_names",
    ]
    present_cols = [c for c in output_cols if c in df.columns]
    df = df[present_cols]

    n_sig_gls = int((df["gls_fdr"] < fdr_threshold).sum())
    n_sig_wilcoxon = int((df["wilcoxon_fdr"].dropna() < fdr_threshold).sum())
    logger.info(
        "ATC enrichment: %d classes tested, %d significant (GLS FDR < %.2f), "
        "%d significant (Wilcoxon FDR < %.2f)",
        len(df),
        n_sig_gls,
        fdr_threshold,
        n_sig_wilcoxon,
        fdr_threshold,
    )

    return df


# ---------------------------------------------------------------------------
# Drug gene-set file reader
# ---------------------------------------------------------------------------


def _read_drug_geneset_file(path: Path) -> dict[str, list[int]]:
    """Read a MAGMA gene-set file (tab-separated, one drug per line).

    Format: ``DRUG_ID<TAB>GENE1<TAB>GENE2<TAB>...``

    Args:
        path: Path to the gene-set file.

    Returns:
        Dict mapping drug_chembl_id to list of Entrez gene IDs (int).
    """
    drug_gene_sets: dict[str, list[int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            drug_id = parts[0]
            gene_ids = [int(g) for g in parts[1:] if g.strip()]
            drug_gene_sets[drug_id] = gene_ids
    return drug_gene_sets


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_atc_enrichment(
    drug_results_path: Path,
    genes_raw_path: Path,
    drug_geneset_path: Path,
    config: ATCEnrichmentConfig,
    output_dir: Path,
    study_name: str,
) -> pd.DataFrame:
    """Orchestrate ATC class enrichment analysis.

    Steps:
        1. Load drug enrichment results (Parquet from 8.3).
        2. Parse ``.genes.raw`` for gene-gene LD correlations.
        3. Build chromosome projections (eigendecompose + whitening).
        4. Load drug gene sets from the MAGMA gene set file (from 8.3).
        5. Compute drug-drug set correlation matrix + precision matrix.
        6. Extract ATC classes by string-slicing drug ATC codes.
        7. For each class: run GLS regression + Wilcoxon.
        8. Optionally: run permutation test.
        9. Assemble results, apply FDR, add diagnostics.
        10. Save Parquet + JSON metadata.

    Args:
        drug_results_path: Path to drug enrichment results Parquet.
        genes_raw_path: Path to MAGMA ``.genes.raw`` file.
        drug_geneset_path: Path to drug gene-set file (tab-separated,
            created by ``drug_enrichment.py``).
        config: ATCEnrichmentConfig.
        output_dir: Base output directory.
        study_name: Study identifier for file naming.

    Returns:
        ATCEnrichmentResult DataFrame.

    Raises:
        FileNotFoundError: If input files are missing.
        ValueError: If required columns are missing from drug results.
    """
    t0 = time.monotonic()

    # Validate input files
    if not drug_results_path.is_file():
        raise FileNotFoundError(f"Drug enrichment results not found: {drug_results_path}")
    if not genes_raw_path.is_file():
        raise FileNotFoundError(f"MAGMA .genes.raw file not found: {genes_raw_path}")
    if not drug_geneset_path.is_file():
        raise FileNotFoundError(f"Drug gene-set file not found: {drug_geneset_path}")

    # 1. Load drug enrichment results
    drug_results = pd.read_parquet(drug_results_path)

    required_cols = {"drug_chembl_id", "n_target_genes", "atc_codes"}
    missing = required_cols - set(drug_results.columns)
    if missing:
        raise ValueError(f"Drug results missing required columns: {missing}")

    if "magma_z" not in drug_results.columns:
        if "magma_beta" in drug_results.columns and "magma_beta_se" in drug_results.columns:
            se = drug_results["magma_beta_se"]
            if (se == 0).any() or se.isna().any():
                n_bad = int((se == 0).sum() + se.isna().sum())
                raise ValueError(
                    f"Cannot derive magma_z: {n_bad} drugs have zero or NaN "
                    f"magma_beta_se. Provide magma_z directly or fix upstream data."
                )
            drug_results = drug_results.copy()
            drug_results["magma_z"] = drug_results["magma_beta"] / se
            logger.info("Derived magma_z from magma_beta / magma_beta_se")
        else:
            raise ValueError(
                "Drug results missing 'magma_z' and cannot derive it "
                "(need 'magma_beta' and 'magma_beta_se')."
            )

    # Deserialise atc_codes to canonical lists (handles JSON strings, plain
    # ATC codes, mixed types, and Parquet round-trip variants safely).
    # Applied unconditionally: _deserialize_atc_codes is safe for all types
    # including lists (passthrough), strings, None, ndarray, and StringDtype.
    if len(drug_results) > 0:
        drug_results["atc_codes"] = drug_results["atc_codes"].apply(_deserialize_atc_codes)

    # Provenance counts (computed before any universe filtering)
    n_drugs_with_atc = int(drug_results["atc_codes"].apply(_has_atc_codes).sum())
    n_drugs_input_total = len(drug_results)

    # Universe filtering (DRUGSETS fidelity: restrict to ATC-annotated drugs)
    if config.atc_universe_mode == "annotated_only":
        has_atc_mask = drug_results["atc_codes"].apply(_has_atc_codes)
        drug_results = drug_results[has_atc_mask].reset_index(drop=True)
        logger.info(
            "ATC universe mode='annotated_only': restricted from %d to %d drugs with ATC codes",
            n_drugs_input_total, len(drug_results),
        )

    logger.info("Loaded %d drug results from %s", len(drug_results), drug_results_path)

    # Early exit if no drugs remain after filtering -
    # skip the expensive matrix computation entirely.
    if len(drug_results) == 0 or n_drugs_with_atc == 0:
        logger.warning("No drugs with ATC codes - skipping correlation matrix, returning empty results")
        atc_dir = output_dir / study_name / "atc_enrichment"
        ensure_directory(atc_dir)
        empty_df = pd.DataFrame()
        empty_df.to_parquet(atc_dir / "atc_enrichment_results.parquet", index=False)

        empty_metadata = {
            "result_type": "atc_enrichment",
            "study": study_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "atc_universe_mode": config.atc_universe_mode,
                "n_drugs_input_total": n_drugs_input_total,
                "n_drugs_tested": len(drug_results),
                "n_drugs_with_atc": n_drugs_with_atc,
                "atc_levels": config.atc_levels,
                "min_drugs_per_class": config.min_drugs_per_class,
                "two_sided": config.two_sided,
                "fdr_method": config.fdr_method,
                "fdr_threshold": config.fdr_threshold,
                "eigenvalue_threshold": config.eigenvalue_threshold,
                "permutation_test": config.permutation_test,
                "n_permutations": config.n_permutations if config.permutation_test else None,
                "n_classes_tested": 0,
                "correlation_matrix_condition_number": None,
            },
            "summary": {
                "n_classes_tested": 0,
                "n_significant_gls": 0,
                "n_significant_wilcoxon": 0,
                "n_concordant_significant": 0,
                "top_class": None,
                "top_class_description": None,
                "top_class_gls_p": None,
                "top_class_gls_fdr": None,
            },
        }
        json_path = atc_dir / "atc_enrichment_metadata.json"
        with open(json_path, "w") as f:
            json.dump(empty_metadata, f, indent=2, default=str)

        return empty_df

    # 2. Parse .genes.raw
    gene_info, chr_corr_matrices = parse_genes_raw(genes_raw_path)

    # 3. Build chromosome projections
    projections, proj_indices = build_chromosome_projections(
        gene_info, chr_corr_matrices, config.eigenvalue_threshold
    )

    # 4. Load drug gene sets
    drug_gene_sets = _read_drug_geneset_file(drug_geneset_path)

    # Filter gene sets to match universe (efficiency + correctness)
    if config.atc_universe_mode == "annotated_only":
        surviving_ids = set(drug_results["drug_chembl_id"])
        drug_gene_sets = {k: v for k, v in drug_gene_sets.items() if k in surviving_ids}

    result_drug_ids = set(drug_results["drug_chembl_id"])
    geneset_drug_ids = set(drug_gene_sets.keys())
    missing_in_geneset = result_drug_ids - geneset_drug_ids
    if missing_in_geneset:
        logger.warning(
            "%d drugs in results but not in gene-set file (will have empty target sets)",
            len(missing_in_geneset),
        )

    # 5. Compute drug-drug set correlations
    drug_z_ordered, set_corrs, set_corrs_inv, correlation_matrix_condition_number = (
        compute_drug_set_correlations(
            gene_info=gene_info,
            projections=projections,
            proj_indices=proj_indices,
            drug_results=drug_results,
            drug_gene_sets=drug_gene_sets,
            eigenvalue_threshold=config.eigenvalue_threshold,
        )
    )

    # 6. Extract ATC classes
    atc_classes = extract_atc_classes(
        drug_results=drug_results,
        atc_levels=config.atc_levels,
        min_drugs_per_class=config.min_drugs_per_class,
    )

    # 6b. Inject curated/custom classes (additive; independent of atc_levels).
    # Must run before the empty-classes early-exit so a run with only custom
    # classes (and no qualifying standard classes) is still tested.
    custom_classes, custom_meta = build_custom_atc_classes(
        drug_results=drug_results,
        custom_classes=config.custom_classes,
        min_drugs_per_class=config.min_drugs_per_class,
    )
    for _code, _ids in custom_classes.items():
        if _code in atc_classes:
            logger.warning(
                "Custom ATC class '%s' collides with a standard class code - "
                "keeping standard, skipping custom.", _code,
            )
            custom_meta.pop(_code, None)
            continue
        atc_classes[_code] = _ids

    if not atc_classes:
        logger.warning("No ATC classes meet the minimum size - returning empty results")
        atc_dir = output_dir / study_name / "atc_enrichment"
        ensure_directory(atc_dir)
        empty_df = pd.DataFrame()
        empty_df.to_parquet(atc_dir / "atc_enrichment_results.parquet", index=False)

        empty_metadata = {
            "result_type": "atc_enrichment",
            "study": study_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "atc_universe_mode": config.atc_universe_mode,
                "n_drugs_input_total": n_drugs_input_total,
                "n_drugs_tested": len(drug_results),
                "n_drugs_with_atc": n_drugs_with_atc,
                "atc_levels": config.atc_levels,
                "min_drugs_per_class": config.min_drugs_per_class,
                "two_sided": config.two_sided,
                "fdr_method": config.fdr_method,
                "fdr_threshold": config.fdr_threshold,
                "eigenvalue_threshold": config.eigenvalue_threshold,
                "permutation_test": config.permutation_test,
                "n_permutations": config.n_permutations if config.permutation_test else None,
                "n_classes_tested": 0,
                "correlation_matrix_condition_number": correlation_matrix_condition_number,
            },
            "summary": {
                "n_classes_tested": 0,
                "n_significant_gls": 0,
                "n_significant_wilcoxon": 0,
                "n_concordant_significant": 0,
                "top_class": None,
                "top_class_description": None,
                "top_class_gls_p": None,
                "top_class_gls_fdr": None,
            },
        }
        json_path = atc_dir / "atc_enrichment_metadata.json"
        with open(json_path, "w") as f:
            json.dump(empty_metadata, f, indent=2, default=str)

        return empty_df

    # Build index mapping for drug IDs
    drug_ids_ordered = drug_results["drug_chembl_id"].tolist()
    drug_id_to_idx = {d: i for i, d in enumerate(drug_ids_ordered)}
    n_genes_arr = drug_results["n_target_genes"].values.astype(float)

    # Build ID-to-name mapping for human-readable companion column.
    # Falls back to the ChEMBL ID when drug_name is absent or blank.
    has_drug_name = "drug_name" in drug_results.columns
    if has_drug_name:
        id_to_name: dict[str, str] = {}
        for cid, name in zip(drug_results["drug_chembl_id"], drug_results["drug_name"]):
            sname = str(name).strip() if pd.notna(name) else ""
            id_to_name[cid] = sname if sname else cid
    else:
        logger.warning("drug_name column absent - contributing_drug_names will use IDs")
        id_to_name = {}

    # Annotation bias global reference
    global_mean_targets = float(drug_results["n_target_genes"].mean())

    # 7. Per-class GLS + Wilcoxon
    class_results: list[dict] = []

    for atc_code, class_drug_ids in atc_classes.items():
        atc_indicator = np.zeros(len(drug_ids_ordered))
        class_drug_idxs = []
        for d in class_drug_ids:
            idx = drug_id_to_idx.get(d)
            if idx is not None:
                atc_indicator[idx] = 1.0
                class_drug_idxs.append(idx)

        n_drugs_in_class = int(atc_indicator.sum())
        if n_drugs_in_class < 2:
            continue

        if atc_code in custom_meta:
            atc_level = custom_meta[atc_code]["level"]
            description = custom_meta[atc_code]["description"]
        else:
            atc_level = _infer_atc_level(atc_code)
            description = ATC_DESCRIPTIONS.get(atc_code, "")

        class_n_genes = n_genes_arr[class_drug_idxs]
        n_genes_total = int(
            len(set(
                gid
                for d in class_drug_ids
                for gid in drug_gene_sets.get(d, [])
            ))
        )
        mean_tpd = float(np.mean(class_n_genes))
        median_tpd = float(np.median(class_n_genes))

        ratio = mean_tpd / global_mean_targets if global_mean_targets > 0 else 0
        if ratio <= 1.5:
            bias_risk = "low"
        elif ratio <= 2.0:
            bias_risk = "medium"
        else:
            bias_risk = "high"

        borderline = 5 <= n_drugs_in_class <= 9

        gls = run_gls_regression(
            drug_z_ordered, atc_indicator, n_genes_arr, set_corrs_inv, config.two_sided
        )
        wilcoxon = run_wilcoxon_test(drug_z_ordered, atc_indicator, config.two_sided)

        n_outside = len(drug_ids_ordered) - n_drugs_in_class
        denom = n_drugs_in_class * n_outside
        wilcoxon_auc = wilcoxon["u_stat"] / denom if denom > 0 else np.nan

        sorted_ids = sorted(class_drug_ids)
        class_results.append({
            "atc_code": atc_code,
            "atc_level": atc_level,
            "atc_description": description,
            "n_drugs": n_drugs_in_class,
            "n_genes_total": n_genes_total,
            "mean_targets_per_drug": mean_tpd,
            "median_targets_per_drug": median_tpd,
            "annotation_bias_risk": bias_risk,
            "borderline_power": borderline,
            "gls_beta": gls["beta"],
            "gls_se": gls["se"],
            "gls_t": gls["t_stat"],
            "gls_p": gls["p_value"],
            "wilcoxon_u": wilcoxon["u_stat"],
            "wilcoxon_p": wilcoxon["p_value"],
            "wilcoxon_auc": wilcoxon_auc,
            "contributing_drugs": ",".join(sorted_ids),
            "contributing_drug_names": ",".join(
                id_to_name.get(d, d) for d in sorted_ids
            ),
        })

    # 8. Optional permutation test
    perm_results = None
    if config.permutation_test:
        logger.info(
            "Running permutation test: %d classes, %d permutations each",
            len(atc_classes),
            config.n_permutations,
        )
        perm_results = run_permutation_test(
            y=drug_z_ordered,
            atc_classes=atc_classes,
            drug_id_to_idx=drug_id_to_idx,
            n_genes=n_genes_arr,
            set_corrs_inv=set_corrs_inv,
            n_permutations=config.n_permutations,
            seed=config.permutation_seed,
            two_sided=config.two_sided,
        )

    # 9. Assemble results + FDR
    results = assemble_atc_results(
        class_results=class_results,
        fdr_method=config.fdr_method,
        fdr_threshold=config.fdr_threshold,
        perm_results=perm_results,
    )

    # 10. Save outputs
    atc_dir = output_dir / study_name / "atc_enrichment"
    ensure_directory(atc_dir)

    parquet_path = atc_dir / "atc_enrichment_results.parquet"
    results.to_parquet(parquet_path, index=False)

    n_sig_gls = int((results["gls_fdr"] < config.fdr_threshold).sum()) if len(results) > 0 else 0
    n_sig_wilcoxon = int((results["wilcoxon_fdr"].dropna() < config.fdr_threshold).sum()) if len(results) > 0 else 0
    n_concordant = 0
    if len(results) > 0:
        n_concordant = int(
            ((results["gls_fdr"] < config.fdr_threshold)
             & (results["wilcoxon_fdr"].fillna(1.0) < config.fdr_threshold)).sum()
        )

    top_class = results.iloc[0]["atc_code"] if len(results) > 0 else None
    top_desc = results.iloc[0]["atc_description"] if len(results) > 0 else None
    top_gls_p = float(results.iloc[0]["gls_p"]) if len(results) > 0 else None
    top_gls_fdr = float(results.iloc[0]["gls_fdr"]) if len(results) > 0 else None

    metadata = {
        "result_type": "atc_enrichment",
        "study": study_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "atc_universe_mode": config.atc_universe_mode,
            "n_drugs_input_total": n_drugs_input_total,
            "n_drugs_tested": len(drug_results),
            "n_drugs_with_atc": n_drugs_with_atc,
            "atc_levels": config.atc_levels,
            "min_drugs_per_class": config.min_drugs_per_class,
            "two_sided": config.two_sided,
            "fdr_method": config.fdr_method,
            "fdr_threshold": config.fdr_threshold,
            "eigenvalue_threshold": config.eigenvalue_threshold,
            "permutation_test": config.permutation_test,
            "n_permutations": config.n_permutations if config.permutation_test else None,
            "n_classes_tested": len(class_results),
            "n_custom_classes": len(custom_meta),
            "custom_class_codes": sorted(custom_meta.keys()),
            "correlation_matrix_condition_number": correlation_matrix_condition_number,
        },
        "summary": {
            "n_classes_tested": len(class_results),
            "n_significant_gls": n_sig_gls,
            "n_significant_wilcoxon": n_sig_wilcoxon,
            "n_concordant_significant": n_concordant,
            "top_class": top_class,
            "top_class_description": top_desc,
            "top_class_gls_p": top_gls_p,
            "top_class_gls_fdr": top_gls_fdr,
        },
    }

    json_path = atc_dir / "atc_enrichment_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    elapsed = time.monotonic() - t0
    logger.info(
        "ATC enrichment analysis completed in %.1fs. "
        "%d classes tested, %d significant at GLS FDR < %.2f.",
        elapsed,
        len(class_results),
        n_sig_gls,
        config.fdr_threshold,
    )

    return results


def _infer_atc_level(atc_code: str) -> int:
    """Infer the ATC hierarchy level from code length."""
    code_len = len(atc_code)
    for level, prefix_len in sorted(ATC_LEVEL_PREFIX.items()):
        if code_len == prefix_len:
            return level
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ATC class enrichment analysis")
    parser.add_argument("--drug-results", type=Path, required=True,
                        help="Path to drug enrichment results Parquet")
    parser.add_argument("--genes-raw", type=Path, required=True,
                        help="Path to MAGMA .genes.raw file")
    parser.add_argument("--drug-geneset", type=Path, required=True,
                        help="Path to drug gene-set file (tab-separated)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to pipeline config YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    from repogen.config.loader import load_config

    pipeline_config = load_config(args.config)

    run_atc_enrichment(
        drug_results_path=args.drug_results,
        genes_raw_path=args.genes_raw,
        drug_geneset_path=args.drug_geneset,
        config=pipeline_config.atc_enrichment,
        output_dir=args.output_dir,
        study_name=args.study_name,
    )
