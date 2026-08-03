"""Drug-gene enrichment analysis using MAGMA competitive gene-set testing.

Primary test: MAGMA competitive regression (corrects for gene size, density, LD).
Secondary metric: Wilcoxon rank-sum AUC (optional descriptive complement).
Optional: Permutation-based enrichment test.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from repogen.analysis.magma_gene import detect_magma_binary
from repogen.analysis.magma_pathway import (
    _get_magma_version,
    run_magma_geneset_analysis,
)
from repogen.config.schema import DrugEnrichmentConfig
from repogen.utils.io import ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

CONFIDENCE_HIERARCHY = {"high": 3, "medium": 2, "low": 1}


# ---------------------------------------------------------------------------
# Gene-set construction
# ---------------------------------------------------------------------------


def build_drug_gene_sets(
    drug_targets: pd.DataFrame,
    gene_results: pd.DataFrame,
    min_genes_per_drug: int = 3,
    min_pchembl: Optional[float] = None,
    max_phase_filter: Optional[int] = None,
    confidence_filter: Optional[str] = None,
    phase_filter_scope: str = "global",
    atc_min_genes_per_drug: Optional[int] = None,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Build drug -> Entrez gene ID mappings from DrugTargetRecord.

    Steps:
    1. Optionally filter by pchembl_value >= min_pchembl (row-level, keeps NaN).
    2. Optionally filter by max_phase >= max_phase_filter (drug-level).
       When ``phase_filter_scope="chembl_only"``, only drugs whose
       ``source`` contains ``"chembl"`` are subject to phase gating;
       PDSP/DGIdb-only drugs are exempt.
    3. Optionally filter by confidence hierarchy (row-level).
    4. Drop rows where gene_entrez_id is null.
    5. Intersect with genes present in gene_results.
    6. Group by drug_chembl_id, collect unique Entrez IDs per drug.
    7. Exclude drugs with fewer than ``atc_min_genes_per_drug`` target
       genes.  Drugs with fewer than ``min_genes_per_drug`` genes still
       enter the gene-set file but are flagged at result-assembly time
       so headline drug-gene FDR / exports / plot can exclude them.

    Args:
        drug_targets: DrugTargetRecord DataFrame from drug_loader.py.
        gene_results: Parsed MAGMA gene results DataFrame.
            Must contain 'gene_entrez_id' column.
        min_genes_per_drug: Headline threshold (default 3).  Used by the
            caller to set ``passes_headline_min_genes`` downstream; this
            function only uses it as the inheritance source for
            ``atc_min_genes_per_drug``.
        min_pchembl: Optional potency threshold. If set, only keep
            interactions with pchembl_value >= this value.
        max_phase_filter: Optional phase filter. If set, only keep
            drugs with max_phase >= this value.
        confidence_filter: Optional confidence level. If set, only keep
            interactions at this level or higher ('high' > 'medium' > 'low').
        phase_filter_scope: ``"global"`` (default, all sources) or
            ``"chembl_only"`` (exempt PDSP/DGIdb-only drugs).
        atc_min_genes_per_drug: Minimum genes for a
            drug to enter the (inclusive) ATC enrichment universe.  When
            ``None`` (default), inherits ``min_genes_per_drug`` for
            byte-identical legacy behaviour.  Production callers pass
            ``config.atc_min_genes_per_drug`` (typically 1).

    Returns:
        Tuple of (drug_gene_sets dict, filter_stats dict with per-filter counts).
    """
    if atc_min_genes_per_drug is None:
        atc_min_genes_per_drug = min_genes_per_drug
    df = drug_targets.copy()
    filter_stats: dict[str, int] = {"n_total_pairs": len(df)}

    if min_pchembl is not None:
        mask = df["pchembl_value"].isna() | (df["pchembl_value"] >= min_pchembl)
        n_before = len(df)
        df = df[mask].copy()
        filter_stats["n_pairs_dropped_pchembl"] = n_before - len(df)
        logger.info(
            "pChEMBL filter (>= %.1f): %d -> %d interactions",
            min_pchembl,
            n_before,
            len(df),
        )
    else:
        filter_stats["n_pairs_dropped_pchembl"] = 0

    if max_phase_filter is not None:
        n_drugs_before = df["drug_chembl_id"].nunique()
        n_before = len(df)

        if phase_filter_scope == "chembl_only" and "source" in df.columns:
            has_chembl = df.groupby("drug_chembl_id")["source"].apply(
                lambda s: s.str.contains("chembl", case=False).any()
            )
            chembl_drug_ids = has_chembl[has_chembl].index
            drugs_with_phase = df.groupby("drug_chembl_id")["max_phase"].max()
            phase_fail = drugs_with_phase[drugs_with_phase < max_phase_filter].index
            drop_ids = set(phase_fail) & set(chembl_drug_ids)
            df = df[~df["drug_chembl_id"].isin(drop_ids)].copy()
            logger.info(
                "Phase filter (>= %d, scope=chembl_only): %d -> %d interactions "
                "(%d ChEMBL-source drugs dropped, non-ChEMBL exempt)",
                max_phase_filter,
                n_before,
                len(df),
                len(drop_ids),
            )
        else:
            drugs_with_phase = df.groupby("drug_chembl_id")["max_phase"].max()
            valid_drugs = drugs_with_phase[drugs_with_phase >= max_phase_filter].index
            df = df[df["drug_chembl_id"].isin(valid_drugs)].copy()
            logger.info(
                "Phase filter (>= %d, scope=global): %d -> %d interactions",
                max_phase_filter,
                n_before,
                len(df),
            )

        filter_stats["n_drugs_dropped_phase"] = n_drugs_before - df["drug_chembl_id"].nunique()
    else:
        filter_stats["n_drugs_dropped_phase"] = 0

    if confidence_filter is not None:
        min_level = CONFIDENCE_HIERARCHY.get(confidence_filter, 0)
        df["_conf_rank"] = df["confidence"].map(CONFIDENCE_HIERARCHY).fillna(0)
        n_before = len(df)
        df = df[df["_conf_rank"] >= min_level].copy()
        df = df.drop(columns=["_conf_rank"])
        filter_stats["n_pairs_dropped_confidence"] = n_before - len(df)
        logger.info(
            "Confidence filter (%s+): %d -> %d interactions",
            confidence_filter,
            n_before,
            len(df),
        )
    else:
        filter_stats["n_pairs_dropped_confidence"] = 0

    n_no_entrez = int(df["gene_entrez_id"].isna().sum())
    filter_stats["n_pairs_dropped_no_entrez"] = n_no_entrez
    if n_no_entrez > 0:
        logger.warning(
            "Dropping %d interactions with no Entrez ID (of %d total)",
            n_no_entrez,
            len(df),
        )
    df = df.dropna(subset=["gene_entrez_id"]).copy()
    df["gene_entrez_id"] = df["gene_entrez_id"].astype(int)

    valid_entrez = set(gene_results["gene_entrez_id"].dropna().astype(int).values)
    n_before = len(df)
    df = df[df["gene_entrez_id"].isin(valid_entrez)].copy()
    filter_stats["n_pairs_dropped_not_in_magma"] = n_before - len(df)
    if filter_stats["n_pairs_dropped_not_in_magma"] > 0:
        logger.info(
            "Gene intersection with MAGMA results: %d -> %d interactions "
            "(%d target genes not in MAGMA results)",
            n_before,
            len(df),
            filter_stats["n_pairs_dropped_not_in_magma"],
        )

    drug_sets: dict[str, list[int]] = {}
    n_headline_pool = 0
    for drug_id, group in df.groupby("drug_chembl_id"):
        entrez_ids = sorted(group["gene_entrez_id"].unique().tolist())
        if len(entrez_ids) >= atc_min_genes_per_drug:
            drug_sets[drug_id] = entrez_ids
            if len(entrez_ids) >= min_genes_per_drug:
                n_headline_pool += 1

    n_drugs_after_filters = df["drug_chembl_id"].nunique()
    filter_stats["n_drugs_dropped_min_genes"] = n_drugs_after_filters - len(drug_sets)
    filter_stats["n_drugs_tested"] = len(drug_sets)
    filter_stats["n_drugs_in_atc_pool"] = len(drug_sets)
    filter_stats["n_drugs_in_headline_pool"] = n_headline_pool
    filter_stats["atc_min_genes_per_drug"] = atc_min_genes_per_drug
    filter_stats["headline_min_genes_per_drug"] = min_genes_per_drug

    logger.info(
        "Drug gene sets: %d drugs after pair filters, %d enter ATC pool "
        "(>= %d genes), %d in headline pool (>= %d genes), %d excluded. "
        "%d initial interactions -> %d used.",
        n_drugs_after_filters,
        len(drug_sets),
        atc_min_genes_per_drug,
        n_headline_pool,
        min_genes_per_drug,
        filter_stats["n_drugs_dropped_min_genes"],
        filter_stats["n_total_pairs"],
        len(df),
    )

    return drug_sets, filter_stats


# ---------------------------------------------------------------------------
# PDSP signature-level deduplication
# ---------------------------------------------------------------------------


def collapse_pdsp_clusters(
    drug_gene_sets: dict[str, list[int]],
    drug_targets: pd.DataFrame,
) -> tuple[dict[str, list[int]], dict]:
    """Collapse PDSP-only drugs with identical target gene sets.

    For each group of PDSP-only drugs sharing the exact same sorted
    Entrez target set, keeps one deterministic representative and removes
    the rest from ``drug_gene_sets``.  Non-PDSP and mixed-source drugs
    are never touched.

    Args:
        drug_gene_sets: Drug -> sorted Entrez IDs (from build_drug_gene_sets).
        drug_targets: Full DrugTargetRecord DataFrame (needs ``drug_chembl_id``,
            ``source``, and optionally ``atc_codes``, ``pchembl_value``).

    Returns:
        Tuple of (collapsed drug_gene_sets copy, cluster_stats dict).
    """
    n_pre = len(drug_gene_sets)

    # --- Identify PDSP-only drugs via comma-split source tokens ---
    tested_ids = set(drug_gene_sets.keys())
    dt = drug_targets[drug_targets["drug_chembl_id"].isin(tested_ids)].copy()

    def _source_tokens(series: pd.Series) -> set[str]:
        tokens: set[str] = set()
        for val in series.dropna():
            for part in str(val).split(","):
                part = part.strip().lower()
                if part:
                    tokens.add(part)
        return tokens

    pdsp_only_ids: set[str] = set()
    if "source" in dt.columns:
        for drug_id, grp in dt.groupby("drug_chembl_id"):
            tokens = _source_tokens(grp["source"])
            if tokens and tokens <= {"pdsp"}:
                pdsp_only_ids.add(drug_id)

    pdsp_in_sets = pdsp_only_ids & tested_ids
    n_pdsp_candidates = len(pdsp_in_sets)

    if n_pdsp_candidates == 0:
        logger.info("PDSP dedup: no PDSP-only drugs in gene sets - nothing to collapse")
        return dict(drug_gene_sets), {
            "n_pdsp_only_candidates": 0,
            "n_pdsp_clusters": 0,
            "n_pdsp_drugs_collapsed": 0,
            "n_drugs_pre_dedup": n_pre,
            "n_drugs_post_dedup": n_pre,
            "cluster_map": {},
        }

    # --- Build per-drug scoring metadata from drug_targets ---
    has_chembl_id: dict[str, bool] = {}
    has_atc: dict[str, bool] = {}
    mean_pchembl: dict[str, float] = {}

    for drug_id in pdsp_in_sets:
        has_chembl_id[drug_id] = str(drug_id).startswith("CHEMBL")

    if "atc_codes" in dt.columns:
        atc_agg = dt[dt["drug_chembl_id"].isin(pdsp_in_sets)].groupby("drug_chembl_id")["atc_codes"].apply(
            lambda x: any(
                (isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0)
                or (isinstance(v, str) and v.strip())
                for v in x.dropna()
            )
        )
        for drug_id in pdsp_in_sets:
            has_atc[drug_id] = bool(atc_agg.get(drug_id, False))
    else:
        for drug_id in pdsp_in_sets:
            has_atc[drug_id] = False

    if "pchembl_value" in dt.columns:
        pch_agg = (
            dt[dt["drug_chembl_id"].isin(pdsp_in_sets)]
            .groupby("drug_chembl_id")["pchembl_value"]
            .mean()
        )
        for drug_id in pdsp_in_sets:
            val = pch_agg.get(drug_id, np.nan)
            mean_pchembl[drug_id] = float(val) if pd.notna(val) else float("-inf")
    else:
        for drug_id in pdsp_in_sets:
            mean_pchembl[drug_id] = float("-inf")

    # --- Group by signature key ---
    sig_to_drugs: dict[tuple[int, ...], list[str]] = {}
    for drug_id in pdsp_in_sets:
        sig = tuple(drug_gene_sets[drug_id])
        sig_to_drugs.setdefault(sig, []).append(drug_id)

    n_clusters = len(sig_to_drugs)

    # --- Select representative per cluster ---
    cluster_map: dict[str, list[str]] = {}
    to_remove: set[str] = set()

    for sig, members in sig_to_drugs.items():
        if len(members) == 1:
            continue

        ranked = sorted(
            members,
            key=lambda d: (
                has_chembl_id.get(d, False),
                has_atc.get(d, False),
                mean_pchembl.get(d, float("-inf")),
                d,
            ),
            reverse=True,
        )
        # reverse=True puts True > False, higher pchembl first.
        # For lexical tie-break we want ascending, so re-sort on
        # the last key. Since Python sort is stable and the first
        # three keys already dominate, we handle the final tie-break
        # by negating: use a tuple where the last element sorts
        # ascending within equal-priority groups.
        ranked = sorted(
            members,
            key=lambda d: (
                not has_chembl_id.get(d, False),
                not has_atc.get(d, False),
                -mean_pchembl.get(d, float("-inf")),
                d,
            ),
        )
        representative = ranked[0]
        collapsed = sorted(m for m in members if m != representative)
        cluster_map[representative] = collapsed
        to_remove.update(collapsed)

    collapsed_sets = {k: v for k, v in drug_gene_sets.items() if k not in to_remove}
    n_collapsed = len(to_remove)
    n_post = len(collapsed_sets)

    logger.info(
        "PDSP dedup: %d PDSP-only candidates, %d unique signatures, "
        "%d drugs collapsed, %d -> %d total drugs",
        n_pdsp_candidates, n_clusters, n_collapsed, n_pre, n_post,
    )

    return collapsed_sets, {
        "n_pdsp_only_candidates": n_pdsp_candidates,
        "n_pdsp_clusters": n_clusters,
        "n_pdsp_drugs_collapsed": n_collapsed,
        "n_drugs_pre_dedup": n_pre,
        "n_drugs_post_dedup": n_post,
        "cluster_map": cluster_map,
    }


# ---------------------------------------------------------------------------
# Gene-set file creation
# ---------------------------------------------------------------------------


def create_drug_geneset_file(
    drug_gene_sets: dict[str, list[int]],
    output_path: Path,
) -> Path:
    """Write MAGMA-compatible gene-set annotation file for drug targets.

    Format: one line per drug, tab-separated:
        DRUG_CHEMBL_ID<tab>ENTREZ_ID_1<tab>ENTREZ_ID_2<tab>...

    Args:
        drug_gene_sets: Dict from build_drug_gene_sets().
        output_path: Path to write the gene-set file.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If drug_gene_sets is empty.
    """
    if not drug_gene_sets:
        raise ValueError("No drug gene sets to write - all drugs filtered out")

    ensure_directory(output_path.parent)

    with open(output_path, "w") as fh:
        for drug_id in sorted(drug_gene_sets.keys()):
            entrez_ids = drug_gene_sets[drug_id]
            line = drug_id + "\t" + "\t".join(str(eid) for eid in entrez_ids)
            fh.write(line + "\n")

    logger.info(
        "Drug gene-set file: %d drugs written -> %s",
        len(drug_gene_sets),
        output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# MAGMA subprocess execution
# ---------------------------------------------------------------------------


def run_magma_drug_enrichment(
    gene_results_raw: Path,
    drug_geneset_file: Path,
    output_prefix: Path,
    magma_binary: Optional[Path] = None,
) -> Path:
    """Run MAGMA competitive gene-set analysis for drug targets.

    Delegates to run_magma_geneset_analysis() from magma_pathway.py,
    which handles subprocess execution with proper token separation.

    Args:
        gene_results_raw: Path to MAGMA .genes.raw file.
        drug_geneset_file: Path from create_drug_geneset_file().
        output_prefix: Output prefix for MAGMA results.
        magma_binary: Optional explicit path to MAGMA binary.

    Returns:
        Path to the MAGMA .gsa.out results file.

    Raises:
        FileNotFoundError: If gene_results_raw or drug_geneset_file missing.
        RuntimeError: If MAGMA binary not found or MAGMA exits non-zero.
        FileNotFoundError: If expected .gsa.out file not created.
    """
    if not gene_results_raw.is_file():
        raise FileNotFoundError(
            f"MAGMA gene results file not found: {gene_results_raw}"
        )
    if not drug_geneset_file.is_file():
        raise FileNotFoundError(
            f"Drug gene-set file not found: {drug_geneset_file}"
        )

    binary = magma_binary if magma_binary is not None else detect_magma_binary()

    return run_magma_geneset_analysis(
        magma_binary=binary,
        gene_results_raw=gene_results_raw,
        geneset_file=drug_geneset_file,
        output_prefix=output_prefix,
    )


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_magma_drug_results(
    gsa_out_path: Path,
) -> pd.DataFrame:
    """Parse MAGMA .gsa.out file for drug enrichment results.

    MAGMA .gsa.out format (whitespace-delimited):
        VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P  [FULL_NAME]

    Uses ``FULL_NAME`` as canonical ``drug_chembl_id`` when present
    (MAGMA truncates long IDs in ``VARIABLE``).  Reads with
    ``errors="replace"`` to handle invalid UTF-8 from truncated
    non-ASCII drug names, and strips comment lines manually to avoid
    ``comment="#"`` truncating IDs that contain ``#`` (e.g. HTML
    entities like ``&#8242;``).

    Returns DataFrame with columns:
        drug_chembl_id, n_genes_in_magma, magma_beta, magma_beta_se, magma_p

    Args:
        gsa_out_path: Path to MAGMA .gsa.out file.

    Returns:
        DataFrame with one row per drug.

    Raises:
        FileNotFoundError: If gsa_out_path does not exist.
        ValueError: If file is empty or has unexpected format.
    """
    from io import StringIO

    if not gsa_out_path.is_file():
        raise FileNotFoundError(
            f"MAGMA .gsa.out file not found: {gsa_out_path}"
        )

    raw = gsa_out_path.read_text(encoding="utf-8", errors="replace")
    data_lines = [
        line for line in raw.splitlines()
        if line and not line.startswith("#")
    ]
    if not data_lines:
        raise ValueError(
            f"MAGMA .gsa.out file is empty (no data rows): {gsa_out_path}"
        )

    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        engine="python",
    )

    if df.empty:
        raise ValueError(
            f"MAGMA .gsa.out file is empty (no data rows): {gsa_out_path}"
        )

    expected = {"VARIABLE", "TYPE", "NGENES", "BETA", "SE", "P"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"MAGMA .gsa.out missing expected columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    if "FULL_NAME" in df.columns:
        drug_ids = df["FULL_NAME"]
        n_recovered = int((df["VARIABLE"] != df["FULL_NAME"]).sum())
        if n_recovered > 0:
            logger.info(
                "Using FULL_NAME as drug_chembl_id (%d / %d had truncated VARIABLE)",
                n_recovered, len(df),
            )
    else:
        drug_ids = df["VARIABLE"]

    beta = df["BETA"].astype(float)
    se = df["SE"].astype(float)

    result = pd.DataFrame({
        "drug_chembl_id": drug_ids,
        "n_genes_in_magma": df["NGENES"].astype(int),
        "magma_beta": beta,
        "magma_beta_se": se,
        "magma_z": beta / se,
        "magma_p": df["P"].astype(float),
    })

    logger.info(
        "Parsed %d drug enrichment results from %s",
        len(result),
        gsa_out_path,
    )

    return result


# ---------------------------------------------------------------------------
# Wilcoxon AUC
# ---------------------------------------------------------------------------


def compute_wilcoxon_auc(
    gene_z_scores: pd.Series,
    drug_gene_sets: dict[str, list[int]],
    gene_entrez_ids: pd.Series,
) -> pd.DataFrame:
    """Compute Wilcoxon rank-sum test and AUC for each drug.

    For each drug:
    1. Split gene Z-scores into target vs non-target groups.
    2. Run one-sided Mann-Whitney U test (alternative='greater').
    3. Compute AUC = U / (n_target x n_non_target).

    Args:
        gene_z_scores: Series of MAGMA Z-scores indexed by position.
        drug_gene_sets: Dict mapping drug_chembl_id -> list of Entrez IDs.
        gene_entrez_ids: Series of Entrez IDs aligned with gene_z_scores.

    Returns:
        DataFrame with columns: drug_chembl_id, wilcoxon_auc, wilcoxon_p
    """
    z_arr = gene_z_scores.values
    entrez_arr = gene_entrez_ids.astype(int).values

    results = []
    for drug_id, target_entrez in drug_gene_sets.items():
        target_mask = np.isin(entrez_arr, target_entrez)
        target_z = z_arr[target_mask]
        non_target_z = z_arr[~target_mask]

        n_target = len(target_z)
        n_non_target = len(non_target_z)

        if n_target == 0 or n_non_target == 0:
            results.append({
                "drug_chembl_id": drug_id,
                "wilcoxon_auc": 0.5,
                "wilcoxon_p": 1.0,
            })
            continue

        if np.all(target_z == target_z[0]):
            results.append({
                "drug_chembl_id": drug_id,
                "wilcoxon_auc": 0.5,
                "wilcoxon_p": 1.0,
            })
            continue

        try:
            u_stat, p_val = stats.mannwhitneyu(
                target_z, non_target_z, alternative="greater"
            )
            auc = u_stat / (n_target * n_non_target)
        except ValueError:
            auc = 0.5
            p_val = 1.0

        results.append({
            "drug_chembl_id": drug_id,
            "wilcoxon_auc": float(auc),
            "wilcoxon_p": float(p_val),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def compute_permutation_enrichment(
    gene_z_scores: np.ndarray,
    drug_gene_sets: dict[str, list[int]],
    gene_entrez_ids: np.ndarray,
    n_permutations: int = 10000,
    permutation_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Compute permutation-based enrichment p-values for each drug.

    For each drug:
    1. Compute observed mean Z-score of target genes.
    2. Draw n_permutations random gene sets of matching size.
    3. Compute mean Z for each random draw.
    4. Empirical p = (count(permuted_mean >= observed_mean) + 1) / (n_perms + 1)

    Vectorised: for each drug, generate a (n_perms x n_target) index matrix.

    Args:
        gene_z_scores: 1D numpy array of MAGMA Z-scores for all genes.
        drug_gene_sets: Dict mapping drug_chembl_id -> list of Entrez IDs.
        gene_entrez_ids: 1D numpy array of Entrez IDs aligned with gene_z_scores.
        n_permutations: Number of permutations per drug (default 10000).
        permutation_seed: Random seed for reproducibility (default 42).

    Returns:
        DataFrame with columns: drug_chembl_id, permutation_p
    """
    rng = np.random.default_rng(permutation_seed)

    entrez_to_idx: dict[int, int] = {}
    for idx, eid in enumerate(gene_entrez_ids):
        entrez_to_idx[int(eid)] = idx

    n_total = len(gene_z_scores)
    n_drugs = len(drug_gene_sets)

    logger.info(
        "Permutation test: %d drugs, %d permutations each. "
        "Minimum achievable p ~ %.1e",
        n_drugs,
        n_permutations,
        1.0 / n_permutations,
    )

    results = []
    for i, (drug_id, target_entrez) in enumerate(drug_gene_sets.items()):
        target_indices = np.array(
            [entrez_to_idx[eid] for eid in target_entrez if eid in entrez_to_idx],
            dtype=int,
        )
        n_target = len(target_indices)

        if n_target == 0:
            results.append({"drug_chembl_id": drug_id, "permutation_p": 1.0})
            continue

        observed_mean = gene_z_scores[target_indices].mean()

        random_indices = np.empty((n_permutations, n_target), dtype=np.intp)
        for p in range(n_permutations):
            random_indices[p] = rng.choice(n_total, size=n_target, replace=False)
        permuted_means = gene_z_scores[random_indices].mean(axis=1)
        p_val = (np.sum(permuted_means >= observed_mean) + 1) / (n_permutations + 1)

        results.append({
            "drug_chembl_id": drug_id,
            "permutation_p": float(p_val),
        })

        if n_drugs > 1000 and (i + 1) % 500 == 0:
            logger.info("Permutation progress: %d / %d drugs", i + 1, n_drugs)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def assemble_drug_results(
    magma_results: pd.DataFrame,
    drug_targets: pd.DataFrame,
    gene_results: pd.DataFrame,
    drug_gene_sets: dict[str, list[int]],
    wilcoxon_results: Optional[pd.DataFrame] = None,
    permutation_results: Optional[pd.DataFrame] = None,
    min_genes_per_drug: int = 3,
) -> pd.DataFrame:
    """Assemble the final DrugEnrichmentResult DataFrame.

    Merges MAGMA results with drug metadata from DrugTargetRecord and
    per-target-gene statistics from MAGMA gene results.

    Args:
        magma_results: From parse_magma_drug_results().  May or may not
            already have ``magma_fdr_q`` applied; the production caller
            attaches FDR after this function returns so the headline
            mask is computed from a single source of truth.
        drug_targets: Full DrugTargetRecord DataFrame.
        gene_results: Parsed MAGMA gene results DataFrame.
        drug_gene_sets: Dict from build_drug_gene_sets().
        wilcoxon_results: Optional DataFrame from compute_wilcoxon_auc().
        permutation_results: Optional DataFrame from compute_permutation_enrichment().
        min_genes_per_drug: Headline threshold (default 3 to match
            ``DrugEnrichmentConfig.min_genes_per_drug`` default).  Used
            to compute ``passes_headline_min_genes`` from the
            *pre-MAGMA* gene-set size (``len(drug_gene_sets[drug_id])``)
            so the flag matches the historical filter denominator.

    Returns:
        DrugEnrichmentResult DataFrame ready for Parquet serialisation.
    """
    results = magma_results.copy()

    tested_drugs = set(results["drug_chembl_id"])
    dt = drug_targets[drug_targets["drug_chembl_id"].isin(tested_drugs)].copy()

    scalar_cols = [
        "drug_name", "drug_inchikey", "drug_pubchem_cid", "drug_smiles",
        "max_phase", "molecule_type", "is_withdrawn",
    ]
    scalar_agg = {}
    for col in scalar_cols:
        if col in dt.columns:
            scalar_agg[col] = "first"

    list_agg_cols = {
        "mechanism_of_action": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0] if len(x) > 0 else None,
    }

    combined_agg = {**scalar_agg, **list_agg_cols}
    if combined_agg:
        drug_meta = dt.groupby("drug_chembl_id").agg(combined_agg).reset_index()
        results = results.merge(drug_meta, on="drug_chembl_id", how="left")

    def _collect_unique_list(series: pd.Series) -> list:
        out: list = []
        for val in series.dropna():
            if isinstance(val, (list, tuple, set, np.ndarray)):
                out.extend(val)
            elif isinstance(val, str) and val:
                out.append(val)
        return sorted(set(out))

    if "atc_codes" in dt.columns:
        atc_agg = dt.groupby("drug_chembl_id")["atc_codes"].apply(_collect_unique_list).reset_index()
        results = results.merge(atc_agg, on="drug_chembl_id", how="left")
    else:
        results["atc_codes"] = [[] for _ in range(len(results))]

    if "indication_mesh" in dt.columns:
        mesh_agg = dt.groupby("drug_chembl_id")["indication_mesh"].apply(_collect_unique_list).reset_index()
        results = results.merge(mesh_agg, on="drug_chembl_id", how="left")
    else:
        results["indication_mesh"] = [[] for _ in range(len(results))]

    if "source" in dt.columns:
        def _atomic_sources(series: pd.Series) -> list[str]:
            tokens: set[str] = set()
            for val in series.dropna():
                for part in str(val).split(","):
                    part = part.strip()
                    if part:
                        tokens.add(part)
            return sorted(tokens)

        src_agg = dt.groupby("drug_chembl_id")["source"].apply(
            _atomic_sources
        ).reset_index().rename(columns={"source": "sources"})
        results = results.merge(src_agg, on="drug_chembl_id", how="left")
    else:
        results["sources"] = [[] for _ in range(len(results))]

    if "confidence" in dt.columns:
        conf_agg = dt.groupby("drug_chembl_id")["confidence"].apply(
            lambda x: sorted(x.dropna().unique().tolist())
        ).reset_index().rename(columns={"confidence": "confidence_levels"})
        results = results.merge(conf_agg, on="drug_chembl_id", how="left")
    else:
        results["confidence_levels"] = [[] for _ in range(len(results))]

    gene_lookup = (
        gene_results[["gene_entrez_id", "gene_symbol", "gene_ensembl_id", "magma_z", "magma_p"]]
        .dropna(subset=["gene_entrez_id"])
        .copy()
    )
    gene_lookup["gene_entrez_id"] = gene_lookup["gene_entrez_id"].astype(int)
    gene_lookup = gene_lookup.drop_duplicates(subset=["gene_entrez_id"])
    gene_map = gene_lookup.set_index("gene_entrez_id").to_dict("index")

    dt_by_drug_gene: dict[tuple[str, int], dict] = {}
    for col in ["interaction_type", "pchembl_value", "source_pmids"]:
        if col not in dt.columns:
            dt[col] = None
    dt_subset = dt[["drug_chembl_id", "gene_entrez_id", "interaction_type", "pchembl_value", "source_pmids"]].copy()
    dt_subset = dt_subset.dropna(subset=["gene_entrez_id"])
    dt_subset["gene_entrez_id"] = dt_subset["gene_entrez_id"].astype(int)
    for row in dt_subset.itertuples(index=False):
        key = (row.drug_chembl_id, row.gene_entrez_id)
        if key not in dt_by_drug_gene:
            dt_by_drug_gene[key] = {
                "interaction_type": row.interaction_type,
                "pchembl_value": row.pchembl_value,
                "source_pmids": row.source_pmids,
            }

    target_genes_list = []
    mean_pchembl_list = []
    mean_target_z_list = []

    for drug_id in results["drug_chembl_id"]:
        entrez_ids = drug_gene_sets.get(drug_id, [])
        genes_detail = []
        pchembl_vals = []
        z_vals = []

        for eid in entrez_ids:
            gene_info = gene_map.get(eid, {})
            dt_info = dt_by_drug_gene.get((drug_id, eid), {})

            z_val = gene_info.get("magma_z")
            p_val = gene_info.get("magma_p")
            pchembl = dt_info.get("pchembl_value")
            pmids = dt_info.get("source_pmids")

            if isinstance(pmids, float) and np.isnan(pmids):
                pmids = None

            genes_detail.append({
                "gene_symbol": gene_info.get("gene_symbol"),
                "gene_entrez_id": eid,
                "gene_ensembl_id": gene_info.get("gene_ensembl_id"),
                "magma_z": float(z_val) if z_val is not None and not (isinstance(z_val, float) and np.isnan(z_val)) else None,
                "magma_p": float(p_val) if p_val is not None and not (isinstance(p_val, float) and np.isnan(p_val)) else None,
                "interaction_type": dt_info.get("interaction_type"),
                "pchembl_value": float(pchembl) if pchembl is not None and not (isinstance(pchembl, float) and np.isnan(pchembl)) else None,
                "source_pmids": pmids if pmids is not None else [],
            })

            if pchembl is not None and not (isinstance(pchembl, float) and np.isnan(pchembl)):
                pchembl_vals.append(float(pchembl))
            if z_val is not None and not (isinstance(z_val, float) and np.isnan(z_val)):
                z_vals.append(float(z_val))

        target_genes_list.append(genes_detail)
        mean_pchembl_list.append(float(np.mean(pchembl_vals)) if pchembl_vals else None)
        mean_target_z_list.append(float(np.mean(z_vals)) if z_vals else None)

    results["target_genes"] = target_genes_list
    results["mean_pchembl"] = mean_pchembl_list
    results["mean_target_z"] = mean_target_z_list

    results["n_target_genes"] = results["n_genes_in_magma"]

    input_set_sizes = results["drug_chembl_id"].map(
        lambda d: len(drug_gene_sets.get(d, []))
    ).astype(int)
    results["n_target_genes_input"] = input_set_sizes
    results["passes_headline_min_genes"] = (
        results["n_target_genes_input"] >= int(min_genes_per_drug)
    )

    discrepancy = results["n_genes_in_magma"] != input_set_sizes
    if discrepancy.any():
        n_disc = discrepancy.sum()
        logger.warning(
            "%d drugs have n_genes_in_magma != len(drug_gene_sets): "
            "MAGMA may have dropped genes with no SNPs in the reference panel",
            n_disc,
        )

    if wilcoxon_results is not None:
        results = results.merge(wilcoxon_results, on="drug_chembl_id", how="left")

    if permutation_results is not None:
        results = results.merge(permutation_results, on="drug_chembl_id", how="left")

    results = results.sort_values("magma_p").reset_index(drop=True)

    logger.info("Drug enrichment results assembled: %d drugs", len(results))

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_drug_enrichment(
    gene_results_raw: Path,
    gene_results_df: pd.DataFrame,
    drug_targets: pd.DataFrame,
    config: DrugEnrichmentConfig,
    output_dir: Path,
    study_name: str,
    magma_binary: Optional[Path] = None,
) -> pd.DataFrame:
    """Full drug-gene enrichment analysis orchestrator.

    Steps:
    1. Build drug gene sets from DrugTargetRecord + MAGMA gene results.
    2. Create MAGMA gene-set annotation file.
    3. Run MAGMA competitive gene-set analysis.
    4. Parse MAGMA results.
    5. Apply BH-FDR correction to MAGMA p-values.
    6. Optionally compute Wilcoxon AUC.
    7. Optionally compute permutation enrichment + separate FDR correction.
    8. Assemble final results with drug metadata.
    9. Write Parquet + JSON metadata sidecar.

    Args:
        gene_results_raw: Path to MAGMA .genes.raw file.
        gene_results_df: Parsed MAGMA gene results DataFrame.
        drug_targets: DrugTargetRecord DataFrame.
        config: DrugEnrichmentConfig from pipeline config.
        output_dir: Base output directory.
        study_name: Study identifier (for file naming).
        magma_binary: Optional explicit MAGMA binary path.

    Returns:
        DrugEnrichmentResult DataFrame.

    Raises:
        ValueError: If no drugs pass the min_genes_per_drug filter.
        RuntimeError: If MAGMA execution fails.
    """
    t0 = time.monotonic()

    de_dir = output_dir / "drug_enrichment"
    de_dir.mkdir(parents=True, exist_ok=True)

    resolved_binary = magma_binary if magma_binary is not None else detect_magma_binary()

    # 1. Build drug gene sets
    # gate on atc_min_genes_per_drug (inclusive); the
    #    headline subset is flagged downstream in assemble_drug_results.
    drug_gene_sets, filter_stats = build_drug_gene_sets(
        drug_targets=drug_targets,
        gene_results=gene_results_df,
        min_genes_per_drug=config.min_genes_per_drug,
        min_pchembl=config.min_pchembl,
        max_phase_filter=config.max_phase_filter,
        confidence_filter=config.confidence_filter,
        phase_filter_scope=config.phase_filter_scope,
        atc_min_genes_per_drug=config.atc_min_genes_per_drug,
    )

    # 1b. Optional PDSP signature collapse
    cluster_stats = None
    if config.pdsp_dedup_mode != "off":
        drug_gene_sets, cluster_stats = collapse_pdsp_clusters(
            drug_gene_sets=drug_gene_sets,
            drug_targets=drug_targets,
        )
        filter_stats["n_drugs_pre_dedup"] = cluster_stats["n_drugs_pre_dedup"]
        filter_stats["n_drugs_tested"] = cluster_stats["n_drugs_post_dedup"]

    if not drug_gene_sets:
        n_dropped_entrez = filter_stats.get("n_pairs_dropped_no_entrez", 0)
        entrez_hint = ""
        if n_dropped_entrez > 0:
            entrez_hint = (
                f" {n_dropped_entrez} interactions had no Entrez ID - "
                "check that reference.ncbi_gene_info and "
                "reference.ncbi_gene_history are set in configs/reference.yaml."
            )
        raise ValueError(
            f"No drugs passed the atc_min_genes_per_drug="
            f"{config.atc_min_genes_per_drug} filter "
            f"(min_genes_per_drug={config.min_genes_per_drug}, "
            f"min_pchembl={config.min_pchembl}, "
            f"max_phase_filter={config.max_phase_filter}, "
            f"phase_filter_scope={config.phase_filter_scope}, "
            f"confidence_filter={config.confidence_filter}). "
            f"Total drugs loaded: {drug_targets['drug_chembl_id'].nunique()}."
            f"{entrez_hint}"
        )

    # 2. Create gene-set file
    geneset_path = de_dir / f"{study_name}_drug_genesets.txt"
    create_drug_geneset_file(drug_gene_sets, geneset_path)

    # 3. Run MAGMA
    output_prefix = de_dir / f"{study_name}_drug_enrichment"
    gsa_out = run_magma_drug_enrichment(
        gene_results_raw=gene_results_raw,
        drug_geneset_file=geneset_path,
        output_prefix=output_prefix,
        magma_binary=resolved_binary,
    )

    # 4. Parse results (no FDR yet - applied post-assembly so the headline
    #    mask is computed from a single source of truth)
    magma_results = parse_magma_drug_results(gsa_out)

    # 5. Optional Wilcoxon AUC (independent of FDR)
    wilcoxon_results = None
    if config.include_wilcoxon_auc:
        wilcoxon_results = compute_wilcoxon_auc(
            gene_z_scores=gene_results_df["magma_z"],
            drug_gene_sets=drug_gene_sets,
            gene_entrez_ids=gene_results_df["gene_entrez_id"],
        )

    # 6. Optional permutation enrichment (no FDR yet - assigned alongside
    #    magma_fdr_q after assembly using the same headline mask)
    permutation_results = None
    if config.permutation_test:
        permutation_results = compute_permutation_enrichment(
            gene_z_scores=gene_results_df["magma_z"].values,
            drug_gene_sets=drug_gene_sets,
            gene_entrez_ids=gene_results_df["gene_entrez_id"].values,
            n_permutations=config.n_permutations,
            permutation_seed=config.permutation_seed,
        )

    # 7. Assemble final results (adds passes_headline_min_genes flag)
    results = assemble_drug_results(
        magma_results=magma_results,
        drug_targets=drug_targets,
        gene_results=gene_results_df,
        drug_gene_sets=drug_gene_sets,
        wilcoxon_results=wilcoxon_results,
        permutation_results=permutation_results,
        min_genes_per_drug=config.min_genes_per_drug,
    )

    # 7b. Centralised FDR scoping
    #     BH is applied to the headline subset only; sub-headline drugs
    #     receive valid magma_p (from MAGMA's competitive test) but
    #     magma_fdr_q = NaN and are excluded from headline outputs.
    headline_mask = results["passes_headline_min_genes"].values.astype(bool)

    magma_q = np.full(len(results), np.nan)
    if headline_mask.any():
        _, q_h, _, _ = multipletests(
            results.loc[headline_mask, "magma_p"].values, method=config.fdr_method
        )
        magma_q[headline_mask] = q_h
    results["magma_fdr_q"] = magma_q

    if config.permutation_test and "permutation_p" in results.columns:
        perm_q = np.full(len(results), np.nan)
        valid = headline_mask & results["permutation_p"].notna().values
        if valid.any():
            _, q_p, _, _ = multipletests(
                results.loc[valid, "permutation_p"].values, method=config.fdr_method
            )
            perm_q[valid] = q_p
        results["permutation_fdr_q"] = perm_q

    # 8. Write outputs
    parquet_path = de_dir / f"{study_name}_drug_enrichment.parquet"
    json_path = de_dir / f"{study_name}_drug_enrichment_metadata.json"

    target_genes_for_parquet = results["target_genes"].apply(json.dumps)
    parquet_df = results.copy()
    parquet_df["target_genes"] = target_genes_for_parquet
    parquet_df.to_parquet(parquet_path, index=False)

    n_sig = int((results["magma_fdr_q"] < config.fdr_threshold).sum())
    n_in_atc_pool = int(len(results))
    n_in_headline_pool = int(results["passes_headline_min_genes"].sum())
    top_drug = results.iloc[0]["drug_chembl_id"] if len(results) > 0 else None

    metadata = {
        "result_type": "drug_enrichment",
        "study": study_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "magma_version": _get_magma_version(resolved_binary),
        "parameters": {
            "min_genes_per_drug": config.min_genes_per_drug,
            "atc_min_genes_per_drug": config.atc_min_genes_per_drug,
            "min_pchembl": config.min_pchembl,
            "max_phase_filter": config.max_phase_filter,
            "phase_filter_scope": config.phase_filter_scope,
            "confidence_filter": config.confidence_filter,
            "pdsp_dedup_mode": config.pdsp_dedup_mode,
            "fdr_method": config.fdr_method,
            "include_wilcoxon_auc": config.include_wilcoxon_auc,
            "permutation_test": config.permutation_test,
            "n_permutations": config.n_permutations if config.permutation_test else None,
        },
        "summary": {
            "n_drugs_loaded": int(drug_targets["drug_chembl_id"].nunique()),
            "n_drugs_tested": filter_stats["n_drugs_tested"],
            "n_drugs_in_atc_pool": n_in_atc_pool,
            "n_drugs_in_headline_pool": n_in_headline_pool,
            "n_drugs_pre_dedup": filter_stats.get("n_drugs_pre_dedup", filter_stats["n_drugs_tested"]),
            "n_drugs_dropped_min_genes": filter_stats["n_drugs_dropped_min_genes"],
            "n_drugs_dropped_phase": filter_stats["n_drugs_dropped_phase"],
            "phase_filter_scope": config.phase_filter_scope,
            "n_pairs_dropped_pchembl": filter_stats["n_pairs_dropped_pchembl"],
            "n_pairs_dropped_confidence": filter_stats["n_pairs_dropped_confidence"],
            "n_pairs_dropped_no_entrez": filter_stats["n_pairs_dropped_no_entrez"],
            "n_pairs_dropped_not_in_magma": filter_stats["n_pairs_dropped_not_in_magma"],
            "n_pdsp_drugs_collapsed": cluster_stats["n_pdsp_drugs_collapsed"] if cluster_stats else 0,
            "n_significant_fdr05": n_sig,
            "top_drug": top_drug,
        },
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if cluster_stats and cluster_stats["cluster_map"]:
        sidecar_path = de_dir / f"{study_name}_pdsp_clusters.json"
        with open(sidecar_path, "w") as f:
            json.dump(cluster_stats["cluster_map"], f, indent=2)

    elapsed = time.monotonic() - t0
    logger.info(
        "Drug enrichment analysis completed in %.1fs. "
        "%d drugs in ATC pool / %d in headline pool, "
        "%d significant at FDR < %.2f.",
        elapsed,
        n_in_atc_pool,
        n_in_headline_pool,
        n_sig,
        config.fdr_threshold,
    )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drug-gene enrichment analysis")
    parser.add_argument("--gene-results-raw", type=Path, required=True,
                        help="Path to MAGMA .genes.raw file")
    parser.add_argument("--gene-results-parquet", type=Path, required=True,
                        help="Path to parsed MAGMA gene results Parquet")
    parser.add_argument("--drug-targets", type=Path, required=True,
                        help="Path to DrugTargetRecord Parquet")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to pipeline config YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    from repogen.config.loader import load_config

    pipeline_config = load_config(args.config)

    gene_results_df = pd.read_parquet(args.gene_results_parquet)
    drug_targets_df = pd.read_parquet(args.drug_targets)

    # Resolve MAGMA here rather than inside run_drug_enrichment: only the
    # full pipeline config knows the resource directory that holds the
    # binary fetched by `repogen setup-resources`.
    resolved_magma = detect_magma_binary(
        pipeline_config.magma.binary_path,
        resource_dir=pipeline_config.resource_dir,
    )

    run_drug_enrichment(
        gene_results_raw=args.gene_results_raw,
        gene_results_df=gene_results_df,
        drug_targets=drug_targets_df,
        config=pipeline_config.drug_enrichment,
        output_dir=args.output_dir,
        study_name=args.study_name,
        magma_binary=resolved_magma,
    )
