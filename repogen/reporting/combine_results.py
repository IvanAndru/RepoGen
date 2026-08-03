"""RepoGen reporting - combine available results from all branches."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from repogen.utils.logging import setup_logging

if TYPE_CHECKING:
    from repogen.config.schema import PipelineConfig

logger = setup_logging(__name__)


@dataclass
class CombinedResults:
    """Container for all available pipeline results.

    Any field except study_name and metadata may be None,
    indicating that branch/analysis was not run.
    """

    study_name: str
    branches_present: list[str] = field(default_factory=list)
    gene_results: pd.DataFrame | None = None
    pathway_results: pd.DataFrame | None = None
    drug_enrichment: pd.DataFrame | None = None
    atc_enrichment: pd.DataFrame | None = None
    spredixcan_meta: pd.DataFrame | None = None
    neg_correlation_summary: pd.DataFrame | None = None
    mr_results: pd.DataFrame | None = None
    mr_drug_matches: pd.DataFrame | None = None
    mr_target_verdicts: pd.DataFrame | None = None
    drug_overlap: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


RESULT_FILE_MAP: dict[str, dict[str, str | None]] = {
    "gene": {
        "subdir": "magma",
        "pattern": "*gene_results.parquet",
        "glob_fallback": "*genes*.parquet",
        "branch": "magma",
        "combined_field": "gene_results",
    },
    "pathway": {
        "subdir": "magma",
        "pattern": "*pathway_results.parquet",
        "glob_fallback": "*pathway*.parquet",
        "branch": "magma",
        "combined_field": "pathway_results",
    },
    "drug": {
        "subdir": "drug_enrichment",
        "pattern": "*drug_enrichment.parquet",
        "glob_fallback": "*drug_enrichment*.parquet",
        "branch": "magma",
        "combined_field": "drug_enrichment",
    },
    "atc": {
        "subdir": "atc_enrichment",
        "pattern": "*atc_enrichment_results.parquet",
        "glob_fallback": "*atc_enrichment*.parquet",
        "branch": "magma",
        "combined_field": "atc_enrichment",
    },
    "spredixcan": {
        "subdir": "spredixcan",
        "pattern": "*meta_analysis.parquet",
        "glob_fallback": "*spredixcan_meta*.parquet",
        "branch": "neg_correlation",
        "combined_field": "spredixcan_meta",
    },
    "spredixcan_per_tissue": {
        "subdir": "spredixcan",
        "pattern": "*spredixcan_per_tissue.parquet",
        "glob_fallback": "*per_tissue*.parquet",
        "branch": "neg_correlation",
        "combined_field": None,
    },
    "correlation": {
        "subdir": "negative_correlation",
        "pattern": "*drug_summary.parquet",
        "glob_fallback": "*drug_summary*.parquet",
        "branch": "neg_correlation",
        "combined_field": "neg_correlation_summary",
    },
    "correlation_per_tissue": {
        "subdir": "negative_correlation",
        "pattern": "*per_tissue_results.parquet",
        "glob_fallback": "*per_tissue*.parquet",
        "branch": "neg_correlation",
        "combined_field": None,
    },
    "mr": {
        "subdir": "mr",
        "pattern": "*mr_results.parquet",
        "glob_fallback": "*mr_results*.parquet",
        "branch": "mr",
        "combined_field": "mr_results",
    },
    "mr_drugs": {
        "subdir": "mr",
        "pattern": "*mr_drug_matches.parquet",
        "glob_fallback": "*drug_match*.parquet",
        "branch": "mr",
        "combined_field": "mr_drug_matches",
    },
    "mr_verdicts": {
        "subdir": "mr",
        "pattern": "*mr_target_verdicts.parquet",
        "glob_fallback": "*target_verdict*.parquet",
        "branch": "mr",
        "combined_field": "mr_target_verdicts",
    },
}


def _discover_file(
    results_dir: Path, study_name: str, result_type: str
) -> Path | None:
    """Find a result file. Returns None if not found."""
    info = RESULT_FILE_MAP[result_type]
    subdir = results_dir / study_name / info["subdir"]
    if not subdir.is_dir():
        return None
    matches = list(subdir.glob(info["pattern"]))
    if not matches:
        matches = list(subdir.glob(info["glob_fallback"]))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple files match %s in %s; using first: %s",
            result_type,
            subdir,
            matches[0].name,
        )
    return matches[0]


def combine_results(
    results_dir: Path,
    study_name: str,
    config: PipelineConfig | None = None,
) -> CombinedResults:
    """Discover available results under results_dir, load what exists,
    skip what is missing, and compute cross-branch drug overlap.

    Args:
        results_dir: Root results directory (e.g., ``Path("results")``).
        study_name: Study identifier (e.g., ``"MDD_PGC3"``).
        config: Optional pipeline config for parameter recording.

    Returns:
        CombinedResults with all discovered results loaded.

    Raises:
        FileNotFoundError: If *results_dir* does not exist.
        ValueError: If no results are found for any branch.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    combined = CombinedResults(study_name=study_name)
    files_loaded: dict[str, str] = {}
    branch_set: set[str] = set()

    for result_type, info in RESULT_FILE_MAP.items():
        if info["combined_field"] is None:
            continue

        fpath = _discover_file(results_dir, study_name, result_type)
        if fpath is None:
            logger.debug("Not found: %s", result_type)
            continue

        try:
            df = pd.read_parquet(fpath)
        except (pd.errors.EmptyDataError, OSError, ValueError) as exc:
            logger.warning(
                "Failed to read %s (%s): %s - treating as missing.",
                result_type,
                fpath,
                exc,
            )
            continue

        setattr(combined, info["combined_field"], df)
        files_loaded[result_type] = str(fpath)
        branch_set.add(info["branch"])
        logger.info("Loaded %s from %s (%d rows)", result_type, fpath.name, len(df))

    combined.branches_present = sorted(branch_set)

    if not combined.branches_present:
        raise ValueError(f"No results found under {results_dir / study_name}")

    drug_branches = 0
    if combined.drug_enrichment is not None:
        drug_branches += 1
    if combined.neg_correlation_summary is not None:
        drug_branches += 1
    if combined.mr_drug_matches is not None:
        drug_branches += 1

    if drug_branches >= 2:
        combined.drug_overlap = _compute_drug_overlap(
            drug_enrichment=combined.drug_enrichment,
            neg_correlation_summary=combined.neg_correlation_summary,
            mr_drug_matches=combined.mr_drug_matches,
        )

    # TODO: Open Targets annotation (v2.0)

    config_dict: dict[str, Any] = {}
    if config is not None:
        try:
            config_dict = config.model_dump(mode="json")
        except (AttributeError, TypeError):
            pass

    combined.metadata = {
        "study_name": study_name,
        "branches_present": combined.branches_present,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_loaded": files_loaded,
        "config": config_dict,
    }

    logger.info(
        "Combined results for '%s': branches=%s, files=%d",
        study_name,
        combined.branches_present,
        len(files_loaded),
    )
    return combined


def _compute_drug_overlap(
    drug_enrichment: pd.DataFrame | None,
    neg_correlation_summary: pd.DataFrame | None,
    mr_drug_matches: pd.DataFrame | None,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame | None:
    """Identify drugs found by >=2 branches.

    Join keys:
        - Primary: drug_chembl_id (exact match)
        - Fallback: normalised drug_name (lowercase, stripped)

    A drug is "convergent" if identified by >=2 of:
        - MAGMA enrichment: magma_fdr_q < fdr_threshold
        - Negative correlation: n_tissues_fdr_significant > 0
        - MR drug match: any confidence_tier

    Returns:
        DataFrame with overlap columns, sorted by n_branches descending.
        Returns None if <2 branches have drug-level results.
    """
    branch_dfs: list[tuple[str, pd.DataFrame]] = []

    if drug_enrichment is not None and "magma_fdr_q" in drug_enrichment.columns:
        mask = drug_enrichment["magma_fdr_q"] < fdr_threshold
        if "passes_headline_min_genes" in drug_enrichment.columns:
            mask = mask & drug_enrichment["passes_headline_min_genes"].fillna(False).astype(bool)
        sig = drug_enrichment[mask].copy()
        if not sig.empty:
            cols = ["drug_name", "drug_chembl_id", "magma_fdr_q"]
            avail = [c for c in cols if c in sig.columns]
            branch_dfs.append(("magma", sig[avail].copy()))

    if (
        neg_correlation_summary is not None
        and "n_tissues_fdr_significant" in neg_correlation_summary.columns
    ):
        sig = neg_correlation_summary[
            neg_correlation_summary["n_tissues_fdr_significant"] > 0
        ].copy()
        if not sig.empty:
            cols = ["drug_name", "drug_chembl_id", "best_spearman_rho"]
            avail = [c for c in cols if c in sig.columns]
            branch_dfs.append(("neg_corr", sig[avail].copy()))

    if mr_drug_matches is not None:
        mr = mr_drug_matches.copy()
        if not mr.empty:
            cols = ["drug_name", "drug_chembl_id", "confidence_tier"]
            avail = [c for c in cols if c in mr.columns]
            branch_dfs.append(("mr", mr[avail].copy()))

    if len(branch_dfs) < 2:
        return None

    for label, df in branch_dfs:
        if "drug_name" in df.columns:
            df["drug_name_norm"] = df["drug_name"].str.lower().str.strip()
        else:
            df["drug_name_norm"] = pd.NA

    all_drugs: list[dict[str, Any]] = []
    for label, df in branch_dfs:
        for row in df.itertuples(index=False):
            all_drugs.append({
                "drug_name": getattr(row, "drug_name", None),
                "drug_chembl_id": getattr(row, "drug_chembl_id", None),
                "drug_name_norm": getattr(row, "drug_name_norm", None),
                "branch": label,
                "magma_fdr_q": getattr(row, "magma_fdr_q", None),
                "neg_corr_best_rho": getattr(row, "best_spearman_rho", None),
                "mr_confidence_tier": getattr(row, "confidence_tier", None),
            })

    if not all_drugs:
        return pd.DataFrame(
            columns=[
                "drug_name",
                "drug_chembl_id",
                "in_magma",
                "in_neg_corr",
                "in_mr",
                "n_branches",
                "magma_fdr_q",
                "neg_corr_best_rho",
                "mr_confidence_tier",
            ]
        )

    by_key: dict[str, dict[str, Any]] = {}
    for d in all_drugs:
        chembl = d.get("drug_chembl_id")
        name_norm = d.get("drug_name_norm")
        branch = d["branch"]

        key = None
        if chembl and pd.notna(chembl) and chembl in by_key:
            key = chembl
        elif name_norm and pd.notna(name_norm):
            for existing_key, existing in by_key.items():
                if existing.get("drug_name_norm") == name_norm:
                    key = existing_key
                    break

        if key is None:
            key = chembl if chembl and pd.notna(chembl) else (name_norm or str(len(by_key)))
            by_key[key] = {
                "drug_name": d["drug_name"],
                "drug_chembl_id": d["drug_chembl_id"],
                "drug_name_norm": name_norm,
                "in_magma": False,
                "in_neg_corr": False,
                "in_mr": False,
                "magma_fdr_q": None,
                "neg_corr_best_rho": None,
                "mr_confidence_tier": None,
            }

        entry = by_key[key]
        if branch == "magma":
            entry["in_magma"] = True
            if d.get("magma_fdr_q") is not None:
                entry["magma_fdr_q"] = d["magma_fdr_q"]
        elif branch == "neg_corr":
            entry["in_neg_corr"] = True
            if d.get("neg_corr_best_rho") is not None:
                entry["neg_corr_best_rho"] = d["neg_corr_best_rho"]
        elif branch == "mr":
            entry["in_mr"] = True
            if d.get("mr_confidence_tier") is not None:
                entry["mr_confidence_tier"] = d["mr_confidence_tier"]

        if d.get("drug_name") and not entry.get("drug_name"):
            entry["drug_name"] = d["drug_name"]
        if d.get("drug_chembl_id") and not entry.get("drug_chembl_id"):
            entry["drug_chembl_id"] = d["drug_chembl_id"]

    result_records = list(by_key.values())
    for rec in result_records:
        rec["n_branches"] = (
            int(rec["in_magma"]) + int(rec["in_neg_corr"]) + int(rec["in_mr"])
        )
        rec.pop("drug_name_norm", None)

    result_df = pd.DataFrame(result_records)
    result_df = result_df[result_df["n_branches"] >= 2]

    out_cols = [
        "drug_name",
        "drug_chembl_id",
        "in_magma",
        "in_neg_corr",
        "in_mr",
        "n_branches",
        "magma_fdr_q",
        "neg_corr_best_rho",
        "mr_confidence_tier",
    ]
    for c in out_cols:
        if c not in result_df.columns:
            result_df[c] = None

    result_df = result_df[out_cols].sort_values(
        ["n_branches", "drug_name"],
        ascending=[False, True],
        na_position="last",
    )

    return result_df.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine RepoGen analysis results from all branches."
    )
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))

    args = parser.parse_args()
    combined = combine_results(
        results_dir=args.results_dir,
        study_name=args.study_name,
    )
    logger.info(
        "Branches present: %s, drug overlap: %s",
        combined.branches_present,
        combined.drug_overlap is not None,
    )
