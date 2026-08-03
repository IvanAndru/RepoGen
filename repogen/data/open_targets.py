"""Post-analysis annotation via Open Targets Platform.

Queries the Open Targets GraphQL API to annotate gene-level or
drug-level results with target-disease association scores, tractability
assessments, and genetic evidence scores.

This module is optional - it degrades gracefully if the API is
unreachable, adding null columns and logging a warning.  It requires
internet access and should NOT be called inside Snakemake rules on
compute nodes without network access.  Use as a post-processing step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from repogen.utils.io import ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

# Open Targets Platform GraphQL API v4 (verified 2026-02)
_OT_API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_RETRY_LIMIT = 3
_RETRY_BACKOFF = 2.0
_REQUEST_TIMEOUT = 30

_DISEASE_ASSOCIATIONS_QUERY = """
query DiseaseAssociations($diseaseId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $diseaseId) {
    associatedTargets(page: {index: $index, size: $size}) {
      count
      rows {
        target {
          id
          approvedSymbol
        }
        score
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""

_TRACTABILITY_QUERY = """
query TargetTractability($ensemblIds: [String!]!) {
  targets(ensemblIds: $ensemblIds) {
    id
    tractability {
      label
      modality
      value
    }
  }
}
"""

_ASSOC_PAGE_SIZE = 500
_TRACTABILITY_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def annotate_with_open_targets(
    results_df: pd.DataFrame,
    disease_efo_id: str,
    gene_column: str = "gene_ensembl_id",
    use_api: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Add Open Targets annotation columns to a results DataFrame.

    Columns added:
        * ``ot_association_score`` - overall target-disease score (0-1).
        * ``ot_genetic_score`` - genetic evidence score.
        * ``ot_known_drug_score`` - known drug evidence score.
        * ``ot_tractability_sm`` - small molecule tractability flag.
        * ``ot_tractability_ab`` - antibody tractability flag.

    If the API is unavailable, adds null columns and logs a warning
    (does not raise).

    Args:
        results_df: DataFrame with gene identifiers.
        disease_efo_id: EFO disease ID (e.g. ``"EFO_0003761"`` for MDD).
        gene_column: Column containing Ensembl gene IDs.
        use_api: If ``True``, query the GraphQL API.  If ``False``,
            attempt to load from cache only.
        cache_dir: Directory for caching API responses.

    Returns:
        A copy of *results_df* with the five new columns appended.
    """
    result = results_df.copy()

    ot_cols = {
        "ot_association_score": float,
        "ot_genetic_score": float,
        "ot_known_drug_score": float,
        "ot_tractability_sm": bool,
        "ot_tractability_ab": bool,
    }

    if gene_column not in result.columns:
        logger.warning(
            "Column '%s' not in DataFrame; skipping Open Targets annotation",
            gene_column,
        )
        for col, dtype in ot_cols.items():
            result[col] = pd.NA if dtype is float else False
        return result

    ensembl_ids = (
        result[gene_column]
        .dropna()
        .unique()
        .tolist()
    )
    ensembl_ids = [eid for eid in ensembl_ids if str(eid).startswith("ENSG")]

    if not ensembl_ids:
        logger.warning("No valid Ensembl IDs found; skipping Open Targets")
        for col, dtype in ot_cols.items():
            result[col] = pd.NA if dtype is float else False
        return result

    logger.info(
        "Querying Open Targets for %d genes, disease=%s",
        len(ensembl_ids), disease_efo_id,
    )

    cached = _load_cache(cache_dir, disease_efo_id) if cache_dir else {}

    uncached_ids = [eid for eid in ensembl_ids if eid not in cached]

    if uncached_ids and use_api:
        api_results = query_open_targets_api(uncached_ids, disease_efo_id)
        cached.update(api_results)
        if cache_dir:
            _save_cache(cache_dir, disease_efo_id, cached)
    elif uncached_ids:
        logger.info(
            "%d Ensembl IDs not in cache and API disabled", len(uncached_ids)
        )

    scores = []
    genetic = []
    drug_scores = []
    tract_sm = []
    tract_ab = []

    for eid in result[gene_column]:
        info = cached.get(eid, {}) if pd.notna(eid) else {}
        scores.append(info.get("association_score"))
        genetic.append(info.get("genetic_score"))
        drug_scores.append(info.get("known_drug_score"))
        tract_sm.append(info.get("tractability_sm", False))
        tract_ab.append(info.get("tractability_ab", False))

    result["ot_association_score"] = pd.array(scores, dtype=pd.Float64Dtype())
    result["ot_genetic_score"] = pd.array(genetic, dtype=pd.Float64Dtype())
    result["ot_known_drug_score"] = pd.array(drug_scores, dtype=pd.Float64Dtype())
    result["ot_tractability_sm"] = tract_sm
    result["ot_tractability_ab"] = tract_ab

    n_annotated = sum(1 for s in scores if s is not None)
    logger.info(
        "Open Targets annotation: %d / %d genes annotated (%.1f%%)",
        n_annotated, len(ensembl_ids),
        100 * n_annotated / max(len(ensembl_ids), 1),
    )
    return result


# ---------------------------------------------------------------------------
# API query - two-step: disease associations + tractability
# ---------------------------------------------------------------------------


def query_open_targets_api(
    ensembl_ids: list[str],
    disease_efo_id: str,
) -> dict[str, dict]:
    """Two-step GraphQL query for target-disease associations and tractability.

    Step 1: Paginate ``disease.associatedTargets`` to get association and
    datatype scores, then filter in Python to our Ensembl ID set.

    Step 2: Batch-query ``targets(ensemblIds: [...])`` for tractability data.

    Args:
        ensembl_ids: Ensembl gene IDs to query.
        disease_efo_id: EFO disease identifier.

    Returns:
        Mapping of Ensembl ID -> annotation dict with keys
        ``association_score``, ``genetic_score``, ``known_drug_score``,
        ``tractability_sm``, ``tractability_ab``.
    """
    results: dict[str, dict] = {
        eid: {
            "association_score": None,
            "genetic_score": None,
            "known_drug_score": None,
            "tractability_sm": False,
            "tractability_ab": False,
        }
        for eid in ensembl_ids
    }

    ensembl_set = set(ensembl_ids)

    assoc_data = _query_disease_associations(disease_efo_id, ensembl_set)
    for eid, scores in assoc_data.items():
        if eid in results:
            results[eid].update(scores)

    tract_data = _query_tractability(ensembl_ids)
    for eid, tract in tract_data.items():
        if eid in results:
            results[eid]["tractability_sm"] = tract.get("tractability_sm", False)
            results[eid]["tractability_ab"] = tract.get("tractability_ab", False)

    return results


def _query_disease_associations(
    disease_efo_id: str,
    ensembl_set: set[str],
) -> dict[str, dict]:
    """Paginate disease.associatedTargets and filter to our gene set."""
    results: dict[str, dict] = {}
    page_index = 0
    total_count = None

    while True:
        payload = {
            "query": _DISEASE_ASSOCIATIONS_QUERY,
            "variables": {
                "diseaseId": disease_efo_id,
                "index": page_index,
                "size": _ASSOC_PAGE_SIZE,
            },
        }

        data = _execute_graphql(payload)
        if data is None:
            break

        disease = data.get("data", {}).get("disease")
        if disease is None:
            logger.warning(
                "Open Targets returned null for disease '%s' - "
                "check if the EFO ID is valid",
                disease_efo_id,
            )
            break

        assoc = disease.get("associatedTargets", {})
        if total_count is None:
            total_count = assoc.get("count", 0)
            logger.info(
                "Open Targets disease '%s': %d total associated targets",
                disease_efo_id, total_count,
            )

        rows = assoc.get("rows", [])
        if not rows:
            break

        for row in rows:
            target_info = row.get("target", {})
            eid = (target_info or {}).get("id", "")
            if eid not in ensembl_set:
                continue

            entry: dict = {
                "association_score": row.get("score"),
                "genetic_score": None,
                "known_drug_score": None,
            }

            for dtype_score in (row.get("datatypeScores") or []):
                score_id = str(dtype_score.get("id", "")).lower()
                score_val = dtype_score.get("score")
                if "genetic" in score_id or "ot_genetics" in score_id:
                    entry["genetic_score"] = score_val
                elif "drug" in score_id or "chembl" in score_id or "known" in score_id:
                    entry["known_drug_score"] = score_val

            results[eid] = entry

        page_index += 1
        if page_index * _ASSOC_PAGE_SIZE >= total_count:
            break

        time.sleep(0.2)

    logger.info(
        "Disease associations: %d / %d of our targets found",
        len(results), len(ensembl_set),
    )
    return results


def _query_tractability(ensembl_ids: list[str]) -> dict[str, dict]:
    """Batch-query targets for tractability data."""
    results: dict[str, dict] = {}

    for i in range(0, len(ensembl_ids), _TRACTABILITY_BATCH_SIZE):
        batch = ensembl_ids[i : i + _TRACTABILITY_BATCH_SIZE]

        payload = {
            "query": _TRACTABILITY_QUERY,
            "variables": {"ensemblIds": batch},
        }

        data = _execute_graphql(payload)
        if data is None:
            continue

        targets = data.get("data", {}).get("targets", [])
        for target in (targets or []):
            eid = target.get("id", "")
            tract_sm = False
            tract_ab = False

            for t in (target.get("tractability") or []):
                modality = str(t.get("modality", "")).upper()
                value = t.get("value", False)
                if modality == "SM" and value:
                    tract_sm = True
                elif modality == "AB" and value:
                    tract_ab = True

            results[eid] = {
                "tractability_sm": tract_sm,
                "tractability_ab": tract_ab,
            }

        if i + _TRACTABILITY_BATCH_SIZE < len(ensembl_ids):
            time.sleep(0.2)

    return results


def _execute_graphql(payload: dict) -> Optional[dict]:
    """Execute a single GraphQL request with retry logic."""
    for attempt in range(1, _RETRY_LIMIT + 1):
        try:
            resp = requests.post(
                _OT_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as exc:
            if attempt < _RETRY_LIMIT:
                wait = _RETRY_BACKOFF ** attempt
                logger.warning(
                    "Open Targets API attempt %d/%d failed: %s. "
                    "Retrying in %.1fs...",
                    attempt, _RETRY_LIMIT, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Open Targets API failed after %d attempts: %s. "
                    "Returning empty results for this request.",
                    _RETRY_LIMIT, exc,
                )
                return None

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "Open Targets API response parsing failed: %s", exc
            )
            return None

    return None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, disease_efo_id: str) -> Path:
    """Return the cache file path for a given disease."""
    safe_id = disease_efo_id.replace(":", "_").replace("/", "_")
    return Path(cache_dir) / f"ot_cache_{safe_id}.json"


def _load_cache(cache_dir: Optional[Path], disease_efo_id: str) -> dict:
    """Load cached Open Targets results if available."""
    if cache_dir is None:
        return {}
    path = _cache_path(cache_dir, disease_efo_id)
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load OT cache: %s", exc)
        return {}


def _save_cache(cache_dir: Path, disease_efo_id: str, data: dict) -> None:
    """Save Open Targets results to cache."""
    cache_dir = ensure_directory(Path(cache_dir))
    path = _cache_path(cache_dir, disease_efo_id)
    try:
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Cached %d OT results to %s", len(data), path)
    except OSError as exc:
        logger.warning("Failed to save OT cache: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate gene results with Open Targets data"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input results file (Parquet or CSV)",
    )
    parser.add_argument(
        "--disease-efo-id", type=str, required=True,
        help="EFO disease ID (e.g. EFO_0003761)",
    )
    parser.add_argument(
        "--gene-column", type=str, default="gene_ensembl_id",
        help="Column with Ensembl gene IDs",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory for caching API responses",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        input_df = pd.read_parquet(input_path)
    else:
        input_df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(input_df), input_path)

    result_df = annotate_with_open_targets(
        results_df=input_df,
        disease_efo_id=args.disease_efo_id,
        gene_column=args.gene_column,
        cache_dir=args.cache_dir,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Saved %d rows to %s", len(result_df), output_path)
