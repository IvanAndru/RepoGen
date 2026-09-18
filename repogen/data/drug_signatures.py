"""LINCS L1000 drug expression signature extraction.

Extracts and aggregates drug gene expression signatures from LINCS L1000
Level 5 data.  Matches LINCS compounds to ChEMBL drugs via InChIKey
(primary), PubChem CID (fallback), or normalised name (last resort).
Outputs the :class:`DrugSignatureRecord` schema.

The correlation analysis logic stays in
``repogen.analysis.negative_correlation``; this module is purely a data
extraction and preparation concern.

Dependencies
~~~~~~~~~~~~
* ``cmapPy`` - for GCTX reading.  Optional; install via
  ``pip install repogen[lincs]``.
* ``rapidfuzz`` - for fuzzy name matching fallback.  Optional; install
  via ``pip install repogen[lincs]``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from repogen.data.schemas import validate_dataframe
from repogen.utils.io import check_file_exists
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Neural cell-line ontology (Phase-0 census validated)
# ---------------------------------------------------------------------------
#
# These tuples are the source of truth for neural-lineage cell-line tokens.
# They are *module constants* (not user config) because they encode a
# biological ontology, not a user preference; the actual behaviour knob is
# ``DrugSignaturesConfig.neural_cell_lines`` which the model validator
# derives from these constants + ``include_neural_tumor_cell_lines``.
#
# Values validated against the LINCS L1000 Level 5 GCTX corpus in Phase-0
# (see ``scripts/lincs_cell_line_census.py``). Tokens with zero
# observed profiles were excluded from defaults.
#
# Primary neural (iPSC-derived and reprogrammed):
#   NEU     (3,065 profiles)  iPSC-derived neurons
#   NPC     (4,128 profiles)  iPSC-derived neural progenitor cells
#   ASC     (2,526 profiles)  iPSC-derived astrocytes
#   FIBRNPC (1,312 profiles)  fibroblast-reprogrammed NPCs
#
# Neural-lineage cancer (sparse but real):
#   LN229   (   12 profiles)  glioma cell line
#
# Not included in current defaults (zero profiles in shipped corpus):
#   MNEU   (motor neurons)      - evaluated, not present
#   SHSY5Y (neuroblastoma)      - evaluated, not present
# If a future LINCS release adds these tokens, users can extend via
# ``drug_signatures.neural_cell_lines`` config override without a code
# change.
NEURAL_PRIMARY_CELL_LINES: tuple[str, ...] = ("NEU", "NPC", "ASC", "FIBRNPC")
NEURAL_TUMOR_CELL_LINES: tuple[str, ...] = ("LN229",)


def resolve_neural_cell_lines_from_yaml(
    ds_cfg: Optional[dict],
) -> list[str]:
    """Single source of truth for
    ``drug_signatures.neural_cell_lines`` derivation.

    Consumed by both:

    * :class:`DrugSignaturesConfig` Pydantic ``@model_validator(mode="after")``
      (the config-validated path), and
    * The Snakemake ``extract_drug_signatures`` rule (which reads raw YAML
      and bypasses Pydantic).

    Resolution logic (matches the Pydantic validator exactly):

    1. If ``neural_cell_lines`` is explicitly non-empty, normalise it
       (upper + strip + dedupe + drop empties) and return it.  A user
       override *bypasses* ``include_neural_tumor_cell_lines`` - the
       switch is only consulted for the derived default.
    2. Otherwise derive from ``include_neural_tumor_cell_lines``
       (default ``True``): primary + tumor when True, primary-only when
       False.

    Args:
        ds_cfg: The ``drug_signatures`` sub-mapping from a raw YAML
            config, or ``None`` (equivalent to an empty mapping).

    Returns:
        Sorted, deduplicated, uppercased list of neural cell-line tokens.
    """
    ds_cfg = ds_cfg or {}
    user_list = ds_cfg.get("neural_cell_lines")
    if user_list:
        return sorted({
            str(c).upper().strip()
            for c in user_list
            if c is not None and str(c).strip()
        })
    include_tumor = ds_cfg.get("include_neural_tumor_cell_lines", True)
    base: list[str] = list(NEURAL_PRIMARY_CELL_LINES)
    if include_tumor:
        base.extend(NEURAL_TUMOR_CELL_LINES)
    return sorted(set(base))


def _normalize_pubchem_cid(val) -> str | None:
    """Normalize a PubChem CID to a clean integer string.

    Handles float strings (``"6005.0"``), int strings (``"6005"``),
    numeric types, and ``NaN``/``None``/empty -> ``None``.
    """
    if val is None:
        return None
    if isinstance(val, float):
        if np.isnan(val):
            return None
        return str(int(val))
    s = str(val).strip()
    if not s or s in ("nan", "None", "<NA>", "NaN"):
        return None
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return None


def _import_cmappy():
    """Lazy import of cmapPy with a clear error if missing."""
    try:
        from cmapPy.pandasGEXpress import parse
        return parse
    except ImportError:
        raise ImportError(
            "cmapPy is required for LINCS L1000 data extraction. "
            "Install with: pip install repogen[lincs]"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_drug_signatures(
    gctx_path: Path,
    compound_metadata: Path,
    gene_metadata: Path,
    drug_targets: Optional[pd.DataFrame] = None,
    aggregation: str = "consensus",
    min_profiles: int = 3,
    preferred_dose: Optional[str] = "10 µM",
    preferred_time: Optional[str] = "24 h",
    cell_line_weighting: str = "uniform",
    neural_cell_lines: Optional[list[str]] = None,
    neural_weight: float = 3.0,
    min_neural_profiles: Optional[int] = None,
) -> pd.DataFrame:
    """Extract and aggregate drug expression signatures from LINCS L1000.

    Args:
        gctx_path: Path to LINCS L1000 Level 5 GCTX file (~18 GB).
        compound_metadata: Path to LINCS Drug Repurposing Hub metadata
            CSV (contains InChIKey, PubChem CID).
        gene_metadata: Path to LINCS gene metadata file
            (maps probe IDs to Entrez Gene IDs).
        drug_targets: :class:`DrugTargetRecord` DataFrame from
            :func:`drug_loader.load_drug_targets`.  Used for InChIKey
            matching.  If ``None``, only name matching is performed.
        aggregation: Profile aggregation strategy:
            ``"consensus"`` (median Z across all profiles),
            ``"best_dose"`` (filter to preferred dose/time, then median),
            ``"per_condition"`` (keep separate signatures per condition).
        min_profiles: Minimum number of profiles to aggregate per drug.
        preferred_dose: Target dose for ``"best_dose"`` aggregation.
        preferred_time: Target time point for ``"best_dose"`` aggregation.

    Returns:
        :class:`DrugSignatureRecord` DataFrame.

    Raises:
        ImportError: If ``cmapPy`` is not installed.
        FileNotFoundError: If required files are missing.
        RuntimeError: If no drug signatures can be extracted.
    """
    parse_gctx = _import_cmappy()

    gctx_path = check_file_exists(gctx_path, label="LINCS L1000 GCTX")
    compound_metadata = check_file_exists(
        compound_metadata, label="LINCS compound metadata"
    )
    gene_metadata = check_file_exists(gene_metadata, label="LINCS gene metadata")

    logger.info("Loading LINCS compound metadata: %s", compound_metadata)
    lincs_meta = pd.read_csv(compound_metadata, low_memory=False)
    logger.info("LINCS metadata: %d compounds", len(lincs_meta))

    logger.info("Loading LINCS gene metadata: %s", gene_metadata)
    gene_meta = _load_gene_metadata(gene_metadata)
    gene_ids = gene_meta["gene_id"].tolist()
    logger.info("LINCS gene metadata: %d genes", len(gene_ids))

    if drug_targets is not None and not drug_targets.empty:
        matched = match_lincs_to_chembl(lincs_meta, drug_targets)
    else:
        logger.warning(
            "No drug_targets DataFrame provided - using name-only matching. "
            "Results will lack ChEMBL IDs and match confidence will be 'name' for all drugs."
        )
        matched = _name_only_matching(lincs_meta)

    if matched.empty:
        raise RuntimeError(
            "No LINCS compounds could be matched to drug targets. "
            "Check compound metadata and drug target files."
        )

    logger.info("Matched %d LINCS compounds to known drugs", len(matched))

    pert_ids = matched["pert_id"].unique().tolist()
    logger.info(
        "Extracting expression profiles for %d perturbagens from GCTX...",
        len(pert_ids),
    )
    profiles = _extract_profiles_chunked(
        parse_gctx, gctx_path, pert_ids, gene_ids,
    )
    logger.info("Extracted profiles for %d drugs", len(profiles))

    signatures = aggregate_signatures(
        profiles=profiles,
        method=aggregation,
        min_profiles=min_profiles,
        preferred_dose=preferred_dose,
        preferred_time=preferred_time,
        cell_line_weighting=cell_line_weighting,
        neural_cell_lines=neural_cell_lines,
        neural_weight=neural_weight,
        min_neural_profiles=min_neural_profiles,
    )
    logger.info("Aggregated %d drug signatures", len(signatures))

    result = _build_signature_records(signatures, matched, gene_ids)

    if result.empty:
        raise RuntimeError(
            "No drug signatures survived aggregation. "
            "Try lowering min_profiles or using a different aggregation strategy."
        )

    errors = validate_dataframe(result, "DrugSignatureRecord")
    if errors:
        logger.warning(
            "DrugSignatureRecord validation: %d issue(s)", len(errors)
        )

    logger.info(
        "Drug signature extraction complete: %d signatures, "
        "%d genes per signature",
        len(result),
        len(gene_ids),
    )
    return result


# ---------------------------------------------------------------------------
# Drug matching
# ---------------------------------------------------------------------------


def match_lincs_to_chembl(
    lincs_metadata: pd.DataFrame,
    drug_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Match LINCS compounds to ChEMBL drugs.

    Priority:
        1. InChIKey exact match (highest confidence).
        2. PubChem CID match (high confidence).
        3. Normalised name match via ``rapidfuzz`` (low confidence).

    Args:
        lincs_metadata: LINCS Drug Repurposing Hub metadata.
        drug_targets: :class:`DrugTargetRecord` DataFrame.

    Returns:
        DataFrame with columns: ``pert_id``, ``drug_name``,
        ``drug_inchikey``, ``drug_chembl_id``, ``match_confidence``.
    """
    lincs_cols = _detect_lincs_metadata_columns(lincs_metadata)
    if lincs_cols is None:
        logger.warning("Could not detect LINCS metadata columns")
        return pd.DataFrame()

    pert_col = lincs_cols["pert_id"]
    name_col = lincs_cols["name"]
    inchikey_col = lincs_cols.get("inchikey")
    pubchem_col = lincs_cols.get("pubchem_cid")

    all_parts: list[pd.DataFrame] = []
    matched_pert_ids: set[str] = set()

    lincs_work = lincs_metadata.copy()
    lincs_work["_pert_id"] = lincs_work[pert_col].astype(str).str.strip()
    lincs_work["_drug_name"] = lincs_work[name_col].astype(str)

    # --- Pass 1: InChIKey (vectorised merge) ---
    n_inchikey = 0
    if inchikey_col and inchikey_col in lincs_work.columns and "drug_inchikey" in drug_targets.columns:
        chembl_ik = (
            drug_targets[["drug_inchikey", "drug_chembl_id"]]
            .dropna(subset=["drug_inchikey"])
            .drop_duplicates(subset=["drug_inchikey"])
        )
        ik_merged = lincs_work.merge(
            chembl_ik,
            left_on=inchikey_col,
            right_on="drug_inchikey",
            how="inner",
        )
        ik_merged = ik_merged.drop_duplicates(subset=["_pert_id"])
        if not ik_merged.empty:
            ik_result = pd.DataFrame({
                "pert_id": ik_merged["_pert_id"].values,
                "drug_name": ik_merged["_drug_name"].values,
                "drug_inchikey": ik_merged[inchikey_col].values,
                "drug_chembl_id": ik_merged["drug_chembl_id"].values,
                "match_confidence": "inchikey",
            })
            all_parts.append(ik_result)
            matched_pert_ids.update(ik_result["pert_id"])
            n_inchikey = len(ik_result)

    logger.info("InChIKey matching: %d compounds matched", n_inchikey)

    # --- Pass 2: PubChem CID (vectorised merge, unmatched only) ---
    n_pubchem = 0
    has_cid_col = "drug_pubchem_cid" in drug_targets.columns or "_pubchem_cid_all" in drug_targets.columns
    if pubchem_col and pubchem_col in lincs_work.columns and has_cid_col:
        unmatched = lincs_work[~lincs_work["_pert_id"].isin(matched_pert_ids)]
        unmatched_str = unmatched.copy()
        unmatched_str["_pc_norm"] = unmatched_str[pubchem_col].apply(_normalize_pubchem_cid)
        unmatched_str = unmatched_str[unmatched_str["_pc_norm"].notna()]

        if "_pubchem_cid_all" in drug_targets.columns:
            rows: list[dict] = []
            for _, row in drug_targets[["drug_chembl_id", "_pubchem_cid_all"]].drop_duplicates("drug_chembl_id").iterrows():
                cids = row["_pubchem_cid_all"]
                if cids is None:
                    continue
                if isinstance(cids, np.ndarray):
                    cids = cids.tolist()
                if isinstance(cids, (list, tuple)):
                    for c in cids:
                        norm = _normalize_pubchem_cid(c)
                        if norm:
                            rows.append({"_pc_norm": norm, "drug_chembl_id": row["drug_chembl_id"]})
            chembl_pc = pd.DataFrame(rows).drop_duplicates(subset=["_pc_norm"])
        else:
            chembl_pc = (
                drug_targets[["drug_pubchem_cid", "drug_chembl_id"]]
                .dropna(subset=["drug_pubchem_cid"])
                .copy()
            )
            chembl_pc["_pc_norm"] = chembl_pc["drug_pubchem_cid"].apply(_normalize_pubchem_cid)
            chembl_pc = chembl_pc[chembl_pc["_pc_norm"].notna()].drop_duplicates(subset=["_pc_norm"])

        if not unmatched_str.empty and not chembl_pc.empty:
            pc_merged = unmatched_str.merge(
                chembl_pc[["_pc_norm", "drug_chembl_id"]],
                on="_pc_norm",
                how="inner",
            )
            pc_merged = pc_merged.drop_duplicates(subset=["_pert_id"])
            if not pc_merged.empty:
                pc_result = pd.DataFrame({
                    "pert_id": pc_merged["_pert_id"].values,
                    "drug_name": pc_merged["_drug_name"].values,
                    "drug_inchikey": pc_merged[inchikey_col].values if inchikey_col and inchikey_col in pc_merged.columns else None,
                    "drug_chembl_id": pc_merged["drug_chembl_id"].values,
                    "match_confidence": "pubchem_cid",
                })
                all_parts.append(pc_result)
                matched_pert_ids.update(pc_result["pert_id"])
                n_pubchem = len(pc_result)

    logger.info("PubChem CID matching: %d compounds matched", n_pubchem)

    # --- Pass 3: Fuzzy name matching ---
    matched_records: list[dict] = []
    n_name = _fuzzy_name_match(
        lincs_metadata, drug_targets, lincs_cols,
        matched_records, matched_pert_ids,
    )
    if matched_records:
        all_parts.append(pd.DataFrame(matched_records))
    logger.info("Name matching: %d compounds matched", n_name)

    logger.info(
        "Match summary: %d InChIKey, %d PubChem CID, %d name -> %d total",
        n_inchikey, n_pubchem, n_name,
        n_inchikey + n_pubchem + n_name,
    )
    if all_parts:
        return pd.concat(all_parts, ignore_index=True)
    return pd.DataFrame()


def _fuzzy_name_match(
    lincs_metadata: pd.DataFrame,
    drug_targets: pd.DataFrame,
    lincs_cols: dict[str, str],
    matched_records: list[dict],
    matched_pert_ids: set[str],
) -> int:
    """Fuzzy name matching fallback using rapidfuzz."""
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        logger.warning(
            "rapidfuzz not installed; skipping name matching fallback. "
            "Install with: pip install repogen[lincs]"
        )
        return 0

    pert_col = lincs_cols["pert_id"]
    name_col = lincs_cols["name"]
    inchikey_col = lincs_cols.get("inchikey")

    dt_dedup = (
        drug_targets
        .dropna(subset=["drug_name"])
        .drop_duplicates(subset=["drug_name"])
        .copy()
    )
    dt_dedup["_name_upper"] = dt_dedup["drug_name"].str.upper().str.strip()
    dt_dedup = dt_dedup[dt_dedup["_name_upper"] != ""]

    _CANONICAL_RE = re.compile(r"^CHEMBL\d+$")
    dt_dedup["_is_canonical"] = dt_dedup["drug_chembl_id"].apply(
        lambda x: bool(_CANONICAL_RE.match(str(x))) if pd.notna(x) else False
    )
    dt_dedup["_has_ik"] = dt_dedup["drug_inchikey"].notna() & (
        dt_dedup["drug_inchikey"].astype(str).str.strip() != ""
    )
    n_collisions = len(dt_dedup) - dt_dedup["_name_upper"].nunique()
    if n_collisions > 0:
        logger.info(
            "Name lookup: collapsing %d normalized-key collisions "
            "(preferring canonical ChEMBL IDs)",
            n_collisions,
        )
    dt_dedup = (
        dt_dedup
        .sort_values(["_is_canonical", "_has_ik"], ascending=[False, False])
        .drop_duplicates(subset=["_name_upper"], keep="first")
    )
    dt_dedup = dt_dedup.drop(columns=["_is_canonical", "_has_ik"])

    chembl_name_map = (
        dt_dedup
        .set_index("_name_upper")[["drug_chembl_id", "drug_inchikey"]]
        .to_dict("index")
    )

    chembl_upper = list(chembl_name_map.keys())
    threshold = 85
    count = 0

    pert_idx = lincs_metadata.columns.get_loc(pert_col)
    name_idx = lincs_metadata.columns.get_loc(name_col)
    ik_idx = lincs_metadata.columns.get_loc(inchikey_col) if inchikey_col and inchikey_col in lincs_metadata.columns else None

    for row in lincs_metadata.itertuples(index=False):
        pert_id = str(row[pert_idx]).strip()
        if pert_id in matched_pert_ids:
            continue
        lincs_name = str(row[name_idx]).strip()
        if not lincs_name:
            continue

        match = process.extractOne(
            lincs_name.upper(), chembl_upper,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if match is None:
            continue

        matched_name, score, _ = match
        info = chembl_name_map.get(matched_name, {})
        matched_records.append({
            "pert_id": pert_id,
            "drug_name": lincs_name,
            "drug_inchikey": str(row[ik_idx]) if ik_idx is not None else None,
            "drug_chembl_id": info.get("drug_chembl_id"),
            "match_confidence": "name",
        })
        matched_pert_ids.add(pert_id)
        count += 1

    return count


def _name_only_matching(lincs_metadata: pd.DataFrame) -> pd.DataFrame:
    """Fallback when no drug_targets are provided - use metadata names."""
    lincs_cols = _detect_lincs_metadata_columns(lincs_metadata)
    if lincs_cols is None:
        return pd.DataFrame()

    pert_col = lincs_cols["pert_id"]
    name_col = lincs_cols["name"]
    inchikey_col = lincs_cols.get("inchikey")

    df = lincs_metadata.copy()
    df["_pert_id"] = df[pert_col].astype(str).str.strip()
    df["_name"] = df[name_col].astype(str).str.strip()
    df = df[(df["_pert_id"] != "") & (df["_name"] != "")]

    if df.empty:
        return pd.DataFrame()

    inchikeys: object
    if inchikey_col and inchikey_col in df.columns:
        inchikeys = df[inchikey_col].where(df[inchikey_col].notna()).astype(str).str.strip()
        inchikeys = inchikeys.where(inchikeys != "", other=None)
    else:
        inchikeys = None

    return pd.DataFrame({
        "pert_id": df["_pert_id"].values,
        "drug_name": df["_name"].values,
        "drug_inchikey": inchikeys.values if inchikeys is not None else None,
        "drug_chembl_id": None,
        "match_confidence": "name",
    })


def _detect_lincs_metadata_columns(
    df: pd.DataFrame,
) -> Optional[dict[str, str]]:
    """Detect Drug Repurposing Hub column names."""
    cols_lower = {c.lower().strip(): c for c in df.columns}

    pert_candidates = ["pert_id", "pert_iname", "broad_id", "compound_id"]
    name_candidates = ["name", "pert_iname", "drug_name", "compound_name",
                       "cmap_name"]
    inchikey_candidates = ["inchi_key", "inchikey", "standard_inchi_key"]
    pubchem_candidates = ["pubchem_cid", "pubchem", "cid"]

    from repogen.data.drug_loader import _first_match

    pert_col = _first_match(cols_lower, pert_candidates)
    name_col = _first_match(cols_lower, name_candidates)

    if pert_col is None or name_col is None:
        return None

    return {
        "pert_id": pert_col,
        "name": name_col,
        "inchikey": _first_match(cols_lower, inchikey_candidates),
        "pubchem_cid": _first_match(cols_lower, pubchem_candidates),
    }


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------


def _load_gene_metadata(gene_metadata_path: Path) -> pd.DataFrame:
    """Load LINCS gene metadata mapping probe IDs to Entrez Gene IDs."""
    df = pd.read_csv(gene_metadata_path, sep="\t", low_memory=False)

    cols_lower = {c.lower().strip(): c for c in df.columns}
    gene_id_col = None
    for candidate in ["gene_id", "pr_gene_id", "entrez_id", "gene_symbol"]:
        if candidate in cols_lower:
            gene_id_col = cols_lower[candidate]
            break

    if gene_id_col is None and len(df.columns) >= 1:
        gene_id_col = df.columns[0]

    if gene_id_col:
        df = df.rename(columns={gene_id_col: "gene_id"})
        df["gene_id"] = pd.to_numeric(df["gene_id"], errors="coerce")
        df = df.dropna(subset=["gene_id"])
        df["gene_id"] = df["gene_id"].astype(int)

    return df


_TIME_SUFFIX_RE = re.compile(r"^(\d+)H$", re.IGNORECASE)
_BRD_PREFIX_RE = re.compile(r"^(BRD-[A-Z]\d{8})")


def _normalize_brd_id(raw_brd: str) -> str:
    """Normalize a BRD identifier to its 2-segment compound form.

    LINCS uses two BRD formats:

    * Compound ID (2-segment): ``BRD-K25050358``
    * Batch ID (5-segment): ``BRD-K25050358-001-01-5``

    The first two segments identify the compound; the trailing segments
    encode batch/plate/well metadata.  The Drug Repurposing Hub uses
    the 2-segment form, so we normalize to that for matching.
    """
    m = _BRD_PREFIX_RE.match(raw_brd)
    return m.group(1) if m else raw_brd


def _parse_gctx_col_id(col_id: str) -> dict[str, str]:
    """Parse a GCTX column ID string into profile metadata components.

    Known LINCS Level 5 column ID formats:

    * 3-token (dominant): ``EXP_CELLLINE_NNH:BRD-xxx:DOSE``
      - time is embedded in the prefix suffix (e.g. ``24H``).
    * 4-token (minority): ``EXP_CELLLINE_XH:BRD-xxx:DOSE:TIME``
      - time is the fourth colon-separated token.

    The ``pert_id`` field is normalized to the 2-segment compound
    identifier (e.g. ``BRD-K25050358-001-01-5`` -> ``BRD-K25050358``).

    Conservative: any field that cannot be deterministically parsed
    is returned as ``"unknown"``.
    """
    result: dict[str, str] = {
        "pert_id": "unknown",
        "cell_line": "unknown",
        "dose": "unknown",
        "time": "unknown",
    }

    tokens = col_id.split(":")
    if len(tokens) < 3:
        return result

    result["pert_id"] = _normalize_brd_id(tokens[1])
    result["dose"] = tokens[2]

    if len(tokens) >= 4:
        result["time"] = tokens[3]

    prefix = tokens[0]
    prefix_parts = prefix.split("_")

    if len(prefix_parts) >= 3:
        last_part = prefix_parts[-1]
        time_match = _TIME_SUFFIX_RE.match(last_part)
        if time_match:
            if len(tokens) == 3:
                result["time"] = time_match.group(1)
            cell_line = "_".join(prefix_parts[1:-1])
            result["cell_line"] = cell_line if cell_line else "unknown"
        elif last_part.upper() == "XH" and len(tokens) >= 4:
            cell_line = "_".join(prefix_parts[1:-1])
            result["cell_line"] = cell_line if cell_line else "unknown"
    elif len(prefix_parts) == 2:
        result["cell_line"] = prefix_parts[1] if prefix_parts[1] else "unknown"

    return result


def _index_gctx_columns_for_perts(
    gctx_path: Path,
    pert_ids: set[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    """Build ``pert_id -> [gctx_col_id]`` index by scanning GCTX column metadata.

    Opens the GCTX file with *h5py* (read-only) and scans all column IDs
    in ``0/META/COL/id``.  BRD identifiers are normalized to their
    2-segment compound form (``BRD-K25050358-001-01-5`` -> ``BRD-K25050358``)
    so that extended-batch GCTX columns match the Drug Repurposing Hub's
    compound-level identifiers.

    Returns:
        pert_to_cids: ``{pert_id: [col_id, ...]}``
        cid_meta: ``{col_id: {"pert_id", "cell_line", "dose", "time"}}``
    """
    import h5py

    pert_to_cids: dict[str, list[str]] = {}
    cid_meta: dict[str, dict[str, str]] = {}
    n_scanned = 0
    n_unresolved = 0

    with h5py.File(str(gctx_path), "r") as f:
        col_ids_raw = f["0/META/COL/id"][:]

    for raw_id in col_ids_raw:
        n_scanned += 1
        col_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)

        parsed = _parse_gctx_col_id(col_id)
        pid = parsed["pert_id"]

        if pid in pert_ids:
            pert_to_cids.setdefault(pid, []).append(col_id)
            cid_meta[col_id] = parsed
        elif pid == "unknown":
            n_unresolved += 1

    n_matched = len(pert_to_cids)
    n_profiles = sum(len(v) for v in pert_to_cids.values())
    n_missing = len(pert_ids) - n_matched

    logger.info(
        "GCTX column index: scanned %d columns, %d matched perturbagens "
        "(%d profiles), %d perturbagens missing GCTX profiles, "
        "%d unresolved column IDs",
        n_scanned,
        n_matched,
        n_profiles,
        n_missing,
        n_unresolved,
    )

    return pert_to_cids, cid_meta


def _extract_profiles_chunked(
    parse_gctx,
    gctx_path: Path,
    pert_ids: list[str],
    gene_ids: list[int],
    chunk_size: int = 2000,
) -> dict[str, list[dict]]:
    """Extract expression profiles from GCTX in chunks.

    Scans GCTX column IDs via *h5py*, builds a ``pert_id -> [col_id]``
    index, then reads expression data in chunks of real GCTX column IDs
    using *cmapPy*.  Profile metadata (cell_line, dose, time) is parsed
    from the column ID strings rather than from external metadata files.

    Z-score vectors are stored as ``numpy.ndarray`` (float32) throughout
    extraction and aggregation to avoid the ~4.5x memory overhead of
    Python lists of boxed floats.  Conversion to Python lists is
    deferred to the output boundary in ``_build_signature_records``.

    Returns a dict mapping pert_id to a list of profile dicts,
    each containing ``z_scores`` (ndarray), ``cell_line``, ``dose``,
    ``time``.
    """
    pert_set = set(pert_ids)
    pert_to_cids, cid_meta = _index_gctx_columns_for_perts(gctx_path, pert_set)

    if not pert_to_cids:
        logger.warning(
            "No GCTX column IDs found for any of the %d requested perturbagens",
            len(pert_ids),
        )
        return {}

    all_cids: list[str] = []
    for pid in pert_ids:
        all_cids.extend(pert_to_cids.get(pid, []))

    n_genes = len(gene_ids)
    n_profiles_total = len(all_cids)
    est_f32_mb = n_profiles_total * n_genes * 4 / (1024 * 1024)
    est_f64_mb = n_profiles_total * n_genes * 8 / (1024 * 1024)
    logger.info(
        "Memory preflight: %d profiles x %d genes, "
        "estimated %.0f MB (float32) / %.0f MB (float64)",
        n_profiles_total, n_genes, est_f32_mb, est_f64_mb,
    )

    profiles: dict[str, list[dict]] = {}
    n_chunks = (len(all_cids) + chunk_size - 1) // chunk_size

    for chunk_start in range(0, len(all_cids), chunk_size):
        chunk_cids = all_cids[chunk_start : chunk_start + chunk_size]
        chunk_num = chunk_start // chunk_size + 1
        logger.info(
            "Extracting GCTX chunk %d/%d (%d profiles)",
            chunk_num,
            n_chunks,
            len(chunk_cids),
        )

        try:
            gctx_data = parse_gctx.parse(
                str(gctx_path),
                cid=chunk_cids,
                rid=gene_ids,
            )
        except Exception as exc:
            logger.warning("GCTX parse error on chunk %d: %s", chunk_num, exc)
            continue

        data_df = gctx_data.data_df

        for col_id in data_df.columns:
            col_id_str = str(col_id)
            meta = cid_meta.get(col_id_str)
            if meta is None:
                continue

            pid = meta["pert_id"]
            z_scores = data_df[col_id].values.astype(np.float32)

            profile_info = {
                "z_scores": z_scores,
                "cell_line": meta["cell_line"],
                "dose": meta["dose"],
                "time": meta["time"],
            }

            profiles.setdefault(pid, []).append(profile_info)

        del gctx_data, data_df

    return profiles


# ---------------------------------------------------------------------------
# Signature aggregation
# ---------------------------------------------------------------------------


def weighted_nanmedian_per_gene(
    z_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Column-wise weighted median with NaN handling, fully vectorised.

    helper. For each gene column of ``z_matrix``, computes
    the *lower weighted median* - the smallest value whose cumulative
    weight (sorted ascending) reaches at least half the total weight.
    NaN values are excluded from both the sort and the weight sum on a
    per-column basis (equivalent to ``np.nanmedian`` semantics).

    Args:
        z_matrix: ``(n_profiles, n_genes)`` float array of z-scores.
        weights: ``(n_profiles,)`` float array of per-profile weights.
            Zero weight excludes a profile from the median for every
            gene column.

    Returns:
        ``(n_genes,)`` float32 array of weighted median values, NaN in
        columns where every value is NaN (or every weight is zero).

    Invariants (test-locked):

    * Equal-weight fast path - when all weights are equal, the result
      is byte-identical to ``np.nanmedian(z_matrix, axis=0).astype(float32)``.
      This preserves the "uniform mode byte-identical" and "zero-neural-
      profiles under ``neural_priority`` equals uniform" guarantees.
    * Unequal-weight convention - lower weighted median (Feller-style
      cumulative-weight crossover).  This is a standard, well-defined
      convention; the alternative (interpolated weighted median) is not
      implemented because it has no canonical extension under NaN
      handling.
    """
    n_profiles, n_genes = z_matrix.shape
    if n_profiles == 0:
        return np.full(n_genes, np.nan, dtype=np.float32)

    # --- Fast path: all equal weights -> delegate to np.nanmedian -------
    # Special-cased for the byte-identity invariant.  Also covers the
    # ``neural_priority`` case where a drug has zero neural profiles
    # (weights collapse to a constant), which is why we do NOT test on
    # ``method == "uniform"`` alone.
    if weights.size > 0 and np.all(weights == weights[0]):
        return np.nanmedian(z_matrix, axis=0).astype(np.float32)

    # --- General path: vectorised argsort + cumulative-weight crossover ---
    # Ascending per-column sort. NumPy places NaN at the end for both
    # 'quicksort' and 'mergesort'; mergesort is stable which makes tie
    # handling deterministic.
    sort_idx = np.argsort(z_matrix, axis=0, kind="mergesort")
    z_sorted = np.take_along_axis(z_matrix, sort_idx, axis=0)
    # Broadcast weights along the sort permutation.  weights[sort_idx]
    # broadcasts a 1-D weight vector across the 2-D sort-index matrix.
    w_sorted = weights[sort_idx]
    # Zero out weights that correspond to NaN z-values so those rows
    # neither contribute to the cumulative sum nor shift the threshold.
    w_sorted = np.where(np.isnan(z_sorted), 0.0, w_sorted)
    # Cumulative weight along the profile axis.
    w_cum = np.cumsum(w_sorted, axis=0)
    w_total = w_cum[-1, :]
    threshold = 0.5 * w_total
    # First index per column where cumulative weight >= threshold.
    # ``argmax`` on a boolean matrix returns the index of the first True.
    exceeds = w_cum >= threshold[np.newaxis, :]
    median_idx = np.argmax(exceeds, axis=0)
    result = z_sorted[median_idx, np.arange(n_genes)]
    # Columns whose entire z_matrix column was NaN (=> w_total == 0)
    # must return NaN, not the first (garbage) sorted value.
    result = np.where(w_total == 0.0, np.nan, result)
    return result.astype(np.float32)


def _neural_composition(
    profiles: list[dict],
    neural_set: set[str],
) -> dict[str, int | float]:
    """Compute composition counts and fractions for a profile list.

    Composition invariant (test-locked):
        ``n_profiles_total == n_profiles_neural
                            + n_profiles_non_neural
                            + n_profiles_unknown_cell_line``

    ``neural_fraction`` uses ``n_profiles_total`` as denominator (0.0 when
    the list is empty).  ``neural_weight_fraction`` is set to
    ``neural_fraction`` here as an informational default; the caller
    overrides it for ``neural_priority`` mode where weights differ from
    counts.
    """
    if not profiles:
        return {
            "n_profiles_total": 0,
            "n_profiles_neural": 0,
            "n_profiles_non_neural": 0,
            "n_profiles_unknown_cell_line": 0,
            "neural_fraction": 0.0,
            "neural_weight_fraction": 0.0,
        }
    per_profile_cl = [str(p.get("cell_line", "unknown")).upper().strip() for p in profiles]
    n_total = len(profiles)
    n_neural = sum(1 for cl in per_profile_cl if cl in neural_set)
    n_unknown = sum(1 for cl in per_profile_cl if cl == "UNKNOWN")
    n_non_neural = n_total - n_neural - n_unknown
    neural_fraction = n_neural / n_total if n_total else 0.0
    return {
        "n_profiles_total": n_total,
        "n_profiles_neural": n_neural,
        "n_profiles_non_neural": n_non_neural,
        "n_profiles_unknown_cell_line": n_unknown,
        "neural_fraction": neural_fraction,
        "neural_weight_fraction": neural_fraction,
    }


def _apply_neural_only_filter(
    profiles: list[dict],
    neural_set: set[str],
) -> list[dict]:
    """Return only profiles whose cell_line (uppercased, stripped) is
    in the neural set.  Used by ``cell_line_weighting == "neural_only"``.
    """
    return [
        p for p in profiles
        if str(p.get("cell_line", "unknown")).upper().strip() in neural_set
    ]


def _profile_weights(
    profiles: list[dict],
    neural_set: set[str],
    neural_weight: float,
) -> np.ndarray:
    """Build a per-profile weight array for ``neural_priority`` mode."""
    return np.array(
        [
            neural_weight
            if str(p.get("cell_line", "unknown")).upper().strip() in neural_set
            else 1.0
            for p in profiles
        ],
        dtype=np.float64,
    )


def aggregate_signatures(
    profiles: dict[str, list[dict]],
    method: str = "consensus",
    min_profiles: int = 3,
    preferred_dose: Optional[str] = "10 µM",
    preferred_time: Optional[str] = "24 h",
    cell_line_weighting: str = "uniform",
    neural_cell_lines: Optional[list[str]] = None,
    neural_weight: float = 3.0,
    min_neural_profiles: Optional[int] = None,
) -> dict[str, dict]:
    """Aggregate multiple profiles per drug into consensus signatures.

    Args:
        profiles: Mapping of pert_id -> list of profile dicts.
        method: ``"consensus"`` (median Z across all), ``"best_dose"``
            (filter to preferred dose/time), ``"per_condition"`` (keep
            separate per cell_line+dose+time - returns one entry per
            condition).
        min_profiles: Minimum profiles to keep a drug.
        preferred_dose: Dose string for ``"best_dose"`` filtering.
        preferred_time: Time string for ``"best_dose"`` filtering.
        cell_line_weighting: - one of ``"uniform"``,
            ``"neural_priority"``, ``"neural_only"``.  ``"uniform"``
            reproduces previous behaviour byte-for-byte.
        neural_cell_lines: List of exact-uppercase LINCS cell-line tokens
            treated as neural.  Ignored in ``"uniform"`` mode; required
            for the other two modes.
        neural_weight: Upweight factor applied to neural profiles in
            ``"neural_priority"`` mode.
        min_neural_profiles: Minimum neural profiles required in
            ``"neural_only"`` mode.  If None, resolves to ``min_profiles``.

    Returns:
        Mapping of pert_id -> aggregated signature dict with keys
        ``z_scores``, ``cell_lines``, ``doses``, ``time_points``,
        ``n_profiles`` plus composition keys (``n_profiles_total``,
        ``n_profiles_neural``, ``n_profiles_non_neural``,
        ``n_profiles_unknown_cell_line``, ``neural_fraction``,
        ``neural_weight_fraction``, ``cell_line_weighting_mode``,
        ``neural_weight``).
    """
    if cell_line_weighting not in ("uniform", "neural_priority", "neural_only"):
        raise ValueError(
            f"cell_line_weighting must be one of "
            f"'uniform'|'neural_priority'|'neural_only', got '{cell_line_weighting}'"
        )
    neural_set: set[str] = {c.upper().strip() for c in (neural_cell_lines or [])}
    if cell_line_weighting != "uniform" and not neural_set:
        raise ValueError(
            f"cell_line_weighting='{cell_line_weighting}' requires a "
            f"non-empty neural_cell_lines list."
        )
    effective_min_neural = min_neural_profiles if min_neural_profiles is not None else min_profiles

    aggregated: dict[str, dict] = {}
    # Cohort-level accounting for the observability log.
    n_neural_only_kept = 0
    n_neural_only_dropped = 0
    n_priority_unchanged = 0
    n_priority_reweighted = 0
    n_with_neural_fraction_gt_0 = 0
    global_cell_line_counts: dict[str, int] = {}

    for pert_id, prof_list in profiles.items():
        if method == "best_dose":
            filtered = [
                p for p in prof_list
                if (preferred_dose is None or preferred_dose in str(p.get("dose", "")))
                and (preferred_time is None or preferred_time in str(p.get("time", "")))
            ]
            if len(filtered) < min_profiles:
                filtered = prof_list
        else:
            filtered = prof_list

        if len(filtered) < min_profiles:
            continue

        # Composition is ALWAYS computed on the pre-neural-filter set
        # (``filtered``) so that ``neural_fraction`` in ``neural_only``
        # mode reflects the drug's original LINCS composition, not the
        # trivially-1.0 post-filter fraction.
        composition = _neural_composition(filtered, neural_set)
        # Update global cell-line counts (for the runtime QC log).
        for p in filtered:
            cl = str(p.get("cell_line", "unknown")).upper().strip()
            global_cell_line_counts[cl] = global_cell_line_counts.get(cl, 0) + 1
        if composition["n_profiles_neural"] > 0:
            n_with_neural_fraction_gt_0 += 1

        if method == "per_condition":
            # Weighting/filtering applies within each condition
            # bucket the same way it does for whole-drug consensus.
            conditions: dict[str, list[dict]] = {}
            for p in filtered:
                key = f"{p['cell_line']}_{p['dose']}_{p['time']}"
                conditions.setdefault(key, []).append(p)

            for cond_key, cond_profiles in conditions.items():
                cond_composition = _neural_composition(cond_profiles, neural_set)
                agg_profiles, agg_weights, mode_used_weighted = _resolve_aggregation_profiles(
                    cond_profiles, cell_line_weighting, neural_set,
                    neural_weight, effective_min_neural, min_profiles,
                )
                if agg_profiles is None:
                    continue
                z_matrix = np.array([p["z_scores"] for p in agg_profiles])
                if mode_used_weighted:
                    consensus_z = weighted_nanmedian_per_gene(z_matrix, agg_weights)
                else:
                    consensus_z = np.nanmedian(z_matrix, axis=0).astype(np.float32)
                agg_id = f"{pert_id}:{cond_key}"
                nw_frac = _compute_neural_weight_fraction(
                    cell_line_weighting, cond_profiles, neural_set, neural_weight,
                    cond_composition["neural_fraction"], agg_profiles,
                )
                aggregated[agg_id] = {
                    "z_scores": consensus_z,
                    "cell_lines": [cond_profiles[0]["cell_line"]],
                    "doses": [cond_profiles[0]["dose"]],
                    "time_points": [cond_profiles[0]["time"]],
                    "n_profiles": len(agg_profiles),
                    "source_pert_id": pert_id,
                    **cond_composition,
                    "neural_weight_fraction": nw_frac,
                    "cell_line_weighting_mode": cell_line_weighting,
                    "neural_weight": float(neural_weight),
                }
        else:
            agg_profiles, agg_weights, mode_used_weighted = _resolve_aggregation_profiles(
                filtered, cell_line_weighting, neural_set,
                neural_weight, effective_min_neural, min_profiles,
            )
            if agg_profiles is None:
                if cell_line_weighting == "neural_only":
                    n_neural_only_dropped += 1
                continue
            if cell_line_weighting == "neural_only":
                n_neural_only_kept += 1
            elif cell_line_weighting == "neural_priority":
                # ``mode_used_weighted`` distinguishes reweighted from
                # constant-weight (equal-weight) aggregations.  The
                # weighted_nanmedian fast path itself delegates to
                # np.nanmedian under equal weights, so this flag
                # measures "would-be-reweighted-if-neural-present".
                if composition["n_profiles_neural"] == 0:
                    n_priority_unchanged += 1
                else:
                    n_priority_reweighted += 1

            z_matrix = np.array([p["z_scores"] for p in agg_profiles])
            if mode_used_weighted:
                consensus_z = weighted_nanmedian_per_gene(z_matrix, agg_weights)
            else:
                consensus_z = np.nanmedian(z_matrix, axis=0).astype(np.float32)

            nw_frac = _compute_neural_weight_fraction(
                cell_line_weighting, filtered, neural_set, neural_weight,
                composition["neural_fraction"], agg_profiles,
            )
            aggregated[pert_id] = {
                "z_scores": consensus_z,
                "cell_lines": sorted(set(p["cell_line"] for p in agg_profiles)),
                "doses": sorted(set(p["dose"] for p in agg_profiles)),
                "time_points": sorted(set(p["time"] for p in agg_profiles)),
                "n_profiles": len(agg_profiles),
                **composition,
                "neural_weight_fraction": nw_frac,
                "cell_line_weighting_mode": cell_line_weighting,
                "neural_weight": float(neural_weight),
            }

    # --- Runtime QC log ---------------------
    logger.info(
        "Aggregation (%s, weighting=%s): %d drugs in -> %d signatures out "
        "(min_profiles=%d)",
        method, cell_line_weighting, len(profiles), len(aggregated), min_profiles,
    )
    if global_cell_line_counts:
        n_total_profiles = sum(global_cell_line_counts.values())
        n_neural_profiles = sum(
            n for cl, n in global_cell_line_counts.items() if cl in neural_set
        )
        neural_frac = n_neural_profiles / n_total_profiles if n_total_profiles else 0.0
        top20 = sorted(global_cell_line_counts.items(), key=lambda kv: -kv[1])[:20]
        top20_str = ", ".join(f"{cl} ({n:,})" for cl, n in top20)
        logger.info(
            "Cell-line composition (top 20 by profile count): %s", top20_str,
        )
        if neural_set:
            neural_hits = {cl: global_cell_line_counts.get(cl, 0) for cl in sorted(neural_set)}
            missing = [cl for cl, n in neural_hits.items() if n == 0]
            hits_str = ", ".join(f"{cl} ({n:,})" for cl, n in neural_hits.items())
            logger.info("Neural cell-line tokens matched: %s", hits_str)
            if missing:
                logger.warning(
                    "Neural cell-line tokens with zero observed profiles: %s "
                    "(check ``neural_cell_lines`` config vs. current GCTX corpus)",
                    missing,
                )
        logger.info(
            "Neural profiles: %d / %d (%.2f%%); drugs with neural_fraction > 0: %d",
            n_neural_profiles, n_total_profiles, 100.0 * neural_frac,
            n_with_neural_fraction_gt_0,
        )
    if cell_line_weighting == "neural_only":
        logger.info(
            "neural_only cohort: kept %d drugs (>= min_neural_profiles=%d neural); "
            "dropped %d drugs below threshold",
            n_neural_only_kept, effective_min_neural, n_neural_only_dropped,
        )
    elif cell_line_weighting == "neural_priority":
        logger.info(
            "neural_priority cohort: %d drugs reweighted (>= 1 neural profile); "
            "%d drugs identical to uniform (0 neural profiles -> equal weights)",
            n_priority_reweighted, n_priority_unchanged,
        )
    return aggregated


def _resolve_aggregation_profiles(
    filtered: list[dict],
    cell_line_weighting: str,
    neural_set: set[str],
    neural_weight: float,
    effective_min_neural: int,
    min_profiles: int,
) -> tuple[Optional[list[dict]], Optional[np.ndarray], bool]:
    """Resolve which profile subset actually feeds the median call and
    what per-profile weights it carries, given the weighting mode.

    Returns ``(agg_profiles, weights, use_weighted_median)``.
    ``agg_profiles`` is None when the drug should be dropped
    (``neural_only`` below threshold).
    """
    if cell_line_weighting == "uniform":
        return filtered, None, False
    if cell_line_weighting == "neural_only":
        neural_profiles = _apply_neural_only_filter(filtered, neural_set)
        if len(neural_profiles) < effective_min_neural:
            return None, None, False
        # Neural-only aggregation is a straight nanmedian over the
        # filtered subset (equal weights among neural profiles).
        return neural_profiles, None, False
    # neural_priority
    weights = _profile_weights(filtered, neural_set, neural_weight)
    return filtered, weights, True


def _compute_neural_weight_fraction(
    cell_line_weighting: str,
    composition_profiles: list[dict],
    neural_set: set[str],
    neural_weight: float,
    fallback_fraction: float,
    agg_profiles: list[dict],
) -> float:
    """neural_weight_fraction: informational, and mode-dependent.

    * ``uniform``:  equals ``neural_fraction`` (all weights == 1).
    * ``neural_priority``:  ``sum(w_neural) / sum(w_all)`` over the
      composition profile list (the actual weight allocation).
    * ``neural_only``:  1.0 (only neural profiles contribute).
    """
    if cell_line_weighting == "uniform":
        return float(fallback_fraction)
    if cell_line_weighting == "neural_only":
        return 1.0 if agg_profiles else 0.0
    # neural_priority
    w = _profile_weights(composition_profiles, neural_set, neural_weight)
    if w.size == 0 or w.sum() == 0:
        return 0.0
    neural_mask = np.array(
        [
            str(p.get("cell_line", "unknown")).upper().strip() in neural_set
            for p in composition_profiles
        ]
    )
    return float(w[neural_mask].sum() / w.sum())


# ---------------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------------


def _build_signature_records(
    signatures: dict[str, dict],
    matched: pd.DataFrame,
    gene_ids: list[int],
) -> pd.DataFrame:
    """Build DrugSignatureRecord DataFrame from aggregated signatures."""
    matched_lookup = (
        matched
        .dropna(subset=["pert_id"])
        .drop_duplicates(subset=["pert_id"])
        .set_index("pert_id")
        .to_dict("index")
    )

    records = []
    for sig_id, sig_data in signatures.items():
        source_pert = sig_data.get("source_pert_id", sig_id)
        base_pert = source_pert.split(":")[0] if ":" in source_pert else source_pert
        drug_info = matched_lookup.get(base_pert, {})

        z = sig_data["z_scores"]
        z_list = z.tolist() if isinstance(z, np.ndarray) else list(z)

        records.append({
            "drug_name": drug_info.get("drug_name", base_pert),
            "drug_inchikey": drug_info.get("drug_inchikey", ""),
            "drug_chembl_id": drug_info.get("drug_chembl_id"),
            "lincs_pert_id": base_pert,
            "n_profiles_aggregated": sig_data["n_profiles"],
            "cell_lines": sig_data["cell_lines"],
            "doses": sig_data["doses"],
            "time_points": sig_data["time_points"],
            "gene_ids": gene_ids,
            "z_scores": z_list,
            "match_confidence": drug_info.get("match_confidence", "name"),
            # composition columns (populated by aggregate_signatures).
            "n_profiles_total": sig_data.get("n_profiles_total"),
            "n_profiles_neural": sig_data.get("n_profiles_neural"),
            "n_profiles_non_neural": sig_data.get("n_profiles_non_neural"),
            "n_profiles_unknown_cell_line": sig_data.get("n_profiles_unknown_cell_line"),
            "neural_fraction": sig_data.get("neural_fraction"),
            "neural_weight_fraction": sig_data.get("neural_weight_fraction"),
            "cell_line_weighting_mode": sig_data.get("cell_line_weighting_mode"),
            "neural_weight": sig_data.get("neural_weight"),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract drug expression signatures from LINCS L1000"
    )
    parser.add_argument(
        "--gctx", type=Path, required=True,
        help="Path to LINCS L1000 Level 5 GCTX file",
    )
    parser.add_argument(
        "--compound-metadata", type=Path, required=True,
        help="Path to Drug Repurposing Hub metadata CSV",
    )
    parser.add_argument(
        "--gene-metadata", type=Path, required=True,
        help="Path to LINCS gene metadata file",
    )
    parser.add_argument(
        "--drug-targets", type=Path, default=None,
        help="Path to DrugTargetRecord Parquet from drug_loader",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--aggregation", type=str, default="consensus",
        choices=["consensus", "best_dose", "per_condition"],
        help="Aggregation method",
    )
    parser.add_argument(
        "--min-profiles", type=int, default=3,
        help="Minimum profiles to aggregate per drug",
    )
    # Cell-line weighting mode-switch
    parser.add_argument(
        "--cell-line-weighting", type=str, default="uniform",
        choices=["uniform", "neural_priority", "neural_only"],
        help="cell-line weighting mode. Default 'uniform' "
             "reproduces previous behaviour byte-for-byte.",
    )
    parser.add_argument(
        "--neural-cell-lines", type=str, default=None,
        help="Comma-separated exact-uppercase LINCS cell-line tokens "
             "treated as neural. If omitted, uses the built-in defaults.",
    )
    parser.add_argument(
        "--neural-weight", type=float, default=3.0,
        help="Upweight factor for neural profiles in neural_priority mode.",
    )
    parser.add_argument(
        "--min-neural-profiles", type=int, default=None,
        help="Minimum neural profiles for neural_only mode. Defaults to "
             "--min-profiles for coherence with the extraction threshold.",
    )
    args = parser.parse_args()

    targets_df = None
    if args.drug_targets:
        targets_df = pd.read_parquet(args.drug_targets)
        logger.info("Loaded %d drug targets", len(targets_df))

    # Resolve neural cell lines from --neural-cell-lines, or fall back
    # to the module-level default (primary + tumor) if a non-uniform mode is
    # requested without an explicit list.
    resolved_neural_cell_lines: Optional[list[str]] = None
    if args.neural_cell_lines is not None:
        resolved_neural_cell_lines = [
            c.strip().upper() for c in args.neural_cell_lines.split(",") if c.strip()
        ]
    elif args.cell_line_weighting != "uniform":
        resolved_neural_cell_lines = list(NEURAL_PRIMARY_CELL_LINES + NEURAL_TUMOR_CELL_LINES)

    result_df = extract_drug_signatures(
        gctx_path=args.gctx,
        compound_metadata=args.compound_metadata,
        gene_metadata=args.gene_metadata,
        drug_targets=targets_df,
        aggregation=args.aggregation,
        min_profiles=args.min_profiles,
        cell_line_weighting=args.cell_line_weighting,
        neural_cell_lines=resolved_neural_cell_lines,
        neural_weight=args.neural_weight,
        min_neural_profiles=args.min_neural_profiles,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Saved %d signatures to %s", len(result_df), output_path)
