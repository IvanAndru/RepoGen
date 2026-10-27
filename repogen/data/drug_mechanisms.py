"""Curated ChEMBL mechanisms by the route through which they reach a gene.

The drug table (``drug_loader``) keeps ChEMBL mechanisms on single-protein
targets only and loses a curated direction when a salt's mechanism row
meets its parent's affinity row in de-duplication.  This module reads
every curated mechanism on a human target once, resolves each molecule to
its parent, and labels each drug-gene pair (records 042, 045, 047):

- ``target_route``: how the mechanism reaches the gene, the first that
  applies of ``single_protein``; ``nucleic_acid`` (a nucleic-acid target
  with one gene, stored by ChEMBL as an Ensembl accession);
  ``site_member`` (a component of a complex's curated binding site);
  ``complex_member`` (a component of a complex, complex group, chimeric
  protein, protein-protein interaction or multi-gene nucleic-acid target
  without a curated site); ``family_member`` (protein family, selectivity
  group); ``off_site`` (a complex component that the curated site leaves
  out).  Pairs with no curated mechanism get ``none``.
- ``mechanism_action``: ChEMBL's action types on that route.
- ``mechanism_direction``: ``positive`` or ``negative`` from ChEMBL's own
  class of the action type (``action_type.parent_type``), a directed
  value winning over an undirected one and opposite values giving
  ``none``; ``off_site`` pairs are never directed.

The labels are added to the finished table as new columns, so no existing
value changes, and the pairs the table lacks are returned separately for
Branch C.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from repogen.utils.constants import INTERACTION_TYPE_STANDARDISATION

logger = logging.getLogger(__name__)

ROUTE_ORDER: tuple[str, ...] = (
    "single_protein", "nucleic_acid", "site_member",
    "complex_member", "family_member", "off_site",
)
MAIN_LIST_ROUTES: frozenset[str] = frozenset({"single_protein", "nucleic_acid"})
MEMBER_ROUTES: frozenset[str] = frozenset({"site_member", "complex_member", "family_member"})
FAMILY_TARGET_TYPES: frozenset[str] = frozenset({"PROTEIN FAMILY", "SELECTIVITY GROUP"})
_CLASS_DIRECTION = {"POSITIVE MODULATOR": "positive", "NEGATIVE MODULATOR": "negative"}
NEW_COLUMNS: tuple[str, ...] = ("target_route", "mechanism_action", "mechanism_direction")

_QUERY_MECHANISM_COMPONENTS = """\
SELECT
    dm.mec_id,
    dm.action_type,
    dm.mechanism_of_action,
    dm.site_id,
    at.parent_type                          AS action_class,
    md.chembl_id                            AS mol_chembl_id,
    md.pref_name                            AS mol_name,
    md.max_phase                            AS mol_phase,
    COALESCE(p.chembl_id, md.chembl_id)     AS drug_chembl_id,
    p.pref_name                             AS parent_name,
    p.max_phase                             AS parent_phase,
    EXISTS (SELECT 1 FROM drug_warning w WHERE w.molregno = md.molregno) AS has_warning,
    td.tid,
    td.target_type,
    tc.component_id,
    cs.component_type,
    cs.accession
FROM drug_mechanism dm
JOIN molecule_dictionary md ON md.molregno = dm.molregno
LEFT JOIN molecule_hierarchy mh ON mh.molregno = md.molregno
LEFT JOIN molecule_dictionary p ON p.molregno = mh.parent_molregno
JOIN target_dictionary td ON td.tid = dm.tid
JOIN target_components tc ON tc.tid = td.tid
JOIN component_sequences cs ON cs.component_id = tc.component_id
LEFT JOIN action_type at ON at.action_type = dm.action_type
WHERE td.organism = 'Homo sapiens'
"""

_QUERY_SITE_COMPONENTS = """\
SELECT site_id, component_id FROM site_components WHERE component_id IS NOT NULL
"""


def _route(target_type: str, n_target_components: int, site_components: Optional[set],
           component_id: int) -> str:
    if target_type == "SINGLE PROTEIN":
        return "single_protein"
    if target_type == "NUCLEIC-ACID":
        return "nucleic_acid" if n_target_components == 1 else "complex_member"
    if target_type in FAMILY_TARGET_TYPES:
        return "family_member"
    if site_components:
        return "site_member" if component_id in site_components else "off_site"
    return "complex_member"


def _direction(classes: pd.Series) -> str:
    dirs = {_CLASS_DIRECTION[c] for c in classes.dropna() if c in _CLASS_DIRECTION}
    return dirs.pop() if len(dirs) == 1 else "none"


def _standardised_type(actions: list[str]) -> str:
    types = {INTERACTION_TYPE_STANDARDISATION.get(a, "other") for a in actions}
    return types.pop() if len(types) == 1 else "other"


def load_curated_mechanism_pairs(
    sqlite_path: Path,
    gene_id_converter,
) -> pd.DataFrame:
    """One row per parent drug and gene that a curated mechanism reaches.

    Args:
        sqlite_path: ChEMBL SQLite database.
        gene_id_converter: :class:`GeneIDConverter`; protein components
            are mapped from their UniProt accession, nucleic-acid
            components from their Ensembl accession.  Components it
            cannot map are dropped.

    Returns:
        Columns ``drug_chembl_id`` (parent), ``drug_name``, ``max_phase``,
        ``is_withdrawn``, the four gene IDs, ``target_route``,
        ``mechanism_action``, ``mechanism_direction``,
        ``mechanism_of_action`` and ``interaction_type``.
    """
    conn = sqlite3.connect(str(sqlite_path))
    try:
        rows = pd.read_sql_query(_QUERY_MECHANISM_COMPONENTS, conn)
        sites = pd.read_sql_query(_QUERY_SITE_COMPONENTS, conn)
    finally:
        conn.close()
    logger.info("Curated mechanisms on human targets: %d mechanism-component rows, %d mechanisms",
                len(rows), rows["mec_id"].nunique() if not rows.empty else 0)
    if rows.empty:
        return pd.DataFrame(columns=[
            "drug_chembl_id", "drug_name", "max_phase", "is_withdrawn", "gene_symbol",
            "gene_ensembl_id", "gene_uniprot_id", "gene_entrez_id", *NEW_COLUMNS,
            "mechanism_of_action", "interaction_type",
        ])

    site_map = sites.groupby("site_id")["component_id"].apply(set).to_dict()
    n_components = rows.groupby("tid")["component_id"].nunique()
    rows["target_route"] = [
        _route(tt, int(n_components.get(tid, 0)), site_map.get(sid) if pd.notna(sid) else None, cid)
        for tt, tid, sid, cid in zip(rows["target_type"], rows["tid"], rows["site_id"],
                                     rows["component_id"])
    ]

    cache: dict[tuple[str, str], Optional[dict]] = {}
    records = []
    for acc, ctype in zip(rows["accession"], rows["component_type"]):
        id_type = "uniprot" if str(ctype).upper() == "PROTEIN" else "ensembl"
        key = (str(acc), id_type)
        if key not in cache:
            cache[key] = gene_id_converter.get_full_record(str(acc), id_type) if pd.notna(acc) else None
        records.append(cache[key])
    rows["gene_symbol"] = [r.get("symbol") if r else None for r in records]
    rows["gene_ensembl_id"] = [r.get("ensembl") if r else None for r in records]
    rows["gene_uniprot_id"] = [r.get("uniprot") if r else None for r in records]
    rows["gene_entrez_id"] = [r.get("entrez") if r else None for r in records]
    n_before = len(rows)
    rows = rows[rows["gene_symbol"].notna()].copy()
    logger.info("Mechanism components mapped to a gene: %d / %d", len(rows), n_before)

    rows["route_rank"] = rows["target_route"].map({r: i for i, r in enumerate(ROUTE_ORDER)})
    best = rows.groupby(["drug_chembl_id", "gene_symbol"])["route_rank"].transform("min")
    on_route = rows[rows["route_rank"] == best].copy()
    on_route.loc[on_route["target_route"] == "off_site", "action_class"] = None

    phase = pd.to_numeric(rows["parent_phase"], errors="coerce").fillna(
        pd.to_numeric(rows["mol_phase"], errors="coerce"))
    rows["_phase"] = np.fmax(phase, pd.to_numeric(rows["mol_phase"], errors="coerce"))

    grouped = on_route.groupby(["drug_chembl_id", "gene_symbol"], sort=True)
    pairs = grouped.agg(
        target_route=("target_route", "first"),
        mechanism_action=("action_type", lambda s: "|".join(sorted(set(s.dropna())))),
        mechanism_direction=("action_class", _direction),
        mechanism_of_action=("mechanism_of_action", lambda s: "|".join(sorted(set(s.dropna())))),
        interaction_type=("action_type", lambda s: _standardised_type(sorted(set(s.dropna())))),
        gene_ensembl_id=("gene_ensembl_id", "first"),
        gene_uniprot_id=("gene_uniprot_id", "first"),
        gene_entrez_id=("gene_entrez_id", "first"),
    ).reset_index()

    per_drug = rows.groupby("drug_chembl_id").agg(
        parent_name=("parent_name", "first"),
        mol_name=("mol_name", "first"),
        max_phase=("_phase", "max"),
        is_withdrawn=("has_warning", "max"),
    )
    pairs = pairs.merge(per_drug, left_on="drug_chembl_id", right_index=True, how="left")
    pairs["drug_name"] = pairs["parent_name"].fillna(pairs["mol_name"]).fillna(pairs["drug_chembl_id"])
    # The drug table's convention (drug_loader._postprocess_chembl).
    pairs["max_phase"] = pd.to_numeric(pairs["max_phase"], errors="coerce").fillna(0).astype(int)
    pairs["is_withdrawn"] = pairs["is_withdrawn"].fillna(0).astype(bool)
    pairs["gene_entrez_id"] = pd.to_numeric(pairs["gene_entrez_id"], errors="coerce").astype("Int64")
    for col in ("mechanism_action", "mechanism_of_action"):
        pairs[col] = pairs[col].replace("", None)
    pairs = pairs.drop(columns=["parent_name", "mol_name"])

    logger.info(
        "Curated mechanism pairs: %d (%s)",
        len(pairs),
        ", ".join(f"{k} {v}" for k, v in pairs["target_route"].value_counts().items()),
    )
    return pairs[[
        "drug_chembl_id", "drug_name", "max_phase", "is_withdrawn", "gene_symbol",
        "gene_ensembl_id", "gene_uniprot_id", "gene_entrez_id", *NEW_COLUMNS,
        "mechanism_of_action", "interaction_type",
    ]]


def _pair_keys(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [df["drug_chembl_id"].astype(str), df["gene_symbol"].astype(str)],
        names=["drug_chembl_id", "gene_symbol"],
    )


def annotate_with_mechanisms(table: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Add ``target_route``, ``mechanism_action`` and ``mechanism_direction``.

    Rows are matched to pairs by parent ``drug_chembl_id`` and
    ``gene_symbol``, whatever the row's source; a row without a curated
    mechanism gets route ``none`` and no action or direction.  No other
    column changes.
    """
    out = table.copy()
    if out.empty:
        for col in NEW_COLUMNS:
            out[col] = pd.Series(dtype=object)
        return out
    indexed = pairs.set_index(_pair_keys(pairs))
    keys = _pair_keys(out)
    route = indexed["target_route"].reindex(keys)
    out["target_route"] = route.fillna("none").values
    for col in ("mechanism_action", "mechanism_direction"):
        vals = indexed[col].reindex(keys)
        out[col] = vals.astype(object).where(vals.notna(), None).values
    logger.info(
        "Drug table rows by curated route: %s",
        ", ".join(f"{k} {v}" for k, v in out["target_route"].value_counts().items()),
    )
    return out


def mechanism_rows_missing_from(table: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Curated mechanism pairs that the drug table lacks, as table rows.

    Branch C matches them beside the table's rows (record 042).  They
    carry ``source`` ``chembl`` and ``confidence`` ``medium`` (a curated
    mechanism without an affinity, the loader's rule), the table's
    columns and the three route columns.
    """
    from repogen.data.drug_loader import _finalize_schema

    present = set(_pair_keys(table)) if not table.empty else set()
    keep = [k not in present for k in _pair_keys(pairs)]
    extra = pairs.loc[keep].copy()
    if extra.empty:
        return extra
    extra["action_type"] = extra["mechanism_action"]
    extra["source"] = "chembl"
    extra["confidence"] = "medium"
    routes = extra[list(NEW_COLUMNS)].reset_index(drop=True)
    out = _finalize_schema(extra.drop(columns=list(NEW_COLUMNS)).reset_index(drop=True))
    out = pd.concat([out, routes], axis=1)
    logger.info(
        "Curated mechanism pairs missing from the drug table: %d (%s)",
        len(out),
        ", ".join(f"{k} {v}" for k, v in out["target_route"].value_counts().items()),
    )
    return out
