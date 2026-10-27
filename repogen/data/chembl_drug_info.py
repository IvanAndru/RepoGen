"""What Branch C's drug layer reads from ChEMBL and DGIdb claims.

Records 042 to 044 and 047: the clinical phase of a DGIdb drug from
ChEMBL's molecule record; the target family that sets a binder's potency
floor; ChEMBL's class of each action type, which also directs typed DGIdb
claims; and, for the record 043 classes a run asks for, a direction for
binders from activity data.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CLASS_DIRECTION = {"POSITIVE MODULATOR": "positive", "NEGATIVE MODULATOR": "negative"}

# DGIdb sources by what their claims assert (record 044).
DGIDB_TARGET_SOURCES: frozenset[str] = frozenset(
    {"ChEMBL", "GuideToPharmacology", "TTD", "TEND", "TdgClinicalTrial", "TALC"}
)
DGIDB_BINDING_SOURCES: frozenset[str] = frozenset({"DTC"})

# Potency floors for reporting a binder, by target family (record 042, after
# Oprea et al. 2018): pChEMBL of 30 nM, 100 nM, 10 uM and 1 uM.
FAMILY_FLOORS: dict[str, float] = {
    "kinase": 7.5, "gpcr": 7.0, "nuclear_receptor": 7.0, "ion_channel": 5.0, "other": 6.0,
}
_GPCR_CLASSES = frozenset({
    "Family A G protein-coupled receptor", "Family B G protein-coupled receptor",
    "Family C G protein-coupled receptor", "Frizzled family G protein-coupled receptor",
    "Taste family G protein-coupled receptor",
})

# Binder direction classes (record 043), behind validation D1; earlier wins.
BINDER_DIRECTION_CLASSES: tuple[str, ...] = (
    "curated_activity", "inferred_enzyme", "inferred_kinase_binding",
)


def parse_dgidb_claims(value) -> list[tuple[str, str]]:
    """``dgidb_sources`` entries (``"source:type"``) as (source, type) pairs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    items = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    out = []
    for item in items:
        source, _, itype = str(item).partition(":")
        out.append((source.strip(), itype.strip().lower()))
    return out


def dgidb_direction(types: list[str], action_classes: Optional[dict[str, str]] = None) -> str:
    """Direction of typed DGIdb claims (record 047, alternative 4).

    A term that is a ChEMBL action type takes ChEMBL's class; other terms
    (antibody, binder, cleavage, potentiator, vaccine, ...) have none.
    Claims, or parts of a combined type, that disagree, or any undirected
    part, give none.
    """
    classes = action_classes or {}
    dirs = set()
    for t in types:
        for part in str(t).split("|"):
            part = part.strip()
            if part:
                dirs.add(_CLASS_DIRECTION.get(classes.get(part.upper()), "none"))
    if len(dirs) == 1 and "none" not in dirs:
        return dirs.pop()
    return "none"


class ChemblDrugInfo:
    """The ChEMBL facts the drug layer needs, read once per run.

    - ``phase``: molecule ``chembl_id`` -> ``max_phase`` as ChEMBL records
      it (record 044 takes a DGIdb drug's phase from here).
    - ``family``: UniProt accession -> ``kinase``, ``gpcr``,
      ``nuclear_receptor``, ``ion_channel`` or ``other`` (record 042's
      floors; a protein in two families takes the stricter floor).
    - ``action_classes``: ChEMBL action type -> its class.
    - ``binder_directions``: (parent ``chembl_id``, UniProt) -> {class:
      direction} for the record 043 classes that were loaded.
    """

    def __init__(self, phase, family, action_classes, binder_directions=None):
        self.phase = phase
        self.family = family
        self.action_classes = action_classes
        self.binder_directions = binder_directions or {}

    def floor_for(self, uniprot) -> float:
        return FAMILY_FLOORS[self.family.get(str(uniprot), "other")]


_Q_PHASE = "SELECT chembl_id, max_phase FROM molecule_dictionary WHERE max_phase IS NOT NULL"
_Q_ACTION = "SELECT action_type, parent_type FROM action_type"
_Q_CLASSES = """\
SELECT cs.accession, cc.protein_class_id
FROM component_sequences cs
JOIN component_class cc ON cc.component_id = cs.component_id
WHERE cs.component_type = 'PROTEIN' AND cs.accession IS NOT NULL
"""
_Q_TREE = "SELECT protein_class_id, parent_id, pref_name FROM protein_classification"
_Q_ACTIVITY_PAIRS = """\
SELECT COALESCE(p.chembl_id, md.chembl_id) AS parent_id, cs.accession AS uniprot,
       act.action_type, act.standard_type, act.pchembl_value, a.assay_type
FROM activities act
JOIN molecule_dictionary md ON md.molregno = act.molregno
LEFT JOIN molecule_hierarchy mh ON mh.molregno = md.molregno
LEFT JOIN molecule_dictionary p ON p.molregno = mh.parent_molregno
JOIN assays a ON a.assay_id = act.assay_id
JOIN target_dictionary td ON td.tid = a.tid
JOIN target_components tc ON tc.tid = td.tid
JOIN component_sequences cs ON cs.component_id = tc.component_id
WHERE td.target_type = 'SINGLE PROTEIN' AND td.organism = 'Homo sapiens'
  AND act.data_validity_comment IS NULL
  AND {where}
"""


def protein_families(conn: sqlite3.Connection) -> tuple[dict[str, str], set[str]]:
    """UniProt -> potency-floor family, and the set of enzyme accessions."""
    tree = pd.read_sql_query(_Q_TREE, conn)
    parent = dict(zip(tree["protein_class_id"], tree["parent_id"]))
    name = dict(zip(tree["protein_class_id"], tree["pref_name"]))

    def lineage(cid) -> set[str]:
        names, seen = set(), set()
        while cid is not None and not (isinstance(cid, float) and np.isnan(cid)) and cid not in seen:
            seen.add(cid)
            names.add(name.get(cid, ""))
            cid = parent.get(cid)
        return names

    cache: dict = {}
    fams: dict[str, set[str]] = {}
    enzymes: set[str] = set()
    for acc, cid in pd.read_sql_query(_Q_CLASSES, conn).itertuples(index=False):
        if cid not in cache:
            cache[cid] = lineage(cid)
        names = cache[cid]
        f = fams.setdefault(acc, set())
        if "Kinase" in names:
            f.add("kinase")
        if names & _GPCR_CLASSES:
            f.add("gpcr")
        if "Nuclear receptor" in names:
            f.add("nuclear_receptor")
        if "Ion channel" in names:
            f.add("ion_channel")
        if "Enzyme" in names:
            enzymes.add(acc)
    family = {acc: (max(fs, key=lambda x: FAMILY_FLOORS[x]) if fs else "other")
              for acc, fs in fams.items()}
    return family, enzymes


def _one_direction(values) -> str:
    dirs = set(values) - {"none"}
    return dirs.pop() if len(dirs) == 1 else "none"


def curated_activity_directions(conn: sqlite3.Connection,
                                action_classes: dict[str, str]) -> pd.DataFrame:
    """Record 043, class ``curated_activity``: ChEMBL's curated action types
    on individual activities; one direction per pair, none if they disagree."""
    act = pd.read_sql_query(_Q_ACTIVITY_PAIRS.format(where="act.action_type IS NOT NULL"), conn)
    act["direction"] = act["action_type"].map(
        lambda a: _CLASS_DIRECTION.get(action_classes.get(str(a)), "none"))
    agg = act.groupby(["parent_id", "uniprot"])["direction"].apply(_one_direction)
    return agg.rename("direction").reset_index()


def inferred_enzyme_calls(conn: sqlite3.Connection, family: dict[str, str],
                          enzymes: set[str]) -> pd.DataFrame:
    """Record 043, class ``inferred_enzyme``: an inhibitor call for an
    enzyme with a functional IC50 or Ki at or above its family floor."""
    act = pd.read_sql_query(_Q_ACTIVITY_PAIRS.format(
        where="a.assay_type = 'F' AND act.standard_type IN ('IC50', 'Ki') "
              "AND act.pchembl_value IS NOT NULL"), conn)
    act = act.loc[act["uniprot"].isin(enzymes)]
    best = act.groupby(["parent_id", "uniprot"])["pchembl_value"].max().reset_index()
    floor = best["uniprot"].map(lambda u: FAMILY_FLOORS[family.get(u, "other")])
    best = best.loc[best["pchembl_value"] >= floor]
    return best.assign(direction="negative")[["parent_id", "uniprot", "direction"]]


def inferred_kinase_binding_calls(conn: sqlite3.Connection, family: dict[str, str]) -> pd.DataFrame:
    """Record 043, class ``inferred_kinase_binding``: a kinase binding-panel
    Kd at pKd >= 7.5 read as inhibition."""
    act = pd.read_sql_query(_Q_ACTIVITY_PAIRS.format(
        where="a.assay_type = 'B' AND act.standard_type = 'Kd' "
              "AND act.pchembl_value IS NOT NULL"), conn)
    act = act.loc[act["uniprot"].map(lambda u: family.get(u) == "kinase")]
    best = act.groupby(["parent_id", "uniprot"])["pchembl_value"].max().reset_index()
    best = best.loc[best["pchembl_value"] >= FAMILY_FLOORS["kinase"]]
    return best.assign(direction="negative")[["parent_id", "uniprot", "direction"]]


def load_chembl_drug_info(sqlite_path: Path, binder_classes=None) -> ChemblDrugInfo:
    """Read phases, protein families, action classes and, for the record 043
    classes asked for (all of them when ``None``), binder directions. Which
    classes a match may use is the run's configuration, so loading more
    classes never changes a result."""
    if binder_classes is None:
        binder_classes = BINDER_DIRECTION_CLASSES
    unknown = set(binder_classes) - set(BINDER_DIRECTION_CLASSES)
    if unknown:
        raise ValueError(f"unknown binder direction classes: {sorted(unknown)}")
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        ph = pd.read_sql_query(_Q_PHASE, conn)
        phase = dict(zip(ph["chembl_id"], ph["max_phase"].astype(float)))
        ac = pd.read_sql_query(_Q_ACTION, conn)
        action_classes = dict(zip(ac["action_type"], ac["parent_type"]))
        family, enzymes = protein_families(conn)
        directions: dict[tuple[str, str], dict[str, str]] = {}
        for cls in BINDER_DIRECTION_CLASSES:
            if cls not in binder_classes:
                continue
            if cls == "curated_activity":
                calls = curated_activity_directions(conn, action_classes)
            elif cls == "inferred_enzyme":
                calls = inferred_enzyme_calls(conn, family, enzymes)
            else:
                calls = inferred_kinase_binding_calls(conn, family)
            for pid, acc, d in calls[["parent_id", "uniprot", "direction"]].itertuples(index=False):
                directions.setdefault((pid, acc), {})[cls] = d
            logger.info("Binder direction class %s: %d pairs", cls, len(calls))
    finally:
        conn.close()
    logger.info("ChEMBL drug info: %d molecule phases, %d protein families",
                len(phase), len(family))
    return ChemblDrugInfo(phase, family, action_classes, directions)
