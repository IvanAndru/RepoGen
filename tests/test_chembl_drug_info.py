"""Tests for repogen.data.chembl_drug_info (records 042 to 044, 047)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from repogen.data.chembl_drug_info import (
    curated_activity_directions,
    inferred_enzyme_calls,
    inferred_kinase_binding_calls,
    load_chembl_drug_info,
    parse_dgidb_claims,
    protein_families,
)


@pytest.fixture()
def chembl_db(tmp_path: Path) -> Path:
    db = tmp_path / "chembl_mini.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT, max_phase REAL);
        CREATE TABLE molecule_hierarchy (molregno INTEGER, parent_molregno INTEGER);
        CREATE TABLE action_type (action_type TEXT, parent_type TEXT);
        CREATE TABLE target_dictionary (tid INTEGER, target_type TEXT, organism TEXT);
        CREATE TABLE target_components (tid INTEGER, component_id INTEGER);
        CREATE TABLE component_sequences (component_id INTEGER, component_type TEXT, accession TEXT);
        CREATE TABLE component_class (component_id INTEGER, protein_class_id INTEGER);
        CREATE TABLE protein_classification (protein_class_id INTEGER, parent_id INTEGER, pref_name TEXT);
        CREATE TABLE assays (assay_id INTEGER, tid INTEGER, assay_type TEXT);
        CREATE TABLE activities (molregno INTEGER, assay_id INTEGER, action_type TEXT,
                                 standard_type TEXT, pchembl_value REAL, data_validity_comment TEXT);
    """)
    conn.executemany("INSERT INTO molecule_dictionary VALUES (?,?,?)",
                     [(1, "CHEMBL1", 4), (2, "CHEMBL2", 4), (3, "CHEMBL3", None), (4, "CHEMBL4", 2)])
    conn.executemany("INSERT INTO molecule_hierarchy VALUES (?,?)", [(1, 1), (2, 1), (3, 3), (4, 4)])
    conn.executemany("INSERT INTO action_type VALUES (?,?)",
                     [("INHIBITOR", "NEGATIVE MODULATOR"), ("AGONIST", "POSITIVE MODULATOR")])
    conn.executemany("INSERT INTO target_dictionary VALUES (?,?,?)", [
        (10, "SINGLE PROTEIN", "Homo sapiens"), (20, "SINGLE PROTEIN", "Homo sapiens"),
        (30, "SINGLE PROTEIN", "Homo sapiens"), (40, "SINGLE PROTEIN", "Rattus norvegicus"),
        (50, "PROTEIN FAMILY", "Homo sapiens"),
    ])
    conn.executemany("INSERT INTO target_components VALUES (?,?)",
                     [(10, 100), (20, 200), (30, 300), (40, 400), (50, 100)])
    conn.executemany("INSERT INTO component_sequences VALUES (?,?,?)", [
        (100, "PROTEIN", "PKIN"), (200, "PROTEIN", "PENZ"), (300, "PROTEIN", "PGPCR"),
        (400, "PROTEIN", "PRAT"),
    ])
    conn.executemany("INSERT INTO protein_classification VALUES (?,?,?)", [
        (0, None, "Protein class"), (1, 0, "Enzyme"), (6, 1, "Kinase"), (11, 0, "Membrane receptor"),
        (1020, 11, "Family A G protein-coupled receptor"), (10, 1, "Oxidoreductase"),
    ])
    conn.executemany("INSERT INTO component_class VALUES (?,?)", [(100, 6), (200, 10), (300, 1020)])
    conn.executemany("INSERT INTO assays VALUES (?,?,?)", [
        (1, 10, "B"), (2, 10, "F"), (3, 20, "F"), (4, 30, "B"), (5, 40, "F"), (6, 50, "B"), (7, 20, "B"),
    ])
    conn.executemany("INSERT INTO activities VALUES (?,?,?,?,?,?)", [
        (2, 1, None, "Kd", 8.0, None),                # salt of CHEMBL1, kinase Kd 8.0
        (4, 1, None, "Kd", 7.0, None),                # kinase Kd below 7.5
        (1, 3, None, "IC50", 6.5, None),              # enzyme functional IC50 at the 'other' floor
        (4, 3, None, "IC50", 5.0, None),              # below it
        (3, 3, None, "IC50", 9.0, "Outside typical range"),  # flagged, ignored
        (1, 7, None, "IC50", 9.0, None),              # binding assay: not functional
        (1, 4, "AGONIST", "EC50", 7.0, None),          # curated agonist on the GPCR
        (4, 4, "AGONIST", "EC50", 7.0, None),
        (4, 4, "INHIBITOR", "IC50", 7.0, None),        # CHEMBL4 disagrees with itself
        (1, 5, "INHIBITOR", "IC50", 9.0, None),        # rat target, ignored
        (1, 6, "INHIBITOR", "IC50", 9.0, None),        # family target, ignored
    ])
    conn.commit()
    conn.close()
    return db


def _conn(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _pairs(df) -> dict:
    return {(p, u): d for p, u, d in df[["parent_id", "uniprot", "direction"]].itertuples(index=False)}


def test_families_and_enzymes(chembl_db: Path) -> None:
    family, enzymes = protein_families(_conn(chembl_db))
    assert family == {"PKIN": "kinase", "PENZ": "other", "PGPCR": "gpcr"}
    assert enzymes == {"PKIN", "PENZ"}


def test_curated_activity(chembl_db: Path) -> None:
    calls = _pairs(curated_activity_directions(
        _conn(chembl_db), {"INHIBITOR": "NEGATIVE MODULATOR", "AGONIST": "POSITIVE MODULATOR"}))
    assert calls == {("CHEMBL1", "PGPCR"): "positive", ("CHEMBL4", "PGPCR"): "none"}


def test_inferred_enzyme(chembl_db: Path) -> None:
    family, enzymes = protein_families(_conn(chembl_db))
    assert _pairs(inferred_enzyme_calls(_conn(chembl_db), family, enzymes)) == {("CHEMBL1", "PENZ"): "negative"}


def test_inferred_kinase_binding_uses_the_parent(chembl_db: Path) -> None:
    family, _ = protein_families(_conn(chembl_db))
    assert _pairs(inferred_kinase_binding_calls(_conn(chembl_db), family)) == {("CHEMBL1", "PKIN"): "negative"}


def test_load_keeps_directions_per_class(chembl_db: Path) -> None:
    info = load_chembl_drug_info(chembl_db)
    assert info.phase == {"CHEMBL1": 4.0, "CHEMBL2": 4.0, "CHEMBL4": 2.0}
    assert info.binder_directions[("CHEMBL1", "PKIN")] == {"inferred_kinase_binding": "negative"}
    assert info.floor_for("PKIN") == 7.5 and info.floor_for("UNKNOWN") == 6.0
    assert load_chembl_drug_info(chembl_db, []).binder_directions == {}
    with pytest.raises(ValueError):
        load_chembl_drug_info(chembl_db, ["guesswork"])


def test_parse_dgidb_claims() -> None:
    assert parse_dgidb_claims(["TTD:", "ChEMBL:Inhibitor"]) == [("TTD", ""), ("ChEMBL", "inhibitor")]
    assert parse_dgidb_claims(None) == [] and parse_dgidb_claims(float("nan")) == []
