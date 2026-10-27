"""Tests for repogen.data.drug_mechanisms (records 042, 045, 047)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from repogen.analysis.drug_enrichment import build_drug_gene_sets
from repogen.data.drug_mechanisms import (
    NEW_COLUMNS,
    annotate_with_mechanisms,
    load_curated_mechanism_pairs,
    mechanism_rows_missing_from,
)

# accession -> (symbol, ensembl, uniprot, entrez)
_GENES = {
    "P04035": ("HMGCR", "ENSG00000113161", "P04035", 3156),
    "P14867": ("GABRA1", "ENSG00000022355", "P14867", 2554),
    "Q16445": ("GABRA6", "ENSG00000145863", "Q16445", 2559),
    "P18507": ("GABRG2", "ENSG00000113327", "P18507", 2566),
    "Q05586": ("GRIN1", "ENSG00000176884", "Q05586", 2902),
    "Q12879": ("GRIN2A", "ENSG00000183454", "Q12879", 2903),
    "ENSG00000169174": ("PCSK9", "ENSG00000169174", "Q8NBP7", 255738),
    "P28223": ("HTR2A", "ENSG00000102468", "P28223", 3356),
    "P14416": ("DRD2", "ENSG00000149295", "P14416", 1813),
    "P35462": ("DRD3", "ENSG00000151577", "P35462", 1814),
    "P08908": ("HTR1A", "ENSG00000178394", "P08908", 3350),
    "Q01959": ("SLC6A3", "ENSG00000142319", "Q01959", 6531),
    "P23219": ("PTGS1", "ENSG00000095303", "P23219", 5742),
}


class _Converter:
    def get_full_record(self, gene_id: str, id_type: str):
        g = _GENES.get(gene_id)
        if g is None:
            return None
        return {"symbol": g[0], "ensembl": g[1], "uniprot": g[2], "entrez": g[3]}


@pytest.fixture()
def chembl_db(tmp_path: Path) -> Path:
    db = tmp_path / "chembl_mini.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT, pref_name TEXT,
                                          max_phase REAL);
        CREATE TABLE molecule_hierarchy (molregno INTEGER, parent_molregno INTEGER);
        CREATE TABLE drug_mechanism (mec_id INTEGER, molregno INTEGER, tid INTEGER,
                                     site_id INTEGER, action_type TEXT, mechanism_of_action TEXT);
        CREATE TABLE action_type (action_type TEXT, parent_type TEXT);
        CREATE TABLE target_dictionary (tid INTEGER, chembl_id TEXT, pref_name TEXT,
                                        target_type TEXT, organism TEXT);
        CREATE TABLE target_components (tid INTEGER, component_id INTEGER);
        CREATE TABLE component_sequences (component_id INTEGER, component_type TEXT,
                                          accession TEXT);
        CREATE TABLE site_components (site_id INTEGER, component_id INTEGER);
        CREATE TABLE drug_warning (molregno INTEGER, warning_type TEXT);
    """)
    conn.executemany("INSERT INTO molecule_dictionary VALUES (?,?,?,?)", [
        (1, "CHEMBL1487", "ATORVASTATIN", 4), (2, "CHEMBL393220", "ATORVASTATIN CALCIUM", 4),
        (3, "CHEMBL12", "DIAZEPAM", 4), (4, "CHEMBL742", "KETAMINE", 4),
        (5, "CHEMBL3989974", "INCLISIRAN", 4), (6, "CHEMBL54", "HALOPERIDOL", 4),
        (7, "CHEMBL900", "MIXED DRUG", 2), (8, "CHEMBL1112", "ARIPIPRAZOLE", 4),
        (9, "CHEMBL405", "AMPHETAMINE", 4), (10, "CHEMBL901", "MODULATING INHIBITOR", 0.5),
        (11, "CHEMBL902", "RAT ONLY", 4), (12, "CHEMBL903", "NSAID", 4),
    ])
    conn.executemany("INSERT INTO molecule_hierarchy VALUES (?,?)",
                     [(1, 1), (2, 1)] + [(i, i) for i in range(3, 13)])
    conn.executemany("INSERT INTO action_type VALUES (?,?)", [
        ("INHIBITOR", "NEGATIVE MODULATOR"), ("ANTAGONIST", "NEGATIVE MODULATOR"),
        ("AGONIST", "POSITIVE MODULATOR"), ("INVERSE AGONIST", "NEGATIVE MODULATOR"),
        ("POSITIVE ALLOSTERIC MODULATOR", "POSITIVE MODULATOR"),
        ("NEGATIVE ALLOSTERIC MODULATOR", "NEGATIVE MODULATOR"),
        ("RNAI INHIBITOR", "NEGATIVE MODULATOR"), ("PARTIAL AGONIST", "POSITIVE MODULATOR"),
        ("RELEASING AGENT", "NEGATIVE MODULATOR"), ("MODULATOR", "OTHER"),
    ])
    conn.executemany("INSERT INTO target_dictionary VALUES (?,?,?,?,?)", [
        (10, "T10", "HMG-CoA reductase", "SINGLE PROTEIN", "Homo sapiens"),
        (20, "T20", "GABA-A receptor", "PROTEIN COMPLEX GROUP", "Homo sapiens"),
        (30, "T30", "NMDA receptor", "PROTEIN COMPLEX GROUP", "Homo sapiens"),
        (40, "T40", "PCSK9 mRNA", "NUCLEIC-ACID", "Homo sapiens"),
        (50, "T50", "5-HT2A", "SINGLE PROTEIN", "Homo sapiens"),
        (60, "T60", "D2-like dopamine receptor", "PROTEIN FAMILY", "Homo sapiens"),
        (70, "T70", "Dopamine D2 receptor", "SINGLE PROTEIN", "Homo sapiens"),
        (80, "T80", "5-HT1A", "SINGLE PROTEIN", "Homo sapiens"),
        (90, "T90", "Dopamine transporter", "SINGLE PROTEIN", "Homo sapiens"),
        (95, "T95", "rat target", "SINGLE PROTEIN", "Rattus norvegicus"),
        (96, "T96", "Cyclooxygenase", "PROTEIN FAMILY", "Homo sapiens"),
    ])
    conn.executemany("INSERT INTO target_components VALUES (?,?)", [
        (10, 100), (20, 201), (20, 202), (20, 203), (30, 301), (30, 302), (40, 401),
        (50, 501), (60, 601), (60, 602), (70, 601), (80, 801), (90, 901), (95, 950),
        (96, 961),
    ])
    conn.executemany("INSERT INTO component_sequences VALUES (?,?,?)", [
        (100, "PROTEIN", "P04035"), (201, "PROTEIN", "P14867"), (202, "PROTEIN", "Q16445"),
        (203, "PROTEIN", "P18507"), (301, "PROTEIN", "Q05586"), (302, "PROTEIN", "Q12879"),
        (401, "RNA", "ENSG00000169174"), (501, "PROTEIN", "P28223"), (601, "PROTEIN", "P14416"),
        (602, "PROTEIN", "P35462"), (801, "PROTEIN", "P08908"), (901, "PROTEIN", "Q01959"),
        (950, "PROTEIN", "P04035"), (961, "PROTEIN", "P23219"),
    ])
    conn.executemany("INSERT INTO site_components VALUES (?,?)", [(1, 201), (1, 203)])
    conn.executemany("INSERT INTO drug_mechanism VALUES (?,?,?,?,?,?)", [
        (1, 2, 10, None, "INHIBITOR", "HMG-CoA reductase inhibitor"),      # salt's mechanism
        (2, 3, 20, 1, "POSITIVE ALLOSTERIC MODULATOR", "GABA-A PAM"),
        (3, 4, 30, None, "NEGATIVE ALLOSTERIC MODULATOR", "NMDA NAM"),
        (4, 5, 40, None, "RNAI INHIBITOR", "PCSK9 mRNA RNAi inhibitor"),
        (5, 6, 50, None, "ANTAGONIST", "5-HT2A antagonist"),
        (6, 6, 60, None, "INVERSE AGONIST", "D2-like inverse agonist"),
        (7, 6, 70, None, "ANTAGONIST", "D2 antagonist"),
        (8, 7, 80, None, "AGONIST", "5-HT1A agonist"),
        (9, 7, 80, None, "ANTAGONIST", "5-HT1A antagonist"),
        (10, 8, 80, None, "PARTIAL AGONIST", "5-HT1A partial agonist"),
        (11, 9, 90, None, "RELEASING AGENT", "DAT releasing agent"),
        (12, 10, 90, None, "MODULATOR", "DAT modulator"),
        (13, 10, 90, None, "INHIBITOR", "DAT inhibitor"),
        (14, 11, 95, None, "INHIBITOR", "rat inhibitor"),
        (15, 12, 96, None, "INHIBITOR", "COX inhibitor"),
    ])
    conn.commit()
    conn.close()
    return db


@pytest.fixture()
def pairs(chembl_db: Path) -> pd.DataFrame:
    return load_curated_mechanism_pairs(chembl_db, _Converter())


def _pair(pairs: pd.DataFrame, drug: str, gene: str) -> pd.Series:
    hit = pairs[(pairs["drug_chembl_id"] == drug) & (pairs["gene_symbol"] == gene)]
    assert len(hit) == 1, (drug, gene, len(hit))
    return hit.iloc[0]


class TestRoutes:
    def test_salt_mechanism_lands_on_parent(self, pairs: pd.DataFrame) -> None:
        p = _pair(pairs, "CHEMBL1487", "HMGCR")
        assert p["target_route"] == "single_protein"
        assert p["mechanism_direction"] == "negative"
        assert "CHEMBL393220" not in set(pairs["drug_chembl_id"])

    def test_binding_site_members_and_off_site(self, pairs: pd.DataFrame) -> None:
        assert _pair(pairs, "CHEMBL12", "GABRA1")["target_route"] == "site_member"
        assert _pair(pairs, "CHEMBL12", "GABRG2")["target_route"] == "site_member"
        off = _pair(pairs, "CHEMBL12", "GABRA6")
        assert off["target_route"] == "off_site"
        assert off["mechanism_direction"] == "none"
        assert _pair(pairs, "CHEMBL12", "GABRA1")["mechanism_direction"] == "positive"

    def test_complex_without_site(self, pairs: pd.DataFrame) -> None:
        for gene in ("GRIN1", "GRIN2A"):
            p = _pair(pairs, "CHEMBL742", gene)
            assert p["target_route"] == "complex_member"
            assert p["mechanism_direction"] == "negative"

    def test_nucleic_acid_target(self, pairs: pd.DataFrame) -> None:
        p = _pair(pairs, "CHEMBL3989974", "PCSK9")
        assert p["target_route"] == "nucleic_acid"
        assert p["mechanism_direction"] == "negative"

    def test_single_protein_beats_family(self, pairs: pd.DataFrame) -> None:
        assert _pair(pairs, "CHEMBL54", "DRD2")["target_route"] == "single_protein"
        assert _pair(pairs, "CHEMBL54", "DRD2")["mechanism_action"] == "ANTAGONIST"
        assert _pair(pairs, "CHEMBL54", "DRD3")["target_route"] == "family_member"

    def test_non_human_target_ignored(self, pairs: pd.DataFrame) -> None:
        assert "CHEMBL902" not in set(pairs["drug_chembl_id"])

    def test_phase_cast_like_the_table(self, pairs: pd.DataFrame) -> None:
        assert _pair(pairs, "CHEMBL901", "SLC6A3")["max_phase"] == 0


class TestDirections:
    def test_opposite_mechanisms_give_none(self, pairs: pd.DataFrame) -> None:
        assert _pair(pairs, "CHEMBL900", "HTR1A")["mechanism_direction"] == "none"

    def test_partial_agonist_raises(self, pairs: pd.DataFrame) -> None:
        assert _pair(pairs, "CHEMBL1112", "HTR1A")["mechanism_direction"] == "positive"

    def test_releasing_agent_lowers(self, pairs: pd.DataFrame) -> None:
        assert _pair(pairs, "CHEMBL405", "SLC6A3")["mechanism_direction"] == "negative"

    def test_directed_wins_over_undirected(self, pairs: pd.DataFrame) -> None:
        p = _pair(pairs, "CHEMBL901", "SLC6A3")
        assert p["mechanism_direction"] == "negative"
        assert p["mechanism_action"] == "INHIBITOR|MODULATOR"


def _table() -> pd.DataFrame:
    return pd.DataFrame({
        "drug_name": ["ATORVASTATIN", "DIAZEPAM", "OTHER DRUG"],
        "drug_chembl_id": ["CHEMBL1487", "CHEMBL12", "CHEMBL777"],
        "gene_symbol": ["HMGCR", "GABRA6", "DRD2"],
        "gene_entrez_id": pd.array([3156, 2559, 1813], dtype="Int64"),
        "interaction_type": ["other", "positive_modulator", "antagonist"],
        "max_phase": [4, 0, 4],
        "source": ["chembl", "dgidb", "chembl"],
        "confidence": ["high", "low", "medium"],
    })


class TestAnnotate:
    def test_columns_added_values_kept(self, pairs: pd.DataFrame) -> None:
        table = _table()
        out = annotate_with_mechanisms(table, pairs)
        pd.testing.assert_frame_equal(out[table.columns], table)
        assert list(out.columns) == list(table.columns) + list(NEW_COLUMNS)

    def test_lost_salt_direction_restored(self, pairs: pd.DataFrame) -> None:
        out = annotate_with_mechanisms(_table(), pairs)
        row = out[out["gene_symbol"] == "HMGCR"].iloc[0]
        assert row["interaction_type"] == "other"
        assert row["mechanism_direction"] == "negative"

    def test_dgidb_row_of_member_pair_carries_route(self, pairs: pd.DataFrame) -> None:
        out = annotate_with_mechanisms(_table(), pairs)
        row = out[out["gene_symbol"] == "GABRA6"].iloc[0]
        assert row["source"] == "dgidb"
        assert row["target_route"] == "off_site"

    def test_row_without_mechanism(self, pairs: pd.DataFrame) -> None:
        out = annotate_with_mechanisms(_table(), pairs)
        row = out[out["drug_chembl_id"] == "CHEMBL777"].iloc[0]
        assert row["target_route"] == "none"
        assert pd.isna(row["mechanism_direction"])
        assert pd.isna(row["mechanism_action"])

    def test_branch_a_drug_sets_unchanged(self, pairs: pd.DataFrame) -> None:
        table = _table()
        genes = pd.DataFrame({"gene_entrez_id": [3156, 2559, 1813]})
        before = build_drug_gene_sets(table, genes, min_genes_per_drug=1)
        after = build_drug_gene_sets(annotate_with_mechanisms(table, pairs), genes,
                                     min_genes_per_drug=1)
        assert before == after


class TestMissingRows:
    def test_pairs_absent_from_table(self, pairs: pd.DataFrame) -> None:
        extra = mechanism_rows_missing_from(_table(), pairs)
        keys = set(zip(extra["drug_chembl_id"], extra["gene_symbol"]))
        assert ("CHEMBL3989974", "PCSK9") in keys
        assert ("CHEMBL12", "GABRA1") in keys
        assert ("CHEMBL1487", "HMGCR") not in keys
        assert ("CHEMBL12", "GABRA6") not in keys
        assert set(extra["source"]) == {"chembl"}
        assert set(extra["confidence"]) == {"medium"}
        for col in ("drug_name", "gene_ensembl_id", "max_phase", *NEW_COLUMNS):
            assert col in extra.columns

    def test_provenance_survives_parquet(self, pairs: pd.DataFrame, tmp_path: Path) -> None:
        extra = mechanism_rows_missing_from(_table(), pairs)
        extra.attrs["provenance"] = {"curated_pairs": int(len(pairs))}
        path = tmp_path / "routes.parquet"
        extra.to_parquet(path, engine="pyarrow", index=False)
        back = pd.read_parquet(path)
        assert back.attrs.get("provenance") == {"curated_pairs": int(len(pairs))}
        assert len(back) == len(extra)
