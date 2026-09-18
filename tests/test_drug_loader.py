"""Tests for repogen.data.drug_loader."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import numpy as np

from repogen.data.drug_loader import (
    ChemblNameIndex,
    _best_affinity_per_pair,
    _build_chembl_synonym_index,
    _collapse_to_parent,
    _deduplicate_records,
    _enrich_pubchem_cid,
    _extract_chembl_id,
    _finalize_schema,
    _join_mechanism_affinity,
    _match_dgidb_to_chembl,
    _match_expression_source_to_chembl,
    _normalize_drug_name,
    _parse_pipe_list,
    _parse_unichem_mapping,
    _postprocess_chembl,
    _propagate_chembl_atc_to_remapped,
    _query_in_chunks,
    _resolve_synonym_index,
    _sanitize_placeholder_id,
    _standardize_interaction_type,
    load_chembl,
    load_chembl_synonym_index,
    load_creeds,
    load_dgidb,
    load_drug_targets,
    load_dsigdb,
    load_pdsp,
    merge_sources,
)


class TestStandardizeInteractionType:
    """Tests for interaction type normalisation."""

    def test_chembl_uppercase(self) -> None:
        assert _standardize_interaction_type("INHIBITOR") == "inhibitor"
        assert _standardize_interaction_type("ANTAGONIST") == "antagonist"
        assert _standardize_interaction_type("FULL AGONIST") == "agonist"
        assert _standardize_interaction_type("PARTIAL AGONIST") == "partial_agonist"
        assert _standardize_interaction_type("BLOCKER") == "blocker"

    def test_dgidb_lowercase(self) -> None:
        assert _standardize_interaction_type("inhibitor") == "inhibitor"
        assert _standardize_interaction_type("channel blocker") == "blocker"

    def test_unknown_returns_other(self) -> None:
        assert _standardize_interaction_type("TOTALLY_NEW_TYPE") == "other"

    def test_none_returns_other(self) -> None:
        assert _standardize_interaction_type(None) == "other"
        assert _standardize_interaction_type("") == "other"


class TestParsePipeList:
    """Tests for pipe-delimited string parsing."""

    def test_single_value(self) -> None:
        assert _parse_pipe_list("N05AH01") == ["N05AH01"]

    def test_multiple_values(self) -> None:
        result = _parse_pipe_list("N05AH01|N05AX08")
        assert result == ["N05AH01", "N05AX08"]

    def test_none_returns_none(self) -> None:
        assert _parse_pipe_list(None) is None
        assert _parse_pipe_list("") is None
        assert _parse_pipe_list(float("nan")) is None


class TestBestAffinityPerPair:
    """Tests for keeping best affinity measurement per drug-target pair."""

    def test_keeps_highest_pchembl(self) -> None:
        df = pd.DataFrame({
            "chembl_id": ["CHEMBL1", "CHEMBL1", "CHEMBL2"],
            "uniprot_id": ["P28223", "P28223", "P15056"],
            "affinity_type": ["Ki", "IC50", "Ki"],
            "affinity_value": [10.0, 100.0, 50.0],
            "affinity_unit": ["nM", "nM", "nM"],
            "pchembl_value": [8.0, 7.0, 7.3],
        })
        result = _best_affinity_per_pair(df)
        assert len(result) == 2
        row1 = result[result["drug_chembl_id"] == "CHEMBL1"].iloc[0]
        assert row1["pchembl_value"] == 8.0

    def test_empty_input(self) -> None:
        result = _best_affinity_per_pair(pd.DataFrame())
        assert result.empty


class TestJoinMechanismAffinity:
    """Tests for full outer join of mechanism and affinity data."""

    def test_both_populated(self) -> None:
        mech = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "uniprot_id": ["P28223"],
            "mechanism_of_action": ["5-HT2A antagonist"],
            "interaction_type": ["ANTAGONIST"],
        })
        aff = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "uniprot_id": ["P28223"],
            "drug_name": ["risperidone"],
            "max_phase": [4],
            "pchembl_value": [8.5],
        })
        result = _join_mechanism_affinity(mech, aff)
        assert len(result) == 1
        assert "mechanism_of_action" in result.columns
        assert "pchembl_value" in result.columns
        assert result["_from_mechanism"].iloc[0] == True  # noqa: E712

    def test_mechanism_only(self) -> None:
        mech = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "uniprot_id": ["P28223"],
            "mechanism_of_action": ["test"],
            "interaction_type": ["ANTAGONIST"],
        })
        result = _join_mechanism_affinity(mech, pd.DataFrame())
        assert len(result) == 1
        assert "pchembl_value" in result.columns
        assert result["_from_mechanism"].iloc[0] == True  # noqa: E712

    def test_affinity_only_branch(self) -> None:
        aff = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL99"],
            "uniprot_id": ["P12345"],
            "drug_name": ["compound_x"],
            "max_phase": [0],
            "pchembl_value": [7.0],
        })
        result = _join_mechanism_affinity(pd.DataFrame(), aff)
        assert len(result) == 1
        assert result["_from_mechanism"].iloc[0] == False  # noqa: E712

    def test_coalesces_metadata(self) -> None:
        mech = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "uniprot_id": ["P28223"],
            "drug_name": ["risperidone"],
            "max_phase": [4],
            "mechanism_of_action": ["5-HT2A antagonist"],
            "interaction_type": ["ANTAGONIST"],
        })
        aff = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "uniprot_id": ["P28223"],
            "drug_name": ["risperidone"],
            "max_phase": [4],
            "pchembl_value": [8.5],
        })
        result = _join_mechanism_affinity(mech, aff)
        assert result["drug_name"].iloc[0] == "risperidone"
        assert result["max_phase"].iloc[0] == 4
        assert "drug_name_aff" not in result.columns

    def test_coalesces_fills_missing_from_affinity(self) -> None:
        mech = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "uniprot_id": ["P28223"],
            "drug_name": [None],
            "max_phase": [None],
            "mechanism_of_action": ["test"],
            "interaction_type": ["ANTAGONIST"],
        })
        aff = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "uniprot_id": ["P28223"],
            "drug_name": ["risperidone"],
            "max_phase": [4],
            "pchembl_value": [8.5],
        })
        result = _join_mechanism_affinity(mech, aff)
        assert result["drug_name"].iloc[0] == "risperidone"
        assert result["max_phase"].iloc[0] == 4

    def test_both_empty(self) -> None:
        result = _join_mechanism_affinity(pd.DataFrame(), pd.DataFrame())
        assert result.empty


class TestAssignConfidence:
    """Tests for ChEMBL confidence assignment (np.select vectorised)."""

    @staticmethod
    def _assign(mech, pchembl):
        """Replicate the np.select logic used in _postprocess_chembl."""
        df = pd.DataFrame({"mechanism_of_action": [mech], "pchembl_value": [pchembl]})
        has_mech = df["mechanism_of_action"].notna()
        has_aff = df["pchembl_value"].notna()
        return np.select(
            [has_mech & has_aff, has_mech | has_aff],
            ["high", "medium"],
            default="low",
        )[0]

    def test_high_confidence(self) -> None:
        assert self._assign("5-HT2A antagonist", 8.5) == "high"

    def test_medium_mechanism_only(self) -> None:
        assert self._assign("test", None) == "medium"

    def test_medium_affinity_only(self) -> None:
        assert self._assign(None, 7.0) == "medium"

    def test_low_neither(self) -> None:
        assert self._assign(None, None) == "low"


class TestFinalizeSchema:
    """Tests for schema finalisation."""

    def test_adds_missing_columns(self) -> None:
        df = pd.DataFrame({
            "drug_name": ["Test"],
            "drug_chembl_id": ["CHEMBL1"],
            "gene_symbol": ["HTR2A"],
            "source": ["chembl"],
            "confidence": ["high"],
        })
        result = _finalize_schema(df)
        assert "drug_inchikey" in result.columns
        assert "atc_codes" in result.columns
        assert "gene_entrez_id" in result.columns

    def test_empty_input(self) -> None:
        result = _finalize_schema(pd.DataFrame())
        assert result.empty

    def test_coerces_float_entrez_to_int64(self) -> None:
        """Float/object gene_entrez_id must be coerced to nullable Int64."""
        df = pd.DataFrame({
            "drug_name": ["A", "B", "C"],
            "drug_chembl_id": ["C1", "C2", "C3"],
            "gene_symbol": ["HTR2A", "DRD2", "SLC6A4"],
            "gene_entrez_id": [3356.0, np.nan, 6532.0],
            "source": ["chembl"] * 3,
            "confidence": ["high"] * 3,
        })
        assert df["gene_entrez_id"].dtype == np.float64
        result = _finalize_schema(df)
        assert result["gene_entrez_id"].dtype == pd.Int64Dtype()
        assert result["gene_entrez_id"].iloc[0] == 3356
        assert pd.isna(result["gene_entrez_id"].iloc[1])
        assert result["gene_entrez_id"].iloc[2] == 6532

    def test_coerces_object_entrez_to_int64(self) -> None:
        """String/object gene_entrez_id must be coerced to nullable Int64."""
        df = pd.DataFrame({
            "drug_name": ["A", "B"],
            "drug_chembl_id": ["C1", "C2"],
            "gene_symbol": ["HTR2A", "DRD2"],
            "gene_entrez_id": ["3356", None],
            "source": ["chembl"] * 2,
            "confidence": ["high"] * 2,
        })
        assert df["gene_entrez_id"].dtype != pd.Int64Dtype()
        result = _finalize_schema(df)
        assert result["gene_entrez_id"].dtype == pd.Int64Dtype()
        assert result["gene_entrez_id"].iloc[0] == 3356
        assert pd.isna(result["gene_entrez_id"].iloc[1])


class TestSanitizePlaceholderId:
    """Tests for ASCII-safe placeholder ID generation."""

    def test_basic_ascii(self) -> None:
        names = pd.Series(["Fluoxetine", "Olanzapine"])
        result = _sanitize_placeholder_id("PDSP_", names)
        assert result.iloc[0] == "PDSP_FLUOXETINE"
        assert result.iloc[1] == "PDSP_OLANZAPINE"

    def test_html_entities_stripped(self) -> None:
        names = pd.Series(["Drug&#8242;Name"])
        result = _sanitize_placeholder_id("PDSP_", names)
        assert "#" not in result.iloc[0]
        assert "DRUG" in result.iloc[0]

    def test_non_ascii_removed(self) -> None:
        names = pd.Series(["C36H50\u00c2\u20ac\u00a2H2O"])
        result = _sanitize_placeholder_id("PDSP_", names)
        assert result.iloc[0].isascii()
        assert "PDSP_" in result.iloc[0]

    def test_spaces_become_underscores(self) -> None:
        names = pd.Series(["My Drug Name"])
        result = _sanitize_placeholder_id("DGIDB_", names)
        assert " " not in result.iloc[0]
        assert "DGIDB_MY_DRUG_NAME" == result.iloc[0]

    def test_nan_does_not_crash(self) -> None:
        """NA / NaN values must not raise TypeError in normalize()."""
        names = pd.Series(["DrugA", None, pd.NA], dtype="string")
        result = _sanitize_placeholder_id("TEST_", names)
        assert result.iloc[0] == "TEST_DRUGA"
        assert result.iloc[1] == "TEST_"
        assert result.iloc[2] == "TEST_"

    def test_float_nan_does_not_crash(self) -> None:
        names = pd.Series(["DrugA", np.nan, "DrugB"])
        result = _sanitize_placeholder_id("TEST_", names)
        assert result.iloc[0] == "TEST_DRUGA"
        assert result.iloc[1] == "TEST_"
        assert result.iloc[2] == "TEST_DRUGB"

    def test_collision_disambiguated_with_hash(self) -> None:
        """Different names that sanitize to the same slug get hash suffixes."""
        names = pd.Series(["BROMOCRIPTINE", "BROMOCRIPTINE,(+)"])
        result = _sanitize_placeholder_id("PDSP_", names)
        assert result.iloc[0] != result.iloc[1]
        assert result.iloc[0].startswith("PDSP_BROMOCRIPTINE_")
        assert result.iloc[1].startswith("PDSP_BROMOCRIPTINE_")
        assert len(result.iloc[0].split("_")[-1]) == 8  # 8-char hex hash

    def test_collision_deterministic(self) -> None:
        """Same input always produces the same disambiguated IDs."""
        names = pd.Series(["Fenfluramine", "Fenfluramine (+)"])
        r1 = _sanitize_placeholder_id("PDSP_", names)
        r2 = _sanitize_placeholder_id("PDSP_", names)
        assert r1.iloc[0] == r2.iloc[0]
        assert r1.iloc[1] == r2.iloc[1]

    def test_no_hash_when_no_collision(self) -> None:
        """Non-colliding names should not get hash suffixes."""
        names = pd.Series(["Fluoxetine", "Olanzapine"])
        result = _sanitize_placeholder_id("PDSP_", names)
        assert result.iloc[0] == "PDSP_FLUOXETINE"
        assert result.iloc[1] == "PDSP_OLANZAPINE"


class TestMergeSources:
    """Tests for multi-source integration."""

    def test_single_source_passthrough(self) -> None:
        df = pd.DataFrame({
            "drug_name": ["Drug A"],
            "drug_chembl_id": ["CHEMBL1"],
            "gene_symbol": ["HTR2A"],
            "interaction_type": ["inhibitor"],
            "max_phase": [4],
            "source": ["chembl"],
            "confidence": ["high"],
        })
        result = merge_sources({"chembl": df})
        assert len(result) == 1
        assert "drug_name" in result.columns

    def test_deduplication_keeps_richest(self) -> None:
        chembl = pd.DataFrame({
            "drug_name": ["Drug A"],
            "drug_chembl_id": ["CHEMBL1"],
            "gene_symbol": ["HTR2A"],
            "interaction_type": ["antagonist"],
            "max_phase": [4],
            "source": ["chembl"],
            "confidence": ["high"],
            "pchembl_value": [8.5],
        })
        dgidb = pd.DataFrame({
            "drug_name": ["Drug A"],
            "drug_chembl_id": ["CHEMBL1"],
            "gene_symbol": ["HTR2A"],
            "interaction_type": ["antagonist"],
            "max_phase": [0],
            "source": ["dgidb"],
            "confidence": ["low"],
        })
        result = merge_sources({"chembl": chembl, "dgidb": dgidb})
        assert len(result) == 1
        assert "chembl" in result.iloc[0]["source"]
        assert "dgidb" in result.iloc[0]["source"]

    def test_dedup_with_list_columns_no_crash(self) -> None:
        """Dedup must not crash when list-like columns (atc_codes) have values."""
        chembl = pd.DataFrame({
            "drug_name": ["Drug A"],
            "drug_chembl_id": ["CHEMBL1"],
            "gene_symbol": ["HTR2A"],
            "interaction_type": ["antagonist"],
            "max_phase": [4],
            "source": ["chembl"],
            "atc_codes": [["N05A", "N06A"]],
            "indication_mesh": [["Depression"]],
        })
        dgidb = pd.DataFrame({
            "drug_name": ["Drug A"],
            "drug_chembl_id": ["CHEMBL1"],
            "gene_symbol": ["HTR2A"],
            "interaction_type": ["antagonist"],
            "max_phase": [0],
            "source": ["dgidb"],
            "atc_codes": [["N05A", "N07B"]],
            "indication_mesh": [None],
        })
        result = merge_sources({"chembl": chembl, "dgidb": dgidb})
        assert len(result) == 1
        atc = result.iloc[0]["atc_codes"]
        assert isinstance(atc, list)
        assert "N05A" in atc and "N06A" in atc and "N07B" in atc
        assert len(atc) == len(set(atc))

    def test_dedup_merges_list_columns(self) -> None:
        """List-like columns should merge unique values across duplicates."""
        chembl = pd.DataFrame({
            "drug_name": ["Drug B"],
            "drug_chembl_id": ["CHEMBL2"],
            "gene_symbol": ["DRD2"],
            "source": ["chembl"],
            "source_pmids": [["PMID1", "PMID2"]],
        })
        pdsp = pd.DataFrame({
            "drug_name": ["Drug B"],
            "drug_chembl_id": ["CHEMBL2"],
            "gene_symbol": ["DRD2"],
            "source": ["pdsp"],
            "source_pmids": [["PMID2", "PMID3"]],
        })
        result = merge_sources({"chembl": chembl, "pdsp": pdsp})
        pmids = result.iloc[0]["source_pmids"]
        assert isinstance(pmids, list)
        assert set(pmids) == {"PMID1", "PMID2", "PMID3"}

    def test_empty_sources(self) -> None:
        result = merge_sources({"chembl": pd.DataFrame()})
        assert result.empty


class TestLoadPdsp:
    """Tests for PDSP Ki loading."""

    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pdsp.csv"
        csv_path.write_text(
            "Drug Name,Target,Ki (nM),Species\n"
            "Clozapine,5-HT2A,4.0,Human\n"
            "Haloperidol,D2,1.5,Human\n"
            "Unknown Drug,Unknown Target,10.0,Human\n"
        )
        result = load_pdsp(pdsp_csv=csv_path)
        assert len(result) == 2
        assert "Clozapine" in result["drug_name"].values
        assert "Haloperidol" in result["drug_name"].values

    def test_filters_non_human(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pdsp.csv"
        csv_path.write_text(
            "Drug Name,Target,Ki (nM),Species\n"
            "Clozapine,5-HT2A,4.0,Human\n"
            "Clozapine,5-HT2A,5.0,Rat\n"
        )
        result = load_pdsp(pdsp_csv=csv_path)
        assert len(result) == 1

    def test_pchembl_calculation(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pdsp.csv"
        csv_path.write_text(
            "Drug Name,Target,Ki (nM),Species\n"
            "Clozapine,5-HT2A,1.0,Human\n"
        )
        result = load_pdsp(pdsp_csv=csv_path)
        assert abs(result.iloc[0]["pchembl_value"] - 9.0) < 0.01


class TestLoadDgidb:
    """Tests for DGIdb loading with edge cases."""

    def test_nan_drug_name_does_not_crash(self, tmp_path: Path) -> None:
        """Rows with missing drug_name must be dropped, not crash sanitizer."""
        tsv_path = tmp_path / "dgidb.tsv"
        tsv_path.write_text(
            "drug_name\tgene_name\tinteraction_type\n"
            "Clozapine\tHTR2A\tantagonist\n"
            "\tDRD2\tantagonist\n"
        )
        result = load_dgidb(interactions_tsv=tsv_path)
        assert len(result) == 1
        assert result.iloc[0]["drug_name"] == "Clozapine"

    def test_namespaced_chembl_concept_id(self, tmp_path: Path) -> None:
        """DGIdb 'chembl:CHEMBL123' concept IDs should yield canonical CHEMBL123."""
        tsv_path = tmp_path / "dgidb.tsv"
        tsv_path.write_text(
            "drug_name\tgene_name\tinteraction_type\tdrug_concept_id\n"
            "Risperidone\tHTR2A\tantagonist\tchembl:CHEMBL85\n"
            "UnknownDrug\tDRD2\tantagonist\twikidata:Q12345\n"
        )
        result = load_dgidb(interactions_tsv=tsv_path)
        assert len(result) == 2
        row_risp = result[result["drug_name"] == "Risperidone"].iloc[0]
        assert row_risp["drug_chembl_id"] == "CHEMBL85"
        row_unk = result[result["drug_name"] == "UnknownDrug"].iloc[0]
        assert row_unk["drug_chembl_id"].startswith("DGIDB_")


class TestExtractChemblId:
    """Tests for _extract_chembl_id helper."""

    def test_bare_id(self) -> None:
        assert _extract_chembl_id("CHEMBL123") == "CHEMBL123"

    def test_namespaced(self) -> None:
        assert _extract_chembl_id("chembl:CHEMBL12345") == "CHEMBL12345"

    def test_mixed_case(self) -> None:
        assert _extract_chembl_id("Chembl:chembl999") == "CHEMBL999"

    def test_whitespace(self) -> None:
        assert _extract_chembl_id("  chembl:CHEMBL42  ") == "CHEMBL42"

    def test_non_chembl_returns_none(self) -> None:
        assert _extract_chembl_id("wikidata:Q12345") is None

    def test_empty_returns_none(self) -> None:
        assert _extract_chembl_id("") is None
        assert _extract_chembl_id(None) is None

    def test_ambiguous_multi_token_returns_none(self) -> None:
        assert _extract_chembl_id("CHEMBL1 CHEMBL2") is None

    def test_repeated_same_token_ok(self) -> None:
        assert _extract_chembl_id("CHEMBL1 chembl:CHEMBL1") == "CHEMBL1"


# --- ChEMBL scope tests -------------------------------------------------


def _make_joined_df() -> pd.DataFrame:
    """Build a synthetic post-join DataFrame with mechanism + affinity rows."""
    return pd.DataFrame({
        "drug_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
        "drug_name": ["risperidone", "", ""],
        "uniprot_id": ["P28223", "P12345", "P67890"],
        "mechanism_of_action": ["5-HT2A antagonist", "D2 antagonist", None],
        "interaction_type": ["ANTAGONIST", "ANTAGONIST", None],
        "max_phase": [4, 3, 0],
        "pchembl_value": [8.5, None, 7.0],
        "_from_mechanism": [True, True, False],
    })


class TestChemblScope:
    """Tests for chembl_scope parameter in _postprocess_chembl."""

    def test_mechanism_only_drops_affinity_only(self) -> None:
        df = _make_joined_df()
        result = _postprocess_chembl(df, chembl_scope="mechanism_only")
        assert "CHEMBL3" not in result["drug_chembl_id"].values

    def test_mechanism_only_drops_blank_name_mechanism_rows(self) -> None:
        df = _make_joined_df()
        result = _postprocess_chembl(df, chembl_scope="mechanism_only")
        assert "CHEMBL2" not in result["drug_chembl_id"].values

    def test_mechanism_or_affinity_retains_affinity_only(self) -> None:
        df = _make_joined_df()
        result = _postprocess_chembl(df, chembl_scope="mechanism_or_affinity")
        assert "CHEMBL3" in result["drug_chembl_id"].values

    def test_mechanism_or_affinity_fallback_name(self) -> None:
        df = _make_joined_df()
        result = _postprocess_chembl(df, chembl_scope="mechanism_or_affinity")
        row3 = result[result["drug_chembl_id"] == "CHEMBL3"]
        assert row3["drug_name"].iloc[0] == "CHEMBL3"

    def test_blank_name_mechanism_row_dropped_in_both_modes(self) -> None:
        df = _make_joined_df()
        mech_only = _postprocess_chembl(df.copy(), chembl_scope="mechanism_only")
        mech_or_aff = _postprocess_chembl(df.copy(), chembl_scope="mechanism_or_affinity")
        assert "CHEMBL2" not in mech_only["drug_chembl_id"].values
        assert "CHEMBL2" not in mech_or_aff["drug_chembl_id"].values

    def test_superset_property(self) -> None:
        df = _make_joined_df()
        mech_only = _postprocess_chembl(df.copy(), chembl_scope="mechanism_only")
        mech_or_aff = _postprocess_chembl(df.copy(), chembl_scope="mechanism_or_affinity")
        mech_only_ids = set(mech_only["drug_chembl_id"])
        mech_or_aff_ids = set(mech_or_aff["drug_chembl_id"])
        assert mech_only_ids.issubset(mech_or_aff_ids)
        assert len(mech_or_aff_ids) >= len(mech_only_ids)

    def test_provenance_flag_dropped_from_output(self) -> None:
        df = _make_joined_df()
        result = _postprocess_chembl(df, chembl_scope="mechanism_only")
        assert "_from_mechanism" not in result.columns
        result2 = _postprocess_chembl(_make_joined_df(), chembl_scope="mechanism_or_affinity")
        assert "_from_mechanism" not in result2.columns

    def test_default_equals_explicit_mechanism_only(self) -> None:
        df1 = _make_joined_df()
        df2 = _make_joined_df()
        default_result = _postprocess_chembl(df1)
        explicit_result = _postprocess_chembl(df2, chembl_scope="mechanism_only")
        pd.testing.assert_frame_equal(default_result, explicit_result)

    def test_backward_compat_identical_output(self) -> None:
        """New code at default scope produces same output as old blank-name logic."""
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "drug_name": ["risperidone", "olanzapine", ""],
            "uniprot_id": ["P28223", "P14416", "P67890"],
            "mechanism_of_action": ["5-HT2A antagonist", "D2 antagonist", None],
            "interaction_type": ["ANTAGONIST", "ANTAGONIST", None],
            "max_phase": [4, 3, 0],
            "pchembl_value": [8.5, 7.2, 6.0],
            "_from_mechanism": [True, True, False],
        })
        result = _postprocess_chembl(df, chembl_scope="mechanism_only")
        assert set(result["drug_chembl_id"]) == {"CHEMBL1", "CHEMBL2"}
        assert "CHEMBL3" not in result["drug_chembl_id"].values
        assert result["drug_name"].tolist() == ["risperidone", "olanzapine"]


# ---------------------------------------------------------------------------
# Parent-aware ATC / indication resolution
# ---------------------------------------------------------------------------

def _build_chembl_mini_db(tmp_path: Path) -> Path:
    """Create a minimal ChEMBL-like SQLite for testing the mechanisms query.

    Schema:
      - mol 10 (CHEMBL10 "drug_A") is parent of itself, has direct ATC N05AH01
      - mol 20 (CHEMBL20 "drug_B") is a salt form whose parent is mol 30
        mol 30 has ATC N06AB03 but is not itself a mechanism drug
      - mol 40 (CHEMBL40 "drug_C") has direct ATC C07AA05 AND its parent
        mol 50 has ATC C07AA05 + C07AB02 (overlap + extra)
    All share the same single mechanism/target for simplicity.
    """
    db_path = tmp_path / "mini_chembl.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE molecule_dictionary (
            molregno INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT,
            max_phase INTEGER, molecule_type TEXT
        );
        CREATE TABLE molecule_hierarchy (
            molregno INTEGER PRIMARY KEY, parent_molregno INTEGER
        );
        CREATE TABLE compound_structures (
            molregno INTEGER PRIMARY KEY, standard_inchi_key TEXT, canonical_smiles TEXT
        );
        CREATE TABLE drug_mechanism (
            mec_id INTEGER PRIMARY KEY, molregno INTEGER,
            mechanism_of_action TEXT, action_type TEXT, tid INTEGER
        );
        CREATE TABLE target_dictionary (
            tid INTEGER PRIMARY KEY, pref_name TEXT,
            target_type TEXT, organism TEXT
        );
        CREATE TABLE target_components (
            tid INTEGER, component_id INTEGER
        );
        CREATE TABLE component_sequences (
            component_id INTEGER PRIMARY KEY, accession TEXT
        );
        CREATE TABLE molecule_atc_classification (
            molregno INTEGER, level5 TEXT
        );
        CREATE TABLE drug_indication (
            molregno INTEGER, mesh_heading TEXT
        );
        CREATE TABLE drug_warning (
            molregno INTEGER, warning_type TEXT
        );
        CREATE TABLE activities (
            assay_id INTEGER, molregno INTEGER, standard_type TEXT,
            standard_value REAL, standard_units TEXT, pchembl_value REAL,
            data_validity_comment TEXT
        );
        CREATE TABLE assays (
            assay_id INTEGER PRIMARY KEY, tid INTEGER
        );
    """)

    c.execute("INSERT INTO target_dictionary VALUES (1, 'Serotonin 2A', 'SINGLE PROTEIN', 'Homo sapiens')")
    c.execute("INSERT INTO target_components VALUES (1, 100)")
    c.execute("INSERT INTO component_sequences VALUES (100, 'P28223')")

    # mol 10: self-parent, direct ATC
    c.execute("INSERT INTO molecule_dictionary VALUES (10, 'CHEMBL10', 'drug_A', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (10, 10)")
    c.execute("INSERT INTO compound_structures VALUES (10, 'INCHI10', 'CCC')")
    c.execute("INSERT INTO drug_mechanism VALUES (1, 10, '5-HT2A antagonist', 'ANTAGONIST', 1)")
    c.execute("INSERT INTO molecule_atc_classification VALUES (10, 'N05AH01')")
    c.execute("INSERT INTO drug_indication VALUES (10, 'Schizophrenia')")

    # mol 20: child of mol 30, NO direct ATC; parent 30 has ATC
    c.execute("INSERT INTO molecule_dictionary VALUES (20, 'CHEMBL20', 'drug_B', 3, 'Small molecule')")
    # mol 30: parent of mol 20.  In real ChEMBL every parent_molregno
    # has a molecule_dictionary row; we add it here (with NULL
    # pref_name so it stays outside the candidate-name set) so
    # parent_chembl_id_of_cid resolves CHEMBL20 -> CHEMBL30 correctly.
    c.execute(
        "INSERT INTO molecule_dictionary VALUES (30, 'CHEMBL30', NULL, "
        "4, 'Small molecule')"
    )
    c.execute("INSERT INTO molecule_hierarchy VALUES (20, 30)")
    c.execute("INSERT INTO compound_structures VALUES (30, 'INCHI30', 'OCC')")
    c.execute("INSERT INTO drug_mechanism VALUES (2, 20, '5-HT2A antagonist', 'ANTAGONIST', 1)")
    c.execute("INSERT INTO molecule_atc_classification VALUES (30, 'N06AB03')")
    # Parent-only indication too
    c.execute("INSERT INTO drug_indication VALUES (30, 'Depression')")

    # mol 40: has direct ATC C07AA05; parent 50 has C07AA05 + C07AB02
    c.execute("INSERT INTO molecule_dictionary VALUES (40, 'CHEMBL40', 'drug_C', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (40, 50)")
    c.execute("INSERT INTO compound_structures VALUES (50, 'INCHI50', 'NCC')")
    c.execute("INSERT INTO drug_mechanism VALUES (3, 40, 'Beta blocker', 'ANTAGONIST', 1)")
    c.execute("INSERT INTO molecule_atc_classification VALUES (40, 'C07AA05')")
    c.execute("INSERT INTO molecule_atc_classification VALUES (50, 'C07AA05')")
    c.execute("INSERT INTO molecule_atc_classification VALUES (50, 'C07AB02')")

    # mol 30 and 50 need hierarchy rows (they are parents)
    c.execute("INSERT INTO molecule_hierarchy VALUES (30, 30)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (50, 50)")

    # mol 60: AFFINITY-ONLY drug (no mechanism row), self-parent, direct ATC.
    # Reproduces the scenario where ATC must reach drugs that have
    # no drug_mechanism row but do have a valid pchembl affinity record.
    c.execute("INSERT INTO molecule_dictionary VALUES (60, 'CHEMBL60', 'drug_D', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (60, 60)")
    c.execute("INSERT INTO compound_structures VALUES (60, 'INCHI60', 'FCC')")
    c.execute("INSERT INTO molecule_atc_classification VALUES (60, 'N06AB04')")
    c.execute("INSERT INTO drug_indication VALUES (60, 'Anxiety')")
    # Affinity row: must satisfy the _CHEMBL_QUERY_AFFINITIES filters
    # (Ki/IC50/EC50/Kd, nM, pchembl IS NOT NULL, validity NULL,
    #  target SINGLE PROTEIN + Homo sapiens via assays.tid -> td.tid).
    c.execute("INSERT INTO assays VALUES (10, 1)")
    c.execute(
        "INSERT INTO activities VALUES (10, 60, 'Ki', 5.0, 'nM', 8.3, NULL)"
    )

    # =========================================================
    # Synonym-index fixture extensions: tiered resolver,
    # parent-consistent rescue, ambiguous-no-parents, ghost-ATC.
    # =========================================================
    c.executescript("""
        CREATE TABLE molecule_synonyms (
            molregno INTEGER, synonyms TEXT, syn_type TEXT
        );
        CREATE TABLE atc_classification (
            level5 TEXT PRIMARY KEY,
            who_name TEXT,
            level1 TEXT, level2 TEXT, level3 TEXT, level4 TEXT
        );
    """)

    # WHO names for the existing ATC codes (used by tier-3 of the
    # synonym index).
    c.execute("INSERT INTO atc_classification VALUES ('N05AH01', 'who_drug_a',"
              " 'N', 'N05', 'N05A', 'N05AH')")
    c.execute("INSERT INTO atc_classification VALUES ('N06AB03', 'who_drug_b',"
              " 'N', 'N06', 'N06A', 'N06AB')")
    c.execute("INSERT INTO atc_classification VALUES ('C07AA05', 'who_drug_c',"
              " 'C', 'C07', 'C07A', 'C07AA')")
    c.execute("INSERT INTO atc_classification VALUES ('C07AB02', 'who_extra',"
              " 'C', 'C07', 'C07A', 'C07AB')")
    c.execute("INSERT INTO atc_classification VALUES ('N06AB04', 'who_drug_d',"
              " 'N', 'N06', 'N06A', 'N06AB')")

    # Synonyms for existing molecules.
    c.execute("INSERT INTO molecule_synonyms VALUES (10, 'syn_drug_a',"
              " 'TRADE_NAME')")
    c.execute("INSERT INTO molecule_synonyms VALUES (20, 'syn_drug_b',"
              " 'TRADE_NAME')")

    # Tier-1 ambiguous + intra-tier parent-collapse -> resolves to
    # parent at INDEX BUILD time (returned as origin="pref" by the
    # resolver since it's already in tier_resolved).
    # mol 70 and mol 71 share normalized pref_name "rescuepc"; both
    # collapse to mol 70.
    c.execute("INSERT INTO molecule_dictionary VALUES (70, 'CHEMBL70',"
              " 'rescue_pc', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_dictionary VALUES (71, 'CHEMBL71',"
              " 'Rescue Pc', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (70, 70)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (71, 70)")

    # Tier-1 genuinely ambiguous (different parents, no rescue path
    # at any tier).
    c.execute("INSERT INTO molecule_dictionary VALUES (72, 'CHEMBL72',"
              " 'conflict_drug', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_dictionary VALUES (73, 'CHEMBL73',"
              " 'conflict_drug', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (72, 72)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (73, 73)")

    # Tier-2 unique resolves when tier-1 misses ("synonly" key only
    # appears as a synonym).
    c.execute("INSERT INTO molecule_dictionary VALUES (74, 'CHEMBL74',"
              " 'tier2_pref', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_synonyms VALUES (74, 'tier2_syn_only',"
              " 'TRADE_NAME')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (74, 74)")

    # Tier-1 ambiguous (different parents) + parent-CONSISTENT rescue
    # via tier-2 synonym whose parent ∈ T1_parents.
    c.execute("INSERT INTO molecule_dictionary VALUES (80, 'CHEMBL80',"
              " 'rescue_target', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_dictionary VALUES (81, 'CHEMBL81',"
              " 'rescue_target', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_dictionary VALUES (82, 'CHEMBL82',"
              " 'rescue_parent_82', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (80, 80)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (81, 82)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (82, 82)")
    # mol 83 carries the synonym "rescue_target", parent=80 (in
    # T1_parents = {80, 82}).
    c.execute("INSERT INTO molecule_dictionary VALUES (83, 'CHEMBL83',"
              " 'rescue_syn_carrier', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_synonyms VALUES (83, 'rescue_target',"
              " 'TRADE_NAME')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (83, 80)")

    # Tier-1 ambiguous + parent-INCONSISTENT (lower-tier rescue
    # rejected because parent ∉ T1_parents).
    c.execute("INSERT INTO molecule_dictionary VALUES (84, 'CHEMBL84',"
              " 'incons_target', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_dictionary VALUES (85, 'CHEMBL85',"
              " 'incons_target', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (84, 84)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (85, 85)")
    c.execute("INSERT INTO molecule_dictionary VALUES (86, 'CHEMBL86',"
              " 'incons_syn_carrier', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_dictionary VALUES (99, 'CHEMBL99',"
              " 'unrelated_parent', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_synonyms VALUES (86, 'incons_target',"
              " 'TRADE_NAME')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (86, 99)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (99, 99)")

    # Ambiguous-no-parents rejection: tier-1 candidates exist with
    # NULL parent_molregno.  Resolver must return
    # ("ambig-no-parents") and NOT fall through to lower tiers.
    c.execute("INSERT INTO molecule_dictionary VALUES (92, 'CHEMBL92',"
              " 'ambig_noparent', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_dictionary VALUES (93, 'CHEMBL93',"
              " 'ambig_noparent', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (92, NULL)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (93, NULL)")

    # "Ghost" drug with ATC: real chembl_id, ATC-coded, but NOT
    # loaded by load_chembl (no mechanism, no affinity row).
    # Used by TestPropagateChemblAtcToRemapped and the integration
    # test to verify ATC propagation works for cids outside the
    # loaded ChEMBL set.
    c.execute("INSERT INTO molecule_dictionary VALUES (110, 'CHEMBL110',"
              " 'ghost_drug', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (110, 110)")
    c.execute("INSERT INTO compound_structures VALUES (110, 'INCHI110', 'NCO')")
    c.execute("INSERT INTO molecule_atc_classification VALUES (110, 'L01XA01')")
    c.execute("INSERT INTO atc_classification VALUES ('L01XA01',"
              " 'who_ghost', 'L', 'L01', 'L01X', 'L01XA')")
    c.execute("INSERT INTO drug_indication VALUES (110, 'Cancer')")

    # =========================================================
    # Parent/salt unification fixture extensions.
    # =========================================================
    # mol 120 ("salt_drug") is a salt form whose parent is mol 121
    # ("parent_drug").  Both are loaded by load_chembl (each has a
    # mechanism row + affinity row hitting the same target=1).
    # Canonicalization should map CHEMBL120 -> CHEMBL121 so the dedup
    # step collapses parent + salt rows on the same gene_symbol.
    c.execute("INSERT INTO molecule_dictionary VALUES (120, 'CHEMBL120',"
              " 'salt_drug', 4, 'Small molecule')")
    c.execute("INSERT INTO molecule_dictionary VALUES (121, 'CHEMBL121',"
              " 'parent_drug', 4, 'Small molecule')")
    c.execute("INSERT INTO compound_structures VALUES (120, 'INCHI120', 'PCC')")
    c.execute("INSERT INTO compound_structures VALUES (121, 'INCHI121', 'PCD')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (120, 121)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (121, 121)")
    c.execute("INSERT INTO drug_mechanism VALUES (120, 120,"
              " '5-HT2A antagonist', 'ANTAGONIST', 1)")
    c.execute("INSERT INTO drug_mechanism VALUES (121, 121,"
              " '5-HT2A antagonist', 'ANTAGONIST', 1)")
    # Direct ATC on the parent (ATC propagation already pushes it to the salt
    # via parent-resolved ATC; canonicalization collapses the IDs themselves).
    c.execute("INSERT INTO molecule_atc_classification VALUES (121, 'N06AA09')")
    c.execute("INSERT INTO atc_classification VALUES ('N06AA09',"
              " 'who_parent_drug', 'N', 'N06', 'N06A', 'N06AA')")

    # mol 130 ("nameless_salt") is a salt form whose parent is mol
    # 131 ("nameless_parent").  Neither has pref_name,
    # synonyms, or who_name in any of the canonical name tables, so
    # they live OUTSIDE the candidate-scoped index - yet load_chembl
    # still pulls both rows via the mechanism path.  Canonicalization
    # canonicalization MUST cover them via the extra_cids extension,
    # Not just the candidate set.  NULL pref_name is the
    # critical bit: SQL ``WHERE pref_name IS NOT NULL`` excludes
    # them from the candidate set, while ``load_chembl`` still
    # picks them up via ``drug_mechanism``.
    c.execute(
        "INSERT INTO molecule_dictionary VALUES (130, 'CHEMBL130',"
        " NULL, 4, 'Small molecule')"
    )
    c.execute(
        "INSERT INTO molecule_dictionary VALUES (131, 'CHEMBL131',"
        " NULL, 4, 'Small molecule')"
    )
    c.execute("INSERT INTO compound_structures VALUES (130, 'INCHI130', 'PCE')")
    c.execute("INSERT INTO compound_structures VALUES (131, 'INCHI131', 'PCF')")
    c.execute("INSERT INTO molecule_hierarchy VALUES (130, 131)")
    c.execute("INSERT INTO molecule_hierarchy VALUES (131, 131)")
    c.execute("INSERT INTO drug_mechanism VALUES (130, 130,"
              " '5-HT2A antagonist', 'ANTAGONIST', 1)")
    c.execute("INSERT INTO drug_mechanism VALUES (131, 131,"
              " '5-HT2A antagonist', 'ANTAGONIST', 1)")

    conn.commit()
    conn.close()
    return db_path


class TestParentAwareATC:
    """Tests for parent-molecule ATC and indication resolution in the SQL query."""

    def test_parent_only_atc_is_captured(self, tmp_path: Path) -> None:
        """Drug B (mol 20) has no direct ATC but parent (mol 30) does."""
        db = _build_chembl_mini_db(tmp_path)
        result = load_chembl(db)
        row_b = result[result["drug_chembl_id"] == "CHEMBL20"]
        assert not row_b.empty, "drug_B should be in results"
        atc = row_b["atc_codes"].iloc[0]
        assert isinstance(atc, list)
        assert "N06AB03" in atc

    def test_direct_atc_still_works(self, tmp_path: Path) -> None:
        """Drug A (mol 10) has direct ATC; should be unaffected by UNION."""
        db = _build_chembl_mini_db(tmp_path)
        result = load_chembl(db)
        row_a = result[result["drug_chembl_id"] == "CHEMBL10"]
        assert not row_a.empty
        atc = row_a["atc_codes"].iloc[0]
        assert isinstance(atc, list)
        assert "N05AH01" in atc

    def test_direct_and_parent_atc_deduplicated(self, tmp_path: Path) -> None:
        """Drug C has direct C07AA05; parent also has C07AA05 + C07AB02.
        Result should contain both codes exactly once."""
        db = _build_chembl_mini_db(tmp_path)
        result = load_chembl(db)
        row_c = result[result["drug_chembl_id"] == "CHEMBL40"]
        assert not row_c.empty
        atc = row_c["atc_codes"].iloc[0]
        assert isinstance(atc, list)
        assert "C07AA05" in atc
        assert "C07AB02" in atc
        assert atc.count("C07AA05") == 1, "duplicate ATC should be deduplicated"

    def test_parent_only_indication_is_captured(self, tmp_path: Path) -> None:
        """Drug B has no direct indication but parent has 'Depression'."""
        db = _build_chembl_mini_db(tmp_path)
        result = load_chembl(db)
        row_b = result[result["drug_chembl_id"] == "CHEMBL20"]
        assert not row_b.empty
        ind = row_b["indication_mesh"].iloc[0]
        assert isinstance(ind, list)
        assert "Depression" in ind


class TestDrugLevelATCDict:
    """Tests for the drug-level (chembl_id-keyed) ATC/indication dict path.

    ATC codes must reach
    affinity-only drugs whose ``_molregno`` is NaN after the outer merge.
    """

    def test_affinity_only_drug_gets_direct_atc(self, tmp_path: Path) -> None:
        """drug_D (mol 60) has no mechanism row but a valid pchembl row;
        under chembl_scope='mechanism_or_affinity' it must keep its ATC."""
        db = _build_chembl_mini_db(tmp_path)
        result = load_chembl(db, chembl_scope="mechanism_or_affinity")
        row_d = result[result["drug_chembl_id"] == "CHEMBL60"]
        assert not row_d.empty, "drug_D should be in mechanism_or_affinity results"
        atc = row_d["atc_codes"].iloc[0]
        assert isinstance(atc, list), f"expected list, got {type(atc)}: {atc!r}"
        assert "N06AB04" in atc

    def test_drug_with_no_atc_yields_none(self, tmp_path: Path) -> None:
        """A drug with no ATC anywhere (self or parent) must have atc_codes=None,
        not [] or NaN, to preserve the Optional[list[str]] schema contract."""
        # Build a tiny DB with a single drug that has no ATC.
        db_path = tmp_path / "no_atc.db"
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.executescript(
            """
            CREATE TABLE molecule_dictionary (
                molregno INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT,
                max_phase INTEGER, molecule_type TEXT
            );
            CREATE TABLE molecule_hierarchy (
                molregno INTEGER PRIMARY KEY, parent_molregno INTEGER
            );
            CREATE TABLE compound_structures (
                molregno INTEGER PRIMARY KEY, standard_inchi_key TEXT, canonical_smiles TEXT
            );
            CREATE TABLE drug_mechanism (
                mec_id INTEGER PRIMARY KEY, molregno INTEGER,
                mechanism_of_action TEXT, action_type TEXT, tid INTEGER
            );
            CREATE TABLE target_dictionary (
                tid INTEGER PRIMARY KEY, pref_name TEXT,
                target_type TEXT, organism TEXT
            );
            CREATE TABLE target_components (
                tid INTEGER, component_id INTEGER
            );
            CREATE TABLE component_sequences (
                component_id INTEGER PRIMARY KEY, accession TEXT
            );
            CREATE TABLE molecule_atc_classification (
                molregno INTEGER, level5 TEXT
            );
            CREATE TABLE drug_indication (
                molregno INTEGER, mesh_heading TEXT
            );
            CREATE TABLE drug_warning (
                molregno INTEGER, warning_type TEXT
            );
            CREATE TABLE activities (
                assay_id INTEGER, molregno INTEGER, standard_type TEXT,
                standard_value REAL, standard_units TEXT, pchembl_value REAL,
                data_validity_comment TEXT
            );
            CREATE TABLE assays (
                assay_id INTEGER PRIMARY KEY, tid INTEGER
            );
            """
        )
        c.execute("INSERT INTO target_dictionary VALUES (1, 'X', 'SINGLE PROTEIN', 'Homo sapiens')")
        c.execute("INSERT INTO target_components VALUES (1, 100)")
        c.execute("INSERT INTO component_sequences VALUES (100, 'P00000')")
        c.execute("INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL1', 'drug_no_atc', 4, 'Small molecule')")
        c.execute("INSERT INTO molecule_hierarchy VALUES (1, 1)")
        c.execute("INSERT INTO compound_structures VALUES (1, 'INCHI1', 'CC')")
        c.execute("INSERT INTO drug_mechanism VALUES (1, 1, 'inhibitor', 'INHIBITOR', 1)")
        conn.commit()
        conn.close()

        result = load_chembl(db_path)
        row = result[result["drug_chembl_id"] == "CHEMBL1"]
        assert not row.empty
        atc = row["atc_codes"].iloc[0]
        assert atc is None, f"expected None for drug with no ATC, got {atc!r}"

    def test_postprocess_fallback_when_dicts_absent(self) -> None:
        """When chembl_to_atc/chembl_to_ind are not provided, _postprocess_chembl
        must fall back to legacy parsing of atc_codes_raw/parent_atc_raw columns,
        preserving backward compatibility for any direct callers."""
        df = pd.DataFrame(
            {
                "drug_chembl_id": ["CHEMBLX"],
                "drug_name": ["legacy_drug"],
                "atc_codes_raw": ["N06AA01|N06AB02"],
                "parent_atc_raw": ["N06AA01"],
                "indications_raw": ["Depression"],
                "parent_indications_raw": [None],
                "max_phase": [4],
                "_from_mechanism": [True],
                "mechanism_of_action": ["test moa"],
                "interaction_type": ["INHIBITOR"],
                "uniprot_id": ["P00000"],
                "pchembl_value": [7.0],
            }
        )
        out = _postprocess_chembl(df.copy())
        assert len(out) == 1
        atc = out["atc_codes"].iloc[0]
        assert isinstance(atc, list)
        assert sorted(atc) == ["N06AA01", "N06AB02"]
        ind = out["indication_mesh"].iloc[0]
        assert isinstance(ind, list)
        assert ind == ["Depression"]

    def test_dict_precedence_over_raw_columns(self) -> None:
        """When both chembl_to_atc and atc_codes_raw are present, the dict
        must win; otherwise the affinity-only fix would be silently
        contaminated by stale per-row raw values."""
        df = pd.DataFrame(
            {
                "drug_chembl_id": ["CHEMBLX"],
                "drug_name": ["precedence_drug"],
                # Raw columns assert one ATC...
                "atc_codes_raw": ["C09AA01"],
                "parent_atc_raw": [None],
                "indications_raw": ["Hypertension"],
                "parent_indications_raw": [None],
                "max_phase": [4],
                "_from_mechanism": [True],
                "mechanism_of_action": ["test moa"],
                "interaction_type": ["INHIBITOR"],
                "uniprot_id": ["P00000"],
                "pchembl_value": [7.0],
            }
        )
        # ...the dict asserts a different one; dict must win.
        out = _postprocess_chembl(
            df.copy(),
            chembl_to_atc={"CHEMBLX": ["N06AB04"]},
            chembl_to_ind={"CHEMBLX": ["Anxiety"]},
        )
        assert len(out) == 1
        atc = out["atc_codes"].iloc[0]
        assert atc == ["N06AB04"], (
            f"dict must take precedence; got {atc!r}"
        )
        ind = out["indication_mesh"].iloc[0]
        assert ind == ["Anxiety"]


# ---------------------------------------------------------------------------
# UniChem PubChem CID enrichment tests
# ---------------------------------------------------------------------------


class TestParseUnichemMapping:

    def _write(self, tmp_path: Path, lines: list[str], gz: bool = False) -> Path:
        fname = "mapping.txt.gz" if gz else "mapping.txt"
        fpath = tmp_path / fname
        if gz:
            import gzip
            with gzip.open(fpath, "wt") as f:
                f.write("\n".join(lines))
        else:
            fpath.write_text("\n".join(lines))
        return fpath

    def test_basic_two_column(self, tmp_path: Path) -> None:
        lines = [
            "From src:'1'\tTo src:'22'",
            "CHEMBL25\t2244",
            "CHEMBL1060\t24203",
        ]
        path = self._write(tmp_path, lines)
        result = _parse_unichem_mapping(path)
        assert result == {"CHEMBL25": ["2244"], "CHEMBL1060": ["24203"]}

    def test_gzip_support(self, tmp_path: Path) -> None:
        lines = [
            "From src:'1'\tTo src:'22'",
            "CHEMBL25\t2244",
        ]
        path = self._write(tmp_path, lines, gz=True)
        result = _parse_unichem_mapping(path)
        assert result == {"CHEMBL25": ["2244"]}

    def test_multi_cid_per_chembl(self, tmp_path: Path) -> None:
        lines = [
            "From src:'1'\tTo src:'22'",
            "CHEMBL1060\t21924748",
            "CHEMBL1060\t24203",
            "CHEMBL1060\t58592228",
        ]
        path = self._write(tmp_path, lines)
        result = _parse_unichem_mapping(path)
        assert set(result["CHEMBL1060"]) == {"21924748", "24203", "58592228"}

    def test_non_numeric_cid_filtered(self, tmp_path: Path) -> None:
        lines = [
            "From src:'1'\tTo src:'22'",
            "CHEMBL25\t2244",
            "CHEMBL99\tNOT_A_CID",
        ]
        path = self._write(tmp_path, lines)
        result = _parse_unichem_mapping(path)
        assert "CHEMBL25" in result
        assert "CHEMBL99" not in result

    def test_header_line_skipped(self, tmp_path: Path) -> None:
        lines = [
            "From src:'1'\tTo src:'22'",
            "CHEMBL25\t2244",
        ]
        path = self._write(tmp_path, lines)
        result = _parse_unichem_mapping(path)
        assert len(result) == 1

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        lines = ["", "CHEMBL25\t2244", "", "CHEMBL50\t100"]
        path = self._write(tmp_path, lines)
        result = _parse_unichem_mapping(path)
        assert len(result) == 2


class TestEnrichPubchemCid:

    def test_fills_empty(self) -> None:
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL25", "CHEMBL50"],
            "drug_pubchem_cid": [pd.NA, pd.NA],
            "gene_symbol": ["GENE1", "GENE2"],
        })
        mapping = {"CHEMBL25": ["2244"], "CHEMBL50": ["5000", "3000"]}
        result = _enrich_pubchem_cid(df, mapping)
        assert result.loc[0, "drug_pubchem_cid"] == "2244"
        assert result.loc[1, "drug_pubchem_cid"] == "3000"

    def test_deterministic_smallest_cid(self) -> None:
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "drug_pubchem_cid": [pd.NA],
        })
        mapping = {"CHEMBL1": ["99999", "100", "5000"]}
        result = _enrich_pubchem_cid(df, mapping)
        assert result.loc[0, "drug_pubchem_cid"] == "100"

    def test_preserves_existing(self) -> None:
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL25"],
            "drug_pubchem_cid": ["9999"],
        })
        mapping = {"CHEMBL25": ["2244"]}
        result = _enrich_pubchem_cid(df, mapping)
        assert result.loc[0, "drug_pubchem_cid"] == "9999"

    def test_all_cids_column_created(self) -> None:
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "drug_pubchem_cid": [pd.NA],
        })
        mapping = {"CHEMBL1": ["300", "100", "200"]}
        result = _enrich_pubchem_cid(df, mapping)
        assert "_pubchem_cid_all" in result.columns
        assert result.loc[0, "_pubchem_cid_all"] == ["100", "200", "300"]

    def test_no_mapping_no_crash(self) -> None:
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL999"],
            "drug_pubchem_cid": [pd.NA],
        })
        mapping = {"CHEMBL25": ["2244"]}
        result = _enrich_pubchem_cid(df, mapping)
        assert pd.isna(result.loc[0, "drug_pubchem_cid"]) or result.loc[0, "drug_pubchem_cid"] in ("", "<NA>", "nan")

    def test_missing_chembl_id_col_passthrough(self) -> None:
        df = pd.DataFrame({"gene_symbol": ["A"]})
        result = _enrich_pubchem_cid(df, {"CHEMBL1": ["100"]})
        assert list(result.columns) == ["gene_symbol"]


# ---------------------------------------------------------------------------
# ChEMBL synonym index, DGIdb name fallback, ATC propagation
# ---------------------------------------------------------------------------


class TestNormalizeDrugName:
    """Normalization key for the synonym index."""

    def test_casefold_and_strip_punctuation(self) -> None:
        assert _normalize_drug_name("Drug-A") == "druga"
        assert _normalize_drug_name(" DRUG_A ") == "druga"
        assert _normalize_drug_name("drug.a (1)") == "druga1"

    def test_nfkd_decomposition(self) -> None:
        # Accented characters decompose to ASCII; without NFKD this
        # would leave the accented form and miss matches.
        assert _normalize_drug_name("étodolac") == "etodolac"
        assert _normalize_drug_name("CITALOPRÄM") == "citalopram"

    def test_idempotence(self) -> None:
        once = _normalize_drug_name("Drug-A!")
        twice = _normalize_drug_name(once)
        assert once == twice == "druga"

    def test_empty_or_missing(self) -> None:
        assert _normalize_drug_name(None) == ""
        assert _normalize_drug_name("") == ""
        assert _normalize_drug_name(float("nan")) == ""
        assert _normalize_drug_name("   ") == ""


class TestCollapseToParent:
    """Pure dict-lookup parent-collapse helper."""

    def test_single_cid_returns_self(self) -> None:
        out = _collapse_to_parent(
            {"CHEMBLA"},
            parent_of_cid={"CHEMBLA": 1},
            molregno_to_cid={1: "CHEMBLA"},
        )
        assert out == "CHEMBLA"

    def test_two_cids_same_parent_returns_parent(self) -> None:
        out = _collapse_to_parent(
            {"CHEMBLA", "CHEMBLB"},
            parent_of_cid={"CHEMBLA": 1, "CHEMBLB": 1},
            molregno_to_cid={1: "CHEMBLP"},
        )
        assert out == "CHEMBLP"

    def test_two_cids_different_parents_returns_none(self) -> None:
        out = _collapse_to_parent(
            {"CHEMBLA", "CHEMBLB"},
            parent_of_cid={"CHEMBLA": 1, "CHEMBLB": 2},
            molregno_to_cid={1: "CHEMBLA", 2: "CHEMBLB"},
        )
        assert out is None

    def test_missing_parent_returns_none(self) -> None:
        out = _collapse_to_parent(
            {"CHEMBLA", "CHEMBLB"},
            parent_of_cid={"CHEMBLA": 1, "CHEMBLB": None},
            molregno_to_cid={1: "CHEMBLA"},
        )
        assert out is None

    def test_parent_without_chembl_id_returns_none(self) -> None:
        out = _collapse_to_parent(
            {"CHEMBLA", "CHEMBLB"},
            parent_of_cid={"CHEMBLA": 1, "CHEMBLB": 1},
            molregno_to_cid={},
        )
        assert out is None


class TestChemblSynonymIndex:
    """Build the tiered name index from the mini ChEMBL fixture."""

    def test_builds_three_tiers(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        assert isinstance(index, ChemblNameIndex)
        assert set(index.tier_resolved.keys()) == {"pref", "syn", "who"}
        assert set(index.tier_candidates.keys()) == {"pref", "syn", "who"}

    def test_carries_chembl_to_atc(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        # CHEMBL10 has direct ATC; CHEMBL110 has direct ATC L01XA01
        # despite being a "ghost" with no mechanism row.
        assert "N05AH01" in index.chembl_to_atc.get("CHEMBL10", [])
        assert "L01XA01" in index.chembl_to_atc.get("CHEMBL110", [])

    def test_pref_unique_resolved(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        assert index.tier_resolved["pref"]["druga"] == "CHEMBL10"
        assert index.tier_resolved["pref"]["ghostdrug"] == "CHEMBL110"

    def test_pref_intra_tier_parent_collapse(self, tmp_path: Path) -> None:
        # mol 70 and mol 71 share normalized pref "rescuepc"; both
        # collapse to mol 70 -> CHEMBL70.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        assert index.tier_resolved["pref"]["rescuepc"] == "CHEMBL70"

    def test_pref_genuine_ambiguity_unresolved(self, tmp_path: Path) -> None:
        # mol 72 / mol 73 have different parents -> no collapse.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        assert "conflictdrug" not in index.tier_resolved["pref"]
        assert index.tier_candidates["pref"]["conflictdrug"] == {"CHEMBL72", "CHEMBL73"}

    def test_synonym_tier_resolves(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        assert index.tier_resolved["syn"]["tier2synonly"] == "CHEMBL74"

    def test_who_tier_resolves(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        # who_drug_a maps to CHEMBL10 via N05AH01.
        assert index.tier_resolved["who"]["whodruga"] == "CHEMBL10"


class TestResolveSynonymIndex:
    """The 7 decision-tree branches of the resolver."""

    def test_pref_unique(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        cid, origin = _resolve_synonym_index("drug_A", index)
        assert cid == "CHEMBL10"
        assert origin == "pref"

    def test_pref_unique_via_intra_tier_parent_collapse(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        cid, origin = _resolve_synonym_index("rescue_pc", index)
        # Collapsed at build time, so resolver sees a unique tier-1
        # match and returns origin="pref".
        assert cid == "CHEMBL70"
        assert origin == "pref"

    def test_pref_genuine_ambiguity_blocks(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        cid, origin = _resolve_synonym_index("conflict_drug", index)
        assert cid is None
        assert origin == "pref-ambig-blocked"

    def test_synonym_resolves_when_pref_misses(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        cid, origin = _resolve_synonym_index("tier2_syn_only", index)
        assert cid == "CHEMBL74"
        assert origin == "syn"

    def test_parent_consistent_synonym_rescue(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        cid, origin = _resolve_synonym_index("rescue_target", index)
        # tier-1 ambiguous (parents = {80, 82}); tier-2 syn unique
        # CHEMBL83 with parent=80 ∈ T1_parents.
        assert cid == "CHEMBL83"
        assert origin == "syn-rescue"

    def test_parent_inconsistent_rescue_rejected(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        cid, origin = _resolve_synonym_index("incons_target", index)
        # tier-1 parents = {84, 85}; tier-2 syn unique CHEMBL86 with
        # parent=99 ∉ T1_parents -> rescue rejected.
        assert cid is None
        assert origin == "pref-ambig-blocked"

    def test_ambig_no_parents_explicit_reject(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        cid, origin = _resolve_synonym_index("ambig_noparent", index)
        # All tier-1 candidates have NULL parent_molregno -> explicit
        # reject; resolver does NOT fall through to lower tiers.
        assert cid is None
        assert origin == "ambig-no-parents"

    def test_empty_input(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        assert _resolve_synonym_index("", index) == (None, "empty")
        assert _resolve_synonym_index(None, index) == (None, "empty")

    def test_complete_miss(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        cid, origin = _resolve_synonym_index("totally_unknown_drug", index)
        assert cid is None
        assert origin == "miss"


class TestDgidbSynonymFallback:
    """The pure-rewriter ``_match_dgidb_to_chembl`` matcher."""

    def _make_dgidb_df(self, rows: list[dict]) -> pd.DataFrame:
        # Reflects the load_dgidb output schema once the synonym index carries
        # _drug_claim_name through.
        defaults = {
            "drug_name": "",
            "drug_chembl_id": "",
            "gene_symbol": "GENE",
            "interaction_type": "other",
            "max_phase": 0,
            "source": "dgidb",
            "confidence": "low",
            "source_pmids": None,
            "_drug_claim_name": "",
        }
        records = []
        for r in rows:
            base = dict(defaults)
            base.update(r)
            records.append(base)
        return pd.DataFrame(records)

    def test_existing_chembl_id_unchanged(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_dgidb_df([
            {"drug_name": "anything", "drug_chembl_id": "CHEMBL10",
             "_drug_claim_name": "anything"},
        ])
        out = _match_dgidb_to_chembl(df, index)
        assert out["drug_chembl_id"].iloc[0] == "CHEMBL10"
        assert "_drug_claim_name" not in out.columns

    def test_drug_name_resolves_remaps(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_dgidb_df([
            {"drug_name": "drug_A", "drug_chembl_id": "DGIDB_drug_A",
             "_drug_claim_name": ""},
        ])
        out = _match_dgidb_to_chembl(df, index)
        assert out["drug_chembl_id"].iloc[0] == "CHEMBL10"

    def test_falls_back_to_drug_claim_name(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_dgidb_df([
            {"drug_name": "Unknown Brand X",
             "drug_chembl_id": "DGIDB_unknown_brand_x",
             "_drug_claim_name": "drug_A"},
        ])
        out = _match_dgidb_to_chembl(df, index)
        assert out["drug_chembl_id"].iloc[0] == "CHEMBL10"

    def test_unresolvable_keeps_placeholder(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_dgidb_df([
            {"drug_name": "totally_made_up_drug",
             "drug_chembl_id": "DGIDB_totally_made_up_drug",
             "_drug_claim_name": ""},
        ])
        out = _match_dgidb_to_chembl(df, index)
        assert out["drug_chembl_id"].iloc[0] == "DGIDB_totally_made_up_drug"

    def test_does_not_mutate_confidence(self, tmp_path: Path) -> None:
        # The matcher must NOT pre-update confidence.
        # The multi-source upgrade is centralized in
        # _deduplicate_records and pre-mutating here would bias
        # downstream confidence_filter behavior.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_dgidb_df([
            {"drug_name": "drug_A", "drug_chembl_id": "DGIDB_drug_A",
             "confidence": "low"},
            {"drug_name": "drug_B", "drug_chembl_id": "DGIDB_drug_B",
             "confidence": "medium"},
        ])
        out = _match_dgidb_to_chembl(df, index)
        assert out["confidence"].tolist() == ["low", "medium"]

    def test_drops_drug_claim_name_column(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_dgidb_df([
            {"drug_name": "x", "drug_chembl_id": "DGIDB_x",
             "_drug_claim_name": "y"},
        ])
        out = _match_dgidb_to_chembl(df, index)
        assert "_drug_claim_name" not in out.columns


class TestPropagateChemblAtcToRemapped:
    """Strictly-additive ATC/indication propagation pass."""

    def test_remapped_dgidb_row_gains_atc(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL10"],
            "drug_name": ["drug_A"],
            "atc_codes": [None],
            "indication_mesh": [None],
            "source": ["dgidb"],
        })
        out = _propagate_chembl_atc_to_remapped(df, index)
        assert out["atc_codes"].iloc[0] == ["N05AH01"]
        # CHEMBL10 also has indication "Schizophrenia" via the ChEMBL dict.
        assert out["indication_mesh"].iloc[0] == ["Schizophrenia"]

    def test_idempotent_for_already_populated(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL10"],
            "drug_name": ["drug_A"],
            "atc_codes": [["EXISTING"]],
            "indication_mesh": [None],
            "source": ["chembl"],
        })
        out = _propagate_chembl_atc_to_remapped(df, index)
        # Existing ATC must NOT be overwritten.
        assert out["atc_codes"].iloc[0] == ["EXISTING"]

    def test_placeholder_id_unchanged(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = pd.DataFrame({
            "drug_chembl_id": ["DGIDB_unknown", "PDSP_unknown"],
            "drug_name": ["unknown1", "unknown2"],
            "atc_codes": [None, None],
            "indication_mesh": [None, None],
            "source": ["dgidb", "pdsp"],
        })
        out = _propagate_chembl_atc_to_remapped(df, index)
        # Placeholder IDs are not in chembl_to_atc; both rows remain
        # with atc_codes=None.
        for v in out["atc_codes"].tolist():
            assert v is None or (isinstance(v, list) and len(v) == 0)

    def test_real_chembl_id_without_atc_keeps_none(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL74"],  # has no ATC entry
            "drug_name": ["tier2_pref"],
            "atc_codes": [None],
            "indication_mesh": [None],
            "source": ["dgidb"],
        })
        out = _propagate_chembl_atc_to_remapped(df, index)
        v = out["atc_codes"].iloc[0]
        assert v is None or (isinstance(v, list) and len(v) == 0)

    def test_ghost_id_outside_loaded_set_still_gets_atc(self, tmp_path: Path) -> None:
        # CHEMBL110 is not loaded by load_chembl(mechanism_only) but
        # has ATC L01XA01 in chembl_to_atc.  Propagation must reach it.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL110"],
            "drug_name": ["ghost_drug"],
            "atc_codes": [None],
            "indication_mesh": [None],
            "source": ["dgidb"],
        })
        # loaded_chembl_ids deliberately omits CHEMBL110 -> "ghost" path.
        out = _propagate_chembl_atc_to_remapped(
            df, index, loaded_chembl_ids={"CHEMBL10", "CHEMBL20"},
        )
        assert out["atc_codes"].iloc[0] == ["L01XA01"]


class TestFinalizeSchemaStrict:
    """Defensive net: _finalize_schema drops non-schema columns."""

    def test_drops_non_schema_columns(self) -> None:
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "drug_name": ["x"],
            "gene_symbol": ["G"],
            "_drug_claim_name": ["leftover"],  # transient leak
            "_some_other_helper": [1],
        })
        out = _finalize_schema(df)
        assert "_drug_claim_name" not in out.columns
        assert "_some_other_helper" not in out.columns

    def test_all_schema_columns_present(self) -> None:
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "drug_name": ["x"],
            "gene_symbol": ["G"],
        })
        out = _finalize_schema(df)
        expected = {
            "drug_name", "drug_chembl_id", "drug_inchikey", "drug_pubchem_cid",
            "drug_smiles", "gene_symbol", "gene_ensembl_id", "gene_uniprot_id",
            "gene_entrez_id", "interaction_type", "action_type",
            "mechanism_of_action", "pchembl_value", "affinity_value",
            "affinity_type", "affinity_unit", "max_phase", "atc_codes",
            "indication_mesh", "molecule_type", "is_withdrawn", "source",
            "confidence", "source_pmids",
        }
        assert set(out.columns) == expected

    def test_no_extra_cols_no_drop_log(self) -> None:
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"],
            "drug_name": ["x"],
            "gene_symbol": ["G"],
        })
        # Should not raise; schema columns added with defaults.
        out = _finalize_schema(df)
        assert len(out) == 1


class TestMergeSourcesIntegration:
    """End-to-end integration: load_chembl + DGIdb + name index."""

    def test_dgidb_remap_and_atc_propagation(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)

        # Simulate ChEMBL output for a SUBSET of the universe (i.e.
        # CHEMBL110 is a "ghost" - not loaded under the current scope).
        chembl_df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL10"],
            "drug_name": ["drug_A"],
            "gene_symbol": ["GENE"],
            "atc_codes": [["N05AH01"]],
            "indication_mesh": [["Schizophrenia"]],
            "source": ["chembl"],
            "confidence": ["high"],
            "max_phase": [4],
        })

        # DGIdb DataFrame: one row that resolves to CHEMBL10 (in loaded
        # set) and one that resolves to CHEMBL110 (ghost).
        dgidb_df = pd.DataFrame({
            "drug_chembl_id": ["DGIDB_a", "DGIDB_g"],
            "drug_name": ["drug_A", "ghost_drug"],
            "gene_symbol": ["GENE2", "GENE3"],
            "atc_codes": [None, None],
            "indication_mesh": [None, None],
            "source": ["dgidb", "dgidb"],
            "confidence": ["low", "low"],
            "max_phase": [0, 0],
            "_drug_claim_name": ["", ""],
        })

        merged = merge_sources(
            {"chembl": chembl_df, "dgidb": dgidb_df},
            chembl_name_index=index,
        )

        # Both DGIdb rows must have been remapped.
        chembl_ids = sorted(merged["drug_chembl_id"].unique().tolist())
        assert "CHEMBL10" in chembl_ids
        assert "CHEMBL110" in chembl_ids

        # Schema is strict - no transient columns.
        assert "_drug_claim_name" not in merged.columns

        # CHEMBL110 (ghost) gained ATC via propagation.
        ghost_row = merged[merged["drug_chembl_id"] == "CHEMBL110"].iloc[0]
        assert ghost_row["atc_codes"] == ["L01XA01"]

    def test_no_index_no_op_preserves_behavior(self, tmp_path: Path) -> None:
        # When chembl_name_index is None, DGIdb remapping and ATC
        # propagation do nothing; the plain matching behaviour is intact.
        chembl_df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL10"],
            "drug_name": ["drug_A"],
            "gene_symbol": ["GENE"],
            "atc_codes": [["N05AH01"]],
            "source": ["chembl"],
            "confidence": ["high"],
            "max_phase": [4],
        })
        dgidb_df = pd.DataFrame({
            "drug_chembl_id": ["DGIDB_a"],
            "drug_name": ["drug_A"],
            "gene_symbol": ["GENE2"],
            "source": ["dgidb"],
            "confidence": ["low"],
            "max_phase": [0],
            "_drug_claim_name": [""],
        })
        merged = merge_sources(
            {"chembl": chembl_df, "dgidb": dgidb_df},
            chembl_name_index=None,
        )
        # DGIdb_a row keeps its placeholder.
        assert "DGIDB_a" in merged["drug_chembl_id"].tolist()
        # Transient column still stripped.
        assert "_drug_claim_name" not in merged.columns


# ---------------------------------------------------------------------------
# Synonym-index follow-ups: chunked SQL IN queries +
# resolver tier-2-ambig blocks tier-3 fallthrough.
# ---------------------------------------------------------------------------


class TestQueryInChunks:
    """Chunked ``SELECT ... WHERE col IN (?, ?, ...)`` helper.

    Regression guard for the production failure where
    ``load_chembl_synonym_index(chembl_35.db)`` raised
    ``OperationalError: too many SQL variables`` because the candidate
    set (~100k molregnos) exceeded SQLite's
    ``SQLITE_MAX_VARIABLE_NUMBER`` (32766 on ≥3.32, 999 on older builds).
    """

    @staticmethod
    def _make_db(tmp_path: Path, n_rows: int) -> sqlite3.Connection:
        db = tmp_path / f"chunks_{n_rows}.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (id INTEGER, val INTEGER)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, i * 10) for i in range(n_rows)],
        )
        conn.commit()
        return conn

    def test_chunked_matches_single_query(self, tmp_path: Path) -> None:
        # 50 rows, chunk_size=10 -> 5 chunks; result must equal the
        # single-query path.
        conn = self._make_db(tmp_path, 50)
        try:
            ids = list(range(50))
            placeholders = ",".join("?" * len(ids))
            single = pd.read_sql_query(
                f"SELECT id, val FROM t WHERE id IN ({placeholders})",
                conn, params=ids,
            )
            chunked = _query_in_chunks(
                conn,
                "SELECT id, val FROM t WHERE id IN ({placeholders})",
                ids,
                chunk_size=10,
            )
            pd.testing.assert_frame_equal(
                single.sort_values("id").reset_index(drop=True),
                chunked.sort_values("id").reset_index(drop=True),
            )
        finally:
            conn.close()

    def test_empty_params_returns_empty(self, tmp_path: Path) -> None:
        conn = self._make_db(tmp_path, 5)
        try:
            out = _query_in_chunks(
                conn,
                "SELECT id, val FROM t WHERE id IN ({placeholders})",
                [],
                chunk_size=10,
            )
            assert isinstance(out, pd.DataFrame)
            assert out.empty
        finally:
            conn.close()

    def test_dedup_collapses_duplicate_params(self, tmp_path: Path) -> None:
        # Duplicate params should not produce duplicate rows because
        # SQL ``IN`` is set-semantic.  The helper dedups before
        # chunking so the chunk count and result row count both match
        # the deduplicated input.
        conn = self._make_db(tmp_path, 10)
        try:
            params_with_dups = [1, 2, 2, 3, 1, 4, 4, 4]
            out = _query_in_chunks(
                conn,
                "SELECT id, val FROM t WHERE id IN ({placeholders})",
                params_with_dups,
                chunk_size=100,
            )
            assert sorted(out["id"].tolist()) == [1, 2, 3, 4]
            # No duplicate rows.
            assert len(out) == out["id"].nunique()
        finally:
            conn.close()

    def test_chunk_size_below_input_length_completes(
        self, tmp_path: Path
    ) -> None:
        # Specifically the regression scenario: n=2000 IDs split
        # across multiple sub-default-limit chunks completes without
        # raising ``too many SQL variables``.
        conn = self._make_db(tmp_path, 2000)
        try:
            ids = list(range(2000))
            out = _query_in_chunks(
                conn,
                "SELECT id, val FROM t WHERE id IN ({placeholders})",
                ids,
                chunk_size=500,
            )
            assert len(out) == 2000
            assert set(out["id"].tolist()) == set(ids)
        finally:
            conn.close()


class TestSynonymIndexLargeCandidateSet:
    """End-to-end regression: ``_build_chembl_synonym_index`` must NOT
    raise on a candidate set larger than SQLite's variable limit.

    The mini-fixture used elsewhere only has ~20 molregnos; this test
    builds a synthetic SQLite with 1500 candidate molregnos
    (well above the 999 conservative limit) and verifies the index
    builds successfully via the chunked path.
    """

    def test_handles_1500_candidate_molregnos(self, tmp_path: Path) -> None:
        db_path = tmp_path / "large_chembl.db"
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE molecule_dictionary (
                molregno INTEGER PRIMARY KEY, chembl_id TEXT, pref_name TEXT,
                max_phase INTEGER, molecule_type TEXT
            );
            CREATE TABLE molecule_hierarchy (
                molregno INTEGER PRIMARY KEY, parent_molregno INTEGER
            );
            CREATE TABLE compound_structures (
                molregno INTEGER PRIMARY KEY, standard_inchi_key TEXT,
                canonical_smiles TEXT
            );
            CREATE TABLE drug_mechanism (
                mec_id INTEGER PRIMARY KEY, molregno INTEGER,
                mechanism_of_action TEXT, action_type TEXT, tid INTEGER
            );
            CREATE TABLE molecule_synonyms (
                molregno INTEGER, synonyms TEXT, syn_type TEXT
            );
            CREATE TABLE molecule_atc_classification (
                molregno INTEGER, level5 TEXT
            );
            CREATE TABLE atc_classification (
                level5 TEXT PRIMARY KEY, who_name TEXT,
                level1 TEXT, level2 TEXT, level3 TEXT, level4 TEXT
            );
            CREATE TABLE drug_indication (
                molregno INTEGER, mesh_heading TEXT
            );
        """)
        # 1500 distinct molecules with unique pref_name and synonym.
        n = 1500
        c.executemany(
            "INSERT INTO molecule_dictionary VALUES (?, ?, ?, ?, ?)",
            [
                (i, f"CHEMBL{i}", f"pref_{i}", 4, "Small molecule")
                for i in range(1, n + 1)
            ],
        )
        c.executemany(
            "INSERT INTO molecule_hierarchy VALUES (?, ?)",
            [(i, i) for i in range(1, n + 1)],
        )
        c.executemany(
            "INSERT INTO molecule_synonyms VALUES (?, ?, ?)",
            [(i, f"syn_{i}", "TRADE_NAME") for i in range(1, n + 1)],
        )
        conn.commit()

        try:
            index = _build_chembl_synonym_index(conn)
        finally:
            conn.close()

        assert isinstance(index, ChemblNameIndex)
        # Every drug should have a unique pref-tier resolution.
        assert len(index.tier_resolved["pref"]) == n
        # Spot-check resolution at the top of the range.
        cid, origin = _resolve_synonym_index(f"pref_{n}", index)
        assert cid == f"CHEMBL{n}"
        assert origin == "pref"


class TestResolverTier2AmbigBlocksTier3:
    """Tier-1 ambiguous + tier-2 ambiguous must NOT fall through to tier-3.

    Symmetric with the tier-1-miss branch's ``syn-ambig-blocked`` rule:
    if the synonym tier itself registered an ambiguity on this name,
    a tier-3 hit cannot resolve it.

    Uses synthetic ``ChemblNameIndex`` instances rather than extending
    the SQLite fixture; the resolver is pure dict-lookup so this is
    the most direct test.
    """

    def _make_index(
        self,
        *,
        pref_resolved: dict[str, str],
        pref_candidates: dict[str, set[str]],
        syn_resolved: dict[str, str],
        syn_candidates: dict[str, set[str]],
        who_resolved: dict[str, str],
        who_candidates: dict[str, set[str]],
        parent_of_cid: dict[str, int],
    ) -> ChemblNameIndex:
        return ChemblNameIndex(
            tier_resolved={
                "pref": pref_resolved, "syn": syn_resolved,
                "who": who_resolved,
            },
            tier_candidates={
                "pref": pref_candidates, "syn": syn_candidates,
                "who": who_candidates,
            },
            parent_of_cid=parent_of_cid,
            parent_chembl_id_of_cid={},
            chembl_to_atc={},
            chembl_to_ind={},
        )

    def test_tier1_ambig_with_tier2_ambig_blocks_tier3_fallthrough(
        self,
    ) -> None:
        # Pref ambiguous (parents {100, 102}); syn ambiguous (no resolved
        # hit); who has consistent unique hit with parent ∈ T1_parents.
        # Pre-fix this returned ("CHEMBL105", "who-rescue"); post-fix
        # it must return (None, "pref-ambig-blocked").
        index = self._make_index(
            pref_resolved={},
            pref_candidates={"k": {"CHEMBL100", "CHEMBL101"}},
            syn_resolved={},
            syn_candidates={"k": {"CHEMBL103", "CHEMBL104"}},
            who_resolved={"k": "CHEMBL105"},
            who_candidates={"k": {"CHEMBL105"}},
            parent_of_cid={
                "CHEMBL100": 100, "CHEMBL101": 102,
                "CHEMBL103": 103, "CHEMBL104": 104,
                "CHEMBL105": 100,  # parent-consistent w/ T1 {100, 102}
            },
        )
        cid, origin = _resolve_synonym_index("k", index)
        assert cid is None
        assert origin == "pref-ambig-blocked"

    def test_tier1_ambig_with_tier2_miss_falls_through_to_tier3(self) -> None:
        # Tier-2 truly missed (not in res_syn AND not in cand_syn).
        # Tier-3 has consistent hit -> who-rescue.  Regression guard
        # against over-correcting Fix 2 into a blanket tier-3 block.
        index = self._make_index(
            pref_resolved={},
            pref_candidates={"k": {"CHEMBL100", "CHEMBL101"}},
            syn_resolved={},
            syn_candidates={},
            who_resolved={"k": "CHEMBL105"},
            who_candidates={"k": {"CHEMBL105"}},
            parent_of_cid={
                "CHEMBL100": 100, "CHEMBL101": 102,
                "CHEMBL105": 100,
            },
        )
        cid, origin = _resolve_synonym_index("k", index)
        assert cid == "CHEMBL105"
        assert origin == "who-rescue"

    def test_tier1_ambig_with_tier2_resolved_inconsistent_falls_through(
        self,
    ) -> None:
        # Documented "lenient parent-inconsistent fall-through" still
        # works: tier-2 resolved-but-parent-inconsistent -> fall through
        # to tier-3, which resolves consistently.
        index = self._make_index(
            pref_resolved={},
            pref_candidates={"k": {"CHEMBL100", "CHEMBL101"}},
            syn_resolved={"k": "CHEMBL103"},
            syn_candidates={"k": {"CHEMBL103"}},
            who_resolved={"k": "CHEMBL105"},
            who_candidates={"k": {"CHEMBL105"}},
            parent_of_cid={
                "CHEMBL100": 100, "CHEMBL101": 102,
                "CHEMBL103": 999,  # parent ∉ T1 {100, 102} -> inconsistent
                "CHEMBL105": 100,  # parent ∈ T1 -> tier-3 wins
            },
        )
        cid, origin = _resolve_synonym_index("k", index)
        assert cid == "CHEMBL105"
        assert origin == "who-rescue"

    def test_tier1_ambig_with_tier2_resolved_consistent_returns_syn_rescue(
        self,
    ) -> None:
        # Sanity check on the happy path: tier-2 resolved + consistent
        # -> syn-rescue (no tier-3 evaluation needed).
        index = self._make_index(
            pref_resolved={},
            pref_candidates={"k": {"CHEMBL100", "CHEMBL101"}},
            syn_resolved={"k": "CHEMBL103"},
            syn_candidates={"k": {"CHEMBL103"}},
            who_resolved={"k": "CHEMBL105"},
            who_candidates={"k": {"CHEMBL105"}},
            parent_of_cid={
                "CHEMBL100": 100, "CHEMBL101": 102,
                "CHEMBL103": 100,  # consistent
                "CHEMBL105": 100,
            },
        )
        cid, origin = _resolve_synonym_index("k", index)
        assert cid == "CHEMBL103"
        assert origin == "syn-rescue"


# ---------------------------------------------------------------------------
# parent/salt unification
# ---------------------------------------------------------------------------


from repogen.data.drug_loader import _canonicalize_to_parent_chembl_id  # noqa: E402


class TestChemblNameIndexParentMaps:
    """``parent_chembl_id_of_cid`` direct-lookup map correctness."""

    def test_parent_map_resolves_salt_to_parent(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        # Salt + parent both have pref_name -> both are in the
        # candidate set, so the candidate-scoped path alone is
        # sufficient here (no extra_cids needed for this assertion).
        index = load_chembl_synonym_index(db)
        # CHEMBL120 is a salt of CHEMBL121 (mol 120 -> parent_molregno 121).
        assert index.parent_chembl_id_of_cid["CHEMBL120"] == "CHEMBL121"
        # CHEMBL121 is its own parent.
        assert index.parent_chembl_id_of_cid["CHEMBL121"] == "CHEMBL121"

    def test_parent_map_self_parent_for_terminal(self, tmp_path: Path) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        # CHEMBL10 is terminal (parent_molregno = 10 = self).
        assert index.parent_chembl_id_of_cid["CHEMBL10"] == "CHEMBL10"

    def test_parent_map_covers_existing_salt_to_parent(
        self, tmp_path: Path,
    ) -> None:
        # CHEMBL20 is a salt of CHEMBL30 (per the existing fixture).
        # Both must end up in parent_chembl_id_of_cid.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        assert index.parent_chembl_id_of_cid["CHEMBL20"] == "CHEMBL30"
        assert index.parent_chembl_id_of_cid["CHEMBL30"] == "CHEMBL30"


class TestLoadChemblSynonymIndexExtraCids:
    """Parent/salt extension of :func:`load_chembl_synonym_index`."""

    def test_extra_cids_none_preserves_v1_coverage(
        self, tmp_path: Path,
    ) -> None:
        # Default: extra_cids omitted -> coverage equals candidate-set.
        # Nameless mols 130/131 (no pref/syn/who name) are NOT in the
        # candidate set -> not in parent_chembl_id_of_cid.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        assert "CHEMBL130" not in index.parent_chembl_id_of_cid
        assert "CHEMBL131" not in index.parent_chembl_id_of_cid

    def test_extra_cids_extends_parent_map_for_unnamed_cids(
        self, tmp_path: Path,
    ) -> None:
        # extra_cids forces parent resolution for cids the candidate
        # path missed.  CHEMBL130 has parent_molregno=131 -> mapped to
        # CHEMBL131 even though neither has a pref_name.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(
            db, extra_cids={"CHEMBL130", "CHEMBL131"},
        )
        assert index.parent_chembl_id_of_cid["CHEMBL130"] == "CHEMBL131"
        assert index.parent_chembl_id_of_cid["CHEMBL131"] == "CHEMBL131"

    def test_extra_cids_self_parent_when_no_hierarchy_row(
        self, tmp_path: Path,
    ) -> None:
        # Synthesise a CID with NO molecule_hierarchy entry.  In real
        # ChEMBL this almost never happens (hierarchy has the same
        # molregnos as molecule_dictionary); but if it did, the
        # extension should fall back to self-parent rather than
        # raising or producing a wrong rewrite.
        db_path = tmp_path / "lone_chembl.db"
        conn = sqlite3.connect(str(db_path))
        try:
            c = conn.cursor()
            c.executescript("""
                CREATE TABLE molecule_dictionary (
                    molregno INTEGER PRIMARY KEY, chembl_id TEXT,
                    pref_name TEXT, max_phase INTEGER, molecule_type TEXT
                );
                CREATE TABLE molecule_hierarchy (
                    molregno INTEGER PRIMARY KEY, parent_molregno INTEGER
                );
                CREATE TABLE compound_structures (
                    molregno INTEGER PRIMARY KEY,
                    standard_inchi_key TEXT, canonical_smiles TEXT
                );
                CREATE TABLE drug_mechanism (
                    mec_id INTEGER PRIMARY KEY, molregno INTEGER,
                    mechanism_of_action TEXT, action_type TEXT, tid INTEGER
                );
                CREATE TABLE molecule_synonyms (
                    molregno INTEGER, synonyms TEXT, syn_type TEXT
                );
                CREATE TABLE atc_classification (
                    level5 TEXT PRIMARY KEY, who_name TEXT,
                    level1 TEXT, level2 TEXT, level3 TEXT, level4 TEXT
                );
                CREATE TABLE molecule_atc_classification (
                    molregno INTEGER, level5 TEXT
                );
                CREATE TABLE drug_indication (
                    molregno INTEGER, mesh_heading TEXT
                );
            """)
            c.execute(
                "INSERT INTO molecule_dictionary VALUES (200, 'CHEMBL200',"
                " '', 4, 'Small molecule')"
            )
            conn.commit()
        finally:
            conn.close()

        index = load_chembl_synonym_index(
            db_path, extra_cids={"CHEMBL200"},
        )
        assert index.parent_chembl_id_of_cid["CHEMBL200"] == "CHEMBL200"

    def test_extra_cids_idempotent_when_overlap_with_candidates(
        self, tmp_path: Path,
    ) -> None:
        # Pass extra_cids that already lived in the candidate set;
        # parent_chembl_id_of_cid must be unchanged for those cids
        # (no double-mapping or errors).
        db = _build_chembl_mini_db(tmp_path)
        index_baseline = load_chembl_synonym_index(db)
        index_overlap = load_chembl_synonym_index(
            db,
            extra_cids={"CHEMBL10", "CHEMBL20", "CHEMBL120"},
        )
        for cid in ("CHEMBL10", "CHEMBL20", "CHEMBL120"):
            assert (
                index_baseline.parent_chembl_id_of_cid[cid]
                == index_overlap.parent_chembl_id_of_cid[cid]
            )

    def test_extra_cids_unknown_cid_is_silently_skipped(
        self, tmp_path: Path,
    ) -> None:
        # Pass a cid that doesn't exist in molecule_dictionary; it
        # shouldn't appear in parent_chembl_id_of_cid (the
        # canonicalization helper treats absent cids as
        # already-canonical).
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(
            db, extra_cids={"CHEMBL_DOES_NOT_EXIST"},
        )
        assert "CHEMBL_DOES_NOT_EXIST" not in index.parent_chembl_id_of_cid


class TestCanonicalizeToParentChemblId:
    """Vectorized parent rewrite helper."""

    def _make_index_with_parent_map(
        self,
        parent_chembl_id_of_cid: dict[str, str],
    ) -> ChemblNameIndex:
        return ChemblNameIndex(
            tier_resolved={"pref": {}, "syn": {}, "who": {}},
            tier_candidates={"pref": {}, "syn": {}, "who": {}},
            parent_of_cid={},
            parent_chembl_id_of_cid=parent_chembl_id_of_cid,
            chembl_to_atc={},
            chembl_to_ind={},
        )

    def test_salt_rewritten_to_parent(self) -> None:
        index = self._make_index_with_parent_map(
            {"CHEMBL120": "CHEMBL121", "CHEMBL121": "CHEMBL121"}
        )
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL120", "CHEMBL121"],
            "gene_symbol": ["GENE", "GENE"],
        })
        out = _canonicalize_to_parent_chembl_id(df, index)
        assert out["drug_chembl_id"].tolist() == ["CHEMBL121", "CHEMBL121"]

    def test_self_parent_unchanged(self) -> None:
        index = self._make_index_with_parent_map(
            {"CHEMBL10": "CHEMBL10"}
        )
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL10"],
            "gene_symbol": ["GENE"],
        })
        out = _canonicalize_to_parent_chembl_id(df, index)
        assert out["drug_chembl_id"].iloc[0] == "CHEMBL10"

    def test_unmapped_cid_kept_as_is(self) -> None:
        index = self._make_index_with_parent_map(
            {"CHEMBL10": "CHEMBL10"}
        )
        df = pd.DataFrame({
            # Mix of known + placeholder + unrelated CHEMBL not in map.
            "drug_chembl_id": ["CHEMBL10", "DGIDB_unknown", "CHEMBL999"],
            "gene_symbol": ["G1", "G2", "G3"],
        })
        out = _canonicalize_to_parent_chembl_id(df, index)
        assert out["drug_chembl_id"].tolist() == [
            "CHEMBL10",
            "DGIDB_unknown",
            "CHEMBL999",
        ]

    def test_does_not_rewrite_drug_name(self) -> None:
        # MVP correctness: only drug_chembl_id is rewritten; drug_name
        # stays whatever the row had.  The design keeps name canonicalization
        # out of scope for this iteration.
        index = self._make_index_with_parent_map(
            {"CHEMBL120": "CHEMBL121", "CHEMBL121": "CHEMBL121"}
        )
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL120"],
            "drug_name": ["salt_drug"],
            "gene_symbol": ["GENE"],
        })
        out = _canonicalize_to_parent_chembl_id(df, index)
        assert out["drug_chembl_id"].iloc[0] == "CHEMBL121"
        assert out["drug_name"].iloc[0] == "salt_drug"

    def test_empty_df_passthrough(self) -> None:
        index = self._make_index_with_parent_map({"CHEMBL10": "CHEMBL10"})
        out = _canonicalize_to_parent_chembl_id(pd.DataFrame(), index)
        assert out.empty

    def test_empty_parent_map_passthrough(self) -> None:
        # If the index has an empty parent map (e.g. older index
        # without parent-map fields), the helper is a no-op rather than
        # corrupting drug_chembl_id values.
        index = self._make_index_with_parent_map({})
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL10", "CHEMBL120"],
            "gene_symbol": ["G", "G"],
        })
        out = _canonicalize_to_parent_chembl_id(df, index)
        assert out["drug_chembl_id"].tolist() == ["CHEMBL10", "CHEMBL120"]

    def test_emits_coverage_metrics_to_log(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        index = self._make_index_with_parent_map(
            {"CHEMBL120": "CHEMBL121", "CHEMBL121": "CHEMBL121"}
        )
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL120", "CHEMBL121", "DGIDB_x"],
            "gene_symbol": ["G", "G", "G"],
        })
        with caplog.at_level("INFO", logger="repogen.data.drug_loader"):
            _canonicalize_to_parent_chembl_id(df, index)
        # All four observability metrics must be logged.
        log_text = caplog.text
        assert "Parent/salt canonicalization" in log_text
        assert "unique_cids_seen" in log_text
        assert "parent_map_hits" in log_text
        assert "rewritten" in log_text
        assert "unmapped_kept_self" in log_text


class TestParentSaltUnificationIntegration:
    """End-to-end behavior of ``merge_sources`` with parent/salt unification enabled."""

    def _make_chembl_df(
        self,
        rows: list[dict],
    ) -> pd.DataFrame:
        defaults = {
            "drug_chembl_id": "",
            "drug_name": "",
            "gene_symbol": "GENE_X",
            "max_phase": 4,
            "source": "chembl",
            "confidence": "high",
            "atc_codes": None,
            "interaction_type": "antagonist",
            "pchembl_value": None,
        }
        records = []
        for r in rows:
            base = dict(defaults)
            base.update(r)
            records.append(base)
        return pd.DataFrame(records)

    def test_merge_sources_collapses_parent_and_salt(
        self, tmp_path: Path,
    ) -> None:
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        # Synthetic chembl rows: parent + salt on the same gene.
        df = self._make_chembl_df([
            {"drug_chembl_id": "CHEMBL120", "drug_name": "salt_drug",
             "atc_codes": None},
            {"drug_chembl_id": "CHEMBL121", "drug_name": "parent_drug",
             "atc_codes": ["N06AA09"]},
        ])
        merged = merge_sources(
            {"chembl": df},
            chembl_name_index=index,
            parent_salt_unification=True,
        )
        # Salt collapsed onto parent; one row per (drug, gene).
        assert merged["drug_chembl_id"].tolist() == ["CHEMBL121"]
        # ATC from the parent is preserved (ATC propagation and dedup interplay).
        atc = merged["atc_codes"].iloc[0]
        atc_list = list(atc) if atc is not None else []
        assert "N06AA09" in atc_list

    def test_merge_sources_unions_genes_across_parent_and_salt(
        self, tmp_path: Path,
    ) -> None:
        # If parent + salt hit different genes, post-canonicalization
        # the merged row set has both genes attributed to the same
        # canonical parent CID.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_chembl_df([
            {"drug_chembl_id": "CHEMBL120", "drug_name": "salt_drug",
             "gene_symbol": "GENE_A"},
            {"drug_chembl_id": "CHEMBL121", "drug_name": "parent_drug",
             "gene_symbol": "GENE_B"},
        ])
        merged = merge_sources(
            {"chembl": df},
            chembl_name_index=index,
            parent_salt_unification=True,
        )
        # Two rows for CHEMBL121 (one per gene); zero rows for CHEMBL120.
        assert (merged["drug_chembl_id"] == "CHEMBL121").sum() == 2
        assert (merged["drug_chembl_id"] == "CHEMBL120").sum() == 0
        assert set(merged["gene_symbol"]) == {"GENE_A", "GENE_B"}

    def test_psu_disabled_keeps_separate_rows(self, tmp_path: Path) -> None:
        # Rollback / debugging path: parent_salt_unification=False
        # leaves child + parent CIDs distinct.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_chembl_df([
            {"drug_chembl_id": "CHEMBL120", "drug_name": "salt_drug",
             "gene_symbol": "GENE_A"},
            {"drug_chembl_id": "CHEMBL121", "drug_name": "parent_drug",
             "gene_symbol": "GENE_A"},
        ])
        merged = merge_sources(
            {"chembl": df},
            chembl_name_index=index,
            parent_salt_unification=False,
        )
        cids = sorted(merged["drug_chembl_id"].unique().tolist())
        assert cids == ["CHEMBL120", "CHEMBL121"]

    def test_extra_cids_collapse_unnamed_parent_salt_pair(
        self, tmp_path: Path,
    ) -> None:
        # CHEMBL130 (salt) / CHEMBL131 (parent) are NOT in the
        # candidate name set, so without extra_cids the index would
        # silently miss them.  With extra_cids, the canonicalization
        # collapses them -> exactly the coverage gap this closes.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(
            db, extra_cids={"CHEMBL130", "CHEMBL131"},
        )
        df = self._make_chembl_df([
            {"drug_chembl_id": "CHEMBL130", "drug_name": "nameless_salt"},
            {"drug_chembl_id": "CHEMBL131", "drug_name": "nameless_parent"},
        ])
        merged = merge_sources(
            {"chembl": df},
            chembl_name_index=index,
            parent_salt_unification=True,
        )
        assert merged["drug_chembl_id"].tolist() == ["CHEMBL131"]

    def test_single_source_path_runs_canonicalization(
        self, tmp_path: Path,
    ) -> None:
        # Regression guard: the single-source
        # path no longer takes a short-circuit return.  Default
        # config (sources=["chembl"]) MUST still trigger unification.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_chembl_df([
            {"drug_chembl_id": "CHEMBL120", "drug_name": "salt_drug"},
            {"drug_chembl_id": "CHEMBL121", "drug_name": "parent_drug"},
        ])
        merged = merge_sources(
            {"chembl": df},  # exactly len(dataframes) == 1
            chembl_name_index=index,
            parent_salt_unification=True,
        )
        # Canonicalization fired even on the single-source path;
        # both rows now share CHEMBL121 and the dedup step
        # collapsed them.
        assert (merged["drug_chembl_id"] == "CHEMBL121").sum() == 1
        assert (merged["drug_chembl_id"] == "CHEMBL120").sum() == 0

    def test_single_source_dgidb_no_chembl_index_no_op(self) -> None:
        # Single-source DGIdb (no chembl, no index): canonicalization
        # is gated off; behaviour matches the plain single-source
        # path (placeholder ID preserved, transient column stripped,
        # row count unchanged).
        df = pd.DataFrame({
            "drug_chembl_id": ["DGIDB_x", "DGIDB_y"],
            "drug_name": ["x", "y"],
            "gene_symbol": ["G", "G"],
            "source": ["dgidb", "dgidb"],
            "confidence": ["low", "low"],
            "max_phase": [0, 0],
            "_drug_claim_name": ["", ""],
        })
        merged = merge_sources(
            {"dgidb": df},
            chembl_name_index=None,
            parent_salt_unification=True,
        )
        assert "_drug_claim_name" not in merged.columns
        assert set(merged["drug_chembl_id"]) == {"DGIDB_x", "DGIDB_y"}

    def test_default_kwarg_enables_unification(
        self, tmp_path: Path,
    ) -> None:
        # Defensive: default for parent_salt_unification is True so
        # callers that DON'T pass the kwarg still get unification.
        db = _build_chembl_mini_db(tmp_path)
        index = load_chembl_synonym_index(db)
        df = self._make_chembl_df([
            {"drug_chembl_id": "CHEMBL120", "drug_name": "salt_drug"},
            {"drug_chembl_id": "CHEMBL121", "drug_name": "parent_drug"},
        ])
        merged = merge_sources({"chembl": df}, chembl_name_index=index)
        # Default behavior: collapsed onto parent.
        assert merged["drug_chembl_id"].tolist() == ["CHEMBL121"]


# ---------------------------------------------------------------------------
# Expression-perturbation evidence
# ---------------------------------------------------------------------------


class TestTokenisedConfidenceUpgrade:
    """``_deduplicate_records`` confidence-upgrade rule.

    Critical-criteria checks:
      * Existing target-only ≥2 source agreement still upgrades
        (regression guard for the target-only behaviour).
      * Compound source tokens (``"chembl,dgidb"``) - created by an
        earlier dedup pass - are tokenised correctly.
      * Mixed target+expression evidence does NOT upgrade.
    """

    def _row(self, **overrides) -> dict:
        base = {
            "drug_chembl_id": "CHEMBL1",
            "drug_name": "drugA",
            "gene_symbol": "GENE1",
            "interaction_type": "inhibitor",
            "max_phase": 0,
            "confidence": "low",
        }
        base.update(overrides)
        return base

    def test_two_target_sources_upgrade_to_high(self) -> None:
        df = pd.DataFrame([
            self._row(source="chembl"),
            self._row(source="dgidb"),
        ])
        out = _deduplicate_records(df)
        assert len(out) == 1
        assert out.loc[0, "confidence"] == "high"
        assert set(out.loc[0, "source"].split(",")) == {"chembl", "dgidb"}

    def test_compound_token_plus_third_target_upgrades(self) -> None:
        # Simulates a row whose ``source`` was already merged by an
        # earlier dedup pass (e.g. "chembl,dgidb") meeting a fresh
        # PDSP row.  Three atomic target-family tokens -> upgrade.
        df = pd.DataFrame([
            self._row(source="chembl,dgidb", confidence="high"),
            self._row(source="pdsp"),
        ])
        out = _deduplicate_records(df)
        assert len(out) == 1
        assert out.loc[0, "confidence"] == "high"
        assert set(out.loc[0, "source"].split(",")) == {
            "chembl", "dgidb", "pdsp",
        }

    def test_target_plus_expression_does_not_upgrade(self) -> None:
        # ChEMBL + CREEDS = 1 target token, 1 expression token.
        # Expression evidence does NOT independently confirm a target
        # claim, so confidence stays at the better of the input rows
        # (low here) - no upgrade to "high".
        df = pd.DataFrame([
            self._row(source="chembl"),
            self._row(source="creeds"),
        ])
        out = _deduplicate_records(df)
        assert len(out) == 1
        assert out.loc[0, "confidence"] != "high"
        assert set(out.loc[0, "source"].split(",")) == {"chembl", "creeds"}

    def test_two_expression_sources_do_not_upgrade(self) -> None:
        # CREEDS + DSigDB: two expression tokens, zero target tokens
        # in the family-intersection.  Must NOT upgrade.
        df = pd.DataFrame([
            self._row(source="creeds"),
            self._row(source="dsigdb"),
        ])
        out = _deduplicate_records(df)
        assert len(out) == 1
        assert out.loc[0, "confidence"] != "high"

    def test_compound_target_token_plus_expression_does_not_upgrade(
        self,
    ) -> None:
        # The harder edge case: pre-merged "chembl,dgidb" + creeds row.
        # The compound row already had high confidence; we must NOT
        # downgrade it, but the expression source must not extend the
        # upgrade either.  The compound row remains "high" simply
        # because it already was.
        df = pd.DataFrame([
            self._row(source="chembl,dgidb", confidence="high"),
            self._row(source="creeds", confidence="low"),
        ])
        out = _deduplicate_records(df)
        assert len(out) == 1
        # Carries forward the existing "high" from the richer row;
        # expression token does NOT *trigger* the upgrade itself.
        assert out.loc[0, "confidence"] == "high"
        assert set(out.loc[0, "source"].split(",")) == {
            "chembl", "dgidb", "creeds",
        }

    def test_single_source_no_upgrade(self) -> None:
        # Defensive regression: one row, one source token.  No
        # multi-source agreement -> no upgrade.
        df = pd.DataFrame([self._row(source="chembl")])
        out = _deduplicate_records(df)
        assert len(out) == 1
        assert out.loc[0, "confidence"] == "low"


class TestLoadCreeds:
    """CREEDS loader."""

    def test_load_minimal_payload(self, tmp_path: Path) -> None:
        import json

        payload = [
            {
                "id": "drug:1",
                "drug_name": "tamoxifen",
                "up_genes": [["ESR1", -3.5], ["TP53", -2.8]],
                "down_genes": [["BRCA1", 2.1]],
            },
            {
                "id": "drug:2",
                "drug_name": "aspirin",
                "up_genes": [["PTGS1", -1.2]],
                "down_genes": [],
            },
        ]
        path = tmp_path / "creeds.json"
        path.write_text(json.dumps(payload))

        df = load_creeds(path)
        # 2 up + 1 down (tamoxifen) + 1 up + 0 down (aspirin) = 4 pairs
        assert len(df) == 4
        # Source labelling
        assert (df["source"] == "creeds").all()
        # Standardised interaction types
        types = set(df["interaction_type"])
        assert types == {"expression_up", "expression_down"}
        # Placeholder IDs use CREEDS_<name>
        assert df["drug_chembl_id"].str.startswith("CREEDS_").all()
        # Low confidence by default
        assert (df["confidence"] == "low").all()

    def test_load_handles_bare_symbol_lists(self, tmp_path: Path) -> None:
        import json

        payload = [{
            "drug_name": "drugX",
            "up_genes": ["GENE1", "GENE2"],
            "down_genes": [],
        }]
        path = tmp_path / "creeds.json"
        path.write_text(json.dumps(payload))
        df = load_creeds(path)
        assert len(df) == 2
        assert set(df["gene_symbol"]) == {"GENE1", "GENE2"}

    def test_load_skips_entries_with_no_drug_name(
        self, tmp_path: Path,
    ) -> None:
        import json

        payload = [{
            "drug_name": "",
            "up_genes": [["G1", -1.0]],
            "down_genes": [],
        }]
        path = tmp_path / "creeds.json"
        path.write_text(json.dumps(payload))
        df = load_creeds(path)
        assert df.empty


class TestLoadDsigdb:
    """DSigDB D3 loader."""

    def test_load_basic_gmt(self, tmp_path: Path) -> None:
        path = tmp_path / "dsigdb_d3.tsv"
        path.write_text(
            "Tamoxifen_MCF7_GSE1\tDescription\tESR1\tPGR\tBRCA1\n"
            "Aspirin_HEPG2_GSE2\tDescription\tPTGS1\tPTGS2\n"
        )
        df = load_dsigdb(path)
        assert len(df) == 5
        assert (df["source"] == "dsigdb").all()
        assert (df["interaction_type"] == "expression_perturbation").all()
        assert df["drug_chembl_id"].str.startswith("DSIGDB_").all()
        # Drug names are the leading underscore-delimited segment.
        assert set(df["drug_name"]) == {"Tamoxifen", "Aspirin"}
        # Low confidence by default.
        assert (df["confidence"] == "low").all()

    def test_skips_blank_and_comment_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "dsigdb_d3.tsv"
        path.write_text(
            "# header comment\n"
            "\n"
            "DrugA_meta\tdesc\tG1\tG2\n"
        )
        df = load_dsigdb(path)
        assert len(df) == 2
        assert set(df["gene_symbol"]) == {"G1", "G2"}

    def test_handles_space_separated_drug_token(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "dsigdb_d3.tsv"
        path.write_text("DrugA Cell GSE1\tdesc\tG1\n")
        df = load_dsigdb(path)
        assert df.loc[0, "drug_name"] == "DrugA"


class TestLoadDrugTargetsExpression:
    """``load_drug_targets`` plumbing for expr sources."""

    def test_disjoint_set_check_at_function_boundary(
        self, tmp_path: Path,
    ) -> None:
        # The Pydantic schema enforces disjoint sets at config-load
        # time.  ``load_drug_targets`` adds a defence-in-depth check
        # for programmatic callers that bypass the schema.
        with pytest.raises(ValueError, match="disjoint"):
            load_drug_targets(
                sources=["chembl", "creeds"],   # cross-domain
                expression_sources=["creeds"],
            )

    def test_mode_off_default_no_expression_sources(self) -> None:
        # When expression_sources is None / empty, nothing about the
        # behaviour vs the legacy call should change.  Verify the
        # signature accepts None and defaults to [].
        import inspect

        sig = inspect.signature(load_drug_targets)
        assert sig.parameters["expression_sources"].default is None
        # Kwarg-only parameter (declared after the * marker).
        assert (
            sig.parameters["expression_sources"].kind
            == inspect.Parameter.KEYWORD_ONLY
        )


class TestExpressionMatcher:
    """``_match_expression_source_to_chembl``."""

    def test_empty_input_passes_through(self) -> None:
        df = pd.DataFrame()
        out = _match_expression_source_to_chembl(
            df, index=None, source_label="creeds",  # type: ignore[arg-type]
        )
        assert out.empty

    def test_real_chembl_id_left_alone(self) -> None:
        # Build a minimal ChemblNameIndex with no resolution paths so
        # the matcher's "needs_resolve" mask is empty for real IDs.
        df = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL83", "CREEDS_unknown"],
            "drug_name": ["tamoxifen", "made-up"],
            "gene_symbol": ["G1", "G2"],
            "source": ["creeds", "creeds"],
            "confidence": ["low", "low"],
        })
        # The synonym resolver expects the three tier keys present;
        # the inner dicts can be empty.  Mirrors how a real index built
        # against an empty ChEMBL would look.
        empty_index = ChemblNameIndex(
            tier_resolved={"pref": {}, "syn": {}, "who": {}},
            tier_candidates={"pref": {}, "syn": {}, "who": {}},
            parent_of_cid={},
            parent_chembl_id_of_cid={},
            chembl_to_atc={},
            chembl_to_ind={},
        )
        out = _match_expression_source_to_chembl(
            df, empty_index, source_label="creeds",
        )
        # The real CHEMBL ID is unchanged; the placeholder was unable
        # to resolve through an empty index, so it stays.
        assert out.loc[0, "drug_chembl_id"] == "CHEMBL83"
        assert out.loc[1, "drug_chembl_id"] == "CREEDS_unknown"
