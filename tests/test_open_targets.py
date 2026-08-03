"""Tests for repogen.data.open_targets."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from repogen.data.open_targets import (
    _ASSOC_PAGE_SIZE,
    annotate_with_open_targets,
    query_open_targets_api,
    _query_disease_associations,
    _query_tractability,
    _load_cache,
    _save_cache,
)


class TestAnnotateWithOpenTargets:
    """Tests for the main annotation function."""

    def test_adds_null_columns_when_no_gene_column(self) -> None:
        df = pd.DataFrame({"gene_symbol": ["BRAF", "PTEN"]})
        result = annotate_with_open_targets(
            df, disease_efo_id="EFO_0003761",
            gene_column="gene_ensembl_id",
        )
        assert "ot_association_score" in result.columns
        assert "ot_tractability_sm" in result.columns
        assert result["ot_association_score"].isna().all()

    def test_adds_null_columns_when_no_ensembl_ids(self) -> None:
        df = pd.DataFrame({"gene_ensembl_id": [None, None]})
        result = annotate_with_open_targets(
            df, disease_efo_id="EFO_0003761",
        )
        assert "ot_association_score" in result.columns

    @patch("repogen.data.open_targets.query_open_targets_api")
    def test_uses_api_results(self, mock_api: MagicMock) -> None:
        mock_api.return_value = {
            "ENSG00000157764": {
                "association_score": 0.85,
                "genetic_score": 0.6,
                "known_drug_score": 0.9,
                "tractability_sm": True,
                "tractability_ab": False,
            },
        }
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG00000157764", "ENSG00000999999"],
        })
        result = annotate_with_open_targets(
            df, disease_efo_id="EFO_0003761",
        )
        assert result.loc[0, "ot_association_score"] == 0.85
        assert bool(result.loc[0, "ot_tractability_sm"]) is True

    def test_does_not_modify_input(self) -> None:
        df = pd.DataFrame({"gene_ensembl_id": ["ENSG00000157764"]})
        original_cols = list(df.columns)
        _ = annotate_with_open_targets(
            df, disease_efo_id="EFO_0003761", use_api=False,
        )
        assert list(df.columns) == original_cols


class TestQueryDiseaseAssociations:
    """Tests for the paginated disease association query."""

    @patch("repogen.data.open_targets._execute_graphql")
    def test_parses_disease_response(self, mock_gql: MagicMock) -> None:
        mock_gql.return_value = {
            "data": {
                "disease": {
                    "associatedTargets": {
                        "count": 1,
                        "rows": [
                            {
                                "target": {"id": "ENSG00000157764", "approvedSymbol": "BRAF"},
                                "score": 0.85,
                                "datatypeScores": [
                                    {"id": "ot_genetics_portal", "score": 0.6},
                                    {"id": "chembl", "score": 0.9},
                                ],
                            },
                        ],
                    },
                },
            },
        }
        result = _query_disease_associations(
            "EFO_0003761", {"ENSG00000157764"},
        )
        assert result["ENSG00000157764"]["association_score"] == 0.85
        assert result["ENSG00000157764"]["genetic_score"] == 0.6
        assert result["ENSG00000157764"]["known_drug_score"] == 0.9

    @patch("repogen.data.open_targets._execute_graphql")
    def test_null_disease_returns_empty(self, mock_gql: MagicMock) -> None:
        mock_gql.return_value = {"data": {"disease": None}}
        result = _query_disease_associations(
            "INVALID_EFO", {"ENSG00000157764"},
        )
        assert result == {}

    @patch("repogen.data.open_targets._execute_graphql")
    def test_filters_to_our_ids(self, mock_gql: MagicMock) -> None:
        mock_gql.return_value = {
            "data": {
                "disease": {
                    "associatedTargets": {
                        "count": 2,
                        "rows": [
                            {"target": {"id": "ENSG_OURS"}, "score": 0.5, "datatypeScores": []},
                            {"target": {"id": "ENSG_OTHER"}, "score": 0.9, "datatypeScores": []},
                        ],
                    },
                },
            },
        }
        result = _query_disease_associations("EFO_001", {"ENSG_OURS"})
        assert "ENSG_OURS" in result
        assert "ENSG_OTHER" not in result


class TestQueryTractability:
    """Tests for the batch tractability query."""

    @patch("repogen.data.open_targets._execute_graphql")
    def test_parses_tractability(self, mock_gql: MagicMock) -> None:
        mock_gql.return_value = {
            "data": {
                "targets": [
                    {
                        "id": "ENSG00000157764",
                        "tractability": [
                            {"label": "sm", "modality": "SM", "value": True},
                            {"label": "ab", "modality": "AB", "value": False},
                        ],
                    },
                ],
            },
        }
        result = _query_tractability(["ENSG00000157764"])
        assert result["ENSG00000157764"]["tractability_sm"] is True
        assert result["ENSG00000157764"]["tractability_ab"] is False

    @patch("repogen.data.open_targets._execute_graphql")
    def test_api_failure_returns_empty(self, mock_gql: MagicMock) -> None:
        mock_gql.return_value = None
        result = _query_tractability(["ENSG00000157764"])
        assert result == {}


class TestDiseaseAssociationsPagination:
    """Test that _query_disease_associations actually paginates."""

    def test_disease_associations_paginates(self, monkeypatch) -> None:
        total_targets = _ASSOC_PAGE_SIZE + 1

        def make_row(ensembl_id, score):
            return {
                "target": {"id": ensembl_id, "approvedSymbol": f"SYM_{ensembl_id}"},
                "score": score,
                "datatypeScores": [],
            }

        page_0_rows = [make_row(f"ENSG{i:011d}", 0.5) for i in range(_ASSOC_PAGE_SIZE)]
        page_1_rows = [make_row(f"ENSG{_ASSOC_PAGE_SIZE:011d}", 0.3)]

        call_count = 0

        def mock_execute(payload):
            nonlocal call_count
            call_count += 1
            page_index = payload.get("variables", {}).get("index", 0)
            rows = page_0_rows if page_index == 0 else page_1_rows
            return {
                "data": {
                    "disease": {
                        "associatedTargets": {
                            "count": total_targets,
                            "rows": rows,
                        }
                    }
                }
            }

        monkeypatch.setattr(
            "repogen.data.open_targets._execute_graphql", mock_execute
        )

        target_ids = {f"ENSG{0:011d}", f"ENSG{_ASSOC_PAGE_SIZE:011d}"}
        result = _query_disease_associations("EFO_0000001", target_ids)

        assert call_count == 2, f"Expected 2 API calls, got {call_count}"
        assert len(result) == 2


class TestCache:
    """Tests for caching functionality."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        data = {"ENSG001": {"association_score": 0.5}}
        _save_cache(tmp_path, "EFO_001", data)
        loaded = _load_cache(tmp_path, "EFO_001")
        assert loaded["ENSG001"]["association_score"] == 0.5

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        result = _load_cache(tmp_path, "NONEXISTENT")
        assert result == {}

    def test_load_none_dir(self) -> None:
        result = _load_cache(None, "EFO_001")
        assert result == {}
