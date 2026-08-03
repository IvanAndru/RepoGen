"""Tests for repogen.data.gene_id_converter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from repogen.data.gene_id_converter import GeneIDConverter

TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture()
def biomart_dir(tmp_path: Path) -> dict[str, Path]:
    """Create small synthetic BioMart dictionary files."""
    dico1 = tmp_path / "biomart_dico1"
    dico1.write_text(
        "ENSG00000102003\tHTR2A\n"
        "ENSG00000157764\tBRAF\n"
        "ENSG00000171862\tPTEN\n"
    )

    dico2 = tmp_path / "biomart_dico2"
    dico2.write_text(
        "HTR2A\tENSG00000102003\n"
        "BRAF\tENSG00000157764\n"
        "PTEN\tENSG00000171862\n"
    )

    dico3 = tmp_path / "biomart_dico3"
    dico3.write_text(
        "P28223\tENSG00000102003\n"
        "P15056\tENSG00000157764\n"
        "P60484\tENSG00000171862\n"
    )

    return {
        "ensembl_to_name": dico1,
        "name_to_ensembl": dico2,
        "uniprot_to_ensembl": dico3,
    }


@pytest.fixture()
def converter(biomart_dir: dict[str, Path]) -> GeneIDConverter:
    """Create a GeneIDConverter from synthetic data."""
    return GeneIDConverter(biomart_dicts=biomart_dir)


class TestGeneIDConverter:
    """Tests for the GeneIDConverter class."""

    def test_ensembl_to_symbol(self, converter: GeneIDConverter) -> None:
        result = converter.convert(["ENSG00000102003"], "ensembl", "symbol")
        assert result["ENSG00000102003"] == "HTR2A"

    def test_symbol_to_ensembl(self, converter: GeneIDConverter) -> None:
        result = converter.convert(["HTR2A"], "symbol", "ensembl")
        assert result["HTR2A"] == "ENSG00000102003"

    def test_uniprot_to_symbol(self, converter: GeneIDConverter) -> None:
        result = converter.convert(["P28223"], "uniprot", "symbol")
        assert result["P28223"] == "HTR2A"

    def test_unknown_id_returns_none(self, converter: GeneIDConverter) -> None:
        result = converter.convert(["NONEXISTENT"], "symbol", "ensembl")
        assert result["NONEXISTENT"] is None

    def test_get_full_record(self, converter: GeneIDConverter) -> None:
        rec = converter.get_full_record("HTR2A", "symbol")
        assert rec is not None
        assert rec["symbol"] == "HTR2A"
        assert rec["ensembl"] == "ENSG00000102003"
        assert rec["uniprot"] == "P28223"

    def test_get_full_record_unknown(self, converter: GeneIDConverter) -> None:
        rec = converter.get_full_record("FAKE_GENE", "symbol")
        assert rec is not None
        assert rec["symbol"] == "FAKE_GENE"
        assert rec["ensembl"] is None

    def test_get_full_record_truly_unknown(self, converter: GeneIDConverter) -> None:
        rec = converter.get_full_record("FAKE_ENSG", "ensembl")
        assert rec is not None
        assert rec["ensembl"] == "FAKE_ENSG"
        assert rec["symbol"] is None

    def test_invalid_id_type_raises(self, converter: GeneIDConverter) -> None:
        with pytest.raises(ValueError, match="Unknown id_type"):
            converter.get_full_record("HTR2A", "invalid_type")

    def test_batch_annotate(self, converter: GeneIDConverter) -> None:
        df = pd.DataFrame({"uniprot": ["P28223", "P15056", "UNKNOWN"]})
        result = converter.batch_annotate(df, "uniprot", "uniprot")
        assert "gene_symbol" in result.columns
        assert "gene_ensembl_id" in result.columns
        assert result.loc[0, "gene_symbol"] == "HTR2A"
        assert result.loc[1, "gene_symbol"] == "BRAF"
        assert pd.isna(result.loc[2, "gene_symbol"])

    def test_get_stats(self, converter: GeneIDConverter) -> None:
        stats = converter.get_stats()
        assert stats["ensembl_to_symbol"] == 3
        assert stats["uniprot_to_ensembl"] == 3

    def test_empty_biomart_tolerates(self, tmp_path: Path) -> None:
        conv = GeneIDConverter(biomart_dicts={})
        assert conv.get_stats()["ensembl_to_symbol"] == 0
        rec = conv.get_full_record("HTR2A", "symbol")
        assert rec is not None
        assert rec["symbol"] == "HTR2A"
        assert rec["ensembl"] is None

    def test_missing_file_warns(self, tmp_path: Path) -> None:
        dicts = {"ensembl_to_name": tmp_path / "nonexistent"}
        conv = GeneIDConverter(biomart_dicts=dicts)
        assert conv.get_stats()["ensembl_to_symbol"] == 0
