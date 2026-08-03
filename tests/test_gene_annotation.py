"""Tests for repogen.data.gene_annotation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from repogen.data.gene_annotation import (
    _classify_id_column,
    _read_gene_location_file,
    annotate_genes,
    create_gene_info,
    create_magma_annotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_gene_loc_6col(path: Path, n: int = 10) -> None:
    """Write a 6-column NCBI-style gene location file."""
    lines = []
    for i in range(1, n + 1):
        lines.append(
            f"ENSG0000000{i:04d}\t{(i % 22) + 1}\t{i * 100000}\t{i * 100000 + 50000}\t+\tGENE{i}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_gene_loc_entrez(path: Path, n: int = 10) -> None:
    """Write a 6-column gene location file with Entrez IDs (like NCBI37.3.gene.loc)."""
    lines = []
    for i in range(1, n + 1):
        lines.append(
            f"{79500 + i}\t{(i % 22) + 1}\t{i * 100000}\t{i * 100000 + 50000}\t+\tGENE{i}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_gene_loc_4col(path: Path, n: int = 5) -> None:
    """Write a 4-column gene location file (no strand, no name)."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"ENSG0000000{i:04d}\t{(i % 22) + 1}\t{i * 100000}\t{i * 100000 + 50000}")
    path.write_text("\n".join(lines) + "\n")


def _make_mock_converter() -> MagicMock:
    """Create a mock GeneIDConverter with batch_annotate."""
    converter = MagicMock()

    def _batch_annotate(df, id_column, id_type):
        result = df.copy()
        if "gene_symbol" not in result.columns:
            result["gene_symbol"] = pd.NA
        if "gene_entrez_id" not in result.columns:
            result["gene_entrez_id"] = pd.NA
        if "gene_uniprot_id" not in result.columns:
            result["gene_uniprot_id"] = pd.NA
        for i in range(len(result)):
            idx = i + 1
            if pd.isna(result.loc[result.index[i], "gene_symbol"]):
                result.loc[result.index[i], "gene_symbol"] = f"GENE{idx}"
            result.loc[result.index[i], "gene_entrez_id"] = 100 + idx
            result.loc[result.index[i], "gene_uniprot_id"] = f"P{idx:05d}"
        return result

    converter.batch_annotate.side_effect = _batch_annotate
    return converter


# ---------------------------------------------------------------------------
# Tests: _read_gene_location_file
# ---------------------------------------------------------------------------


class TestReadGeneLocationFile:
    """Tests for gene location file parsing."""

    def test_six_column_file(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_6col(f, n=5)
        df = _read_gene_location_file(f)
        assert len(df) == 5
        assert "gene_ensembl_id" in df.columns
        assert "chr" in df.columns
        assert "start" in df.columns
        assert "end" in df.columns
        assert "gene_symbol" in df.columns
        assert df["gene_symbol"].iloc[0] == "GENE1"

    def test_four_column_file(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_4col(f, n=3)
        df = _read_gene_location_file(f)
        assert len(df) == 3
        assert df["gene_symbol"].isna().all()

    def test_chr_prefix_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        f.write_text("ENSG00000000001\tchr1\t100000\t150000\t+\tBRCA1\n")
        df = _read_gene_location_file(f)
        assert df["chr"].iloc[0] == 1

    def test_biotype_default_protein_coding(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_6col(f, n=2)
        df = _read_gene_location_file(f)
        assert (df["biotype"] == "protein_coding").all()

    def test_entrez_id_detection(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_entrez(f, n=5)
        df = _read_gene_location_file(f)
        assert df["gene_entrez_id"].notna().all()
        assert df["gene_ensembl_id"].isna().all()
        assert df["gene_symbol"].iloc[0] == "GENE1"

    def test_ensembl_id_detection(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_6col(f, n=5)
        df = _read_gene_location_file(f)
        assert df["gene_ensembl_id"].notna().all()
        assert df["gene_ensembl_id"].iloc[0].startswith("ENSG")

    def test_classify_id_column_ensembl(self) -> None:
        s = pd.Series(["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"])
        assert _classify_id_column(s) == "ensembl"

    def test_classify_id_column_entrez(self) -> None:
        s = pd.Series(["79501", "100996442", "729759", "81399"])
        assert _classify_id_column(s) == "entrez"

    def test_classify_id_column_mixed(self) -> None:
        s = pd.Series(["ENSG00000000001", "79501", "ENSG00000000002", "81399"])
        assert _classify_id_column(s) == "mixed"

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises((RuntimeError, pd.errors.EmptyDataError)):
            _read_gene_location_file(f)


# ---------------------------------------------------------------------------
# Tests: annotate_genes
# ---------------------------------------------------------------------------


class TestAnnotateGenes:
    """Tests for the main annotation function."""

    def test_basic_annotation_no_converter(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_6col(f, n=5)
        result = annotate_genes(reference_gene_file=f)
        assert len(result) == 5
        assert "gene_ensembl_id" in result.columns
        assert "gene_symbol" in result.columns
        assert result["gene_uniprot_id"].isna().all()

    def test_with_converter(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_6col(f, n=5)
        converter = _make_mock_converter()
        result = annotate_genes(
            reference_gene_file=f, gene_id_converter=converter,
        )
        assert len(result) == 5
        converter.batch_annotate.assert_called_once()
        assert result["gene_entrez_id"].notna().all()
        assert result["gene_uniprot_id"].notna().all()

    def test_biotype_filter(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_6col(f, n=10)
        result = annotate_genes(
            reference_gene_file=f, biotype_filter=["protein_coding"],
        )
        assert len(result) == 10
        assert (result["biotype"] == "protein_coding").all()

    def test_biotype_filter_excludes_all_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_6col(f, n=5)
        with pytest.raises(RuntimeError, match="No genes remained"):
            annotate_genes(
                reference_gene_file=f, biotype_filter=["lincRNA"],
            )

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            annotate_genes(reference_gene_file=tmp_path / "missing.txt")

    def test_deduplication(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        f.write_text(
            "ENSG00000000001\t1\t100000\t150000\t+\tGENEA\n"
            "ENSG00000000001\t1\t100000\t150000\t+\tGENEB\n"
            "ENSG00000000002\t2\t200000\t250000\t+\tGENEC\n"
        )
        result = annotate_genes(reference_gene_file=f)
        assert len(result) == 2

    def test_dedup_partial_ensembl_preserves_entrez_rows(self, tmp_path: Path) -> None:
        """When converter enriches only some rows with Ensembl IDs,
        rows with distinct Entrez IDs but missing Ensembl must not be
        collapsed (regression: NA == NA in drop_duplicates)."""
        f = tmp_path / "genes.txt"
        _write_gene_loc_entrez(f, n=3)
        converter = MagicMock()

        def _partial_annotate(df, id_column, id_type):
            result = df.copy()
            for col in ("gene_symbol", "gene_ensembl_id", "gene_uniprot_id", "gene_entrez_id"):
                if col not in result.columns:
                    result[col] = pd.NA
            result.loc[result.index[0], "gene_ensembl_id"] = "ENSG00000099999"
            return result

        converter.batch_annotate.side_effect = _partial_annotate
        result = annotate_genes(reference_gene_file=f, gene_id_converter=converter)
        assert len(result) == 3

    def test_output_columns(self, tmp_path: Path) -> None:
        f = tmp_path / "genes.txt"
        _write_gene_loc_6col(f, n=3)
        result = annotate_genes(reference_gene_file=f)
        expected_cols = {
            "gene_symbol", "gene_ensembl_id", "gene_entrez_id",
            "gene_uniprot_id", "chr", "start", "end", "biotype", "description",
        }
        assert set(result.columns) == expected_cols

    def test_entrez_gene_loc_preserves_ids(self, tmp_path: Path) -> None:
        """Entrez-style gene.loc should retain gene_entrez_id and gene_symbol."""
        f = tmp_path / "genes.txt"
        _write_gene_loc_entrez(f, n=5)
        result = annotate_genes(reference_gene_file=f)
        assert len(result) == 5
        assert result["gene_entrez_id"].notna().all()
        assert result["gene_symbol"].notna().all()
        assert result["gene_symbol"].iloc[0] == "GENE1"

    def test_entrez_gene_loc_with_converter_non_destructive(self, tmp_path: Path) -> None:
        """batch_annotate must not erase existing gene_symbol or gene_entrez_id."""
        f = tmp_path / "genes.txt"
        _write_gene_loc_entrez(f, n=5)
        converter = MagicMock()

        def _noop_annotate(df, id_column, id_type):
            result = df.copy()
            for col in ("gene_symbol", "gene_ensembl_id", "gene_uniprot_id", "gene_entrez_id"):
                if col not in result.columns:
                    result[col] = pd.NA
            return result

        converter.batch_annotate.side_effect = _noop_annotate
        result = annotate_genes(reference_gene_file=f, gene_id_converter=converter)
        assert result["gene_entrez_id"].notna().all()
        assert result["gene_symbol"].notna().all()

    def test_empty_gene_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises((RuntimeError, pd.errors.EmptyDataError)):
            annotate_genes(reference_gene_file=f)


# ---------------------------------------------------------------------------
# Tests: create_magma_annotation
# ---------------------------------------------------------------------------


class TestCreateMagmaAnnotation:
    """Tests for MAGMA annotation file generation."""

    def _make_gene_annotations(self) -> pd.DataFrame:
        return pd.DataFrame({
            "gene_ensembl_id": ["ENSG001", "ENSG002"],
            "gene_symbol": ["GENE1", "GENE2"],
            "chr": [1, 1],
            "start": [10000, 50000],
            "end": [20000, 60000],
            "biotype": ["protein_coding", "protein_coding"],
        })

    def _make_bim(self, path: Path) -> Path:
        bim = path / "ref.bim"
        bim.write_text(
            "1\trs1\t0\t15000\tA\tG\n"
            "1\trs2\t0\t55000\tC\tT\n"
            "1\trs3\t0\t90000\tA\tC\n"
            "2\trs4\t0\t10000\tA\tG\n"
        )
        return bim

    def test_basic_annotation(self, tmp_path: Path) -> None:
        genes = self._make_gene_annotations()
        bim = self._make_bim(tmp_path)
        gwas = pd.DataFrame({"SNP": ["rs1", "rs2", "rs3", "rs4"]})
        result = create_magma_annotation(genes, gwas, bim, window_kb=0)
        assert result.exists()
        content = result.read_text()
        assert "ENSG001" in content
        assert "rs1" in content

    def test_window_extends_boundaries(self, tmp_path: Path) -> None:
        genes = self._make_gene_annotations()
        bim = self._make_bim(tmp_path)
        gwas = pd.DataFrame({"SNP": ["rs1", "rs2", "rs3", "rs4"]})
        result_narrow = create_magma_annotation(genes, gwas, bim, window_kb=0)
        result_wide = create_magma_annotation(genes, gwas, bim, window_kb=100)
        content_narrow = result_narrow.read_text()
        content_wide = result_wide.read_text()
        assert len(content_wide) >= len(content_narrow)

    def test_no_matching_snps(self, tmp_path: Path) -> None:
        genes = pd.DataFrame({
            "gene_ensembl_id": ["ENSG001"],
            "gene_symbol": ["GENE1"],
            "chr": [22],
            "start": [10000],
            "end": [20000],
            "biotype": ["protein_coding"],
        })
        bim = self._make_bim(tmp_path)
        gwas = pd.DataFrame({"SNP": ["rs1"]})
        result = create_magma_annotation(genes, gwas, bim, window_kb=0)
        content = result.read_text().strip()
        assert content == ""


# ---------------------------------------------------------------------------
# Tests: create_gene_info
# ---------------------------------------------------------------------------


class TestCreateGeneInfo:
    """Tests for gene info DataFrame creation."""

    def test_selects_expected_columns(self) -> None:
        df = pd.DataFrame({
            "gene_symbol": ["A"],
            "gene_ensembl_id": ["ENSG1"],
            "gene_entrez_id": [100],
            "gene_uniprot_id": ["P00001"],
            "biotype": ["protein_coding"],
            "chr": [1],
            "start": [1000],
            "end": [2000],
            "description": ["test gene"],
        })
        result = create_gene_info(df)
        assert "gene_symbol" in result.columns
        assert "gene_ensembl_id" in result.columns
        assert "description" not in result.columns

    def test_handles_missing_columns(self) -> None:
        df = pd.DataFrame({
            "gene_symbol": ["A"],
            "gene_ensembl_id": ["ENSG1"],
            "chr": [1],
            "start": [1000],
            "end": [2000],
        })
        result = create_gene_info(df)
        assert "gene_symbol" in result.columns
        assert "gene_uniprot_id" not in result.columns


# ---------------------------------------------------------------------------
# Tests: parse_magma_results compatibility
# ---------------------------------------------------------------------------


class TestParseMagmaResultsCompat:
    """Verify that Entrez-only annotations work with parse_magma_results."""

    def test_entrez_annotations_merge_with_magma(self, tmp_path: Path) -> None:
        """gene_annotations with Entrez IDs (no Ensembl) should merge into
        MAGMA results without raising on .astype(int)."""
        from repogen.analysis.magma_gene import parse_magma_results

        genes_out = tmp_path / "test.genes.out"
        genes_out.write_text(
            "GENE CHR START STOP NSNPS NPARAM N ZSTAT P\n"
            "79501 1 69091 70008 5 3 50000 2.5 0.006\n"
            "148398 1 859993 879961 10 5 50000 3.1 0.001\n"
        )
        annot = pd.DataFrame({
            "gene_entrez_id": pd.array([79501, 148398], dtype=pd.Int64Dtype()),
            "gene_symbol": ["OR4F5", "SAMD11"],
            "gene_ensembl_id": [pd.NA, pd.NA],
            "biotype": ["protein_coding", "protein_coding"],
        })

        result = parse_magma_results(genes_out, annot, exclude_mhc=False)
        assert len(result) == 2
        assert result["gene_symbol"].notna().all()
        assert (result["gene_symbol"] == ["SAMD11", "OR4F5"]).all()
