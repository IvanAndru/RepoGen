"""Tests for repogen.data.gene_sets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from repogen.data.gene_sets import (
    _build_url,
    _infer_category,
    _underscores_to_title,
    create_magma_pathway_file,
    load_gene_sets,
    parse_gmt_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_gmt(path: Path, entries: list[tuple[str, str, list[str]]]) -> None:
    """Write a GMT file from (name, description, genes) tuples."""
    lines = []
    for name, desc, genes in entries:
        lines.append(f"{name}\t{desc}\t" + "\t".join(genes))
    path.write_text("\n".join(lines) + "\n")


def _standard_gmt_entries() -> list[tuple[str, str, list[str]]]:
    """Return a mix of standard MSigDB entries for testing."""
    genes_20 = [f"GENE{i}" for i in range(1, 21)]
    genes_5 = [f"GENE{i}" for i in range(1, 6)]
    genes_3 = [f"GENE{i}" for i in range(1, 4)]
    return [
        ("GOBP_SYNAPTIC_SIGNALING", "GO:0099536", genes_20),
        ("KEGG_MAPK_SIGNALING_PATHWAY", "https://kegg.jp/hsa04010", genes_20),
        ("REACTOME_SIGNALING_BY_GPCRS", "R-HSA-372790", genes_20),
        ("WP_ALZHEIMERS_DISEASE", "NA", genes_20),
        ("HALLMARK_TNFA_SIGNALING_VIA_NFKB", "na", genes_20),
        ("BIOCARTA_AKT_PATHWAY", "NA", genes_5),
        ("PID_AURORA_B_PATHWAY", "NA", genes_3),
        ("HP_SEIZURE", "NA", genes_20),
        ("CUSTOM_SET_XYZ", "Some custom set", genes_20),
    ]


# ---------------------------------------------------------------------------
# Tests: parse_gmt_metadata
# ---------------------------------------------------------------------------


class TestParseGmtMetadata:
    """Tests for MSigDB name parsing."""

    def test_gobp_prefix(self) -> None:
        m = parse_gmt_metadata("GOBP_SYNAPTIC_SIGNALING")
        assert m["source_db"] == "GO_BP"
        assert m["name"] == "Synaptic Signaling"
        assert m["category"] == "biological_process"

    def test_kegg_prefix(self) -> None:
        m = parse_gmt_metadata("KEGG_MAPK_SIGNALING_PATHWAY")
        assert m["source_db"] == "KEGG"
        assert "Mapk" in m["name"]
        assert m["category"] == "canonical_pathways"

    def test_reactome_prefix(self) -> None:
        m = parse_gmt_metadata("REACTOME_SIGNALING_BY_GPCRS")
        assert m["source_db"] == "REACTOME"
        assert m["category"] == "canonical_pathways"

    def test_wp_prefix(self) -> None:
        m = parse_gmt_metadata("WP_ALZHEIMERS_DISEASE")
        assert m["source_db"] == "WIKIPATHWAYS"

    def test_hallmark_prefix(self) -> None:
        m = parse_gmt_metadata("HALLMARK_TNFA_SIGNALING_VIA_NFKB")
        assert m["source_db"] == "HALLMARK"
        assert m["category"] == "hallmark"
        assert "Tnfa" in m["name"]

    def test_biocarta_prefix(self) -> None:
        m = parse_gmt_metadata("BIOCARTA_AKT_PATHWAY")
        assert m["source_db"] == "BIOCARTA"

    def test_pid_prefix(self) -> None:
        m = parse_gmt_metadata("PID_AURORA_B_PATHWAY")
        assert m["source_db"] == "PID"

    def test_hpo_prefix(self) -> None:
        m = parse_gmt_metadata("HP_SEIZURE")
        assert m["source_db"] == "HPO"
        assert m["category"] == "phenotype"

    def test_go_colon_prefix(self) -> None:
        m = parse_gmt_metadata("GO:0099536")
        assert m["source_db"] == "GO"

    def test_unknown_prefix(self) -> None:
        m = parse_gmt_metadata("CUSTOM_SET_NAME")
        assert m["source_db"] == "OTHER"

    def test_unknown_with_reactome_in_name(self) -> None:
        m = parse_gmt_metadata("SOME_REACTOME_THING")
        assert m["source_db"] == "REACTOME"


# ---------------------------------------------------------------------------
# Tests: _underscores_to_title
# ---------------------------------------------------------------------------


class TestUnderscoresToTitle:
    """Tests for the name formatter."""

    def test_standard_conversion(self) -> None:
        assert _underscores_to_title("MAPK_SIGNALING_PATHWAY") == "Mapk Signaling Pathway"

    def test_short_uppercase_preserved(self) -> None:
        assert _underscores_to_title("TNF_VIA_NF") == "TNF VIA NF"

    def test_mixed_length_words(self) -> None:
        result = _underscores_to_title("ALZHEIMERS_DISEASE")
        assert result == "Alzheimers Disease"

    def test_single_word(self) -> None:
        assert _underscores_to_title("SEIZURE") == "Seizure"


# ---------------------------------------------------------------------------
# Tests: _infer_category
# ---------------------------------------------------------------------------


class TestInferCategory:
    """Tests for category mapping."""

    def test_known_sources(self) -> None:
        assert _infer_category("GO_BP") == "biological_process"
        assert _infer_category("GO_CC") == "cellular_component"
        assert _infer_category("GO_MF") == "molecular_function"
        assert _infer_category("KEGG") == "canonical_pathways"
        assert _infer_category("REACTOME") == "canonical_pathways"
        assert _infer_category("HALLMARK") == "hallmark"
        assert _infer_category("HPO") == "phenotype"

    def test_unknown_source(self) -> None:
        assert _infer_category("CUSTOM") is None


# ---------------------------------------------------------------------------
# Tests: _build_url
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for URL construction."""

    def test_go_url(self) -> None:
        url = _build_url("GO_BP", "GOBP_SYNAPTIC_SIGNALING", "GO:0099536")
        assert url is not None
        assert "GO:0099536" in url

    def test_no_url_for_unknown(self) -> None:
        url = _build_url("KEGG", "KEGG_SOMETHING", "NA")
        assert url is None


# ---------------------------------------------------------------------------
# Tests: load_gene_sets
# ---------------------------------------------------------------------------


class TestLoadGeneSets:
    """Tests for the main GMT loading function."""

    def test_basic_loading(self, tmp_path: Path) -> None:
        gmt = tmp_path / "test.gmt"
        _write_gmt(gmt, _standard_gmt_entries())
        df = load_gene_sets([gmt], min_size=1, max_size=1000)
        assert len(df) > 0
        assert "pathway_id" in df.columns
        assert "source_db" in df.columns
        assert "genes" in df.columns
        assert "n_genes" in df.columns

    def test_size_filter_min(self, tmp_path: Path) -> None:
        gmt = tmp_path / "test.gmt"
        _write_gmt(gmt, _standard_gmt_entries())
        df = load_gene_sets([gmt], min_size=10, max_size=1000)
        assert all(df["n_genes"] >= 10)
        assert len(df) < len(_standard_gmt_entries())

    def test_size_filter_max(self, tmp_path: Path) -> None:
        gmt = tmp_path / "test.gmt"
        large_genes = [f"G{i}" for i in range(600)]
        _write_gmt(gmt, [("BIG_SET", "NA", large_genes)])
        df = load_gene_sets([gmt], min_size=1, max_size=500)
        assert len(df) == 0

    def test_source_filter(self, tmp_path: Path) -> None:
        gmt = tmp_path / "test.gmt"
        _write_gmt(gmt, _standard_gmt_entries())
        df = load_gene_sets(
            [gmt], min_size=1, max_size=1000, sources=["GO_BP", "KEGG"],
        )
        assert set(df["source_db"].unique()).issubset({"GO_BP", "KEGG"})

    def test_deduplication(self, tmp_path: Path) -> None:
        genes = [f"GENE{i}" for i in range(20)]
        entries = [
            ("GOBP_TEST_PATHWAY", "NA", genes),
            ("GOBP_TEST_PATHWAY", "duplicate", genes),
        ]
        gmt = tmp_path / "test.gmt"
        _write_gmt(gmt, entries)
        df = load_gene_sets([gmt], min_size=1, max_size=1000)
        assert len(df[df["pathway_id"] == "GOBP_TEST_PATHWAY"]) == 1

    def test_multiple_gmt_files(self, tmp_path: Path) -> None:
        genes = [f"GENE{i}" for i in range(15)]
        gmt1 = tmp_path / "a.gmt"
        gmt2 = tmp_path / "b.gmt"
        _write_gmt(gmt1, [("KEGG_SET_A", "NA", genes)])
        _write_gmt(gmt2, [("REACTOME_SET_B", "NA", genes)])
        df = load_gene_sets([gmt1, gmt2], min_size=1, max_size=1000)
        assert len(df) == 2

    def test_empty_gmt(self, tmp_path: Path) -> None:
        gmt = tmp_path / "empty.gmt"
        gmt.write_text("")
        df = load_gene_sets([gmt])
        assert len(df) == 0
        assert "pathway_id" in df.columns

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        gmt = tmp_path / "bad.gmt"
        gmt.write_text("SHORT_LINE\n\nGOBP_GOOD_SET\tNA\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10\tG11\n")
        df = load_gene_sets([gmt], min_size=1, max_size=1000)
        assert len(df) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gene_sets([tmp_path / "nonexistent.gmt"])

    def test_genes_column_is_list(self, tmp_path: Path) -> None:
        genes = [f"GENE{i}" for i in range(15)]
        gmt = tmp_path / "test.gmt"
        _write_gmt(gmt, [("KEGG_TEST", "NA", genes)])
        df = load_gene_sets([gmt], min_size=1, max_size=1000)
        assert isinstance(df["genes"].iloc[0], list)
        assert len(df["genes"].iloc[0]) == 15


# ---------------------------------------------------------------------------
# Tests: create_magma_pathway_file
# ---------------------------------------------------------------------------


class TestCreateMagmaPathwayFile:
    """Tests for MAGMA pathway file generation."""

    def test_basic_output(self, tmp_path: Path) -> None:
        gene_sets = pd.DataFrame({
            "pathway_id": ["KEGG_TEST"],
            "genes": [["BRCA1", "TP53", "EGFR"]],
            "n_genes": [3],
        })
        gene_annot = pd.DataFrame({
            "gene_symbol": ["BRCA1", "TP53", "EGFR"],
            "gene_ensembl_id": ["ENSG001", "ENSG002", "ENSG003"],
        })
        out = tmp_path / "magma.sets"
        result = create_magma_pathway_file(gene_sets, gene_annot, out)
        content = result.read_text()
        assert "KEGG_TEST" in content
        assert "ENSG001" in content

    def test_unmapped_genes_excluded(self, tmp_path: Path) -> None:
        gene_sets = pd.DataFrame({
            "pathway_id": ["SET1"],
            "genes": [["BRCA1", "UNKNOWN_GENE", "TP53"]],
            "n_genes": [3],
        })
        gene_annot = pd.DataFrame({
            "gene_symbol": ["BRCA1", "TP53"],
            "gene_ensembl_id": ["ENSG001", "ENSG002"],
        })
        out = tmp_path / "magma.sets"
        result = create_magma_pathway_file(gene_sets, gene_annot, out)
        content = result.read_text()
        assert "SET1" in content
        assert "UNKNOWN_GENE" not in content

    def test_pathway_skipped_if_fewer_than_2_mapped(self, tmp_path: Path) -> None:
        gene_sets = pd.DataFrame({
            "pathway_id": ["SMALL_SET"],
            "genes": [["ONLY_ONE"]],
            "n_genes": [1],
        })
        gene_annot = pd.DataFrame({
            "gene_symbol": ["ONLY_ONE"],
            "gene_ensembl_id": ["ENSG999"],
        })
        out = tmp_path / "magma.sets"
        result = create_magma_pathway_file(gene_sets, gene_annot, out)
        content = result.read_text().strip()
        assert "SMALL_SET" not in content

    def test_multiple_pathways(self, tmp_path: Path) -> None:
        gene_sets = pd.DataFrame({
            "pathway_id": ["SET_A", "SET_B"],
            "genes": [["G1", "G2"], ["G2", "G3"]],
            "n_genes": [2, 2],
        })
        gene_annot = pd.DataFrame({
            "gene_symbol": ["G1", "G2", "G3"],
            "gene_ensembl_id": ["E1", "E2", "E3"],
        })
        out = tmp_path / "magma.sets"
        result = create_magma_pathway_file(gene_sets, gene_annot, out)
        content = result.read_text()
        assert "SET_A" in content
        assert "SET_B" in content
