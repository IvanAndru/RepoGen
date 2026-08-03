"""Tests for repogen.analysis.magma_pathway."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from statsmodels.stats.multitest import multipletests

from repogen.analysis.magma_pathway import (
    _compute_driving_genes,
    apply_fdr_correction,
    assemble_pathway_results,
    create_geneset_file,
    parse_magma_geneset_results,
    run_magma_geneset_analysis,
    run_pathway_analysis,
)
from repogen.config.schema import MagmaConfig, PathwayConfig, PipelineConfig, StudyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_gene_results() -> pd.DataFrame:
    """Synthetic MAGMA gene results (20 genes)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "gene_entrez_id": [str(i) for i in range(1, 21)],
        "gene_symbol": [f"GENE{i}" for i in range(1, 21)],
        "magma_z": rng.normal(0, 2, 20),
        "magma_p": rng.uniform(0, 1, 20),
        "n_snps": [50] * 20,
        "chr": [1] * 20,
        "start": list(range(1000, 21000, 1000)),
        "end": list(range(2000, 22000, 1000)),
    })


@pytest.fixture()
def sample_gene_sets() -> pd.DataFrame:
    """Synthetic PathwayRecord DataFrame (5 pathways)."""
    return pd.DataFrame({
        "pathway_id": ["GO:0001", "GO:0002", "KEGG_PATH1", "REACT_PATH1", "HALLMARK_P1"],
        "pathway_name": ["Pathway A", "Pathway B", "Pathway C", "Pathway D", "Pathway E"],
        "source_db": ["GO_BP", "GO_BP", "KEGG", "REACTOME", "HALLMARK"],
        "genes": [
            [f"GENE{i}" for i in range(1, 8)],
            [f"GENE{i}" for i in range(5, 16)],
            [f"GENE{i}" for i in range(1, 13)],
            [f"GENE{i}" for i in range(10, 21)],
            [f"GENE{i}" for i in range(1, 21)],
        ],
        "n_genes": [7, 11, 12, 11, 20],
    })


@pytest.fixture()
def sample_gsa_out(tmp_path: Path) -> Path:
    """Write a synthetic MAGMA .gsa.out file."""
    content = (
        "# MAGMA gene-set analysis\n"
        "# TOTAL_GENES = 20\n"
        "VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P\n"
        "GO:0001  COMPETITIVE  7  0.15  0.08  0.12  0.10\n"
        "GO:0002  COMPETITIVE  11  0.45  0.22  0.09  0.001\n"
        "KEGG_PATH1  COMPETITIVE  12  0.30  0.15  0.10  0.02\n"
        "REACT_PATH1  COMPETITIVE  11  0.05  0.03  0.11  0.35\n"
        "HALLMARK_P1  COMPETITIVE  20  0.50  0.25  0.08  0.0005\n"
    )
    path = tmp_path / "test.gsa.out"
    path.write_text(content)
    return path


def _make_config(
    tmp_path: Path,
    sources: list[str] | None = None,
    extra_gmt_files: list[Path] | None = None,
) -> PipelineConfig:
    """Build a minimal PipelineConfig for testing."""
    gwas_path = tmp_path / "test.gwas.gz"
    gwas_path.touch()
    pathway_kwargs: dict = {}
    if sources is not None:
        pathway_kwargs["sources"] = sources
    if extra_gmt_files is not None:
        pathway_kwargs["extra_gmt_files"] = extra_gmt_files
    return PipelineConfig(
        study=StudyConfig(name="test_study", gwas_input=gwas_path),
        pathway=PathwayConfig(**pathway_kwargs),
    )


# ---------------------------------------------------------------------------
# 8.1 Gene-Set File Creation (4 tests)
# ---------------------------------------------------------------------------


class TestCreateGenesetFile:

    def test_basic(
        self, tmp_path: Path, sample_gene_results: pd.DataFrame
    ) -> None:
        """3 pathways with different sizes - verify output file format."""
        gene_sets = pd.DataFrame({
            "pathway_id": ["SET_A", "SET_B", "SET_C"],
            "genes": [
                [f"GENE{i}" for i in range(1, 6)],
                [f"GENE{i}" for i in range(1, 11)],
                [f"GENE{i}" for i in range(1, 16)],
            ],
            "n_genes": [5, 10, 15],
            "pathway_name": ["A", "B", "C"],
            "source_db": ["GO_BP", "GO_BP", "KEGG"],
        })
        out = tmp_path / "test.geneset"
        path, filtered = create_geneset_file(
            gene_sets, sample_gene_results, out, min_set_size=3, max_set_size=500
        )
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            parts = line.split("\t")
            assert parts[0] in ("SET_A", "SET_B", "SET_C")
            assert len(parts) > 1
        assert "n_genes_mapped" in filtered.columns

    def test_filters_by_size(
        self, tmp_path: Path, sample_gene_results: pd.DataFrame
    ) -> None:
        """Pathways outside size range are excluded."""
        gene_sets = pd.DataFrame({
            "pathway_id": ["TINY", "GOOD10", "GOOD50", "HUGE"],
            "genes": [
                [f"GENE{i}" for i in range(1, 4)],
                [f"GENE{i}" for i in range(1, 11)],
                [f"GENE{i}" for i in range(1, 16)],
                [f"GENE{i}" for i in range(1, 21)] * 30,
            ],
            "n_genes": [3, 10, 15, 600],
            "pathway_name": ["T", "G10", "G50", "H"],
            "source_db": ["GO_BP"] * 4,
        })
        out = tmp_path / "test.geneset"
        _, filtered = create_geneset_file(
            gene_sets, sample_gene_results, out, min_set_size=10, max_set_size=500
        )
        assert len(filtered) == 2
        assert set(filtered["pathway_id"]) == {"GOOD10", "GOOD50"}

    def test_unmapped_genes(
        self, tmp_path: Path, sample_gene_results: pd.DataFrame
    ) -> None:
        """Pathway with partially unmapped genes - only mapped ones appear."""
        gene_sets = pd.DataFrame({
            "pathway_id": ["MIXED"],
            "genes": [
                [f"GENE{i}" for i in range(1, 9)] + [f"UNKNOWN{i}" for i in range(1, 8)]
            ],
            "n_genes": [15],
            "pathway_name": ["Mixed"],
            "source_db": ["GO_BP"],
        })
        out = tmp_path / "test.geneset"
        _, filtered = create_geneset_file(
            gene_sets, sample_gene_results, out, min_set_size=5, max_set_size=500
        )
        assert filtered.iloc[0]["n_genes_mapped"] == 8
        line = out.read_text().strip()
        entrez_ids = line.split("\t")[1:]
        assert len(entrez_ids) == 8

    def test_empty_after_mapping_raises(
        self, tmp_path: Path, sample_gene_results: pd.DataFrame
    ) -> None:
        """All pathways shrink below min_set_size - RuntimeError."""
        gene_sets = pd.DataFrame({
            "pathway_id": ["EMPTY1", "EMPTY2"],
            "genes": [
                ["NOTREAL1", "NOTREAL2"],
                ["NOTREAL3"],
            ],
            "n_genes": [2, 1],
            "pathway_name": ["E1", "E2"],
            "source_db": ["GO_BP", "GO_BP"],
        })
        out = tmp_path / "test.geneset"
        with pytest.raises(RuntimeError, match="empty or outside size range"):
            create_geneset_file(
                gene_sets, sample_gene_results, out, min_set_size=10, max_set_size=500
            )


# ---------------------------------------------------------------------------
# 8.2 MAGMA Subprocess Execution (3 tests)
# ---------------------------------------------------------------------------


class TestRunMagmaGenesetAnalysis:

    @staticmethod
    def _write_valid_geneset(path: Path) -> None:
        path.write_text("SET_A\t1234\t5678\t9012\nSET_B\t3456\t7890\n")

    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Verify correct command construction with separate tokens."""
        gsa_out = tmp_path / "test.gsa.out"
        gsa_out.write_text("VARIABLE TYPE NGENES BETA BETA_STD SE P\n")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        binary = Path("/usr/bin/magma")
        raw = tmp_path / "test.genes.raw"
        raw.touch()
        geneset = tmp_path / "test.geneset"
        self._write_valid_geneset(geneset)
        prefix = tmp_path / "test"

        result = run_magma_geneset_analysis(binary, raw, geneset, prefix)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == str(binary)
        assert "--gene-results" in cmd
        assert "--set-annot" in cmd
        assert "col=2,1" not in cmd
        set_annot_idx = cmd.index("--set-annot")
        assert cmd[set_annot_idx + 1] == str(geneset)
        assert cmd[set_annot_idx + 2] == "--out"
        assert result == gsa_out

    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Non-zero exit code raises RuntimeError with stderr."""
        geneset = tmp_path / "test.geneset"
        self._write_valid_geneset(geneset)
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="MAGMA error: bad input"
        )
        with pytest.raises(RuntimeError, match="MAGMA error: bad input"):
            run_magma_geneset_analysis(
                Path("/usr/bin/magma"),
                tmp_path / "test.genes.raw",
                geneset,
                tmp_path / "test",
            )

    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_failure_empty_stderr_includes_log_tail(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Empty stderr still produces actionable RuntimeError via MAGMA log."""
        geneset = tmp_path / "test.geneset"
        self._write_valid_geneset(geneset)
        magma_log = tmp_path / "test.log"
        magma_log.write_text(
            "reading gene results\nprocessing sets\n"
            "ERROR - processing input: no input variables to analyse\n"
        )
        mock_run.return_value = MagicMock(returncode=2, stdout="", stderr="")
        with pytest.raises(RuntimeError, match="no input variables to analyse"):
            run_magma_geneset_analysis(
                Path("/usr/bin/magma"),
                tmp_path / "test.genes.raw",
                geneset,
                tmp_path / "test",
            )

    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_missing_output_raises(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Success exit code but no .gsa.out - FileNotFoundError."""
        geneset = tmp_path / "test.geneset"
        self._write_valid_geneset(geneset)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with pytest.raises(FileNotFoundError, match="gsa.out"):
            run_magma_geneset_analysis(
                Path("/usr/bin/magma"),
                tmp_path / "test.genes.raw",
                geneset,
                tmp_path / "test",
            )

    def test_preflight_empty_file(self, tmp_path: Path) -> None:
        """Empty geneset file fails preflight before MAGMA is called."""
        geneset = tmp_path / "empty.geneset"
        geneset.write_text("")
        with pytest.raises(RuntimeError, match="empty or contains no valid sets"):
            run_magma_geneset_analysis(
                Path("/usr/bin/magma"),
                tmp_path / "test.genes.raw",
                geneset,
                tmp_path / "test",
            )

    def test_preflight_malformed_lines(self, tmp_path: Path) -> None:
        """Geneset with no tab separators fails preflight."""
        geneset = tmp_path / "bad.geneset"
        geneset.write_text("SET_A_NO_GENES\n")
        with pytest.raises(RuntimeError, match="malformed line"):
            run_magma_geneset_analysis(
                Path("/usr/bin/magma"),
                tmp_path / "test.genes.raw",
                geneset,
                tmp_path / "test",
            )

    def test_preflight_missing_file(self, tmp_path: Path) -> None:
        """Non-existent geneset file fails preflight."""
        with pytest.raises(RuntimeError, match="does not exist"):
            run_magma_geneset_analysis(
                Path("/usr/bin/magma"),
                tmp_path / "test.genes.raw",
                tmp_path / "nonexistent.geneset",
                tmp_path / "test",
            )


# ---------------------------------------------------------------------------
# 8.3 Result Parsing (3 tests)
# ---------------------------------------------------------------------------


class TestParseGenesetResults:

    def test_basic(self, sample_gsa_out: Path) -> None:
        """Parse 5 pathways from .gsa.out file."""
        df = parse_magma_geneset_results(sample_gsa_out)
        assert len(df) == 5
        assert set(df.columns) == {
            "pathway_id", "n_genes_tested", "beta", "beta_std_error", "p_value"
        }
        assert "GO:0001" in df["pathway_id"].values

    def test_empty_raises(self, tmp_path: Path) -> None:
        """Header-only .gsa.out file raises RuntimeError."""
        path = tmp_path / "empty.gsa.out"
        path.write_text(
            "# comment\n"
            "VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P\n"
        )
        with pytest.raises(RuntimeError, match="empty"):
            parse_magma_geneset_results(path)

    def test_column_types(self, sample_gsa_out: Path) -> None:
        """Verify correct dtypes after parsing."""
        df = parse_magma_geneset_results(sample_gsa_out)
        assert df["p_value"].dtype == np.float64
        assert df["n_genes_tested"].dtype in (np.int32, np.int64)
        assert df["beta"].dtype == np.float64

    def test_full_name_used_when_variable_truncated(self, tmp_path: Path) -> None:
        """Parser must use FULL_NAME as pathway_id when VARIABLE is truncated."""
        content = (
            "# MAGMA gene-set analysis\n"
            "# TOTAL_GENES = 20\n"
            "VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P  FULL_NAME\n"
            "GOBP_2_OXOGLUTARATE_METABOLI  COMPETITIVE  7  0.15  0.08  0.12  0.10  GOBP_2_OXOGLUTARATE_METABOLIC_PROCESS\n"
            "SHORT_SET  COMPETITIVE  12  0.30  0.15  0.10  0.02  SHORT_SET\n"
        )
        path = tmp_path / "trunc.gsa.out"
        path.write_text(content)
        df = parse_magma_geneset_results(path)
        assert "GOBP_2_OXOGLUTARATE_METABOLIC_PROCESS" in df["pathway_id"].values
        assert "SHORT_SET" in df["pathway_id"].values
        assert "GOBP_2_OXOGLUTARATE_METABOLI" not in df["pathway_id"].values

    def test_no_full_name_falls_back_to_variable(self, sample_gsa_out: Path) -> None:
        """When FULL_NAME column absent, VARIABLE is used as pathway_id."""
        df = parse_magma_geneset_results(sample_gsa_out)
        assert "GO:0001" in df["pathway_id"].values


# ---------------------------------------------------------------------------
# 8.4 FDR Correction (3 tests)
# ---------------------------------------------------------------------------


class TestApplyFDR:

    def test_basic(self) -> None:
        """FDR q-values are >= p-values and monotonic when sorted by p."""
        rng = np.random.default_rng(99)
        df = pd.DataFrame({"p_value": rng.uniform(0.001, 0.5, 10)})
        result = apply_fdr_correction(df)
        assert "fdr_q" in result.columns
        assert (result["fdr_q"] >= result["p_value"]).all()
        sorted_result = result.sort_values("p_value")
        q_sorted = sorted_result["fdr_q"].values
        assert all(q_sorted[i] <= q_sorted[i + 1] for i in range(len(q_sorted) - 1))

    def test_matches_statsmodels(self) -> None:
        """FDR values match direct statsmodels call."""
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 0.001, 0.02, 0.03, 0.9])
        df = pd.DataFrame({"p_value": pvals})
        result = apply_fdr_correction(df, method="fdr_bh")
        _, expected_q, _, _ = multipletests(pvals, method="fdr_bh")
        np.testing.assert_array_almost_equal(result["fdr_q"].values, expected_q)

    def test_all_ones(self) -> None:
        """All p-values = 1.0 => all q-values = 1.0."""
        df = pd.DataFrame({"p_value": [1.0] * 5})
        result = apply_fdr_correction(df)
        assert (result["fdr_q"] == 1.0).all()


# ---------------------------------------------------------------------------
# 8.5 Driving Gene Extraction (4 tests)
# ---------------------------------------------------------------------------


class TestDrivingGenes:

    def _build_lookup(self) -> dict:
        rng = np.random.default_rng(42)
        z_scores = rng.normal(0, 2, 10)
        p_values = [0.001, 0.01, 0.04, 0.1, 0.2, 0.5, 0.03, 0.8, 0.02, 0.9]
        lookup = {}
        for i in range(1, 11):
            lookup[f"GENE{i}"] = {
                "gene_entrez_id": str(i),
                "z_score": z_scores[i - 1],
                "p_value": p_values[i - 1],
            }
        return lookup

    def test_basic(self) -> None:
        """3 genes with p < 0.05 should be driving genes."""
        lookup = self._build_lookup()
        genes = [f"GENE{i}" for i in range(1, 11)]
        driving, all_genes = _compute_driving_genes(genes, lookup, 0.05)
        sig_count = sum(1 for g in lookup.values() if g["p_value"] < 0.05)
        assert len(driving) == sig_count
        assert len(all_genes) == 10

    def test_none_significant(self) -> None:
        """All gene p-values > threshold => empty driving list."""
        lookup = {
            f"GENE{i}": {"gene_entrez_id": str(i), "z_score": 0.5, "p_value": 0.5}
            for i in range(1, 6)
        }
        genes = [f"GENE{i}" for i in range(1, 6)]
        driving, all_genes = _compute_driving_genes(genes, lookup, 0.05)
        assert len(driving) == 0
        assert len(all_genes) == 5

    def test_custom_threshold(self) -> None:
        """Threshold of 0.01 - only genes with p < 0.01 are driving."""
        lookup = self._build_lookup()
        genes = [f"GENE{i}" for i in range(1, 11)]
        driving, _ = _compute_driving_genes(genes, lookup, 0.01)
        for g in driving:
            assert g["p_value"] < 0.01
        sig_count = sum(1 for g in lookup.values() if g["p_value"] < 0.01)
        assert len(driving) == sig_count

    def test_sorting(self) -> None:
        """Both driving and all_genes are sorted by z_score descending."""
        lookup = self._build_lookup()
        genes = [f"GENE{i}" for i in range(1, 11)]
        driving, all_genes = _compute_driving_genes(genes, lookup, 0.05)
        for lst in (driving, all_genes):
            if len(lst) > 1:
                z_scores = [g["z_score"] for g in lst]
                assert z_scores == sorted(z_scores, reverse=True)


# ---------------------------------------------------------------------------
# 8.6 End-to-End Orchestration (4 tests)
# ---------------------------------------------------------------------------


def _write_gsa_out(path: Path, pathways: list[tuple]) -> None:
    """Helper to write a synthetic .gsa.out file.

    pathways: list of (name, ngenes, beta, beta_std, se, p).
    """
    lines = ["# MAGMA output\n", "VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P\n"]
    for name, ng, beta, beta_std, se, p in pathways:
        lines.append(f"{name}  COMPETITIVE  {ng}  {beta}  {beta_std}  {se}  {p}\n")
    path.write_text("".join(lines))


class TestRunPathwayAnalysis:

    @patch("repogen.analysis.magma_pathway.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_full(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        tmp_path: Path,
        sample_gene_results: pd.DataFrame,
        sample_gene_sets: pd.DataFrame,
    ) -> None:
        """Full pipeline with mocked MAGMA produces Parquet + JSON."""
        mock_detect.return_value = Path("/usr/bin/magma")

        config = _make_config(tmp_path)
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        magma_out_dir = out_dir / "test_study" / "magma"
        magma_out_dir.mkdir(parents=True)
        raw_file = magma_out_dir / "fake.genes.raw"
        raw_file.touch()

        gsa_path = magma_out_dir / "test_study.gsa.out"
        _write_gsa_out(gsa_path, [
            ("GO:0002", 11, 0.45, 0.22, 0.09, 0.001),
            ("KEGG_PATH1", 12, 0.30, 0.15, 0.10, 0.02),
            ("REACT_PATH1", 11, 0.05, 0.03, 0.11, 0.35),
            ("HALLMARK_P1", 20, 0.50, 0.25, 0.08, 0.0005),
        ])

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        results = run_pathway_analysis(
            config=config,
            gene_results_raw=raw_file,
            gene_results_df=sample_gene_results,
            gene_sets_df=sample_gene_sets,
            output_dir=out_dir,
        )

        parquet_path = out_dir / "test_study_pathway_results.parquet"
        json_path = out_dir / "test_study_pathway_results_meta.json"
        assert parquet_path.exists()
        assert json_path.exists()

        assert "pathway_id" in results.columns
        assert "driving_genes" in results.columns
        assert "fdr_q" in results.columns
        assert "significant" in results.columns

        has_drivers = results["n_driving_genes"].sum() > 0
        assert has_drivers or len(results) > 0

    @patch("repogen.analysis.magma_pathway.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_with_extra_gmt(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        tmp_path: Path,
        sample_gene_results: pd.DataFrame,
        sample_gene_sets: pd.DataFrame,
    ) -> None:
        """Extra GMT files add pathways to the analysis."""
        mock_detect.return_value = Path("/usr/bin/magma")

        gmt_path = tmp_path / "extra.gmt"
        gmt_path.write_text(
            "EXTRA_SET1\thttp://example.com\t"
            + "\t".join(f"GENE{i}" for i in range(1, 16))
            + "\n"
        )

        config = _make_config(tmp_path, extra_gmt_files=[gmt_path])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        magma_out_dir = out_dir / "test_study" / "magma"
        magma_out_dir.mkdir(parents=True)
        raw_file = magma_out_dir / "fake.genes.raw"
        raw_file.touch()

        gsa_path = magma_out_dir / "test_study.gsa.out"
        _write_gsa_out(gsa_path, [
            ("GO:0002", 11, 0.45, 0.22, 0.09, 0.001),
            ("KEGG_PATH1", 12, 0.30, 0.15, 0.10, 0.02),
            ("HALLMARK_P1", 20, 0.50, 0.25, 0.08, 0.0005),
            ("EXTRA_SET1", 15, 0.20, 0.10, 0.11, 0.05),
        ])

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        results = run_pathway_analysis(
            config=config,
            gene_results_raw=raw_file,
            gene_results_df=sample_gene_results,
            gene_sets_df=sample_gene_sets,
            output_dir=out_dir,
        )

        assert "EXTRA_SET1" in results["pathway_id"].values

    @patch("repogen.analysis.magma_pathway.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_output_metadata(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        tmp_path: Path,
        sample_gene_results: pd.DataFrame,
        sample_gene_sets: pd.DataFrame,
    ) -> None:
        """JSON metadata contains all required fields."""
        mock_detect.return_value = Path("/usr/bin/magma")
        config = _make_config(tmp_path)
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        magma_out_dir = out_dir / "test_study" / "magma"
        magma_out_dir.mkdir(parents=True)
        raw_file = magma_out_dir / "fake.genes.raw"
        raw_file.touch()

        gsa_path = magma_out_dir / "test_study.gsa.out"
        _write_gsa_out(gsa_path, [
            ("GO:0002", 11, 0.45, 0.22, 0.09, 0.001),
            ("HALLMARK_P1", 20, 0.50, 0.25, 0.08, 0.0005),
        ])

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        run_pathway_analysis(
            config=config,
            gene_results_raw=raw_file,
            gene_results_df=sample_gene_results,
            gene_sets_df=sample_gene_sets,
            output_dir=out_dir,
        )

        json_path = out_dir / "test_study_pathway_results_meta.json"
        with open(json_path) as f:
            meta = json.load(f)

        required_keys = {
            "study_name", "analysis", "test_mode", "gene_results_file",
            "n_gene_sets_tested", "n_significant_fdr05", "fdr_method",
            "fdr_threshold", "driver_gene_p_threshold", "gene_set_sources",
            "extra_gmt_files", "annotation_mode", "timestamp",
        }
        assert required_keys <= set(meta.keys())
        assert meta["analysis"] == "magma_pathway"
        assert meta["test_mode"] == "competitive"
        assert meta["fdr_method"] == "fdr_bh"

    @patch("repogen.analysis.magma_pathway.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_source_filter(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        tmp_path: Path,
        sample_gene_results: pd.DataFrame,
        sample_gene_sets: pd.DataFrame,
    ) -> None:
        """Source filter restricts to specified collections only."""
        mock_detect.return_value = Path("/usr/bin/magma")
        config = _make_config(tmp_path, sources=["KEGG"])
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        magma_out_dir = out_dir / "test_study" / "magma"
        magma_out_dir.mkdir(parents=True)
        raw_file = magma_out_dir / "fake.genes.raw"
        raw_file.touch()

        gsa_path = magma_out_dir / "test_study.gsa.out"
        _write_gsa_out(gsa_path, [
            ("KEGG_PATH1", 12, 0.30, 0.15, 0.10, 0.02),
        ])

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        results = run_pathway_analysis(
            config=config,
            gene_results_raw=raw_file,
            gene_results_df=sample_gene_results,
            gene_sets_df=sample_gene_sets,
            output_dir=out_dir,
        )

        assert all(results["source_db"] == "KEGG")


# ---------------------------------------------------------------------------
# 8.7 Config Validation (3 tests)
# ---------------------------------------------------------------------------


class TestPathwayConfig:

    def test_defaults(self) -> None:
        """Verify default values for all PathwayConfig fields."""
        pc = PathwayConfig()
        assert pc.fdr_method == "fdr_bh"
        assert pc.fdr_threshold == 0.05
        assert pc.min_set_size == 10
        assert pc.max_set_size == 500
        assert pc.sources is None
        assert pc.driver_gene_p_threshold == 0.05
        assert pc.extra_gmt_files == []

    def test_invalid_fdr_method(self) -> None:
        """Invalid fdr_method raises ValidationError."""
        with pytest.raises(ValidationError, match="fdr_method"):
            PathwayConfig(fdr_method="invalid")

    def test_extra_gmt_files_default_empty(self) -> None:
        """extra_gmt_files defaults to empty list."""
        pc = PathwayConfig()
        assert isinstance(pc.extra_gmt_files, list)
        assert len(pc.extra_gmt_files) == 0


# ---------------------------------------------------------------------------
# 8.8 Integration Test (1 test, skipped without MAGMA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("magma") is None,
    reason="MAGMA not installed",
)
class TestIntegrationRealMagma:

    def test_real_magma(
        self,
        tmp_path: Path,
        sample_gene_results: pd.DataFrame,
        sample_gene_sets: pd.DataFrame,
    ) -> None:
        """End-to-end with real MAGMA binary (skipped if not available)."""
        config = _make_config(tmp_path)
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        results = run_pathway_analysis(
            config=config,
            gene_results_raw=raw_file,
            gene_results_df=sample_gene_results,
            gene_sets_df=sample_gene_sets,
            output_dir=out_dir,
        )
        assert len(results) > 0
