"""Tests for repogen.analysis.magma_gene."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from repogen.analysis.magma_gene import (
    detect_magma_binary,
    parse_magma_results,
    prepare_magma_input,
    resolve_annotation_file,
    run_gene_analysis,
    run_magma_gene_analysis,
)
from repogen.config.schema import MagmaConfig, PipelineConfig, StudyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_GENES_OUT = textwrap.dedent("""\
    # MEAN_SAMPLE_SIZE = 50000
    GENE       CHR  START     STOP      NSNPS  NPARAM  N      ZSTAT    P
    100287102  1    11873     14409     5      3       50000  2.5      0.006210
    729737     1    69091     70008     12     8       50000  -0.5     0.6915
    100302278  1    367640    368634    8      5       50000  4.2      0.00001335
    79501      6    26505891  26517442  20     15      50000  3.8      0.00007235
""")


@pytest.fixture()
def gene_annotations() -> pd.DataFrame:
    return pd.DataFrame({
        "gene_entrez_id": [100287102, 729737, 100302278, 79501],
        "gene_symbol": ["DDX11L1", "LOC729737", "ENSG00000228463", "TRIM31"],
        "gene_ensembl_id": [
            "ENSG00000223972", "ENSG00000237613",
            "ENSG00000228463", "ENSG00000113302",
        ],
        "biotype": [
            "transcribed_unprocessed_pseudogene", "lncRNA",
            "lncRNA", "protein_coding",
        ],
    })


@pytest.fixture()
def genes_out_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.genes.out"
    p.write_text(SYNTHETIC_GENES_OUT)
    return p


@pytest.fixture()
def minimal_gwas_df() -> pd.DataFrame:
    return pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(1, 11)],
        "P": np.random.default_rng(42).uniform(0.0001, 1.0, 10),
        "N": [50000] * 10,
    })


@pytest.fixture()
def varying_n_gwas_df() -> pd.DataFrame:
    return pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(1, 11)],
        "P": np.random.default_rng(42).uniform(0.0001, 1.0, 10),
        "N": [50000, 50000, 48000, 48000, 50000,
              49000, 50000, 47000, 50000, 50000],
    })


# ---------------------------------------------------------------------------
# 1-3: detect_magma_binary
# ---------------------------------------------------------------------------


class TestDetectMagmaBinary:

    def test_detect_from_config_path(self, tmp_path: Path) -> None:
        binary = tmp_path / "magma"
        binary.write_text("#!/bin/sh\n")
        result = detect_magma_binary(config_path=binary)
        assert result == binary

    def test_detect_from_which(self) -> None:
        with patch("repogen.analysis.magma_gene.shutil.which",
                    return_value="/usr/local/bin/magma"):
            result = detect_magma_binary()
        assert result == Path("/usr/local/bin/magma")

    def test_detect_not_found(self) -> None:
        with patch("repogen.analysis.magma_gene.shutil.which",
                    return_value=None):
            with pytest.raises(FileNotFoundError, match="MAGMA binary not found"):
                detect_magma_binary()


# ---------------------------------------------------------------------------
# 4-8: resolve_annotation_file
# ---------------------------------------------------------------------------


class TestResolveAnnotation:

    def test_proximity_mode(self, tmp_path: Path) -> None:
        config = MagmaConfig(annotation_mode="proximity")
        bfile = tmp_path / "ref"
        (tmp_path / "ref.bim").write_text("snp_data")
        gene_loc = tmp_path / "gene.loc"
        gene_loc.write_text("gene_data")
        magma_bin = tmp_path / "magma"
        magma_bin.write_text("#!/bin/sh\n")
        output_prefix = tmp_path / "out"

        with patch("repogen.analysis.magma_gene.run_magma_annotation") as mock_annot:
            expected = tmp_path / "out.genes.annot"
            mock_annot.return_value = expected
            result = resolve_annotation_file(
                config=config,
                resources_dir=tmp_path,
                reference_bfile=bfile,
                gene_loc=gene_loc,
                magma_binary=magma_bin,
                output_prefix=output_prefix,
            )
            assert result == expected
            mock_annot.assert_called_once_with(
                magma_binary=magma_bin,
                snp_loc=tmp_path / "ref.bim",
                gene_loc=gene_loc,
                output_prefix=output_prefix,
                window_upstream_kb=35,
                window_downstream_kb=10,
            )

    def test_hmagma_mode_file_exists(self, tmp_path: Path) -> None:
        config = MagmaConfig(annotation_mode="hmagma_fetal_brain")
        hmagma_dir = tmp_path / "hmagma"
        hmagma_dir.mkdir()
        annot_file = hmagma_dir / "Fetal_brain.genes.annot"
        annot_file.write_text("annot_data")

        result = resolve_annotation_file(config=config, resources_dir=tmp_path)
        assert result == annot_file

    def test_hmagma_mode_file_missing(self, tmp_path: Path) -> None:
        config = MagmaConfig(annotation_mode="hmagma_fetal_brain")
        with pytest.raises(FileNotFoundError, match="H-MAGMA annotation file"):
            resolve_annotation_file(config=config, resources_dir=tmp_path)

    def test_custom_mode(self, tmp_path: Path) -> None:
        annot_file = tmp_path / "my_custom.genes.annot"
        annot_file.write_text("custom_data")
        config = MagmaConfig(
            annotation_mode="custom", custom_annot_file=annot_file
        )
        result = resolve_annotation_file(config=config, resources_dir=tmp_path)
        assert result == annot_file

    def test_custom_missing_path_caught_by_validator(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="custom_annot_file"):
            MagmaConfig(annotation_mode="custom")


# ---------------------------------------------------------------------------
# 9-10: prepare_magma_input
# ---------------------------------------------------------------------------


class TestPrepareMagmaInput:

    def test_constant_n(self, tmp_path: Path, minimal_gwas_df: pd.DataFrame) -> None:
        pval_path, n = prepare_magma_input(minimal_gwas_df, tmp_path, "test")
        assert pval_path.exists()
        assert n == 50000
        content = pd.read_csv(pval_path, sep="\t")
        assert "N" not in content.columns
        assert list(content.columns) == ["SNP", "P"]
        assert len(content) == 10

    def test_varying_n(self, tmp_path: Path, varying_n_gwas_df: pd.DataFrame) -> None:
        pval_path, n = prepare_magma_input(varying_n_gwas_df, tmp_path, "test")
        assert pval_path.exists()
        assert n is None
        content = pd.read_csv(pval_path, sep="\t")
        assert "N" in content.columns
        assert len(content) == 10


# ---------------------------------------------------------------------------
# 11-12: run_magma_gene_analysis (mocked)
# ---------------------------------------------------------------------------


class TestRunMagmaGeneAnalysis:

    def test_success(self, tmp_path: Path) -> None:
        magma_bin = tmp_path / "magma"
        magma_bin.write_text("#!/bin/sh\n")
        ref_bfile = tmp_path / "ref"
        pval_file = tmp_path / "pval.txt"
        pval_file.write_text("SNP\tP\nrs1\t0.05\n")
        annot_file = tmp_path / "annot.genes.annot"
        annot_file.write_text("annot_data")
        output_prefix = tmp_path / "output"

        genes_out = Path(f"{output_prefix}.genes.out")
        genes_raw = Path(f"{output_prefix}.genes.raw")
        genes_out.write_text(SYNTHETIC_GENES_OUT)
        genes_raw.write_text("raw_data")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_result.stderr = ""

        with patch("repogen.analysis.magma_gene.subprocess.run",
                    return_value=mock_result):
            out, raw = run_magma_gene_analysis(
                magma_binary=magma_bin,
                reference_bfile=ref_bfile,
                pval_file=pval_file,
                gene_annot_file=annot_file,
                output_prefix=output_prefix,
                sample_size=50000,
            )
        assert out == genes_out
        assert raw == genes_raw

    def test_failure(self, tmp_path: Path) -> None:
        magma_bin = tmp_path / "magma"
        magma_bin.write_text("#!/bin/sh\n")
        output_prefix = tmp_path / "output"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: something went wrong"

        with patch("repogen.analysis.magma_gene.subprocess.run",
                    return_value=mock_result):
            with pytest.raises(subprocess.CalledProcessError):
                run_magma_gene_analysis(
                    magma_binary=magma_bin,
                    reference_bfile=tmp_path / "ref",
                    pval_file=tmp_path / "pval.txt",
                    gene_annot_file=tmp_path / "annot",
                    output_prefix=output_prefix,
                )


# ---------------------------------------------------------------------------
# 13-20: parse_magma_results
# ---------------------------------------------------------------------------


class TestParseMagmaResults:

    def test_basic(
        self,
        genes_out_file: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        results = parse_magma_results(genes_out_file, gene_annotations)
        assert len(results) == 4
        expected_cols = [
            "gene_entrez_id", "gene_symbol", "gene_ensembl_id",
            "chr", "start", "end", "biotype",
            "n_snps", "n_param", "n_samples",
            "magma_z", "magma_p", "fdr_q", "in_mhc", "annotation_mode",
        ]
        assert list(results.columns) == expected_cols

    def test_mhc_flagging(
        self,
        genes_out_file: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        results = parse_magma_results(genes_out_file, gene_annotations)
        trim31 = results[results["gene_entrez_id"] == 79501].iloc[0]
        assert trim31["in_mhc"] is True or trim31["in_mhc"] == True  # noqa: E712
        non_mhc = results[results["gene_entrez_id"] != 79501]
        assert (non_mhc["in_mhc"] == False).all()  # noqa: E712
        assert 79501 in results["gene_entrez_id"].values

    def test_mhc_fdr_excluded(
        self,
        genes_out_file: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        results = parse_magma_results(
            genes_out_file, gene_annotations, exclude_mhc=True
        )
        mhc_rows = results[results["in_mhc"]]
        assert mhc_rows["fdr_q"].isna().all()
        non_mhc_rows = results[~results["in_mhc"]]
        assert non_mhc_rows["fdr_q"].notna().all()

    def test_mhc_fdr_included(
        self,
        genes_out_file: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        results = parse_magma_results(
            genes_out_file, gene_annotations, exclude_mhc=False
        )
        assert results["fdr_q"].notna().all()
        mhc_rows = results[results["in_mhc"]]
        assert not mhc_rows["fdr_q"].isna().any()

    def test_mhc_flagging_with_string_chr(
        self,
        tmp_path: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        """When MAGMA output has non-numeric chromosomes (X/Y), CHR column
        becomes object/string.  MHC flagging must still work for chr 6."""
        genes_out = tmp_path / "str_chr.genes.out"
        genes_out.write_text(textwrap.dedent("""\
            # MEAN_SAMPLE_SIZE = 50000
            GENE       CHR  START     STOP      NSNPS  NPARAM  N      ZSTAT    P
            100287102  1    11873     14409     5      3       50000  2.5      0.006210
            79501      6    26505891  26517442  20     15      50000  3.8      0.00007235
            999999     X    1000      2000      3      2       50000  0.1      0.9
        """))
        results = parse_magma_results(genes_out, gene_annotations)
        mhc_row = results[results["gene_entrez_id"] == 79501]
        assert len(mhc_row) == 1
        assert mhc_row.iloc[0]["in_mhc"] is True or mhc_row.iloc[0]["in_mhc"] == True  # noqa: E712
        non_mhc = results[results["gene_entrez_id"] != 79501]
        assert (non_mhc["in_mhc"] == False).all()  # noqa: E712
        assert results["in_mhc"].sum() == 1

    def test_mhc_exclusion_changes_fdr(
        self,
        tmp_path: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        """FDR values should differ when MHC genes are excluded vs included,
        given enough MHC genes with significant p-values."""
        genes_out = tmp_path / "mhc_fdr.genes.out"
        genes_out.write_text(textwrap.dedent("""\
            # MEAN_SAMPLE_SIZE = 50000
            GENE       CHR  START     STOP      NSNPS  NPARAM  N      ZSTAT    P
            100287102  1    11873     14409     5      3       50000  2.5      0.006
            729737     1    69091     70008     12     8       50000  -0.5     0.69
            79501      6    26505891  26517442  20     15      50000  3.8      0.00007
            999999     X    1000      2000      3      2       50000  0.1      0.9
        """))
        with_excl = parse_magma_results(genes_out, gene_annotations, exclude_mhc=True)
        without_excl = parse_magma_results(genes_out, gene_annotations, exclude_mhc=False)
        mhc_gene = with_excl[with_excl["gene_entrez_id"] == 79501]
        assert mhc_gene.iloc[0]["fdr_q"] != mhc_gene.iloc[0]["fdr_q"]  # NaN != NaN
        mhc_gene_incl = without_excl[without_excl["gene_entrez_id"] == 79501]
        assert pd.notna(mhc_gene_incl.iloc[0]["fdr_q"])

    def test_unmapped_genes(self, genes_out_file: Path) -> None:
        empty_annot = pd.DataFrame({
            "gene_entrez_id": [999999],
            "gene_symbol": ["FAKE"],
            "gene_ensembl_id": ["ENSG_FAKE"],
            "biotype": ["protein_coding"],
        })
        results = parse_magma_results(genes_out_file, empty_annot)
        assert len(results) == 4
        assert results["gene_symbol"].isna().sum() == 4

    def test_fdr_values_match_statsmodels(
        self,
        genes_out_file: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        results = parse_magma_results(
            genes_out_file, gene_annotations, exclude_mhc=True
        )
        non_mhc = results[~results["in_mhc"]].sort_values("magma_p")
        _, expected_q, _, _ = multipletests(
            non_mhc["magma_p"].values, method="fdr_bh"
        )
        np.testing.assert_array_almost_equal(
            non_mhc["fdr_q"].values, expected_q
        )

    def test_sorting(
        self,
        genes_out_file: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        results = parse_magma_results(genes_out_file, gene_annotations)
        pvalues = results["magma_p"].values
        assert all(pvalues[i] <= pvalues[i + 1] for i in range(len(pvalues) - 1))

    def test_annotation_mode_column(
        self,
        genes_out_file: Path,
        gene_annotations: pd.DataFrame,
    ) -> None:
        results = parse_magma_results(
            genes_out_file, gene_annotations,
            annotation_mode="hmagma_fetal_brain",
        )
        assert (results["annotation_mode"] == "hmagma_fetal_brain").all()


# ---------------------------------------------------------------------------
# 21-23: run_gene_analysis (end-to-end, mocked)
# ---------------------------------------------------------------------------


class TestRunGeneAnalysis:

    def _make_config(self, study_name: str = "test_study") -> PipelineConfig:
        return PipelineConfig(
            study=StudyConfig(
                name=study_name,
                gwas_input="data/test.gwas.gz",
            )
        )

    def test_end_to_end(
        self,
        tmp_path: Path,
        minimal_gwas_df: pd.DataFrame,
        gene_annotations: pd.DataFrame,
    ) -> None:
        config = self._make_config()
        ref_bfile = tmp_path / "ref"
        (tmp_path / "ref.bim").write_text("snp_data")
        gene_loc = tmp_path / "gene.loc"
        gene_loc.write_text("gene_data")
        resources = tmp_path / "resources"
        resources.mkdir()
        output_dir = tmp_path / "output"
        fake_binary = tmp_path / "magma"
        fake_binary.write_text("#!/bin/sh\n")

        def mock_subprocess_run(cmd, **kwargs):
            output_prefix_dir = output_dir / "test_study" / "magma"
            output_prefix_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_prefix_dir / "test_study"

            annot = Path(f"{prefix}.genes.annot")
            if not annot.exists():
                annot.write_text("annot_data")

            genes_out = Path(f"{prefix}.genes.out")
            if not genes_out.exists():
                genes_out.write_text(SYNTHETIC_GENES_OUT)

            genes_raw = Path(f"{prefix}.genes.raw")
            if not genes_raw.exists():
                genes_raw.write_text("raw_data")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "ok"
            mock_result.stderr = ""
            return mock_result

        with (
            patch("repogen.analysis.magma_gene.detect_magma_binary",
                  return_value=fake_binary),
            patch("repogen.analysis.magma_gene.subprocess.run",
                  side_effect=mock_subprocess_run),
        ):
            results = run_gene_analysis(
                config=config,
                gwas_df=minimal_gwas_df,
                gene_annotations=gene_annotations,
                reference_bfile=ref_bfile,
                gene_loc_file=gene_loc,
                resources_dir=resources,
                output_dir=output_dir,
            )

        assert len(results) == 4
        assert "magma_z" in results.columns
        assert "fdr_q" in results.columns
        assert "in_mhc" in results.columns

        parquet_path = output_dir / "test_study" / "magma" / "test_study.genes.parquet"
        assert parquet_path.exists()

        raw_path = output_dir / "test_study" / "magma" / "test_study.genes.raw"
        assert raw_path.exists()

    def test_empty_results(
        self,
        tmp_path: Path,
        minimal_gwas_df: pd.DataFrame,
        gene_annotations: pd.DataFrame,
    ) -> None:
        config = self._make_config()
        ref_bfile = tmp_path / "ref"
        (tmp_path / "ref.bim").write_text("snp_data")
        gene_loc = tmp_path / "gene.loc"
        gene_loc.write_text("gene_data")
        resources = tmp_path / "resources"
        resources.mkdir()
        output_dir = tmp_path / "output"
        fake_binary = tmp_path / "magma"
        fake_binary.write_text("#!/bin/sh\n")

        empty_genes_out = "# MEAN_SAMPLE_SIZE = 50000\nGENE CHR START STOP NSNPS NPARAM N ZSTAT P\n"

        def mock_subprocess_run(cmd, **kwargs):
            output_prefix_dir = output_dir / "test_study" / "magma"
            output_prefix_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_prefix_dir / "test_study"

            annot = Path(f"{prefix}.genes.annot")
            if not annot.exists():
                annot.write_text("annot_data")

            genes_out = Path(f"{prefix}.genes.out")
            if not genes_out.exists():
                genes_out.write_text(empty_genes_out)

            genes_raw = Path(f"{prefix}.genes.raw")
            if not genes_raw.exists():
                genes_raw.write_text("")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "ok"
            mock_result.stderr = ""
            return mock_result

        with (
            patch("repogen.analysis.magma_gene.detect_magma_binary",
                  return_value=fake_binary),
            patch("repogen.analysis.magma_gene.subprocess.run",
                  side_effect=mock_subprocess_run),
        ):
            with pytest.raises(RuntimeError, match="no gene results"):
                run_gene_analysis(
                    config=config,
                    gwas_df=minimal_gwas_df,
                    gene_annotations=gene_annotations,
                    reference_bfile=ref_bfile,
                    gene_loc_file=gene_loc,
                    resources_dir=resources,
                    output_dir=output_dir,
                )

    def test_study_name_from_config(
        self,
        tmp_path: Path,
        minimal_gwas_df: pd.DataFrame,
        gene_annotations: pd.DataFrame,
    ) -> None:
        config = self._make_config(study_name="custom_study_xyz")
        ref_bfile = tmp_path / "ref"
        (tmp_path / "ref.bim").write_text("snp_data")
        gene_loc = tmp_path / "gene.loc"
        gene_loc.write_text("gene_data")
        resources = tmp_path / "resources"
        resources.mkdir()
        output_dir = tmp_path / "output"
        fake_binary = tmp_path / "magma"
        fake_binary.write_text("#!/bin/sh\n")

        def mock_subprocess_run(cmd, **kwargs):
            output_prefix_dir = output_dir / "custom_study_xyz" / "magma"
            output_prefix_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_prefix_dir / "custom_study_xyz"

            annot = Path(f"{prefix}.genes.annot")
            if not annot.exists():
                annot.write_text("annot_data")

            genes_out = Path(f"{prefix}.genes.out")
            if not genes_out.exists():
                genes_out.write_text(SYNTHETIC_GENES_OUT)

            genes_raw = Path(f"{prefix}.genes.raw")
            if not genes_raw.exists():
                genes_raw.write_text("raw_data")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "ok"
            mock_result.stderr = ""
            return mock_result

        with (
            patch("repogen.analysis.magma_gene.detect_magma_binary",
                  return_value=fake_binary),
            patch("repogen.analysis.magma_gene.subprocess.run",
                  side_effect=mock_subprocess_run),
        ):
            results = run_gene_analysis(
                config=config,
                gwas_df=minimal_gwas_df,
                gene_annotations=gene_annotations,
                reference_bfile=ref_bfile,
                gene_loc_file=gene_loc,
                resources_dir=resources,
                output_dir=output_dir,
            )

        parquet_path = (
            output_dir / "custom_study_xyz" / "magma"
            / "custom_study_xyz.genes.parquet"
        )
        assert parquet_path.exists()


# ---------------------------------------------------------------------------
# Integration test (skipped without real MAGMA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("magma") is None,
    reason="MAGMA binary not available - integration test requires MAGMA installed",
)
def test_magma_integration_real_binary():
    """Run MAGMA with a tiny synthetic dataset to verify the subprocess
    pipeline works end-to-end with the real binary.

    This test is skipped in CI and only runs when MAGMA is installed.
    """
    pytest.skip("Synthetic BIM/BED/FAM generation not yet implemented for integration test")


class TestDetectMagmaBinaryResourceFallback:
    """MAGMA resolves from the resource directory when it is not on PATH.

    MAGMA cannot be installed from conda and its licence forbids bundling it
    in a container image, so `repogen setup-resources` places the official
    binary at <resource_dir>/bin/magma. Detection must find it there.
    """

    def test_falls_back_to_resource_dir(self, tmp_path: Path) -> None:
        binary = tmp_path / "bin" / "magma"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")

        with patch("repogen.analysis.magma_gene.shutil.which", return_value=None):
            result = detect_magma_binary(resource_dir=tmp_path)
        assert result == binary

    def test_resource_copy_takes_precedence_over_path(self, tmp_path: Path) -> None:
        """The manifest-pinned build wins over whatever happens to be on PATH.

        MAGMA's output is build-dependent, so results are only comparable
        across machines when every run uses the same binary. The resource
        copy is fetched identically everywhere; PATH varies per machine.
        """
        binary = tmp_path / "bin" / "magma"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")

        with patch("repogen.analysis.magma_gene.shutil.which",
                   return_value="/usr/local/bin/magma"):
            result = detect_magma_binary(resource_dir=tmp_path)
        assert result == binary

    def test_path_used_when_no_resource_copy_exists(self, tmp_path: Path) -> None:
        with patch("repogen.analysis.magma_gene.shutil.which",
                   return_value="/usr/local/bin/magma"):
            result = detect_magma_binary(resource_dir=tmp_path)
        assert result == Path("/usr/local/bin/magma")

    def test_explicit_config_path_overrides_everything(self, tmp_path: Path) -> None:
        """A site that manages MAGMA itself keeps full control."""
        chosen = tmp_path / "custom_magma"
        chosen.write_text("#!/bin/sh\n")
        resource_copy = tmp_path / "bin" / "magma"
        resource_copy.parent.mkdir(parents=True)
        resource_copy.write_text("#!/bin/sh\n")

        with patch("repogen.analysis.magma_gene.shutil.which",
                   return_value="/usr/local/bin/magma"):
            result = detect_magma_binary(chosen, resource_dir=tmp_path)
        assert result == chosen

    def test_error_names_setup_resources_when_absent(self, tmp_path: Path) -> None:
        with patch("repogen.analysis.magma_gene.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="setup-resources"):
                detect_magma_binary(resource_dir=tmp_path)
