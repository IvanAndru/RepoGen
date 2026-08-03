"""Tests for the preflight resource checks behind ``repogen validate``.

The point of these checks is to fail in seconds on a login node rather than
forty minutes into a queued job, so the cases that matter are: missing files
are reported as missing, optional sources are only demanded when configured,
and the exit code distinguishes "nothing can run" from "some branches can".
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from repogen.cli import main
from repogen.config.schema import PipelineConfig
from repogen.config.validate import (
    BRANCH_A,
    BRANCH_B,
    BRANCH_C,
    SHARED,
    check_resources,
    summarise_by_branch,
)


def _config(tmp_path: Path, **overrides) -> PipelineConfig:
    payload = {
        "study": {"name": "TEST", "gwas_input": str(tmp_path / "gwas.tsv.gz")},
        "resource_dir": str(tmp_path / "resources"),
        "reference": {
            "genome_dir": str(tmp_path / "resources" / "reference"),
            "bfile_prefix": "g1000_eur",
            "gene_loc_file": str(tmp_path / "resources" / "reference" / "NCBI37.3.gene.loc"),
            "liftover_chain": str(tmp_path / "resources" / "reference" / "chain.gz"),
            "predixcan_model_dir": str(tmp_path / "resources" / "predixcan_models"),
        },
        # mr.eqtl_sources defaults to a path relative to the working directory.
        # Left at the default these tests would silently read the developer's
        # own resources/ tree and pass for the wrong reason, then fail for
        # anyone who cloned the repository without that data.
        "mr": {
            "eqtl_sources": [
                {"source": "eqtlgen", "path": str(tmp_path / "resources" / "eqtl" / "eqtlgen")}
            ]
        },
    }
    payload.update(overrides)
    return PipelineConfig(**payload)


def _populate_all(tmp_path: Path) -> None:
    """Create every file the checks look for."""
    res = tmp_path / "resources"
    (tmp_path / "gwas.tsv.gz").write_text("x")
    (res / "reference").mkdir(parents=True)
    for suffix in (".bed", ".bim", ".fam"):
        (res / "reference" / f"g1000_eur{suffix}").write_text("x")
    (res / "reference" / "NCBI37.3.gene.loc").write_text("x")
    (res / "reference" / "chain.gz").write_text("x")
    (res / "drugs").mkdir(parents=True)
    (res / "drugs" / "chembl_35.db").write_text("x")
    (res / "pathways").mkdir(parents=True)
    (res / "pathways" / "c5.all.gmt").write_text("x")
    (res / "drug_signatures").mkdir(parents=True)
    (res / "drug_signatures" / "level5.gctx").write_text("x")
    (res / "drug_signatures" / "lincs_gene_info.tsv").write_text("x")
    (res / "drug_signatures" / "repurposing_hub.csv").write_text("x")
    (res / "drug_signatures" / "geneinfo_beta.txt").write_text("x")
    (res / "predixcan_models").mkdir(parents=True)
    (res / "eqtl" / "eqtlgen").mkdir(parents=True)
    (res / "bin").mkdir(parents=True)
    (res / "bin" / "magma").write_text("x")


class TestResourceChecks:
    def test_everything_missing_is_reported(self, tmp_path: Path) -> None:
        # mr.eqtl_sources defaults to a *relative* path, which would resolve
        # against the working directory (the repo, where it exists). Point it
        # somewhere definitely absent so the check is exercised honestly.
        cfg = _config(
            tmp_path,
            mr={"eqtl_sources": [{"source": "eqtlgen", "path": str(tmp_path / "absent")}]},
        )
        with patch("repogen.config.validate.shutil.which", return_value=None):
            checks = check_resources(cfg)
        assert checks, "no checks were produced"
        still_ok = [c.name for c in checks if c.ok]
        assert not still_ok, f"reported present despite nothing existing: {still_ok}"

    def test_everything_present_passes(self, tmp_path: Path) -> None:
        _populate_all(tmp_path)
        cfg = _config(
            tmp_path,
            mr={"eqtl_sources": [{"source": "eqtlgen",
                                  "path": str(tmp_path / "resources" / "eqtl" / "eqtlgen")}]},
        )
        with patch("repogen.config.validate.shutil.which", return_value="/usr/bin/tool"):
            checks = check_resources(cfg)
        failures = [c.name for c in checks if not c.ok]
        assert not failures, f"unexpected failures: {failures}"

    def test_optional_drug_sources_only_checked_when_configured(self, tmp_path: Path) -> None:
        """A ChEMBL-only run must not be told PDSP is missing."""
        _populate_all(tmp_path)
        cfg = _config(tmp_path, drug_enrichment={"sources": ["chembl"]})
        with patch("repogen.config.validate.shutil.which", return_value="/usr/bin/tool"):
            names = {c.name for c in check_resources(cfg)}
        assert "PDSP Ki database" not in names
        assert "DGIdb interactions" not in names

        cfg = _config(tmp_path, drug_enrichment={"sources": ["chembl", "pdsp", "dgidb"]})
        with patch("repogen.config.validate.shutil.which", return_value="/usr/bin/tool"):
            names = {c.name for c in check_resources(cfg)}
        assert "PDSP Ki database" in names
        assert "DGIdb interactions" in names

    def test_magma_found_in_resource_dir_when_absent_from_path(self, tmp_path: Path) -> None:
        """setup-resources installs MAGMA under the resource root, not on PATH."""
        _populate_all(tmp_path)
        with patch("repogen.config.validate.shutil.which", return_value=None):
            checks = check_resources(_config(tmp_path))
        magma = next(c for c in checks if c.name == "MAGMA binary")
        assert magma.ok
        assert magma.location.endswith("magma")

    def test_every_branch_is_covered(self, tmp_path: Path) -> None:
        checks = check_resources(_config(tmp_path))
        assert {c.branch for c in checks} == {SHARED, BRANCH_A, BRANCH_B, BRANCH_C}

    def test_lincs_gene_info_is_checked(self, tmp_path: Path) -> None:
        """Branch B reads a derived landmark gene list, so validate must see it.

        A real cluster run failed at the negative_correlation rule while
        validate had reported Branch B fully satisfied, because this file was
        the one Branch B input nothing checked for.
        """
        _populate_all(tmp_path)
        (tmp_path / "resources" / "drug_signatures" / "lincs_gene_info.tsv").unlink()
        with patch("repogen.config.validate.shutil.which", return_value="/usr/bin/tool"):
            checks = check_resources(_config(tmp_path))
        missing = [c.name for c in checks if not c.ok]
        assert any("landmark" in n.lower() for n in missing), (
            f"a missing lincs_gene_info.tsv was not reported; missing={missing}"
        )

    def test_summary_counts(self, tmp_path: Path) -> None:
        _populate_all(tmp_path)
        with patch("repogen.config.validate.shutil.which", return_value="/usr/bin/tool"):
            summary = summarise_by_branch(check_resources(_config(tmp_path)))
        for branch, (n_ok, n_total) in summary.items():
            assert n_ok == n_total, f"{branch} not fully satisfied"


class TestValidateCommand:
    def _write_config(self, tmp_path: Path, resource_dir: Path) -> Path:
        cfg = tmp_path / "config.yaml"
        ref = resource_dir / "reference"
        eqtl = (resource_dir / "eqtl" / "eqtlgen").as_posix()
        cfg.write_text(
            "study:\n"
            "  name: TEST\n"
            f"  gwas_input: {(tmp_path / 'gwas.tsv.gz').as_posix()}\n"
            f"resource_dir: {resource_dir.as_posix()}\n"
            # Every path below is pinned into tmp_path on purpose. The defaults
            # are relative to the working directory, so left unset these tests
            # would read the developer's own resources/ tree and pass for the
            # wrong reason, then fail for anyone who cloned the repository.
            "reference:\n"
            f"  genome_dir: {ref.as_posix()}\n"
            "  bfile_prefix: g1000_eur\n"
            f"  gene_loc_file: {(ref / 'NCBI37.3.gene.loc').as_posix()}\n"
            f"  liftover_chain: {(ref / 'chain.gz').as_posix()}\n"
            f"  predixcan_model_dir: {(resource_dir / 'predixcan_models').as_posix()}\n"
            "mr:\n"
            "  eqtl_sources:\n"
            "    - source: eqtlgen\n"
            f"      path: {eqtl}\n",
            encoding="utf-8",
        )
        return cfg

    def test_missing_shared_prerequisites_exit_nonzero(self, tmp_path: Path) -> None:
        """Without a GWAS file or reference panel, nothing can run."""
        cfg = self._write_config(tmp_path, tmp_path / "resources")
        result = CliRunner().invoke(main, ["validate", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "no branch can run" in result.output.lower()

    def test_reports_ready_branches(self, tmp_path: Path) -> None:
        _populate_all(tmp_path)
        cfg = self._write_config(tmp_path, tmp_path / "resources")
        with patch("repogen.config.validate.shutil.which", return_value="/usr/bin/tool"):
            result = CliRunner().invoke(main, ["validate", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "Ready to run" in result.output
        assert "A (MAGMA" in result.output

    def test_strict_flag_fails_on_any_missing_resource(self, tmp_path: Path) -> None:
        """Shared prerequisites present, one branch incomplete."""
        _populate_all(tmp_path)
        (tmp_path / "resources" / "drug_signatures" / "level5.gctx").unlink()
        cfg = self._write_config(tmp_path, tmp_path / "resources")

        with patch("repogen.config.validate.shutil.which", return_value="/usr/bin/tool"):
            lenient = CliRunner().invoke(main, ["validate", "--config", str(cfg)])
            strict = CliRunner().invoke(main, ["validate", "--config", str(cfg), "--strict"])

        assert lenient.exit_code == 0, "a missing branch resource must not block other branches"
        assert strict.exit_code == 1
        assert "missing" in strict.output.lower()

    def test_invalid_config_still_fails_first(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("study:\n  name: TEST\n", encoding="utf-8")  # no gwas_input
        result = CliRunner().invoke(main, ["validate", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "Validation failed" in result.output
