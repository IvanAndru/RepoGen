"""Tests for the plotting and reporting command-line entry points.

These are the interfaces the Snakemake rules call. The rendering paths are
covered by tests/test_plotting.py; what matters here is the orchestration
around them - which inputs trigger a skip, what the status marker records,
and whether stale figures are cleared on a rerun.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from repogen.plotting import cli as plot_cli
from repogen.plotting.status import remove_if_exists, write_plot_status
from repogen.reporting import cli as report_cli


# ---------------------------------------------------------------------------
# Status markers
# ---------------------------------------------------------------------------


class TestPlotStatus:
    def test_writes_minimal_marker(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "status.json"
        write_plot_status(target, "rendered")
        assert json.loads(target.read_text()) == {"status": "rendered"}

    def test_records_optional_fields(self, tmp_path: Path) -> None:
        target = tmp_path / "status.json"
        write_plot_status(
            target, "rendered", figure=tmp_path / "f.png", n_rows=12,
            n_significant=3, formats=["png"],
        )
        payload = json.loads(target.read_text())
        assert payload["n_rows"] == 12
        assert payload["n_significant"] == 3
        assert payload["formats"] == ["png"]
        assert payload["figure"].endswith("f.png")

    def test_skip_records_reason(self, tmp_path: Path) -> None:
        target = tmp_path / "status.json"
        write_plot_status(target, "skipped_empty", reason="zero rows")
        payload = json.loads(target.read_text())
        assert payload == {"status": "skipped_empty", "reason": "zero rows"}

    def test_rejects_unknown_status(self, tmp_path: Path) -> None:
        """A typo must not produce a marker downstream consumers misread."""
        with pytest.raises(ValueError, match="Invalid plot status"):
            write_plot_status(tmp_path / "s.json", "done")

    def test_remove_if_exists_is_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / "figure.png"
        target.write_bytes(b"stale")
        remove_if_exists(target)
        assert not target.exists()
        remove_if_exists(target)  # absent: must not raise


# ---------------------------------------------------------------------------
# Plotting CLI - skip paths
# ---------------------------------------------------------------------------


def _status(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Minimal valid pipeline config; commands read style and study name from it."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "study:\n"
        "  name: TEST_STUDY\n"
        "  gwas_input: data/test.gwas.gz\n",
        encoding="utf-8",
    )
    return cfg


class TestPlottingCliSkipPaths:
    """Empty or inapplicable inputs must skip cleanly, never half-render."""

    def test_atc_empty_skips_and_clears_stale_figure(
        self, tmp_path: Path, config_file: Path
    ) -> None:
        parquet = tmp_path / "atc_enrichment_results.parquet"
        pd.DataFrame(columns=["atc_code", "gls_fdr"]).to_parquet(parquet)
        figure = tmp_path / "atc_enrichment.png"
        figure.write_bytes(b"figure from an earlier run")
        status = tmp_path / "status.json"

        plot_cli.main([
            "atc-enrichment", "--config", str(config_file),
            "--atc-results", str(parquet),
            "--figure", str(figure), "--status", str(status),
        ])

        assert _status(status)["status"] == "skipped_empty"
        assert not figure.exists(), "stale figure must not survive a skip"

    def test_mr_coloc_without_pp_h4_is_not_applicable(self, tmp_path: Path) -> None:
        parquet = tmp_path / "mr_results.parquet"
        pd.DataFrame({"gene_ensembl_id": ["ENSG1"], "mr_pval": [0.01]}).to_parquet(parquet)
        status = tmp_path / "status.json"

        plot_cli.main([
            "mr-coloc", "--mr-results", str(parquet),
            "--figure", str(tmp_path / "coloc.png"), "--status", str(status),
        ])

        payload = _status(status)
        assert payload["status"] == "skipped_not_applicable"
        assert "pp_h4" in payload["reason"]

    def test_mr_coloc_with_all_null_pp_h4_is_not_applicable(self, tmp_path: Path) -> None:
        """Column present but unpopulated is the same non-result."""
        parquet = tmp_path / "mr_results.parquet"
        pd.DataFrame({"pp_h4": [None, None]}).to_parquet(parquet)
        status = tmp_path / "status.json"

        plot_cli.main([
            "mr-coloc", "--mr-results", str(parquet),
            "--figure", str(tmp_path / "coloc.png"), "--status", str(status),
        ])
        assert _status(status)["status"] == "skipped_not_applicable"

    def test_mr_drug_summary_empty_skips(self, tmp_path: Path) -> None:
        parquet = tmp_path / "mr_drug_matches.parquet"
        pd.DataFrame(columns=["drug_name"]).to_parquet(parquet)
        status = tmp_path / "status.json"

        plot_cli.main([
            "mr-drug-summary", "--mr-drug-matches", str(parquet),
            "--figure", str(tmp_path / "f.png"), "--status", str(status),
        ])
        assert _status(status)["status"] == "skipped_empty"

    def test_empty_input_fails_loudly_when_figure_is_required(self, tmp_path: Path) -> None:
        """A mandatory figure must fail rather than emit an empty plot."""
        parquet = tmp_path / "gene.parquet"
        pd.DataFrame(columns=["gene_symbol"]).to_parquet(parquet)
        with pytest.raises(ValueError, match="cannot plot"):
            plot_cli._read_results(parquet, "Gene results parquet")


# ---------------------------------------------------------------------------
# Convergence branch selection
# ---------------------------------------------------------------------------


class TestConvergenceBranchSets:
    def _write(self, tmp_path: Path, drug: pd.DataFrame, nc: pd.DataFrame,
               mr: pd.DataFrame) -> tuple[Path, Path, Path]:
        p1, p2, p3 = (tmp_path / n for n in ("drug.parquet", "nc.parquet", "mr.parquet"))
        drug.to_parquet(p1); nc.to_parquet(p2); mr.to_parquet(p3)
        return p1, p2, p3

    def test_collects_significant_drugs_per_branch(self, tmp_path: Path) -> None:
        paths = self._write(
            tmp_path,
            pd.DataFrame({"drug_name": ["a", "b"], "magma_fdr_q": [0.01, 0.9]}),
            pd.DataFrame({"drug_name": ["c"], "n_tissues_fdr_significant": [2]}),
            pd.DataFrame({"drug_name": ["d", "e"]}),
        )
        sets = plot_cli.collect_branch_drug_sets(*paths)
        assert sets == {"MAGMA": {"a"}, "Neg-Corr": {"c"}, "MR": {"d", "e"}}

    def test_headline_gene_threshold_is_applied(self, tmp_path: Path) -> None:
        """Sub-headline drugs inform class statistics but are not headline hits."""
        paths = self._write(
            tmp_path,
            pd.DataFrame({
                "drug_name": ["a", "b"],
                "magma_fdr_q": [0.01, 0.01],
                "passes_headline_min_genes": [True, False],
            }),
            pd.DataFrame({"drug_name": [], "n_tissues_fdr_significant": []}),
            pd.DataFrame({"drug_name": []}),
        )
        assert plot_cli.collect_branch_drug_sets(*paths) == {"MAGMA": {"a"}}

    def test_empty_branches_are_dropped(self, tmp_path: Path) -> None:
        paths = self._write(
            tmp_path,
            pd.DataFrame({"drug_name": ["a"], "magma_fdr_q": [0.9]}),
            pd.DataFrame({"drug_name": [], "n_tissues_fdr_significant": []}),
            pd.DataFrame({"drug_name": []}),
        )
        assert plot_cli.collect_branch_drug_sets(*paths) == {}

    def test_single_branch_skips_the_figure(self, tmp_path: Path) -> None:
        paths = self._write(
            tmp_path,
            pd.DataFrame({"drug_name": ["a"], "magma_fdr_q": [0.01]}),
            pd.DataFrame({"drug_name": [], "n_tissues_fdr_significant": []}),
            pd.DataFrame({"drug_name": []}),
        )
        status = tmp_path / "status.json"
        plot_cli.main([
            "convergence", "--drug-results", str(paths[0]),
            "--nc-summary", str(paths[1]), "--mr-drug-matches", str(paths[2]),
            "--figure", str(tmp_path / "conv.png"), "--status", str(status),
        ])
        payload = _status(status)
        assert payload["status"] == "skipped_not_applicable"
        assert "Fewer than 2" in payload["reason"]


# ---------------------------------------------------------------------------
# Argument surfaces
# ---------------------------------------------------------------------------


class TestCliArgumentSurfaces:
    """The workflow calls these by name; a rename would break the rules."""

    PLOT_COMMANDS = {
        "magma-gene", "magma-pathway", "drug-enrichment", "atc-enrichment",
        "tissue-signature", "correlation", "mr-forest", "mr-coloc",
        "mr-drug-summary", "convergence",
    }
    REPORT_COMMANDS = {
        "export-magma", "export-drug", "export-correlation", "export-mr",
        "combine", "export-combined", "html-report",
    }

    def _subcommands(self, parser) -> set[str]:
        return {
            name
            for action in parser._actions
            if hasattr(action, "choices") and action.choices
            for name in action.choices
        }

    def test_plotting_commands_present(self) -> None:
        assert self.PLOT_COMMANDS <= self._subcommands(plot_cli.build_parser())

    def test_reporting_commands_present(self) -> None:
        assert self.REPORT_COMMANDS <= self._subcommands(report_cli.build_parser())

    def test_missing_required_argument_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit):
            plot_cli.main(["magma-pathway", "--config", "c.yaml"])

    def test_combine_records_mode(self, tmp_path: Path, monkeypatch) -> None:
        """The metadata sidecar must distinguish a full run from a partial one."""
        class _Combined:
            branches_present = ["magma"]
            metadata = {"files_loaded": {"gene": "x.parquet"}}

        monkeypatch.setattr(report_cli, "_study_name", lambda _p: "STUDY")
        monkeypatch.setattr(report_cli, "_combined", lambda *_a, **_k: _Combined())

        out = tmp_path / "combined_metadata.json"
        report_cli.main([
            "combine", "--config", "c.yaml", "--results-dir", str(tmp_path),
            "--mode", "available", "--output", str(out),
        ])
        payload = json.loads(out.read_text())
        assert payload["mode"] == "available"
        assert payload["study_name"] == "STUDY"
        assert payload["branches_present"] == ["magma"]
