"""Command-line entry points for export, aggregation and report rendering.

Counterpart to :mod:`repogen.plotting.cli`: the workflow calls these
subcommands instead of embedding Python in the Snakemake file, so every step
is importable, testable, and able to run inside a container.

Usage::

    python -m repogen.reporting.cli export-magma --config config.yaml \\
        --gene results/magma/study_gene_results.parquet ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repogen.utils.io import ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

_EXPORT_FORMATS = ["csv", "json", "xlsx"]


def _study_name(config_path: Path) -> str:
    from repogen.config.loader import load_config

    return load_config(config_path).study.name


def _combined(results_dir: Path, study_name: str):
    """Load whichever branch results exist under *results_dir*."""
    from repogen.reporting.combine_results import combine_results

    return combine_results(results_dir=results_dir, study_name=study_name)


# --- Per-branch exports --------------------------------------------------

def _export_each(pairs: list[tuple[str, Path, Path | None]], out_dir: Path,
                 study: str) -> None:
    from repogen.reporting.export import export_results

    for result_type, results_path, metadata_path in pairs:
        export_results(
            result_type=result_type,
            results_path=results_path,
            metadata_path=metadata_path,
            output_dir=out_dir,
            study_name=study,
            formats=_EXPORT_FORMATS,
        )


def cmd_export_magma(args: argparse.Namespace) -> None:
    study = _study_name(args.config)
    # The pathway exporter enriches its JSON from the sidecar written beside
    # the parquet; absent sidecar simply means less metadata, not an error.
    pathway_meta = args.pathway.with_name(args.pathway.stem + "_meta.json")
    _export_each(
        [
            ("gene", args.gene, None),
            ("pathway", args.pathway, pathway_meta if pathway_meta.is_file() else None),
        ],
        args.output_dir, study,
    )


def cmd_export_drug(args: argparse.Namespace) -> None:
    study = _study_name(args.config)
    _export_each(
        [("drug", args.drug, None), ("atc", args.atc, None)],
        args.output_dir, study,
    )


def cmd_export_correlation(args: argparse.Namespace) -> None:
    study = _study_name(args.config)
    _export_each(
        [
            ("spredixcan", args.spx_meta, None),
            ("spredixcan_per_tissue", args.spx_tissue, None),
            ("correlation", args.nc_summary, None),
            ("correlation_per_tissue", args.nc_tissue, None),
        ],
        args.output_dir, study,
    )


def cmd_export_mr(args: argparse.Namespace) -> None:
    study = _study_name(args.config)
    _export_each(
        [("mr", args.mr_results, None), ("mr_drugs", args.mr_drugs, None)],
        args.output_dir, study,
    )


# --- Cross-branch aggregation and reporting ------------------------------

def cmd_combine(args: argparse.Namespace) -> None:
    """Write the metadata sidecar describing which branches were found."""
    study = _study_name(args.config)
    combined = _combined(args.results_dir, study)

    meta = {
        "study_name": study,
        "branches_present": combined.branches_present,
        "files_loaded": combined.metadata.get("files_loaded", {}),
        "mode": args.mode,
    }
    ensure_directory(args.output.parent)
    args.output.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Combined metadata written: %s (branches: %s)",
        args.output, ", ".join(combined.branches_present) or "none",
    )


def cmd_export_combined(args: argparse.Namespace) -> None:
    from repogen.reporting.export import export_combined

    study = _study_name(args.config)
    combined = _combined(args.results_dir, study)
    export_combined(combined, output_dir=args.output_dir, formats=["json", "xlsx"])


def cmd_html_report(args: argparse.Namespace) -> None:
    from repogen.reporting.html_report import generate_html_report

    study = _study_name(args.config)
    combined = _combined(args.results_dir, study)
    ensure_directory(args.output.parent)
    generate_html_report(combined, plot_dir=args.plots_dir, output_path=args.output)


# --- Argument parsing ----------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m repogen.reporting.cli",
        description="Export RepoGen results and render reports.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True,
                       help="Pipeline config YAML (supplies the study name).")
        p.set_defaults(handler=handler)
        return p

    p = add("export-magma", cmd_export_magma)
    p.add_argument("--gene", type=Path, required=True)
    p.add_argument("--pathway", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p = add("export-drug", cmd_export_drug)
    p.add_argument("--drug", type=Path, required=True)
    p.add_argument("--atc", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p = add("export-correlation", cmd_export_correlation)
    p.add_argument("--spx-meta", type=Path, required=True)
    p.add_argument("--spx-tissue", type=Path, required=True)
    p.add_argument("--nc-summary", type=Path, required=True)
    p.add_argument("--nc-tissue", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p = add("export-mr", cmd_export_mr)
    p.add_argument("--mr-results", type=Path, required=True)
    p.add_argument("--mr-drugs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p = add("combine", cmd_combine)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Pipeline output root (the parent of the study directory).")
    p.add_argument("--mode", choices=["full", "available"], required=True)
    p.add_argument("--output", type=Path, required=True)

    p = add("export-combined", cmd_export_combined)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p = add("html-report", cmd_html_report)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--plots-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
