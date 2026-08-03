"""Command-line entry points for the figure-producing pipeline steps.

Each subcommand corresponds to one workflow rule. Keeping this orchestration
in the package (rather than embedded in the Snakemake file) means the steps
are importable, testable, and runnable inside a container - Snakemake can only
apply ``conda:``/``container:`` directives to rules that shell out.

Plot functions themselves stay pure: they build and return a Figure, and the
caller decides where it is written. This module is that caller.

Usage::

    python -m repogen.plotting.cli magma-gene --config config.yaml \\
        --gene-results results/magma/study_gene_results.parquet ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from repogen.plotting.status import remove_if_exists, write_plot_status
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


def _load_style_and_study(config_path: Path) -> tuple[object, str]:
    """Return the plot style block and study name from a pipeline config."""
    from repogen.config.loader import load_config

    config = load_config(config_path)
    return config.output.plot_style, config.study.name


def _read_results(path: Path, label: str, *, allow_empty: bool = False) -> pd.DataFrame:
    """Read a results parquet, failing loudly when a figure needs rows.

    A silently empty figure is worse than a failed rule: it looks like a
    negative result. Rules whose emptiness is a legitimate outcome pass
    ``allow_empty=True`` and record it in the status marker instead.
    """
    df = pd.read_parquet(path)
    if df.empty and not allow_empty:
        raise ValueError(f"{label} is empty ({path}) - cannot plot")
    return df


# --- MAGMA ---------------------------------------------------------------

def cmd_magma_gene(args: argparse.Namespace) -> None:
    from repogen.plotting._io import load_gene_sign_lookup
    from repogen.plotting.base import save_figure
    from repogen.plotting.manhattan import plot_gene_volcano, plot_manhattan
    from repogen.plotting.qq import plot_qq

    style, study = _load_style_and_study(args.config)
    df = _read_results(args.gene_results, "Gene results parquet")
    plot_meta = {"study": study, "n_genes": len(df)}

    fig = plot_manhattan(df, style=style.manhattan, meta=plot_meta)
    save_figure(fig, args.manhattan, style=style, rasterize_scatter=True)

    n_eff = int(df["n_samples"].median()) if "n_samples" in df.columns else None
    fig = plot_qq(df, n_effective=n_eff, meta=plot_meta)
    save_figure(fig, args.qq, style=style, rasterize_scatter=True)

    # The signed volcano needs each gene's top-SNP direction, which is
    # recovered from the MAGMA annotation plus the prepared GWAS. It is a
    # presentational refinement, so a failure downgrades to the raw MAGMA Z
    # rather than failing the rule.
    sign_lookup: dict = {}
    if style.volcano.x_axis_mode == "top_snp_beta":
        try:
            sign_lookup = load_gene_sign_lookup(args.annot, args.prepared_gwas)
        except Exception as exc:  # noqa: BLE001 - fall back, never fail the figure
            logger.warning(
                "Gene sign lookup failed (%r); falling back to raw magma_z", exc
            )
            sign_lookup = {}

    fig = plot_gene_volcano(
        df, style=style.volcano, sign_lookup=sign_lookup, meta=plot_meta
    )
    save_figure(fig, args.volcano, style=style, rasterize_scatter=True)

    write_plot_status(
        args.status, "rendered",
        n_rows=len(df),
        n_significant=int((df["fdr_q"] < 0.05).sum()),
        x_axis_mode=style.volcano.x_axis_mode,
        sign_lookup_used=bool(sign_lookup),
        formats=list(style.figure_formats),
    )


def cmd_magma_pathway(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.enrichment import plot_pathway_enrichment

    style, study = _load_style_and_study(args.config)
    df = _read_results(args.pathway_results, "Pathway results parquet")

    fig = plot_pathway_enrichment(df, style=style, meta={"study": study})
    save_figure(fig, args.figure, style=style)

    write_plot_status(
        args.status, "rendered",
        n_rows=len(df),
        n_significant=int((df["fdr_q"] < 0.05).sum()),
        formats=list(style.figure_formats),
    )


# --- Drug / ATC ----------------------------------------------------------

def cmd_drug_enrichment(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.enrichment import plot_drug_enrichment

    style, study = _load_style_and_study(args.config)
    df = _read_results(args.drug_results, "Drug enrichment parquet")

    drug_meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    drug_meta["study"] = study

    fig = plot_drug_enrichment(df, style=style, meta=drug_meta)
    save_figure(fig, args.figure, style=style)

    n_significant = int((df["magma_fdr_q"] < 0.05).sum())
    write_plot_status(
        args.status, "rendered",
        n_rows=len(df),
        n_significant=n_significant,
        no_hit_regime=bool(n_significant == 0),
        formats=list(style.figure_formats),
    )


def cmd_atc_enrichment(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.enrichment import plot_atc_enrichment

    style, study = _load_style_and_study(args.config)
    df = _read_results(args.atc_results, "ATC enrichment parquet", allow_empty=True)

    atc_meta: dict = {}
    meta_path = Path(args.atc_results).parent / "atc_enrichment_metadata.json"
    if meta_path.is_file():
        atc_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    atc_meta["study"] = study

    if df.empty:
        remove_if_exists(args.figure)
        write_plot_status(
            args.status, "skipped_empty",
            reason="ATC enrichment parquet has zero rows",
        )
        return

    fig = plot_atc_enrichment(df, style=style, meta=atc_meta)
    save_figure(fig, args.figure, style=style)
    write_plot_status(
        args.status, "rendered",
        figure=args.figure,
        n_rows=len(df),
        n_significant=int((df["gls_fdr"] < 0.05).sum()),
        formats=list(style.figure_formats),
    )


# --- Correlation (Branch B) ----------------------------------------------

def cmd_tissue_signature(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.correlation import plot_tissue_signature

    df = _read_results(args.per_tissue, "S-PrediXcan per-tissue parquet")
    fig = plot_tissue_signature(df)
    save_figure(fig, args.figure)
    write_plot_status(args.status, "rendered", n_rows=len(df))


def cmd_correlation(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.correlation import (
        plot_correlation_heatmap,
        plot_correlation_scatter,
    )

    df = _read_results(args.per_tissue, "Correlation per-tissue parquet")

    fig = plot_correlation_scatter(df)
    save_figure(fig, args.scatter)

    fig = plot_correlation_heatmap(df)
    save_figure(fig, args.heatmap)

    write_plot_status(args.status, "rendered", n_rows=len(df))


# --- Mendelian randomisation (Branch C) ----------------------------------

def cmd_mr_forest(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.mr import plot_mr_forest, select_forest_rows

    df = _read_results(args.mr_results, "MR results parquet")

    # Row selection is reported alongside the figure: a forest plot showing
    # 123 of 14,298 rows must not be mistaken for the whole result set.
    n_total = len(df)
    n_plotted = len(select_forest_rows(df))

    fig = plot_mr_forest(df)
    save_figure(fig, args.figure, formats=["png"])

    write_plot_status(
        args.status, "rendered",
        n_rows=n_total,
        n_rows_total=n_total,
        n_rows_plotted=n_plotted,
        truncated=n_total > n_plotted,
    )


def cmd_mr_coloc(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.mr import plot_coloc_posteriors

    df = _read_results(args.mr_results, "MR results parquet", allow_empty=True)

    has_pp_h4 = "pp_h4" in df.columns and df["pp_h4"].notna().any()
    if not has_pp_h4:
        remove_if_exists(args.figure)
        write_plot_status(
            args.status, "skipped_not_applicable",
            reason="No plottable pp_h4 values in MR results",
        )
        return

    fig = plot_coloc_posteriors(df)
    save_figure(fig, args.figure)
    write_plot_status(
        args.status, "rendered",
        figure=args.figure,
        n_rows=int(df["pp_h4"].notna().sum()),
    )


def cmd_mr_drug_summary(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.mr import plot_mr_drug_summary

    df = _read_results(args.mr_drug_matches, "MR drug matches parquet", allow_empty=True)

    if df.empty:
        remove_if_exists(args.figure)
        write_plot_status(
            args.status, "skipped_empty",
            reason="MR drug matches parquet has zero rows",
        )
        return

    fig = plot_mr_drug_summary(df)
    save_figure(fig, args.figure)
    write_plot_status(args.status, "rendered", figure=args.figure, n_rows=len(df))


# --- Cross-branch convergence --------------------------------------------

def collect_branch_drug_sets(
    drug_results: Path, nc_summary: Path, mr_drug_matches: Path
) -> dict[str, set]:
    """Return the significant drug set contributed by each branch.

    Only branches with at least one significant drug are included, so the
    caller can tell whether a convergence figure is meaningful.
    """
    branch_drugs: dict[str, set] = {}

    drug_df = pd.read_parquet(drug_results)
    if not drug_df.empty and "magma_fdr_q" in drug_df.columns:
        mask = drug_df["magma_fdr_q"] < 0.05
        # Drugs below the headline gene-count threshold contribute valid
        # class-level statistics but are not headline drug-level results.
        if "passes_headline_min_genes" in drug_df.columns:
            mask = mask & drug_df["passes_headline_min_genes"].fillna(False).astype(bool)
        sig = drug_df.loc[mask]
        if "drug_name" in sig.columns and not sig.empty:
            branch_drugs["MAGMA"] = set(sig["drug_name"].dropna())

    nc_df = pd.read_parquet(nc_summary)
    if not nc_df.empty and "n_tissues_fdr_significant" in nc_df.columns:
        sig = nc_df.loc[nc_df["n_tissues_fdr_significant"] > 0]
        if "drug_name" in sig.columns and not sig.empty:
            branch_drugs["Neg-Corr"] = set(sig["drug_name"].dropna())

    mr_df = pd.read_parquet(mr_drug_matches)
    if not mr_df.empty and "drug_name" in mr_df.columns:
        branch_drugs["MR"] = set(mr_df["drug_name"].dropna())

    return {k: v for k, v in branch_drugs.items() if v}


def cmd_convergence(args: argparse.Namespace) -> None:
    from repogen.plotting.base import save_figure
    from repogen.plotting.convergence import plot_convergence

    non_empty = collect_branch_drug_sets(
        args.drug_results, args.nc_summary, args.mr_drug_matches
    )

    if len(non_empty) < 2:
        remove_if_exists(args.figure)
        write_plot_status(
            args.status, "skipped_not_applicable",
            reason=f"Fewer than 2 non-empty branch drug sets ({len(non_empty)})",
        )
        return

    fig = plot_convergence(non_empty)
    save_figure(fig, args.figure)
    write_plot_status(
        args.status, "rendered",
        figure=args.figure,
        n_rows=sum(len(v) for v in non_empty.values()),
    )


# --- Argument parsing ----------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m repogen.plotting.cli",
        description="Render RepoGen figures and their status markers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, *, needs_config: bool = True):
        p = sub.add_parser(name)
        if needs_config:
            p.add_argument("--config", type=Path, required=True,
                           help="Pipeline config YAML (supplies plot style and study name).")
        p.add_argument("--status", type=Path, required=True,
                       help="Status marker JSON to write.")
        p.set_defaults(handler=handler)
        return p

    p = add("magma-gene", cmd_magma_gene)
    p.add_argument("--gene-results", type=Path, required=True)
    p.add_argument("--annot", type=Path, required=True)
    p.add_argument("--prepared-gwas", type=Path, required=True)
    p.add_argument("--manhattan", type=Path, required=True)
    p.add_argument("--qq", type=Path, required=True)
    p.add_argument("--volcano", type=Path, required=True)

    p = add("magma-pathway", cmd_magma_pathway)
    p.add_argument("--pathway-results", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)

    p = add("drug-enrichment", cmd_drug_enrichment)
    p.add_argument("--drug-results", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)

    p = add("atc-enrichment", cmd_atc_enrichment)
    p.add_argument("--atc-results", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)

    p = add("tissue-signature", cmd_tissue_signature, needs_config=False)
    p.add_argument("--per-tissue", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)

    p = add("correlation", cmd_correlation, needs_config=False)
    p.add_argument("--per-tissue", type=Path, required=True)
    p.add_argument("--scatter", type=Path, required=True)
    p.add_argument("--heatmap", type=Path, required=True)

    p = add("mr-forest", cmd_mr_forest, needs_config=False)
    p.add_argument("--mr-results", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)

    p = add("mr-coloc", cmd_mr_coloc, needs_config=False)
    p.add_argument("--mr-results", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)

    p = add("mr-drug-summary", cmd_mr_drug_summary, needs_config=False)
    p.add_argument("--mr-drug-matches", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)

    p = add("convergence", cmd_convergence, needs_config=False)
    p.add_argument("--drug-results", type=Path, required=True)
    p.add_argument("--nc-summary", type=Path, required=True)
    p.add_argument("--mr-drug-matches", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
