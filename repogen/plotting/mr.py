"""Visualizations for Mendelian Randomisation results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from repogen.plotting.base import (
    FIG_WIDTH_DOUBLE,
    COLOR_NONSIGNIFICANT,
    COLOR_ZERO_LINE,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MEDIUM,
    COLOR_SECONDARY_TEXT,
    TIER_COLORS,
    COLOC_COLORS,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_MINI,
    FONT_SIZE_TICK,
    format_pvalue,
    truncate_label,
    add_significance_line,
)
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise ValueError if df is empty or missing required columns."""
    if df.empty:
        raise ValueError(f"{name} is empty - nothing to plot")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {name}: {missing}")


_SOURCE_MARKERS: dict[str, str] = {
    "eqtlgen": "o",
    "metabrain_cortex": "^",
}


def select_forest_rows(df: pd.DataFrame, max_rows: int = 250) -> pd.DataFrame:
    """Select rows for forest plot display (single source of truth).

    Selection policy:
        1. If ``mr_significant`` column exists and has any True values,
           return only significant rows sorted by mr_pval ascending,
           capped at ``max_rows``.
        2. Otherwise, return top ``max_rows`` rows by mr_pval ascending.

    Args:
        df: Full MR results DataFrame (must contain ``mr_pval``).
        max_rows: Maximum rows to include in the forest plot.

    Returns:
        Filtered DataFrame ready for plotting.
    """
    n_total = len(df)

    if "mr_significant" in df.columns and df["mr_significant"].any():
        selected = df[df["mr_significant"] == True].copy()  # noqa: E712
        selected = selected.sort_values("mr_pval", ascending=True).head(max_rows)
    else:
        selected = df.sort_values("mr_pval", ascending=True).head(max_rows)

    n_plotted = len(selected)
    if n_plotted < n_total:
        logger.warning(
            "Forest plot truncated: %d/%d rows selected (max_rows=%d)",
            n_plotted, n_total, max_rows,
        )

    return selected


def plot_mr_forest(
    mr_results: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    max_rows: int = 250,
    max_height: float = 24.0,
) -> Figure:
    """MR forest plot - effect estimates with confidence intervals.

    Args:
        mr_results: DataFrame with columns: gene_symbol, eqtl_source, mr_beta,
            mr_se, mr_pval, confidence_tier, n_instruments.
            Optional: mr_method, mr_significant.
        figsize: Override. None computes from number of genes.
        title: Override figure title.
        max_rows: Maximum rows to display (default 250).
        max_height: Maximum figure height in inches (default 24).

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If mr_results is empty or missing required columns.
    """
    required = {"gene_symbol", "eqtl_source", "mr_beta", "mr_se", "mr_pval", "confidence_tier", "n_instruments"}
    _validate_columns(mr_results, required, "mr_results")

    df = select_forest_rows(mr_results, max_rows=max_rows)
    sources = sorted(df["eqtl_source"].unique())
    n_rows = len(df)

    height = min(max(3.0, 0.25 * n_rows + 2), max_height)
    if figsize is None:
        figsize = (FIG_WIDTH_DOUBLE, height)
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = 0
    y_positions = []
    y_labels = []
    group_boundaries: list[float] = []

    for s_idx, source in enumerate(sources):
        source_df = df[df["eqtl_source"] == source].sort_values("mr_pval")
        if s_idx > 0 and len(source_df) > 0:
            group_boundaries.append(y_pos - 0.5)
            y_pos += 0.5

        for row in source_df.itertuples():
            ci_lo = row.mr_beta - 1.96 * row.mr_se
            ci_hi = row.mr_beta + 1.96 * row.mr_se
            color = TIER_COLORS.get(row.confidence_tier, COLOR_NONSIGNIFICANT)
            marker = _SOURCE_MARKERS.get(row.eqtl_source, "o")

            ax.plot([ci_lo, ci_hi], [y_pos, y_pos], color=color, linewidth=1.5, zorder=2)
            ax.scatter([row.mr_beta], [y_pos], c=color, marker=marker, s=40, zorder=3, edgecolors="white", linewidths=0.5)

            ax.text(
                ax.get_xlim()[1] if ax.get_xlim()[1] != 1.0 else 1.0,
                y_pos,
                f"  {format_pvalue(row.mr_pval)}  k={row.n_instruments}",
                va="center",
                ha="left",
                fontsize=FONT_SIZE_ANNOTATION,
                clip_on=False,
            )

            y_positions.append(y_pos)
            y_labels.append(row.gene_symbol)
            y_pos += 1

    ax.axvline(0, color=COLOR_ZERO_LINE, linestyle="-", linewidth=0.8, zorder=1)

    for boundary in group_boundaries:
        ax.axhline(boundary, color=COLOR_NEUTRAL_LIGHT, linestyle=":", linewidth=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("MR \u03b2 (log-OR scale)")
    ax.set_title(title or "Mendelian Randomisation Forest Plot")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    tier_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TIER_COLORS[t],
               markersize=6, label=t.replace("_", " ").title())
        for t in ["high", "medium", "low", "direction_conflict"]
        if t in df["confidence_tier"].values
    ]
    if len(sources) > 1:
        for source in sources:
            marker = _SOURCE_MARKERS.get(source, "o")
            tier_handles.append(
                Line2D([0], [0], marker=marker, color="none", markerfacecolor=COLOR_NEUTRAL_MEDIUM,
                       markersize=6, label=source)
            )
    if tier_handles:
        ax.legend(handles=tier_handles, loc="lower right", frameon=False)

    fig.tight_layout()
    return fig


def plot_coloc_posteriors(
    mr_results: pd.DataFrame,
    n_top: int = 20,
    pp_h4_threshold: float = 0.8,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Colocalisation PP.H4 bar chart.

    Args:
        mr_results: DataFrame with columns: gene_symbol, pp_h4, pp_h3,
            coloc_status. Rows with NaN pp_h4 are excluded.
        n_top: Number of top genes by pp_h4.
        pp_h4_threshold: Dashed vertical line threshold.
        figsize: Override. None computes from n_top.
        title: Override figure title.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If no genes have valid (non-NaN) pp_h4 values.
    """
    required = {"gene_symbol", "pp_h4", "pp_h3", "coloc_status"}
    missing = required - set(mr_results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = mr_results.dropna(subset=["pp_h4"]).copy()
    if df.empty:
        raise ValueError("No genes have valid (non-NaN) pp_h4 values")

    df = df.nlargest(n_top, "pp_h4").sort_values("pp_h4", ascending=True)

    colors = df["coloc_status"].map(
        lambda x: COLOC_COLORS.get(x, COLOR_NONSIGNIFICANT)
    ).values

    height = max(3.0, 0.3 * len(df) + 1.5)
    if figsize is None:
        figsize = (FIG_WIDTH_DOUBLE, height)
    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(len(df))
    ax.barh(y_positions, df["pp_h4"].values, color=colors, edgecolor="white", linewidth=0.5, zorder=2)

    add_significance_line(ax, pp_h4_threshold, f"PP.H4 = {pp_h4_threshold}", "vertical")

    for i, row in enumerate(df.itertuples()):
        if pd.notna(row.pp_h3) and row.pp_h3 > 0.5:
            ax.text(
                row.pp_h4 + 0.01,
                i,
                f"H3={row.pp_h3:.2f}",
                va="center",
                ha="left",
                fontsize=FONT_SIZE_ANNOTATION,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["gene_symbol"].values, fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("PP.H4")
    ax.set_xlim(0, 1.05)
    ax.set_title(title or "Colocalisation Posterior Probabilities")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    unique_statuses = df["coloc_status"].unique()
    status_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOC_COLORS.get(s, COLOR_NONSIGNIFICANT),
               markersize=8, label=s.replace("_", " ").title())
        for s in unique_statuses
    ]
    if status_handles:
        ax.legend(handles=status_handles, loc="lower right", frameon=False)

    fig.tight_layout()
    return fig


def plot_mr_drug_summary(
    drug_matches: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """MR drug matching summary - gene x drug tile/dot matrix.

    Aggregates across eQTL sources before plotting.

    Args:
        drug_matches: DataFrame with columns: gene_symbol, drug_name,
            confidence_tier, direction_concordant, interaction_type, max_phase.
            Optional: eqtl_source.
        figsize: Override. None computes from matrix dimensions.
        title: Override figure title.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If drug_matches is empty or missing required columns.
    """
    required = {"gene_symbol", "drug_name", "confidence_tier", "direction_concordant", "interaction_type", "max_phase"}
    _validate_columns(drug_matches, required, "drug_matches")

    df = drug_matches.copy()

    tier_rank = {"high": 0, "medium": 1, "low": 2, "direction_conflict": 3}
    concordance_rank = {True: 0, False: 1, None: 2}

    n_before = len(df)
    df["_tier_rank"] = df["confidence_tier"].map(tier_rank).fillna(3)
    df["_conc_rank"] = df["direction_concordant"].map(concordance_rank).fillna(2)
    df = df.sort_values(["gene_symbol", "drug_name", "_tier_rank", "_conc_rank"])
    df = df.drop_duplicates(subset=["gene_symbol", "drug_name"], keep="first")
    n_after = len(df)
    if n_before != n_after:
        logger.info("Aggregated %d drug match rows to %d gene-drug pairs across eQTL sources", n_before, n_after)
    df = df.drop(columns=["_tier_rank", "_conc_rank"])

    genes = sorted(df["gene_symbol"].unique())
    drugs = sorted(df["drug_name"].unique(), key=lambda d: (-df.loc[df["drug_name"] == d, "max_phase"].max(), d))
    if len(drugs) > 30:
        logger.warning("Truncating drug list from %d to 30 for readability", len(drugs))
        drugs = drugs[:30]
        df = df[df["drug_name"].isin(drugs)]

    n_genes = len(genes)
    n_drugs = len(drugs)
    gene_idx = {g: i for i, g in enumerate(genes)}
    drug_idx = {d: i for i, d in enumerate(drugs)}

    width = max(FIG_WIDTH_DOUBLE, 0.5 * n_drugs + 2)
    height = max(3.0, 0.3 * n_genes + 2)
    if figsize is None:
        figsize = (width, height)
    fig, ax = plt.subplots(figsize=figsize)

    for row in df.itertuples():
        x = drug_idx.get(row.drug_name)
        y = gene_idx.get(row.gene_symbol)
        if x is None or y is None:
            continue

        color = TIER_COLORS.get(row.confidence_tier, COLOR_NONSIGNIFICANT)
        concordant = row.direction_concordant

        if pd.isna(concordant):
            ax.scatter([x], [y], c=color, marker="D", s=60, zorder=3, edgecolors="white", linewidths=0.5)
        elif bool(concordant):
            ax.scatter([x], [y], c=color, marker="o", s=80, zorder=3, edgecolors="white", linewidths=0.5)
        else:
            ax.scatter([x], [y], facecolors="none", edgecolors=color, marker="o", s=80, linewidths=1.5, zorder=3)

        itype = str(row.interaction_type)[:3] if pd.notna(row.interaction_type) else ""
        ax.text(x, y + 0.3, itype, ha="center", va="top", fontsize=FONT_SIZE_MINI, color=COLOR_SECONDARY_TEXT)

    ax.set_xticks(range(n_drugs))
    ax.set_xticklabels(
        [truncate_label(d, 25) for d in drugs],
        rotation=45,
        ha="right",
        fontsize=FONT_SIZE_TICK,
    )
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(genes, fontsize=FONT_SIZE_TICK)
    ax.set_xlim(-0.5, n_drugs - 0.5)
    ax.set_ylim(-0.5, n_genes - 0.5)
    ax.invert_yaxis()
    ax.set_title(title or "MR Drug Matching Summary")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_NEUTRAL_MEDIUM, markersize=6, label="Concordant"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=COLOR_NEUTRAL_MEDIUM,
               markeredgewidth=1.5, markersize=6, label="Discordant"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLOR_NEUTRAL_MEDIUM, markersize=5, label="Ambiguous"),
    ]
    for tier_name in ["high", "medium", "low", "direction_conflict"]:
        if tier_name in df["confidence_tier"].values:
            legend_handles.append(
                Line2D([0], [0], marker="s", color="none", markerfacecolor=TIER_COLORS[tier_name],
                       markersize=6, label=tier_name.replace("_", " ").title())
            )
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=FONT_SIZE_ANNOTATION)

    fig.tight_layout()
    return fig
