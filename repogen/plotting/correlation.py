"""Negative correlation scatter, tissue heatmaps, and S-PrediXcan signature visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import linkage, leaves_list

from repogen.plotting.base import (
    FIGSIZE_STANDARD,
    FIG_WIDTH_DOUBLE,
    COLOR_BRANCH_B,
    COLOR_SIGNIFICANT,
    COLOR_NONSIGNIFICANT,
    COLOR_NEUTRAL_MEDIUM,
    PALETTE_DIVERGING,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_TICK,
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


def plot_correlation_scatter(
    correlation_results: pd.DataFrame,
    n_top_labels: int = 15,
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Negative correlation volcano/scatter plot.

    Args:
        correlation_results: DataFrame with columns: drug_name, tissue,
            spearman_rho, fdr_global.
        n_top_labels: Number of top drug names to label.
        fdr_threshold: FDR threshold for colouring.
        figsize: Override figure size. None uses FIGSIZE_STANDARD.
        title: Override figure title.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If correlation_results is empty or missing required columns.
    """
    required = {"drug_name", "tissue", "spearman_rho", "fdr_global"}
    _validate_columns(correlation_results, required, "correlation_results")

    df = correlation_results.copy()
    df["fdr_global"] = df["fdr_global"].clip(lower=np.finfo(float).tiny)
    df["neg_log_fdr"] = -np.log10(df["fdr_global"].values)

    is_sig = df["fdr_global"] < fdr_threshold
    is_neg = df["spearman_rho"] < 0

    colors = np.full(len(df), COLOR_NONSIGNIFICANT, dtype=object)
    colors[is_sig & is_neg] = COLOR_BRANCH_B
    colors[is_sig & ~is_neg] = COLOR_SIGNIFICANT

    if figsize is None:
        figsize = FIGSIZE_STANDARD
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df["spearman_rho"].values,
        df["neg_log_fdr"].values,
        c=colors,
        s=8,
        linewidths=0,
        alpha=0.7,
        zorder=2,
    )

    add_significance_line(ax, -np.log10(fdr_threshold), f"FDR = {fdr_threshold}")

    if n_top_labels > 0:
        candidates = df[is_sig & is_neg].nsmallest(n_top_labels, "fdr_global")
        if not candidates.empty:
            try:
                from adjustText import adjust_text
                texts = [
                    ax.text(
                        row.spearman_rho,
                        row.neg_log_fdr,
                        row.drug_name,
                        fontsize=FONT_SIZE_ANNOTATION,
                        ha="center",
                        va="bottom",
                    )
                    for row in candidates.itertuples()
                ]
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=COLOR_NEUTRAL_MEDIUM, lw=0.5))
            except ImportError:
                logger.warning("adjustText not installed - skipping scatter labels")

    ax.set_xlabel("Spearman \u03c1")
    ax.set_ylabel(r"$-\log_{10}$(global FDR)")
    ax.set_title(title or "Drug-Disease Correlation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    per_tissue_results: pd.DataFrame,
    n_top: int = 30,
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    cluster_rows: bool = True,
    cluster_cols: bool = False,
) -> Figure:
    """Drug-tissue correlation heatmap.

    Pivots the long-form input internally. Rows are drugs, columns are tissues,
    colour encodes Spearman rho.

    Args:
        per_tissue_results: Long-form DataFrame with columns: drug_name,
            tissue, spearman_rho, fdr_global.
        n_top: Number of top drugs (by minimum fdr_global across tissues).
        fdr_threshold: Threshold for significance markers in cells.
        figsize: Override. None computes from n_top.
        title: Override figure title.
        cluster_rows: Hierarchical clustering on drug rows.
        cluster_cols: Hierarchical clustering on tissue columns.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If per_tissue_results is empty or missing required columns.
    """
    required = {"drug_name", "tissue", "spearman_rho", "fdr_global"}
    _validate_columns(per_tissue_results, required, "per_tissue_results")

    min_fdr = per_tissue_results.groupby("drug_name")["fdr_global"].min()
    top_drugs = min_fdr.nsmallest(n_top).index.tolist()

    df = per_tissue_results[per_tissue_results["drug_name"].isin(top_drugs)]
    rho_pivot = df.pivot_table(index="drug_name", columns="tissue", values="spearman_rho", aggfunc="first")
    fdr_pivot = df.pivot_table(index="drug_name", columns="tissue", values="fdr_global", aggfunc="first")

    if cluster_rows and len(rho_pivot) > 1:
        filled = rho_pivot.fillna(0).values
        row_link = linkage(filled, method="average", metric="euclidean")
        row_order = leaves_list(row_link)
        rho_pivot = rho_pivot.iloc[row_order]
        fdr_pivot = fdr_pivot.iloc[row_order]

    if cluster_cols and len(rho_pivot.columns) > 1:
        filled_t = rho_pivot.fillna(0).values.T
        col_link = linkage(filled_t, method="average", metric="euclidean")
        col_order = leaves_list(col_link)
        rho_pivot = rho_pivot.iloc[:, col_order]
        fdr_pivot = fdr_pivot.iloc[:, col_order]

    height = min(15.0, max(4.0, 0.3 * len(rho_pivot) + 2))
    if figsize is None:
        figsize = (FIG_WIDTH_DOUBLE, height)
    fig, ax = plt.subplots(figsize=figsize)

    data_range = max(abs(np.nanmin(rho_pivot.values)), abs(np.nanmax(rho_pivot.values)), 0.1)
    sns.heatmap(
        rho_pivot,
        ax=ax,
        cmap=PALETTE_DIVERGING,
        center=0,
        vmin=-data_range,
        vmax=data_range,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman \u03c1", "shrink": 0.6},
    )

    for i in range(len(rho_pivot)):
        for j in range(len(rho_pivot.columns)):
            fdr_val = fdr_pivot.iloc[i, j]
            if pd.notna(fdr_val) and fdr_val < fdr_threshold:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_ANNOTATION,
                    color="black",
                    fontweight="bold",
                )

    ax.set_title(title or "Drug-Tissue Correlation Heatmap")
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.tight_layout()
    return fig


def plot_tissue_signature(
    signature_per_tissue: pd.DataFrame,
    n_top: int = 30,
    p_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    cluster_rows: bool = True,
) -> Figure:
    """S-PrediXcan tissue-gene heatmap.

    Shows gene x tissue S-PrediXcan Z-scores.

    Args:
        signature_per_tissue: DataFrame with columns: gene_symbol, tissue,
            zscore, pvalue.
        n_top: Number of top genes to show (by min pvalue across tissues).
        p_threshold: Per-tissue significance marker threshold.
        figsize: Override. None computes from n_top.
        title: Override figure title.
        cluster_rows: Hierarchical clustering on gene rows.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If signature_per_tissue is empty or missing required columns.
    """
    required = {"gene_symbol", "tissue", "zscore", "pvalue"}
    _validate_columns(signature_per_tissue, required, "signature_per_tissue")

    min_p = signature_per_tissue.groupby("gene_symbol")["pvalue"].min()
    top_genes = min_p.nsmallest(n_top).index.tolist()

    df = signature_per_tissue[signature_per_tissue["gene_symbol"].isin(top_genes)]
    z_pivot = df.pivot_table(index="gene_symbol", columns="tissue", values="zscore", aggfunc="first")
    p_pivot = df.pivot_table(index="gene_symbol", columns="tissue", values="pvalue", aggfunc="first")

    if cluster_rows and len(z_pivot) > 1:
        filled = z_pivot.fillna(0).values
        row_link = linkage(filled, method="average", metric="euclidean")
        row_order = leaves_list(row_link)
        z_pivot = z_pivot.iloc[row_order]
        p_pivot = p_pivot.iloc[row_order]

    height = min(15.0, max(4.0, 0.3 * len(z_pivot) + 2))
    if figsize is None:
        figsize = (FIG_WIDTH_DOUBLE, height)
    fig, ax = plt.subplots(figsize=figsize)

    data_range = max(abs(np.nanmin(z_pivot.values)), abs(np.nanmax(z_pivot.values)), 0.1)
    sns.heatmap(
        z_pivot,
        ax=ax,
        cmap=PALETTE_DIVERGING,
        center=0,
        vmin=-data_range,
        vmax=data_range,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Z-score", "shrink": 0.6},
    )

    for i in range(len(z_pivot)):
        for j in range(len(z_pivot.columns)):
            p_val = p_pivot.iloc[i, j]
            if pd.notna(p_val) and p_val < p_threshold:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_ANNOTATION,
                    color="black",
                    fontweight="bold",
                )

    ax.set_title(title or "S-PrediXcan Tissue Signature")
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.tight_layout()
    return fig
