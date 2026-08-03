"""Horizontal dot plots for pathway, drug, and ATC class enrichment results."""

from __future__ import annotations

from textwrap import fill, shorten
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from repogen.plotting.base import (
    FIGSIZE_STANDARD,
    FIGSIZE_FOREST,
    FIG_WIDTH_DOUBLE,
    PALETTE_CATEGORICAL,
    PHASE_COLORS,
    ATC_LEVEL_COLORS,
    SIG_MARKERS,
    COLOR_SIGNIFICANT,
    COLOR_NONSIGNIFICANT,
    COLOR_NEUTRAL_MEDIUM,
    COLOR_ZERO_LINE,
    FONT_SIZE_TICK,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_MINI,
    render_threshold_line,
    fdr_implied_p_cutoff,
    bonferroni_cutoff,
    annotate_stats,
    figure_footer,
    panel_label,
    humanise_name,
    no_hit_overlay,
    truncate_label,
)
from repogen.utils.logging import setup_logging

if TYPE_CHECKING:
    from repogen.config.schema import PlotStyleConfig

logger = setup_logging(__name__)


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} is empty - nothing to plot")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {name}: {missing}")


def _clamp_log10(series: pd.Series) -> np.ndarray:
    return -np.log10(np.maximum(series.values.astype(float), np.finfo(float).tiny))


# ---------------------------------------------------------------------------
# Pathway enrichment (§6.2)
# ---------------------------------------------------------------------------

def plot_pathway_enrichment(
    pathway_results: pd.DataFrame,
    n_top: int = 20,
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    *,
    style: PlotStyleConfig | None = None,
    panel: str | None = None,
    meta: dict | None = None,
) -> Figure:
    """Pathway enrichment dot plot -- x-axis is -log10(p_value), FDR via marker fill."""
    required = {"pathway_name", "source_db", "p_value", "fdr_q", "n_genes_in_set"}
    _validate_columns(pathway_results, required, "pathway_results")

    max_label_len = style.max_label_len if style else 40

    df = pathway_results.nsmallest(n_top, "p_value").copy()
    if "pathway_id" in df.columns:
        df["pathway_name"] = df["pathway_name"].fillna(df["pathway_id"])
    df["pathway_name"] = df["pathway_name"].fillna("UNKNOWN")
    df["source_db"] = df["source_db"].fillna("UNKNOWN")

    df["neg_log_p"] = _clamp_log10(df["p_value"])
    df["_display_name"] = df["pathway_name"].apply(lambda x: humanise_name(str(x), "pathway"))
    df["label"] = df.apply(
        lambda row: fill(str(row["_display_name"]), width=max_label_len), axis=1
    )
    df = df.sort_values("neg_log_p", ascending=True)

    sources = df["source_db"].unique()
    source_color_map = {s: PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)] for i, s in enumerate(sources)}
    colors = df["source_db"].map(source_color_map).values

    gene_counts = df["n_genes_in_set"].values.astype(float)
    gc_min, gc_max = gene_counts.min(), gene_counts.max()
    gc_range = gc_max - gc_min if gc_max > gc_min else 1
    sizes = 30 + 200 * np.sqrt((gene_counts - gc_min) / gc_range)

    is_fdr = df["fdr_q"].values < fdr_threshold
    facecolors = np.array([c if sig else "none" for c, sig in zip(colors, is_fdr)])
    edgecolors = colors

    height = max(4.5, 0.3 * len(df) + 1)
    if figsize is None:
        figsize = (FIG_WIDTH_DOUBLE, height)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    y_positions = np.arange(len(df))
    ax.scatter(df["neg_log_p"].values, y_positions, c=facecolors, s=sizes,
               linewidths=0.8, edgecolors=edgecolors, zorder=2)

    # Threshold lines
    p_cut = fdr_implied_p_cutoff(df["p_value"].values, df["fdr_q"].values, fdr_threshold)
    if p_cut is not None:
        render_threshold_line(ax, -np.log10(p_cut),
                              f"FDR {fdr_threshold} (p\u2264{p_cut:.2g})",
                              orientation="vertical", position="above")
    else:
        n_tested = len(pathway_results)
        bonf = bonferroni_cutoff(n_tested)
        render_threshold_line(ax, -np.log10(bonf), "Bonferroni (no FDR hits)",
                              orientation="vertical", position="above")
    render_threshold_line(ax, -np.log10(0.05), "nominal p=0.05",
                          orientation="vertical", color="#BBBBBB", linestyle=":", position="above")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["label"].values, fontsize=FONT_SIZE_TICK)
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_title(title or "Pathway Enrichment")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend outside
    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=source_color_map[s],
               markersize=6, label=s)
        for s in sources
    ]
    color_handles.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#333333", markersize=6, label=f"FDR \u2265 {fdr_threshold}")
    )
    ax.legend(handles=color_handles, title="Source", loc="lower right",
              fontsize=6, framealpha=0.8)

    # Stats block
    n_tested = len(pathway_results)
    n_fdr = int((pathway_results["fdr_q"] < fdr_threshold).sum())
    source_list = ", ".join(sorted(pathway_results["source_db"].dropna().unique()))
    annotate_stats(ax, f"N tested = {n_tested:,} | N FDR<{fdr_threshold} = {n_fdr} | sources = {source_list}")

    if panel:
        panel_label(ax, panel)
    if meta:
        figure_footer(fig, meta)

    return fig


# ---------------------------------------------------------------------------
# Drug enrichment (§6.3)
# ---------------------------------------------------------------------------

def plot_drug_enrichment(
    drug_results: pd.DataFrame,
    n_top: int = 20,
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    *,
    style: PlotStyleConfig | None = None,
    panel: str | None = None,
    meta: dict | None = None,
) -> Figure:
    """Drug enrichment dot plot -- x-axis is -log10(magma_p), with no-hit regime support."""
    required = {"drug_name", "magma_p", "magma_fdr_q", "max_phase"}
    _validate_columns(drug_results, required, "drug_results")

    if "passes_headline_min_genes" in drug_results.columns:
        headline_mask = drug_results["passes_headline_min_genes"].fillna(False).astype(bool)
        drug_results = drug_results.loc[headline_mask].copy()

    max_label_len = style.max_label_len if style else 40
    size_by = "wilcoxon_auc"
    if style and hasattr(style, "drug_enrichment"):
        size_by = style.drug_enrichment.size_by
        n_top = style.drug_enrichment.top_n

    df = drug_results.nsmallest(n_top, "magma_p").copy()
    df["neg_log_p"] = _clamp_log10(df["magma_p"])

    df["_display_name"] = df["drug_name"].apply(lambda x: humanise_name(str(x), "drug"))
    df["label"] = df["_display_name"].apply(lambda x: truncate_label(x, max_label_len))
    df = df.sort_values("neg_log_p", ascending=True)

    n_sig = int((df["magma_fdr_q"] < fdr_threshold).sum())
    no_hit = n_sig == 0

    # Phase colours
    if no_hit:
        colors = np.full(len(df), COLOR_NEUTRAL_MEDIUM, dtype=object)
    else:
        colors = df["max_phase"].map(lambda x: PHASE_COLORS.get(int(x), COLOR_NONSIGNIFICANT)).values

    # Size
    has_size_col = size_by in df.columns and df[size_by].notna().any()
    if has_size_col:
        vals = df[size_by].fillna(0.5 if size_by == "wilcoxon_auc" else 1).values.astype(float)
        v_min, v_max = vals.min(), vals.max()
        v_range = v_max - v_min if v_max > v_min else 1
        sizes = 30 + 200 * (vals - v_min) / v_range
    else:
        sizes = np.full(len(df), 60)

    # FDR fill
    is_fdr = df["magma_fdr_q"].values < fdr_threshold
    facecolors = np.array([c if sig else "none" for c, sig in zip(colors, is_fdr)])
    edgecolors = colors

    height = max(4.5, 0.3 * len(df) + 1)
    if figsize is None:
        figsize = (FIG_WIDTH_DOUBLE, height)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    y_positions = np.arange(len(df))
    ax.scatter(df["neg_log_p"].values, y_positions, c=facecolors, s=sizes,
               linewidths=0.8, edgecolors=edgecolors, zorder=2)

    # Direction overlay (inner dot for negative beta)
    if "magma_beta" in df.columns:
        neg_beta = df["magma_beta"] < 0
        if neg_beta.any():
            neg_sub = df[neg_beta]
            neg_sizes = sizes[neg_beta.values] * 0.09
            neg_y = y_positions[neg_beta.values]
            ax.scatter(neg_sub["neg_log_p"].values, neg_y, c="black",
                       s=neg_sizes, linewidths=0, zorder=3)

    # No-hit overlay
    if no_hit:
        no_hit_overlay(ax, n_sig=0, threshold=fdr_threshold, kind="drugs")

    # Phase chip labels
    phase_labels = {0: "Pre", 1: "Ph1", 2: "Ph2", 3: "Ph3", 4: "Ph4"}
    for i, row in enumerate(df.itertuples()):
        ph = int(row.max_phase)
        chip = phase_labels.get(ph, f"Ph{ph}")
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else df["neg_log_p"].max() * 1.05,
                i, f" [{chip}]", fontsize=5, va="center", ha="left",
                color=PHASE_COLORS.get(ph, "#999999"))

    # Threshold lines
    if not no_hit:
        p_cut = fdr_implied_p_cutoff(df["magma_p"].values, df["magma_fdr_q"].values, fdr_threshold)
        if p_cut is not None:
            render_threshold_line(ax, -np.log10(p_cut),
                                  f"FDR {fdr_threshold} (p\u2264{p_cut:.2g})",
                                  orientation="vertical", position="above")

    if meta:
        n_tested_meta = meta.get("n_drugs_in_headline_pool")
        if n_tested_meta is None:
            n_tested_meta = meta.get("n_drugs_tested", len(drug_results))
    else:
        n_tested_meta = len(drug_results)
    bonf = bonferroni_cutoff(n_tested_meta)
    render_threshold_line(ax, -np.log10(bonf),
                          "Bonferroni" + (" (no FDR hits)" if no_hit else ""),
                          orientation="vertical", color="#555555", position="above")
    render_threshold_line(ax, -np.log10(0.05), "nominal p=0.05",
                          orientation="vertical", color="#BBBBBB", linestyle=":", position="above")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["label"].values, fontsize=FONT_SIZE_TICK)
    ax.set_xlabel(r"$-\log_{10}(p)$")

    if no_hit:
        ax.set_title(title or f"Drug Enrichment \u2014 top {n_top} nominal (no drugs pass FDR {fdr_threshold})")
    else:
        ax.set_title(title or "Drug Enrichment")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Phase legend (only present phases)
    present_phases = sorted(df["max_phase"].dropna().astype(int).unique())
    phase_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=PHASE_COLORS.get(p, COLOR_NONSIGNIFICANT),
               markersize=6, label=f"Phase {p}" if p > 0 else "Preclinical")
        for p in present_phases
    ]
    if phase_handles:
        ax.legend(handles=phase_handles, title="Clinical Phase", loc="lower right",
                  fontsize=6, framealpha=0.8)

    # Stats block - counts from headline pool (not the full inclusive ATC pool)
    if meta:
        n_drugs_tested = meta.get("n_drugs_in_headline_pool")
        if n_drugs_tested is None:
            n_drugs_tested = meta.get("n_drugs_tested", len(drug_results))
    else:
        n_drugs_tested = len(drug_results)
    n_sig_fdr05 = meta.get("n_significant_fdr05", n_sig) if meta else n_sig
    n_with_atc = 0
    if "atc_codes" in drug_results.columns:
        n_with_atc = int(drug_results["atc_codes"].apply(
            lambda x: x is not None and (isinstance(x, (list, np.ndarray)) and len(x) > 0)
        ).sum())
    top_p = f"{float(df['magma_p'].min()):.2e}" if not df.empty else "N/A"
    stats_text = (
        f"N drugs tested = {n_drugs_tested:,} | N with ATC = {n_with_atc}\n"
        f"N FDR<{fdr_threshold} = {n_sig_fdr05} | top p = {top_p}"
    )
    annotate_stats(ax, stats_text)

    if panel:
        panel_label(ax, panel)
    if meta:
        figure_footer(fig, meta)

    return fig


# ---------------------------------------------------------------------------
# ATC enrichment -- forest-style (§6.4)
# ---------------------------------------------------------------------------

def plot_atc_enrichment(
    atc_results: pd.DataFrame,
    n_top: int = 20,
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    *,
    style: PlotStyleConfig | None = None,
    panel: str | None = None,
    meta: dict | None = None,
) -> Figure:
    """ATC class enrichment forest plot -- x-axis is gls_beta with error bars."""
    required = {"atc_code", "atc_description", "atc_level", "gls_p", "gls_fdr", "n_drugs"}
    _validate_columns(atc_results, required, "atc_results")

    max_label_len = style.max_label_len if style else 40

    df = atc_results.nsmallest(n_top, "gls_fdr").copy()
    df = df.sort_values("gls_fdr", ascending=True)

    has_beta = "gls_beta" in df.columns
    has_se = "gls_se" in df.columns

    # Hierarchy indent: sort by level, then group level-3 under level-2 parent
    df["_level"] = df["atc_level"].astype(int)
    df["_parent_code"] = df["atc_code"].str[:3]
    df = df.sort_values(["_parent_code", "_level", "gls_fdr"])

    # Build labels
    labels = []
    for _, row in df.iterrows():
        raw = f"{row['atc_code']} {row['atc_description']}"
        name = humanise_name(raw, "atc")
        name = fill(name, width=max_label_len)
        if int(row["atc_level"]) == 3:
            name = "   " + name
        labels.append(name)
    df["label"] = labels

    # Significance markers
    sig_texts = []
    for _, row in df.iterrows():
        fdr = float(row["gls_fdr"])
        p = float(row["gls_p"])
        if fdr < 0.001:
            sig_texts.append(SIG_MARKERS["fdr_001"])
        elif fdr < 0.01:
            sig_texts.append(SIG_MARKERS["fdr_01"])
        elif fdr < 0.05:
            sig_texts.append(SIG_MARKERS["fdr_05"])
        elif p < 0.05:
            sig_texts.append(SIG_MARKERS["nom_05"])
        else:
            sig_texts.append("")

    # Colour by -log10(gls_fdr)
    neg_log_fdr = _clamp_log10(df["gls_fdr"])
    norm = plt.Normalize(vmin=0, vmax=max(neg_log_fdr.max(), 1))
    cmap = plt.get_cmap("cividis")
    point_colors = [cmap(norm(v)) for v in neg_log_fdr]
    edge_colors = [ATC_LEVEL_COLORS.get(int(lv), COLOR_NONSIGNIFICANT) for lv in df["atc_level"]]

    n_rows = len(df)
    height = max(3.5, 0.4 * n_rows + 1.2)
    if figsize is None:
        figsize = (FIG_WIDTH_DOUBLE, height)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    y_positions = np.arange(n_rows)[::-1]

    if has_beta:
        betas = df["gls_beta"].values.astype(float)
        if has_se:
            ses = df["gls_se"].values.astype(float)
            xerr = 1.96 * ses
        else:
            xerr = None

        ax.errorbar(betas, y_positions, xerr=xerr, fmt="none",
                     ecolor="#999999", elinewidth=0.8, capsize=2, zorder=1)
        ax.scatter(betas, y_positions, c=point_colors, s=50,
                   edgecolors=edge_colors, linewidths=1.0, zorder=2)

        ax.axvline(0, color="#AAAAAA", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xlabel(r"GLS $\beta$ (± 1.96 SE)")
    else:
        neg_log_p = _clamp_log10(df["gls_p"])
        ax.scatter(neg_log_p, y_positions, c=point_colors, s=50,
                   edgecolors=edge_colors, linewidths=1.0, zorder=2)
        ax.set_xlabel(r"$-\log_{10}(p)$")

    # Significance markers right of each point
    x_max = ax.get_xlim()[1]
    for i, (yp, sig_text) in enumerate(zip(y_positions, sig_texts)):
        if sig_text:
            ax.text(x_max * 0.98 if has_beta else neg_log_fdr.max() + 0.3,
                    yp, sig_text, fontsize=FONT_SIZE_ANNOTATION,
                    va="center", ha="left", color=COLOR_SIGNIFICANT, fontweight="bold")

    # Warning flags
    if "annotation_bias_risk" in df.columns or "borderline_power" in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            flags = []
            if row.get("annotation_bias_risk", False):
                flags.append("\u26a0")
            if row.get("borderline_power", False):
                flags.append("\u25d0")
            if flags:
                ax.text(-0.02, y_positions[i], " ".join(flags),
                        transform=ax.get_yaxis_transform(),
                        fontsize=8, color="#888888", va="center", ha="right")

    # Separator lines between parent groups
    prev_parent = None
    for i, (_, row) in enumerate(df.iterrows()):
        parent = row["_parent_code"]
        if prev_parent is not None and parent != prev_parent:
            ax.axhline(y=y_positions[i] + 0.5, color="#EEEEEE", lw=0.5)
        prev_parent = parent

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["label"].values, fontsize=FONT_SIZE_TICK)
    ax.set_title(title or "ATC Class Enrichment")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    level_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#999999", markeredgecolor=ATC_LEVEL_COLORS.get(lv, COLOR_NONSIGNIFICANT),
               markersize=6, label=f"Level {lv}")
        for lv in sorted(df["_level"].unique())
    ]
    sig_legend_items = [
        Line2D([0], [0], marker="", color="none", label=f"{v} = FDR<{k.split('_')[1]}")
        for k, v in SIG_MARKERS.items() if k.startswith("fdr")
    ]
    all_handles = level_handles + sig_legend_items
    ax.legend(handles=all_handles, loc="lower right", fontsize=6, framealpha=0.8)

    # Stats block
    n_classes = len(atc_results)
    n_fdr = int((atc_results["gls_fdr"] < fdr_threshold).sum())
    top_row = df.iloc[0] if not df.empty else None
    top_label = f"{top_row['atc_code']}" if top_row is not None else "N/A"
    top_beta = f" (\u03b2={float(top_row['gls_beta']):.2f})" if top_row is not None and has_beta else ""
    condition_str = ""
    if meta and "condition_number" in meta:
        condition_str = f" | \u03ba(\u03a3) = {meta['condition_number']}"
    stats_text = f"N classes = {n_classes} | FDR<{fdr_threshold} = {n_fdr} | top = {top_label}{top_beta}{condition_str}"
    annotate_stats(ax, stats_text)

    if panel:
        panel_label(ax, panel)
    if meta:
        figure_footer(fig, meta)

    return fig
