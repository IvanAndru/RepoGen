"""Gene-level Manhattan plot and gene volcano plot for MAGMA gene analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from adjustText import adjust_text

from repogen.plotting.base import (
    FIGSIZE_WIDE_SHORT,
    FIGSIZE_SQUARE_LARGE,
    COLOR_SIGNIFICANT,
    COLOR_NONSIGNIFICANT,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MEDIUM,
    FONT_SIZE_ANNOTATION,
    render_threshold_line,
    fdr_implied_p_cutoff,
    bonferroni_cutoff,
    figure_footer,
    panel_label,
    annotate_stats,
    collapse_loci,
    _dot_size_legend,
    chromosome_colors,
)
from repogen.utils.logging import setup_logging

if TYPE_CHECKING:
    from repogen.config.schema import ManhattanStyleConfig, VolcanoStyleConfig

logger = setup_logging(__name__)

_CHR_ORDER: dict[str, int] = {
    **{str(i): i for i in range(1, 23)},
    "X": 23, "Y": 24, "MT": 25, "M": 25,
}

_CHR_FILTER: dict[str, set[str]] = {
    "autosomes": {str(i) for i in range(1, 23)},
    "autosomes+X": {str(i) for i in range(1, 23)} | {"X"},
    "all": {str(i) for i in range(1, 23)} | {"X", "Y", "MT", "M"},
}

_MHC_CHR, _MHC_START, _MHC_END = "6", 25_000_000, 35_000_000


def _normalize_chr(label: object) -> str:
    """Normalize a chromosome label to a canonical string (e.g. '1', 'X')."""
    s = str(label).strip().upper()
    if s.startswith("CHR"):
        s = s[3:]
    return s


def _chr_sort_key(label: str) -> int:
    """Return a numeric sort key for canonical chromosome labels."""
    return _CHR_ORDER.get(label, 100)


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    """Raise ValueError if df is empty or missing required columns."""
    if df.empty:
        raise ValueError(f"{name} is empty - nothing to plot")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {name}: {missing}")


# ---------------------------------------------------------------------------
# Manhattan plot (§3)
# ---------------------------------------------------------------------------

def plot_manhattan(
    gene_results: pd.DataFrame,
    *,
    style: ManhattanStyleConfig | None = None,
    fdr_threshold: float = 0.05,
    suggestive_threshold: float = 1e-4,
    genome_wide_threshold: float = 5e-8,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    panel: str | None = None,
    meta: dict | None = None,
) -> Figure:
    """Gene-level Manhattan plot with rank-scale chromosomes and MHC handling."""
    from repogen.config.schema import ManhattanStyleConfig as _MSC
    if style is None:
        style = _MSC()

    required = {"gene_symbol", "chr", "start", "magma_p", "fdr_q"}
    _validate_columns(gene_results, required, "gene_results")

    df = gene_results.copy()
    df["_chr_label"] = df["chr"].map(_normalize_chr)
    df["_chr_order"] = df["_chr_label"].map(_chr_sort_key)

    allowed_chrs = _CHR_FILTER.get(style.chromosome_set, _CHR_FILTER["autosomes+X"])
    df = df[df["_chr_label"].isin(allowed_chrs)].copy()
    if df.empty:
        raise ValueError("No genes remain after chromosome filtering")

    df = df.sort_values(["_chr_order", "start"]).reset_index(drop=True)
    df["neg_log_p"] = -np.log10(np.maximum(df["magma_p"].values, np.finfo(float).tiny))

    if "in_mhc" not in df.columns:
        df["in_mhc"] = (
            (df["_chr_label"] == _MHC_CHR)
            & (df["start"] >= _MHC_START)
            & (df["start"] <= _MHC_END)
        )

    chromosomes = sorted(df["_chr_label"].unique(), key=_chr_sort_key)
    chr_color_list = chromosome_colors(len(chromosomes))

    # Chromosome layout
    use_rank = style.chromosome_scale == "rank"
    cumulative_offset = 0.0
    chr_offsets: dict[str, float] = {}
    chr_widths: dict[str, float] = {}
    chr_centers: dict[str, float] = {}
    chr_min_start: dict[str, int] = {}

    if use_rank:
        gene_counts = df["_chr_label"].value_counts()
        total_genes = gene_counts.sum()
        gap_total = 0.02 * len(chromosomes)
        usable = 1.0 - gap_total
        for chrom in chromosomes:
            n = gene_counts.get(chrom, 1)
            width = usable * n / total_genes
            chr_offsets[chrom] = cumulative_offset
            chr_widths[chrom] = width
            chr_centers[chrom] = cumulative_offset + width / 2
            cumulative_offset += width + 0.02
        ranks = df.groupby("_chr_label").cumcount()
        counts = df["_chr_label"].map(gene_counts)
        df["x_pos"] = (
            df["_chr_label"].map(chr_offsets)
            + (ranks / counts.clip(lower=1)) * df["_chr_label"].map(chr_widths)
        )
    else:
        for chrom in chromosomes:
            chr_offsets[chrom] = cumulative_offset
            chr_df = df[df["_chr_label"] == chrom]
            start_min = chr_df["start"].min()
            start_max = chr_df["start"].max()
            chr_min_start[chrom] = start_min
            chr_span = max(start_max - start_min + 1, 1)
            chr_widths[chrom] = chr_span
            chr_centers[chrom] = cumulative_offset + chr_span / 2
            cumulative_offset += chr_span + chr_span * 0.05
        df["x_pos"] = (
            df["start"]
            - df["_chr_label"].map(chr_min_start)
            + df["_chr_label"].map(chr_offsets)
        )

    total_width = cumulative_offset

    is_sig = df["fdr_q"] < fdr_threshold

    if figsize is None:
        figsize = FIGSIZE_WIDE_SHORT
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # MHC band
    has_mhc = df["in_mhc"].any()
    if has_mhc and _MHC_CHR in chr_offsets:
        if use_rank:
            mhc_genes = df[df["in_mhc"]]
            if not mhc_genes.empty:
                x0 = mhc_genes["x_pos"].min() - 0.002
                x1 = mhc_genes["x_pos"].max() + 0.002
                ax.axvspan(x0, x1, color="#FFD700", alpha=0.15, zorder=0)
        else:
            off = chr_offsets[_MHC_CHR]
            mn = chr_min_start.get(_MHC_CHR, 0)
            x0 = off + _MHC_START - mn
            x1 = off + _MHC_END - mn
            ax.axvspan(x0, x1, color="#FFD700", alpha=0.15, zorder=0)

    # Scatter layers
    for idx, chrom in enumerate(chromosomes):
        mask_chr = df["_chr_label"] == chrom
        chr_color = chr_color_list[idx % len(chr_color_list)]

        nonsig_nomhc = mask_chr & ~is_sig & ~df["in_mhc"]
        if nonsig_nomhc.any():
            sub = df[nonsig_nomhc]
            ax.scatter(sub["x_pos"].values, sub["neg_log_p"].values,
                       c=chr_color, s=6, marker="o", linewidths=0, zorder=1,
                       gid="dense_scatter")

        nonsig_mhc = mask_chr & ~is_sig & df["in_mhc"]
        if nonsig_mhc.any():
            sub = df[nonsig_mhc]
            ax.scatter(sub["x_pos"].values, sub["neg_log_p"].values,
                       facecolors="none", edgecolors=chr_color, s=6,
                       marker="^", linewidths=0.8, zorder=1, gid="dense_scatter")

        sig_nomhc = mask_chr & is_sig & ~df["in_mhc"]
        if sig_nomhc.any():
            sub = df[sig_nomhc]
            ax.scatter(sub["x_pos"].values, sub["neg_log_p"].values,
                       c=COLOR_SIGNIFICANT, s=18, marker="o", linewidths=0, zorder=3)

        sig_mhc = mask_chr & is_sig & df["in_mhc"]
        if sig_mhc.any():
            sub = df[sig_mhc]
            ax.scatter(sub["x_pos"].values, sub["neg_log_p"].values,
                       facecolors="none", edgecolors=COLOR_SIGNIFICANT, s=18,
                       marker="^", linewidths=0.8, zorder=3)

    # Y-cap
    y_cap = style.y_cap
    n_clipped = 0
    if y_cap is not None:
        clipped_mask = df["neg_log_p"] > y_cap
        n_clipped = int(clipped_mask.sum())
        if n_clipped > 0:
            clipped = df[clipped_mask]
            ax.scatter(clipped["x_pos"].values,
                       np.full(n_clipped, y_cap), c=COLOR_SIGNIFICANT,
                       s=30, marker="^", linewidths=0, zorder=4)
            ax.set_ylim(-0.3, y_cap + 0.5)
        else:
            ax.set_ylim(-0.3, df["neg_log_p"].max() + 1)
    else:
        ax.set_ylim(-0.3, df["neg_log_p"].max() + 1)

    # Threshold lines
    n_genes = len(df)
    bonf_p = bonferroni_cutoff(n_genes)
    render_threshold_line(ax, -np.log10(bonf_p), "Bonferroni",
                          color="#555555", linestyle="--", position="gutter")
    render_threshold_line(ax, -np.log10(genome_wide_threshold), "GWAS 5e-8",
                          color="#000000", linestyle="-", position="gutter")

    p_cut = fdr_implied_p_cutoff(df["magma_p"].values, df["fdr_q"].values, fdr_threshold)
    if p_cut is not None:
        render_threshold_line(ax, -np.log10(p_cut),
                              f"FDR {fdr_threshold} (p\u2264{p_cut:.2g})",
                              color=COLOR_SIGNIFICANT, position="gutter")

    if suggestive_threshold:
        render_threshold_line(ax, -np.log10(suggestive_threshold),
                              f"p={suggestive_threshold:.0e}",
                              color=COLOR_NEUTRAL_MEDIUM, linestyle=":", position="gutter")

    # Labels
    n_labels = style.n_top_labels
    if n_labels > 0:
        sig_df = df[is_sig].copy()
        if not sig_df.empty:
            collapsed = collapse_loci(sig_df, window_kb=style.locus_window_kb)
            top = collapsed.nsmallest(n_labels, "magma_p")
        else:
            top = df.nsmallest(min(n_labels, 5), "magma_p")

        if style.highlight_genes:
            hl = df[df["gene_symbol"].isin(style.highlight_genes)]
            top = pd.concat([top, hl]).drop_duplicates(subset=["gene_symbol"])

        y_plot = top["neg_log_p"].clip(upper=y_cap) if y_cap else top["neg_log_p"]
        texts = [
            ax.text(row.x_pos, yp, row.gene_symbol,
                    fontsize=FONT_SIZE_ANNOTATION, ha="center", va="bottom",
                    fontweight="bold" if row.gene_symbol in (style.highlight_genes or []) else "normal")
            for row, yp in zip(top.itertuples(), y_plot)
        ]
        if style.highlight_genes:
            for row, yp in zip(top.itertuples(), y_plot):
                if row.gene_symbol in style.highlight_genes:
                    ax.scatter([row.x_pos], [yp], s=60, facecolors="none",
                               edgecolors=COLOR_SIGNIFICANT, linewidths=1.2, zorder=5)
        if texts:
            adjust_text(texts, ax=ax, force_text=(0.5, 0.8), expand_points=(1.2, 1.4),
                        arrowprops=dict(arrowstyle="-", color=COLOR_NEUTRAL_MEDIUM, lw=0.5))

    # Axis furniture
    ax.set_xticks([chr_centers[c] for c in chromosomes])
    ax.set_xticklabels(chromosomes)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title or "Gene-level Manhattan Plot")
    ax.set_xlim(-0.01 if use_rank else 0, total_width)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = []
    handles.append(Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=COLOR_SIGNIFICANT, markersize=5,
                          label=f"FDR < {fdr_threshold}"))
    handles.append(Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=COLOR_NONSIGNIFICANT, markersize=4,
                          label="Non-significant"))
    if has_mhc:
        handles.append(Line2D([0], [0], marker="^", color="none",
                              markerfacecolor="none", markeredgecolor=COLOR_NEUTRAL_MEDIUM,
                              markersize=5, label="MHC gene"))
    ax.legend(handles=handles, loc="upper left", fontsize=6, framealpha=0.8)

    # Stats block
    n_bonf = int((df["magma_p"] < bonf_p).sum())
    n_fdr = int(is_sig.sum())
    chi2_vals = np.clip(-2 * np.log(np.maximum(df["magma_p"].values, 1e-300)), 0, None)
    lambda_gc = float(np.median(chi2_vals) / 0.4549) if len(chi2_vals) > 0 else 0
    stats_text = (
        f"N genes = {n_genes:,}\n"
        f"Bonferroni hits = {n_bonf}\n"
        f"FDR q < {fdr_threshold} = {n_fdr}\n"
        f"\u03bb_GC = {lambda_gc:.3f}"
    )
    if n_clipped > 0:
        stats_text += f"\n+{n_clipped} genes > {y_cap}"
    annotate_stats(ax, stats_text)

    if panel:
        panel_label(ax, panel)
    if meta:
        figure_footer(fig, meta)

    return fig


# ---------------------------------------------------------------------------
# Gene volcano - simple variant (supplement, §5.1)
# ---------------------------------------------------------------------------

def plot_gene_volcano_simple(
    gene_results: pd.DataFrame,
    n_top_labels: int = 20,
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Raw signed-Z volcano -- for MAGMA supplements."""
    required = {"gene_symbol", "magma_z", "magma_p", "fdr_q"}
    _validate_columns(gene_results, required, "gene_results")

    df = gene_results.copy()
    df["neg_log_p"] = -np.log10(np.maximum(df["magma_p"].values, np.finfo(float).tiny))
    is_sig = df["fdr_q"] < fdr_threshold
    colors = np.where(is_sig, COLOR_SIGNIFICANT, COLOR_NONSIGNIFICANT)

    if figsize is None:
        figsize = FIGSIZE_SQUARE_LARGE
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.scatter(df["magma_z"].values, df["neg_log_p"].values,
               c=colors, s=np.where(is_sig, 16, 8), linewidths=0, zorder=2,
               gid="dense_scatter")

    sig_pvals = df.loc[is_sig, "magma_p"]
    if not sig_pvals.empty:
        boundary_p = sig_pvals.max()
        render_threshold_line(ax, -np.log10(boundary_p), f"FDR = {fdr_threshold}",
                              position="gutter")

    if n_top_labels > 0:
        top = df.nsmallest(n_top_labels, "magma_p")
        texts = [
            ax.text(row.magma_z, row.neg_log_p, row.gene_symbol,
                    fontsize=FONT_SIZE_ANNOTATION, ha="center", va="bottom")
            for row in top.itertuples()
        ]
        if texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color=COLOR_NEUTRAL_MEDIUM, lw=0.5))

    ax.set_xlabel("MAGMA Z-score")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title or "Gene Volcano Plot")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


# ---------------------------------------------------------------------------
# Gene volcano - publication variant (§5.2–5.9)
# ---------------------------------------------------------------------------

def plot_gene_volcano(
    gene_results: pd.DataFrame,
    *,
    style: VolcanoStyleConfig | None = None,
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    sign_lookup: dict[int, float] | None = None,
    panel: str | None = None,
    meta: dict | None = None,
) -> Figure:
    """Gene-level volcano (x-axis depends on style.x_axis_mode, y = -log10 p).

    Default mode 'magma_z' plots raw signed Z with a 2-colour FDR-only scheme.
    Opt-in mode 'top_snp_beta' requires sign_lookup (Entrez-keyed) and renders
    a four-quadrant up/down colour scheme.
    """
    from repogen.config.schema import VolcanoStyleConfig as _VSC
    if style is None:
        style = _VSC()

    required = {"gene_symbol", "magma_z", "magma_p", "fdr_q"}
    _validate_columns(gene_results, required, "gene_results")

    df = gene_results.copy()
    df["neg_log_p"] = -np.log10(np.maximum(df["magma_p"].values, np.finfo(float).tiny))

    is_sig = df["fdr_q"] < fdr_threshold
    n_genes = len(df)

    # X-axis
    mode = style.x_axis_mode
    if mode == "top_snp_beta" and sign_lookup:
        if "gene_entrez_id" in df.columns:
            signs = df["gene_entrez_id"].map(sign_lookup).fillna(+1.0)
        else:
            signs = pd.Series(+1.0, index=df.index)
            logger.debug("gene_entrez_id not in columns; defaulting all signs to +1")
        x = df["magma_z"].abs().values * signs.values
        x_label = r"Signed MAGMA $|Z|$ $\times$ sign(top-SNP $\beta$)"
        subtitle = "Effect direction: top-SNP \u03b2 from prepared GWAS"
    else:
        if mode == "top_snp_beta" and not sign_lookup:
            logger.debug("top_snp_beta mode requested but sign_lookup empty; falling back to raw magma_z")
        x = df["magma_z"].values
        x_label = "MAGMA Z (snp-wise=mean)"
        subtitle = "Sign of Z reflects p<0.5 vs p>0.5, not effect direction"

    df["signed_x"] = x

    # Size encoding
    n_snps = df["n_snps"].values if "n_snps" in df.columns else np.ones(n_genes)
    sizes = 10 + 40 * np.sqrt(np.clip(n_snps, 1, 2000) / 2000)

    # Colour scheme
    if mode == "top_snp_beta" and sign_lookup:
        colors = np.full(n_genes, COLOR_NONSIGNIFICANT, dtype=object)
        colors[is_sig & (x > 0)] = COLOR_SIGNIFICANT
        colors[is_sig & (x < 0)] = "#0072B2"
        nonsig_large = ~is_sig & (np.abs(x) >= 2)
        colors[nonsig_large] = COLOR_NEUTRAL_MEDIUM
        alphas = np.where(is_sig, 1.0, np.where(nonsig_large, 0.6, 0.5))
    else:
        colors = np.where(is_sig, COLOR_SIGNIFICANT, COLOR_NONSIGNIFICANT)
        alphas = np.where(is_sig, 1.0, 0.5)

    if figsize is None:
        figsize = FIGSIZE_SQUARE_LARGE
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.scatter(x, df["neg_log_p"].values, c=colors, s=sizes,
               alpha=alphas, linewidths=0, zorder=2, gid="dense_scatter")

    # MHC overlay
    in_mhc = df.get("in_mhc", pd.Series(False, index=df.index))
    if in_mhc.any():
        mhc = df[in_mhc]
        ax.scatter(mhc["signed_x"].values, mhc["neg_log_p"].values,
                   facecolors="none", edgecolors=colors[in_mhc.values] if hasattr(colors, '__getitem__') else COLOR_SIGNIFICANT,
                   s=sizes[in_mhc.values] * 1.5, marker="^", linewidths=0.8, zorder=4)

    # Y-cap
    y_cap = style.y_cap
    if y_cap is not None:
        clipped = df["neg_log_p"] > y_cap
        if clipped.any():
            ax.scatter(df.loc[clipped, "signed_x"].values,
                       np.full(clipped.sum(), y_cap),
                       c=COLOR_SIGNIFICANT, s=30, marker="^", linewidths=0, zorder=5)
            ax.set_ylim(-0.3, y_cap + 0.5)
        else:
            ax.set_ylim(-0.3, df["neg_log_p"].max() + 1)
    else:
        ax.set_ylim(-0.3, df["neg_log_p"].max() + 1)

    # Axes furniture
    ax.axvline(0, color=COLOR_NEUTRAL_LIGHT, linestyle="--", linewidth=0.8, zorder=0)

    p_cut = fdr_implied_p_cutoff(df["magma_p"].values, df["fdr_q"].values, fdr_threshold)
    if p_cut is not None:
        render_threshold_line(ax, -np.log10(p_cut),
                              f"FDR {fdr_threshold} (p\u2264{p_cut:.2g})",
                              orientation="horizontal", position="gutter")
    else:
        bonf = bonferroni_cutoff(n_genes)
        render_threshold_line(ax, -np.log10(bonf), "Bonferroni (no FDR hits)",
                              orientation="horizontal", position="gutter")

    # Labels
    n_labels = style.n_top_labels
    if n_labels > 0:
        cands = df[is_sig].copy() if is_sig.any() else df.copy()
        cands["_abs_x"] = cands["signed_x"].abs()
        top = cands.nlargest(n_labels, "_abs_x")
        y_plot = top["neg_log_p"].clip(upper=y_cap) if y_cap else top["neg_log_p"]
        texts = [
            ax.text(row.signed_x, yp, row.gene_symbol,
                    fontsize=FONT_SIZE_ANNOTATION, ha="center", va="bottom")
            for row, yp in zip(top.itertuples(), y_plot)
        ]
        if texts:
            adjust_text(texts, ax=ax, force_text=(0.5, 0.8), expand_points=(1.2, 1.4),
                        arrowprops=dict(arrowstyle="-", color=COLOR_NEUTRAL_MEDIUM, lw=0.5))

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    vol_title = title or "Gene Volcano Plot"
    ax.set_title(vol_title, fontsize=10)
    ax.text(0.5, -0.02, subtitle, transform=ax.transAxes, fontsize=6,
            ha="center", va="top", color="#666666", style="italic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Stats block
    n_sig = int(is_sig.sum())
    n_mhc = int(in_mhc.sum()) if in_mhc.any() else 0
    if mode == "top_snp_beta" and sign_lookup:
        n_up = int((is_sig & (x > 0)).sum())
        n_down = int((is_sig & (x < 0)).sum())
        stats_text = f"n_sig_up = {n_up}   n_sig_down = {n_down}   n_MHC = {n_mhc}"
    else:
        stats_text = f"n_sig = {n_sig}   n_MHC = {n_mhc}"
    annotate_stats(ax, stats_text)

    # Size legend
    if "n_snps" in df.columns:
        _dot_size_legend(ax, sizes, n_snps, "N SNPs", loc="lower right")

    if panel:
        panel_label(ax, panel)
    if meta:
        figure_footer(fig, meta)

    return fig
