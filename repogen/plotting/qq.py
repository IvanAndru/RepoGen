"""Quantile-quantile plot with confidence envelope for MAGMA gene-level p-values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

from repogen.plotting.base import (
    COLOR_THRESHOLD,
    COLOR_CI_BAND,
    COLOR_SIGNIFICANT,
    COLOR_NONSIGNIFICANT,
    FONT_SIZE_ANNOTATION,
    annotate_stats,
    figure_footer,
    panel_label,
    render_threshold_line,
)
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


def plot_qq(
    gene_results: pd.DataFrame | np.ndarray,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ci: float = 0.95,
    n_effective: int | None = None,
    split_mhc: bool = True,
    inset: bool = True,
    panel: str | None = None,
    meta: dict | None = None,
) -> Figure:
    """QQ plot with confidence bands, MHC split, lambda stats, and optional inset.

    Backward-compatible: accepts both a DataFrame (reads 'magma_p', optional 'in_mhc')
    and a plain 1D p-value array.
    """
    if isinstance(gene_results, pd.DataFrame):
        p = gene_results["magma_p"].values.astype(float)
        in_mhc = gene_results["in_mhc"].values if "in_mhc" in gene_results.columns else None
    else:
        p = np.asarray(gene_results, dtype=float)
        in_mhc = None

    valid = ~np.isnan(p)
    p = p[valid]
    if in_mhc is not None:
        in_mhc = in_mhc[valid]

    if len(p) == 0:
        raise ValueError("No valid p-values for QQ plot")

    p = np.maximum(p, np.finfo(float).tiny)
    p_sorted_idx = np.argsort(p)
    p_sorted = p[p_sorted_idx]
    if in_mhc is not None:
        mhc_sorted = in_mhc[p_sorted_idx]
    else:
        mhc_sorted = None

    n = len(p_sorted)
    ranks = np.arange(1, n + 1)
    expected = -np.log10((ranks - 0.5) / n)
    observed = -np.log10(p_sorted)

    # Lambda GC and lambda_1000
    chi2_vals = sp_stats.chi2.isf(np.clip(p, 1e-300, 1.0), 1)
    lambda_gc = float(np.median(chi2_vals) / 0.4549)
    lambda_1000 = None
    n_eff_val = n_effective
    if n_eff_val is None and meta and "n_effective" in meta:
        n_eff_val = meta["n_effective"]
    if n_eff_val is None and isinstance(gene_results, pd.DataFrame):
        for col in ("n_samples", "nsamp", "N"):
            if col in gene_results.columns:
                median_val = gene_results[col].dropna().median()
                if pd.notna(median_val) and median_val > 0:
                    n_eff_val = int(median_val)
                    break
    if n_eff_val is not None and n_eff_val > 0:
        lambda_1000 = 1 + (lambda_gc - 1) * (1000.0 / n_eff_val)

    if figsize is None:
        figsize = (4.2, 4.0)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # CI wedge
    if n >= 2:
        alpha_ci = (1 - ci) / 2
        ci_lower = -np.log10(sp_stats.beta.ppf(1 - alpha_ci, ranks, n - ranks + 1))
        ci_upper = -np.log10(
            np.maximum(sp_stats.beta.ppf(alpha_ci, ranks, n - ranks + 1), np.finfo(float).tiny)
        )
        ax.fill_between(expected, ci_lower, ci_upper, color=COLOR_CI_BAND, alpha=0.25, zorder=0)

    # Reference line
    x_max = float(expected.max()) + 0.5
    y_max = min(float(observed.max()) + 1, 20.0)
    diag_max = max(x_max, y_max) * 1.05
    ax.plot([0, diag_max], [0, diag_max], color=COLOR_THRESHOLD, linestyle="-",
            linewidth=0.8, zorder=1)

    # Scatter (MHC split)
    do_mhc_split = split_mhc and mhc_sorted is not None and np.any(mhc_sorted)

    if do_mhc_split:
        non_mhc_mask = ~mhc_sorted.astype(bool)
        mhc_mask = mhc_sorted.astype(bool)
        ax.scatter(expected[non_mhc_mask], observed[non_mhc_mask],
                   c=COLOR_NONSIGNIFICANT, s=6, linewidths=0, zorder=2, gid="dense_scatter")
        n_mhc_pts = int(mhc_mask.sum())
        ax.scatter(expected[mhc_mask], observed[mhc_mask],
                   c=COLOR_SIGNIFICANT, s=8, linewidths=0, zorder=3,
                   label=f"MHC (n={n_mhc_pts})")
    else:
        ax.scatter(expected, observed, c="#0072B2", s=6, linewidths=0, zorder=2,
                   gid="dense_scatter")

    # Genome-wide reference guideline
    gw_line = -np.log10(5e-8)
    if gw_line <= y_max:
        ax.axhline(gw_line, color="#DDDDDD", linestyle=":", linewidth=0.5, zorder=0)
        ax.axvline(gw_line, color="#DDDDDD", linestyle=":", linewidth=0.5, zorder=0)

    # Stats annotation
    stats_parts = [f"\u03bb_GC = {lambda_gc:.3f}"]
    if lambda_1000 is not None:
        stats_parts.append(f"\u03bb_1000 = {lambda_1000:.3f}")
    annotate_stats(ax, "\n".join(stats_parts), loc="upper left")

    # Inset
    inset_trigger = float(observed.max()) / max(float(expected.max()), 0.01) > 3
    if inset and inset_trigger:
        ax_ins = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
        ins_mask = (expected <= 5) & (observed <= 5)
        if do_mhc_split:
            non_mhc_ins = ins_mask & non_mhc_mask
            mhc_ins = ins_mask & mhc_mask
            ax_ins.scatter(expected[non_mhc_ins], observed[non_mhc_ins],
                           c=COLOR_NONSIGNIFICANT, s=3, linewidths=0)
            ax_ins.scatter(expected[mhc_ins], observed[mhc_ins],
                           c=COLOR_SIGNIFICANT, s=4, linewidths=0)
        else:
            ax_ins.scatter(expected[ins_mask], observed[ins_mask],
                           c="#0072B2", s=3, linewidths=0)
        ax_ins.plot([0, 5], [0, 5], color=COLOR_THRESHOLD, linestyle="-", linewidth=0.5)
        ax_ins.set_xlim(0, 5)
        ax_ins.set_ylim(0, 5)
        ax_ins.tick_params(labelsize=5)
        ax_ins.set_xlabel("")
        ax_ins.set_ylabel("")

    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title(title or "QQ Plot")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if do_mhc_split:
        handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_NONSIGNIFICANT, markersize=4, label="Non-MHC"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=COLOR_SIGNIFICANT, markersize=4,
                   label=f"MHC (n={int(mhc_mask.sum())})"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.8)

    if panel:
        panel_label(ax, panel)
    if meta:
        figure_footer(fig, meta)

    return fig
