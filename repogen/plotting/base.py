"""Shared theme constants, colour palettes, typography, and utility functions for RepoGen plots."""

from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from repogen.utils.logging import setup_logging

if TYPE_CHECKING:
    from repogen.config.schema import PlotStyleConfig

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

"""Okabe-Ito colorblind-safe categorical palette (Wong, Nature Methods 2011)."""
PALETTE_CATEGORICAL: list[str] = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
]

COLOR_BRANCH_A: str = "#E69F00"  # MAGMA pipeline (Orange)
COLOR_BRANCH_B: str = "#56B4E9"  # Negative correlation pipeline (Sky Blue)
COLOR_BRANCH_C: str = "#009E73"  # Mendelian randomisation (Bluish Green)

PALETTE_SEQUENTIAL: str = "viridis"
PALETTE_SEQUENTIAL_REVERSED: str = "viridis_r"
PALETTE_DIVERGING: str = "RdBu_r"

COLOR_SIGNIFICANT: str = "#D55E00"
COLOR_NONSIGNIFICANT: str = "#999999"
COLOR_THRESHOLD: str = "#333333"

COLOR_CI_BAND: str = "#CCCCCC"
COLOR_ZERO_LINE: str = "#AAAAAA"
COLOR_NEUTRAL_LIGHT: str = "#DDDDDD"
COLOR_NEUTRAL_MEDIUM: str = "#666666"
COLOR_SECONDARY_TEXT: str = "#444444"

ATC_LEVEL_COLORS: dict[int, str] = {
    2: "#56B4E9",  # Sky Blue (lighter)
    3: "#0072B2",  # Blue (darker)
}

PHASE_COLORS: dict[int, str] = {
    0: "#999999",  # Preclinical / unknown
    1: "#BDD7E7",  # Phase 1
    2: "#6BAED6",  # Phase 2
    3: "#2171B5",  # Phase 3
    4: "#D55E00",  # Marketed / Phase 4
}

TIER_COLORS: dict[str, str] = {
    "high": "#009E73",
    "medium": "#0072B2",
    "low": "#999999",
    "direction_conflict": "#D55E00",
}

COLOC_COLORS: dict[str, str] = {
    "colocalised": "#009E73",
    "distinct_signals": "#E69F00",
    "insufficient_data": "#999999",
    "unsupported": "#CC79A7",
}

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY: str = "Arial"
FONT_SIZE_TITLE: int = 12
FONT_SIZE_AXIS_LABEL: int = 10
FONT_SIZE_TICK: int = 8
FONT_SIZE_ANNOTATION: int = 7
FONT_SIZE_LEGEND: int = 8
FONT_SIZE_PANEL_LABEL: int = 10
FONT_SIZE_MINI: int = 5

FONT_SIZES: dict[str, int] = {
    "title": FONT_SIZE_TITLE,
    "axis_label": FONT_SIZE_AXIS_LABEL,
    "tick": FONT_SIZE_TICK,
    "annotation": FONT_SIZE_ANNOTATION,
    "legend": FONT_SIZE_LEGEND,
    "panel_label": FONT_SIZE_PANEL_LABEL,
    "mini": FONT_SIZE_MINI,
}

# ---------------------------------------------------------------------------
# Figure dimensions
# ---------------------------------------------------------------------------

FIG_WIDTH_SINGLE: float = 3.5
FIG_WIDTH_ONEANDAHALF: float = 5.3
FIG_WIDTH_DOUBLE: float = 7.2

FIG_HEIGHT_DEFAULT: float = 4.5
FIG_HEIGHT_TALL: float = 8.0

FIGSIZE_SINGLE: tuple[float, float] = (FIG_WIDTH_SINGLE, 2.5)
FIGSIZE_STANDARD: tuple[float, float] = (FIG_WIDTH_DOUBLE, 4.5)
FIGSIZE_TALL: tuple[float, float] = (FIG_WIDTH_DOUBLE, 8.0)
FIGSIZE_SQUARE: tuple[float, float] = (FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE)
FIGSIZE_SQUARE_LARGE: tuple[float, float] = (4.0, 4.0)
FIGSIZE_WIDE_SHORT: tuple[float, float] = (7.2, 3.3)
FIGSIZE_FOREST: tuple[float, float] = (6.8, 0.35)

SIG_MARKERS: dict[str, str] = {
    "fdr_001": "***",
    "fdr_01":  "**",
    "fdr_05":  "*",
    "nom_05":  "\u2020",
}

# ---------------------------------------------------------------------------
# DPI and output
# ---------------------------------------------------------------------------

DPI_SCREEN: int = 150
DPI_PUBLICATION: int = 300
DPI_HIGH: int = 600
DEFAULT_FORMATS: list[str] = ["png", "pdf"]

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

RCPARAMS: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT_SIZE_TICK,
    "axes.labelsize": FONT_SIZE_AXIS_LABEL,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.linewidth": 0.8,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": FONT_SIZE_LEGEND,
    "legend.frameon": False,
    "figure.dpi": DPI_SCREEN,
    "savefig.dpi": DPI_PUBLICATION,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def apply_theme() -> None:
    """Set matplotlib rcParams for RepoGen publication style.

    Call once at the start of a plotting session or at module import.
    Modifies matplotlib.rcParams in place.
    """
    plt.rcParams.update(RCPARAMS)


def save_figure(
    fig: Figure,
    output_path: Path,
    *,
    style: PlotStyleConfig | None = None,
    rasterize_scatter: bool = False,
    formats: list[str] | None = None,
    dpi: int | None = None,
    tight: bool = True,
) -> list[Path]:
    """Save figure in one or more formats.

    Resolution order for formats:
        1. explicit ``formats=`` (legacy kwarg) -- highest priority (tests use this)
        2. ``style.figure_formats`` when ``style`` provided
        3. module constant ``DEFAULT_FORMATS`` (= ["pdf", "png"]) -- unchanged

    Resolution order for DPI:
        1. explicit ``dpi=`` -- wins
        2. ``style.png_dpi`` for PNG / ``style.pdf_dpi`` for PDF/SVG -- per-format
        3. ``DPI_PUBLICATION`` -- unchanged legacy default

    ``rasterize_scatter=True`` finds every artist tagged ``set_gid("dense_scatter")``
    in ``fig.axes[*].collections`` and calls ``.set_rasterized(True)`` before save,
    so PDF/SVG keep vector text + axes while compressing tens of thousands of points.
    """
    if rasterize_scatter:
        for ax in fig.axes:
            for coll in ax.collections:
                if coll.get_gid() == "dense_scatter":
                    coll.set_rasterized(True)

    fmt_list: list[str]
    if formats is not None:
        fmt_list = formats
    elif style is not None:
        fmt_list = list(style.figure_formats)
    else:
        fmt_list = DEFAULT_FORMATS

    output_path = Path(output_path)
    stem_path = output_path.with_suffix("")
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for fmt in fmt_list:
        if dpi is not None:
            fmt_dpi = dpi
        elif style is not None:
            fmt_dpi = style.png_dpi if fmt == "png" else style.pdf_dpi
        else:
            fmt_dpi = DPI_PUBLICATION

        path = stem_path.with_suffix(f".{fmt}")
        fig.savefig(
            path,
            dpi=fmt_dpi,
            bbox_inches="tight" if tight else None,
            facecolor="white",
            edgecolor="none",
        )
        logger.info("Saved figure: %s", path)
        saved.append(path)

    return saved


def format_pvalue(p: float, threshold: float = 1e-300) -> str:
    """Format p-value for display on plots.

    Args:
        p: The p-value to format.
        threshold: Below this, display as "p < {threshold}".

    Returns:
        Formatted string.

    Example:
        >>> format_pvalue(3.2e-8)
        'p = 3.2×10⁻⁸'
    """
    if p <= 0 or p < threshold:
        exp = int(math.floor(math.log10(threshold)))
        return f"p < 1×10{_superscript(exp)}"
    if p >= 0.01:
        return f"p = {p:.3f}"
    exp = int(math.floor(math.log10(p)))
    mantissa = p / (10**exp)
    return f"p = {mantissa:.1f}×10{_superscript(exp)}"


def _superscript(n: int) -> str:
    """Convert an integer to Unicode superscript characters."""
    sup_map = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(n).translate(sup_map)


def truncate_label(label: str, max_len: int = 40) -> str:
    """Truncate long pathway/drug names with ellipsis.

    Args:
        label: The label string to truncate.
        max_len: Maximum character length before truncation.

    Returns:
        Original label if short enough, or truncated with "\u2026".

    Example:
        >>> truncate_label("GO_REGULATION_OF_SYNAPTIC_TRANSMISSION", 40)
        'GO_REGULATION_OF_SYNAPTIC_TRANSMISSION'
    """
    label = str(label) if not isinstance(label, str) else label
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "\u2026"


def add_significance_line(
    ax: Axes,
    threshold: float,
    label: str | None = None,
    orientation: str = "horizontal",
    color: str = COLOR_THRESHOLD,
) -> None:
    """Add a dashed significance threshold line to an axis.

    Args:
        ax: The matplotlib Axes to draw on.
        threshold: The value at which to draw the line.
        label: Optional text label for the line.
        orientation: "horizontal" or "vertical".
        color: Line colour.

    Example:
        >>> add_significance_line(ax, -np.log10(0.05), "FDR = 0.05")
    """
    kwargs: dict = dict(color=color, linestyle="--", linewidth=0.8, zorder=1)
    if orientation == "horizontal":
        ax.axhline(threshold, **kwargs)
    elif orientation == "vertical":
        ax.axvline(threshold, **kwargs)
    else:
        raise ValueError(f"orientation must be 'horizontal' or 'vertical', got '{orientation}'")

    if label is not None:
        if orientation == "horizontal":
            ax.text(
                ax.get_xlim()[1],
                threshold,
                f" {label}",
                ha="left",
                va="bottom",
                fontsize=FONT_SIZE_ANNOTATION,
                color=color,
            )
        else:
            ax.text(
                threshold,
                ax.get_ylim()[1],
                f" {label}",
                ha="left",
                va="top",
                fontsize=FONT_SIZE_ANNOTATION,
                color=color,
                rotation=90,
            )


def chromosome_colors(n_chr: int = 22) -> list[str]:
    """Return alternating grey/dark-grey colours for chromosome Manhattan layout.

    Args:
        n_chr: Number of chromosomes (default 22 autosomes).

    Returns:
        List of hex colour strings, alternating between two greys.
    """
    light, dark = "#AAAAAA", "#666666"
    return [light if i % 2 == 0 else dark for i in range(n_chr)]


# ---------------------------------------------------------------------------
# Publication helper functions
# ---------------------------------------------------------------------------

def render_threshold_line(
    ax: Axes,
    value: float,
    label: str,
    *,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    color: str = COLOR_THRESHOLD,
    position: Literal["gutter", "inside", "above"] = "gutter",
    linestyle: str = "--",
) -> None:
    """Dashed threshold line with a *non-occluding* label.

    'gutter':  label placed outside the axis (right for horizontal, top for vertical).
    'above':   label rotated above the top axis.
    'inside':  legacy behaviour -- label inside plotting area.
    """
    lw = 0.8
    if orientation == "horizontal":
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=lw, zorder=1)
    else:
        ax.axvline(value, color=color, linestyle=linestyle, linewidth=lw, zorder=1)

    if not label:
        return

    fs = FONT_SIZE_ANNOTATION
    if position == "gutter":
        if orientation == "horizontal":
            ax.annotate(
                label, xy=(1.01, value), xycoords=("axes fraction", "data"),
                fontsize=fs, color=color, va="center", ha="left",
                annotation_clip=False,
            )
        else:
            ax.annotate(
                label, xy=(value, 1.02), xycoords=("data", "axes fraction"),
                fontsize=fs, color=color, va="bottom", ha="center",
                rotation=90, annotation_clip=False,
            )
    elif position == "above":
        ax.annotate(
            label,
            xy=(value, 1.02) if orientation == "vertical" else (1.01, value),
            xycoords=("data", "axes fraction") if orientation == "vertical"
                     else ("axes fraction", "data"),
            fontsize=fs, color=color, va="bottom", ha="center",
            rotation=90 if orientation == "vertical" else 0,
            annotation_clip=False,
        )
    else:
        if orientation == "horizontal":
            ax.text(
                ax.get_xlim()[1], value, f" {label}",
                ha="left", va="bottom", fontsize=fs, color=color,
            )
        else:
            ax.text(
                value, ax.get_ylim()[1], f" {label}",
                ha="left", va="top", fontsize=fs, color=color, rotation=90,
            )


def fdr_implied_p_cutoff(
    p: np.ndarray,
    fdr_q: np.ndarray,
    alpha: float = 0.05,
) -> float | None:
    """Return the largest p-value whose Benjamini-Hochberg q is < alpha.

    Use this when the plot x-axis is -log10(p) but the significance threshold
    is an FDR threshold. Drawing a vertical at -log10(p*) correctly separates
    FDR-passing from non-passing points on a p-axis plot.

    Returns None if no row passes FDR < alpha.
    """
    mask = fdr_q < alpha
    if not np.any(mask):
        return None
    return float(np.max(p[mask]))


def bonferroni_cutoff(n_tests: int, alpha: float = 0.05) -> float:
    """Trivial helper: alpha / n_tests."""
    return alpha / max(n_tests, 1)


def figure_footer(fig: Figure, meta: dict) -> None:
    """Provenance stamp at bottom-right (5pt, #888888)."""
    parts = ["RepoGen"]
    if "study" in meta:
        parts.append(f"study={meta['study']}")
    if "n_genes" in meta:
        parts.append(f"n_genes={meta['n_genes']:,}")
    parts.append(str(date.today()))
    text = " | ".join(parts)
    fig.text(0.99, 0.005, text, fontsize=5, color="#888888",
             ha="right", va="bottom", transform=fig.transFigure)


def panel_label(ax: Axes, letter: str) -> None:
    """Bold panel label at (-0.12, 1.02) axes coords, 10pt."""
    ax.text(
        -0.12, 1.02, letter,
        transform=ax.transAxes, fontsize=FONT_SIZE_PANEL_LABEL,
        fontweight="bold", va="bottom", ha="right",
    )


def annotate_stats(ax: Axes, text: str, *, loc: str = "upper right") -> None:
    """Framed multi-line stats block (white bbox, alpha=0.85, 7pt)."""
    x_map = {"upper right": 0.98, "upper left": 0.02, "lower right": 0.98, "lower left": 0.02}
    y_map = {"upper right": 0.97, "upper left": 0.97, "lower right": 0.03, "lower left": 0.03}
    ha_map = {"upper right": "right", "upper left": "left", "lower right": "right", "lower left": "left"}
    va_map = {"upper right": "top", "upper left": "top", "lower right": "bottom", "lower left": "bottom"}
    ax.text(
        x_map.get(loc, 0.98), y_map.get(loc, 0.97), text,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_ANNOTATION, fontfamily="monospace",
        ha=ha_map.get(loc, "right"), va=va_map.get(loc, "top"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )


_PATHWAY_STOP_WORDS = {"of", "to", "via", "in", "on", "by", "the", "and", "for", "with", "a", "an"}
_BIO_ACRONYMS = {"DNA", "RNA", "ATP", "GTP", "mRNA", "MAPK", "GABA", "NMDA", "GPCR", "CNS", "ADHD"}
_DRUG_PREFIX_RE = re.compile(r"^(DGIDB_|CHEMBL:|CHEBI:|DRUGBANK_)", re.IGNORECASE)

_SOURCE_PREFIX_MAP = {
    "GO_BP": "GO:BP",
    "GO_CC": "GO:CC",
    "GO_MF": "GO:MF",
    "GOBP": "GO:BP",
    "GOCC": "GO:CC",
    "GOMF": "GO:MF",
    "HP": "HPO",
    "KEGG": "KEGG",
    "REACTOME": "Reactome",
    "WP": "WikiPathways",
    "BIOCARTA": "BioCarta",
}


def humanise_name(raw: str, kind: Literal["pathway", "drug", "atc"]) -> str:
    """Normalise raw pathway/drug/ATC names for display."""
    if not isinstance(raw, str) or not raw.strip():
        return str(raw)

    if kind == "pathway":
        name = raw
        prefix = ""
        for key, label in _SOURCE_PREFIX_MAP.items():
            if name.upper().startswith(key + "_"):
                prefix = f"{label} \u00b7 "
                name = name[len(key) + 1:]
                break
        name = name.replace("_", " ")
        words = name.split()
        result = []
        for w in words:
            if w.upper() in _BIO_ACRONYMS:
                result.append(w.upper())
            elif w.lower() in _PATHWAY_STOP_WORDS and result:
                result.append(w.lower())
            else:
                result.append(w.capitalize())
        return prefix + " ".join(result)

    elif kind == "drug":
        name = _DRUG_PREFIX_RE.sub("", raw).strip()
        if name.startswith("CHEMBL") and name[6:].isdigit():
            return name
        words = name.replace("_", " ").split()
        result = []
        for w in words:
            if w.upper() in _BIO_ACRONYMS:
                result.append(w.upper())
            elif w.isupper() and len(w) > 3:
                result.append(w.capitalize())
            else:
                result.append(w.capitalize())
        return " ".join(result) if result else name

    elif kind == "atc":
        parts = raw.strip().split(None, 1)
        if len(parts) == 2:
            return f"{parts[0]} \u00b7 {parts[1]}"
        return raw

    return raw


def collapse_loci(
    df: pd.DataFrame,
    *,
    chr_col: str = "chr",
    pos_col: str = "start",
    pval_col: str = "magma_p",
    window_kb: int = 500,
) -> pd.DataFrame:
    """Return one row per +/-window_kb locus, keeping the minimum-p-value gene."""
    if df.empty:
        return df
    window_bp = window_kb * 1000
    sdf = df.sort_values([chr_col, pos_col]).copy()
    keep_idx: list[int] = []
    last_chr: object = None
    last_pos: float = -float("inf")
    last_p: float = float("inf")
    last_idx: int = -1

    for row in sdf.itertuples():
        idx = row.Index
        chrom = getattr(row, chr_col)
        pos = getattr(row, pos_col)
        pval = getattr(row, pval_col)

        if chrom != last_chr or (pos - last_pos) > window_bp:
            if last_idx >= 0:
                keep_idx.append(last_idx)
            last_chr = chrom
            last_pos = pos
            last_p = pval
            last_idx = idx
        else:
            if pval < last_p:
                last_pos = pos
                last_p = pval
                last_idx = idx

    if last_idx >= 0:
        keep_idx.append(last_idx)

    return df.loc[keep_idx]


def no_hit_overlay(
    ax: Axes,
    *,
    n_sig: int,
    threshold: float,
    kind: str,
) -> bool:
    """If n_sig == 0, desaturate all scatter points to grey and stamp a
    red banner 'No {kind} pass FDR < {threshold}; showing top nominal'.
    Returns True if overlay was applied."""
    if n_sig > 0:
        return False
    for coll in ax.collections:
        coll.set_facecolor(COLOR_NONSIGNIFICANT)
        coll.set_alpha(0.5)
    ax.text(
        0.5, 0.5,
        f"No {kind} pass FDR < {threshold}\nshowing top nominal",
        transform=ax.transAxes, fontsize=9, color="#CC0000",
        ha="center", va="center", alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#CC0000"),
    )
    return True


def _dot_size_legend(
    ax: Axes,
    sizes: np.ndarray,
    values: np.ndarray,
    label: str,
    *,
    n_dots: int = 4,
    loc: str = "lower right",
) -> None:
    """Bubble-size legend with n representative dots."""
    from matplotlib.lines import Line2D

    if len(values) == 0:
        return
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmin == vmax:
        ticks = [vmin]
    else:
        ticks = np.linspace(vmin, vmax, n_dots).tolist()

    smin, smax = float(np.nanmin(sizes)), float(np.nanmax(sizes))
    handles = []
    for t in ticks:
        frac = (t - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        s = smin + frac * (smax - smin)
        handles.append(
            Line2D([0], [0], marker="o", color="none", markeredgecolor="#666666",
                   markerfacecolor="#CCCCCC", markersize=math.sqrt(s),
                   label=f"{t:.0f}")
        )
    ax.legend(
        handles=handles, title=label, loc=loc,
        fontsize=FONT_SIZE_MINI, title_fontsize=FONT_SIZE_ANNOTATION,
        frameon=True, framealpha=0.8, labelspacing=1.0,
        borderpad=0.8, handletextpad=0.5,
    )


# Apply theme on import
apply_theme()
