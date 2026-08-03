"""Cross-branch convergence UpSet plot showing which drugs are found by multiple branches."""

import warnings
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from repogen.plotting.base import (
    FIGSIZE_STANDARD,
    PALETTE_CATEGORICAL,
    COLOR_NONSIGNIFICANT,
    COLOR_THRESHOLD,
    COLOR_NEUTRAL_LIGHT,
    FONT_SIZE_TITLE,
    FONT_SIZE_TICK,
    FONT_SIZE_ANNOTATION,
)
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


def plot_convergence(
    branch_drugs: dict[str, set[str]],
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Cross-branch convergence UpSet plot.

    Uses the upsetplot library when available and compatible. Falls back to
    a native matplotlib implementation when upsetplot has pandas CoW issues.

    Args:
        branch_drugs: Mapping from branch name to set of drug identifiers.
        figsize: Override. None uses FIGSIZE_STANDARD.
        title: Override figure title.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If branch_drugs is empty or all sets are empty.
        ValueError: If fewer than 2 branches are provided.
    """
    if len(branch_drugs) < 2:
        raise ValueError("UpSet plot requires at least 2 branches")

    all_drugs: set[str] = set()
    for drugs in branch_drugs.values():
        all_drugs |= drugs
    if not all_drugs:
        raise ValueError("All branch drug sets are empty")

    if figsize is None:
        figsize = FIGSIZE_STANDARD

    try:
        return _plot_with_upsetplot(branch_drugs, all_drugs, figsize, title)
    except (ValueError, TypeError, AttributeError):
        logger.info("upsetplot incompatible with current pandas; using native implementation")
        return _plot_native_upset(branch_drugs, all_drugs, figsize, title)


def _plot_with_upsetplot(
    branch_drugs: dict[str, set[str]],
    all_drugs: set[str],
    figsize: tuple[float, float],
    title: str | None,
) -> Figure:
    """Attempt UpSet plot via upsetplot library."""
    import upsetplot

    branch_names = list(branch_drugs.keys())
    records: list[dict] = []
    for drug in sorted(all_drugs):
        row = {name: drug in branch_drugs[name] for name in branch_names}
        records.append(row)

    indicator_df = pd.DataFrame(records)
    indicator_df = indicator_df.set_index(branch_names)
    counts = indicator_df.groupby(level=branch_names).size()

    upset = upsetplot.UpSet(
        counts,
        sort_by="degree",
        sort_categories_by=None,
        show_counts=True,
        facecolor=PALETTE_CATEGORICAL[4],
    )

    fig = plt.figure(figsize=figsize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        upset.plot(fig=fig)

    if title:
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE, y=1.02)
    return fig


def _plot_native_upset(
    branch_drugs: dict[str, set[str]],
    all_drugs: set[str],
    figsize: tuple[float, float],
    title: str | None,
) -> Figure:
    """Pure-matplotlib UpSet-style plot."""
    branch_names = list(branch_drugs.keys())
    n_branches = len(branch_names)

    membership_counter: Counter[tuple[bool, ...]] = Counter()
    for drug in all_drugs:
        key = tuple(drug in branch_drugs[name] for name in branch_names)
        membership_counter[key] += 1

    intersections = sorted(
        membership_counter.items(),
        key=lambda item: (-sum(item[0]), -item[1]),
    )

    n_intersections = len(intersections)
    bar_color_multi = PALETTE_CATEGORICAL[4]

    fig, (ax_bar, ax_matrix) = plt.subplots(
        2, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    x = np.arange(n_intersections)
    counts = [c for _, c in intersections]
    degrees = [sum(k) for k, _ in intersections]
    bar_colors = [bar_color_multi if d > 1 else COLOR_NONSIGNIFICANT for d in degrees]

    ax_bar.bar(x, counts, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.6)
    for i, c in enumerate(counts):
        ax_bar.text(i, c + max(counts) * 0.02, str(c), ha="center", va="bottom",
                    fontsize=FONT_SIZE_ANNOTATION)

    ax_bar.set_ylabel("Intersection size")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_xlim(-0.5, n_intersections - 0.5)

    ax_matrix.set_ylim(-0.5, n_branches - 0.5)
    ax_matrix.set_xlim(-0.5, n_intersections - 0.5)
    ax_matrix.invert_yaxis()

    for i, (key, _) in enumerate(intersections):
        active_rows = [j for j, active in enumerate(key) if active]
        inactive_rows = [j for j, active in enumerate(key) if not active]

        for j in inactive_rows:
            ax_matrix.scatter([i], [j], c=COLOR_NEUTRAL_LIGHT, s=40, zorder=2)
        for j in active_rows:
            ax_matrix.scatter([i], [j], c=COLOR_THRESHOLD, s=40, zorder=3)

        if len(active_rows) > 1:
            ax_matrix.plot(
                [i, i],
                [min(active_rows), max(active_rows)],
                color=COLOR_THRESHOLD,
                linewidth=1.5,
                zorder=2,
            )

    ax_matrix.set_yticks(range(n_branches))
    ax_matrix.set_yticklabels(branch_names, fontsize=FONT_SIZE_TICK)
    ax_matrix.set_xticks([])
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)
    ax_matrix.spines["bottom"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE)

    fig.tight_layout()
    return fig
