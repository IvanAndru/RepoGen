"""RepoGen plotting subpackage - publication-quality figures for genomic drug repurposing."""

from repogen.plotting.base import (
    apply_theme,
    save_figure,
    format_pvalue,
    truncate_label,
    add_significance_line,
    chromosome_colors,
    PALETTE_CATEGORICAL,
    PALETTE_SEQUENTIAL,
    PALETTE_DIVERGING,
    COLOR_SIGNIFICANT,
    COLOR_NONSIGNIFICANT,
    COLOR_BRANCH_A,
    COLOR_BRANCH_B,
    COLOR_BRANCH_C,
    COLOR_THRESHOLD,
    COLOR_CI_BAND,
    COLOR_ZERO_LINE,
    COLOR_NEUTRAL_LIGHT,
    COLOR_NEUTRAL_MEDIUM,
    COLOR_SECONDARY_TEXT,
    ATC_LEVEL_COLORS,
    FONT_SIZE_MINI,
    FIGSIZE_SINGLE,
    FIGSIZE_STANDARD,
    FIGSIZE_TALL,
    FIGSIZE_SQUARE,
    DPI_SCREEN,
    DPI_PUBLICATION,
)
from repogen.plotting.manhattan import plot_manhattan, plot_gene_volcano
from repogen.plotting.qq import plot_qq
from repogen.plotting.enrichment import (
    plot_pathway_enrichment,
    plot_drug_enrichment,
    plot_atc_enrichment,
)
from repogen.plotting.correlation import (
    plot_correlation_scatter,
    plot_correlation_heatmap,
    plot_tissue_signature,
)
from repogen.plotting.mr import (
    plot_mr_forest,
    plot_coloc_posteriors,
    plot_mr_drug_summary,
)
from repogen.plotting.convergence import plot_convergence

__all__ = [
    # Base utilities
    "apply_theme",
    "save_figure",
    "format_pvalue",
    "truncate_label",
    "add_significance_line",
    "chromosome_colors",
    # Manhattan
    "plot_manhattan",
    "plot_gene_volcano",
    # Qq
    "plot_qq",
    # Enrichment
    "plot_pathway_enrichment",
    "plot_drug_enrichment",
    "plot_atc_enrichment",
    # Correlation
    "plot_correlation_scatter",
    "plot_correlation_heatmap",
    "plot_tissue_signature",
    # MR
    "plot_mr_forest",
    "plot_coloc_posteriors",
    "plot_mr_drug_summary",
    # Convergence
    "plot_convergence",
]
