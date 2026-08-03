"""Tests for repogen.plotting - figure creation and structure (not pixel-perfect)."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("Agg")

from repogen.plotting.base import (
    apply_theme,
    save_figure,
    format_pvalue,
    truncate_label,
    chromosome_colors,
    add_significance_line,
    PALETTE_CATEGORICAL,
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
    select_forest_rows,
)
from repogen.plotting.convergence import plot_convergence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gene_results_df():
    """Synthetic MAGMA gene results - 100 genes, 5 significant."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "gene_symbol": [f"GENE{i}" for i in range(n)],
        "chr": rng.integers(1, 23, size=n),
        "start": rng.integers(1_000_000, 200_000_000, size=n),
        "magma_z": rng.normal(0, 1.5, size=n),
        "magma_p": np.concatenate([rng.uniform(1e-10, 1e-6, 5), rng.uniform(0.01, 1.0, n - 5)]),
        "fdr_q": np.concatenate([rng.uniform(0.001, 0.04, 5), rng.uniform(0.1, 1.0, n - 5)]),
        "in_mhc": [False] * 95 + [True] * 5,
        "biotype": ["protein_coding"] * n,
    })


@pytest.fixture
def pathway_results_df():
    """Synthetic pathway enrichment results."""
    rng = np.random.default_rng(42)
    n = 50
    sources = ["GO_BP", "KEGG", "REACTOME", "WIKIPATHWAYS"]
    return pd.DataFrame({
        "pathway_name": [f"PATHWAY_{i}" for i in range(n)],
        "source_db": rng.choice(sources, size=n),
        "p_value": np.concatenate([rng.uniform(1e-8, 1e-4, 3), rng.uniform(0.01, 1.0, n - 3)]),
        "fdr_q": np.concatenate([rng.uniform(0.001, 0.04, 3), rng.uniform(0.1, 1.0, n - 3)]),
        "n_genes_in_set": rng.integers(10, 500, size=n),
        "beta": rng.normal(0.5, 0.3, size=n),
    })


@pytest.fixture
def drug_results_df():
    """Synthetic drug enrichment results."""
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame({
        "drug_name": [f"DRUG_{i}" for i in range(n)],
        "magma_p": np.concatenate([rng.uniform(1e-6, 1e-3, 5), rng.uniform(0.05, 1.0, n - 5)]),
        "magma_fdr_q": np.concatenate([rng.uniform(0.001, 0.04, 5), rng.uniform(0.1, 1.0, n - 5)]),
        "max_phase": rng.integers(0, 5, size=n),
        "magma_beta": rng.normal(0.3, 0.2, size=n),
        "wilcoxon_auc": rng.uniform(0.4, 0.9, size=n),
    })


@pytest.fixture
def atc_results_df():
    """Synthetic ATC enrichment results."""
    rng = np.random.default_rng(42)
    n = 25
    return pd.DataFrame({
        "atc_code": [f"N0{i}" for i in range(n)],
        "atc_description": [f"Nervous System Drug Class {i}" for i in range(n)],
        "atc_level": rng.choice([2, 3], size=n),
        "gls_p": np.concatenate([rng.uniform(1e-5, 1e-3, 4), rng.uniform(0.05, 1.0, n - 4)]),
        "gls_fdr": np.concatenate([rng.uniform(0.005, 0.04, 4), rng.uniform(0.1, 1.0, n - 4)]),
        "n_drugs": rng.integers(5, 50, size=n),
        "gls_beta": rng.normal(0.2, 0.15, size=n),
    })


@pytest.fixture
def correlation_results_df():
    """Synthetic negative correlation per-tissue results."""
    rng = np.random.default_rng(42)
    drugs = [f"DRUG_{i}" for i in range(20)]
    tissues = ["Brain_Cortex", "Brain_Hippocampus", "Brain_Amygdala"]
    rows = []
    for d in drugs:
        for t in tissues:
            rows.append({
                "drug_name": d,
                "tissue": t,
                "spearman_rho": rng.uniform(-0.8, 0.3),
                "fdr_global": rng.uniform(1e-5, 1.0),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tissue_signature_df():
    """Synthetic S-PrediXcan per-tissue results."""
    rng = np.random.default_rng(42)
    genes = [f"GENE_{i}" for i in range(30)]
    tissues = ["Brain_Cortex", "Brain_Hippocampus", "Brain_Amygdala"]
    rows = []
    for g in genes:
        for t in tissues:
            rows.append({
                "gene_symbol": g,
                "tissue": t,
                "zscore": rng.normal(0, 2),
                "pvalue": rng.uniform(1e-6, 1.0),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mr_results_df():
    """Synthetic MR results."""
    rng = np.random.default_rng(42)
    n = 15
    tiers = ["high", "medium", "low", "direction_conflict"]
    statuses = ["colocalised", "distinct_signals", "insufficient_data", "unsupported"]
    return pd.DataFrame({
        "gene_symbol": [f"GENE_{i}" for i in range(n)],
        "eqtl_source": ["eqtlgen"] * 8 + ["metabrain_cortex"] * 7,
        "mr_beta": rng.normal(0, 0.3, size=n),
        "mr_se": rng.uniform(0.05, 0.2, size=n),
        "mr_pval": np.concatenate([rng.uniform(1e-6, 0.01, 5), rng.uniform(0.05, 1.0, n - 5)]),
        "confidence_tier": rng.choice(tiers, size=n),
        "n_instruments": rng.integers(1, 10, size=n),
        "pp_h4": np.concatenate([rng.uniform(0.7, 0.99, 8), [np.nan] * 7]),
        "pp_h3": np.concatenate([rng.uniform(0.01, 0.4, 8), [np.nan] * 7]),
        "coloc_status": rng.choice(statuses, size=n),
    })


@pytest.fixture
def drug_matches_df():
    """Synthetic MR drug match results - includes tri-state direction and duplicates."""
    rng = np.random.default_rng(42)
    rows = []
    for g in ["GENE_0", "GENE_1", "GENE_2"]:
        for d in ["Aspirin", "Fluoxetine", "Risperidone"]:
            rows.append({
                "gene_symbol": g,
                "drug_name": d,
                "confidence_tier": rng.choice(["high", "medium", "low"]),
                "direction_concordant": rng.choice([True, False, None]),
                "interaction_type": rng.choice(["inhibitor", "agonist", "antagonist", "modulator"]),
                "max_phase": int(rng.integers(0, 5)),
                "eqtl_source": "eqtlgen",
            })
            if rng.random() > 0.5:
                rows.append({
                    "gene_symbol": g,
                    "drug_name": d,
                    "confidence_tier": rng.choice(["high", "medium", "low"]),
                    "direction_concordant": rng.choice([True, False, None]),
                    "interaction_type": rng.choice(["inhibitor", "agonist", "antagonist", "modulator"]),
                    "max_phase": int(rng.integers(0, 5)),
                    "eqtl_source": "metabrain_cortex",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# base.py tests
# ---------------------------------------------------------------------------


class TestBase:

    def test_format_pvalue_scientific(self):
        result = format_pvalue(3.2e-8)
        assert "10⁻⁸" in result

    def test_format_pvalue_decimal(self):
        result = format_pvalue(0.042)
        assert result.startswith("p = 0.042")

    def test_format_pvalue_zero(self):
        result = format_pvalue(0.0)
        assert "p < " in result

    def test_truncate_label_short(self):
        assert truncate_label("short", 40) == "short"

    def test_truncate_label_long(self):
        result = truncate_label("a" * 50, 40)
        assert len(result) == 40
        assert result.endswith("\u2026")

    def test_apply_theme(self):
        apply_theme()
        assert "sans-serif" in plt.rcParams["font.family"]
        assert plt.rcParams["pdf.fonttype"] == 42

    def test_save_figure(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        paths = save_figure(fig, tmp_path / "test_fig")
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        plt.close(fig)

    def test_chromosome_colors(self):
        colors = chromosome_colors(22)
        assert len(colors) == 22
        assert colors[0] != colors[1]

    def test_add_significance_line_horizontal(self):
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        add_significance_line(ax, 5.0, "test", "horizontal")
        plt.close(fig)

    def test_add_significance_line_invalid_orientation(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="orientation"):
            add_significance_line(ax, 5.0, orientation="diagonal")
        plt.close(fig)


# ---------------------------------------------------------------------------
# manhattan.py tests
# ---------------------------------------------------------------------------


class TestManhattan:

    def test_plot_manhattan_returns_figure(self, gene_results_df):
        fig = plot_manhattan(gene_results_df)
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_manhattan_axes(self, gene_results_df):
        fig = plot_manhattan(gene_results_df)
        ax = fig.axes[0]
        assert ax.get_ylabel() != ""
        assert ax.get_xlabel() != ""
        plt.close(fig)

    def test_plot_manhattan_empty_raises(self, gene_results_df):
        with pytest.raises(ValueError, match="empty"):
            plot_manhattan(gene_results_df.iloc[:0])

    def test_plot_manhattan_missing_columns(self):
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            plot_manhattan(df)

    def test_plot_manhattan_no_labels(self, gene_results_df):
        from repogen.config.schema import ManhattanStyleConfig
        fig = plot_manhattan(gene_results_df, style=ManhattanStyleConfig(n_top_labels=0))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_manhattan_single_chromosome(self, gene_results_df):
        df = gene_results_df.copy()
        df["chr"] = 1
        fig = plot_manhattan(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_manhattan_string_chr_with_x(self, gene_results_df):
        """String chromosomes including 'X' must not raise."""
        df = gene_results_df.copy()
        labels = [str(i) for i in range(1, 23)] + ["X"]
        rng = np.random.default_rng(99)
        df["chr"] = rng.choice(labels, size=len(df))
        fig = plot_manhattan(df)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "X" in tick_labels
        plt.close(fig)

    def test_plot_manhattan_mixed_chr_formats(self, gene_results_df):
        """Handles mixed formats: int-like strings, 'X', 'chr' prefixed."""
        df = gene_results_df.copy()
        labels = ["1", "2", "X", "chr3", "22"]
        rng = np.random.default_rng(77)
        df["chr"] = rng.choice(labels, size=len(df))
        fig = plot_manhattan(df)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "3" in tick_labels, "chr3 should be normalized to 3"
        plt.close(fig)

    def test_plot_manhattan_integer_chr_still_works(self, gene_results_df):
        """Existing integer chromosome input continues to work."""
        df = gene_results_df.copy()
        df["chr"] = np.random.default_rng(42).integers(1, 23, size=len(df))
        fig = plot_manhattan(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_gene_volcano_returns_figure(self, gene_results_df):
        fig = plot_gene_volcano(gene_results_df)
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_gene_volcano_empty_raises(self, gene_results_df):
        with pytest.raises(ValueError, match="empty"):
            plot_gene_volcano(gene_results_df.iloc[:0])

    def test_plot_gene_volcano_missing_columns(self):
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            plot_gene_volcano(df)


# ---------------------------------------------------------------------------
# qq.py tests
# ---------------------------------------------------------------------------


class TestQQ:

    def test_plot_qq_returns_figure(self, gene_results_df):
        fig = plot_qq(gene_results_df["magma_p"].values)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_qq_axes(self, gene_results_df):
        fig = plot_qq(gene_results_df["magma_p"].values)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_plot_qq_empty_raises(self):
        with pytest.raises(ValueError, match="No valid"):
            plot_qq(np.array([]))

    def test_plot_qq_nan_values(self):
        p = np.array([0.1, np.nan, 0.5, np.nan, 0.9])
        fig = plot_qq(p)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_qq_single_pvalue(self):
        fig = plot_qq(np.array([0.05]))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_qq_zero_pvalue(self):
        p = np.array([0.0, 0.1, 0.5, 0.9])
        fig = plot_qq(p)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_qq_all_nan_raises(self):
        with pytest.raises(ValueError, match="No valid"):
            plot_qq(np.array([np.nan, np.nan]))


# ---------------------------------------------------------------------------
# enrichment.py tests
# ---------------------------------------------------------------------------


class TestEnrichment:

    def test_plot_pathway_enrichment_returns_figure(self, pathway_results_df):
        fig = plot_pathway_enrichment(pathway_results_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_pathway_enrichment_empty_raises(self, pathway_results_df):
        with pytest.raises(ValueError, match="empty"):
            plot_pathway_enrichment(pathway_results_df.iloc[:0])

    def test_plot_pathway_enrichment_missing_columns(self):
        with pytest.raises(ValueError, match="Missing"):
            plot_pathway_enrichment(pd.DataFrame({"x": [1]}))

    def test_plot_drug_enrichment_returns_figure(self, drug_results_df):
        fig = plot_drug_enrichment(drug_results_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_drug_enrichment_empty_raises(self, drug_results_df):
        with pytest.raises(ValueError, match="empty"):
            plot_drug_enrichment(drug_results_df.iloc[:0])

    def test_plot_drug_enrichment_without_auc(self, drug_results_df):
        df = drug_results_df.drop(columns=["wilcoxon_auc"])
        fig = plot_drug_enrichment(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_atc_enrichment_returns_figure(self, atc_results_df):
        fig = plot_atc_enrichment(atc_results_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_atc_enrichment_empty_raises(self, atc_results_df):
        with pytest.raises(ValueError, match="empty"):
            plot_atc_enrichment(atc_results_df.iloc[:0])

    def test_plot_pathway_enrichment_nan_labels(self):
        """Pathway plot with NaN pathway_name/source_db must not crash."""
        rng = np.random.default_rng(42)
        n = 10
        df = pd.DataFrame({
            "pathway_id": [f"PATH_{i}" for i in range(n)],
            "pathway_name": [f"Pathway {i}" if i < 5 else np.nan for i in range(n)],
            "source_db": ["GO_BP" if i < 5 else np.nan for i in range(n)],
            "p_value": rng.uniform(1e-6, 0.5, n),
            "fdr_q": rng.uniform(0.01, 0.5, n),
            "n_genes_in_set": rng.integers(10, 100, n),
        })
        fig = plot_pathway_enrichment(df)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# correlation.py tests
# ---------------------------------------------------------------------------


class TestCorrelation:

    def test_plot_correlation_scatter_returns_figure(self, correlation_results_df):
        fig = plot_correlation_scatter(correlation_results_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_correlation_scatter_empty_raises(self, correlation_results_df):
        with pytest.raises(ValueError, match="empty"):
            plot_correlation_scatter(correlation_results_df.iloc[:0])

    def test_plot_correlation_heatmap_returns_figure(self, correlation_results_df):
        fig = plot_correlation_heatmap(correlation_results_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_correlation_heatmap_pivot_shape(self, correlation_results_df):
        fig = plot_correlation_heatmap(correlation_results_df, n_top=5)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_correlation_heatmap_empty_raises(self, correlation_results_df):
        with pytest.raises(ValueError, match="empty"):
            plot_correlation_heatmap(correlation_results_df.iloc[:0])

    def test_plot_tissue_signature_returns_figure(self, tissue_signature_df):
        fig = plot_tissue_signature(tissue_signature_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_tissue_signature_single_tissue(self, tissue_signature_df):
        df = tissue_signature_df[tissue_signature_df["tissue"] == "Brain_Cortex"]
        fig = plot_tissue_signature(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_tissue_signature_empty_raises(self, tissue_signature_df):
        with pytest.raises(ValueError, match="empty"):
            plot_tissue_signature(tissue_signature_df.iloc[:0])


# ---------------------------------------------------------------------------
# mr.py tests
# ---------------------------------------------------------------------------


class TestMR:

    def test_plot_mr_forest_returns_figure(self, mr_results_df):
        fig = plot_mr_forest(mr_results_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mr_forest_single_source(self, mr_results_df):
        df = mr_results_df[mr_results_df["eqtl_source"] == "eqtlgen"]
        fig = plot_mr_forest(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mr_forest_empty_raises(self, mr_results_df):
        with pytest.raises(ValueError, match="empty"):
            plot_mr_forest(mr_results_df.iloc[:0])

    def test_plot_mr_forest_missing_columns(self):
        with pytest.raises(ValueError, match="Missing"):
            plot_mr_forest(pd.DataFrame({"x": [1]}))

    def test_plot_coloc_posteriors_returns_figure(self, mr_results_df):
        fig = plot_coloc_posteriors(mr_results_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_coloc_all_nan_raises(self, mr_results_df):
        df = mr_results_df.copy()
        df["pp_h4"] = np.nan
        with pytest.raises(ValueError, match="non-NaN"):
            plot_coloc_posteriors(df)

    def test_plot_coloc_unsupported_status(self, mr_results_df):
        df = mr_results_df.copy()
        df["coloc_status"] = "unsupported"
        df.loc[df["pp_h4"].notna(), "pp_h4"] = 0.5
        fig = plot_coloc_posteriors(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mr_drug_summary_returns_figure(self, drug_matches_df):
        fig = plot_mr_drug_summary(drug_matches_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mr_drug_summary_single_pair(self):
        df = pd.DataFrame({
            "gene_symbol": ["GENE_0"],
            "drug_name": ["Aspirin"],
            "confidence_tier": ["high"],
            "direction_concordant": [True],
            "interaction_type": ["inhibitor"],
            "max_phase": [4],
        })
        fig = plot_mr_drug_summary(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mr_drug_summary_aggregates_sources(self, drug_matches_df):
        n_unique_pairs = drug_matches_df.drop_duplicates(subset=["gene_symbol", "drug_name"]).shape[0]
        fig = plot_mr_drug_summary(drug_matches_df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mr_drug_summary_direction_none(self):
        df = pd.DataFrame({
            "gene_symbol": ["GENE_0"],
            "drug_name": ["Modafinil"],
            "confidence_tier": ["medium"],
            "direction_concordant": [None],
            "interaction_type": ["modulator"],
            "max_phase": [3],
        })
        fig = plot_mr_drug_summary(df)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mr_drug_summary_empty_raises(self, drug_matches_df):
        with pytest.raises(ValueError, match="empty"):
            plot_mr_drug_summary(drug_matches_df.iloc[:0])


# ---------------------------------------------------------------------------
# convergence.py tests
# ---------------------------------------------------------------------------


class TestConvergence:

    def test_plot_convergence_two_branches(self):
        branch_drugs = {
            "MAGMA": {"DRUG_A", "DRUG_B", "DRUG_C"},
            "Neg Corr": {"DRUG_B", "DRUG_C", "DRUG_D"},
        }
        fig = plot_convergence(branch_drugs)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_convergence_three_branches(self):
        branch_drugs = {
            "MAGMA": {"DRUG_A", "DRUG_B"},
            "Neg Corr": {"DRUG_B", "DRUG_C"},
            "MR": {"DRUG_A", "DRUG_C"},
        }
        fig = plot_convergence(branch_drugs)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_convergence_no_overlap(self):
        branch_drugs = {
            "MAGMA": {"DRUG_A"},
            "Neg Corr": {"DRUG_B"},
        }
        fig = plot_convergence(branch_drugs)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_convergence_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            plot_convergence({"MAGMA": {"DRUG_A"}})

    def test_plot_convergence_all_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            plot_convergence({"MAGMA": set(), "Neg Corr": set()})


# ---------------------------------------------------------------------------
# Forest plot row selection and OOM prevention tests
# ---------------------------------------------------------------------------


class TestSelectForestRows:
    """Tests for select_forest_rows() - single source of truth for row selection."""

    def _make_df(self, n: int, n_significant: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "gene_symbol": [f"GENE_{i}" for i in range(n)],
            "eqtl_source": ["eqtlgen"] * n,
            "mr_beta": rng.normal(0, 0.3, size=n),
            "mr_se": rng.uniform(0.05, 0.2, size=n),
            "mr_pval": rng.uniform(1e-8, 1.0, size=n),
            "confidence_tier": ["medium"] * n,
            "n_instruments": [5] * n,
        })
        if n_significant > 0:
            df["mr_significant"] = [True] * n_significant + [False] * (n - n_significant)
        return df

    def test_significant_only_prioritized(self):
        """With 10 significant out of 500, return exactly 10."""
        df = self._make_df(500, n_significant=10)
        selected = select_forest_rows(df, max_rows=250)
        assert len(selected) == 10
        assert (selected["mr_significant"] == True).all()  # noqa: E712

    def test_no_significant_fallback(self):
        """With 0 significant, return top max_rows by mr_pval."""
        df = self._make_df(500, n_significant=0)
        df["mr_significant"] = False
        selected = select_forest_rows(df, max_rows=250)
        assert len(selected) == 250

    def test_max_rows_enforced_on_significant(self):
        """With 400 significant, cap at max_rows=250."""
        df = self._make_df(400, n_significant=400)
        selected = select_forest_rows(df, max_rows=250)
        assert len(selected) == 250
        assert (selected["mr_significant"] == True).all()  # noqa: E712

    def test_no_mr_significant_column(self):
        """Without mr_significant column, use top by mr_pval."""
        df = self._make_df(500, n_significant=0)
        selected = select_forest_rows(df, max_rows=100)
        assert len(selected) == 100

    def test_sorted_by_pval(self):
        """Selected rows should be sorted by mr_pval ascending."""
        df = self._make_df(500, n_significant=50)
        selected = select_forest_rows(df, max_rows=250)
        pvals = selected["mr_pval"].values
        assert (pvals[:-1] <= pvals[1:]).all()

    def test_small_input_no_truncation(self):
        """Input smaller than max_rows is returned fully."""
        df = self._make_df(10, n_significant=3)
        selected = select_forest_rows(df, max_rows=250)
        assert len(selected) == 3


class TestPlotMrForestBounded:
    """Tests that plot_mr_forest respects max_rows and max_height."""

    def _make_large_df(self, n: int) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "gene_symbol": [f"GENE_{i}" for i in range(n)],
            "eqtl_source": ["eqtlgen"] * n,
            "mr_beta": rng.normal(0, 0.3, size=n),
            "mr_se": rng.uniform(0.05, 0.2, size=n),
            "mr_pval": rng.uniform(1e-8, 1.0, size=n),
            "confidence_tier": ["medium"] * n,
            "n_instruments": [5] * n,
            "mr_significant": [True] * min(n, 50) + [False] * max(0, n - 50),
        })

    def test_max_height_enforced(self):
        """Figure height never exceeds max_height."""
        df = self._make_large_df(1000)
        fig = plot_mr_forest(df, max_rows=250, max_height=24.0)
        assert fig.get_size_inches()[1] <= 24.0
        plt.close(fig)

    def test_large_input_does_not_oom(self):
        """14k-row input produces bounded figure without crash."""
        df = self._make_large_df(14000)
        fig = plot_mr_forest(df, max_rows=250, max_height=24.0)
        assert isinstance(fig, Figure)
        h = fig.get_size_inches()[1]
        assert h <= 24.0
        plt.close(fig)

    def test_returns_figure(self):
        """API contract: returns Figure."""
        df = self._make_large_df(100)
        fig = plot_mr_forest(df)
        assert isinstance(fig, Figure)
        plt.close(fig)
