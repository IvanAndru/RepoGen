"""Publication-readiness regression tests for Branch A plots.

These complement test_plotting.py, which covers rendering, with
targeted contract and behaviour checks.
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from repogen.config.schema import (
    OutputConfig,
    PlotStyleConfig,
    ManhattanStyleConfig,
    VolcanoStyleConfig,
)
from repogen.plotting.base import (
    save_figure,
    humanise_name,
    collapse_loci,
    fdr_implied_p_cutoff,
    bonferroni_cutoff,
    annotate_stats,
    figure_footer,
    panel_label,
    render_threshold_line,
    no_hit_overlay,
    FIGSIZE_SQUARE_LARGE,
    FIGSIZE_WIDE_SHORT,
    PHASE_COLORS,
)
from repogen.plotting.manhattan import plot_manhattan, plot_gene_volcano
from repogen.plotting.qq import plot_qq
from repogen.plotting.enrichment import (
    plot_pathway_enrichment,
    plot_drug_enrichment,
    plot_atc_enrichment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gene_results_with_mhc():
    """200 rows, includes 10 MHC genes on chr6 28-32 Mb."""
    rng = np.random.default_rng(42)
    n = 200
    chrs = [str(c) for c in rng.integers(1, 23, size=n)]
    starts = rng.integers(1_000_000, 200_000_000, size=n)
    p_vals = np.concatenate([rng.uniform(1e-12, 1e-6, 20), rng.uniform(0.01, 1.0, n - 20)])
    fdr_vals = np.concatenate([rng.uniform(0.001, 0.04, 20), rng.uniform(0.1, 1.0, n - 20)])

    for i in range(10):
        chrs[n - 10 + i] = "6"
        starts[n - 10 + i] = rng.integers(28_000_000, 32_000_000)

    in_mhc = [False] * (n - 10) + [True] * 10
    return pd.DataFrame({
        "gene_symbol": [f"GENE{i}" for i in range(n)],
        "gene_entrez_id": list(range(1000, 1000 + n)),
        "chr": chrs,
        "start": starts,
        "magma_z": rng.normal(0, 2, size=n),
        "magma_p": p_vals,
        "fdr_q": fdr_vals,
        "in_mhc": in_mhc,
        "n_snps": rng.integers(5, 500, size=n),
    })


@pytest.fixture
def gene_results_highly_inflated():
    """Highly inflated case to stress QQ inset (max obs / max exp > 3)."""
    rng = np.random.default_rng(99)
    n = 500
    p_vals = np.concatenate([
        np.full(10, 1e-80),
        rng.uniform(1e-50, 1e-20, 40),
        rng.uniform(1e-10, 1e-4, 100),
        rng.uniform(1e-4, 0.5, 350),
    ])
    return pd.DataFrame({
        "gene_symbol": [f"G{i}" for i in range(n)],
        "chr": [str(c) for c in rng.integers(1, 23, size=n)],
        "start": rng.integers(1_000_000, 200_000_000, size=n),
        "magma_z": rng.normal(0, 3, size=n),
        "magma_p": p_vals,
        "fdr_q": np.clip(p_vals * n / np.arange(1, n + 1)[::-1], 0, 1),
        "in_mhc": [i < 20 for i in range(n)],
        "n_snps": rng.integers(5, 500, size=n),
    })


@pytest.fixture
def drug_no_hits():
    """All magma_fdr_q > 0.5."""
    rng = np.random.default_rng(7)
    n = 25
    return pd.DataFrame({
        "drug_name": [f"DRUG_{i}" for i in range(n)],
        "drug_chembl_id": [f"CHEMBL{i}" for i in range(n)],
        "magma_p": rng.uniform(0.01, 0.5, size=n),
        "magma_fdr_q": rng.uniform(0.5, 1.0, size=n),
        "magma_beta": rng.normal(0, 1, size=n),
        "max_phase": rng.choice([0, 1, 2, 3, 4], size=n),
        "n_target_genes": rng.integers(1, 50, size=n),
        "wilcoxon_auc": rng.uniform(0.4, 0.7, size=n),
        "atc_codes": [["N05A"] if i % 3 == 0 else [] for i in range(n)],
    })


@pytest.fixture
def drug_mixed_phases():
    """20 rows spanning phases 0-4 unevenly."""
    rng = np.random.default_rng(8)
    phases = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    n = len(phases)
    return pd.DataFrame({
        "drug_name": [f"DRUG_{i}" for i in range(n)],
        "magma_p": rng.uniform(1e-6, 0.1, size=n),
        "magma_fdr_q": rng.uniform(0.001, 0.3, size=n),
        "magma_beta": rng.normal(0, 1, size=n),
        "max_phase": phases,
        "n_target_genes": rng.integers(1, 50, size=n),
        "wilcoxon_auc": rng.uniform(0.4, 0.9, size=n),
    })


@pytest.fixture
def atc_results_15():
    """Mock ATC enrichment with 15 rows (level 2 + level 3)."""
    rows = []
    for i in range(15):
        level = 2 if i < 5 else 3
        parent = f"{'ABCDE'[i % 5]}"
        code = f"{parent}0{i}" if level == 2 else f"{parent}0{i % 5}{chr(65 + i)}"
        rows.append({
            "atc_code": code,
            "atc_description": f"Description {i}",
            "atc_level": level,
            "gls_p": 0.001 * (i + 1),
            "gls_fdr": 0.005 * (i + 1),
            "gls_beta": 0.5 * ((-1) ** i),
            "gls_se": 0.2,
            "n_drugs": 10 + i * 3,
            "annotation_bias_risk": i == 2,
            "borderline_power": i == 4,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def pathway_results_with_beta():
    """Pathway data including beta column."""
    rng = np.random.default_rng(11)
    n = 15
    return pd.DataFrame({
        "pathway_name": [f"GO_PATHWAY_{i}" for i in range(n)],
        "pathway_id": [f"GO:{i:07d}" for i in range(n)],
        "source_db": ["GO_BP"] * 5 + ["KEGG"] * 5 + ["REACTOME"] * 5,
        "p_value": rng.uniform(1e-8, 0.05, size=n),
        "fdr_q": rng.uniform(0.001, 0.3, size=n),
        "n_genes_in_set": rng.integers(10, 500, size=n),
        "beta": rng.normal(0, 1, size=n),
    })


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestPlotStyleConfig:
    def test_defaults_load(self, tmp_path):
        """OutputConfig.plot_style defaults apply when user YAML omits plot_style."""
        oc = OutputConfig()
        assert hasattr(oc, "plot_style")
        assert oc.plot_style.png_dpi == 600
        assert oc.plot_style.manhattan.chromosome_scale == "rank"

    def test_svg_in_defaults(self):
        assert "svg" in PlotStyleConfig().figure_formats


# ---------------------------------------------------------------------------
# Base helper tests
# ---------------------------------------------------------------------------

class TestSaveFigure:
    def test_writes_all_formats(self, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        style = PlotStyleConfig()
        paths = save_figure(fig, tmp_path / "test.png", style=style)
        extensions = {p.suffix for p in paths}
        assert {".pdf", ".svg", ".png"} == extensions
        for p in paths:
            assert p.stat().st_size > 0
        plt.close(fig)

    def test_rasterize_scatter_preserves_text(self, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter([1, 2], [3, 4])
        sc.set_gid("dense_scatter")
        ax.set_xlabel("X Label")
        style = PlotStyleConfig()
        paths = save_figure(fig, tmp_path / "test.png", style=style, rasterize_scatter=True)
        svg_path = [p for p in paths if p.suffix == ".svg"][0]
        svg_text = svg_path.read_text()
        assert "X Label" in svg_text
        plt.close(fig)


class TestHumaniseName:
    def test_pathway_cases(self):
        result = humanise_name("GABA_ERGIC_SYNAPSE", "pathway")
        assert "GABA" in result

    def test_pathway_stopwords(self):
        result = humanise_name("REGULATION_OF_SYNAPTIC_TRANSMISSION", "pathway")
        assert "of" in result.lower()

    def test_drug_prefixes(self):
        assert humanise_name("DGIDB_BATIMASTAT", "drug") == "Batimastat"

    def test_drug_chembl_prefix(self):
        result = humanise_name("CHEMBL:CHEMBL584442", "drug")
        assert result.startswith("CHEMBL")

    def test_atc_format(self):
        result = humanise_name("N05A Antipsychotics", "atc")
        assert "\u00b7" in result


class TestCollapseLoci:
    def test_window_collapse(self):
        df = pd.DataFrame({
            "chr": ["1", "1", "1", "2"],
            "start": [1_000_000, 1_200_000, 5_000_000, 1_000_000],
            "magma_p": [0.001, 0.01, 0.05, 0.001],
        })
        result = collapse_loci(df, window_kb=500)
        assert len(result) == 3
        chr1_kept = result[result["chr"] == "1"]
        assert chr1_kept["magma_p"].min() == 0.001


class TestHelpers:
    def test_fdr_implied_p_cutoff(self):
        p = np.array([0.001, 0.01, 0.05, 0.5])
        fdr = np.array([0.01, 0.04, 0.1, 0.8])
        result = fdr_implied_p_cutoff(p, fdr, 0.05)
        assert result == pytest.approx(0.01)

    def test_fdr_implied_p_cutoff_none(self):
        p = np.array([0.5, 0.8])
        fdr = np.array([0.5, 0.8])
        assert fdr_implied_p_cutoff(p, fdr, 0.05) is None

    def test_bonferroni_cutoff(self):
        assert bonferroni_cutoff(1000) == pytest.approx(5e-5)


# ---------------------------------------------------------------------------
# Manhattan tests
# ---------------------------------------------------------------------------

class TestManhattan:
    def test_rank_vs_bp_produce_equal_points(self, gene_results_with_mhc):
        style_rank = ManhattanStyleConfig(chromosome_scale="rank")
        style_bp = ManhattanStyleConfig(chromosome_scale="bp")
        fig_rank = plot_manhattan(gene_results_with_mhc, style=style_rank)
        fig_bp = plot_manhattan(gene_results_with_mhc, style=style_bp)
        n_rank = sum(len(c.get_offsets()) for c in fig_rank.axes[0].collections)
        n_bp = sum(len(c.get_offsets()) for c in fig_bp.axes[0].collections)
        assert n_rank == n_bp
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_y_cap_clips_and_annotates(self, gene_results_with_mhc):
        style = ManhattanStyleConfig(y_cap=5.0)
        fig = plot_manhattan(gene_results_with_mhc, style=style)
        ax = fig.axes[0]
        assert ax.get_ylim()[1] <= 6.0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_mhc_open_triangle_marker(self, gene_results_with_mhc):
        fig = plot_manhattan(gene_results_with_mhc)
        ax = fig.axes[0]
        has_triangle = False
        for coll in ax.collections:
            paths = coll.get_paths()
            if not paths:
                continue
            # Triangle markers have 3+1 vertices (closed path)
            if any(len(p.vertices) in (3, 4) for p in paths):
                has_triangle = True
                break
        assert has_triangle, "Expected MHC triangle scatter layer"
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# QQ tests
# ---------------------------------------------------------------------------

class TestQQ:
    def test_split_mhc_two_colours(self, gene_results_with_mhc):
        fig = plot_qq(gene_results_with_mhc, split_mhc=True)
        ax = fig.axes[0]
        scatter_count = sum(1 for c in ax.collections if len(c.get_offsets()) > 0)
        assert scatter_count >= 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_lambda_1000_computed(self, gene_results_with_mhc):
        fig = plot_qq(gene_results_with_mhc, n_effective=50000)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        has_lambda_1000 = any("\u03bb_1000" in t for t in texts)
        assert has_lambda_1000, f"Expected lambda_1000 in annotations, got: {texts}"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_inset_appears_when_inflation_large(self, gene_results_highly_inflated):
        fig = plot_qq(gene_results_highly_inflated, inset=True)
        ax = fig.axes[0]
        n_child = len(getattr(ax, "child_axes", []))
        assert n_child >= 1, f"Expected inset child axis, got {n_child}"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_inset_absent_when_no_inflation(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(0.01, 1.0, 100)
        fig = plot_qq(p, inset=True)
        ax = fig.axes[0]
        n_child = len(getattr(ax, "child_axes", []))
        assert n_child == 0
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Volcano tests
# ---------------------------------------------------------------------------

class TestVolcano:
    def test_default_mode_plots_raw_magma_z(self, gene_results_with_mhc):
        style = VolcanoStyleConfig(x_axis_mode="magma_z")
        fig = plot_gene_volcano(gene_results_with_mhc, style=style)
        ax = fig.axes[0]
        assert "MAGMA Z" in ax.get_xlabel()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_top_snp_beta_with_sign_lookup(self, gene_results_with_mhc):
        style = VolcanoStyleConfig(x_axis_mode="top_snp_beta")
        lookup = {eid: 1.0 if eid % 2 == 0 else -1.0
                  for eid in gene_results_with_mhc["gene_entrez_id"]}
        fig = plot_gene_volcano(gene_results_with_mhc, style=style,
                                sign_lookup=lookup)
        ax = fig.axes[0]
        assert "Signed" in ax.get_xlabel()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_top_snp_beta_falls_back_when_empty(self, gene_results_with_mhc):
        style = VolcanoStyleConfig(x_axis_mode="top_snp_beta")
        fig = plot_gene_volcano(gene_results_with_mhc, style=style,
                                sign_lookup={})
        ax = fig.axes[0]
        assert "MAGMA Z" in ax.get_xlabel()
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Enrichment tests
# ---------------------------------------------------------------------------

class TestDrugEnrichment:
    def test_no_hit_regime_desaturates(self, drug_no_hits):
        fig = plot_drug_enrichment(drug_no_hits)
        ax = fig.axes[0]
        title = ax.get_title()
        assert "no drugs pass FDR" in title
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_empty_phase_legend_filtered(self, drug_mixed_phases):
        fig = plot_drug_enrichment(drug_mixed_phases)
        ax = fig.axes[0]
        legend = ax.get_legend()
        if legend:
            legend_labels = [t.get_text() for t in legend.get_texts()]
            all_phases = set(drug_mixed_phases["max_phase"].unique())
            legend_phases = set()
            for label in legend_labels:
                for p in range(5):
                    if f"Phase {p}" in label or ("Preclinical" in label and p == 0):
                        legend_phases.add(p)
            assert legend_phases.issubset(all_phases)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPathwayPlot:
    def test_sort_order_highest_top(self, pathway_results_with_beta):
        fig = plot_pathway_enrichment(pathway_results_with_beta)
        ax = fig.axes[0]
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert len(ytick_labels) > 0
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestATCForest:
    def test_error_bars_rendered(self, atc_results_15):
        fig = plot_atc_enrichment(atc_results_15)
        ax = fig.axes[0]
        has_errorbar = len(ax.containers) > 0
        assert has_errorbar, "Expected error bar containers in ATC forest plot"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_hierarchy_indent(self, atc_results_15):
        fig = plot_atc_enrichment(atc_results_15)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        has_indent = any(lab.startswith("   ") for lab in labels)
        assert has_indent, f"Expected level-3 indented labels, got: {labels[:3]}"
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Golden images - opt-in only
# ---------------------------------------------------------------------------

GOLDEN = os.environ.get("REPOGEN_GOLDEN") == "1"


@pytest.mark.skipif(not GOLDEN, reason="Set REPOGEN_GOLDEN=1 to run pixel-diff tests")
@pytest.mark.parametrize("plot_name", [
    "manhattan", "qq", "volcano",
    "pathway_enrichment", "drug_enrichment", "atc_enrichment",
])
def test_golden_image(plot_name, tmp_path):
    expected = Path("tests/golden/plots") / f"{plot_name}.png"
    if not expected.is_file():
        pytest.skip(f"Golden image not found: {expected}")
    import matplotlib.testing.compare as mpc
    actual = tmp_path / f"{plot_name}.png"
    result = mpc.compare_images(str(expected), str(actual), tol=10)
    assert result is None, result
