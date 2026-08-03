"""Tests for repogen.analysis.atc_enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import stats

from repogen.analysis.atc_enrichment import (
    ATC_LEVEL_PREFIX,
    _deserialize_atc_codes,
    _has_atc_codes,
    _infer_atc_level,
    _read_drug_geneset_file,
    assemble_atc_results,
    build_chromosome_projections,
    build_custom_atc_classes,
    compute_drug_set_correlations,
    extract_atc_classes,
    parse_genes_raw,
    run_atc_enrichment,
    run_gls_regression,
    run_permutation_test,
    run_wilcoxon_test,
)
from repogen.config.schema import (
    ATCEnrichmentConfig,
    CustomATCClass,
    PipelineConfig,
    StudyConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_genes_raw(tmp_path):
    """Minimal .genes.raw with 6 genes across 2 chromosomes, known correlations."""
    content = """\
# MEAN_SAMPLE_SIZE = 1000
# TOTAL_GENES = 6
1001 1 10 5 1000 0.5 2.0
1002 1 15 6 1000 0.4 1.5 0.3
1003 1 20 7 1000 0.6 -0.5 0.1 0.2
1004 1 12 5 1000 0.3 3.0 0.0 0.4 0.1
2001 2 8 4 1000 0.7 0.5
2002 2 18 8 1000 0.2 -1.0 0.5
"""
    path = tmp_path / "test.genes.raw"
    path.write_text(content)
    return path


@pytest.fixture()
def synthetic_drug_results():
    """Drug results where N05A drugs have elevated Z-scores."""
    rng = np.random.default_rng(42)
    drugs = []
    for i in range(30):
        if i < 8:
            z = rng.normal(2.5, 0.5)
            atc = [f"N05AH{i:02d}"]
        elif i < 15:
            z = rng.normal(0.0, 1.0)
            atc = [f"C08CA{i:02d}"]
        else:
            z = rng.normal(0.0, 1.0)
            atc = [] if i > 25 else [f"A01A{chr(65 + i % 6)}{i:02d}"]
        drugs.append({
            "drug_chembl_id": f"CHEMBL{1000 + i}",
            "drug_name": f"Drug_{i}",
            "magma_beta": z * 0.3,
            "magma_se": 0.3,
            "magma_p": float(stats.norm.sf(z)),
            "magma_z": z,
            "n_target_genes": int(rng.integers(3, 20)),
            "atc_codes": atc,
            "drug_inchikey": f"INCHI{i}",
            "drug_pubchem_cid": f"CID{i}",
            "max_phase": 4,
            "mechanism_of_action": "test",
        })
    return pd.DataFrame(drugs)


@pytest.fixture()
def synthetic_drug_gene_sets():
    """Map each drug to a subset of genes from the synthetic genes.raw."""
    gene_pool = [1001, 1002, 1003, 1004, 2001, 2002]
    rng = np.random.default_rng(42)
    sets = {}
    for i in range(30):
        n_genes = rng.integers(2, 5)
        chosen = rng.choice(gene_pool, size=n_genes, replace=False).tolist()
        sets[f"CHEMBL{1000 + i}"] = chosen
    return sets


@pytest.fixture()
def drug_geneset_file(tmp_path, synthetic_drug_gene_sets):
    """Write a tab-separated drug gene-set file."""
    path = tmp_path / "drug_genesets.txt"
    with open(path, "w") as f:
        for drug_id, gene_ids in sorted(synthetic_drug_gene_sets.items()):
            line = drug_id + "\t" + "\t".join(str(g) for g in gene_ids)
            f.write(line + "\n")
    return path


@pytest.fixture()
def gene_info_and_corr(synthetic_genes_raw):
    """Parsed gene info and correlation matrices."""
    return parse_genes_raw(synthetic_genes_raw)


@pytest.fixture()
def projections_and_indices(gene_info_and_corr):
    """Build projections from the synthetic data."""
    gene_info, chr_corrs = gene_info_and_corr
    return build_chromosome_projections(gene_info, chr_corrs, eigenvalue_threshold=0.01)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseGenesRaw:
    def test_basic(self, synthetic_genes_raw):
        gene_info, chr_corrs = parse_genes_raw(synthetic_genes_raw)
        assert len(gene_info) == 6
        assert set(gene_info.columns) == {"gene", "chr", "nsnps", "nparam", "nsamp", "mac", "zstat"}
        assert gene_info["gene"].dtype == int
        assert gene_info["zstat"].dtype == float

    def test_chromosome_grouping(self, synthetic_genes_raw):
        gene_info, chr_corrs = parse_genes_raw(synthetic_genes_raw)
        assert set(chr_corrs.keys()) == {"1", "2"}
        assert (gene_info["chr"] == "1").sum() == 4
        assert (gene_info["chr"] == "2").sum() == 2

    def test_correlation_matrix_shape(self, synthetic_genes_raw):
        _, chr_corrs = parse_genes_raw(synthetic_genes_raw)
        assert chr_corrs["1"].shape == (4, 4)
        assert chr_corrs["2"].shape == (2, 2)

    def test_correlation_matrix_symmetric(self, synthetic_genes_raw):
        _, chr_corrs = parse_genes_raw(synthetic_genes_raw)
        for chrom, mat in chr_corrs.items():
            np.testing.assert_array_almost_equal(mat, mat.T)
            np.testing.assert_array_almost_equal(np.diag(mat), np.ones(mat.shape[0]))

    def test_known_correlation_values(self, synthetic_genes_raw):
        _, chr_corrs = parse_genes_raw(synthetic_genes_raw)
        c1 = chr_corrs["1"]
        assert c1[1, 0] == pytest.approx(0.3)
        assert c1[2, 0] == pytest.approx(0.1)
        assert c1[2, 1] == pytest.approx(0.2)
        c2 = chr_corrs["2"]
        assert c2[1, 0] == pytest.approx(0.5)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_genes_raw(tmp_path / "nonexistent.genes.raw")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.genes.raw"
        path.write_text("# MEAN_SAMPLE_SIZE = 1000\n")
        with pytest.raises(ValueError, match="no gene records"):
            parse_genes_raw(path)

    def test_v110_format_with_covariates(self, tmp_path):
        """MAGMA v1.10+ format: GENE CHR START STOP NSNPS NPARAM NSAMP MAC ZSTAT."""
        content = (
            "# VERSION = 110\n"
            "# COVAR = NSAMP MAC\n"
            "1001 1 800000 900000 180 25 411000 144.0 1.41\n"
            "1002 1 860000 911000 136 25 397000 93.0 1.99 0.60\n"
            "1003 1 867000 920000 149 27 399000 113.0 2.01 0.49 0.89\n"
            "2001 2 500000 600000 200 30 405000 150.0 0.50\n"
            "2002 2 700000 800000 100 15 395000 80.0 -1.00 0.40\n"
        )
        path = tmp_path / "v110.genes.raw"
        path.write_text(content)
        gene_info, chr_corrs = parse_genes_raw(path)

        assert len(gene_info) == 5
        assert set(gene_info.columns) == {"gene", "chr", "nsnps", "nparam", "nsamp", "mac", "zstat"}

        row0 = gene_info.iloc[0]
        assert row0["gene"] == 1001
        assert row0["chr"] == "1"
        assert row0["nsnps"] == 180
        assert row0["nparam"] == 25
        assert row0["nsamp"] == 411000
        assert row0["mac"] == pytest.approx(144.0)
        assert row0["zstat"] == pytest.approx(1.41)

        assert chr_corrs["1"].shape == (3, 3)
        assert chr_corrs["2"].shape == (2, 2)
        np.testing.assert_array_almost_equal(np.diag(chr_corrs["1"]), [1.0, 1.0, 1.0])
        assert chr_corrs["1"][1, 0] == pytest.approx(0.60)
        assert chr_corrs["1"][2, 0] == pytest.approx(0.49)
        assert chr_corrs["1"][2, 1] == pytest.approx(0.89)
        assert chr_corrs["2"][1, 0] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------


class TestBuildChromosomeProjections:
    def test_dimensions(self, gene_info_and_corr):
        gene_info, chr_corrs = gene_info_and_corr
        projections, proj_indices = build_chromosome_projections(
            gene_info, chr_corrs, eigenvalue_threshold=0.01
        )
        for chrom in gene_info["chr"].unique():
            n_genes = (gene_info["chr"] == chrom).sum()
            proj = projections[chrom]
            assert proj.shape[0] == n_genes
            assert proj.shape[1] > 0
            assert proj.shape[1] <= n_genes

    def test_eigenvalue_clipping(self, gene_info_and_corr):
        gene_info, chr_corrs = gene_info_and_corr
        projections, _ = build_chromosome_projections(
            gene_info, chr_corrs, eigenvalue_threshold=0.5
        )
        for chrom, proj in projections.items():
            assert proj.shape[1] <= chr_corrs[chrom].shape[0]

    def test_all_eigenvalues_below_threshold(self, gene_info_and_corr):
        gene_info, chr_corrs = gene_info_and_corr
        projections, _ = build_chromosome_projections(
            gene_info, chr_corrs, eigenvalue_threshold=1e6
        )
        for chrom, proj in projections.items():
            assert proj.shape[1] >= 1


# ---------------------------------------------------------------------------
# Drug-drug correlation tests
# ---------------------------------------------------------------------------


class TestComputeDrugSetCorrelations:
    def test_output_shape(self, gene_info_and_corr, projections_and_indices,
                          synthetic_drug_results, synthetic_drug_gene_sets):
        gene_info, _ = gene_info_and_corr
        projections, proj_indices = projections_and_indices
        n = len(synthetic_drug_results)
        z, corr, inv, cond_num = compute_drug_set_correlations(
            gene_info, projections, proj_indices,
            synthetic_drug_results, synthetic_drug_gene_sets, eigenvalue_threshold=0.01,
        )
        assert z.shape == (n,)
        assert corr.shape == (n, n)
        assert inv.shape == (n, n)
        assert isinstance(cond_num, float)
        assert cond_num >= 1.0

    def test_symmetric(self, gene_info_and_corr, projections_and_indices,
                       synthetic_drug_results, synthetic_drug_gene_sets):
        gene_info, _ = gene_info_and_corr
        projections, proj_indices = projections_and_indices
        _, corr, _, _ = compute_drug_set_correlations(
            gene_info, projections, proj_indices,
            synthetic_drug_results, synthetic_drug_gene_sets, eigenvalue_threshold=0.01,
        )
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_diagonal_one(self, gene_info_and_corr, projections_and_indices,
                          synthetic_drug_results, synthetic_drug_gene_sets):
        gene_info, _ = gene_info_and_corr
        projections, proj_indices = projections_and_indices
        _, corr, _, _ = compute_drug_set_correlations(
            gene_info, projections, proj_indices,
            synthetic_drug_results, synthetic_drug_gene_sets, eigenvalue_threshold=0.01,
        )
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(corr.shape[0]))

    def test_known_overlap(self, gene_info_and_corr, projections_and_indices):
        gene_info, _ = gene_info_and_corr
        projections, proj_indices = projections_and_indices
        all_genes = gene_info["gene"].tolist()
        drug_results = pd.DataFrame({
            "drug_chembl_id": ["D1", "D2", "D3"],
            "magma_z": [1.0, 1.0, 1.0],
            "n_target_genes": [len(all_genes), len(all_genes), 1],
        })
        drug_gene_sets = {
            "D1": all_genes,
            "D2": all_genes,
            "D3": [all_genes[0]],
        }
        _, corr, _, _ = compute_drug_set_correlations(
            gene_info, projections, proj_indices,
            drug_results, drug_gene_sets, eigenvalue_threshold=0.01,
        )
        assert corr[0, 1] > 0.9


# ---------------------------------------------------------------------------
# ATC class extraction tests
# ---------------------------------------------------------------------------


class TestExtractATCClasses:
    def _make_df(self, atc_codes_list):
        return pd.DataFrame({
            "drug_chembl_id": [f"D{i}" for i in range(len(atc_codes_list))],
            "atc_codes": atc_codes_list,
        })

    def test_level_2(self):
        df = self._make_df([["N05AH01"], ["N05AH02"], ["N05AH03"], ["N05AH04"], ["N05AH05"]])
        classes = extract_atc_classes(df, [2], min_drugs_per_class=2)
        assert "N05" in classes
        assert len(classes["N05"]) == 5

    def test_level_3(self):
        df = self._make_df([["N05AH01"], ["N05AH02"], ["N05AH03"], ["C08CA01"], ["C08CA02"]])
        classes = extract_atc_classes(df, [3], min_drugs_per_class=2)
        assert "N05A" in classes
        assert "C08C" in classes
        assert len(classes["N05A"]) == 3
        assert len(classes["C08C"]) == 2

    def test_level_4(self):
        df = self._make_df([["N05AH01"], ["N05AH02"], ["N05AH03"]])
        classes = extract_atc_classes(df, [4], min_drugs_per_class=2)
        assert "N05AH" in classes

    def test_min_drugs_filter(self):
        df = self._make_df([["N05AH01"], ["N05AH02"]])
        classes = extract_atc_classes(df, [3], min_drugs_per_class=5)
        assert len(classes) == 0

    def test_multi_atc_drug(self):
        df = self._make_df([["N05AH01", "C08CA01"], ["N05AH02"], ["N05AH03"],
                            ["C08CA02"], ["C08CA03"]])
        classes = extract_atc_classes(df, [3], min_drugs_per_class=2)
        assert "N05A" in classes
        assert "C08C" in classes
        assert "D0" in classes["N05A"]
        assert "D0" in classes["C08C"]

    def test_no_atc_excluded(self):
        df = self._make_df([[], [], ["N05AH01"], ["N05AH02"], ["N05AH03"]])
        classes = extract_atc_classes(df, [3], min_drugs_per_class=2)
        for code, drug_ids in classes.items():
            assert "D0" not in drug_ids
            assert "D1" not in drug_ids

    def test_dedup_same_level(self):
        df = self._make_df([["N05AH01", "N05AH02"], ["N05AH03"], ["N05AH04"]])
        classes = extract_atc_classes(df, [3], min_drugs_per_class=2)
        assert "N05A" in classes
        assert len(classes["N05A"]) == 3


# ---------------------------------------------------------------------------
# Custom (curated) ATC class tests
# ---------------------------------------------------------------------------


class TestBuildCustomATCClasses:
    def _make_df(self, atc_codes_list):
        return pd.DataFrame({
            "drug_chembl_id": [f"D{i}" for i in range(len(atc_codes_list))],
            "atc_codes": atc_codes_list,
        })

    def test_membership_by_l5_code(self):
        df = self._make_df([
            ["N06AB03"], ["N06AB04"], ["N06AB05"], ["N06AB06"], ["N06AB08"], ["A01AA01"],
        ])
        cc = CustomATCClass(
            code="N06AB_SSRI", level=4, description="SSRIs",
            atc_members=["N06AB03", "N06AB04", "N06AB05", "N06AB06", "N06AB08"],
        )
        classes, meta = build_custom_atc_classes(df, [cc], min_drugs_per_class=5)
        assert classes["N06AB_SSRI"] == ["D0", "D1", "D2", "D3", "D4"]
        assert meta["N06AB_SSRI"] == {"level": 4, "description": "SSRIs"}

    def test_below_floor_skipped(self):
        df = self._make_df([["N06AB03"], ["N06AB04"], ["A01AA01"]])
        cc = CustomATCClass(code="N06AB_SSRI", atc_members=["N06AB03", "N06AB04"])
        classes, meta = build_custom_atc_classes(df, [cc], min_drugs_per_class=5)
        assert classes == {}
        assert meta == {}

    def test_multi_code_drug_counted_once(self):
        df = self._make_df([
            ["N06AB03", "N06AB04"], ["N06AB05"], ["N06AB06"], ["N06AB08"], ["N06AB10"],
        ])
        cc = CustomATCClass(
            code="N06AB_SSRI",
            atc_members=["N06AB03", "N06AB04", "N06AB05", "N06AB06", "N06AB08", "N06AB10"],
        )
        classes, _ = build_custom_atc_classes(df, [cc], min_drugs_per_class=5)
        assert classes["N06AB_SSRI"].count("D0") == 1
        assert len(classes["N06AB_SSRI"]) == 5

    def test_empty_custom_classes(self):
        df = self._make_df([["N06AB03"], ["N06AB04"]])
        classes, meta = build_custom_atc_classes(df, [], min_drugs_per_class=5)
        assert classes == {}
        assert meta == {}

    def test_empty_universe(self):
        df = pd.DataFrame({"drug_chembl_id": [], "atc_codes": []})
        cc = CustomATCClass(code="N06AB_SSRI", atc_members=["N06AB03"])
        classes, meta = build_custom_atc_classes(df, [cc], min_drugs_per_class=5)
        assert classes == {}
        assert meta == {}


# ---------------------------------------------------------------------------
# GLS regression tests
# ---------------------------------------------------------------------------


class TestGLSRegression:
    def test_enriched_class(self):
        rng = np.random.default_rng(42)
        N = 50
        y = rng.normal(0, 1, N)
        y[:10] = rng.normal(3, 0.5, 10)
        indicator = np.zeros(N)
        indicator[:10] = 1.0
        n_genes = np.full(N, 5.0)
        sigma_inv = np.eye(N)
        res = run_gls_regression(y, indicator, n_genes, sigma_inv, two_sided=False)
        assert res["beta"] > 0
        assert res["p_value"] < 0.05

    def test_null_class(self):
        rng = np.random.default_rng(42)
        N = 50
        y = rng.normal(0, 1, N)
        indicator = np.zeros(N)
        indicator[:10] = 1.0
        n_genes = np.full(N, 5.0)
        sigma_inv = np.eye(N)
        res = run_gls_regression(y, indicator, n_genes, sigma_inv, two_sided=False)
        assert res["p_value"] > 0.05

    def test_one_sided_vs_two_sided(self):
        rng = np.random.default_rng(42)
        N = 50
        y = rng.normal(0, 1, N)
        y[:10] = rng.normal(2, 0.5, 10)
        indicator = np.zeros(N)
        indicator[:10] = 1.0
        n_genes = np.full(N, 5.0)
        sigma_inv = np.eye(N)
        one = run_gls_regression(y, indicator, n_genes, sigma_inv, two_sided=False)
        two = run_gls_regression(y, indicator, n_genes, sigma_inv, two_sided=True)
        assert one["p_value"] < two["p_value"]

    def test_returns_all_keys(self):
        N = 20
        y = np.ones(N)
        indicator = np.zeros(N)
        indicator[:5] = 1.0
        n_genes = np.full(N, 3.0)
        sigma_inv = np.eye(N)
        res = run_gls_regression(y, indicator, n_genes, sigma_inv)
        assert set(res.keys()) == {"beta", "se", "t_stat", "p_value"}


# ---------------------------------------------------------------------------
# Wilcoxon tests
# ---------------------------------------------------------------------------


class TestWilcoxon:
    def test_enriched_class(self):
        rng = np.random.default_rng(42)
        N = 50
        y = rng.normal(0, 1, N)
        y[:10] = rng.normal(3, 0.5, 10)
        indicator = np.zeros(N)
        indicator[:10] = 1.0
        res = run_wilcoxon_test(y, indicator, two_sided=False)
        assert res["p_value"] < 0.05

    def test_null_class(self):
        rng = np.random.default_rng(42)
        N = 50
        y = rng.normal(0, 1, N)
        indicator = np.zeros(N)
        indicator[:10] = 1.0
        res = run_wilcoxon_test(y, indicator, two_sided=False)
        assert res["p_value"] > 0.05

    def test_degenerate(self):
        y = np.array([1.0, 2.0, 3.0])
        indicator = np.array([1.0, 0.0, 0.0])
        res = run_wilcoxon_test(y, indicator)
        assert np.isnan(res["u_stat"])


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------


class TestPermutation:
    def test_valid_pvalues(self):
        rng = np.random.default_rng(42)
        N = 30
        y = rng.normal(0, 1, N)
        drug_id_to_idx = {f"D{i}": i for i in range(N)}
        classes = {"CLS1": [f"D{i}" for i in range(5)]}
        n_genes = np.full(N, 5.0)
        sigma_inv = np.eye(N)
        res = run_permutation_test(
            y, classes, drug_id_to_idx, n_genes, sigma_inv,
            n_permutations=100, seed=42, two_sided=False,
        )
        for code, pval in res.items():
            assert 0 <= pval <= 1

    def test_indicator_permuted_n_genes_fixed(self):
        """Verify indicator varies across permutations while n_genes stays unchanged."""
        from unittest.mock import patch

        N = 20
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, N)
        drug_id_to_idx = {f"D{i}": i for i in range(N)}
        classes = {"CLS1": [f"D{i}" for i in range(5)]}
        n_genes = np.arange(1.0, N + 1.0)
        sigma_inv = np.eye(N)

        captured_calls: list[dict] = []
        original_gls = run_gls_regression

        def spy_gls(y_, indicator_, n_genes_, inv_, two_sided_=False):
            captured_calls.append({
                "indicator": indicator_.copy(),
                "n_genes": n_genes_.copy(),
            })
            return original_gls(y_, indicator_, n_genes_, inv_, two_sided_)

        with patch("repogen.analysis.atc_enrichment.run_gls_regression", side_effect=spy_gls):
            run_permutation_test(
                y, classes, drug_id_to_idx, n_genes, sigma_inv,
                n_permutations=20, seed=42, two_sided=False,
            )

        # First call is the observed (real indicator); rest are permutations
        assert len(captured_calls) == 21
        obs_indicator = captured_calls[0]["indicator"]
        perm_indicators = [c["indicator"] for c in captured_calls[1:]]

        # At least some permuted indicators differ from the observed
        n_different = sum(not np.array_equal(pi, obs_indicator) for pi in perm_indicators)
        assert n_different > 0, "Permuted indicators should differ from observed"

        # n_genes must be identical (same object values) in every call
        for call in captured_calls:
            np.testing.assert_array_equal(call["n_genes"], n_genes)

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(42)
        N = 30
        y = rng.normal(0, 1, N)
        drug_id_to_idx = {f"D{i}": i for i in range(N)}
        classes = {"CLS1": [f"D{i}" for i in range(5)]}
        n_genes = np.full(N, 5.0)
        sigma_inv = np.eye(N)
        r1 = run_permutation_test(y, classes, drug_id_to_idx, n_genes, sigma_inv,
                                  n_permutations=100, seed=123, two_sided=False)
        r2 = run_permutation_test(y, classes, drug_id_to_idx, n_genes, sigma_inv,
                                  n_permutations=100, seed=123, two_sided=False)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Assembly tests
# ---------------------------------------------------------------------------


class TestAssembleResults:
    def test_basic(self):
        results = [
            {"atc_code": "N05A", "atc_level": 3, "atc_description": "Antipsychotics",
             "n_drugs": 8, "n_genes_total": 50, "mean_targets_per_drug": 6.25,
             "median_targets_per_drug": 6.0, "annotation_bias_risk": "low",
             "borderline_power": False, "gls_beta": 1.5, "gls_se": 0.3,
             "gls_t": 5.0, "gls_p": 0.001, "wilcoxon_u": 100.0,
             "wilcoxon_p": 0.01, "wilcoxon_auc": 0.625,
             "contributing_drugs": "CHEMBL1,CHEMBL2",
             "contributing_drug_names": "DrugA,DrugB"},
            {"atc_code": "C08C", "atc_level": 3, "atc_description": "Selective CCBs",
             "n_drugs": 7, "n_genes_total": 35, "mean_targets_per_drug": 5.0,
             "median_targets_per_drug": 5.0, "annotation_bias_risk": "low",
             "borderline_power": True, "gls_beta": 0.1, "gls_se": 0.5,
             "gls_t": 0.2, "gls_p": 0.42, "wilcoxon_u": 50.0,
             "wilcoxon_p": 0.5, "wilcoxon_auc": 0.5,
             "contributing_drugs": "CHEMBL3,CHEMBL4",
             "contributing_drug_names": "DrugC,DrugD"},
        ]
        df = assemble_atc_results(results, fdr_method="fdr_bh", fdr_threshold=0.05)
        assert len(df) == 2
        assert "gls_fdr" in df.columns
        assert "wilcoxon_fdr" in df.columns
        assert "wilcoxon_auc" in df.columns
        assert "perm_p" in df.columns
        assert "contributing_drug_names" in df.columns
        assert df.iloc[0]["gls_p"] <= df.iloc[1]["gls_p"]

    def test_empty_input(self):
        df = assemble_atc_results([], fdr_method="fdr_bh", fdr_threshold=0.05)
        assert df.empty

    def test_with_permutation(self):
        results = [
            {"atc_code": "N05A", "atc_level": 3, "atc_description": "",
             "n_drugs": 5, "n_genes_total": 30, "mean_targets_per_drug": 6.0,
             "median_targets_per_drug": 6.0, "annotation_bias_risk": "low",
             "borderline_power": True, "gls_beta": 1.0, "gls_se": 0.3,
             "gls_t": 3.3, "gls_p": 0.01, "wilcoxon_u": 80.0,
             "wilcoxon_p": 0.02, "wilcoxon_auc": 0.64,
             "contributing_drugs": "D1,D2",
             "contributing_drug_names": "Drug1,Drug2"},
        ]
        perm = {"N05A": 0.005}
        df = assemble_atc_results(results, "fdr_bh", 0.05, perm_results=perm)
        assert df.iloc[0]["perm_p"] == pytest.approx(0.005)

    def test_wilcoxon_auc_nan_passthrough(self):
        """NaN wilcoxon_auc (degenerate case) passes through assembly intact."""
        results = [
            {"atc_code": "X01A", "atc_level": 3, "atc_description": "",
             "n_drugs": 3, "n_genes_total": 10, "mean_targets_per_drug": 3.0,
             "median_targets_per_drug": 3.0, "annotation_bias_risk": "low",
             "borderline_power": True, "gls_beta": 0.5, "gls_se": 0.2,
             "gls_t": 2.5, "gls_p": 0.05, "wilcoxon_u": np.nan,
             "wilcoxon_p": np.nan, "wilcoxon_auc": np.nan,
             "contributing_drugs": "D1,D2,D3",
             "contributing_drug_names": "Drug1,Drug2,Drug3"},
        ]
        df = assemble_atc_results(results, "fdr_bh", 0.05)
        assert "wilcoxon_auc" in df.columns
        assert np.isnan(df.iloc[0]["wilcoxon_auc"])

    def test_contributing_drug_names_preserved(self):
        """assemble_atc_results passes through contributing_drug_names."""
        results = [
            {"atc_code": "X01A", "atc_level": 3, "atc_description": "",
             "n_drugs": 3, "n_genes_total": 10, "mean_targets_per_drug": 3.0,
             "median_targets_per_drug": 3.0, "annotation_bias_risk": "low",
             "borderline_power": True, "gls_beta": 0.5, "gls_se": 0.2,
             "gls_t": 2.5, "gls_p": 0.05, "wilcoxon_u": 10.0,
             "wilcoxon_p": 0.1, "wilcoxon_auc": 0.55,
             "contributing_drugs": "CHEMBL1,CHEMBL2,CHEMBL3",
             "contributing_drug_names": "ASPIRIN,IBUPROFEN,NAPROXEN"},
        ]
        df = assemble_atc_results(results, "fdr_bh", 0.05)
        assert df.iloc[0]["contributing_drug_names"] == "ASPIRIN,IBUPROFEN,NAPROXEN"
        assert df.iloc[0]["contributing_drugs"] == "CHEMBL1,CHEMBL2,CHEMBL3"


# ---------------------------------------------------------------------------
# Drug gene-set file reader
# ---------------------------------------------------------------------------


class TestReadDrugGenesetFile:
    def test_basic(self, drug_geneset_file):
        sets = _read_drug_geneset_file(drug_geneset_file)
        assert len(sets) == 30
        for drug_id, genes in sets.items():
            assert all(isinstance(g, int) for g in genes)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestATCEnrichmentConfig:
    def test_defaults(self):
        cfg = ATCEnrichmentConfig()
        assert cfg.atc_levels == [2, 3]
        assert cfg.min_drugs_per_class == 5
        assert cfg.two_sided is False
        assert cfg.fdr_method == "fdr_bh"
        assert cfg.eigenvalue_threshold == 0.1
        assert cfg.permutation_test is False

    def test_invalid_atc_level(self):
        with pytest.raises(ValidationError):
            ATCEnrichmentConfig(atc_levels=[0, 2])

    def test_invalid_fdr_method(self):
        with pytest.raises(ValidationError):
            ATCEnrichmentConfig(fdr_method="invalid")

    def test_level_dedup_and_sort(self):
        cfg = ATCEnrichmentConfig(atc_levels=[3, 2, 3, 1])
        assert cfg.atc_levels == [1, 2, 3]

    def test_in_pipeline_config(self):
        cfg = PipelineConfig(
            study=StudyConfig(name="test", gwas_input=Path("test.tsv")),
            atc_enrichment=ATCEnrichmentConfig(atc_levels=[1, 2, 3, 4]),
        )
        assert cfg.atc_enrichment.atc_levels == [1, 2, 3, 4]

    def test_custom_classes_default_empty(self):
        cfg = ATCEnrichmentConfig()
        assert cfg.custom_classes == []

    def test_custom_classes_valid(self):
        cfg = ATCEnrichmentConfig(custom_classes=[
            {"code": "N06AB_SSRI", "level": 4, "description": "SSRIs",
             "atc_members": ["n06ab03", " N06AB04 "]},
        ])
        cc = cfg.custom_classes[0]
        assert cc.code == "N06AB_SSRI"
        # members are stripped, uppercased, deduped and sorted
        assert cc.atc_members == ["N06AB03", "N06AB04"]

    def test_custom_class_empty_members_rejected(self):
        with pytest.raises(ValidationError):
            CustomATCClass(code="N06AB_SSRI", atc_members=[])

    def test_custom_class_blank_code_rejected(self):
        with pytest.raises(ValidationError):
            CustomATCClass(code="   ", atc_members=["N06AB03"])

    def test_custom_class_level_bounds(self):
        with pytest.raises(ValidationError):
            CustomATCClass(code="X", level=5, atc_members=["N06AB03"])

    def test_duplicate_custom_codes_rejected(self):
        with pytest.raises(ValidationError):
            ATCEnrichmentConfig(custom_classes=[
                {"code": "DUP", "atc_members": ["N06AB03"]},
                {"code": "DUP", "atc_members": ["N06AB04"]},
            ])


# ---------------------------------------------------------------------------
# Integration / End-to-end tests
# ---------------------------------------------------------------------------


class TestRunATCEnrichmentEndToEnd:
    def test_full_pipeline(self, tmp_path, synthetic_genes_raw, synthetic_drug_results,
                           drug_geneset_file):
        results_path = tmp_path / "drug_results.parquet"
        synthetic_drug_results.to_parquet(results_path, index=False)

        config = ATCEnrichmentConfig(
            atc_levels=[3],
            min_drugs_per_class=2,
            eigenvalue_threshold=0.01,
        )

        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="test_study",
        )

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "gls_fdr" in result.columns
            assert "wilcoxon_fdr" in result.columns
            assert "perm_p" in result.columns

        out_dir = tmp_path / "test_study" / "atc_enrichment"
        assert (out_dir / "atc_enrichment_results.parquet").is_file()
        assert (out_dir / "atc_enrichment_metadata.json").is_file()

        with open(out_dir / "atc_enrichment_metadata.json") as f:
            meta = json.load(f)
        assert meta["result_type"] == "atc_enrichment"
        assert meta["study"] == "test_study"
        assert "n_classes_tested" in meta["parameters"]
        assert meta["parameters"]["correlation_matrix_condition_number"] is not None
        assert meta["parameters"]["correlation_matrix_condition_number"] >= 1.0

    def test_custom_class_end_to_end(self, tmp_path, synthetic_genes_raw,
                                     synthetic_drug_results, drug_geneset_file):
        results_path = tmp_path / "drug_results.parquet"
        synthetic_drug_results.to_parquet(results_path, index=False)

        # Curated class over the high-Z N05AH drugs (CHEMBL1000..CHEMBL1004).
        custom = CustomATCClass(
            code="N05AH_CURATED",
            level=4,
            description="Curated antipsychotic subset",
            atc_members=[f"N05AH{i:02d}" for i in range(5)],
        )
        config = ATCEnrichmentConfig(
            atc_levels=[3],
            min_drugs_per_class=2,
            eigenvalue_threshold=0.01,
            custom_classes=[custom],
        )

        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="custom_study",
        )

        assert "N05AH_CURATED" in set(result["atc_code"])
        row = result[result["atc_code"] == "N05AH_CURATED"].iloc[0]
        # Display metadata comes from the custom definition, not code inference.
        assert int(row["atc_level"]) == 4
        assert row["atc_description"] == "Curated antipsychotic subset"
        assert row["n_drugs"] == 5
        assert "gls_fdr" in result.columns

        out_dir = tmp_path / "custom_study" / "atc_enrichment"
        with open(out_dir / "atc_enrichment_metadata.json") as f:
            meta = json.load(f)
        assert meta["parameters"]["n_custom_classes"] == 1
        assert meta["parameters"]["custom_class_codes"] == ["N05AH_CURATED"]

    def test_empty_atc(self, tmp_path, synthetic_genes_raw, drug_geneset_file):
        drugs = pd.DataFrame({
            "drug_chembl_id": [f"D{i}" for i in range(10)],
            "drug_name": [f"Drug{i}" for i in range(10)],
            "magma_z": np.random.default_rng(42).normal(0, 1, 10),
            "magma_beta": np.zeros(10),
            "magma_se": np.ones(10),
            "magma_p": np.ones(10) * 0.5,
            "n_target_genes": [5] * 10,
            "atc_codes": [[]] * 10,
            "drug_inchikey": [""] * 10,
            "drug_pubchem_cid": [""] * 10,
            "max_phase": [4] * 10,
            "mechanism_of_action": [""] * 10,
        })
        results_path = tmp_path / "empty_atc.parquet"
        drugs.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig(atc_levels=[2, 3], min_drugs_per_class=2)

        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="empty_test",
        )
        assert result.empty

        out_dir = tmp_path / "empty_test" / "atc_enrichment"
        assert (out_dir / "atc_enrichment_results.parquet").is_file()
        assert (out_dir / "atc_enrichment_metadata.json").is_file()

        with open(out_dir / "atc_enrichment_metadata.json") as f:
            meta = json.load(f)
        assert meta["result_type"] == "atc_enrichment"
        assert meta["parameters"]["n_classes_tested"] == 0
        assert meta["parameters"]["n_drugs_with_atc"] == 0
        assert meta["parameters"]["correlation_matrix_condition_number"] is None
        assert meta["summary"]["n_significant_gls"] == 0

    def test_empty_atc_classes_with_atc_drugs(self, tmp_path, synthetic_genes_raw,
                                               drug_geneset_file):
        """Drugs have ATC codes but no class meets min_drugs_per_class.

        Matrix IS computed so condition number should be non-null.
        """
        drugs = pd.DataFrame({
            "drug_chembl_id": [f"D{i}" for i in range(10)],
            "drug_name": [f"Drug{i}" for i in range(10)],
            "magma_z": np.random.default_rng(42).normal(0, 1, 10),
            "magma_beta": np.zeros(10),
            "magma_se": np.ones(10),
            "magma_p": np.ones(10) * 0.5,
            "n_target_genes": [5] * 10,
            "atc_codes": [[f"A0{i}"] for i in range(10)],
            "drug_inchikey": [""] * 10,
            "drug_pubchem_cid": [""] * 10,
            "max_phase": [4] * 10,
            "mechanism_of_action": [""] * 10,
        })
        results_path = tmp_path / "atc_no_class.parquet"
        drugs.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig(atc_levels=[2, 3], min_drugs_per_class=100)

        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="no_class_test",
        )
        assert result.empty

        out_dir = tmp_path / "no_class_test" / "atc_enrichment"
        assert (out_dir / "atc_enrichment_metadata.json").is_file()

        with open(out_dir / "atc_enrichment_metadata.json") as f:
            meta = json.load(f)
        assert meta["parameters"]["n_classes_tested"] == 0
        assert meta["parameters"]["n_drugs_with_atc"] == 10
        assert meta["parameters"]["correlation_matrix_condition_number"] is not None
        assert meta["parameters"]["correlation_matrix_condition_number"] >= 1.0

    def test_output_columns(self, tmp_path, synthetic_genes_raw, synthetic_drug_results,
                            drug_geneset_file):
        results_path = tmp_path / "drug_results.parquet"
        synthetic_drug_results.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig(
            atc_levels=[3],
            min_drugs_per_class=2,
            eigenvalue_threshold=0.01,
        )

        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="col_test",
        )

        if len(result) > 0:
            expected_cols = {
                "atc_code", "atc_level", "atc_description", "n_drugs",
                "n_genes_total", "mean_targets_per_drug", "median_targets_per_drug",
                "annotation_bias_risk", "borderline_power",
                "gls_beta", "gls_se", "gls_t", "gls_p", "gls_fdr",
                "wilcoxon_u", "wilcoxon_p", "wilcoxon_auc", "wilcoxon_fdr",
                "perm_p", "contributing_drugs", "contributing_drug_names",
            }
            assert expected_cols == set(result.columns)

    def test_wilcoxon_auc_bounded(self, tmp_path, synthetic_genes_raw,
                                  synthetic_drug_results, drug_geneset_file):
        """wilcoxon_auc must be in [0, 1] for all non-NaN rows."""
        results_path = tmp_path / "drug_results.parquet"
        synthetic_drug_results.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig(
            atc_levels=[3], min_drugs_per_class=2, eigenvalue_threshold=0.01,
        )
        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="auc_test",
        )
        if len(result) == 0:
            pytest.skip("No ATC classes produced")
        assert "wilcoxon_auc" in result.columns
        valid = result["wilcoxon_auc"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_wilcoxon_auc_formula(self, tmp_path, synthetic_genes_raw,
                                  synthetic_drug_results, drug_geneset_file):
        """wilcoxon_auc == wilcoxon_u / (n_in * n_out) for each class."""
        results_path = tmp_path / "drug_results.parquet"
        synthetic_drug_results.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig(
            atc_levels=[3], min_drugs_per_class=2, eigenvalue_threshold=0.01,
            atc_universe_mode="all_drugs",
        )
        n_total_drugs = len(synthetic_drug_results)
        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="auc_formula_test",
        )
        if len(result) == 0:
            pytest.skip("No ATC classes produced")
        for _, row in result.iterrows():
            if np.isnan(row["wilcoxon_auc"]):
                continue
            n_in = row["n_drugs"]
            n_out = n_total_drugs - n_in
            expected = row["wilcoxon_u"] / (n_in * n_out)
            assert row["wilcoxon_auc"] == pytest.approx(expected, rel=1e-9)

    def test_contributing_drug_names_mapping(
        self, tmp_path, synthetic_genes_raw, synthetic_drug_results, drug_geneset_file,
    ):
        """contributing_drug_names must contain the resolved drug names,
        aligned to the same sorted-ID order as contributing_drugs."""
        results_path = tmp_path / "drug_results.parquet"
        synthetic_drug_results.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig(
            atc_levels=[3],
            min_drugs_per_class=2,
            eigenvalue_threshold=0.01,
        )
        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="name_test",
        )
        if len(result) == 0:
            pytest.skip("No ATC classes produced (synthetic data too small)")

        id_to_name = dict(zip(
            synthetic_drug_results["drug_chembl_id"],
            synthetic_drug_results["drug_name"],
        ))
        for _, row in result.iterrows():
            ids = row["contributing_drugs"].split(",")
            names = row["contributing_drug_names"].split(",")
            assert len(ids) == len(names), "ID/name count mismatch"
            assert ids == sorted(ids), "IDs should be sorted"
            for cid, name in zip(ids, names):
                assert name == id_to_name[cid], f"{cid} -> expected {id_to_name[cid]}, got {name}"

    def test_contributing_drug_names_fallback(
        self, tmp_path, synthetic_genes_raw, drug_geneset_file,
    ):
        """When drug_name is blank, the ID should be used as fallback."""
        drugs = []
        for i in range(30):
            if i < 8:
                z = 2.5
                atc = [f"N05AH{i:02d}"]
            elif i < 15:
                z = 0.0
                atc = [f"C08CA{i:02d}"]
            else:
                z = 0.0
                atc = [] if i > 25 else [f"A01A{chr(65 + i % 6)}{i:02d}"]
            name = "" if i == 0 else f"Drug_{i}"
            drugs.append({
                "drug_chembl_id": f"CHEMBL{1000 + i}",
                "drug_name": name,
                "magma_beta": z * 0.3, "magma_se": 0.3,
                "magma_p": 0.5, "magma_z": z,
                "n_target_genes": 5, "atc_codes": atc,
                "drug_inchikey": "", "drug_pubchem_cid": "",
                "max_phase": 4, "mechanism_of_action": "",
            })
        df = pd.DataFrame(drugs)
        results_path = tmp_path / "drug_results.parquet"
        df.to_parquet(results_path, index=False)

        config = ATCEnrichmentConfig(
            atc_levels=[3], min_drugs_per_class=2, eigenvalue_threshold=0.01,
        )
        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="fallback_test",
        )
        if len(result) == 0:
            pytest.skip("No ATC classes produced")

        for _, row in result.iterrows():
            ids = row["contributing_drugs"].split(",")
            names = row["contributing_drug_names"].split(",")
            for cid, name in zip(ids, names):
                if cid == "CHEMBL1000":
                    assert name == "CHEMBL1000", "blank name should fall back to ID"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_class(self, tmp_path, synthetic_genes_raw, drug_geneset_file):
        drugs = pd.DataFrame({
            "drug_chembl_id": [f"CHEMBL{1000 + i}" for i in range(10)],
            "drug_name": [f"Drug{i}" for i in range(10)],
            "magma_z": np.random.default_rng(42).normal(0, 1, 10),
            "magma_beta": np.zeros(10),
            "magma_se": np.ones(10),
            "magma_p": np.ones(10) * 0.5,
            "n_target_genes": [5] * 10,
            "atc_codes": [["N05AH01"]] * 5 + [[]] * 5,
            "drug_inchikey": [""] * 10,
            "drug_pubchem_cid": [""] * 10,
            "max_phase": [4] * 10,
            "mechanism_of_action": [""] * 10,
        })
        results_path = tmp_path / "single.parquet"
        drugs.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig(atc_levels=[3], min_drugs_per_class=2,
                                     eigenvalue_threshold=0.01)
        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="single_test",
        )
        assert len(result) == 1

    def test_missing_drug_results_file(self, tmp_path, synthetic_genes_raw, drug_geneset_file):
        config = ATCEnrichmentConfig()
        with pytest.raises(FileNotFoundError):
            run_atc_enrichment(
                drug_results_path=tmp_path / "nonexistent.parquet",
                genes_raw_path=synthetic_genes_raw,
                drug_geneset_path=drug_geneset_file,
                config=config,
                output_dir=tmp_path,
                study_name="test",
            )

    def test_missing_required_column(self, tmp_path, synthetic_genes_raw, drug_geneset_file):
        bad_df = pd.DataFrame({"drug_chembl_id": ["D1"], "magma_z": [1.0]})
        results_path = tmp_path / "bad.parquet"
        bad_df.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig()
        with pytest.raises(ValueError, match="missing required columns"):
            run_atc_enrichment(
                drug_results_path=results_path,
                genes_raw_path=synthetic_genes_raw,
                drug_geneset_path=drug_geneset_file,
                config=config,
                output_dir=tmp_path,
                study_name="test",
            )

    def test_derives_magma_z_from_beta_se(
        self, tmp_path, synthetic_drug_results, synthetic_genes_raw,
        drug_geneset_file,
    ):
        """ATC should derive magma_z when absent but magma_beta/se present."""
        df = synthetic_drug_results.copy()
        df["magma_beta_se"] = df.pop("magma_se")
        df = df.drop(columns=["magma_z"])
        results_path = tmp_path / "no_z.parquet"
        df.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig()
        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=drug_geneset_file,
            config=config,
            output_dir=tmp_path,
            study_name="test",
        )
        assert not result.empty

    def test_zero_se_raises_when_no_magma_z(
        self, tmp_path, synthetic_genes_raw, drug_geneset_file,
    ):
        """Zero SE must raise actionable error when magma_z is absent."""
        df = pd.DataFrame({
            "drug_chembl_id": ["D1", "D2"],
            "magma_beta": [0.5, -0.3],
            "magma_beta_se": [0.1, 0.0],
            "n_target_genes": [5, 4],
            "atc_codes": [["N05AH01"], ["C08CA01"]],
        })
        results_path = tmp_path / "zero_se.parquet"
        df.to_parquet(results_path, index=False)
        config = ATCEnrichmentConfig()
        with pytest.raises(ValueError, match="zero or NaN"):
            run_atc_enrichment(
                drug_results_path=results_path,
                genes_raw_path=synthetic_genes_raw,
                drug_geneset_path=drug_geneset_file,
                config=config,
                output_dir=tmp_path,
                study_name="test",
            )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_infer_atc_level(self):
        assert _infer_atc_level("N") == 1
        assert _infer_atc_level("N05") == 2
        assert _infer_atc_level("N05A") == 3
        assert _infer_atc_level("N05AH") == 4
        assert _infer_atc_level("N05AH02") == 5

    def test_atc_level_prefix_consistency(self):
        assert ATC_LEVEL_PREFIX[1] == 1
        assert ATC_LEVEL_PREFIX[2] == 3
        assert ATC_LEVEL_PREFIX[3] == 4
        assert ATC_LEVEL_PREFIX[4] == 5
        assert ATC_LEVEL_PREFIX[5] == 7


# ---------------------------------------------------------------------------
# ATC-code helper tests
# ---------------------------------------------------------------------------


class TestDeserializeAtcCodes:
    """Unit tests for _deserialize_atc_codes normalizer."""

    def test_json_list_string(self):
        assert _deserialize_atc_codes('["N06A", "N06AX"]') == ["N06A", "N06AX"]

    def test_plain_atc_string(self):
        assert _deserialize_atc_codes("N06AX11") == ["N06AX11"]

    def test_json_scalar_string(self):
        assert _deserialize_atc_codes('"N06AX11"') == ["N06AX11"]

    def test_none(self):
        assert _deserialize_atc_codes(None) == []

    def test_nan(self):
        assert _deserialize_atc_codes(float("nan")) == []

    def test_list_passthrough(self):
        assert _deserialize_atc_codes(["A03", "C10A"]) == ["A03", "C10A"]

    def test_tuple_passthrough(self):
        assert _deserialize_atc_codes(("A03",)) == ["A03"]

    def test_ndarray_passthrough(self):
        result = _deserialize_atc_codes(np.array(["C10A", "N05A"]))
        assert result == ["C10A", "N05A"]

    def test_empty_string(self):
        assert _deserialize_atc_codes("") == []

    def test_whitespace_string(self):
        assert _deserialize_atc_codes("   ") == []

    def test_malformed_json(self):
        assert _deserialize_atc_codes("not{valid json[") == ["not{valid json["]

    def test_pd_na(self):
        assert _deserialize_atc_codes(pd.NA) == []

    def test_empty_list(self):
        assert _deserialize_atc_codes([]) == []


class TestHasAtcCodes:
    """Unit tests for _has_atc_codes checker."""

    def test_valid_list(self):
        assert _has_atc_codes(["N06A"]) is True

    def test_multiple_codes(self):
        assert _has_atc_codes(["N06A", "N06AX11"]) is True

    def test_empty_list(self):
        assert _has_atc_codes([]) is False

    def test_none(self):
        assert _has_atc_codes(None) is False

    def test_list_of_none(self):
        assert _has_atc_codes([None]) is False

    def test_list_of_nan(self):
        assert _has_atc_codes([float("nan")]) is False

    def test_list_of_empty_string(self):
        assert _has_atc_codes([""]) is False

    def test_list_of_whitespace(self):
        assert _has_atc_codes(["  "]) is False

    def test_ndarray_valid(self):
        assert _has_atc_codes(np.array(["C10A"])) is True

    def test_pd_na(self):
        assert _has_atc_codes(pd.NA) is False

    def test_tuple_valid(self):
        assert _has_atc_codes(("A03",)) is True

    def test_nan_float(self):
        assert _has_atc_codes(float("nan")) is False

    def test_bare_string_rejected(self):
        assert _has_atc_codes("N06A") is False


# ---------------------------------------------------------------------------
# ATC universe mode tests
# ---------------------------------------------------------------------------


class TestATCUniverseMode:
    """Tests for atc_universe_mode filtering and metadata provenance."""

    @pytest.fixture()
    def universe_fixture(self, tmp_path, synthetic_genes_raw):
        """20 drugs: 5 with ATC codes (N05A), 15 without."""
        rng = np.random.default_rng(99)
        drugs = []
        for i in range(20):
            z = rng.normal(1.0 if i < 5 else 0.0, 1.0)
            atc = [f"N05AH{i:02d}"] if i < 5 else []
            drugs.append({
                "drug_chembl_id": f"CHEMBL{2000 + i}",
                "drug_name": f"UnivDrug_{i}",
                "magma_z": z,
                "n_target_genes": int(rng.integers(3, 10)),
                "atc_codes": atc,
            })
        df = pd.DataFrame(drugs)
        results_path = tmp_path / "univ_drug_results.parquet"
        df.to_parquet(results_path, index=False)

        gene_pool = [1001, 1002, 1003, 1004, 2001, 2002]
        geneset_path = tmp_path / "univ_genesets.txt"
        with open(geneset_path, "w") as f:
            for i in range(20):
                n_genes = rng.integers(2, 5)
                chosen = rng.choice(gene_pool, size=n_genes, replace=False)
                line = f"CHEMBL{2000 + i}\t" + "\t".join(str(g) for g in chosen)
                f.write(line + "\n")

        return results_path, synthetic_genes_raw, geneset_path

    def test_annotated_only_filters_universe(self, tmp_path, universe_fixture):
        """In annotated_only mode, only ATC-annotated drugs enter the test."""
        from unittest.mock import patch

        results_path, genes_raw, geneset_path = universe_fixture
        config = ATCEnrichmentConfig(
            atc_universe_mode="annotated_only",
            min_drugs_per_class=2,
            atc_levels=[3],
        )

        captured_args = []

        original_gls = run_gls_regression

        def mock_gls(drug_z, indicator, *args, **kwargs):
            captured_args.append((drug_z.copy(), indicator.copy()))
            return original_gls(drug_z, indicator, *args, **kwargs)

        with patch(
            "repogen.analysis.atc_enrichment.run_gls_regression",
            side_effect=mock_gls,
        ):
            run_atc_enrichment(
                drug_results_path=results_path,
                genes_raw_path=genes_raw,
                drug_geneset_path=geneset_path,
                config=config,
                output_dir=tmp_path,
                study_name="test_univ",
            )

        assert len(captured_args) > 0, "GLS should have been called"
        drug_z, indicator = captured_args[0]
        assert drug_z.shape[0] == 5, "Universe should contain only 5 ATC-annotated drugs"
        assert indicator.shape[0] == 5

    def test_all_drugs_preserves_full_universe(self, tmp_path, universe_fixture):
        """In all_drugs mode, all drugs enter the test universe."""
        from unittest.mock import patch

        results_path, genes_raw, geneset_path = universe_fixture
        config = ATCEnrichmentConfig(
            atc_universe_mode="all_drugs",
            min_drugs_per_class=2,
            atc_levels=[3],
        )

        captured_args = []

        original_gls = run_gls_regression

        def mock_gls(drug_z, indicator, *args, **kwargs):
            captured_args.append((drug_z.copy(), indicator.copy()))
            return original_gls(drug_z, indicator, *args, **kwargs)

        with patch(
            "repogen.analysis.atc_enrichment.run_gls_regression",
            side_effect=mock_gls,
        ):
            run_atc_enrichment(
                drug_results_path=results_path,
                genes_raw_path=genes_raw,
                drug_geneset_path=geneset_path,
                config=config,
                output_dir=tmp_path,
                study_name="test_univ",
            )

        assert len(captured_args) > 0, "GLS should have been called"
        drug_z, indicator = captured_args[0]
        assert drug_z.shape[0] == 20, "Universe should contain all 20 drugs"
        assert indicator.shape[0] == 20

    def test_annotated_only_metadata_provenance(self, tmp_path, universe_fixture):
        """Metadata should contain correct provenance fields."""
        results_path, genes_raw, geneset_path = universe_fixture
        config = ATCEnrichmentConfig(
            atc_universe_mode="annotated_only",
            min_drugs_per_class=2,
            atc_levels=[3],
        )

        run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=genes_raw,
            drug_geneset_path=geneset_path,
            config=config,
            output_dir=tmp_path,
            study_name="test_meta",
        )

        meta_path = tmp_path / "test_meta" / "atc_enrichment" / "atc_enrichment_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)

        params = meta["parameters"]
        assert params["atc_universe_mode"] == "annotated_only"
        assert params["n_drugs_input_total"] == 20
        assert params["n_drugs_tested"] == 5
        assert params["n_drugs_with_atc"] == 5

    def test_all_drugs_metadata_provenance(self, tmp_path, universe_fixture):
        """Metadata for all_drugs mode reports full universe."""
        results_path, genes_raw, geneset_path = universe_fixture
        config = ATCEnrichmentConfig(
            atc_universe_mode="all_drugs",
            min_drugs_per_class=2,
            atc_levels=[3],
        )

        run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=genes_raw,
            drug_geneset_path=geneset_path,
            config=config,
            output_dir=tmp_path,
            study_name="test_meta_all",
        )

        meta_path = tmp_path / "test_meta_all" / "atc_enrichment" / "atc_enrichment_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        params = meta["parameters"]
        assert params["atc_universe_mode"] == "all_drugs"
        assert params["n_drugs_input_total"] == 20
        assert params["n_drugs_tested"] == 20
        assert params["n_drugs_with_atc"] == 5

    def test_mixed_type_atc_codes_no_crash(self, tmp_path, synthetic_genes_raw):
        """Mixed-type atc_codes column should not crash and filters correctly."""
        # Simulate mixed ATC storage as all-string (object dtype) column
        # which is what Parquet round-trips produce when types are heterogeneous.
        drugs = [
            {"drug_chembl_id": "D1", "drug_name": "A", "magma_z": 1.0,
             "n_target_genes": 4, "atc_codes": '["N05AH01"]'},
            {"drug_chembl_id": "D2", "drug_name": "B", "magma_z": 0.5,
             "n_target_genes": 3, "atc_codes": "C08CA01"},
            {"drug_chembl_id": "D3", "drug_name": "C", "magma_z": -0.2,
             "n_target_genes": 5, "atc_codes": None},
            {"drug_chembl_id": "D4", "drug_name": "D", "magma_z": 0.1,
             "n_target_genes": 3, "atc_codes": ""},
            {"drug_chembl_id": "D5", "drug_name": "E", "magma_z": 1.5,
             "n_target_genes": 4, "atc_codes": '["A03FA"]'},
            {"drug_chembl_id": "D6", "drug_name": "F", "magma_z": 0.8,
             "n_target_genes": 3, "atc_codes": "[]"},
            {"drug_chembl_id": "D7", "drug_name": "G", "magma_z": -0.1,
             "n_target_genes": 4, "atc_codes": '[""]'},
        ]
        df = pd.DataFrame(drugs)
        results_path = tmp_path / "mixed_atc.parquet"
        df.to_parquet(results_path, index=False)

        gene_pool = [1001, 1002, 1003, 1004, 2001, 2002]
        geneset_path = tmp_path / "mixed_genesets.txt"
        with open(geneset_path, "w") as f:
            for drug in drugs:
                f.write(f"{drug['drug_chembl_id']}\t1001\t1002\t2001\n")

        config = ATCEnrichmentConfig(
            atc_universe_mode="annotated_only",
            min_drugs_per_class=2,
            atc_levels=[3],
        )

        result = run_atc_enrichment(
            drug_results_path=results_path,
            genes_raw_path=synthetic_genes_raw,
            drug_geneset_path=geneset_path,
            config=config,
            output_dir=tmp_path,
            study_name="test_mixed",
        )

        meta_path = tmp_path / "test_mixed" / "atc_enrichment" / "atc_enrichment_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        # D1 (JSON list), D2 (plain string), D5 (list) should pass.
        # D3 (None), D4 (""), D6 ([None]), D7 ([""]) should be excluded.
        assert meta["parameters"]["n_drugs_with_atc"] == 3
        assert meta["parameters"]["n_drugs_input_total"] == 7
        assert meta["parameters"]["n_drugs_tested"] == 3
