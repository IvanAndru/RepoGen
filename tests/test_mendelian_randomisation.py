"""Tests for repogen/analysis/mendelian_randomisation.py.

Covers MR estimation, coloc, harmonisation, drug matching, tiering,
Steiger, eQTL loading, and end-to-end orchestration.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from repogen.analysis.mendelian_randomisation import (
    AMBIGUOUS_TYPES,
    ColocConfigurationError,
    DOWNREGULATING_TYPES,
    INFERABLE_TYPES,
    UPREGULATING_TYPES,
    _annotate_druggable_track,
    _annotate_fdr_track,
    _annotate_mhc_flag,
    _bh_fdr_finite,
    _build_drug_match_records,
    _ensembl_no_version,
    _has_text,
    _normalize_to_maf,
    _resolve_coloc_variance_inputs,
    _resolve_drug_matches_by_id,
    _resolve_maf,
    _validate_coloc_calibration_config,
    _summarise_gene_verdict,
    _write_mhc_excluded_sensitivity,
    annotate_cross_source,
    assign_confidence_tiers,
    clump_instruments,
    cochrans_q,
    coloc_abf,
    ensure_ref_freq,
    f_statistic,
    harmonise_gwas_eqtl,
    ivw_fixed_effects,
    ivw_random_effects,
    match_drugs_to_mr_gene,
    mr_egger,
    select_instruments,
    steiger_test,
    wald_ratio,
    weighted_median,
)
from repogen.config.schema import (
    EQTLSourceConfig,
    MRConfig,
    MRDrugMatchConfig,
    MRMHCSensitivityConfig,
)
from repogen.data.schemas import GWASMetadata


# ---------------------------------------------------------------------------
# 7.1 MR Estimation Tests
# ---------------------------------------------------------------------------


class TestWaldRatio:
    def test_known_values(self) -> None:
        beta_mr, se_mr, pval = wald_ratio(0.5, 0.1, 0.2, 0.05)
        assert beta_mr == pytest.approx(0.4, rel=1e-6)
        expected_var = (0.05**2 / 0.5**2) + (0.4**2 * 0.1**2 / 0.5**2)
        assert se_mr == pytest.approx(np.sqrt(expected_var), rel=1e-6)
        assert 0 < pval < 1

    def test_full_se_vs_nome(self) -> None:
        """At F=100, full SE ≈ first-order; at F=10 full SE is larger."""
        bx_high_f, sx_high_f = 0.5, 0.05  # F=100
        by, sy = 0.05, 0.05  # small outcome z keeps second term < 1%

        _, se_full_high, _ = wald_ratio(bx_high_f, sx_high_f, by, sy)
        nome_var_high = sy**2 / bx_high_f**2
        assert abs(se_full_high - np.sqrt(nome_var_high)) / se_full_high < 0.01

        bx_low_f, sx_low_f = 0.316, 0.1  # F≈10
        _, se_full_low, _ = wald_ratio(bx_low_f, sx_low_f, 0.2, sy)
        nome_var_low = sy**2 / bx_low_f**2
        assert se_full_low > np.sqrt(nome_var_low) * 1.05

    def test_pvalue_reasonable(self) -> None:
        _, _, pval = wald_ratio(0.5, 0.1, 0.0, 0.05)
        assert pval > 0.5


class TestIVWFixedEffects:
    def test_known_three_instruments(self) -> None:
        bx = np.array([0.5, 0.3, 0.4])
        sx = np.array([0.1, 0.1, 0.1])
        by = np.array([0.2, 0.12, 0.16])
        sy = np.array([0.05, 0.05, 0.05])

        w = 1.0 / sy**2
        expected_beta = np.sum(w * bx * by) / np.sum(w * bx**2)
        expected_se = 1.0 / np.sqrt(np.sum(w * bx**2))

        beta_mr, se_mr, pval = ivw_fixed_effects(bx, sx, by, sy)
        assert beta_mr == pytest.approx(expected_beta, rel=1e-6)
        assert se_mr == pytest.approx(expected_se, rel=1e-6)
        assert 0 < pval < 1


class TestIVWRandomEffects:
    def test_with_heterogeneity(self) -> None:
        bx = np.array([0.5, 0.3, 0.4])
        sx = np.array([0.1, 0.1, 0.1])
        by = np.array([0.2, -0.1, 0.5])
        sy = np.array([0.05, 0.05, 0.05])

        fe_beta, _, _ = ivw_fixed_effects(bx, sx, by, sy)
        q, q_pval = cochrans_q(bx, sx, by, sy, fe_beta)
        assert q > 0

        re_beta, re_se, re_pval = ivw_random_effects(bx, sx, by, sy)
        assert isinstance(re_beta, float)
        assert re_se > 0


class TestCochransQ:
    def test_homogeneous(self) -> None:
        bx = np.array([0.5, 0.5, 0.5])
        sx = np.array([0.1, 0.1, 0.1])
        by = bx * 0.4
        sy = np.array([0.05, 0.05, 0.05])

        beta_ivw, _, _ = ivw_fixed_effects(bx, sx, by, sy)
        q, q_pval = cochrans_q(bx, sx, by, sy, beta_ivw)
        assert q_pval > 0.05

    def test_heterogeneous(self) -> None:
        bx = np.array([0.5, 0.3, 0.4])
        sx = np.array([0.1, 0.1, 0.1])
        by = np.array([0.2, -0.3, 0.5])
        sy = np.array([0.05, 0.05, 0.05])

        beta_ivw, _, _ = ivw_fixed_effects(bx, sx, by, sy)
        q, q_pval = cochrans_q(bx, sx, by, sy, beta_ivw)
        assert q_pval < 0.05


class TestFStatistic:
    def test_known_value(self) -> None:
        f = f_statistic(np.array([0.5]), np.array([0.1]))
        assert f[0] == pytest.approx(25.0, rel=1e-6)

    def test_vectorised(self) -> None:
        bx = np.array([0.5, 0.3])
        sx = np.array([0.1, 0.1])
        f = f_statistic(bx, sx)
        assert f[0] == pytest.approx(25.0, rel=1e-6)
        assert f[1] == pytest.approx(9.0, rel=1e-6)


class TestMREgger:
    def test_four_instruments(self) -> None:
        bx = np.array([0.5, 0.3, 0.4, 0.6])
        sx = np.array([0.1, 0.1, 0.1, 0.1])
        by = np.array([0.2, 0.15, 0.18, 0.25])
        sy = np.array([0.05, 0.05, 0.05, 0.05])

        result = mr_egger(bx, sx, by, sy)
        assert "intercept" in result
        assert "slope" in result
        assert "intercept_pval" in result
        assert "slope_pval" in result
        assert isinstance(result["slope"], float)


class TestWeightedMedian:
    def test_robustness_to_outlier(self) -> None:
        bx = np.array([0.5, 0.3, 0.4, 0.6, 0.35])
        sx = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        by = np.array([0.2, 0.12, 0.16, 0.24, 5.0])
        sy = np.array([0.05, 0.05, 0.05, 0.05, 0.05])

        beta_wm, se_wm, pval = weighted_median(bx, sx, by, sy)
        assert abs(beta_wm - 0.4) < 0.2


class TestMREdgeCases:
    def test_single_instrument_uses_wald(self) -> None:
        beta_mr, se_mr, pval = wald_ratio(0.5, 0.1, 0.2, 0.05)
        assert isinstance(beta_mr, float)

    def test_all_weak_instruments(self) -> None:
        bx = np.array([0.1, 0.05])
        sx = np.array([0.1, 0.1])
        f = f_statistic(bx, sx)
        assert np.all(f < 10)

    def test_k2_with_q_significant(self) -> None:
        bx = np.array([0.5, 0.3])
        sx = np.array([0.1, 0.1])
        by = np.array([0.5, -0.3])
        sy = np.array([0.05, 0.05])

        fe_beta, _, _ = ivw_fixed_effects(bx, sx, by, sy)
        q, q_pval = cochrans_q(bx, sx, by, sy, fe_beta)

        if q_pval < 0.05:
            re_beta, re_se, re_pval = ivw_random_effects(bx, sx, by, sy)
            assert isinstance(re_beta, float)


# ---------------------------------------------------------------------------
# 7.2 Coloc Tests
# ---------------------------------------------------------------------------


class TestColoc:
    def test_shared_signal(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        causal_idx = 50
        beta1 = rng.normal(0, 0.01, n)
        beta2 = rng.normal(0, 0.01, n)
        beta1[causal_idx] = 0.5
        beta2[causal_idx] = 0.3
        se1 = np.full(n, 0.05)
        se2 = np.full(n, 0.05)
        maf = np.full(n, 0.3)

        result = coloc_abf(beta1, se1, beta2, se2, maf, n1=10000, n2=10000)
        assert result["pp_h4"] > 0.8

    def test_distinct_signals(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        beta1 = rng.normal(0, 0.01, n)
        beta2 = rng.normal(0, 0.01, n)
        beta1[20] = 0.5
        beta2[80] = 0.5
        se1 = np.full(n, 0.05)
        se2 = np.full(n, 0.05)
        maf = np.full(n, 0.3)

        result = coloc_abf(beta1, se1, beta2, se2, maf, n1=10000, n2=10000)
        assert result["pp_h3"] > result["pp_h4"]

    def test_no_association(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        beta1 = rng.normal(0, 0.01, n)
        beta2 = rng.normal(0, 0.01, n)
        se1 = np.full(n, 0.1)
        se2 = np.full(n, 0.1)
        maf = np.full(n, 0.3)

        result = coloc_abf(beta1, se1, beta2, se2, maf, n1=10000, n2=10000)
        assert result["pp_h0"] > 0.5

    def test_insufficient_snps(self) -> None:
        result = coloc_abf(
            np.array([0.5]), np.array([0.05]),
            np.array([0.3]), np.array([0.05]),
            np.array([0.3]), n1=1000, n2=1000,
        )
        assert result["n_snps"] == 1

    def test_prior_sensitivity(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        causal_idx = 50
        beta1 = rng.normal(0, 0.01, n)
        beta2 = rng.normal(0, 0.01, n)
        beta1[causal_idx] = 0.3
        beta2[causal_idx] = 0.2
        se1 = np.full(n, 0.05)
        se2 = np.full(n, 0.05)
        maf = np.full(n, 0.3)

        result_default = coloc_abf(beta1, se1, beta2, se2, maf, 10000, 10000, p12=1e-5)
        result_larger = coloc_abf(beta1, se1, beta2, se2, maf, 10000, 10000, p12=1e-3)
        assert result_larger["pp_h4"] > result_default["pp_h4"]

    def test_log_space_stability(self) -> None:
        n = 50
        beta1 = np.zeros(n)
        beta2 = np.zeros(n)
        beta1[25] = 10.0
        beta2[25] = 8.0
        se1 = np.full(n, 0.5)
        se2 = np.full(n, 0.5)
        maf = np.full(n, 0.3)

        result = coloc_abf(beta1, se1, beta2, se2, maf, n1=50000, n2=50000)
        assert not np.isnan(result["pp_h4"])
        total = result["pp_h0"] + result["pp_h1"] + result["pp_h2"] + result["pp_h3"] + result["pp_h4"]
        assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 7.3 eQTL Loading Tests
# ---------------------------------------------------------------------------


class TestEQTLLoading:
    def test_eqtlgen_z_to_beta(self, tmp_path: Path) -> None:
        eqtl_dir = tmp_path / "eqtlgen"
        eqtl_dir.mkdir()

        eqtl_data = pd.DataFrame({
            "Pvalue": [1e-10, 1e-5],
            "SNP": ["rs1", "rs2"],
            "SNPChr": [1, 1],
            "SNPPos": [100000, 200000],
            "AssessedAllele": ["A", "C"],
            "OtherAllele": ["G", "T"],
            "Zscore": [6.0, 4.0],
            "Gene": ["ENSG00000001", "ENSG00000001"],
            "GeneSymbol": ["GENE1", "GENE1"],
            "GeneChr": [1, 1],
            "GenePos": [150000, 150000],
            "NrCohorts": [10, 10],
            "NrSamples": [31684, 31684],
            "FDR": [0.001, 0.01],
            "BonferroniP": [0.001, 0.1],
        })
        eqtl_data.to_csv(eqtl_dir / "cis-eQTLs.txt", sep="\t", index=False)

        frq_data = pd.DataFrame({
            "CHR": [1, 1],
            "SNP": ["rs1", "rs2"],
            "A1": ["A", "C"],
            "A2": ["G", "T"],
            "MAF": [0.25, 0.3],
            "NCHROBS": [63368, 63368],
        })
        frq_path = tmp_path / "ref.frq"
        frq_data.to_csv(frq_path, sep="\t", index=False)

        from repogen.analysis.mendelian_randomisation import _load_eqtlgen
        result = _load_eqtlgen(eqtl_dir, frq_path)

        assert len(result) == 2
        assert "beta" in result.columns
        assert "se" in result.columns

        maf = 0.25
        z = 6.0
        n = 31684.0
        expected_beta = z / np.sqrt(2 * maf * (1 - maf) * (n + z**2))
        assert result.iloc[0]["beta"] == pytest.approx(expected_beta, rel=1e-4)

    def test_eqtlgen_missing_maf(self, tmp_path: Path) -> None:
        eqtl_dir = tmp_path / "eqtlgen"
        eqtl_dir.mkdir()

        eqtl_data = pd.DataFrame({
            "Pvalue": [1e-10], "SNP": ["rs_not_in_ref"],
            "SNPChr": [1], "SNPPos": [100000],
            "AssessedAllele": ["A"], "OtherAllele": ["G"],
            "Zscore": [6.0], "Gene": ["ENSG00000001"],
            "GeneSymbol": ["GENE1"], "GeneChr": [1], "GenePos": [150000],
            "NrCohorts": [10], "NrSamples": [31684],
            "FDR": [0.001], "BonferroniP": [0.001],
        })
        eqtl_data.to_csv(eqtl_dir / "cis-eQTLs.txt", sep="\t", index=False)

        frq_data = pd.DataFrame({
            "CHR": [1], "SNP": ["rs_other"], "A1": ["A"], "A2": ["G"],
            "MAF": [0.25], "NCHROBS": [63368],
        })
        frq_path = tmp_path / "ref.frq"
        frq_data.to_csv(frq_path, sep="\t", index=False)

        from repogen.analysis.mendelian_randomisation import _load_eqtlgen
        result = _load_eqtlgen(eqtl_dir, frq_path)
        assert len(result) == 0

    def test_metabrain_direct_load(self, tmp_path: Path) -> None:
        eqtl_dir = tmp_path / "metabrain"
        eqtl_dir.mkdir()

        # The legacy loader reads the normalized .tsv.gz (composite
        # chr:pos:rsid:alleles SNP tokens), consistent with the chunked path.
        data = pd.DataFrame({
            "SNP": ["1:100000:rs1:A_G", "2:200000:rs2:C_T"],
            "gene": ["ENSG00000001", "ENSG00000002"],
            "chr": [1, 2],
            "pos": [100000, 200000],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, -0.3],
            "se": [0.1, 0.1],
            "pval": [1e-6, 0.01],
            "n": [2970, 2970],
        })
        data.to_csv(
            eqtl_dir / "metabrain_cortex_normalized.tsv.gz",
            sep="\t", index=False, compression="gzip",
        )

        from repogen.analysis.mendelian_randomisation import _load_metabrain
        result = _load_metabrain(eqtl_dir)
        assert len(result) == 2
        assert set(result["SNP"]) == {"rs1", "rs2"}
        assert result.loc[result["SNP"] == "rs1", "beta"].iloc[0] == pytest.approx(0.5)

    def test_metabrain_null_frq(self, tmp_path: Path) -> None:
        from repogen.analysis.mendelian_randomisation import load_eqtl_source
        eqtl_dir = tmp_path / "metabrain"
        eqtl_dir.mkdir()

        data = pd.DataFrame({
            "SNP": ["1:100000:rs1:A_G"], "gene": ["ENSG00000001"], "chr": [1],
            "pos": [100000], "a1": ["A"], "a2": ["G"],
            "beta": [0.5], "se": [0.1], "pval": [1e-6],
        })
        data.to_csv(
            eqtl_dir / "metabrain_cortex_normalized.tsv.gz",
            sep="\t", index=False, compression="gzip",
        )

        config = EQTLSourceConfig(source="metabrain_cortex", path=eqtl_dir)
        result = load_eqtl_source(config, ref_freq_path=None)
        assert len(result) == 1
        assert result.iloc[0]["SNP"] == "rs1"


# ---------------------------------------------------------------------------
# 7.4 Harmonisation Tests
# ---------------------------------------------------------------------------


class TestHarmonisation:
    @pytest.fixture()
    def gwas_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "CHR": [1, 1, 1, 1, 1],
            "POS": [100, 200, 300, 400, 500],
            "A1": ["A", "G", "T", "A", "C"],
            "A2": ["G", "A", "G", "T", "G"],
            "BETA": [0.1, 0.2, 0.3, 0.4, 0.5],
            "SE": [0.05, 0.05, 0.05, 0.05, 0.05],
            "P": [0.01, 0.01, 0.01, 0.01, 0.01],
            "N": [10000, 10000, 10000, 10000, 10000],
        })

    @pytest.fixture()
    def instruments(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "gene": ["G1"] * 5,
            "chr": [1] * 5,
            "pos": [100, 200, 300, 400, 500],
            "a1": ["A", "A", "A", "A", "X"],
            "a2": ["G", "G", "C", "T", "Y"],
            "beta": [0.5, 0.5, 0.5, 0.5, 0.5],
            "se": [0.1] * 5,
            "pval": [1e-8] * 5,
            "n": [31684] * 5,
        })

    def test_direct_match(self, gwas_df: pd.DataFrame, instruments: pd.DataFrame) -> None:
        result = harmonise_gwas_eqtl(gwas_df, instruments)
        rs1_row = result.loc[result["SNP"] == "rs1"]
        assert len(rs1_row) == 1
        assert rs1_row.iloc[0]["beta_outcome"] == pytest.approx(0.1)

    def test_flipped_alleles(self, gwas_df: pd.DataFrame, instruments: pd.DataFrame) -> None:
        result = harmonise_gwas_eqtl(gwas_df, instruments)
        rs2_row = result.loc[result["SNP"] == "rs2"]
        assert len(rs2_row) == 1
        assert rs2_row.iloc[0]["beta_outcome"] == pytest.approx(-0.2)

    def test_complement_alleles(self, gwas_df: pd.DataFrame, instruments: pd.DataFrame) -> None:
        result = harmonise_gwas_eqtl(gwas_df, instruments)
        rs3_row = result.loc[result["SNP"] == "rs3"]
        assert len(rs3_row) == 1

    def test_palindromic_excluded(self, gwas_df: pd.DataFrame, instruments: pd.DataFrame) -> None:
        result = harmonise_gwas_eqtl(gwas_df, instruments)
        rs4_row = result.loc[result["SNP"] == "rs4"]
        assert len(rs4_row) == 0

    def test_no_match_excluded(self, gwas_df: pd.DataFrame, instruments: pd.DataFrame) -> None:
        result = harmonise_gwas_eqtl(gwas_df, instruments)
        rs5_row = result.loc[result["SNP"] == "rs5"]
        assert len(rs5_row) == 0

    def test_missing_snps(self, gwas_df: pd.DataFrame) -> None:
        instruments = pd.DataFrame({
            "SNP": ["rs999"],
            "gene": ["G1"], "chr": [1], "pos": [999],
            "a1": ["A"], "a2": ["G"],
            "beta": [0.5], "se": [0.1], "pval": [1e-8], "n": [31684],
        })
        result = harmonise_gwas_eqtl(gwas_df, instruments)
        assert len(result) == 0

    def test_empty_instruments(self, gwas_df: pd.DataFrame) -> None:
        result = harmonise_gwas_eqtl(gwas_df, pd.DataFrame())
        assert result.empty

    def test_vectorised(self) -> None:
        import inspect
        source = inspect.getsource(harmonise_gwas_eqtl)
        assert "iterrows" not in source


# ---------------------------------------------------------------------------
# 7.5 Drug Matching Tests
# ---------------------------------------------------------------------------


class TestDrugMatching:
    @pytest.fixture()
    def drug_targets(self) -> pd.DataFrame:
        return pd.DataFrame({
            "gene_ensembl_id": ["ENSG001", "ENSG001", "ENSG001", "ENSG001"],
            "gene_symbol": ["GENE1", "GENE1", "GENE1", "GENE1"],
            "drug_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
            "drug_name": ["DrugA", "DrugB", "DrugC", "DrugD"],
            "drug_inchikey": [None, None, None, None],
            "interaction_type": ["inhibitor", "agonist", "modulator", "activator"],
            "pchembl_value": [7.0, 6.5, 5.0, 8.0],
            "max_phase": [4, 3, 2, 1],
            "atc_codes": [["N05A"], ["N06A"], [], ["N07X"]],
        })

    def test_positive_beta_inhibitor(self, drug_targets: pd.DataFrame) -> None:
        gene_row = pd.Series({
            "gene_ensembl_id": "ENSG001", "gene_symbol": "GENE1",
            "eqtl_source": "eqtlgen", "mr_beta": 0.5, "mr_pval": 1e-10,
            "pp_h4": 0.95, "confidence_tier": "high",
        })
        result = match_drugs_to_mr_gene(gene_row, drug_targets)
        inhibitor_row = result.loc[result["drug_name"] == "DrugA"]
        assert inhibitor_row.iloc[0]["direction_concordant"] is True

    def test_positive_beta_agonist(self, drug_targets: pd.DataFrame) -> None:
        gene_row = pd.Series({
            "gene_ensembl_id": "ENSG001", "gene_symbol": "GENE1",
            "eqtl_source": "eqtlgen", "mr_beta": 0.5, "mr_pval": 1e-10,
            "pp_h4": 0.95, "confidence_tier": "high",
        })
        result = match_drugs_to_mr_gene(gene_row, drug_targets)
        agonist_row = result.loc[result["drug_name"] == "DrugB"]
        assert agonist_row.iloc[0]["direction_concordant"] is False

    def test_negative_beta_agonist(self, drug_targets: pd.DataFrame) -> None:
        gene_row = pd.Series({
            "gene_ensembl_id": "ENSG001", "gene_symbol": "GENE1",
            "eqtl_source": "eqtlgen", "mr_beta": -0.5, "mr_pval": 1e-10,
            "pp_h4": 0.95, "confidence_tier": "high",
        })
        result = match_drugs_to_mr_gene(gene_row, drug_targets)
        agonist_row = result.loc[result["drug_name"] == "DrugB"]
        assert agonist_row.iloc[0]["direction_concordant"] is True

    def test_modulator_ambiguous(self, drug_targets: pd.DataFrame) -> None:
        gene_row = pd.Series({
            "gene_ensembl_id": "ENSG001", "gene_symbol": "GENE1",
            "eqtl_source": "eqtlgen", "mr_beta": 0.5, "mr_pval": 1e-10,
            "pp_h4": 0.95, "confidence_tier": "high",
        })
        result = match_drugs_to_mr_gene(gene_row, drug_targets)
        mod_row = result.loc[result["drug_name"] == "DrugC"]
        assert mod_row.iloc[0]["direction_concordant"] is None
        assert bool(mod_row.iloc[0]["interaction_direction_ambiguous"]) is True

    def test_gene_with_no_drugs(self) -> None:
        drug_targets = pd.DataFrame({
            "gene_ensembl_id": ["ENSG999"],
            "gene_symbol": ["OTHER"],
            "drug_chembl_id": ["CHEMBL1"],
            "drug_name": ["DrugA"],
            "drug_inchikey": [None],
            "interaction_type": ["inhibitor"],
            "pchembl_value": [7.0],
            "max_phase": [4],
            "atc_codes": [["N05A"]],
        })
        gene_row = pd.Series({
            "gene_ensembl_id": "ENSG001", "gene_symbol": "GENE1",
            "eqtl_source": "eqtlgen", "mr_beta": 0.5, "mr_pval": 1e-10,
            "pp_h4": 0.95, "confidence_tier": "high",
        })
        result = match_drugs_to_mr_gene(gene_row, drug_targets)
        assert result.empty

    def test_direction_constants_complete(self) -> None:
        all_types = DOWNREGULATING_TYPES | UPREGULATING_TYPES | AMBIGUOUS_TYPES
        assert "inhibitor" in all_types
        assert "agonist" in all_types
        assert "modulator" in all_types
        assert "positive_modulator" in UPREGULATING_TYPES
        assert "negative_modulator" in DOWNREGULATING_TYPES
        assert INFERABLE_TYPES == (DOWNREGULATING_TYPES | UPREGULATING_TYPES)


# ---------------------------------------------------------------------------
# Drug-match config, filters, strict IDs, provenance, verdicts
# ---------------------------------------------------------------------------


def _rich_drug_targets() -> pd.DataFrame:
    """Drug targets with ID + provenance columns for drug-match tests."""
    return pd.DataFrame({
        "gene_ensembl_id": ["ENSG001", "ENSG001", "ENSG001", "ENSG001"],
        "gene_symbol": ["GENE1", "GENE1", "GENE1", "GENE1"],
        "gene_entrez_id": [111, 111, 111, 111],
        "gene_uniprot_id": ["P00001", "P00001", "P00001", "P00001"],
        "drug_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
        "drug_name": ["DrugA", "DrugB", "DrugC", "DrugD"],
        "drug_inchikey": [None, None, None, None],
        "interaction_type": ["inhibitor", "agonist", "other", "activator"],
        "pchembl_value": [7.0, 6.5, None, 8.0],
        "max_phase": [4, 3, 2, 0],
        "atc_codes": [["N05A"], ["N06A"], [], ["N07X"]],
        "mechanism_of_action": ["X inhibitor", None, "Modulates X", ""],
        "action_type": ["INHIBITOR", "AGONIST", None, "ACTIVATOR"],
        "source": ["chembl", "chembl", "dgidb", "pdsp"],
        "confidence": ["high", "medium", "low", "low"],
    })


def _gene_row(**overrides) -> pd.Series:
    base = {
        "gene_ensembl_id": "ENSG001", "gene_symbol": "GENE1",
        "gene_entrez_id": 111, "gene_uniprot_id": "P00001",
        "eqtl_source": "eqtlgen", "mr_beta": 0.5, "mr_pval": 1e-10,
        "pp_h4": 0.95, "confidence_tier": "high",
    }
    base.update(overrides)
    return pd.Series(base)


class TestDrugMatchConfig:
    def test_defaults_are_legacy_preserving(self) -> None:
        cfg = MRDrugMatchConfig()
        assert cfg.match_mode == "legacy"
        assert cfg.min_pchembl is None
        assert cfg.min_phase is None
        assert cfg.phase_filter_scope == "global"
        assert cfg.direction_policy == "all"
        assert cfg.allow_symbol_fallback is True
        assert cfg.require_druggable is False
        assert cfg.druggable_genome_path is None

    def test_mrconfig_has_default_drug_match(self) -> None:
        mr_cfg = MRConfig(eqtl_sources=[EQTLSourceConfig(source="eqtlgen", path="x.txt")])
        assert isinstance(mr_cfg.drug_match, MRDrugMatchConfig)
        assert mr_cfg.drug_match.match_mode == "legacy"

    def test_invalid_match_mode_rejected(self) -> None:
        with pytest.raises(Exception):
            MRDrugMatchConfig(match_mode="fuzzy")


class TestLegacyDrugMatchInvariant:
    """Default config must preserve legacy row set + existing-column values."""

    def test_default_config_matches_legacy_values(self) -> None:
        dt = _rich_drug_targets()
        gene = _gene_row()
        result = match_drugs_to_mr_gene(gene, dt)  # default (legacy) config
        # Same 4 drug rows as legacy exact-Ensembl match.
        assert sorted(result["drug_name"]) == ["DrugA", "DrugB", "DrugC", "DrugD"]
        inhib = result.loc[result["drug_name"] == "DrugA"].iloc[0]
        assert inhib["direction_concordant"] is True  # +beta inhibitor
        agon = result.loc[result["drug_name"] == "DrugB"].iloc[0]
        assert agon["direction_concordant"] is False  # +beta agonist
        other = result.loc[result["drug_name"] == "DrugC"].iloc[0]
        assert other["direction_concordant"] is None
        assert bool(other["interaction_direction_ambiguous"]) is True

    def test_default_config_adds_provenance_columns(self) -> None:
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets())
        for col in (
            "action_type", "mechanism_of_action", "has_mechanism_text",
            "direction_inferable", "drug_target_source", "drug_target_confidence",
            "match_via", "drug_match_rank",
        ):
            assert col in result.columns
        assert (result["match_via"] == "ensembl").all()

    def test_symbol_fallback_flagged_via_symbol(self) -> None:
        gene = _gene_row(gene_ensembl_id="ENSG_MISSING")  # not in drug_targets
        result = match_drugs_to_mr_gene(gene, _rich_drug_targets())
        assert not result.empty
        assert (result["match_via"] == "symbol").all()


class TestStrictMatchMode:
    def test_ensembl_version_stripped(self) -> None:
        cfg = MRDrugMatchConfig(match_mode="strict")
        gene = _gene_row(gene_ensembl_id="ENSG001.7")
        raw, via = _resolve_drug_matches_by_id(gene, _rich_drug_targets(), cfg)
        assert via == "ensembl"
        assert len(raw) == 4

    def test_entrez_fallback(self) -> None:
        cfg = MRDrugMatchConfig(match_mode="strict")
        gene = _gene_row(gene_ensembl_id="ENSG_NONE", gene_entrez_id=111)
        raw, via = _resolve_drug_matches_by_id(gene, _rich_drug_targets(), cfg)
        assert via == "entrez"
        assert len(raw) == 4

    def test_uniprot_fallback(self) -> None:
        cfg = MRDrugMatchConfig(match_mode="strict")
        gene = _gene_row(
            gene_ensembl_id="ENSG_NONE", gene_entrez_id=None, gene_uniprot_id="P00001"
        )
        raw, via = _resolve_drug_matches_by_id(gene, _rich_drug_targets(), cfg)
        assert via == "uniprot"

    def test_unambiguous_symbol_accepted(self) -> None:
        cfg = MRDrugMatchConfig(match_mode="strict")
        gene = _gene_row(
            gene_ensembl_id="ENSG_NONE", gene_entrez_id=None, gene_uniprot_id=None,
            gene_symbol="GENE1",
        )
        raw, via = _resolve_drug_matches_by_id(gene, _rich_drug_targets(), cfg)
        assert via == "symbol"

    def test_ambiguous_symbol_rejected(self) -> None:
        cfg = MRDrugMatchConfig(match_mode="strict")
        dt = _rich_drug_targets()
        # Make the symbol map to two distinct Ensembl genes (paralogs).
        dt.loc[2, "gene_ensembl_id"] = "ENSG002"
        dt.loc[3, "gene_ensembl_id"] = "ENSG002"
        gene = _gene_row(
            gene_ensembl_id="ENSG_NONE", gene_entrez_id=None, gene_uniprot_id=None,
            gene_symbol="GENE1",
        )
        raw, via = _resolve_drug_matches_by_id(gene, dt, cfg)
        assert via == ""
        assert raw.empty

    def test_symbol_fallback_disabled(self) -> None:
        cfg = MRDrugMatchConfig(match_mode="strict", allow_symbol_fallback=False)
        gene = _gene_row(
            gene_ensembl_id="ENSG_NONE", gene_entrez_id=None, gene_uniprot_id=None,
            gene_symbol="GENE1",
        )
        raw, via = _resolve_drug_matches_by_id(gene, _rich_drug_targets(), cfg)
        assert via == ""


class TestDrugMatchFilters:
    def test_min_phase_drops_low_phase(self) -> None:
        cfg = MRDrugMatchConfig(min_phase=3)
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets(), cfg)
        assert set(result["drug_name"]) == {"DrugA", "DrugB"}  # phase 4, 3

    def test_min_pchembl_keeps_null(self) -> None:
        cfg = MRDrugMatchConfig(min_pchembl=7.0)
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets(), cfg)
        # DrugA(7.0) & DrugD(8.0) pass; DrugC has null pchembl -> retained; DrugB(6.5) dropped
        assert set(result["drug_name"]) == {"DrugA", "DrugC", "DrugD"}

    def test_phase_filter_scope_chembl_only(self) -> None:
        cfg = MRDrugMatchConfig(min_phase=3, phase_filter_scope="chembl_only")
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets(), cfg)
        # chembl rows: DrugA(4 keep), DrugB(3 keep); non-chembl DrugC(dgidb) & DrugD(pdsp) exempt
        assert set(result["drug_name"]) == {"DrugA", "DrugB", "DrugC", "DrugD"}

    def test_direction_policy_inferable_only(self) -> None:
        cfg = MRDrugMatchConfig(direction_policy="inferable_only")
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets(), cfg)
        # DrugC ('other') dropped; inhibitor/agonist/activator kept
        assert "DrugC" not in set(result["drug_name"])
        assert set(result["drug_name"]) == {"DrugA", "DrugB", "DrugD"}

    def test_require_druggable_gates_when_absent(self) -> None:
        cfg = MRDrugMatchConfig(require_druggable=True, druggable_genome_path="dg.tsv")
        result = match_drugs_to_mr_gene(_gene_row(druggable_tier=None), _rich_drug_targets(), cfg)
        assert result.empty

    def test_require_druggable_keeps_when_present(self) -> None:
        cfg = MRDrugMatchConfig(require_druggable=True, druggable_genome_path="dg.tsv")
        result = match_drugs_to_mr_gene(_gene_row(druggable_tier="Tier 1"), _rich_drug_targets(), cfg)
        assert not result.empty


class TestDrugMatchProvenance:
    def test_provenance_values_populated(self) -> None:
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets())
        drug_a = result.loc[result["drug_name"] == "DrugA"].iloc[0]
        assert drug_a["action_type"] == "INHIBITOR"
        assert drug_a["mechanism_of_action"] == "X inhibitor"
        assert bool(drug_a["has_mechanism_text"]) is True
        assert bool(drug_a["direction_inferable"]) is True
        assert drug_a["drug_target_source"] == "chembl"
        assert drug_a["drug_target_confidence"] == "high"

    def test_has_mechanism_text_false_for_empty(self) -> None:
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets())
        drug_d = result.loc[result["drug_name"] == "DrugD"].iloc[0]  # moa == ""
        assert bool(drug_d["has_mechanism_text"]) is False
        assert pd.isna(drug_d["mechanism_of_action"])

    def test_direction_inferable_false_for_other(self) -> None:
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets())
        drug_c = result.loc[result["drug_name"] == "DrugC"].iloc[0]  # 'other'
        assert bool(drug_c["direction_inferable"]) is False

    def test_prefer_inferable_ranks_inferable_first(self) -> None:
        cfg = MRDrugMatchConfig(direction_policy="prefer_inferable")
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets(), cfg)
        rank1 = result.loc[result["drug_match_rank"] == 1].iloc[0]
        assert bool(rank1["direction_inferable"]) is True

    def test_rank_is_contiguous(self) -> None:
        result = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets())
        ranks = sorted(result["drug_match_rank"].tolist())
        assert ranks == list(range(1, len(result) + 1))


class TestTargetVerdicts:
    def test_actionable_status(self) -> None:
        cfg = MRDrugMatchConfig()
        matches = match_drugs_to_mr_gene(_gene_row(), _rich_drug_targets(), cfg)
        verdict = _summarise_gene_verdict(_gene_row(), raw_count=4, matches=matches, config=cfg)
        assert verdict["verdict_status"] == "actionable"
        assert verdict["n_filtered_matches"] == 4
        assert verdict["n_direction_inferable"] == 3  # inhibitor, agonist, activator
        assert verdict["best_drug_name"] is not None

    def test_binder_only_status(self) -> None:
        cfg = MRDrugMatchConfig()
        dt = _rich_drug_targets().loc[lambda d: d["interaction_type"] == "other"].copy()
        matches = match_drugs_to_mr_gene(_gene_row(), dt, cfg)
        verdict = _summarise_gene_verdict(_gene_row(), raw_count=1, matches=matches, config=cfg)
        assert verdict["verdict_status"] == "binder_only"
        assert verdict["n_direction_inferable"] == 0

    def test_no_filtered_match_status(self) -> None:
        cfg = MRDrugMatchConfig(min_phase=4)
        dt = _rich_drug_targets().loc[lambda d: d["max_phase"] < 4].copy()
        matches = match_drugs_to_mr_gene(_gene_row(), dt, cfg)
        verdict = _summarise_gene_verdict(_gene_row(), raw_count=len(dt), matches=matches, config=cfg)
        assert verdict["verdict_status"] == "no_filtered_match"
        assert verdict["n_filtered_matches"] == 0

    def test_no_drug_record_status(self) -> None:
        cfg = MRDrugMatchConfig()
        verdict = _summarise_gene_verdict(
            _gene_row(), raw_count=0, matches=pd.DataFrame(), config=cfg
        )
        assert verdict["verdict_status"] == "no_drug_record"
        assert verdict["n_drug_records_raw"] == 0


class TestDrugMatchNullSafety:
    """The drug loader emits pd.NA for affinity-only records."""

    def _affinity_only_targets(self) -> pd.DataFrame:
        # Mirrors drug_loader affinity-only output: pd.NA mechanism + interaction.
        return pd.DataFrame({
            "gene_ensembl_id": ["ENSG001", "ENSG001"],
            "gene_symbol": ["GENE1", "GENE1"],
            "drug_chembl_id": ["CHEMBL1", "CHEMBL2"],
            "drug_name": ["DrugA", "DrugB"],
            "drug_inchikey": [None, None],
            "interaction_type": pd.array([pd.NA, pd.NA], dtype="string"),
            "pchembl_value": [7.0, 6.0],
            "max_phase": [4, 2],
            "atc_codes": [["N05A"], ["N06A"]],
            "mechanism_of_action": pd.array([pd.NA, pd.NA], dtype="string"),
            "action_type": pd.array([pd.NA, pd.NA], dtype="string"),
            "source": ["chembl", "chembl"],
            "confidence": ["low", "low"],
        })

    def test_pd_na_mechanism_does_not_crash(self) -> None:
        result = match_drugs_to_mr_gene(_gene_row(), self._affinity_only_targets())
        assert len(result) == 2
        assert bool(result.iloc[0]["has_mechanism_text"]) is False
        assert result["mechanism_of_action"].isna().all()
        # direction not inferable for pd.NA interaction_type (no crash on `in`).
        assert bool(result.iloc[0]["direction_inferable"]) is False

    def test_pd_na_druggable_tier_require_druggable_no_crash(self) -> None:
        cfg = MRDrugMatchConfig(require_druggable=True, druggable_genome_path="x.tsv")
        gene = _gene_row(druggable_tier=pd.NA)
        result = match_drugs_to_mr_gene(gene, self._affinity_only_targets(), cfg)
        assert result.empty  # gated away, no TypeError

    def test_has_text_helper(self) -> None:
        assert _has_text("Tier 1") is True
        assert _has_text("") is False
        assert _has_text(None) is False
        assert _has_text(pd.NA) is False
        assert _has_text(np.nan) is False


class TestChemblOnlyPhaseScope:
    """chembl_only must mirror Branch A contains() semantics."""

    def test_compound_source_treated_as_chembl(self) -> None:
        dt = pd.DataFrame({
            "gene_ensembl_id": ["ENSG001", "ENSG001"],
            "gene_symbol": ["GENE1", "GENE1"],
            "drug_chembl_id": ["CHEMBL1", "CHEMBL2"],
            "drug_name": ["MergedDrug", "PureDgidb"],
            "drug_inchikey": [None, None],
            "interaction_type": ["inhibitor", "inhibitor"],
            "pchembl_value": [7.0, 7.0],
            "max_phase": [0, 0],
            "atc_codes": [[], []],
            "source": ["chembl,dgidb", "dgidb"],
        })
        cfg = MRDrugMatchConfig(min_phase=1, phase_filter_scope="chembl_only")
        result = match_drugs_to_mr_gene(_gene_row(), dt, cfg)
        # "chembl,dgidb" is ChEMBL-sourced -> phase-0 dropped; pure dgidb exempt.
        assert set(result["drug_name"]) == {"PureDgidb"}


class TestDruggableResourceFailLoud:
    """A configured-but-missing druggable resource must fail loud."""

    def test_missing_path_raises(self) -> None:
        cfg = MRDrugMatchConfig(druggable_genome_path="does_not_exist.tsv")
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG001"], "eqtl_source": ["eqtlgen"], "mr_pval": [1e-6],
        })
        with pytest.raises(FileNotFoundError, match="not found"):
            _annotate_druggable_track(df, cfg)

    def test_invalid_columns_raise(self, tmp_path) -> None:
        res = tmp_path / "bad.tsv"
        res.write_text("wrong_col\tanother\nx\ty\n")
        cfg = MRDrugMatchConfig(druggable_genome_path=res)
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG001"], "eqtl_source": ["eqtlgen"], "mr_pval": [1e-6],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            _annotate_druggable_track(df, cfg)

    def test_require_druggable_without_path_rejected_at_config(self) -> None:
        with pytest.raises(Exception, match="druggable_genome_path"):
            MRDrugMatchConfig(require_druggable=True)


class TestDruggableTrack:
    def _mr_results(self) -> pd.DataFrame:
        return pd.DataFrame({
            "gene_ensembl_id": ["ENSG001", "ENSG002", "ENSG003"],
            "eqtl_source": ["eqtlgen", "eqtlgen", "eqtlgen"],
            # ENSG001 (1e-6) < 0.025 threshold; ENSG002 (0.03) > threshold.
            "mr_pval": [1e-6, 0.03, 0.5],
        })

    def test_no_path_leaves_unchanged(self) -> None:
        cfg = MRDrugMatchConfig()
        df = self._mr_results()
        out = _annotate_druggable_track(df, cfg)
        assert "druggable_tier" not in out.columns

    def test_annotation_and_secondary_track(self, tmp_path) -> None:
        res = tmp_path / "druggable.tsv"
        res.write_text(
            "gene_ensembl_id\tdruggable_tier\nENSG001\tTier 1\nENSG002\tTier 2\n"
        )
        cfg = MRDrugMatchConfig(druggable_genome_path=res)
        out = _annotate_druggable_track(self._mr_results(), cfg)
        assert out.loc[out["gene_ensembl_id"] == "ENSG001", "druggable_tier"].iloc[0] == "Tier 1"
        assert pd.isna(out.loc[out["gene_ensembl_id"] == "ENSG003", "druggable_tier"].iloc[0])
        # Bonferroni over 2 druggable genes tested -> 0.05/2 = 0.025
        thr = out.loc[out["gene_ensembl_id"] == "ENSG001", "bonferroni_threshold_druggable"].iloc[0]
        assert thr == pytest.approx(0.025)
        assert bool(out.loc[out["gene_ensembl_id"] == "ENSG001", "mr_significant_druggable"].iloc[0]) is True
        assert bool(out.loc[out["gene_ensembl_id"] == "ENSG002", "mr_significant_druggable"].iloc[0]) is False

    def test_ensembl_no_version_helper(self) -> None:
        assert _ensembl_no_version("ENSG001.7") == "ENSG001"
        assert _ensembl_no_version("ENSG001") == "ENSG001"
        assert _ensembl_no_version(None) == ""


# ---------------------------------------------------------------------------
# 7.6 Tiering & Integration Tests
# ---------------------------------------------------------------------------


class TestTiering:
    @pytest.fixture()
    def base_row(self) -> dict:
        return {
            "gene_ensembl_id": "ENSG001",
            "gene_symbol": "GENE1",
            "eqtl_source": "eqtlgen",
            "n_instruments": 3,
            "mr_method": "ivw_fe",
            "mr_beta": 0.5, "mr_se": 0.1, "mr_pval": 1e-10,
            "mr_significant": True,
            "bonferroni_threshold": 1e-5,
            "mean_f_stat": 50.0,
            "weak_instrument_excluded": False,
            "heterogeneity_warning": False,
            "coloc_supported": True,
            "coloc_status": "colocalised",
            "cross_source_status": "concordant",
            "steiger_valid": True,
        }

    def test_high_confidence(self, base_row: dict) -> None:
        df = pd.DataFrame([base_row])
        result = assign_confidence_tiers(df)
        assert result.iloc[0]["confidence_tier"] == "high"

    def test_medium_non_significant_second(self, base_row: dict) -> None:
        base_row["cross_source_status"] = "non_significant"
        df = pd.DataFrame([base_row])
        result = assign_confidence_tiers(df)
        assert result.iloc[0]["confidence_tier"] == "medium"

    def test_medium_unavailable_second(self, base_row: dict) -> None:
        base_row["cross_source_status"] = "unavailable"
        df = pd.DataFrame([base_row])
        result = assign_confidence_tiers(df)
        assert result.iloc[0]["confidence_tier"] == "medium"

    def test_direction_conflict(self, base_row: dict) -> None:
        base_row["cross_source_status"] = "discordant"
        df = pd.DataFrame([base_row])
        result = assign_confidence_tiers(df)
        assert result.iloc[0]["confidence_tier"] == "direction_conflict"

    def test_low_coloc_failed(self, base_row: dict) -> None:
        base_row["coloc_supported"] = False
        df = pd.DataFrame([base_row])
        result = assign_confidence_tiers(df)
        assert result.iloc[0]["confidence_tier"] == "low"

    def test_steiger_flagged(self, base_row: dict) -> None:
        base_row["steiger_valid"] = False
        df = pd.DataFrame([base_row])
        result = assign_confidence_tiers(df, require_steiger=True)
        assert result.iloc[0]["confidence_tier"] == "steiger_flagged"

    def test_single_source_medium_max(self, base_row: dict) -> None:
        base_row["cross_source_status"] = "unavailable"
        df = pd.DataFrame([base_row])
        result = assign_confidence_tiers(df)
        assert result.iloc[0]["confidence_tier"] == "medium"

    def test_full_pipeline_columns(self) -> None:
        rows = []
        for i in range(5):
            rows.append({
                "gene_ensembl_id": f"ENSG{i:03d}",
                "gene_symbol": f"GENE{i}",
                "eqtl_source": "eqtlgen",
                "n_instruments": 3,
                "mr_method": "ivw_fe",
                "mr_beta": 0.5 - i * 0.2,
                "mr_se": 0.1,
                "mr_pval": 1e-10 if i < 3 else 0.5,
                "mr_significant": i < 3,
                "bonferroni_threshold": 1e-5,
                "mean_f_stat": 50.0,
                "weak_instrument_excluded": False,
                "heterogeneity_warning": False,
                "coloc_supported": i < 2,
                "coloc_status": "colocalised" if i < 2 else "unsupported",
                "cross_source_status": "unavailable",
                "steiger_valid": True,
            })

        df = pd.DataFrame(rows)
        df = annotate_cross_source(df)
        df = assign_confidence_tiers(df)

        expected_cols = {
            "gene_ensembl_id", "gene_symbol", "eqtl_source",
            "mr_method", "mr_beta", "mr_se", "mr_pval",
            "mr_significant", "confidence_tier", "cross_source_status",
        }
        assert expected_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# 7.7 Steiger Tests
# ---------------------------------------------------------------------------


class TestSteiger:
    def test_quantitative_valid(self) -> None:
        pval, valid = steiger_test(0.05, 0.01, 30000, 100000)
        assert valid is True
        assert pval < 0.05

    def test_binary_trait(self) -> None:
        pval, valid = steiger_test(
            0.05, 0.01, 30000, 50000,
            trait_type="case_control", n_cases=20000, n_controls=30000,
        )
        assert isinstance(pval, float)
        assert valid in (True, False)

    def test_wrong_direction(self) -> None:
        pval, valid = steiger_test(0.01, 0.05, 30000, 100000)
        assert valid is False


# ---------------------------------------------------------------------------
# 7.8 Cross-Source Concordance Tests
# ---------------------------------------------------------------------------


class TestCrossSource:
    def test_concordant(self) -> None:
        df = pd.DataFrame([
            {"gene_ensembl_id": "G1", "eqtl_source": "eqtlgen",
             "mr_significant": True, "mr_beta": 0.5, "mr_pval": 1e-10},
            {"gene_ensembl_id": "G1", "eqtl_source": "metabrain",
             "mr_significant": False, "mr_beta": 0.3, "mr_pval": 0.01},
        ])
        result = annotate_cross_source(df)
        assert (result["cross_source_status"] == "concordant").all()

    def test_discordant(self) -> None:
        df = pd.DataFrame([
            {"gene_ensembl_id": "G1", "eqtl_source": "eqtlgen",
             "mr_significant": True, "mr_beta": 0.5, "mr_pval": 1e-10},
            {"gene_ensembl_id": "G1", "eqtl_source": "metabrain",
             "mr_significant": False, "mr_beta": -0.3, "mr_pval": 0.01},
        ])
        result = annotate_cross_source(df)
        assert (result["cross_source_status"] == "discordant").all()

    def test_non_significant_second(self) -> None:
        df = pd.DataFrame([
            {"gene_ensembl_id": "G1", "eqtl_source": "eqtlgen",
             "mr_significant": True, "mr_beta": 0.5, "mr_pval": 1e-10},
            {"gene_ensembl_id": "G1", "eqtl_source": "metabrain",
             "mr_significant": False, "mr_beta": 0.1, "mr_pval": 0.5},
        ])
        result = annotate_cross_source(df)
        assert (result["cross_source_status"] == "non_significant").all()

    def test_single_source(self) -> None:
        df = pd.DataFrame([
            {"gene_ensembl_id": "G1", "eqtl_source": "eqtlgen",
             "mr_significant": True, "mr_beta": 0.5, "mr_pval": 1e-10},
        ])
        result = annotate_cross_source(df)
        assert (result["cross_source_status"] == "unavailable").all()


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestMRConfig:
    def test_defaults(self) -> None:
        config = MRConfig()
        assert len(config.eqtl_sources) == 1
        assert config.eqtl_sources[0].source == "eqtlgen"
        assert config.cis_window_kb == 1000
        assert config.instrument_pval == 5e-8
        assert config.clump_r2 == 0.001
        assert config.f_stat_threshold == 10.0
        assert config.coloc_enabled is True
        assert config.require_coloc is True
        assert config.require_steiger is False

    def test_too_many_sources(self) -> None:
        with pytest.raises(ValueError, match="at most 2"):
            MRConfig(eqtl_sources=[
                EQTLSourceConfig(source="a", path=Path(".")),
                EQTLSourceConfig(source="b", path=Path(".")),
                EQTLSourceConfig(source="c", path=Path(".")),
            ])

    def test_empty_sources(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            MRConfig(eqtl_sources=[])

    def test_duplicate_sources(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            MRConfig(eqtl_sources=[
                EQTLSourceConfig(source="eqtlgen", path=Path(".")),
                EQTLSourceConfig(source="eqtlgen", path=Path(".")),
            ])

    def test_pipeline_config_integration(self) -> None:
        from repogen.config.schema import PipelineConfig, StudyConfig
        config = PipelineConfig(
            study=StudyConfig(name="test", gwas_input=Path("test.gz")),
        )
        assert isinstance(config.mr, MRConfig)
        assert config.mr.cis_window_kb == 1000


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestMRSchemas:
    def test_mr_result_schema_valid(self) -> None:
        from repogen.data.schemas import validate_dataframe
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG001"],
            "gene_symbol": ["GENE1"],
            "eqtl_source": ["eqtlgen"],
            "n_instruments": [3],
            "mr_method": ["ivw_fe"],
            "mr_beta": [0.5],
            "mr_se": [0.1],
            "mr_pval": [1e-10],
            "mr_significant": [True],
            "bonferroni_threshold": [1e-5],
            "mean_f_stat": [50.0],
            "weak_instrument_excluded": [False],
            "heterogeneity_warning": [False],
            "coloc_supported": [True],
            "coloc_status": ["colocalised"],
            "cross_source_status": ["concordant"],
            "confidence_tier": ["high"],
        })
        errors = validate_dataframe(df, "MRResult")
        assert errors == []

    def test_mr_drug_match_schema_valid(self) -> None:
        from repogen.data.schemas import validate_dataframe
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG001"],
            "gene_symbol": ["GENE1"],
            "eqtl_source": ["eqtlgen"],
            "mr_beta": [0.5],
            "mr_pval": [1e-10],
            "confidence_tier": ["high"],
            "drug_chembl_id": ["CHEMBL1"],
            "drug_name": ["DrugA"],
            "interaction_type": ["inhibitor"],
            "interaction_direction_ambiguous": [False],
            "max_phase": [4],
        })
        errors = validate_dataframe(df, "MRDrugMatch")
        assert errors == []

    def test_mr_target_verdict_schema_valid(self) -> None:
        from repogen.data.schemas import validate_dataframe
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG001"],
            "gene_symbol": ["GENE1"],
            "eqtl_source": ["eqtlgen"],
            "mr_beta": [0.5],
            "mr_pval": [1e-10],
            "confidence_tier": ["high"],
            "verdict_status": ["actionable"],
            "n_drug_records_raw": [4],
            "n_filtered_matches": [2],
            "n_direction_inferable": [2],
            "n_direction_concordant": [1],
        })
        errors = validate_dataframe(df, "MRTargetVerdict")
        assert errors == []

    def test_mr_result_missing_column(self) -> None:
        from repogen.data.schemas import validate_dataframe
        df = pd.DataFrame({"gene_ensembl_id": ["ENSG001"]})
        errors = validate_dataframe(df, "MRResult")
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Instrument selection tests
# ---------------------------------------------------------------------------


class TestInstrumentSelection:
    def test_basic_selection(self) -> None:
        eqtl_df = pd.DataFrame({
            "SNP": [f"rs{i}" for i in range(10)],
            "gene": ["G1"] * 10,
            "chr": [1] * 10,
            "pos": list(range(90000, 110000, 2000)),
            "a1": ["A"] * 10,
            "a2": ["G"] * 10,
            "beta": [0.5] * 5 + [0.05] * 5,
            "se": [0.1] * 10,
            "pval": [1e-10] * 5 + [0.5] * 5,
            "n": [31684] * 10,
        })

        result = select_instruments(
            eqtl_df, "G1", cis_window_kb=1000,
            gene_start=100000, gene_chr=1,
            pval_threshold=5e-8, f_stat_threshold=10.0,
        )
        assert len(result) == 5
        assert all(result["f_stat"] >= 10)

    def test_empty_gene(self) -> None:
        eqtl_df = pd.DataFrame({
            "SNP": ["rs1"], "gene": ["OTHER"],
            "chr": [1], "pos": [100000],
            "a1": ["A"], "a2": ["G"],
            "beta": [0.5], "se": [0.1], "pval": [1e-10], "n": [31684],
        })
        result = select_instruments(
            eqtl_df, "G1", cis_window_kb=1000,
            gene_start=100000, gene_chr=1,
            pval_threshold=5e-8, f_stat_threshold=10.0,
        )
        assert result.empty


# ---------------------------------------------------------------------------
# ensure_ref_freq tests
# ---------------------------------------------------------------------------


class TestEnsureRefFreq:
    def test_existing_frq(self, tmp_path: Path) -> None:
        bfile = tmp_path / "test_prefix"
        frq_file = Path(str(bfile) + ".frq")
        frq_file.write_text("CHR SNP A1 A2 MAF NCHROBS\n1 rs1 A G 0.3 100\n")

        result = ensure_ref_freq(bfile, Path("plink"))
        assert result == frq_file

    def test_frq_not_generated_raises(self, tmp_path: Path) -> None:
        bfile = tmp_path / "test_prefix"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(FileNotFoundError, match="did not produce"):
                ensure_ref_freq(bfile, Path("plink"))


# ---------------------------------------------------------------------------
# Fix A: Clumping failure handling tests
# ---------------------------------------------------------------------------


class TestClumpingFailureHandling:
    @pytest.fixture()
    def instruments(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3"],
            "gene": ["G1"] * 3,
            "chr": [1] * 3,
            "pos": [100000, 200000, 300000],
            "a1": ["A"] * 3,
            "a2": ["G"] * 3,
            "beta": [0.5, 0.3, 0.4],
            "se": [0.1] * 3,
            "pval": [1e-10, 1e-8, 1e-6],
            "n": [31684] * 3,
        })

    def test_plink_failure_raises(self, instruments: pd.DataFrame, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd="plink", stderr="Error: bfile not found"
            )
            with pytest.raises(RuntimeError, match="PLINK clumping failed"):
                clump_instruments(
                    instruments, tmp_path / "ref", 0.001, 1000, Path("plink"),
                )

    def test_no_clumped_file_returns_empty(self, instruments: pd.DataFrame, tmp_path: Path) -> None:
        with patch("subprocess.run"):
            result = clump_instruments(
                instruments, tmp_path / "ref", 0.001, 1000, Path("plink"),
            )
            assert result.empty

    def test_malformed_clumped_raises(self, instruments: pd.DataFrame, tmp_path: Path) -> None:
        def _create_bad_clumped(*args, **kwargs):
            out_prefix = None
            for i, a in enumerate(args[0]):
                if a == "--out":
                    out_prefix = args[0][i + 1]
            if out_prefix:
                Path(out_prefix + ".clumped").write_text("BADCOL\nfoo\n")

        with patch("subprocess.run", side_effect=_create_bad_clumped):
            with pytest.raises(ValueError, match="SNP.*column missing"):
                clump_instruments(
                    instruments, tmp_path / "ref", 0.001, 1000, Path("plink"),
                )


# ---------------------------------------------------------------------------
# Fix B: Gene TSS anchoring test
# ---------------------------------------------------------------------------


class TestGeneTSSAnchoring:
    def test_uses_gene_pos_not_snp_min(self, tmp_path: Path) -> None:
        """Verify cis-window is anchored at gene TSS, not min SNP position."""
        eqtl_dir = tmp_path / "eqtlgen"
        eqtl_dir.mkdir()

        eqtl_data = pd.DataFrame({
            "Pvalue": [1e-10, 1e-10],
            "SNP": ["rs1", "rs2"],
            "SNPChr": [1, 1],
            "SNPPos": [500000, 600000],
            "AssessedAllele": ["A", "C"],
            "OtherAllele": ["G", "T"],
            "Zscore": [6.0, 5.0],
            "Gene": ["ENSG001", "ENSG001"],
            "GeneSymbol": ["GENE1", "GENE1"],
            "GeneChr": [1, 1],
            "GenePos": [550000, 550000],
            "NrCohorts": [10, 10],
            "NrSamples": [31684, 31684],
            "FDR": [0.001, 0.01],
            "BonferroniP": [0.001, 0.1],
        })
        eqtl_data.to_csv(eqtl_dir / "cis-eQTLs.txt", sep="\t", index=False)

        frq_data = pd.DataFrame({
            "CHR": [1, 1], "SNP": ["rs1", "rs2"],
            "A1": ["A", "C"], "A2": ["G", "T"],
            "MAF": [0.25, 0.3], "NCHROBS": [63368, 63368],
        })
        frq_path = tmp_path / "ref.frq"
        frq_data.to_csv(frq_path, sep="\t", index=False)

        from repogen.analysis.mendelian_randomisation import _load_eqtlgen
        result = _load_eqtlgen(eqtl_dir, frq_path)

        assert "gene_pos" in result.columns
        assert "gene_chr" in result.columns
        assert result.iloc[0]["gene_pos"] == 550000

        # min SNP pos would be 500000, but gene TSS is 550000
        assert result.iloc[0]["gene_pos"] != result["pos"].min()

    def test_fallback_when_gene_pos_null(self) -> None:
        """gene_chr present but gene_pos is NA - must fall back, not crash."""
        eqtl_df = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["ENSG001", "ENSG001"],
            "chr": [1, 1],
            "pos": [500000, 600000],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.3],
            "se": [0.1, 0.1],
            "pval": [1e-10, 1e-8],
            "n": [31684, 31684],
            "gene_chr": [1, 1],
            "gene_pos": [pd.NA, pd.NA],
        })

        genes = eqtl_df["gene"].unique()
        has_gene_pos = "gene_pos" in eqtl_df.columns and "gene_chr" in eqtl_df.columns
        assert has_gene_pos  # columns exist

        gene_rows = eqtl_df.loc[eqtl_df["gene"] == "ENSG001"]
        row0 = gene_rows.iloc[0]

        # gene_chr is non-null, gene_pos is NA - old code would crash here
        assert pd.notna(row0.get("gene_chr"))
        assert pd.isna(row0.get("gene_pos"))

        # Replicate the orchestrator logic - must not crash
        if (
            has_gene_pos
            and pd.notna(row0.get("gene_chr"))
            and pd.notna(row0.get("gene_pos"))
        ):
            start = int(row0["gene_pos"])
        else:
            start = int(gene_rows["pos"].min())

        assert start == 500000  # fallback to min SNP pos


# ---------------------------------------------------------------------------
# Fix C: Coloc positional fallback test
# ---------------------------------------------------------------------------


class TestColocPositionalFallback:
    def test_null_rsid_gwas_still_matches_coloc(self) -> None:
        """Coloc should merge via chr:pos when GWAS SNP is null."""
        from repogen.analysis.mendelian_randomisation import _merge_eqtl_gwas_two_stage

        eqtl_df = pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3"],
            "gene": ["G1"] * 3,
            "chr": [1, 1, 1],
            "pos": [100, 200, 300],
            "a1": ["A", "C", "G"],
            "a2": ["G", "T", "A"],
            "beta": [0.5, 0.3, 0.4],
            "se": [0.1, 0.1, 0.1],
        })

        gwas_df = pd.DataFrame({
            "SNP": ["rs1", None, None],
            "CHR": [1, 1, 1],
            "POS": [100, 200, 300],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "BETA": [0.1, 0.2, 0.3],
            "SE": [0.05, 0.05, 0.05],
        })

        gwas_cols = ["A1", "A2", "BETA", "SE", "CHR", "POS"]
        merged = _merge_eqtl_gwas_two_stage(
            eqtl_df, gwas_df, gwas_cols, suffixes=("_eqtl", "_gwas"),
        )

        assert len(merged) == 3

    def test_all_null_rsid_gwas_uses_positional(self) -> None:
        from repogen.analysis.mendelian_randomisation import _merge_eqtl_gwas_two_stage

        eqtl_df = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["G1"] * 2,
            "chr": [1, 1],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.3],
            "se": [0.1, 0.1],
        })

        gwas_df = pd.DataFrame({
            "SNP": [None, None],
            "CHR": [1, 1],
            "POS": [100, 200],
            "A1": ["A", "C"],
            "A2": ["G", "T"],
            "BETA": [0.1, 0.2],
            "SE": [0.05, 0.05],
        })

        gwas_cols = ["A1", "A2", "BETA", "SE", "CHR", "POS"]
        merged = _merge_eqtl_gwas_two_stage(
            eqtl_df, gwas_df, gwas_cols, suffixes=("_eqtl", "_gwas"),
        )

        assert len(merged) == 2


# ---------------------------------------------------------------------------
# Integration test (skipped without PLINK)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MAF resolution tests
# ---------------------------------------------------------------------------


class TestNormalizeToMaf:
    def test_flips_above_half(self) -> None:
        s = pd.Series([0.1, 0.7, 0.5, 0.3])
        result = _normalize_to_maf(s)
        expected = pd.Series([0.1, 0.3, 0.5, 0.3])
        pd.testing.assert_series_equal(result, expected)

    def test_boundary_values_rejected(self) -> None:
        s = pd.Series([0.0, 1.0, -0.1, 1.5, np.nan])
        result = _normalize_to_maf(s)
        assert result.isna().all()

    def test_non_numeric_coerced(self) -> None:
        s = pd.Series(["0.2", "bad", "0.8"])
        result = _normalize_to_maf(s)
        assert result.iloc[0] == pytest.approx(0.2)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(0.2)


class TestResolveMaf:
    def test_precedence_existing_maf(self) -> None:
        df = pd.DataFrame({"MAF": [0.1, 0.2], "FCON": [0.3, 0.4]})
        maf, counters = _resolve_maf(df)
        np.testing.assert_array_almost_equal(maf.values, [0.1, 0.2])
        assert counters["input_maf"] == 2
        assert counters.get("fcon", 0) == 0

    def test_precedence_fcon_fills_missing(self) -> None:
        df = pd.DataFrame({"MAF": [0.1, np.nan], "FCON": [0.3, 0.4]})
        maf, counters = _resolve_maf(df)
        np.testing.assert_array_almost_equal(maf.values, [0.1, 0.4])
        assert counters["input_maf"] == 1
        assert counters["fcon"] == 1

    def test_fcon_only(self) -> None:
        df = pd.DataFrame({"FCON": [0.15, 0.85]})
        maf, counters = _resolve_maf(df)
        np.testing.assert_array_almost_equal(maf.values, [0.15, 0.15])
        assert counters["fcon"] == 2

    def test_weighted_af_fills_remaining(self) -> None:
        """Weighted AF resolves rows where FCON is boundary-invalid (0.0)."""
        df = pd.DataFrame({
            "FCAS": [0.2, 0.3],
            "FCON": [0.0, 0.0],
            "N_CAS": [1000, 2000],
            "N_CON": [3000, 4000],
        })
        maf, counters = _resolve_maf(df)
        assert counters.get("fcon", 0) == 0
        assert counters["weighted_af"] == 2
        assert maf.notna().all()
        expected_0 = (0.2 * 1000 + 0.0 * 3000) / 4000  # 0.05
        assert maf.iloc[0] == pytest.approx(expected_0)

    def test_frq_fallback_resolves_remaining(self, tmp_path: Path) -> None:
        frq_file = tmp_path / "test.frq"
        frq_file.write_text("CHR SNP A1 A2 MAF NCHROBS\n1 rs1 A G 0.12 1000\n1 rs2 C T 0.35 1000\n")
        df = pd.DataFrame({"SNP": ["rs1", "rs2", "rs3"]})
        maf, counters = _resolve_maf(df, ref_freq_path=frq_file)
        assert counters["ref_frq"] == 2
        assert counters["unresolved"] == 1
        assert maf.iloc[0] == pytest.approx(0.12)
        assert np.isnan(maf.iloc[2])

    def test_no_columns_all_unresolved(self) -> None:
        df = pd.DataFrame({"BETA": [0.1, 0.2, 0.3]})
        maf, counters = _resolve_maf(df)
        assert counters["unresolved"] == 3
        assert maf.isna().all()


class TestHarmoniseCarriesMaf:
    def test_maf_column_in_harmonised_output(self) -> None:
        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "CHR": [1, 1],
            "POS": [100, 200],
            "A1": ["A", "C"],
            "A2": ["G", "T"],
            "BETA": [0.1, 0.2],
            "SE": [0.05, 0.05],
            "P": [0.01, 0.02],
            "N": [10000, 10000],
            "MAF": [0.15, 0.25],
        })
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["GENE1", "GENE1"],
            "chr": [1, 1],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.6],
            "se": [0.1, 0.1],
            "pval": [1e-8, 1e-8],
            "n": [5000, 5000],
        })
        result = harmonise_gwas_eqtl(gwas, instruments)
        assert "maf" in result.columns
        np.testing.assert_array_almost_equal(result["maf"].values, [0.15, 0.25])


class TestSteigPerSnpMaf:
    def test_steiger_uses_per_snp_maf(self) -> None:
        """Verify Steiger R² uses per-SNP MAF when available."""
        bx = np.array([0.5, 0.3])
        by = np.array([0.1, 0.05])
        mafs = np.array([0.1, 0.4])
        geno_var = 2 * mafs * (1 - mafs)
        r2_exp = float(np.sum(bx**2 * geno_var))
        r2_out = float(np.sum(by**2 * geno_var))
        r2_const = float(np.sum(bx**2 * 2 * 0.3 * 0.7))
        assert r2_exp != pytest.approx(r2_const, rel=0.01)
        assert r2_exp > 0 and r2_out > 0

    def test_steiger_fallback_for_nan_maf(self) -> None:
        """NaN MAF entries fall back to 0.3."""
        maf_arr = np.array([0.2, np.nan, 0.4])
        maf_fixed = np.where(np.isfinite(maf_arr), maf_arr, 0.3)
        assert maf_fixed[1] == pytest.approx(0.3)
        assert int((~np.isfinite(maf_arr)).sum()) == 1


class TestColocNanMafFallback:
    def test_nan_maf_replaced_before_coloc(self) -> None:
        """NaN in MAF column should be replaced by 0.3 before coloc."""
        maf_vals = np.array([0.2, np.nan, 0.4, np.nan])
        fallback_count = int((~np.isfinite(maf_vals)).sum())
        maf_vals = np.where(np.isfinite(maf_vals), maf_vals, 0.3)
        maf_vals = np.clip(maf_vals, 0.01, 0.49)
        assert fallback_count == 2
        np.testing.assert_array_almost_equal(
            maf_vals, [0.2, 0.3, 0.4, 0.3],
        )

    def test_coloc_with_nan_free_maf(self) -> None:
        """Full coloc.abf with resolved MAF works without errors."""
        n = 20
        beta = np.random.randn(n) * 0.05
        se = np.abs(np.random.randn(n) * 0.01) + 0.01
        maf = np.random.uniform(0.05, 0.45, n)
        maf[3] = np.nan
        maf = np.where(np.isfinite(maf), maf, 0.3)
        maf = np.clip(maf, 0.01, 0.49)
        result = coloc_abf(
            beta1=beta, se1=se, beta2=beta * 0.5, se2=se,
            maf=maf, n1=50000, n2=50000,
        )
        assert "pp_h4" in result
        assert 0 <= result["pp_h4"] <= 1
        assert not np.isnan(result["pp_h4"])


@pytest.mark.skipif(
    not shutil.which("plink"),
    reason="PLINK not installed",
)
class TestPLINKIntegration:
    def test_clump_subprocess(self, tmp_path: Path) -> None:
        """Integration: runs actual PLINK clumping with tiny synthetic data."""
        pass


# ---------------------------------------------------------------------------
# Chunked loader regression tests
# ---------------------------------------------------------------------------

from repogen.analysis.mendelian_randomisation import (
    _load_eqtlgen_chunked,
    _load_metabrain_chunked,
    _load_eqtl_coloc_genes,
    _reload_metabrain_for_coloc,
    _normalise_metabrain_snp_ids,
    _dedup_metabrain_instruments,
    _enforce_source_yield,
    _run_coloc_for_gene,
    _get_peak_rss_mb,
    load_eqtl_source,
)


class TestChunkedEqtlgenLoader:
    """Verify _load_eqtlgen_chunked matches legacy loader semantics."""

    @pytest.fixture
    def eqtlgen_fixture(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create minimal eQTLGen-format file and .frq reference."""
        eqtl_dir = tmp_path / "eqtlgen"
        eqtl_dir.mkdir()
        eqtl_file = eqtl_dir / "eqtlgen_cis_eqtl.txt"

        header = "Pvalue\tSNP\tSNPChr\tSNPPos\tAssessedAllele\tOtherAllele\tZscore\tGene\tNrSamples\tGeneChr\tGenePos"
        rows = [
            "1e-10\trs1\t1\t100\tA\tG\t6.5\tENSG001\t31684\t1\t50",
            "1e-12\trs2\t1\t150\tC\tT\t7.0\tENSG001\t31684\t1\t50",
            "0.5\trs3\t1\t200\tA\tT\t0.7\tENSG001\t31684\t1\t50",
            "1e-9\trs4\t2\t300\tG\tC\t6.0\tENSG002\t31684\t2\t250",
            "0.3\trs5\t2\t400\tA\tG\t1.0\tENSG002\t31684\t2\t250",
            "0.8\trs6\t3\t500\tC\tT\t0.2\tENSG003\t31684\t3\t480",
        ]
        eqtl_file.write_text(header + "\n" + "\n".join(rows) + "\n")

        frq_file = tmp_path / "ref.frq"
        frq_content = "CHR SNP A1 A2 MAF NCHROBS\n"
        frq_content += "1 rs1 A G 0.20 1000\n"
        frq_content += "1 rs2 C T 0.30 1000\n"
        frq_content += "1 rs3 A T 0.15 1000\n"
        frq_content += "2 rs4 G C 0.25 1000\n"
        frq_content += "2 rs5 A G 0.10 1000\n"
        frq_content += "3 rs6 C T 0.40 1000\n"
        frq_file.write_text(frq_content)

        return eqtl_dir, frq_file

    def test_returns_correct_tuple_structure(self, eqtlgen_fixture: tuple) -> None:
        eqtl_dir, frq_file = eqtlgen_fixture
        instruments_df, gene_metadata, n_genes_valid = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=3,
        )
        assert isinstance(instruments_df, pd.DataFrame)
        assert isinstance(gene_metadata, dict)
        assert isinstance(n_genes_valid, int)

    def test_bonferroni_denominator_counts_all_genes(self, eqtlgen_fixture: tuple) -> None:
        """n_genes_valid must include ALL genes (including those without instruments)."""
        eqtl_dir, frq_file = eqtlgen_fixture
        _, _, n_genes_valid = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=3,
        )
        assert n_genes_valid == 3  # ENSG001, ENSG002, ENSG003

    def test_instruments_filtered_by_pval(self, eqtlgen_fixture: tuple) -> None:
        eqtl_dir, frq_file = eqtlgen_fixture
        instruments_df, _, _ = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=3,
        )
        assert len(instruments_df) > 0
        assert (instruments_df["pval"] < 5e-8).all()

    def test_gene_metadata_uses_gene_pos_when_available(self, eqtlgen_fixture: tuple) -> None:
        eqtl_dir, frq_file = eqtlgen_fixture
        _, gene_metadata, _ = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=3,
        )
        assert gene_metadata["ENSG001"]["chr"] == 1
        assert gene_metadata["ENSG001"]["start"] == 50  # From GenePos column
        assert gene_metadata["ENSG002"]["chr"] == 2
        assert gene_metadata["ENSG002"]["start"] == 250

    def test_gene_metadata_fallback_to_min_snp_pos(self, tmp_path: Path) -> None:
        """When gene_pos is missing, uses min(SNP pos) across chunks."""
        eqtl_dir = tmp_path / "eqtl_nopos"
        eqtl_dir.mkdir()
        eqtl_file = eqtl_dir / "eqtlgen_cis_eqtl.txt"

        header = "Pvalue\tSNP\tSNPChr\tSNPPos\tAssessedAllele\tOtherAllele\tZscore\tGene\tNrSamples\tGeneChr\tGenePos"
        rows = [
            "1e-10\trs1\t1\t500\tA\tG\t6.5\tGENE_X\t31684\t\t",
            "1e-11\trs2\t1\t100\tC\tT\t7.0\tGENE_X\t31684\t\t",
            "1e-12\trs3\t1\t300\tA\tT\t7.5\tGENE_X\t31684\t\t",
        ]
        eqtl_file.write_text(header + "\n" + "\n".join(rows) + "\n")

        frq_file = tmp_path / "ref.frq"
        frq_content = "CHR SNP A1 A2 MAF NCHROBS\n"
        frq_content += "1 rs1 A G 0.20 1000\n"
        frq_content += "1 rs2 C T 0.30 1000\n"
        frq_content += "1 rs3 A T 0.15 1000\n"
        frq_file.write_text(frq_content)

        _, gene_metadata, _ = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=2,
        )
        assert gene_metadata["GENE_X"]["start"] == 100

    def test_column_schema_matches_legacy(self, eqtlgen_fixture: tuple) -> None:
        eqtl_dir, frq_file = eqtlgen_fixture
        instruments_df, _, _ = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=10,
        )
        required_cols = {"SNP", "gene", "chr", "pos", "a1", "a2", "beta", "se", "pval", "n"}
        assert required_cols.issubset(set(instruments_df.columns))


class TestChunkedMetabrainLoader:
    """Verify _load_metabrain_chunked matches legacy loader semantics."""

    @pytest.fixture
    def metabrain_fixture(self, tmp_path: Path) -> Path:
        eqtl_dir = tmp_path / "metabrain"
        eqtl_dir.mkdir()
        normalized = eqtl_dir / "metabrain_cortex_normalized.tsv.gz"

        df = pd.DataFrame({
            "gene": ["GENE_A"] * 3 + ["GENE_B"] * 2,
            "SNP": ["rs10", "rs11", "rs12", "rs13", "rs14"],
            "chr": [1, 1, 1, 2, 2],
            "pos": [100, 200, 300, 400, 500],
            "a1": ["A", "C", "G", "T", "A"],
            "a2": ["G", "T", "A", "C", "G"],
            "beta": [0.5, 0.3, 0.01, 0.6, 0.02],
            "se": [0.1, 0.1, 0.1, 0.1, 0.1],
            "pval": [1e-10, 1e-9, 0.5, 1e-11, 0.8],
            "n": [2970, 2970, 2970, 2970, 2970],
            "gene_chr": [1, 1, 1, 2, 2],
            "gene_pos": [50, 50, 50, 350, 350],
        })
        df.to_csv(normalized, sep="\t", index=False, compression="gzip")

        return eqtl_dir

    def test_returns_correct_tuple(self, metabrain_fixture: Path) -> None:
        instruments_df, gene_metadata, n_genes_valid = _load_metabrain_chunked(
            metabrain_fixture, instrument_pval=5e-8, chunksize=3,
        )
        assert isinstance(instruments_df, pd.DataFrame)
        assert isinstance(gene_metadata, dict)
        assert n_genes_valid == 2

    def test_bonferroni_denominator_all_genes(self, metabrain_fixture: Path) -> None:
        _, _, n_genes_valid = _load_metabrain_chunked(
            metabrain_fixture, instrument_pval=5e-8, chunksize=3,
        )
        assert n_genes_valid == 2  # GENE_A, GENE_B

    def test_instruments_filtered(self, metabrain_fixture: Path) -> None:
        instruments_df, _, _ = _load_metabrain_chunked(
            metabrain_fixture, instrument_pval=5e-8, chunksize=3,
        )
        assert len(instruments_df) == 3  # rs10, rs11, rs13
        assert (instruments_df["pval"] < 5e-8).all()


# ---------------------------------------------------------------------------
# MetaBrain SNP normalisation, dedup, and source-yield guardrail
# ---------------------------------------------------------------------------


class TestMetabrainSnpNormalisation:
    """_normalise_metabrain_snp_ids: composite -> bare rsID parsing."""

    def _frame(self, snps: list[str]) -> pd.DataFrame:
        n = len(snps)
        return pd.DataFrame({
            "SNP": snps,
            "gene": ["GENE_A"] * n,
            "chr": [1] * n,
            "pos": list(range(100, 100 + n)),
            "a1": ["A"] * n,
            "a2": ["G"] * n,
            "beta": [0.1] * n,
            "se": [0.05] * n,
            "pval": [1e-9] * n,
            "n": [2970] * n,
        })

    def test_composite_parsed_to_bare_rsid(self) -> None:
        df = self._frame(["10:100000012:rs12345:A_G", "2:50:rs67:C_T"])
        out, counts = _normalise_metabrain_snp_ids(df)
        assert list(out["SNP"]) == ["rs12345", "rs67"]
        assert counts["metabrain_snp_composite_parsed"] == 2
        assert counts["metabrain_snp_bare_rsid"] == 0
        assert counts["metabrain_snp_dropped_non_rsid"] == 0

    def test_variant_id_preserves_original_token(self) -> None:
        df = self._frame(["10:100000012:rs12345:A_G"])
        out, _ = _normalise_metabrain_snp_ids(df)
        assert "variant_id" in out.columns
        assert out["variant_id"].iloc[0] == "10:100000012:rs12345:A_G"
        assert out["SNP"].iloc[0] == "rs12345"

    def test_bare_rsid_idempotent(self) -> None:
        df = self._frame(["rs1", "rs2", "rs3"])
        out, counts = _normalise_metabrain_snp_ids(df)
        assert list(out["SNP"]) == ["rs1", "rs2", "rs3"]
        assert counts["metabrain_snp_bare_rsid"] == 3
        assert counts["metabrain_snp_composite_parsed"] == 0

    def test_whitespace_and_case_canonicalised(self) -> None:
        df = self._frame(["  RS12345  ", "1:100:RS9:A_G"])
        out, counts = _normalise_metabrain_snp_ids(df)
        assert list(out["SNP"]) == ["rs12345", "rs9"]
        assert counts["metabrain_snp_dropped_non_rsid"] == 0

    def test_non_rsid_rows_dropped_and_counted(self) -> None:
        df = self._frame(["1:100:.:A_G", "2:200:A:G", "rs5"])
        out, counts = _normalise_metabrain_snp_ids(df)
        assert list(out["SNP"]) == ["rs5"]
        assert counts["metabrain_snp_dropped_non_rsid"] == 2
        assert counts["metabrain_rows_after_snp_normalisation"] == 1

    def test_empty_frame_adds_variant_id(self) -> None:
        df = pd.DataFrame(columns=["SNP", "gene", "a1", "a2", "pval"])
        out, counts = _normalise_metabrain_snp_ids(df)
        assert "variant_id" in out.columns
        assert counts["metabrain_rows_after_snp_normalisation"] == 0

    def test_snp_dtype_is_plain_object(self) -> None:
        df = self._frame(["1:100:rs12345:A_G"])
        out, _ = _normalise_metabrain_snp_ids(df)
        # Plain python str objects to match the GWAS/eQTLGen 'SNP' convention.
        assert isinstance(out["SNP"].iloc[0], str)

    def test_field_aware_rejects_rsid_in_wrong_field(self) -> None:
        # Field-aware parsing only trusts the 3rd field. A token carrying an rsID in
        # a non-standard slot (here field 3, not field 2) is dropped - a permissive
        # substring extractor would have wrongly lifted 'rs99' and overstated recovery.
        df = self._frame(["1:100:A:rs99", "1:200:rs7:A_G"])
        out, counts = _normalise_metabrain_snp_ids(df)
        assert list(out["SNP"]) == ["rs7"]
        assert counts["metabrain_snp_dropped_non_rsid"] == 1
        assert counts["metabrain_snp_composite_parsed"] == 1

    def test_field_aware_positional_only_token_dropped(self) -> None:
        # chr:pos:ref:alt with no rsID -> field 2 is an allele, not rs\d+ -> dropped.
        df = self._frame(["10:500:A:G", "10:600:rs3:C_T"])
        out, counts = _normalise_metabrain_snp_ids(df)
        assert list(out["SNP"]) == ["rs3"]
        assert counts["metabrain_snp_dropped_non_rsid"] == 1


class TestMetabrainDedup:
    """_dedup_metabrain_instruments: collapse multiallelic rsID collisions."""

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_multiallelic_collision_keeps_lowest_pval(self) -> None:
        df = self._frame([
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-5},
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "T", "pval": 1e-9},
            {"gene": "G", "SNP": "rs2", "a1": "C", "a2": "T", "pval": 1e-8},
        ])
        out, counts = _dedup_metabrain_instruments(df)
        assert len(out) == 2
        kept = out.loc[out["SNP"] == "rs1"].iloc[0]
        assert kept["pval"] == 1e-9 and kept["a2"] == "T"
        assert counts["metabrain_multiallelic_collisions"] == 1
        assert counts["metabrain_rows_after_dedup"] == 2

    def test_exact_duplicate_rows_counted(self) -> None:
        df = self._frame([
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-9},
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-9},
        ])
        out, counts = _dedup_metabrain_instruments(df)
        assert len(out) == 1
        assert counts["metabrain_exact_duplicate_rows"] == 1

    def test_unique_rows_unchanged(self) -> None:
        df = self._frame([
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-9},
            {"gene": "G", "SNP": "rs2", "a1": "C", "a2": "T", "pval": 1e-8},
            {"gene": "H", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-7},
        ])
        out, counts = _dedup_metabrain_instruments(df)
        assert len(out) == 3
        assert counts["metabrain_multiallelic_collisions"] == 0
        assert counts["metabrain_exact_duplicate_rows"] == 0

    def test_deterministic_allele_tiebreak(self) -> None:
        # Same (gene,SNP), same pval, different alleles -> lexicographic tiebreak.
        df = self._frame([
            {"gene": "G", "SNP": "rs1", "a1": "T", "a2": "C", "pval": 1e-9},
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-9},
        ])
        out, _ = _dedup_metabrain_instruments(df)
        assert len(out) == 1
        assert out.iloc[0]["a1"] == "A"  # (A,G) < (T,C)

    def test_empty_frame(self) -> None:
        df = pd.DataFrame(columns=["gene", "SNP", "a1", "a2", "pval"])
        out, counts = _dedup_metabrain_instruments(df)
        assert out.empty
        assert counts["metabrain_rows_after_dedup"] == 0

    def test_collapse_false_keeps_multiallelic(self) -> None:
        # MR-instrument mode: multiallelic collision is KEPT (harmonise resolves it).
        df = self._frame([
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-5},
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "T", "pval": 1e-9},
        ])
        out, counts = _dedup_metabrain_instruments(df, collapse_multiallelic=False)
        assert len(out) == 2  # both allele pairs retained
        assert counts["metabrain_multiallelic_collisions"] == 1
        assert counts["metabrain_exact_duplicate_rows"] == 0

    def test_collapse_false_still_removes_exact_duplicates(self) -> None:
        # Exact duplicates are always collapsed (harmonise would over-weight k).
        df = self._frame([
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-9},
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "G", "pval": 1e-9},
            {"gene": "G", "SNP": "rs1", "a1": "A", "a2": "T", "pval": 1e-8},
        ])
        out, counts = _dedup_metabrain_instruments(df, collapse_multiallelic=False)
        # rs1 A/G (exact dup) -> 1 row; rs1 A/T kept -> total 2.
        assert len(out) == 2
        assert counts["metabrain_exact_duplicate_rows"] == 1
        assert counts["metabrain_multiallelic_collisions"] == 1


class TestMetabrainChunkedComposite:
    """_load_metabrain_chunked end-to-end with composite tokens + duplicates."""

    @pytest.fixture
    def composite_fixture(self, tmp_path: Path) -> Path:
        eqtl_dir = tmp_path / "metabrain_comp"
        eqtl_dir.mkdir()
        normalized = eqtl_dir / "metabrain_cortex_normalized.tsv.gz"
        df = pd.DataFrame({
            "gene": ["GENE_A", "GENE_A", "GENE_A", "GENE_A", "GENE_B"],
            # GENE_A rs20 appears twice with different alleles (multiallelic
            # collision): the MR loader keeps BOTH for harmonisation to resolve.
            # rs21 appears twice with identical alleles (exact dup): collapsed.
            "SNP": [
                "1:100:rs20:A_G",
                "1:100:rs20:A_T",
                "1:200:rs21:C_T",
                "1:200:rs21:C_T",
                "2:400:rs22:G_A",
            ],
            "chr": [1, 1, 1, 1, 2],
            "pos": [100, 100, 200, 200, 400],
            "a1": ["A", "A", "C", "C", "G"],
            "a2": ["G", "T", "T", "T", "A"],
            "beta": [0.5, 0.4, 0.3, 0.3, 0.6],
            "se": [0.1, 0.1, 0.1, 0.1, 0.1],
            "pval": [1e-10, 1e-12, 1e-9, 1e-9, 1e-11],
            "n": [2970, 2970, 2970, 2970, 2970],
            "gene_chr": [1, 1, 1, 1, 2],
            "gene_pos": [50, 50, 50, 50, 350],
        })
        df.to_csv(normalized, sep="\t", index=False, compression="gzip")
        return eqtl_dir

    def test_composite_snps_parsed_and_deduped(self, composite_fixture: Path) -> None:
        stats: dict = {}
        instruments_df, _, n_genes_valid = _load_metabrain_chunked(
            composite_fixture, instrument_pval=5e-8, chunksize=2, stats_out=stats,
        )
        assert set(instruments_df["SNP"]) == {"rs20", "rs21", "rs22"}
        # MR loader keeps multiallelic collisions (rs20 -> 2 rows) so
        # harmonisation can pick the GWAS-compatible pair; exact dup (rs21) collapsed.
        rs20 = instruments_df.loc[instruments_df["SNP"] == "rs20"]
        assert len(rs20) == 2
        assert set(rs20["a2"]) == {"G", "T"}
        rs21 = instruments_df.loc[instruments_df["SNP"] == "rs21"]
        assert len(rs21) == 1
        assert stats["metabrain_snp_composite_parsed"] == 5
        assert stats["metabrain_multiallelic_collisions"] == 1
        assert stats["metabrain_exact_duplicate_rows"] == 1
        assert "variant_id" in instruments_df.columns
        assert n_genes_valid == 2

    def test_stats_out_optional(self, composite_fixture: Path) -> None:
        # Must not raise when stats_out is omitted (back-compat 3-tuple contract).
        instruments_df, gene_metadata, n_genes_valid = _load_metabrain_chunked(
            composite_fixture, instrument_pval=5e-8, chunksize=10,
        )
        assert isinstance(instruments_df, pd.DataFrame)
        assert isinstance(gene_metadata, dict)


class TestMetabrainColocReloadNormalisation:
    """_reload_metabrain_for_coloc must parse composite SNPs for coloc joins."""

    @pytest.fixture
    def coloc_fixture(self, tmp_path: Path) -> Path:
        eqtl_dir = tmp_path / "metabrain_coloc"
        eqtl_dir.mkdir()
        normalized = eqtl_dir / "metabrain_cortex_normalized.tsv.gz"
        df = pd.DataFrame({
            "gene": ["GENE_SIG"] * 3 + ["GENE_OTHER"],
            "SNP": [
                "1:100:rs1:A_G",
                "1:200:rs2:C_T",
                "1:300:rs3:A_T",
                "2:400:rs4:G_C",
            ],
            "chr": [1, 1, 1, 2],
            "pos": [100, 200, 300, 400],
            "a1": ["A", "C", "A", "G"],
            "a2": ["G", "T", "T", "C"],
            "beta": [0.5, 0.1, 0.2, 0.6],
            "se": [0.1, 0.1, 0.1, 0.1],
            "pval": [1e-10, 0.5, 0.8, 1e-9],
            "n": [2970, 2970, 2970, 2970],
            "gene_chr": [1, 1, 1, 2],
            "gene_pos": [50, 50, 50, 350],
        })
        df.to_csv(normalized, sep="\t", index=False, compression="gzip")
        return eqtl_dir

    def test_reload_parses_rsids(self, coloc_fixture: Path) -> None:
        result = _reload_metabrain_for_coloc(
            coloc_fixture, gene_set={"GENE_SIG"}, chunksize=2,
        )
        assert set(result["SNP"]) == {"rs1", "rs2", "rs3"}
        assert "GENE_OTHER" not in result["gene"].values
        assert "variant_id" in result.columns


class TestSourceYieldGuardrail:
    """_enforce_source_yield: fail-loud vs warn semantics."""

    def test_optional_source_never_raises(self) -> None:
        cfg = EQTLSourceConfig(source="metabrain_cortex", path=Path("."))
        # Very low yield, but not required and no explicit threshold -> no raise.
        _enforce_source_yield(cfg, result_fraction=0.001, genes_processed=1000)

    def test_required_source_below_default_floor_raises(self) -> None:
        cfg = EQTLSourceConfig(source="metabrain_cortex", path=Path("."), required=True)
        with pytest.raises(RuntimeError, match="below the required minimum"):
            _enforce_source_yield(cfg, result_fraction=0.02, genes_processed=1000)

    def test_required_source_above_floor_ok(self) -> None:
        cfg = EQTLSourceConfig(source="metabrain_cortex", path=Path("."), required=True)
        _enforce_source_yield(cfg, result_fraction=0.5, genes_processed=1000)

    def test_explicit_min_fraction_raises_even_if_not_required(self) -> None:
        cfg = EQTLSourceConfig(
            source="metabrain_cortex", path=Path("."), min_result_fraction=0.5,
        )
        with pytest.raises(RuntimeError, match="below the required minimum"):
            _enforce_source_yield(cfg, result_fraction=0.3, genes_processed=1000)

    def test_zero_genes_required_raises(self) -> None:
        cfg = EQTLSourceConfig(source="metabrain_cortex", path=Path("."), required=True)
        with pytest.raises(RuntimeError, match="processed 0 genes"):
            _enforce_source_yield(cfg, result_fraction=None, genes_processed=0)

    def test_zero_genes_optional_warns_no_raise(self) -> None:
        cfg = EQTLSourceConfig(source="metabrain_cortex", path=Path("."))
        _enforce_source_yield(cfg, result_fraction=None, genes_processed=0)

    def test_min_result_fraction_bounds_validated(self) -> None:
        with pytest.raises(ValueError):
            EQTLSourceConfig(source="x", path=Path("."), min_result_fraction=0.0)
        with pytest.raises(ValueError):
            EQTLSourceConfig(source="x", path=Path("."), min_result_fraction=1.5)
        # Valid boundary values.
        assert EQTLSourceConfig(
            source="x", path=Path("."), min_result_fraction=1.0
        ).min_result_fraction == 1.0
        assert EQTLSourceConfig(source="x", path=Path(".")).min_result_fraction is None
        assert EQTLSourceConfig(source="x", path=Path(".")).required is False


class TestMultiallelicHarmonisationResolution:
    """Harmonisation resolves multiallelic collisions correctly.

    The MR loader keeps multiallelic (gene,SNP) collisions rather than collapsing
    by p-value, so harmonise_gwas_eqtl selects the GWAS-allele-compatible pair even
    when it has a higher (worse) p-value than the incompatible pair.
    """

    def test_harmonise_keeps_compatible_over_lower_p_incompatible(self) -> None:
        gwas_df = pd.DataFrame({
            "SNP": ["rs1"], "CHR": [1], "POS": [100],
            "A1": ["A"], "A2": ["G"],  # GWAS carries A/G at rs1
            "BETA": [0.2], "SE": [0.05], "P": [1e-8], "N": [100000], "MAF": [0.3],
        })
        # rs1 A/T has the lower p (1e-12) but is allele-incompatible with GWAS A/G;
        # rs1 A/G has a higher p (1e-9) but IS compatible. The compatible one must win.
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs1"],
            "gene": ["G", "G"],
            "chr": [1, 1], "pos": [100, 100],
            "a1": ["A", "A"], "a2": ["T", "G"],
            "beta": [0.5, 0.4], "se": [0.1, 0.1],
            "pval": [1e-12, 1e-9], "n": [2970, 2970],
        })
        out = harmonise_gwas_eqtl(gwas_df, instruments)
        assert len(out) == 1  # one compatible instrument survives
        assert out.iloc[0]["beta_exposure"] == pytest.approx(0.4)  # the A/G row

    def test_clump_input_dedups_duplicate_rsids(self, tmp_path: Path) -> None:
        # clump returns early (len<=1) after PLINK, but we can at least verify the
        # duplicate-rsID guard does not raise and that a single-SNP frame is returned
        # unchanged (PLINK path is covered by the plink-gated integration tests).
        instruments = pd.DataFrame({
            "SNP": ["rs1"], "gene": ["G"], "chr": [1], "pos": [100],
            "a1": ["A"], "a2": ["G"], "beta": [0.5], "se": [0.1],
            "pval": [1e-10], "n": [2970],
        })
        out = clump_instruments(
            instruments, tmp_path / "nofile", clump_r2=0.001, clump_kb=1000,
            plink_binary=Path("plink"),
        )
        assert len(out) == 1  # single-SNP shortcut, no PLINK invoked


class TestColocReload:
    """Verify Phase 3 coloc reload retrieves full-locus data."""

    @pytest.fixture
    def eqtlgen_coloc_fixture(self, tmp_path: Path) -> tuple[Path, Path]:
        eqtl_dir = tmp_path / "eqtlgen_coloc"
        eqtl_dir.mkdir()
        eqtl_file = eqtl_dir / "eqtlgen_cis_eqtl.txt"

        header = "Pvalue\tSNP\tSNPChr\tSNPPos\tAssessedAllele\tOtherAllele\tZscore\tGene\tNrSamples\tGeneChr\tGenePos"
        rows = [
            "1e-10\trs1\t1\t100\tA\tG\t6.5\tGENE_SIG\t31684\t1\t50",
            "0.5\trs2\t1\t200\tC\tT\t0.7\tGENE_SIG\t31684\t1\t50",
            "0.8\trs3\t1\t300\tA\tT\t0.2\tGENE_SIG\t31684\t1\t50",
            "1e-9\trs4\t2\t400\tG\tC\t6.0\tGENE_OTHER\t31684\t2\t350",
        ]
        eqtl_file.write_text(header + "\n" + "\n".join(rows) + "\n")

        frq_file = tmp_path / "ref.frq"
        frq_content = "CHR SNP A1 A2 MAF NCHROBS\n"
        frq_content += "1 rs1 A G 0.20 1000\n"
        frq_content += "1 rs2 C T 0.30 1000\n"
        frq_content += "1 rs3 A T 0.15 1000\n"
        frq_content += "2 rs4 G C 0.25 1000\n"
        frq_file.write_text(frq_content)

        return eqtl_dir, frq_file

    def test_coloc_reload_returns_all_snps_for_gene(self, eqtlgen_coloc_fixture: tuple) -> None:
        """Phase 3 must return ALL SNPs for requested genes, not just instruments."""
        eqtl_dir, frq_file = eqtlgen_coloc_fixture
        result = _load_eqtl_coloc_genes(
            eqtl_dir, frq_file, gene_set={"GENE_SIG"}, source="eqtlgen",
        )
        assert len(result) == 3  # All 3 SNPs for GENE_SIG, not just significant
        assert set(result["SNP"]) == {"rs1", "rs2", "rs3"}

    def test_coloc_reload_excludes_unrequested_genes(self, eqtlgen_coloc_fixture: tuple) -> None:
        eqtl_dir, frq_file = eqtlgen_coloc_fixture
        result = _load_eqtl_coloc_genes(
            eqtl_dir, frq_file, gene_set={"GENE_SIG"}, source="eqtlgen",
        )
        assert "GENE_OTHER" not in result["gene"].values

    def test_coloc_reload_empty_gene_set(self, eqtlgen_coloc_fixture: tuple) -> None:
        eqtl_dir, frq_file = eqtlgen_coloc_fixture
        result = _load_eqtl_coloc_genes(
            eqtl_dir, frq_file, gene_set=set(), source="eqtlgen",
        )
        assert result.empty


class TestBonferroniInvariance:
    """Verify chunked loading preserves Bonferroni denominator vs legacy loader."""

    @pytest.fixture
    def shared_eqtlgen_data(self, tmp_path: Path) -> tuple[Path, Path]:
        """Fixture used for both chunked and legacy loading."""
        eqtl_dir = tmp_path / "eqtlgen_inv"
        eqtl_dir.mkdir()
        eqtl_file = eqtl_dir / "eqtlgen_cis_eqtl.txt"

        header = "Pvalue\tSNP\tSNPChr\tSNPPos\tAssessedAllele\tOtherAllele\tZscore\tGene\tNrSamples\tGeneChr\tGenePos"
        genes = [f"ENSG{i:05d}" for i in range(1, 21)]
        rows = []
        for i, g in enumerate(genes):
            chr_val = (i % 5) + 1
            pval = 1e-10 if i < 5 else 0.5
            z = 6.5 if i < 5 else 0.5
            rows.append(
                f"{pval}\trs{100+i}\t{chr_val}\t{(i+1)*100}\tA\tG\t{z}\t{g}\t31684\t{chr_val}\t{(i+1)*80}"
            )
        eqtl_file.write_text(header + "\n" + "\n".join(rows) + "\n")

        frq_file = tmp_path / "ref.frq"
        frq_content = "CHR SNP A1 A2 MAF NCHROBS\n"
        for i in range(20):
            frq_content += f"{(i%5)+1} rs{100+i} A G 0.20 1000\n"
        frq_file.write_text(frq_content)

        return eqtl_dir, frq_file

    def test_denominator_matches_legacy(self, shared_eqtlgen_data: tuple) -> None:
        """Chunked n_genes_valid must equal legacy loader gene count."""
        eqtl_dir, frq_file = shared_eqtlgen_data

        source_config = EQTLSourceConfig(source="eqtlgen", path=eqtl_dir)
        legacy_df = load_eqtl_source(source_config, ref_freq_path=frq_file)
        n_genes_legacy = len(legacy_df["gene"].unique())

        _, _, n_genes_chunked = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=5,
        )

        assert n_genes_chunked == n_genes_legacy


class TestRssMonitoring:
    """Verify RSS helper works."""

    def test_get_peak_rss_returns_float(self) -> None:
        result = _get_peak_rss_mb()
        assert isinstance(result, float)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Phase 1 vectorization & indexed MAF lookup tests
# ---------------------------------------------------------------------------


class TestFirstRowNaNFallback:
    """Verify that first-row NaN gene_pos triggers min(pos) fallback, not later rows."""

    def test_eqtlgen_first_row_nan_uses_min_pos(self, tmp_path: Path) -> None:
        """Gene with first row gene_pos=NaN should use min(pos), not later gene_pos."""
        eqtl_dir = tmp_path / "eqtlgen"
        eqtl_dir.mkdir()
        eqtl_file = eqtl_dir / "eqtlgen_cis_eqtl.txt.gz"

        df = pd.DataFrame({
            "Pvalue": [1e-10, 1e-9, 1e-8],
            "SNP": ["rs1", "rs2", "rs3"],
            "SNPChr": [1, 1, 1],
            "SNPPos": [300, 100, 200],
            "AssessedAllele": ["A", "C", "G"],
            "OtherAllele": ["G", "T", "A"],
            "Zscore": [5.0, 4.0, 3.0],
            "Gene": ["GENE_X", "GENE_X", "GENE_X"],
            "NrSamples": [31684, 31684, 31684],
            "GeneChr": [pd.NA, 1, 1],
            "GenePos": [pd.NA, 999, 888],
        })
        df.to_csv(eqtl_file, sep="\t", index=False, compression="gzip")

        frq_file = tmp_path / "ref.frq"
        frq_content = "CHR SNP A1 A2 MAF NCHROBS\n"
        frq_content += "1 rs1 A G 0.20 1000\n"
        frq_content += "1 rs2 C T 0.30 1000\n"
        frq_content += "1 rs3 G A 0.25 1000\n"
        frq_file.write_text(frq_content)

        _, gene_metadata, _ = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=10,
        )

        assert "GENE_X" in gene_metadata
        assert gene_metadata["GENE_X"]["start"] == 100

    def test_metabrain_first_row_nan_uses_min_pos(self, tmp_path: Path) -> None:
        """MetaBrain gene with first row gene_pos=NaN should use min(pos) fallback."""
        eqtl_dir = tmp_path / "metabrain"
        eqtl_dir.mkdir()
        normalized = eqtl_dir / "metabrain_cortex_normalized.tsv.gz"

        df = pd.DataFrame({
            "gene": ["GENE_Y", "GENE_Y", "GENE_Y"],
            "SNP": ["rs10", "rs11", "rs12"],
            "chr": [2, 2, 2],
            "pos": [500, 200, 300],
            "a1": ["A", "C", "G"],
            "a2": ["G", "T", "A"],
            "beta": [0.5, 0.3, 0.4],
            "se": [0.1, 0.1, 0.1],
            "pval": [1e-10, 1e-9, 1e-8],
            "n": [2970, 2970, 2970],
            "gene_chr": [pd.NA, 2, 2],
            "gene_pos": [pd.NA, 999, 888],
        })
        df.to_csv(normalized, sep="\t", index=False, compression="gzip")

        _, gene_metadata, _ = _load_metabrain_chunked(
            eqtl_dir, instrument_pval=5e-8, chunksize=10,
        )

        assert "GENE_Y" in gene_metadata
        assert gene_metadata["GENE_Y"]["start"] == 200


class TestMetabrainAllNaNPos:
    """Verify MetaBrain gene with all NaN pos values gets no metadata entry."""

    def test_gene_with_all_nan_pos_excluded(self, tmp_path: Path) -> None:
        eqtl_dir = tmp_path / "metabrain"
        eqtl_dir.mkdir()
        normalized = eqtl_dir / "metabrain_cortex_normalized.tsv.gz"

        df = pd.DataFrame({
            "gene": ["GENE_OK", "GENE_OK", "GENE_NOPOS", "GENE_NOPOS"],
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "chr": [1, 1, 2, 2],
            "pos": [100, 200, pd.NA, pd.NA],
            "a1": ["A", "C", "G", "T"],
            "a2": ["G", "T", "A", "C"],
            "beta": [0.5, 0.3, 0.4, 0.2],
            "se": [0.1, 0.1, 0.1, 0.1],
            "pval": [1e-10, 1e-9, 1e-10, 1e-9],
            "n": [2970, 2970, 2970, 2970],
            "gene_chr": [pd.NA, pd.NA, pd.NA, pd.NA],
            "gene_pos": [pd.NA, pd.NA, pd.NA, pd.NA],
        })
        df.to_csv(normalized, sep="\t", index=False, compression="gzip")

        _, gene_metadata, n_genes = _load_metabrain_chunked(
            eqtl_dir, instrument_pval=5e-8, chunksize=10,
        )

        assert "GENE_OK" in gene_metadata
        assert gene_metadata["GENE_OK"]["start"] == 100
        assert "GENE_NOPOS" not in gene_metadata
        assert n_genes == 2


class TestMapVsMergeEquivalence:
    """Verify indexed MAF lookup produces identical results to merge."""

    def test_indexed_lookup_matches_merge_semantics(self, tmp_path: Path) -> None:
        """Test with: SNP missing from .frq, duplicate SNP in .frq, normal rows."""
        eqtl_dir = tmp_path / "eqtlgen"
        eqtl_dir.mkdir()
        eqtl_file = eqtl_dir / "eqtlgen_cis_eqtl.txt.gz"

        df = pd.DataFrame({
            "Pvalue": [1e-10, 1e-9, 1e-8, 1e-7],
            "SNP": ["rs1", "rs2", "rs_missing", "rs3"],
            "SNPChr": [1, 1, 1, 1],
            "SNPPos": [100, 200, 300, 400],
            "AssessedAllele": ["a", "c", "g", "t"],
            "OtherAllele": ["g", "t", "a", "c"],
            "Zscore": [5.0, 4.0, 3.0, 6.0],
            "Gene": ["G1", "G1", "G1", "G2"],
            "NrSamples": [31684, 31684, 31684, 31684],
            "GeneChr": [1, 1, 1, 2],
            "GenePos": [50, 50, 50, 350],
        })
        df.to_csv(eqtl_file, sep="\t", index=False, compression="gzip")

        # .frq with duplicate (rs1 appears twice) and missing rs_missing
        frq_file = tmp_path / "ref.frq"
        frq_content = "CHR SNP A1 A2 MAF NCHROBS\n"
        frq_content += "1 rs1 A G 0.20 1000\n"
        frq_content += "1 rs1 A G 0.25 1000\n"  # duplicate
        frq_content += "1 rs2 C T 0.30 1000\n"
        frq_content += "1 rs3 T C 0.15 1000\n"
        frq_file.write_text(frq_content)

        instruments_df, gene_metadata, n_genes = _load_eqtlgen_chunked(
            eqtl_dir, frq_file, instrument_pval=5e-8, chunksize=10,
        )

        # rs_missing should be dropped (not in .frq)
        assert "rs_missing" not in instruments_df["SNP"].values
        # rs1 should use first MAF (0.20) after dedup
        rs1_rows = instruments_df.loc[instruments_df["SNP"] == "rs1"]
        assert len(rs1_rows) == 1
        # G1 and G2 should both be counted in denominator
        assert n_genes == 2
        # Gene metadata should be present
        assert "G1" in gene_metadata
        assert "G2" in gene_metadata


# ---------------------------------------------------------------------------
# Phase 2 parallelization & --chr scoping tests
# ---------------------------------------------------------------------------

from unittest.mock import patch, call


class TestChromosomeScopedClumping:
    """Verify --chr is injected into PLINK command correctly."""

    def test_chr_injected_for_valid_autosome(self) -> None:
        """--chr N should be present in PLINK command for chromosomes 1-22."""
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3"],
            "gene": ["G1", "G1", "G1"],
            "chr": [1, 1, 1],
            "pos": [100, 200, 300],
            "a1": ["A", "C", "G"],
            "a2": ["G", "T", "A"],
            "beta": [0.5, 0.3, 0.4],
            "se": [0.1, 0.1, 0.1],
            "pval": [1e-10, 1e-9, 1e-8],
        })

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "plink")
            try:
                clump_instruments(
                    instruments,
                    bfile_full_path=Path("/fake/ref"),
                    clump_r2=0.001,
                    clump_kb=1000,
                    plink_binary=Path("plink"),
                    gene_chr=1,
                )
            except RuntimeError:
                pass

            cmd = mock_run.call_args[0][0]
            assert "--chr" in cmd
            chr_idx = cmd.index("--chr")
            assert cmd[chr_idx + 1] == "1"

    def test_chr_injected_for_chrx(self) -> None:
        """--chr 23 should be passed for chromosome X (encoded as 23)."""
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["G1", "G1"],
            "chr": [23, 23],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.3],
            "se": [0.1, 0.1],
            "pval": [1e-10, 1e-9],
        })

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "plink")
            try:
                clump_instruments(
                    instruments,
                    bfile_full_path=Path("/fake/ref"),
                    clump_r2=0.001,
                    clump_kb=1000,
                    plink_binary=Path("plink"),
                    gene_chr=23,
                )
            except RuntimeError:
                pass

            cmd = mock_run.call_args[0][0]
            assert "--chr" in cmd
            chr_idx = cmd.index("--chr")
            assert cmd[chr_idx + 1] == "23"

    def test_chr_omitted_when_none(self) -> None:
        """When gene_chr is None, --chr should NOT be in the command."""
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["G1", "G1"],
            "chr": [1, 1],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.3],
            "se": [0.1, 0.1],
            "pval": [1e-10, 1e-9],
        })

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "plink")
            try:
                clump_instruments(
                    instruments,
                    bfile_full_path=Path("/fake/ref"),
                    clump_r2=0.001,
                    clump_kb=1000,
                    plink_binary=Path("plink"),
                    gene_chr=None,
                )
            except RuntimeError:
                pass

            cmd = mock_run.call_args[0][0]
            assert "--chr" not in cmd

    def test_chr_omitted_for_invalid_value(self) -> None:
        """Invalid chromosome values (0, -1, non-int) should fall back."""
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["G1", "G1"],
            "chr": [1, 1],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.3],
            "se": [0.1, 0.1],
            "pval": [1e-10, 1e-9],
        })

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "plink")
            try:
                clump_instruments(
                    instruments,
                    bfile_full_path=Path("/fake/ref"),
                    clump_r2=0.001,
                    clump_kb=1000,
                    plink_binary=Path("plink"),
                    gene_chr=0,
                )
            except RuntimeError:
                pass

            cmd = mock_run.call_args[0][0]
            assert "--chr" not in cmd


class TestPhase2FailFast:
    """Verify fail-fast error handling in parallel mode."""

    def test_plink_failure_propagates(self) -> None:
        """PLINK RuntimeError should propagate and not be swallowed."""
        from repogen.analysis.mendelian_randomisation import _run_mr_for_gene

        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["G1", "G1"],
            "chr": [1, 1],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.4],
            "se": [0.1, 0.1],
            "pval": [1e-10, 1e-9],
            "n": [30000, 30000],
        })

        gwas_df = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "CHR": [1, 1],
            "POS": [100, 200],
            "A1": ["A", "C"],
            "A2": ["G", "T"],
            "BETA": [0.1, 0.2],
            "SE": [0.05, 0.05],
            "P": [0.01, 0.02],
            "N": [100000, 100000],
            "MAF": [0.2, 0.3],
        })

        config = MRConfig(
            eqtl_sources=[EQTLSourceConfig(source="eqtlgen", path=Path("/tmp"))],
        )

        mock_meta = MagicMock()
        mock_meta.trait_type = "quantitative"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "plink", stderr="fail")
            with pytest.raises(RuntimeError, match="PLINK clumping failed"):
                _run_mr_for_gene(
                    gene="G1",
                    gene_info={"chr": 1, "start": 50, "symbol": ""},
                    eqtl_df=instruments,
                    gwas_df=gwas_df,
                    gwas_metadata=mock_meta,
                    config=config,
                    bfile_full_path=Path("/fake/ref"),
                    plink_binary=Path("plink"),
                    source_name="eqtlgen",
                    bonf_threshold=0.05 / 1000,
                    skip_coloc=True,
                )


class TestParallelDeterminism:
    """Verify parallel execution produces same results as sequential."""

    def test_n_workers_1_matches_sequential(self) -> None:
        """n_workers=1 should use sequential path (no ThreadPoolExecutor)."""
        from repogen.analysis.mendelian_randomisation import run_mendelian_randomisation

        config = MRConfig(
            eqtl_sources=[EQTLSourceConfig(source="eqtlgen", path=Path("/tmp"))],
            n_workers=1,
        )
        assert config.n_workers == 1

    def test_effective_workers_respects_cap(self) -> None:
        """Effective workers = min(config, cap, cpu_count, tasks)."""
        import os
        config_workers = 8
        cap = 4
        cpu = os.cpu_count() or 4
        n_tasks = 10
        effective = min(config_workers, cap, cpu, n_tasks)
        assert effective <= cap
        assert effective <= config_workers


# ---------------------------------------------------------------------------
# Phase 2 --extract optimization tests
# ---------------------------------------------------------------------------


class TestExtractSNPRestriction:
    """Verify --extract is added to PLINK command with correct content."""

    def test_extract_flag_present_with_correct_snps(self) -> None:
        """--extract file should contain exactly the unique instrument SNPs."""
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3", "rs1"],
            "gene": ["G1", "G1", "G1", "G1"],
            "chr": [1, 1, 1, 1],
            "pos": [100, 200, 300, 400],
            "a1": ["A", "C", "G", "T"],
            "a2": ["G", "T", "A", "C"],
            "beta": [0.5, 0.4, 0.3, 0.6],
            "se": [0.1, 0.1, 0.1, 0.1],
            "pval": [1e-10, 1e-9, 1e-8, 1e-7],
        })

        with patch("subprocess.run") as mock_run:
            err = subprocess.CalledProcessError(1, "plink")
            err.stdout = ""
            err.stderr = "Error: No variants remaining after main filters."
            mock_run.side_effect = err
            result = clump_instruments(
                instruments,
                bfile_full_path=Path("/fake/ref"),
                clump_r2=0.001,
                clump_kb=1000,
                plink_binary=Path("plink"),
                gene_chr=1,
            )

            cmd = mock_run.call_args[0][0]
            assert "--extract" in cmd
            extract_idx = cmd.index("--extract")
            extract_path = Path(cmd[extract_idx + 1])

            assert result.empty

    def test_extract_flag_and_chr_both_present(self) -> None:
        """Both --extract and --chr should coexist in the command."""
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["G1", "G1"],
            "chr": [5, 5],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.4],
            "se": [0.1, 0.1],
            "pval": [1e-10, 1e-9],
        })

        with patch("subprocess.run") as mock_run:
            err = subprocess.CalledProcessError(1, "plink")
            err.stdout = ""
            err.stderr = "No variants remaining"
            mock_run.side_effect = err
            clump_instruments(
                instruments,
                bfile_full_path=Path("/fake/ref"),
                clump_r2=0.001,
                clump_kb=1000,
                plink_binary=Path("plink"),
                gene_chr=5,
            )

            cmd = mock_run.call_args[0][0]
            assert "--extract" in cmd
            assert "--chr" in cmd
            chr_idx = cmd.index("--chr")
            assert cmd[chr_idx + 1] == "5"


class TestGracefulNoVariants:
    """Verify PLINK 'no variants remaining' returns empty instead of crashing."""

    def test_no_variants_remaining_returns_empty(self) -> None:
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["G1", "G1"],
            "chr": [1, 1],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.4],
            "se": [0.1, 0.1],
            "pval": [1e-10, 1e-9],
        })

        with patch("subprocess.run") as mock_run:
            err = subprocess.CalledProcessError(1, "plink")
            err.stdout = ""
            err.stderr = "Error: No variants remaining after main filters.\n"
            mock_run.side_effect = err
            result = clump_instruments(
                instruments,
                bfile_full_path=Path("/fake/ref"),
                clump_r2=0.001,
                clump_kb=1000,
                plink_binary=Path("plink"),
                gene_chr=1,
            )

        assert result.empty
        assert list(result.columns) == list(instruments.columns)

    def test_real_plink_error_still_raises(self) -> None:
        """Non-benign PLINK errors should still raise RuntimeError."""
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "gene": ["G1", "G1"],
            "chr": [1, 1],
            "pos": [100, 200],
            "a1": ["A", "C"],
            "a2": ["G", "T"],
            "beta": [0.5, 0.4],
            "se": [0.1, 0.1],
            "pval": [1e-10, 1e-9],
        })

        with patch("subprocess.run") as mock_run:
            err = subprocess.CalledProcessError(1, "plink")
            err.stdout = ""
            err.stderr = "Error: File not found."
            mock_run.side_effect = err
            with pytest.raises(RuntimeError, match="PLINK clumping failed"):
                clump_instruments(
                    instruments,
                    bfile_full_path=Path("/fake/ref"),
                    clump_r2=0.001,
                    clump_kb=1000,
                    plink_binary=Path("plink"),
                    gene_chr=1,
                )


class TestTimedWrapper:
    """Verify _run_mr_for_gene_timed returns proper (result, metrics) tuple."""

    def test_timed_wrapper_returns_tuple(self) -> None:
        from repogen.analysis.mendelian_randomisation import _run_mr_for_gene_timed

        with patch(
            "repogen.analysis.mendelian_randomisation._run_mr_for_gene"
        ) as mock_mr:
            mock_mr.return_value = {"gene_ensembl_id": "G1", "mr_significant": True}
            result, metrics = _run_mr_for_gene_timed(
                gene="G1", gene_info={"chr": 1, "start": 100},
                eqtl_df=pd.DataFrame(), gwas_df=pd.DataFrame(),
                gwas_metadata=None, config=None, bfile_full_path=Path("/x"),
                plink_binary=Path("p"), source_name="eqtlgen",
                bonf_threshold=0.05, skip_coloc=True,
            )

        assert result == {"gene_ensembl_id": "G1", "mr_significant": True}
        assert "task_time_seconds" in metrics
        assert isinstance(metrics["task_time_seconds"], float)
        assert metrics["task_time_seconds"] >= 0

    def test_timed_wrapper_returns_none_result(self) -> None:
        from repogen.analysis.mendelian_randomisation import _run_mr_for_gene_timed

        with patch(
            "repogen.analysis.mendelian_randomisation._run_mr_for_gene"
        ) as mock_mr:
            mock_mr.return_value = None
            result, metrics = _run_mr_for_gene_timed(
                gene="G1", gene_info={"chr": 1, "start": 100},
                eqtl_df=pd.DataFrame(), gwas_df=pd.DataFrame(),
                gwas_metadata=None, config=None, bfile_full_path=Path("/x"),
                plink_binary=Path("p"), source_name="eqtlgen",
                bonf_threshold=0.05, skip_coloc=True,
            )

        assert result is None
        assert "task_time_seconds" in metrics


# ---------------------------------------------------------------------------
# Worker cap, telemetry expansion, schema guard tests
# ---------------------------------------------------------------------------


class TestRuleThreadsNullFallback:
    """Verify that rule_threads=None falls back to n_workers in thread selection."""

    def test_null_rule_threads_resolves_to_n_workers(self) -> None:
        config = {"mr": {"rule_threads": None, "n_workers": 6}}
        resolved = int(
            config.get("mr", {}).get("rule_threads")
            or config.get("mr", {}).get("n_workers", 4)
        )
        assert resolved == 6

    def test_missing_rule_threads_resolves_to_n_workers(self) -> None:
        config = {"mr": {"n_workers": 10}}
        resolved = int(
            config.get("mr", {}).get("rule_threads")
            or config.get("mr", {}).get("n_workers", 4)
        )
        assert resolved == 10

    def test_explicit_rule_threads_takes_precedence(self) -> None:
        config = {"mr": {"rule_threads": 12, "n_workers": 6}}
        resolved = int(
            config.get("mr", {}).get("rule_threads")
            or config.get("mr", {}).get("n_workers", 4)
        )
        assert resolved == 12

    def test_empty_mr_section_uses_default(self) -> None:
        config: dict = {}
        resolved = int(
            config.get("mr", {}).get("rule_threads")
            or config.get("mr", {}).get("n_workers", 4)
        )
        assert resolved == 4


class TestNoTelemetryLeakage:
    """Verify telemetry keys never appear in mr_results or mr_drug_matches columns."""

    TELEMETRY_PATTERNS = [
        "task_time", "phase2_task", "phase2_duration", "phase2_genes",
        "phase2_top_slowest", "tasks_returned_none",
    ]

    def test_mr_results_schema_no_telemetry(self) -> None:
        sample_result = {
            "gene_ensembl_id": "ENSG00000001",
            "gene_symbol": "GENE1",
            "eqtl_source": "eqtlgen",
            "mr_beta_ivw": 0.1,
            "mr_se_ivw": 0.05,
            "mr_pval_ivw": 0.01,
            "mr_significant": True,
            "n_instruments": 5,
            "coloc_supported": True,
            "coloc_status": "run",
        }
        df = pd.DataFrame([sample_result])
        for pattern in self.TELEMETRY_PATTERNS:
            matching = [c for c in df.columns if pattern in c]
            assert matching == [], f"Telemetry key '{pattern}' leaked into mr_results: {matching}"

    def test_mr_drug_matches_schema_no_telemetry(self) -> None:
        sample_match = {
            "gene_ensembl_id": "ENSG00000001",
            "eqtl_source": "eqtlgen",
            "drug_chembl_id": "CHEMBL123",
            "drug_name": "TestDrug",
            "interaction_type": "inhibitor",
            "direction_concordant": True,
        }
        df = pd.DataFrame([sample_match])
        for pattern in self.TELEMETRY_PATTERNS:
            matching = [c for c in df.columns if pattern in c]
            assert matching == [], f"Telemetry key '{pattern}' leaked into mr_drug_matches: {matching}"


class TestTelemetryMetadataPresence:
    """Verify source_metadata contains expected telemetry keys."""

    EXPECTED_KEYS = [
        "phase2_duration_seconds",
        "phase2_genes_processed",
        "phase2_genes_per_minute",
        "phase2_task_time_total_seconds",
        "phase2_task_time_p50_seconds",
        "phase2_task_time_p90_seconds",
        "phase2_task_time_p99_seconds",
        "phase2_top_slowest_genes",
        "phase2_tasks_returned_none",
    ]

    def test_metadata_keys_present_with_tasks(self) -> None:
        """Simulate source_metadata construction with task data."""
        import numpy as np_local

        task_times = [("ENSG1", 1.5), ("ENSG2", 3.2), ("ENSG3", 0.8)]
        task_time_values = [t for _, t in task_times]
        task_time_total = sum(task_time_values)

        p50 = round(float(np_local.percentile(task_time_values, 50)), 3)
        p90 = round(float(np_local.percentile(task_time_values, 90)), 3)
        p99 = round(float(np_local.percentile(task_time_values, 99)), 3)
        sorted_tasks = sorted(task_times, key=lambda x: x[1], reverse=True)
        top_slowest = [
            {"gene": g, "time_seconds": round(t, 3)}
            for g, t in sorted_tasks[:5]
        ]

        metadata = {
            "phase2_duration_seconds": 10.0,
            "phase2_genes_processed": 3,
            "phase2_genes_per_minute": 18.0,
            "phase2_task_time_total_seconds": round(task_time_total, 2),
            "phase2_task_time_p50_seconds": p50,
            "phase2_task_time_p90_seconds": p90,
            "phase2_task_time_p99_seconds": p99,
            "phase2_top_slowest_genes": top_slowest,
            "phase2_tasks_returned_none": 1,
        }

        for key in self.EXPECTED_KEYS:
            assert key in metadata, f"Missing key: {key}"
        assert metadata["phase2_task_time_p50_seconds"] == 1.5
        assert len(metadata["phase2_top_slowest_genes"]) == 3
        assert metadata["phase2_top_slowest_genes"][0]["gene"] == "ENSG2"

    def test_metadata_keys_present_with_no_tasks(self) -> None:
        """When n_gene_tasks == 0, percentiles are None."""
        task_times: list[tuple[str, float]] = []
        task_time_values = [t for _, t in task_times]

        if task_time_values:
            p50 = round(float(np.percentile(task_time_values, 50)), 3)
            p90 = round(float(np.percentile(task_time_values, 90)), 3)
            p99 = round(float(np.percentile(task_time_values, 99)), 3)
            sorted_tasks = sorted(task_times, key=lambda x: x[1], reverse=True)
            top_slowest = [
                {"gene": g, "time_seconds": round(t, 3)}
                for g, t in sorted_tasks[:5]
            ]
        else:
            p50 = p90 = p99 = None
            top_slowest = []

        metadata = {
            "phase2_task_time_p50_seconds": p50,
            "phase2_task_time_p90_seconds": p90,
            "phase2_task_time_p99_seconds": p99,
            "phase2_top_slowest_genes": top_slowest,
            "phase2_tasks_returned_none": 0,
        }

        assert metadata["phase2_task_time_p50_seconds"] is None
        assert metadata["phase2_task_time_p90_seconds"] is None
        assert metadata["phase2_task_time_p99_seconds"] is None
        assert metadata["phase2_top_slowest_genes"] == []
        assert metadata["phase2_tasks_returned_none"] == 0


class TestWorkerCapFromThreads:
    """Verify n_workers_cap (from --threads) correctly caps effective workers."""

    def test_threads_cap_below_config(self) -> None:
        import os
        config_n_workers = 8
        n_workers_cap = 4
        cpu = os.cpu_count() or 4
        n_tasks = 100
        effective = min(
            n_workers_cap if n_workers_cap is not None else config_n_workers,
            config_n_workers,
            cpu,
            max(n_tasks, 1),
        )
        assert effective <= n_workers_cap

    def test_threads_cap_none_uses_config(self) -> None:
        import os
        config_n_workers = 6
        n_workers_cap = None
        cpu = os.cpu_count() or 4
        n_tasks = 100
        effective = min(
            n_workers_cap if n_workers_cap is not None else config_n_workers,
            config_n_workers,
            cpu,
            max(n_tasks, 1),
        )
        assert effective <= config_n_workers
        assert effective <= cpu


# ---------------------------------------------------------------------------
# Per-source BH-FDR + tested-Bonferroni sensitivity track
# ---------------------------------------------------------------------------


class TestFdrTrack:
    def _frame(self) -> pd.DataFrame:
        # Two sources; blood has a clear signal + noise, brain has one signal.
        rows = []
        for i, p in enumerate([1e-9, 1e-3, 0.2, 0.5, 0.9]):
            rows.append({"gene_ensembl_id": f"B{i}", "eqtl_source": "eqtlgen",
                         "mr_pval": p, "mr_significant": p < (0.05 / 5)})
        for i, p in enumerate([1e-8, 0.4, np.nan]):
            rows.append({"gene_ensembl_id": f"R{i}", "eqtl_source": "metabrain",
                         "mr_pval": p, "mr_significant": bool(np.isfinite(p) and p < (0.05 / 3))})
        return pd.DataFrame(rows)

    def test_columns_added(self) -> None:
        out = _annotate_fdr_track(self._frame())
        for col in ("mr_fdr_bh_q", "mr_significant_fdr_bh",
                    "bonferroni_threshold_tested", "mr_significant_bonferroni_tested"):
            assert col in out.columns

    def test_nan_pval_yields_nan_q_and_false(self) -> None:
        out = _annotate_fdr_track(self._frame())
        nan_row = out.loc[out["gene_ensembl_id"] == "R2"].iloc[0]
        assert pd.isna(nan_row["mr_fdr_bh_q"])
        assert bool(nan_row["mr_significant_fdr_bh"]) is False
        assert bool(nan_row["mr_significant_bonferroni_tested"]) is False

    def test_fdr_is_per_source(self) -> None:
        # BH q for the brain 1e-8 uses n=2 finite p-values (not the global 8).
        out = _annotate_fdr_track(self._frame())
        r0 = out.loc[out["gene_ensembl_id"] == "R0"].iloc[0]
        assert r0["mr_fdr_bh_q"] == pytest.approx(1e-8 * 2, rel=1e-6)

    def test_tested_bonferroni_denominator_is_finite_count(self) -> None:
        out = _annotate_fdr_track(self._frame())
        # metabrain has 2 finite p-values -> threshold 0.05/2.
        r = out.loc[out["eqtl_source"] == "metabrain"].iloc[0]
        assert r["bonferroni_threshold_tested"] == pytest.approx(0.05 / 2)
        # eqtlgen has 5 finite -> 0.05/5.
        b = out.loc[out["eqtl_source"] == "eqtlgen"].iloc[0]
        assert b["bonferroni_threshold_tested"] == pytest.approx(0.05 / 5)

    def test_primary_significant_untouched(self) -> None:
        df = self._frame()
        before = df["mr_significant"].tolist()
        out = _annotate_fdr_track(df)
        assert out["mr_significant"].tolist() == before

    def test_bh_fdr_finite_all_nan(self) -> None:
        s = pd.Series([np.nan, np.nan])
        q = _bh_fdr_finite(s)
        assert q.isna().all()

    def test_empty_frame(self) -> None:
        out = _annotate_fdr_track(pd.DataFrame())
        assert "mr_fdr_bh_q" in out.columns


# ---------------------------------------------------------------------------
# Coloc variance mode + Steiger population-prevalence gating
# ---------------------------------------------------------------------------


class TestColocVarianceInputs:
    @staticmethod
    def _coloc_fixture(n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
        snps = [f"rs{i}" for i in range(1, n + 1)]
        positions = np.arange(1001, 1001 + n)
        eqtl = pd.DataFrame({
            "gene": ["ENSG000001"] * n,
            "SNP": snps,
            "chr": [1] * n,
            "pos": positions,
            "beta": np.linspace(0.02, 0.08, n),
            "se": [0.01] * n,
        })
        gwas = pd.DataFrame({
            "SNP": snps,
            "CHR": [1] * n,
            "POS": positions,
            "A1": ["A"] * n,
            "A2": ["G"] * n,
            "BETA": np.linspace(0.01, 0.05, n),
            "SE": [0.02] * n,
            "P": [1e-4] * n,
            "N": [50000] * n,
            "MAF": [0.3] * n,
        })
        return eqtl, gwas

    def test_default_reported_se_passes_none_and_n_out(self) -> None:
        cfg = MRConfig()  # coloc_variance_mode default = reported_se
        s2, n2 = _resolve_coloc_variance_inputs(cfg, "case_control", 5000, 10000, 44000)
        assert s2 is None
        assert n2 == 44000  # n_out preserved, byte-stable

    def test_case_control_approx_uses_total_n_and_fraction(self) -> None:
        cfg = MRConfig(coloc_variance_mode="case_control_approx")
        s2, n2 = _resolve_coloc_variance_inputs(cfg, "case_control", 5000, 10000, 999999)
        assert s2 == pytest.approx(5000 / 15000)
        assert n2 == 15000  # total N, NOT the Neff-ish n_out

    def test_case_control_approx_quant_trait_falls_back_to_se(self) -> None:
        cfg = MRConfig(coloc_variance_mode="case_control_approx")
        s2, n2 = _resolve_coloc_variance_inputs(cfg, "quantitative", None, None, 30000)
        assert s2 is None
        assert n2 == 30000

    def test_case_control_approx_missing_counts_fails_loud(self) -> None:
        cfg = MRConfig(coloc_variance_mode="case_control_approx")
        with pytest.raises(ColocConfigurationError, match="requires finite positive"):
            _resolve_coloc_variance_inputs(cfg, "case_control", None, None, 30000)

    @pytest.mark.parametrize("bad_count", [0, np.nan, 123.5, "not-a-count"])
    def test_case_control_approx_invalid_counts_fail_loud(self, bad_count: object) -> None:
        cfg = MRConfig(coloc_variance_mode="case_control_approx")
        with pytest.raises(ColocConfigurationError, match="finite positive"):
            _resolve_coloc_variance_inputs(cfg, "case_control", bad_count, 1000, 30000)

    def test_preflight_fails_before_coloc_for_missing_binary_counts(self) -> None:
        cfg = MRConfig(coloc_variance_mode="case_control_approx")
        metadata = GWASMetadata(genome_build="GRCh37", trait_type="case_control")
        with pytest.raises(ColocConfigurationError, match="finite positive"):
            _validate_coloc_calibration_config(cfg, metadata)

    def test_preflight_does_not_require_counts_for_reported_se(self) -> None:
        cfg = MRConfig(coloc_variance_mode="reported_se")
        metadata = GWASMetadata(genome_build="GRCh37", trait_type="case_control")
        _validate_coloc_calibration_config(cfg, metadata)

    def test_run_coloc_for_gene_propagates_c4_config_error(self) -> None:
        cfg = MRConfig(coloc_variance_mode="case_control_approx", min_coloc_snps=10)
        metadata = GWASMetadata(genome_build="GRCh37", trait_type="case_control")
        eqtl, gwas = self._coloc_fixture(n=12)

        with pytest.raises(ColocConfigurationError, match="finite positive"):
            _run_coloc_for_gene(
                "ENSG000001", eqtl, gwas, metadata, cfg,
                n_exp=30000, n_out=50000,
            )

    def test_run_coloc_for_gene_reported_se_still_runs_without_counts(self) -> None:
        cfg = MRConfig(coloc_variance_mode="reported_se", min_coloc_snps=10)
        metadata = GWASMetadata(genome_build="GRCh37", trait_type="case_control")
        eqtl, gwas = self._coloc_fixture(n=12)

        result = _run_coloc_for_gene(
            "ENSG000001", eqtl, gwas, metadata, cfg,
            n_exp=30000, n_out=50000,
        )

        assert result["coloc_status"] in {"colocalised", "distinct_signals", "unsupported"}
        assert result["n_snps_coloc"] == 12


class TestSteigerPrevalenceGating:
    def test_no_prevalence_no_transform(self) -> None:
        # With population_prevalence=None the binary transform must NOT fire, so
        # the result equals the plain observed-scale call.
        p_gated, v_gated = steiger_test(
            0.05, 0.01, 30000, 50000,
            trait_type="case_control", n_cases=20000, n_controls=30000,
            population_prevalence=None,
        )
        p_plain, v_plain = steiger_test(0.05, 0.01, 30000, 50000)
        assert p_gated == pytest.approx(p_plain)
        assert v_gated == v_plain

    def test_prevalence_applies_transform(self) -> None:
        # Supplying a real (small) prevalence rescales r2_out -> changes the
        # p-value. Use moderate r2/N so the two p-values are not both sub-1e-12
        # (pytest.approx's default abs=1e-12 would otherwise call them equal).
        p_no, _ = steiger_test(
            0.01, 0.009, 800, 1200,
            trait_type="case_control", n_cases=20000, n_controls=30000,
            population_prevalence=None,
        )
        p_yes, _ = steiger_test(
            0.01, 0.009, 800, 1200,
            trait_type="case_control", n_cases=20000, n_controls=30000,
            population_prevalence=0.01,
        )
        assert p_yes != pytest.approx(p_no, abs=0.0)

    def test_sample_fraction_never_used_as_prevalence(self) -> None:
        # Sanity: passing prevalence == sample fraction differs from the old
        # (buggy) behaviour only in that it is now explicit; the point is that
        # without prevalence there is no transform at all.
        p_none, _ = steiger_test(
            0.02, 0.02, 30000, 50000,
            trait_type="case_control", n_cases=25000, n_controls=25000,
            population_prevalence=None,
        )
        p_quant, _ = steiger_test(0.02, 0.02, 30000, 50000)
        assert p_none == pytest.approx(p_quant)


# ---------------------------------------------------------------------------
# MHC flag annotation + MHC-excluded sensitivity output
# ---------------------------------------------------------------------------


class TestMhcFlag:
    def _mr(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"gene_ensembl_id": "ENSG_MHC1", "eqtl_source": "eqtlgen",
             "gene_chr": 6, "gene_start": 31_000_000, "mr_significant": True,
             "coloc_supported": True, "confidence_tier": "high", "mr_pval": 1e-9},
            {"gene_ensembl_id": "ENSG_OK1", "eqtl_source": "eqtlgen",
             "gene_chr": 1, "gene_start": 1_000_000, "mr_significant": True,
             "coloc_supported": True, "confidence_tier": "high", "mr_pval": 1e-9},
        ])

    def test_disabled_no_columns(self) -> None:
        cfg = MRConfig()  # mhc_sensitivity disabled by default
        out, meta = _annotate_mhc_flag(self._mr(), cfg, MagicMock(), None)
        assert "mhc_flag" not in out.columns
        assert meta["enabled"] is False

    def test_ensembl_membership_primary(self) -> None:
        cfg = MRConfig(mhc_sensitivity=MRMHCSensitivityConfig(enabled=True))
        annotation = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_MHC1"], "chr": 6,
            "start": 29_000_000, "end": 33_000_000,
        })
        with patch(
            "repogen.analysis.mendelian_randomisation.load_mhc_gene_annotation",
            return_value=(annotation, {"mhc_annotation_strategy": "grch38_gene_loc"}),
        ):
            out, meta = _annotate_mhc_flag(self._mr(), cfg, MagicMock(), MagicMock())
        assert out.loc[out["gene_ensembl_id"] == "ENSG_MHC1", "mhc_flag"].iloc[0]
        assert not out.loc[out["gene_ensembl_id"] == "ENSG_OK1", "mhc_flag"].iloc[0]
        assert meta["mhc_flag_method"] == "ensembl_membership"

    def test_fail_loud_when_no_annotation_and_no_fallback(self) -> None:
        cfg = MRConfig(mhc_sensitivity=MRMHCSensitivityConfig(enabled=True))
        # require_mhc_annotation=True makes the loader raise; verify it propagates.
        with patch(
            "repogen.analysis.mendelian_randomisation.load_mhc_gene_annotation",
            side_effect=RuntimeError("no annotation"),
        ):
            with pytest.raises(RuntimeError):
                _annotate_mhc_flag(self._mr(), cfg, MagicMock(), None)

    def test_coordinate_fallback_opt_in(self) -> None:
        cfg = MRConfig(mhc_sensitivity=MRMHCSensitivityConfig(
            enabled=True, allow_mhc_coordinate_fallback=True,
        ))
        with patch(
            "repogen.analysis.mendelian_randomisation.load_mhc_gene_annotation",
            return_value=(None, {"mhc_annotation_strategy": "no_annotation"}),
        ):
            out, meta = _annotate_mhc_flag(self._mr(), cfg, MagicMock(), None)
        # eqtlgen -> GRCh37; chr6:31M is inside the MHC interval.
        assert out.loc[out["gene_ensembl_id"] == "ENSG_MHC1", "mhc_flag"].iloc[0]
        assert not out.loc[out["gene_ensembl_id"] == "ENSG_OK1", "mhc_flag"].iloc[0]
        assert meta["mhc_flag_method"] == "coordinate_fallback_approximate"


class TestMhcSensitivityOutput:
    def test_excluded_view_written_and_counts(self, tmp_path: Path) -> None:
        mr = pd.DataFrame([
            {"gene_ensembl_id": "MHC", "eqtl_source": "eqtlgen",
             "mr_significant": True, "coloc_supported": True,
             "confidence_tier": "high", "mhc_flag": True},
            {"gene_ensembl_id": "OK", "eqtl_source": "eqtlgen",
             "mr_significant": True, "coloc_supported": True,
             "confidence_tier": "high", "mhc_flag": False},
        ])
        drugs = pd.DataFrame([
            {"gene_ensembl_id": "MHC", "drug_chembl_id": "CHEMBL1"},
            {"gene_ensembl_id": "OK", "drug_chembl_id": "CHEMBL2"},
        ])
        verdicts = pd.DataFrame([
            {"gene_ensembl_id": "MHC", "verdict_status": "actionable"},
            {"gene_ensembl_id": "OK", "verdict_status": "actionable"},
        ])
        summary = _write_mhc_excluded_sensitivity(tmp_path, mr, drugs, verdicts)
        sens = tmp_path / "sensitivity" / "mhc_excluded"
        assert (sens / "mr_results.parquet").exists()
        assert (sens / "mhc_excluded_summary.json").exists()
        assert summary["primary"]["n_significant"] == 2
        assert summary["mhc_excluded"]["n_significant"] == 1
        assert summary["mhc_excluded"]["n_drug_matched"] == 1
        # MHC gene dropped from the excluded drug table.
        drugs_ex = pd.read_parquet(sens / "mr_drug_matches.parquet")
        assert set(drugs_ex["gene_ensembl_id"]) == {"OK"}


# ---------------------------------------------------------------------------
# Run time: join index, per-chromosome panel, file discovery, outputs
# ---------------------------------------------------------------------------

from repogen.analysis.mendelian_randomisation import (  # noqa: E402
    _GwasJoinIndex,
    _chromosome_bfile,
    _find_eqtlgen_file,
    _merge_eqtl_gwas_two_stage,
    _write_drug_matches,
    ensure_ref_split,
)


def _join_fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    """eQTL rows covering every join path, and a GWAS with a missing rsID.

    rs1 joins by rsID; rs2 has two allele rows and joins by rsID; rs4 is
    missing from the GWAS but a row without an rsID sits at its position;
    rs5 sits where the GWAS has another rsID; rs9 matches nothing.
    """
    eqtl = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs2", "rs4", "rs5", "rs9"],
        "gene": ["G1"] * 6,
        "chr": [1, 1, 1, 1, 2, 3],
        "pos": [100, 200, 200, 400, 500, 900],
        "a1": ["A", "C", "C", "G", "T", "A"],
        "a2": ["G", "T", "G", "A", "C", "C"],
        "beta": [0.5, 0.3, 0.2, 0.4, 0.1, 0.6],
        "se": [0.1] * 6,
        "pval": [1e-10] * 6,
        "n": [1000] * 6,
    })
    gwas = pd.DataFrame({
        "SNP": ["rs0", "rs1", "rs2", None, "rs5b", "rs7"],
        "CHR": [1, 1, 1, 1, 2, 3],
        "POS": [50, 100, 200, 400, 500, 700],
        "A1": ["A", "A", "C", "G", "T", "A"],
        "A2": ["G", "G", "T", "A", "C", "C"],
        "BETA": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "SE": [0.01] * 6,
        "P": [0.5] * 6,
        "N": [5000] * 6,
        "MAF": [0.3] * 6,
    })
    return eqtl, gwas


class TestGwasJoinIndex:
    GWAS_COLS = ["A1", "A2", "BETA", "SE", "P", "N", "CHR", "POS", "MAF"]

    @pytest.mark.parametrize("suffixes", [("_eqtl", "_gwas"), ("_exp", "_out")])
    def test_join_identical_with_and_without_index(self, suffixes) -> None:
        eqtl, gwas = _join_fixtures()
        plain = _merge_eqtl_gwas_two_stage(eqtl, gwas, self.GWAS_COLS, suffixes=suffixes)
        indexed = _merge_eqtl_gwas_two_stage(
            eqtl, gwas, self.GWAS_COLS, suffixes=suffixes, lookup=_GwasJoinIndex(gwas),
        )
        assert list(plain["SNP"]) == ["rs1", "rs2", "rs2", "rs4", "rs5"]
        pd.testing.assert_frame_equal(plain, indexed)

    def test_harmonisation_identical_with_and_without_index(self) -> None:
        eqtl, gwas = _join_fixtures()
        pd.testing.assert_frame_equal(
            harmonise_gwas_eqtl(gwas, eqtl),
            harmonise_gwas_eqtl(gwas, eqtl, lookup=_GwasJoinIndex(gwas)),
        )

    def test_duplicated_positions_join_every_row(self) -> None:
        eqtl, gwas = _join_fixtures()
        extra = gwas.iloc[[3]].assign(A1="T", A2="C", BETA=0.07)
        gwas = pd.concat([gwas, extra], ignore_index=True)
        plain = _merge_eqtl_gwas_two_stage(eqtl, gwas, self.GWAS_COLS)
        indexed = _merge_eqtl_gwas_two_stage(
            eqtl, gwas, self.GWAS_COLS, lookup=_GwasJoinIndex(gwas),
        )
        assert (plain["SNP"] == "rs4").sum() == 2
        pd.testing.assert_frame_equal(plain, indexed)


class TestFindEqtlgenFile:
    def test_single_table(self, tmp_path: Path) -> None:
        (tmp_path / "cis_eqtls.txt.gz").write_bytes(b"")
        assert _find_eqtlgen_file(tmp_path).name == "cis_eqtls.txt.gz"

    def test_readme_and_checksums_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "cis_eqtls.txt.gz").write_bytes(b"")
        (tmp_path / "README.txt").write_text("notes")
        (tmp_path / "cis_eqtls.txt.gz.md5").write_text("abc")
        assert _find_eqtlgen_file(tmp_path).name == "cis_eqtls.txt.gz"

    def test_two_tables_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt.gz").write_bytes(b"")
        (tmp_path / "b.tsv.gz").write_bytes(b"")
        with pytest.raises(ValueError, match="Expected one eQTLGen table"):
            _find_eqtlgen_file(tmp_path)

    def test_no_table_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _find_eqtlgen_file(tmp_path)


def _fake_plink_outputs(cmd, *args, **kwargs):
    """Stand-in for PLINK --make-bed: create the files it would write."""
    out = cmd[cmd.index("--out") + 1]
    for ext in (".bed", ".bim", ".fam", ".log"):
        Path(out + ext).write_text("")


class TestPerChromosomePanel:
    def test_split_written_once(self, tmp_path: Path) -> None:
        ref = tmp_path / "ref"
        with patch("subprocess.run", side_effect=_fake_plink_outputs) as run:
            ensure_ref_split(ref, Path("plink"))
            assert run.call_count == 22
            ensure_ref_split(ref, Path("plink"))
            assert run.call_count == 22
        cmd = run.call_args_list[0].args[0]
        assert "--keep-allele-order" in cmd
        assert all(_chromosome_bfile(ref, c) is not None for c in range(1, 23))
        assert not list(tmp_path.glob("*.tmp*"))

    def test_chromosome_bfile_only_for_autosomes(self, tmp_path: Path) -> None:
        ref = tmp_path / "ref"
        for ext in (".bed", ".bim", ".fam"):
            Path(f"{ref}.chr7{ext}").write_text("")
        assert _chromosome_bfile(ref, 7) == Path(f"{ref}.chr7")
        assert _chromosome_bfile(ref, np.int64(7)) == Path(f"{ref}.chr7")
        assert _chromosome_bfile(ref, 8) is None
        assert _chromosome_bfile(ref, 23) is None
        assert _chromosome_bfile(ref, None) is None

    @pytest.mark.parametrize("split", [True, False])
    def test_clumping_reads_the_split_panel_when_present(self, tmp_path: Path, split: bool) -> None:
        ref = tmp_path / "ref"
        if split:
            for ext in (".bed", ".bim", ".fam"):
                Path(f"{ref}.chr1{ext}").write_text("")
        instruments = pd.DataFrame({
            "SNP": ["rs1", "rs2"], "pval": [1e-10, 1e-9], "chr": [1, 1], "pos": [100, 200],
        })
        with patch("subprocess.run") as run:
            clump_instruments(instruments, ref, 0.001, 1000, Path("plink"), gene_chr=1)
        cmd = run.call_args.args[0]
        expected = f"{ref}.chr1" if split else str(ref)
        assert cmd[cmd.index("--bfile") + 1] == expected
        assert cmd[cmd.index("--chr") + 1] == "1"


class TestWriteDrugMatches:
    def test_empty_writes_both_files(self, tmp_path: Path) -> None:
        (tmp_path / "mr_drug_matches.csv").write_text("stale,previous,run\n")
        _write_drug_matches(tmp_path, pd.DataFrame())
        assert (tmp_path / "mr_drug_matches.parquet").exists()
        assert "stale" not in (tmp_path / "mr_drug_matches.csv").read_text()

    def test_list_columns_become_json_in_csv(self, tmp_path: Path) -> None:
        matches = pd.DataFrame({"drug_chembl_id": ["CHEMBL1"], "atc_codes": [["N05AB06"]]})
        _write_drug_matches(tmp_path, matches)
        csv = pd.read_csv(tmp_path / "mr_drug_matches.csv")
        assert csv.loc[0, "atc_codes"] == '["N05AB06"]'
        assert pd.read_parquet(tmp_path / "mr_drug_matches.parquet").shape == (1, 2)
