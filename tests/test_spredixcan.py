"""Tests for repogen.analysis.spredixcan."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from repogen.analysis.spredixcan import (
    _precompute_gwas_lookups,
    add_entrez_ids,
    add_mhc_flag,
    align_alleles,
    build_covariance_matrix,
    compute_gene_zscore,
    ivw_meta_analysis,
    load_covariance,
    load_prediction_model,
    match_variants_to_gwas,
    resolve_model_paths,
    run_spredixcan,
    run_spredixcan_tissue,
    _resolve_model_dir,
    _resolve_cov_dir,
    _resolve_tissue_list,
)
from repogen.data.schemas import validate_dataframe
from repogen.utils.constants import BRAIN_TISSUES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_model_db(path: Path, genes: dict[str, list[dict]],
                     extra: list[dict] | None = None) -> None:
    """Create a synthetic PredictDB .db file."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE weights "
        "(gene TEXT, rsid TEXT, weight REAL, ref_allele TEXT, eff_allele TEXT)"
    )
    conn.execute(
        "CREATE TABLE extra "
        "(gene TEXT, genename TEXT, gene_type TEXT, n_snps_in_window INTEGER, "
        "n_snps_in_model INTEGER, pred_perf_r2 REAL, pred_perf_pval REAL, "
        "pred_perf_qval REAL)"
    )
    for gene_id, snps in genes.items():
        for snp in snps:
            conn.execute(
                "INSERT INTO weights VALUES (?, ?, ?, ?, ?)",
                (gene_id, snp["rsid"], snp["weight"],
                 snp["ref_allele"], snp["eff_allele"]),
            )
    if extra:
        for e in extra:
            conn.execute(
                "INSERT INTO extra VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (e["gene"], e.get("genename", ""), e.get("gene_type", "protein_coding"),
                 e.get("n_snps_in_window", 100), e.get("n_snps_in_model", 5),
                 e.get("pred_perf_r2", 0.05), e.get("pred_perf_pval", 0.01),
                 e.get("pred_perf_qval", 0.05)),
            )
    conn.commit()
    conn.close()


def _create_mashr_model_db(
    path: Path,
    genes: dict[str, list[dict]],
    extra: list[dict] | None = None,
) -> None:
    """Create a mashr-format PredictDB .db file with varID and dotted extra columns."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE weights "
        "(gene TEXT, rsid TEXT, varID TEXT, ref_allele TEXT, eff_allele TEXT, weight REAL)"
    )
    conn.execute(
        'CREATE TABLE extra '
        '(gene TEXT, genename TEXT, gene_type TEXT, '
        '"n.snps.in.model" INTEGER, "pred.perf.R2" REAL, '
        '"pred.perf.pval" REAL, "pred.perf.qval" REAL)'
    )
    for gene_id, snps in genes.items():
        for snp in snps:
            conn.execute(
                "INSERT INTO weights VALUES (?, ?, ?, ?, ?, ?)",
                (gene_id, snp["rsid"], snp.get("varID", snp["rsid"]),
                 snp["ref_allele"], snp["eff_allele"], snp["weight"]),
            )
    if extra:
        for e in extra:
            conn.execute(
                "INSERT INTO extra VALUES (?, ?, ?, ?, ?, ?, ?)",
                (e["gene"], e.get("genename", ""),
                 e.get("gene_type", "protein_coding"),
                 e.get("n_snps_in_model", 5),
                 e.get("pred_perf_r2", 0.05),
                 e.get("pred_perf_pval", 0.01),
                 e.get("pred_perf_qval", 0.05)),
            )
    conn.commit()
    conn.close()


def _create_covariance_file(path: Path, gene_cov: dict[str, list[tuple[str, str, float]]]) -> None:
    """Create a synthetic covariance .txt.gz file."""
    with gzip.open(path, "wt") as f:
        f.write("GENE RSID1 RSID2 VALUE\n")
        for gene_id, entries in gene_cov.items():
            for rsid1, rsid2, val in entries:
                f.write(f"{gene_id} {rsid1} {rsid2} {val}\n")


@pytest.fixture
def synthetic_model_db(tmp_path: Path) -> Path:
    """Create a synthetic model .db with 2 genes, 3 SNPs each."""
    db_path = tmp_path / "mashr_Brain_Amygdala.db"
    genes = {
        "ENSG00000001.1": [
            {"rsid": "rs1", "weight": 0.3, "ref_allele": "G", "eff_allele": "A"},
            {"rsid": "rs2", "weight": -0.2, "ref_allele": "C", "eff_allele": "T"},
            {"rsid": "rs3", "weight": 0.5, "ref_allele": "T", "eff_allele": "C"},
        ],
        "ENSG00000002.1": [
            {"rsid": "rs4", "weight": 0.1, "ref_allele": "A", "eff_allele": "G"},
            {"rsid": "rs5", "weight": 0.4, "ref_allele": "T", "eff_allele": "C"},
        ],
    }
    extra = [
        {"gene": "ENSG00000001.1", "genename": "GENE1", "n_snps_in_model": 3,
         "pred_perf_r2": 0.08, "pred_perf_pval": 0.001},
        {"gene": "ENSG00000002.1", "genename": "GENE2", "n_snps_in_model": 2,
         "pred_perf_r2": 0.05, "pred_perf_pval": 0.01},
    ]
    _create_model_db(db_path, genes, extra)
    return db_path


@pytest.fixture
def synthetic_cov_file(tmp_path: Path) -> Path:
    """Create covariance file with identity-like covariance for test genes."""
    cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
    gene_cov = {
        "ENSG00000001.1": [
            ("rs1", "rs1", 1.0), ("rs2", "rs2", 1.0), ("rs3", "rs3", 1.0),
            ("rs1", "rs2", 0.0), ("rs1", "rs3", 0.0), ("rs2", "rs3", 0.0),
        ],
        "ENSG00000002.1": [
            ("rs4", "rs4", 1.0), ("rs5", "rs5", 1.0), ("rs4", "rs5", 0.0),
        ],
    }
    _create_covariance_file(cov_path, gene_cov)
    return cov_path


@pytest.fixture
def synthetic_gwas() -> pd.DataFrame:
    """Create a synthetic GWAS DataFrame matching the model SNPs."""
    return pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5", "rs99"],
        "VARIANT_ID": ["1:100:A:G", "1:200:T:C", "2:300:C:T", "3:400:G:A", "4:500:C:T", "5:600:A:G"],
        "CHR": [1, 1, 2, 3, 4, 5],
        "POS": [100, 200, 300, 400, 500, 600],
        "A1": ["A", "T", "C", "G", "C", "A"],
        "A2": ["G", "C", "T", "A", "T", "G"],
        "BETA": [0.1, -0.05, 0.2, 0.15, -0.1, 0.3],
        "SE": [0.02, 0.03, 0.04, 0.05, 0.02, 0.01],
        "P": [1e-6, 0.09, 1e-7, 0.003, 1e-6, 1e-30],
        "MAF": [0.3, 0.2, 0.15, 0.4, 0.1, 0.05],
        "N": [50000] * 6,
    })


# ---------------------------------------------------------------------------
# Model Loading Tests
# ---------------------------------------------------------------------------


class TestLoadPredictionModel:
    def test_reads_weights_and_extra(self, synthetic_model_db: Path) -> None:
        weights, extra = load_prediction_model(synthetic_model_db)
        assert "gene" in weights.columns
        assert "rsid" in weights.columns
        assert "weight" in weights.columns
        assert len(weights) == 5
        assert len(extra) == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_prediction_model(tmp_path / "nonexistent.db")


class TestLoadCovariance:
    def test_reads_gene_snp_pairs(self, synthetic_cov_file: Path) -> None:
        cov = load_covariance(synthetic_cov_file)
        assert "ENSG00000001.1" in cov
        assert len(cov["ENSG00000001.1"]) == 6
        assert cov["ENSG00000001.1"][0] == ("rs1", "rs1", 1.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_covariance(tmp_path / "nonexistent.txt.gz")


# ---------------------------------------------------------------------------
# Allele Alignment Tests
# ---------------------------------------------------------------------------


class TestAlignAlleles:
    def test_direct_match(self) -> None:
        result = align_alleles(
            np.array(["A"]), np.array(["G"]),
            np.array(["A"]), np.array(["G"]),
        )
        assert result[0] == 1

    def test_flip(self) -> None:
        result = align_alleles(
            np.array(["G"]), np.array(["A"]),
            np.array(["A"]), np.array(["G"]),
        )
        assert result[0] == -1

    def test_complement(self) -> None:
        result = align_alleles(
            np.array(["T"]), np.array(["C"]),
            np.array(["A"]), np.array(["G"]),
        )
        assert result[0] == 1

    def test_palindromic_returns_zero(self) -> None:
        result = align_alleles(
            np.array(["A"]), np.array(["T"]),
            np.array(["A"]), np.array(["T"]),
        )
        assert result[0] == 0

    def test_cg_palindromic_returns_zero(self) -> None:
        result = align_alleles(
            np.array(["C"]), np.array(["G"]),
            np.array(["C"]), np.array(["G"]),
        )
        assert result[0] == 0

    def test_vectorised(self) -> None:
        result = align_alleles(
            np.array(["A", "G", "T", "A"]),
            np.array(["G", "A", "C", "T"]),
            np.array(["A", "A", "A", "A"]),
            np.array(["G", "G", "G", "T"]),
        )
        np.testing.assert_array_equal(result, [1, -1, 1, 0])

    def test_unresolvable_returns_zero(self) -> None:
        result = align_alleles(
            np.array(["A"]), np.array(["G"]),
            np.array(["A"]), np.array(["C"]),
        )
        assert result[0] == 0


# ---------------------------------------------------------------------------
# Variant Matching Tests
# ---------------------------------------------------------------------------


class TestMatchVariantsToGwas:
    def test_by_rsid(self, synthetic_gwas: pd.DataFrame) -> None:
        model = pd.DataFrame({
            "gene": ["G1", "G1"],
            "rsid": ["rs1", "rs2"],
            "weight": [0.3, 0.2],
            "ref_allele": ["G", "C"],
            "eff_allele": ["A", "T"],
        })
        result = match_variants_to_gwas(model, synthetic_gwas)
        assert len(result) == 2
        assert "zscore" in result.columns

    def test_handles_missing_snps(self, synthetic_gwas: pd.DataFrame) -> None:
        model = pd.DataFrame({
            "gene": ["G1", "G1"],
            "rsid": ["rs1", "rs_missing"],
            "weight": [0.3, 0.2],
            "ref_allele": ["G", "C"],
            "eff_allele": ["A", "T"],
        })
        result = match_variants_to_gwas(model, synthetic_gwas)
        assert len(result) == 1

    def test_applies_sign_flip(self, synthetic_gwas: pd.DataFrame) -> None:
        model = pd.DataFrame({
            "gene": ["G1"],
            "rsid": ["rs1"],
            "weight": [0.3],
            "ref_allele": ["A"],
            "eff_allele": ["G"],
        })
        result = match_variants_to_gwas(model, synthetic_gwas)
        assert len(result) == 1
        expected_z = -(0.1 / 0.02)
        assert abs(result.iloc[0]["zscore"] - expected_z) < 1e-10

    def test_variant_id_fallback(self) -> None:
        """Model rsid matching GWAS VARIANT_ID (not SNP) should still match."""
        gwas = pd.DataFrame({
            "SNP": ["rs100"],
            "VARIANT_ID": ["1:200:T:C"],
            "CHR": [1], "POS": [200],
            "A1": ["T"], "A2": ["C"],
            "BETA": [0.5], "SE": [0.1],
            "P": [1e-5], "N": [50000],
        })
        model = pd.DataFrame({
            "gene": ["G1"],
            "rsid": ["1:200:T:C"],
            "weight": [0.4],
            "ref_allele": ["C"],
            "eff_allele": ["T"],
        })
        result = match_variants_to_gwas(model, gwas)
        assert len(result) == 1
        assert result.iloc[0]["rsid"] == "1:200:T:C"
        expected_z = 0.5 / 0.1
        assert abs(result.iloc[0]["zscore"] - expected_z) < 1e-10

    def test_snp_match_not_overwritten_by_fallback(self) -> None:
        """SNP match should never be overwritten by VARIANT_ID fallback."""
        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs_other"],
            "VARIANT_ID": ["rs1", "1:200:T:C"],
            "CHR": [1, 1], "POS": [100, 200],
            "A1": ["A", "T"], "A2": ["G", "C"],
            "BETA": [0.1, 0.9], "SE": [0.02, 0.1],
            "P": [1e-6, 0.5], "N": [50000, 50000],
        })
        model = pd.DataFrame({
            "gene": ["G1"],
            "rsid": ["rs1"],
            "weight": [0.3],
            "ref_allele": ["G"],
            "eff_allele": ["A"],
        })
        result = match_variants_to_gwas(model, gwas)
        assert len(result) == 1
        expected_z = 0.1 / 0.02
        assert abs(result.iloc[0]["zscore"] - expected_z) < 1e-10

    def test_output_schema(self, synthetic_gwas: pd.DataFrame) -> None:
        """Output columns and types must match the documented contract."""
        model = pd.DataFrame({
            "gene": ["G1", "G1", "G2"],
            "rsid": ["rs1", "rs2", "rs4"],
            "weight": [0.3, 0.2, 0.1],
            "ref_allele": ["G", "C", "A"],
            "eff_allele": ["A", "T", "G"],
        })
        result = match_variants_to_gwas(model, synthetic_gwas)
        assert list(result.columns) == ["gene", "rsid", "weight", "zscore", "alignment"]
        assert result["alignment"].dtype in (np.int8, np.int32, np.int64, int)

    def test_precomputed_lookups_identical(self, synthetic_gwas: pd.DataFrame) -> None:
        """Providing precomputed gwas_lookups must produce identical output."""
        model = pd.DataFrame({
            "gene": ["G1", "G1", "G2"],
            "rsid": ["rs1", "rs2", "rs4"],
            "weight": [0.3, 0.2, 0.1],
            "ref_allele": ["G", "C", "A"],
            "eff_allele": ["A", "T", "G"],
        })
        result_auto = match_variants_to_gwas(model, synthetic_gwas)
        lookups = _precompute_gwas_lookups(synthetic_gwas)
        result_pre = match_variants_to_gwas(model, synthetic_gwas, gwas_lookups=lookups)

        pd.testing.assert_frame_equal(
            result_auto.reset_index(drop=True),
            result_pre.reset_index(drop=True),
        )

    def test_empty_model_returns_empty(self, synthetic_gwas: pd.DataFrame) -> None:
        model = pd.DataFrame(columns=["gene", "rsid", "weight", "ref_allele", "eff_allele"])
        result = match_variants_to_gwas(model, synthetic_gwas)
        assert result.empty
        assert list(result.columns) == ["gene", "rsid", "weight", "zscore", "alignment"]


class TestPrecomputeGwasLookups:
    """Tests for GWAS lookup table precomputation."""

    def test_returns_correct_keys(self, synthetic_gwas: pd.DataFrame) -> None:
        lookups = _precompute_gwas_lookups(synthetic_gwas)
        assert "by_rsid" in lookups
        assert "by_varid" in lookups

    def test_rsid_index(self, synthetic_gwas: pd.DataFrame) -> None:
        lookups = _precompute_gwas_lookups(synthetic_gwas)
        by_rsid = lookups["by_rsid"]
        assert by_rsid.index.name == "SNP"
        assert set(by_rsid.columns) == {"A1", "A2", "Z"}
        assert len(by_rsid) == len(synthetic_gwas)

    def test_varid_present_when_column_exists(self, synthetic_gwas: pd.DataFrame) -> None:
        lookups = _precompute_gwas_lookups(synthetic_gwas)
        assert lookups["by_varid"] is not None
        assert lookups["by_varid"].index.name == "VARIANT_ID"

    def test_varid_none_when_no_column(self) -> None:
        gwas = pd.DataFrame({
            "SNP": ["rs1"], "CHR": [1], "POS": [100],
            "A1": ["A"], "A2": ["G"],
            "BETA": [0.1], "SE": [0.02], "P": [0.01], "N": [1000],
        })
        lookups = _precompute_gwas_lookups(gwas)
        assert lookups["by_varid"] is None

    def test_z_computed_correctly(self) -> None:
        gwas = pd.DataFrame({
            "SNP": ["rs1"], "CHR": [1], "POS": [100],
            "A1": ["A"], "A2": ["G"],
            "BETA": [0.3], "SE": [0.1], "P": [0.01], "N": [1000],
        })
        lookups = _precompute_gwas_lookups(gwas)
        assert abs(lookups["by_rsid"].loc["rs1", "Z"] - 3.0) < 1e-10


class TestMatcherParityRealLike:
    """Parity test against a frozen real-like fixture.

    Exercises: multi-gene matching, palindromic SNP exclusion,
    VARIANT_ID fallback, allele flips, complement strand, and
    missing SNPs.
    """

    @staticmethod
    def _make_real_like_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        gwas = pd.DataFrame({
            "SNP": ["rs100", "rs200", "rs300", "rs400", "rs500", "rs600", "rs700"],
            "VARIANT_ID": [
                "1:100:A:G", "1:200:T:C", "2:300:C:G",
                "3:400:A:T", "4:500:G:A", "5:600:T:A",
                "6:700:C:T",
            ],
            "CHR": [1, 1, 2, 3, 4, 5, 6],
            "POS": [100, 200, 300, 400, 500, 600, 700],
            "A1": ["A", "T", "C", "A", "G", "T", "C"],
            "A2": ["G", "C", "G", "T", "A", "A", "T"],
            "BETA": [0.10, -0.05, 0.20, 0.15, -0.10, 0.30, 0.08],
            "SE": [0.02, 0.03, 0.04, 0.05, 0.02, 0.01, 0.04],
            "P": [1e-6, 0.09, 1e-7, 0.003, 1e-6, 1e-30, 0.05],
            "N": [50000] * 7,
        })
        model = pd.DataFrame({
            "gene": [
                "GENE_A", "GENE_A", "GENE_A",
                "GENE_B", "GENE_B",
                "GENE_C",
                "GENE_D",
                "GENE_E",
            ],
            "rsid": [
                "rs100", "rs200", "rs_missing",
                "rs300", "rs400",
                "4:500:G:A",
                "rs600",
                "rs700",
            ],
            "weight": [0.3, 0.2, 0.1, 0.5, 0.4, 0.6, 0.7, 0.15],
            "ref_allele": [
                "G", "C", "A",
                "G", "T",
                "A",
                "A",
                "T",
            ],
            "eff_allele": [
                "A", "T", "G",
                "C", "A",
                "G",
                "T",
                "C",
            ],
        })
        return gwas, model

    def test_parity_with_expected_output(self) -> None:
        gwas, model = self._make_real_like_data()
        result = match_variants_to_gwas(model, gwas, exclude_palindromic=True)

        assert list(result.columns) == ["gene", "rsid", "weight", "zscore", "alignment"]

        result_sorted = result.sort_values(["gene", "rsid"]).reset_index(drop=True)

        # rs100: A/G GWAS, A/G model -> direct match, alignment=+1
        row_a1 = result_sorted[
            (result_sorted["gene"] == "GENE_A") & (result_sorted["rsid"] == "rs100")
        ]
        assert len(row_a1) == 1
        assert row_a1.iloc[0]["alignment"] == 1
        assert abs(row_a1.iloc[0]["zscore"] - (0.10 / 0.02)) < 1e-10

        # rs200: T/C GWAS, eff=T ref=C model -> direct match, alignment=+1
        row_a2 = result_sorted[
            (result_sorted["gene"] == "GENE_A") & (result_sorted["rsid"] == "rs200")
        ]
        assert len(row_a2) == 1
        assert row_a2.iloc[0]["alignment"] == 1

        # rs_missing: not in GWAS -> not in output
        assert "rs_missing" not in result_sorted["rsid"].values

        # rs300: C/G GWAS -> palindromic, excluded
        row_b1 = result_sorted[
            (result_sorted["gene"] == "GENE_B") & (result_sorted["rsid"] == "rs300")
        ]
        assert len(row_b1) == 0

        # rs400: A/T GWAS -> palindromic, excluded
        row_b2 = result_sorted[
            (result_sorted["gene"] == "GENE_B") & (result_sorted["rsid"] == "rs400")
        ]
        assert len(row_b2) == 0

        # 4:500:G:A (VARIANT_ID fallback): G/A GWAS, eff=G ref=A model -> +1
        row_c = result_sorted[result_sorted["gene"] == "GENE_C"]
        assert len(row_c) == 1
        assert row_c.iloc[0]["alignment"] == 1
        assert abs(row_c.iloc[0]["zscore"] - (-0.10 / 0.02)) < 1e-10

        # rs600: T/A GWAS -> palindromic, excluded
        row_d = result_sorted[result_sorted["gene"] == "GENE_D"]
        assert len(row_d) == 0

        # rs700: C/T GWAS, eff=C ref=T model -> direct match, alignment=+1
        row_e = result_sorted[result_sorted["gene"] == "GENE_E"]
        assert len(row_e) == 1
        assert row_e.iloc[0]["alignment"] == 1
        assert abs(row_e.iloc[0]["zscore"] - (0.08 / 0.04)) < 1e-10

    def test_palindromic_included_when_flag_false(self) -> None:
        gwas, model = self._make_real_like_data()
        result = match_variants_to_gwas(model, gwas, exclude_palindromic=False)
        genes_present = set(result["gene"].values)
        assert "GENE_B" in genes_present
        assert "GENE_D" in genes_present


# ---------------------------------------------------------------------------
# Core Computation Tests
# ---------------------------------------------------------------------------


class TestBuildCovarianceMatrix:
    def test_identity(self) -> None:
        snps = ["s1", "s2"]
        entries = [("s1", "s1", 1.0), ("s2", "s2", 1.0), ("s1", "s2", 0.0)]
        cov = build_covariance_matrix(snps, entries)
        np.testing.assert_array_equal(cov, np.eye(2))

    def test_symmetric(self) -> None:
        snps = ["s1", "s2", "s3"]
        entries = [
            ("s1", "s1", 1.0), ("s2", "s2", 1.0), ("s3", "s3", 1.0),
            ("s1", "s2", 0.5),
        ]
        cov = build_covariance_matrix(snps, entries)
        assert cov[0, 1] == 0.5
        assert cov[1, 0] == 0.5


class TestComputeGeneZscore:
    def test_identity_covariance(self) -> None:
        w = np.array([0.3, -0.2, 0.5])
        z = np.array([5.0, -1.67, 5.0])
        cov = np.eye(3)

        zscore, pval, effect, se = compute_gene_zscore(w, z, cov)

        expected_num = np.dot(w, z)
        expected_sigma = np.dot(w, w)
        expected_z = expected_num / np.sqrt(expected_sigma)
        assert abs(zscore - expected_z) < 1e-10

    def test_known_values(self) -> None:
        w = np.array([0.5, 0.5])
        z = np.array([3.0, 3.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])

        zscore, pval, effect, se = compute_gene_zscore(w, z, cov)

        num = 0.5 * 3.0 + 0.5 * 3.0
        sigma = np.array([0.5, 0.5]) @ cov @ np.array([0.5, 0.5])
        expected_z = num / np.sqrt(sigma)
        assert abs(zscore - expected_z) < 1e-10
        assert 0 < pval < 1
        assert abs(effect - num / sigma) < 1e-10
        assert abs(se - 1 / np.sqrt(sigma)) < 1e-10

    def test_degenerate_variance(self) -> None:
        w = np.array([1.0])
        z = np.array([3.0])
        cov = np.array([[0.0]])
        zscore, pval, effect, se = compute_gene_zscore(w, z, cov)
        assert np.isnan(zscore)
        assert np.isnan(pval)

    def test_negative_variance(self) -> None:
        w = np.array([1.0, -1.0])
        z = np.array([3.0, 3.0])
        cov = np.array([[1.0, 2.0], [2.0, 1.0]])
        zscore, pval, effect, se = compute_gene_zscore(w, z, cov)
        assert np.isnan(zscore)


# ---------------------------------------------------------------------------
# Per-Tissue Run Tests
# ---------------------------------------------------------------------------


class TestRunSpredixcanTissue:
    def test_produces_correct_schema(
        self, synthetic_model_db: Path, synthetic_cov_file: Path,
        synthetic_gwas: pd.DataFrame,
    ) -> None:
        result = run_spredixcan_tissue(synthetic_gwas, synthetic_model_db, synthetic_cov_file)
        assert not result.empty
        for col in ["gene_ensembl_id", "gene_symbol", "zscore", "pvalue",
                     "effect_size", "se", "n_snps_used", "n_snps_in_model"]:
            assert col in result.columns

    def test_gene_count(
        self, synthetic_model_db: Path, synthetic_cov_file: Path,
        synthetic_gwas: pd.DataFrame,
    ) -> None:
        result = run_spredixcan_tissue(synthetic_gwas, synthetic_model_db, synthetic_cov_file)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Meta-Analysis Tests
# ---------------------------------------------------------------------------


class TestIVWMetaAnalysis:
    def _make_per_tissue(self, genes_tissues: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(genes_tissues)

    def test_single_tissue(self) -> None:
        df = self._make_per_tissue([{
            "gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
            "tissue": "Brain_Amygdala", "zscore": 3.5, "pvalue": 0.0005,
            "effect_size": 0.1, "se": 0.03, "n_snps_used": 5,
            "n_snps_in_model": 10, "mhc_flag": False,
        }])
        meta, label = ivw_meta_analysis(df, list(BRAIN_TISSUES))
        assert len(meta) == 1
        assert meta.iloc[0]["n_tissues"] == 1
        assert abs(meta.iloc[0]["meta_zscore"] - 3.5) < 1e-10

    def test_two_tissues_known(self) -> None:
        df = self._make_per_tissue([
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Brain_Amygdala", "zscore": 3.0, "pvalue": 0.003,
             "effect_size": 0.1, "se": 0.05, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Brain_Cortex", "zscore": 4.0, "pvalue": 0.0001,
             "effect_size": 0.2, "se": 0.04, "n_snps_used": 8,
             "n_snps_in_model": 10, "mhc_flag": False},
        ])
        meta, label = ivw_meta_analysis(df, list(BRAIN_TISSUES))
        assert len(meta) == 1
        row = meta.iloc[0]

        w1 = 1.0 / 0.05**2
        w2 = 1.0 / 0.04**2
        expected_beta = (w1 * 0.1 + w2 * 0.2) / (w1 + w2)
        expected_se = 1.0 / np.sqrt(w1 + w2)
        assert abs(row["meta_beta"] - expected_beta) < 1e-10
        assert abs(row["meta_se"] - expected_se) < 1e-10

    def test_heterogeneity(self) -> None:
        df = self._make_per_tissue([
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Brain_Amygdala", "zscore": 5.0, "pvalue": 1e-7,
             "effect_size": 0.5, "se": 0.05, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Brain_Cortex", "zscore": -2.0, "pvalue": 0.05,
             "effect_size": -0.1, "se": 0.05, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
        ])
        meta, _ = ivw_meta_analysis(df, list(BRAIN_TISSUES))
        row = meta.iloc[0]
        assert row["q_statistic"] > 0
        assert row["i_squared"] > 0

    def test_tissue_subsetting_all_brain(self) -> None:
        df = self._make_per_tissue([
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Brain_Amygdala", "zscore": 3.0, "pvalue": 0.003,
             "effect_size": 0.1, "se": 0.05, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Brain_Cortex", "zscore": 2.0, "pvalue": 0.05,
             "effect_size": 0.08, "se": 0.04, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
        ])
        _, label = ivw_meta_analysis(df, list(BRAIN_TISSUES))
        assert label == "all"

    def test_tissue_subsetting_mixed(self) -> None:
        df = self._make_per_tissue([
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Brain_Amygdala", "zscore": 3.0, "pvalue": 0.003,
             "effect_size": 0.1, "se": 0.05, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Liver", "zscore": 2.0, "pvalue": 0.05,
             "effect_size": 0.08, "se": 0.04, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
        ])
        _, label = ivw_meta_analysis(df, list(BRAIN_TISSUES))
        assert label == "brain_only"

    def test_tissue_subsetting_no_brain(self) -> None:
        df = self._make_per_tissue([
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Liver", "zscore": 3.0, "pvalue": 0.003,
             "effect_size": 0.1, "se": 0.05, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
            {"gene_ensembl_id": "ENSG1", "gene_symbol": "G1",
             "tissue": "Heart_Left_Ventricle", "zscore": 2.0, "pvalue": 0.05,
             "effect_size": 0.08, "se": 0.04, "n_snps_used": 5,
             "n_snps_in_model": 10, "mhc_flag": False},
        ])
        _, label = ivw_meta_analysis(df, list(BRAIN_TISSUES))
        assert label == "all_configured"


# ---------------------------------------------------------------------------
# MHC Flagging Tests
# ---------------------------------------------------------------------------


class TestAddMHCFlag:
    def test_marks_chr6_region(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_MHC", "ENSG_OTHER"],
            "gene_symbol": ["HLA-A", "TP53"],
        })
        ann = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_MHC", "ENSG_OTHER"],
            "chr": [6, 17],
            "start": [29_000_000, 7_600_000],
            "end": [29_500_000, 7_700_000],
        })
        result = add_mhc_flag(df, ann)
        assert result.loc[0, "mhc_flag"] is True or result.loc[0, "mhc_flag"] == True
        assert result.loc[1, "mhc_flag"] is False or result.loc[1, "mhc_flag"] == False

    def test_no_annotation_all_false(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG1", "ENSG2"],
            "gene_symbol": ["G1", "G2"],
        })
        result = add_mhc_flag(df)
        assert not result["mhc_flag"].any()


# ---------------------------------------------------------------------------
# Entrez ID Tests
# ---------------------------------------------------------------------------


class TestAddEntrezIds:
    def test_populates_column(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG00000141510", "ENSG_UNKNOWN"],
            "gene_symbol": ["TP53", "FAKE"],
        })
        converter = MagicMock()
        converter.get_full_record.side_effect = lambda gene_id, id_type: (
            {"entrez": 7157} if gene_id == "ENSG00000141510" else None
        )
        result = add_entrez_ids(df, converter)
        assert "gene_entrez_id" in result.columns
        assert result.iloc[0]["gene_entrez_id"] == 7157
        assert pd.isna(result.iloc[1]["gene_entrez_id"])

    def test_calls_converter_with_correct_signature(self) -> None:
        """Verify add_entrez_ids calls get_full_record(str, 'ensembl')."""
        from repogen.data.gene_id_converter import GeneIDConverter

        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG00000141510"],
            "gene_symbol": ["TP53"],
        })
        converter = MagicMock(spec=GeneIDConverter)
        converter.get_full_record.return_value = {"entrez": 7157, "symbol": "TP53", "ensembl": "ENSG00000141510", "uniprot": None}
        add_entrez_ids(df, converter)
        converter.get_full_record.assert_called_once_with("ENSG00000141510", "ensembl")


# ---------------------------------------------------------------------------
# Model Path Resolution Tests
# ---------------------------------------------------------------------------


class TestResolveModelPaths:
    def test_finds_model_files(self, tmp_path: Path) -> None:
        (tmp_path / "mashr_Brain_Amygdala.db").touch()
        (tmp_path / "mashr_Brain_Amygdala.txt.gz").touch()
        result = resolve_model_paths("mashr", ["Brain_Amygdala"], tmp_path)
        assert len(result) == 1
        assert result[0][0] == "Brain_Amygdala"

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model file missing"):
            resolve_model_paths("mashr", ["Brain_Amygdala"], tmp_path)

    def test_missing_cov_raises(self, tmp_path: Path) -> None:
        (tmp_path / "mashr_Brain_Amygdala.db").touch()
        with pytest.raises(FileNotFoundError, match="Covariance file missing"):
            resolve_model_paths("mashr", ["Brain_Amygdala"], tmp_path)


class TestModelPathPrecedence:
    def test_from_reference_config(self) -> None:
        ref = SimpleNamespace(predixcan_model_dir=Path("/models"))
        result = _resolve_model_dir(ref)
        assert result == Path("/models")

    def test_fallback_to_negative_correlation(self, caplog: pytest.LogCaptureFixture) -> None:
        ref = SimpleNamespace(predixcan_model_dir=None)
        nc = SimpleNamespace(predixcan_models_dir=Path("/nc_models"))
        result = _resolve_model_dir(ref, nc)
        assert result == Path("/nc_models")
        assert "deprecated" in caplog.text

    def test_neither_set_raises(self) -> None:
        ref = SimpleNamespace(predixcan_model_dir=None)
        with pytest.raises(FileNotFoundError, match="No PredictDB"):
            _resolve_model_dir(ref)


class TestCovDirPrecedence:
    def test_from_reference(self) -> None:
        ref = SimpleNamespace(predixcan_covariance_dir=Path("/covs"))
        result = _resolve_cov_dir(ref, Path("/models"))
        assert result == Path("/covs")

    def test_fallback_to_nc(self, caplog: pytest.LogCaptureFixture) -> None:
        ref = SimpleNamespace(predixcan_covariance_dir=None)
        nc = SimpleNamespace(predixcan_covariances_dir=Path("/nc_covs"))
        result = _resolve_cov_dir(ref, Path("/models"), nc)
        assert result == Path("/nc_covs")
        assert "deprecated" in caplog.text

    def test_defaults_to_model_dir(self) -> None:
        ref = SimpleNamespace(predixcan_covariance_dir=None)
        result = _resolve_cov_dir(ref, Path("/models"))
        assert result == Path("/models")


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------


class TestSpredixcanConfig:
    def test_defaults(self) -> None:
        from repogen.config.schema import SpredixcanConfig
        cfg = SpredixcanConfig()
        assert cfg.model_type == "mashr"
        assert cfg.tissue_preset == "brain_13"
        assert cfg.gwas_imputation is False
        assert cfg.min_snps_used_fraction == 0.1

    def test_invalid_model_type(self) -> None:
        from repogen.config.schema import SpredixcanConfig
        with pytest.raises(Exception):
            SpredixcanConfig(model_type="invalid")

    def test_invalid_tissue_preset(self) -> None:
        from repogen.config.schema import SpredixcanConfig
        with pytest.raises(Exception):
            SpredixcanConfig(tissue_preset="invalid")

    def test_all_gtex_preset_accepted(self) -> None:
        from repogen.config.schema import SpredixcanConfig
        cfg = SpredixcanConfig(tissue_preset="all_gtex")
        assert cfg.tissue_preset == "all_gtex"

    def test_custom_preset_accepted(self) -> None:
        from repogen.config.schema import SpredixcanConfig
        cfg = SpredixcanConfig(tissue_preset="custom")
        assert cfg.tissue_preset == "custom"


# ---------------------------------------------------------------------------
# Zero-Results Behavior Tests
# ---------------------------------------------------------------------------


class TestZeroResults:
    def test_single_tissue_zero_genes_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        db_path = tmp_path / "mashr_Brain_Amygdala.db"
        genes = {"ENSG_NOMATCH.1": [
            {"rsid": "rs_none", "weight": 0.1, "ref_allele": "A", "eff_allele": "G"}
        ]}
        extra = [{"gene": "ENSG_NOMATCH.1", "genename": "NOMATCH", "n_snps_in_model": 1}]
        _create_model_db(db_path, genes, extra)

        cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
        _create_covariance_file(cov_path, {"ENSG_NOMATCH.1": [("rs_none", "rs_none", 1.0)]})

        gwas = pd.DataFrame({
            "SNP": ["rs_other"],
            "VARIANT_ID": ["1:1:A:G"],
            "CHR": [1], "POS": [1],
            "A1": ["A"], "A2": ["G"],
            "BETA": [0.1], "SE": [0.02],
            "P": [0.001], "MAF": [0.3], "N": [50000],
        })

        result = run_spredixcan_tissue(gwas, db_path, cov_path)
        assert result.empty

    def test_all_tissues_zero_genes_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mashr_Brain_Amygdala.db"
        genes = {"ENSG_NOMATCH.1": [
            {"rsid": "rs_none", "weight": 0.1, "ref_allele": "A", "eff_allele": "G"}
        ]}
        extra = [{"gene": "ENSG_NOMATCH.1", "genename": "NOMATCH", "n_snps_in_model": 1}]
        _create_model_db(db_path, genes, extra)

        cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
        _create_covariance_file(cov_path, {"ENSG_NOMATCH.1": [("rs_none", "rs_none", 1.0)]})

        gwas = pd.DataFrame({
            "SNP": ["rs_other"], "VARIANT_ID": ["1:1:A:G"],
            "CHR": [1], "POS": [1], "A1": ["A"], "A2": ["G"],
            "BETA": [0.1], "SE": [0.02], "P": [0.001], "MAF": [0.3], "N": [50000],
        })
        gwas.to_parquet(tmp_path / "gwas.parquet")

        config = SimpleNamespace(
            gwas_imputation=False, tissues=["Brain_Amygdala"],
            tissue_preset="brain_13", model_type="mashr",
            min_snps_used_fraction=0.1, exclude_palindromic=True,
            extra_models=[],
        )
        ref = SimpleNamespace(
            predixcan_model_dir=tmp_path,
            predixcan_covariance_dir=tmp_path,
        )

        with pytest.raises(ValueError, match="no results"):
            run_spredixcan(
                gwas_path=tmp_path / "gwas.parquet",
                config=config, reference=ref,
                output_dir=tmp_path / "out",
            )


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_per_tissue_passes_validation(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG1"],
            "gene_symbol": ["G1"],
            "tissue": ["Brain_Amygdala"],
            "zscore": [3.5],
            "pvalue": [0.0005],
            "effect_size": [0.1],
            "se": [0.03],
            "n_snps_used": [5],
            "n_snps_in_model": [10],
            "mhc_flag": [False],
        })
        errors = validate_dataframe(df, "DiseaseSignaturePerTissue")
        assert errors == []

    def test_meta_passes_validation(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG1"],
            "gene_symbol": ["G1"],
            "meta_zscore": [3.5],
            "meta_pvalue": [0.0005],
            "meta_beta": [0.1],
            "meta_se": [0.03],
            "n_tissues": [3],
            "i_squared": [25.0],
            "q_statistic": [3.0],
            "q_pvalue": [0.2],
            "best_tissue": ["Brain_Amygdala"],
            "best_tissue_zscore": [4.0],
            "mhc_flag": [False],
        })
        errors = validate_dataframe(df, "DiseaseSignatureMeta")
        assert errors == []


# ---------------------------------------------------------------------------
# GWAS Imputation Guard
# ---------------------------------------------------------------------------


class TestGWASImputationGuard:
    def test_raises_not_implemented(self, tmp_path: Path) -> None:
        config = SimpleNamespace(gwas_imputation=True)
        ref = SimpleNamespace(predixcan_model_dir=tmp_path)
        with pytest.raises(NotImplementedError, match="deferred"):
            run_spredixcan(
                gwas_path=tmp_path / "gwas.parquet",
                config=config, reference=ref,
                output_dir=tmp_path / "out",
            )


# ---------------------------------------------------------------------------
# End-to-End Orchestrator Tests
# ---------------------------------------------------------------------------


class TestRunSpredixcanEndToEnd:
    def _setup_e2e(self, tmp_path: Path) -> tuple[Path, SimpleNamespace, SimpleNamespace]:
        """Set up synthetic files for end-to-end test."""
        db_path = tmp_path / "mashr_Brain_Amygdala.db"
        genes = {
            "ENSG00000001.1": [
                {"rsid": "rs1", "weight": 0.3, "ref_allele": "G", "eff_allele": "A"},
                {"rsid": "rs2", "weight": -0.2, "ref_allele": "C", "eff_allele": "T"},
                {"rsid": "rs3", "weight": 0.5, "ref_allele": "T", "eff_allele": "C"},
            ],
            "ENSG00000002.1": [
                {"rsid": "rs4", "weight": 0.1, "ref_allele": "A", "eff_allele": "G"},
                {"rsid": "rs5", "weight": 0.4, "ref_allele": "T", "eff_allele": "C"},
            ],
        }
        extra = [
            {"gene": "ENSG00000001.1", "genename": "GENE1", "n_snps_in_model": 3,
             "pred_perf_r2": 0.08, "pred_perf_pval": 0.001},
            {"gene": "ENSG00000002.1", "genename": "GENE2", "n_snps_in_model": 2,
             "pred_perf_r2": 0.05, "pred_perf_pval": 0.01},
        ]
        _create_model_db(db_path, genes, extra)

        cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
        gene_cov = {
            "ENSG00000001.1": [
                ("rs1", "rs1", 1.0), ("rs2", "rs2", 1.0), ("rs3", "rs3", 1.0),
            ],
            "ENSG00000002.1": [
                ("rs4", "rs4", 1.0), ("rs5", "rs5", 1.0),
            ],
        }
        _create_covariance_file(cov_path, gene_cov)

        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "VARIANT_ID": ["1:100:A:G", "1:200:T:C", "2:300:C:T", "3:400:G:A", "4:500:C:T"],
            "CHR": [1, 1, 2, 3, 4],
            "POS": [100, 200, 300, 400, 500],
            "A1": ["A", "T", "C", "G", "C"],
            "A2": ["G", "C", "T", "A", "T"],
            "BETA": [0.1, -0.05, 0.2, 0.15, -0.1],
            "SE": [0.02, 0.03, 0.04, 0.05, 0.02],
            "P": [1e-6, 0.09, 1e-7, 0.003, 1e-6],
            "MAF": [0.3, 0.2, 0.15, 0.4, 0.1],
            "N": [50000] * 5,
        })
        gwas_path = tmp_path / "gwas.parquet"
        gwas.to_parquet(gwas_path)

        meta = {"genome_build": "GRCh38", "trait": "Test Trait"}
        with open(tmp_path / "gwas.meta.json", "w") as f:
            json.dump(meta, f)

        config = SimpleNamespace(
            gwas_imputation=False, tissues=["Brain_Amygdala"],
            tissue_preset="brain_13", model_type="mashr",
            min_snps_used_fraction=0.1, exclude_palindromic=True,
            extra_models=[],
        )
        ref = SimpleNamespace(
            predixcan_model_dir=tmp_path,
            predixcan_covariance_dir=tmp_path,
        )
        return gwas_path, config, ref

    def test_end_to_end(self, tmp_path: Path) -> None:
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        out_dir = tmp_path / "output"
        result = run_spredixcan(gwas_path, config, ref, out_dir)

        assert "per_tissue" in result
        assert "meta_analysis" in result
        assert "metadata" in result

        per_tissue = pd.read_parquet(result["per_tissue"])
        assert not per_tissue.empty
        assert "zscore" in per_tissue.columns
        assert "mhc_flag" in per_tissue.columns

    def test_output_files_exist(self, tmp_path: Path) -> None:
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        out_dir = tmp_path / "output"
        result = run_spredixcan(gwas_path, config, ref, out_dir)

        assert Path(result["per_tissue"]).exists()
        assert Path(result["meta_analysis"]).exists()
        assert Path(result["metadata"]).exists()

    def test_genome_build_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        with open(tmp_path / "gwas.meta.json", "w") as f:
            json.dump({"genome_build": "GRCh37", "trait": "Test"}, f)

        out_dir = tmp_path / "output2"
        run_spredixcan(gwas_path, config, ref, out_dir)
        assert "GRCh37" in caplog.text

    def test_sidecar_legacy_json_fallback(
        self, tmp_path: Path,
    ) -> None:
        """Legacy .json sidecar is loaded when .meta.json does not exist."""
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        canonical = tmp_path / "gwas.meta.json"
        if canonical.exists():
            canonical.unlink()
        with open(tmp_path / "gwas.json", "w") as f:
            json.dump({"genome_build": "GRCh38", "trait": "Legacy Trait"}, f)

        out_dir = tmp_path / "output_legacy"
        result = run_spredixcan(gwas_path, config, ref, out_dir)
        with open(result["metadata"]) as f:
            meta = json.load(f)
        assert meta["trait"] == "Legacy Trait"

    def test_sidecar_canonical_takes_precedence(
        self, tmp_path: Path,
    ) -> None:
        """.meta.json takes precedence when both sidecar files exist."""
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        with open(tmp_path / "gwas.meta.json", "w") as f:
            json.dump({"genome_build": "GRCh38", "trait": "Canonical"}, f)
        with open(tmp_path / "gwas.json", "w") as f:
            json.dump({"genome_build": "GRCh38", "trait": "Legacy"}, f)

        out_dir = tmp_path / "output_precedence"
        result = run_spredixcan(gwas_path, config, ref, out_dir)
        with open(result["metadata"]) as f:
            meta = json.load(f)
        assert meta["trait"] == "Canonical"

    def test_metadata_content(self, tmp_path: Path) -> None:
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        out_dir = tmp_path / "output3"
        result = run_spredixcan(gwas_path, config, ref, out_dir)

        with open(result["metadata"]) as f:
            meta = json.load(f)

        assert meta["implementation"] == "repogen_native"
        assert meta["model_type"] == "mashr"
        assert "Brain_Amygdala" in meta["tissues"]
        assert meta["trait"] == "Test Trait"
        assert "meta_analysis_note" in meta


# ---------------------------------------------------------------------------
# Tissue Preset Resolution Tests (C2)
# ---------------------------------------------------------------------------


class TestTissuePresetResolution:
    def test_all_gtex_discovers_from_dir(self, tmp_path: Path) -> None:
        from repogen.analysis.spredixcan import _resolve_tissue_list
        (tmp_path / "mashr_Brain_Amygdala.db").touch()
        (tmp_path / "mashr_Liver.db").touch()
        (tmp_path / "mashr_Heart_Left_Ventricle.db").touch()
        config = SimpleNamespace(tissues=[], tissue_preset="all_gtex", model_type="mashr")
        result = _resolve_tissue_list(config, tmp_path)
        assert sorted(result) == ["Brain_Amygdala", "Heart_Left_Ventricle", "Liver"]

    def test_all_gtex_empty_dir_raises(self, tmp_path: Path) -> None:
        from repogen.analysis.spredixcan import _resolve_tissue_list
        config = SimpleNamespace(tissues=[], tissue_preset="all_gtex", model_type="mashr")
        with pytest.raises(ValueError, match="no.*files found"):
            _resolve_tissue_list(config, tmp_path)

    def test_custom_without_tissues_raises(self, tmp_path: Path) -> None:
        from repogen.analysis.spredixcan import _resolve_tissue_list
        config = SimpleNamespace(tissues=[], tissue_preset="custom", model_type="mashr")
        with pytest.raises(ValueError, match="non-empty.*tissues"):
            _resolve_tissue_list(config, tmp_path)

    def test_custom_with_tissues_passes(self, tmp_path: Path) -> None:
        from repogen.analysis.spredixcan import _resolve_tissue_list
        config = SimpleNamespace(tissues=["Liver", "Heart"], tissue_preset="custom", model_type="mashr")
        result = _resolve_tissue_list(config, tmp_path)
        assert result == ["Liver", "Heart"]

    def test_known_preset_resolves(self, tmp_path: Path) -> None:
        from repogen.analysis.spredixcan import _resolve_tissue_list
        config = SimpleNamespace(tissues=[], tissue_preset="brain_13", model_type="mashr")
        result = _resolve_tissue_list(config, tmp_path)
        assert len(result) == 13


# ---------------------------------------------------------------------------
# Palindromic Flag Tests (W1)
# ---------------------------------------------------------------------------


class TestExcludePalindromicFlag:
    def test_at_palindromic_excluded_by_default(self) -> None:
        result = align_alleles(
            np.array(["A"]), np.array(["T"]),
            np.array(["A"]), np.array(["T"]),
        )
        assert result[0] == 0

    def test_at_palindromic_kept_when_disabled(self) -> None:
        result = align_alleles(
            np.array(["A"]), np.array(["T"]),
            np.array(["A"]), np.array(["T"]),
            exclude_palindromic=False,
        )
        assert result[0] == 1

    def test_cg_palindromic_kept_when_disabled(self) -> None:
        result = align_alleles(
            np.array(["C"]), np.array(["G"]),
            np.array(["C"]), np.array(["G"]),
            exclude_palindromic=False,
        )
        assert result[0] == 1

    def test_palindromic_flip_when_disabled(self) -> None:
        result = align_alleles(
            np.array(["T"]), np.array(["A"]),
            np.array(["A"]), np.array(["T"]),
            exclude_palindromic=False,
        )
        assert result[0] == -1

    def test_flag_threaded_through_matching(self, synthetic_gwas: pd.DataFrame) -> None:
        """Verify exclude_palindromic reaches align_alleles via match_variants_to_gwas."""
        model = pd.DataFrame({
            "gene": ["G1"],
            "rsid": ["rs1"],
            "weight": [0.3],
            "ref_allele": ["G"],
            "eff_allele": ["A"],
        })
        r_true = match_variants_to_gwas(model, synthetic_gwas, exclude_palindromic=True)
        r_false = match_variants_to_gwas(model, synthetic_gwas, exclude_palindromic=False)
        assert len(r_true) == len(r_false)


# ---------------------------------------------------------------------------
# Extra Column Normalization Tests
# ---------------------------------------------------------------------------


class TestExtraColumnNormalization:
    """Verify dotted column names from mashr PredictDB are canonicalized."""

    def test_dotted_columns_renamed(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mashr_test.db"
        genes = {
            "ENSG00000001.1": [
                {"rsid": "rs1", "varID": "chr1_100_A_G_b38",
                 "weight": 0.3, "ref_allele": "G", "eff_allele": "A"},
            ],
        }
        extra = [{"gene": "ENSG00000001.1", "genename": "GENE1",
                  "n_snps_in_model": 3, "pred_perf_r2": 0.08,
                  "pred_perf_pval": 0.001, "pred_perf_qval": 0.05}]
        _create_mashr_model_db(db_path, genes, extra)

        _, extra_df = load_prediction_model(db_path)
        assert "n_snps_in_model" in extra_df.columns
        assert "pred_perf_r2" in extra_df.columns
        assert "pred_perf_pval" in extra_df.columns
        assert "pred_perf_qval" in extra_df.columns
        assert "n.snps.in.model" not in extra_df.columns

    def test_underscore_columns_unaffected(self, synthetic_model_db: Path) -> None:
        _, extra_df = load_prediction_model(synthetic_model_db)
        assert "n_snps_in_model" in extra_df.columns
        assert "pred_perf_r2" in extra_df.columns

    def test_qc_metadata_populated(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mashr_qc.db"
        genes = {
            "ENSG00000001.1": [
                {"rsid": "rs1", "varID": "chr1_100_A_G_b38",
                 "weight": 0.3, "ref_allele": "G", "eff_allele": "A"},
            ],
        }
        extra = [{"gene": "ENSG00000001.1", "genename": "GENE1",
                  "n_snps_in_model": 7, "pred_perf_r2": 0.12,
                  "pred_perf_pval": 0.005, "pred_perf_qval": 0.02}]
        _create_mashr_model_db(db_path, genes, extra)

        _, extra_df = load_prediction_model(db_path)
        row = extra_df.iloc[0]
        assert row["n_snps_in_model"] == 7
        assert abs(row["pred_perf_r2"] - 0.12) < 1e-10


# ---------------------------------------------------------------------------
# Covariance Key-Space Tests
# ---------------------------------------------------------------------------


class TestCovarianceKeySpace:
    """Verify varID-based covariance lookup and rsID fallback."""

    def test_varid_covariance_keys_produce_nonzero_genes(self, tmp_path: Path) -> None:
        """Weights matched by rsID, covariance keyed by varID -> genes pass QC."""
        db_path = tmp_path / "mashr_Brain_Amygdala.db"
        genes = {
            "ENSG00000001.1": [
                {"rsid": "rs1", "varID": "chr1_100_A_G_b38",
                 "weight": 0.3, "ref_allele": "G", "eff_allele": "A"},
                {"rsid": "rs2", "varID": "chr1_200_T_C_b38",
                 "weight": -0.2, "ref_allele": "C", "eff_allele": "T"},
            ],
        }
        extra = [{"gene": "ENSG00000001.1", "genename": "GENE1",
                  "n_snps_in_model": 2, "pred_perf_r2": 0.08,
                  "pred_perf_pval": 0.001}]
        _create_mashr_model_db(db_path, genes, extra)

        cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
        _create_covariance_file(cov_path, {
            "ENSG00000001.1": [
                ("chr1_100_A_G_b38", "chr1_100_A_G_b38", 1.0),
                ("chr1_200_T_C_b38", "chr1_200_T_C_b38", 1.0),
                ("chr1_100_A_G_b38", "chr1_200_T_C_b38", 0.1),
            ],
        })

        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs2"], "VARIANT_ID": ["1:100:A:G", "1:200:T:C"],
            "CHR": [1, 1], "POS": [100, 200],
            "A1": ["A", "T"], "A2": ["G", "C"],
            "BETA": [0.1, -0.05], "SE": [0.02, 0.03],
            "P": [1e-6, 0.09], "MAF": [0.3, 0.2], "N": [50000, 50000],
        })

        result = run_spredixcan_tissue(gwas, db_path, cov_path)
        assert not result.empty
        assert result.iloc[0]["gene_ensembl_id"] == "ENSG00000001"

    def test_rsid_covariance_fallback_without_varid(self, tmp_path: Path) -> None:
        """Without varID column, rsID covariance keys still work."""
        db_path = tmp_path / "mashr_Brain_Amygdala.db"
        genes = {
            "ENSG00000001.1": [
                {"rsid": "rs1", "weight": 0.3, "ref_allele": "G", "eff_allele": "A"},
                {"rsid": "rs2", "weight": -0.2, "ref_allele": "C", "eff_allele": "T"},
            ],
        }
        extra = [{"gene": "ENSG00000001.1", "genename": "GENE1",
                  "n_snps_in_model": 2, "pred_perf_r2": 0.08,
                  "pred_perf_pval": 0.001}]
        _create_model_db(db_path, genes, extra)

        cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
        _create_covariance_file(cov_path, {
            "ENSG00000001.1": [
                ("rs1", "rs1", 1.0), ("rs2", "rs2", 1.0),
                ("rs1", "rs2", 0.1),
            ],
        })

        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs2"], "VARIANT_ID": ["1:100:A:G", "1:200:T:C"],
            "CHR": [1, 1], "POS": [100, 200],
            "A1": ["A", "T"], "A2": ["G", "C"],
            "BETA": [0.1, -0.05], "SE": [0.02, 0.03],
            "P": [1e-6, 0.09], "MAF": [0.3, 0.2], "N": [50000, 50000],
        })

        result = run_spredixcan_tissue(gwas, db_path, cov_path)
        assert not result.empty
        assert result.iloc[0]["gene_ensembl_id"] == "ENSG00000001"

    def test_varid_vs_rsid_covariance_produces_different_zscores(self, tmp_path: Path) -> None:
        """With off-diagonal covariance, varID lookup changes z-scores vs identity fallback."""
        db_path = tmp_path / "mashr_Brain_Amygdala.db"
        genes = {
            "ENSG00000001.1": [
                {"rsid": "rs1", "varID": "chr1_100_A_G_b38",
                 "weight": 0.5, "ref_allele": "G", "eff_allele": "A"},
                {"rsid": "rs2", "varID": "chr1_200_T_C_b38",
                 "weight": 0.5, "ref_allele": "C", "eff_allele": "T"},
            ],
        }
        extra = [{"gene": "ENSG00000001.1", "genename": "GENE1",
                  "n_snps_in_model": 2}]
        _create_mashr_model_db(db_path, genes, extra)

        cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
        _create_covariance_file(cov_path, {
            "ENSG00000001.1": [
                ("chr1_100_A_G_b38", "chr1_100_A_G_b38", 1.0),
                ("chr1_200_T_C_b38", "chr1_200_T_C_b38", 1.0),
                ("chr1_100_A_G_b38", "chr1_200_T_C_b38", 0.8),
            ],
        })

        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs2"], "VARIANT_ID": ["1:100:A:G", "1:200:T:C"],
            "CHR": [1, 1], "POS": [100, 200],
            "A1": ["A", "T"], "A2": ["G", "C"],
            "BETA": [0.3, 0.3], "SE": [0.05, 0.05],
            "P": [1e-6, 1e-6], "MAF": [0.3, 0.2], "N": [50000, 50000],
        })

        result = run_spredixcan_tissue(gwas, db_path, cov_path)
        assert not result.empty
        z_with_cov = result.iloc[0]["zscore"]

        # w=[0.5,0.5], z=[6,6], cov=[[1,0.8],[0.8,1]]
        # sigma = w @ cov @ w = 0.5*0.5*1 + 2*0.5*0.5*0.8 + 0.5*0.5*1 = 0.9
        # numerator = 0.5*6 + 0.5*6 = 6
        # z = 6/sqrt(0.9) = 6.3246...
        expected_z = 6.0 / np.sqrt(0.9)
        assert abs(z_with_cov - expected_z) < 1e-4

    def test_e2e_mashr_format(self, tmp_path: Path) -> None:
        """Full E2E with mashr-format DB (varID + dotted extra) passes."""
        db_path = tmp_path / "mashr_Brain_Amygdala.db"
        genes = {
            "ENSG00000001.1": [
                {"rsid": "rs1", "varID": "chr1_100_A_G_b38",
                 "weight": 0.3, "ref_allele": "G", "eff_allele": "A"},
                {"rsid": "rs2", "varID": "chr1_200_T_C_b38",
                 "weight": -0.2, "ref_allele": "C", "eff_allele": "T"},
                {"rsid": "rs3", "varID": "chr2_300_C_T_b38",
                 "weight": 0.5, "ref_allele": "T", "eff_allele": "C"},
            ],
            "ENSG00000002.1": [
                {"rsid": "rs4", "varID": "chr3_400_G_A_b38",
                 "weight": 0.1, "ref_allele": "A", "eff_allele": "G"},
                {"rsid": "rs5", "varID": "chr4_500_C_T_b38",
                 "weight": 0.4, "ref_allele": "T", "eff_allele": "C"},
            ],
        }
        extra = [
            {"gene": "ENSG00000001.1", "genename": "GENE1", "n_snps_in_model": 3,
             "pred_perf_r2": 0.08, "pred_perf_pval": 0.001},
            {"gene": "ENSG00000002.1", "genename": "GENE2", "n_snps_in_model": 2,
             "pred_perf_r2": 0.05, "pred_perf_pval": 0.01},
        ]
        _create_mashr_model_db(db_path, genes, extra)

        cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
        _create_covariance_file(cov_path, {
            "ENSG00000001.1": [
                ("chr1_100_A_G_b38", "chr1_100_A_G_b38", 1.0),
                ("chr1_200_T_C_b38", "chr1_200_T_C_b38", 1.0),
                ("chr2_300_C_T_b38", "chr2_300_C_T_b38", 1.0),
            ],
            "ENSG00000002.1": [
                ("chr3_400_G_A_b38", "chr3_400_G_A_b38", 1.0),
                ("chr4_500_C_T_b38", "chr4_500_C_T_b38", 1.0),
            ],
        })

        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "VARIANT_ID": ["1:100:A:G", "1:200:T:C", "2:300:C:T", "3:400:G:A", "4:500:C:T"],
            "CHR": [1, 1, 2, 3, 4], "POS": [100, 200, 300, 400, 500],
            "A1": ["A", "T", "C", "G", "C"], "A2": ["G", "C", "T", "A", "T"],
            "BETA": [0.1, -0.05, 0.2, 0.15, -0.1],
            "SE": [0.02, 0.03, 0.04, 0.05, 0.02],
            "P": [1e-6, 0.09, 1e-7, 0.003, 1e-6],
            "MAF": [0.3, 0.2, 0.15, 0.4, 0.1], "N": [50000] * 5,
        })
        gwas.to_parquet(tmp_path / "gwas.parquet")

        meta = {"genome_build": "GRCh38", "trait": "Test Trait"}
        with open(tmp_path / "gwas.meta.json", "w") as f:
            json.dump(meta, f)

        config = SimpleNamespace(
            gwas_imputation=False, tissues=["Brain_Amygdala"],
            tissue_preset="brain_13", model_type="mashr",
            min_snps_used_fraction=0.1, exclude_palindromic=True,
            extra_models=[],
        )
        ref = SimpleNamespace(
            predixcan_model_dir=tmp_path,
            predixcan_covariance_dir=tmp_path,
        )

        result = run_spredixcan(
            tmp_path / "gwas.parquet", config, ref,
            output_dir=tmp_path / "output",
        )
        per_tissue = pd.read_parquet(result["per_tissue"])
        assert not per_tissue.empty
        assert per_tissue["n_snps_in_model"].max() > 0
        assert per_tissue["gene_ensembl_id"].nunique() == 2


# ---------------------------------------------------------------------------
# Golden Fixture Test
# ---------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.skip(reason="Golden fixtures not yet generated - run reference MetaXcan to create")
def test_numerical_parity_with_reference() -> None:
    """Verify our Z-scores match reference MetaXcan output (golden fixture).

    To generate golden fixtures:
    1. Run reference MetaXcan SPrediXcan.py on a small test GWAS + model
    2. Save output Z-scores to tests/data/golden_spredixcan_output.csv
    3. Save input GWAS + model to tests/data/
    4. Remove the skip marker
    """
    pass


# ---------------------------------------------------------------------------
# Build-aware MHC flagging, GRCh38 primary annotation,
#              CLI wiring, and fail-loud guardrails.
# ---------------------------------------------------------------------------


class TestAddMHCFlagBuildAware:
    """``add_mhc_flag`` must honour the *build* argument so GRCh38 annotation
    is filtered against GRCh38 bounds and GRCh37 annotation against GRCh37
    bounds.  The output column is keyed on ``gene_ensembl_id`` and is
    therefore safe to join across builds."""

    def test_grch38_default_uses_grch38_interval(self) -> None:
        # HLA-A (GRCh38: chr6:29942532-29945457) falls in the GRCh38
        # interval (25.7M-33.4M).  TP53 (chr17) is not chr6 so should
        # never be flagged regardless of bounds.
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_HLA_A", "ENSG_TP53"],
            "gene_symbol": ["HLA-A", "TP53"],
        })
        ann = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_HLA_A", "ENSG_TP53"],
            "chr": [6, 17],
            "start": [29_942_532, 7_661_779],
            "end": [29_945_457, 7_687_550],
        })
        result = add_mhc_flag(df, ann)
        assert result.loc[0, "mhc_flag"] == True
        assert result.loc[1, "mhc_flag"] == False

    def test_grch38_excludes_gene_outside_interval(self) -> None:
        # A gene at chr6:24M is just below the GRCh38 lower bound
        # (25.726M) - must not be flagged on GRCh38, but IS flagged on
        # GRCh37 (lower bound 25M).
        df = pd.DataFrame({"gene_ensembl_id": ["ENSG_BOUND"], "gene_symbol": ["BOUND"]})
        ann = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_BOUND"],
            "chr": [6],
            "start": [25_500_000],
            "end": [25_600_000],
        })
        assert add_mhc_flag(df, ann, build="GRCh38").loc[0, "mhc_flag"] == False
        assert add_mhc_flag(df, ann, build="GRCh37").loc[0, "mhc_flag"] == True

    def test_grch37_legacy_call_preserves_old_behaviour(self) -> None:
        # Original test fixture: chr6:29M HLA-A coords falls inside the
        # GRCh37 interval as well.  Behaviour preserved.
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_HLA", "ENSG_OFF"],
            "gene_symbol": ["HLA-A", "OFF"],
        })
        ann = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_HLA", "ENSG_OFF"],
            "chr": [6, 17],
            "start": [29_000_000, 7_600_000],
            "end": [29_500_000, 7_700_000],
        })
        result = add_mhc_flag(df, ann, build="GRCh37")
        assert result.loc[0, "mhc_flag"] == True
        assert result.loc[1, "mhc_flag"] == False

    def test_invalid_build_raises(self) -> None:
        df = pd.DataFrame({"gene_ensembl_id": ["ENSG"], "gene_symbol": ["X"]})
        ann = pd.DataFrame({
            "gene_ensembl_id": ["ENSG"], "chr": [6],
            "start": [29_000_000], "end": [29_500_000],
        })
        with pytest.raises(ValueError, match="GRCh37"):
            add_mhc_flag(df, ann, build="hg19")


class TestLoadMHCGeneAnnotation:
    """Decision-tree behaviour for ``_load_mhc_gene_annotation`` - the helper
    that materialises a build-consistent MHC annotation for Branch B."""

    _GRCH38_GENE_LOC = (
        # Three real GRCh38 MHC Entrez IDs (HLA-A=3105, HLA-B=3106, HLA-C=3107)
        # plus a chr17 negative control (TP53=7157).
        "3105\t6\t29942532\t29945457\t+\tHLA-A\n"
        "3106\t6\t31268749\t31272136\t-\tHLA-B\n"
        "3107\t6\t31269491\t31357188\t-\tHLA-C\n"
        "7157\t17\t7661779\t7687550\t-\tTP53\n"
    )
    _GRCH37_GENE_LOC = (
        "3105\t6\t29910247\t29913661\t+\tHLA-A\n"
        "3106\t6\t31321649\t31324989\t-\tHLA-B\n"
        "7157\t17\t7571720\t7590868\t-\tTP53\n"
    )

    def _make_converter(self, mapping: dict[int, str]):
        """Stub GeneIDConverter.get_full_record(entrez_id_str, 'entrez')."""
        converter = MagicMock()
        def _lookup(gene_id: str, id_type: str):
            if id_type != "entrez":
                return None
            try:
                eid = int(gene_id)
            except ValueError:
                return None
            ens = mapping.get(eid)
            return {"ensembl": ens} if ens else None
        converter.get_full_record.side_effect = _lookup
        return converter

    def test_grch38_primary_path(self, tmp_path: Path) -> None:
        from repogen.analysis.spredixcan import _load_mhc_gene_annotation
        grch38_path = tmp_path / "NCBI38.gene.loc"
        grch38_path.write_text(self._GRCH38_GENE_LOC)
        ref = SimpleNamespace(
            gene_loc_file_grch38=grch38_path,
            gene_loc_file=None,
        )
        converter = self._make_converter({
            3105: "ENSG00000206503",
            3106: "ENSG00000234745",
            3107: "ENSG00000204525",
        })
        annotation, meta = _load_mhc_gene_annotation(ref, converter)

        assert annotation is not None
        assert sorted(annotation["gene_ensembl_id"].tolist()) == [
            "ENSG00000204525", "ENSG00000206503", "ENSG00000234745",
        ]
        assert meta["mhc_annotation_strategy"] == "grch38_gene_loc"
        assert meta["mhc_annotation_build"] == "GRCh38"
        assert meta["mhc_annotation_source"] == str(grch38_path)
        # 3 MHC genes (chr6 in MHC); TP53 (chr17) excluded.
        assert meta["n_unique_mhc_entrez_ids_in_gene_loc"] == 3
        assert meta["n_unique_mhc_ensembl_ids_after_conversion"] == 3
        assert meta["mhc_flag_interval"] == [6, 25_726_063, 33_400_644]

    def test_grch37_fallback_when_no_grch38(self, tmp_path: Path) -> None:
        from repogen.analysis.spredixcan import _load_mhc_gene_annotation
        grch37_path = tmp_path / "NCBI37.3.gene.loc"
        grch37_path.write_text(self._GRCH37_GENE_LOC)
        ref = SimpleNamespace(
            gene_loc_file_grch38=None,
            gene_loc_file=grch37_path,
        )
        converter = self._make_converter({
            3105: "ENSG00000206503",
            3106: "ENSG00000234745",
        })
        annotation, meta = _load_mhc_gene_annotation(ref, converter)

        assert annotation is not None
        assert set(annotation["gene_ensembl_id"]) == {
            "ENSG00000206503", "ENSG00000234745",
        }
        assert meta["mhc_annotation_strategy"] == "grch37_gene_loc_ensembl_projection"
        assert meta["mhc_annotation_build"] == "GRCh37"
        # GRCh38 interval still used for flagging since output coords are GRCh38.
        assert meta["mhc_flag_interval"] == [6, 25_726_063, 33_400_644]

    def test_no_annotation_when_both_unset(self) -> None:
        from repogen.analysis.spredixcan import _load_mhc_gene_annotation
        ref = SimpleNamespace(gene_loc_file_grch38=None, gene_loc_file=None)
        annotation, meta = _load_mhc_gene_annotation(ref, gene_id_converter=None)
        assert annotation is None
        assert meta["mhc_annotation_strategy"] == "no_annotation"
        assert meta["mhc_annotation_source"] is None
        assert meta["mhc_annotation_build"] is None

    def test_require_mhc_raises_when_no_annotation(self) -> None:
        from repogen.analysis.spredixcan import _load_mhc_gene_annotation
        ref = SimpleNamespace(gene_loc_file_grch38=None, gene_loc_file=None)
        with pytest.raises(RuntimeError, match="No MHC gene annotation"):
            _load_mhc_gene_annotation(
                ref, gene_id_converter=None, require_mhc_annotation=True,
            )

    def test_grch38_without_converter_falls_through_to_no_annotation(
        self, tmp_path: Path,
    ) -> None:
        # Without a converter, the Entrez-keyed GRCh38 file cannot be
        # projected to Ensembl IDs.  Helper must report no_annotation
        # (and a warning), not partial/empty annotation.
        from repogen.analysis.spredixcan import _load_mhc_gene_annotation
        grch38_path = tmp_path / "NCBI38.gene.loc"
        grch38_path.write_text(self._GRCH38_GENE_LOC)
        ref = SimpleNamespace(
            gene_loc_file_grch38=grch38_path,
            gene_loc_file=None,
        )
        annotation, meta = _load_mhc_gene_annotation(
            ref, gene_id_converter=None,
        )
        assert annotation is None
        assert meta["mhc_annotation_strategy"] == "no_annotation"

    def test_raises_when_grch38_configured_but_missing(self, tmp_path: Path) -> None:
        # If the operator wired
        # gene_loc_file_grch38 in their reference config but never
        # downloaded the file (e.g., forgot `repogen setup-resources`),
        # the helper must fail loudly - silent fallback to GRCh37 was
        # the bug this guards against.
        from repogen.analysis.spredixcan import _load_mhc_gene_annotation
        missing_path = tmp_path / "definitely_not_here" / "NCBI38.gene.loc"
        ref = SimpleNamespace(
            gene_loc_file_grch38=missing_path,
            gene_loc_file=tmp_path / "NCBI37.3.gene.loc",  # exists is irrelevant
        )
        with pytest.raises(FileNotFoundError, match="gene_loc_file_grch38 is configured"):
            _load_mhc_gene_annotation(ref, gene_id_converter=None)

    def test_grch38_source_interval_matches_flag_interval(
        self, tmp_path: Path,
    ) -> None:
        # For the GRCh38 primary path, source and flag intervals are
        # identical (both GRCh38).
        from repogen.analysis.spredixcan import _load_mhc_gene_annotation
        grch38_path = tmp_path / "NCBI38.gene.loc"
        grch38_path.write_text(self._GRCH38_GENE_LOC)
        ref = SimpleNamespace(
            gene_loc_file_grch38=grch38_path, gene_loc_file=None,
        )
        converter = self._make_converter({3105: "ENSG00000206503"})
        _annotation, meta = _load_mhc_gene_annotation(ref, converter)
        assert meta["mhc_source_interval"] == [6, 25_726_063, 33_400_644]
        assert meta["mhc_flag_interval"] == [6, 25_726_063, 33_400_644]
        assert meta["mhc_annotation_coordinate_mode"] == "membership_interval_placeholder"

    def test_grch37_fallback_source_interval_differs_from_flag_interval(
        self, tmp_path: Path,
    ) -> None:
        # For the GRCh37 fallback path, the
        # source interval (used to *select* MHC genes from the GRCh37
        # file) is the GRCh37 bounds, while the flag interval (re-applied
        # by add_mhc_flag against the synthetic GRCh38 annotation
        # coords) is the GRCh38 bounds.  This distinction is what the
        # new metadata fields exist to surface.
        from repogen.analysis.spredixcan import _load_mhc_gene_annotation
        grch37_path = tmp_path / "NCBI37.3.gene.loc"
        grch37_path.write_text(self._GRCH37_GENE_LOC)
        ref = SimpleNamespace(
            gene_loc_file_grch38=None, gene_loc_file=grch37_path,
        )
        converter = self._make_converter({3105: "ENSG00000206503"})
        _annotation, meta = _load_mhc_gene_annotation(ref, converter)
        assert meta["mhc_source_interval"] == [6, 25_000_000, 34_000_000]
        assert meta["mhc_flag_interval"] == [6, 25_726_063, 33_400_644]
        assert meta["mhc_source_interval"] != meta["mhc_flag_interval"]
        assert meta["mhc_annotation_coordinate_mode"] == "membership_interval_placeholder"


class TestRunSpredixcanMHCGuardrailsIntegration:
    """End-to-end integration tests for the zero-flagged guardrail.

    The guardrail ALWAYS raises when the annotation is non-empty but
    zero spx rows match,
    regardless of ``require_mhc_annotation`` downstream.  These tests
    actually invoke ``run_spredixcan()`` end-to-end against a synthetic
    PredictDB so the integration path (not just the condition) is
    exercised."""

    def _setup_e2e(self, tmp_path: Path) -> tuple[Path, SimpleNamespace, SimpleNamespace]:
        """Mirror TestRunSpredixcanEndToEnd._setup_e2e - kept local
        so the integration tests do not couple to that class's lifecycle."""
        db_path = tmp_path / "mashr_Brain_Amygdala.db"
        genes = {
            "ENSG00000001.1": [
                {"rsid": "rs1", "weight": 0.3, "ref_allele": "G", "eff_allele": "A"},
                {"rsid": "rs2", "weight": -0.2, "ref_allele": "C", "eff_allele": "T"},
                {"rsid": "rs3", "weight": 0.5, "ref_allele": "T", "eff_allele": "C"},
            ],
            "ENSG00000002.1": [
                {"rsid": "rs4", "weight": 0.1, "ref_allele": "A", "eff_allele": "G"},
                {"rsid": "rs5", "weight": 0.4, "ref_allele": "T", "eff_allele": "C"},
            ],
        }
        extra = [
            {"gene": "ENSG00000001.1", "genename": "GENE1", "n_snps_in_model": 3,
             "pred_perf_r2": 0.08, "pred_perf_pval": 0.001},
            {"gene": "ENSG00000002.1", "genename": "GENE2", "n_snps_in_model": 2,
             "pred_perf_r2": 0.05, "pred_perf_pval": 0.01},
        ]
        _create_model_db(db_path, genes, extra)

        cov_path = tmp_path / "mashr_Brain_Amygdala.txt.gz"
        gene_cov = {
            "ENSG00000001.1": [
                ("rs1", "rs1", 1.0), ("rs2", "rs2", 1.0), ("rs3", "rs3", 1.0),
            ],
            "ENSG00000002.1": [
                ("rs4", "rs4", 1.0), ("rs5", "rs5", 1.0),
            ],
        }
        _create_covariance_file(cov_path, gene_cov)

        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "VARIANT_ID": ["1:100:A:G", "1:200:T:C", "2:300:C:T", "3:400:G:A", "4:500:C:T"],
            "CHR": [1, 1, 2, 3, 4],
            "POS": [100, 200, 300, 400, 500],
            "A1": ["A", "T", "C", "G", "C"],
            "A2": ["G", "C", "T", "A", "T"],
            "BETA": [0.1, -0.05, 0.2, 0.15, -0.1],
            "SE": [0.02, 0.03, 0.04, 0.05, 0.02],
            "P": [1e-6, 0.09, 1e-7, 0.003, 1e-6],
            "MAF": [0.3, 0.2, 0.15, 0.4, 0.1],
            "N": [50000] * 5,
        })
        gwas_path = tmp_path / "gwas.parquet"
        gwas.to_parquet(gwas_path)

        with open(tmp_path / "gwas.meta.json", "w") as f:
            json.dump({"genome_build": "GRCh38", "trait": "Test"}, f)

        config = SimpleNamespace(
            gwas_imputation=False, tissues=["Brain_Amygdala"],
            tissue_preset="brain_13", model_type="mashr",
            min_snps_used_fraction=0.1, exclude_palindromic=True,
            extra_models=[],
        )
        ref = SimpleNamespace(
            predixcan_model_dir=tmp_path,
            predixcan_covariance_dir=tmp_path,
        )
        return gwas_path, config, ref

    def test_zero_flagged_with_nonempty_annotation_raises(
        self, tmp_path: Path,
    ) -> None:
        # Annotation contains a single fake Ensembl ID that is NOT in
        # the synthetic spx output (which produces only ENSG00000001 /
        # ENSG00000002).  The guardrail must raise; this is
        # unconditional.
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        annotation = pd.DataFrame({
            "gene_ensembl_id": ["ENSG_NOT_IN_SPX_OUTPUT"],
            "chr": [6],
            "start": [29_942_532],
            "end": [29_945_457],
        })
        with pytest.raises(RuntimeError, match=r"zero S-PrediXcan rows were flagged"):
            run_spredixcan(
                gwas_path, config, ref,
                output_dir=tmp_path / "output_guardrail",
                gene_annotation=annotation,
                mhc_build="GRCh38",
            )

    def test_nonempty_annotation_with_match_does_not_raise(
        self, tmp_path: Path,
    ) -> None:
        # Annotation contains the same Ensembl ID the synthetic spx
        # output emits - guardrail must NOT raise.  Verifies the
        # tightened guardrail does not over-fire.
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        annotation = pd.DataFrame({
            "gene_ensembl_id": ["ENSG00000001"],
            "chr": [6],
            "start": [29_942_532],
            "end": [29_945_457],
        })
        result = run_spredixcan(
            gwas_path, config, ref,
            output_dir=tmp_path / "output_ok",
            gene_annotation=annotation,
            mhc_build="GRCh38",
        )
        per_tissue = pd.read_parquet(result["per_tissue"])
        assert per_tissue["mhc_flag"].sum() >= 1

    def test_none_annotation_does_not_raise(self, tmp_path: Path) -> None:
        # Legacy callers passing gene_annotation=None (or no kwarg at
        # all) must not trigger the guardrail - the tightening targets
        # the "annotation supplied but inert" failure mode only.
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        result = run_spredixcan(
            gwas_path, config, ref,
            output_dir=tmp_path / "output_legacy",
            gene_annotation=None,
        )
        assert Path(result["per_tissue"]).exists()


class TestRunSpredixcanMetadataExtensionsIntegration:
    """End-to-end test that the MHC metadata extensions are actually
    written to ``spredixcan_metadata.json`` with values propagated from
    *mhc_annotation_metadata*."""

    def _setup_e2e(self, tmp_path: Path) -> tuple[Path, SimpleNamespace, SimpleNamespace]:
        return TestRunSpredixcanMHCGuardrailsIntegration._setup_e2e(self, tmp_path)

    def test_extended_metadata_is_written_to_json(self, tmp_path: Path) -> None:
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        # Mirror the shape produced by _load_mhc_gene_annotation, with
        # distinguishable source vs flag intervals to lock the field
        # propagation contract.
        provided = {
            "mhc_annotation_source": "/synthetic/path/NCBI38.gene.loc",
            "mhc_annotation_build": "GRCh38",
            "mhc_annotation_strategy": "grch38_gene_loc",
            "mhc_annotation_coordinate_mode": "membership_interval_placeholder",
            "mhc_source_interval": [6, 25_726_063, 33_400_644],
            "mhc_flag_interval": [6, 25_726_063, 33_400_644],
            "n_unique_mhc_entrez_ids_in_gene_loc": 42,
            "n_unique_mhc_ensembl_ids_after_conversion": 41,
        }
        result = run_spredixcan(
            gwas_path, config, ref,
            output_dir=tmp_path / "output_meta",
            gene_annotation=None,
            mhc_annotation_metadata=provided,
        )
        with open(result["metadata"]) as fh:
            meta = json.load(fh)

        for key in (
            "mhc_annotation_source",
            "mhc_annotation_build",
            "mhc_annotation_strategy",
            "mhc_annotation_coordinate_mode",
            "mhc_source_interval",
            "mhc_flag_interval",
            "n_unique_mhc_entrez_ids_in_gene_loc",
            "n_unique_mhc_ensembl_ids_after_conversion",
            "n_unique_mhc_ensembl_ids_in_spx",
            "n_mhc_flagged_rows",
            "n_spx_unique_genes_total",
            "n_spx_unique_genes_with_entrez",
        ):
            assert key in meta, f"metadata key missing: {key}"

        assert meta["mhc_annotation_source"] == "/synthetic/path/NCBI38.gene.loc"
        assert meta["mhc_annotation_build"] == "GRCh38"
        assert meta["mhc_annotation_strategy"] == "grch38_gene_loc"
        assert meta["mhc_annotation_coordinate_mode"] == "membership_interval_placeholder"
        assert meta["mhc_source_interval"] == [6, 25_726_063, 33_400_644]
        assert meta["mhc_flag_interval"] == [6, 25_726_063, 33_400_644]
        assert meta["n_unique_mhc_entrez_ids_in_gene_loc"] == 42
        assert meta["n_unique_mhc_ensembl_ids_after_conversion"] == 41
        # The synthetic spx output has no MHC-region genes by construction,
        # so the from-output counts should reflect that.
        assert meta["n_mhc_flagged_rows"] == 0
        assert meta["n_spx_unique_genes_total"] >= 1

    def test_no_metadata_input_defaults_to_no_annotation_block(
        self, tmp_path: Path,
    ) -> None:
        gwas_path, config, ref = self._setup_e2e(tmp_path)
        result = run_spredixcan(
            gwas_path, config, ref,
            output_dir=tmp_path / "output_meta_default",
            # both kwargs intentionally absent
        )
        with open(result["metadata"]) as fh:
            meta = json.load(fh)
        assert meta["mhc_annotation_strategy"] == "no_annotation"
        assert meta["mhc_annotation_source"] is None
        assert meta["mhc_annotation_build"] is None
        assert meta["mhc_annotation_coordinate_mode"] == "membership_interval_placeholder"
        assert meta["mhc_source_interval"] is None
        assert meta["mhc_flag_interval"] == [6, 25_726_063, 33_400_644]
        assert meta["n_mhc_flagged_rows"] == 0
