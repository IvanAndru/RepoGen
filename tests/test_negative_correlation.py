"""Tests for repogen.analysis.negative_correlation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from repogen.analysis.negative_correlation import (
    _apply_fdr_and_aggregate,
    _atomic_write_json,
    _build_summary,
    _collect_string_values,
    _directional_pvalue,
    _lambda_gc,
    _run_calibration,
    aggregate_drug_metadata,
    compute_spearman,
    compute_top_contributing_genes,
    compute_xsum,
    filter_drug_signatures,
    load_lincs_gene_info,
    prepare_disease_signatures,
    run_negative_correlation,
)
from repogen.config.schema import (
    NegativeCorrelationConfig,
    PermutationCalibrationConfig,
    PipelineConfig,
)
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_disease_signature():
    """Small disease signature: 3 tissues, 20 genes each."""
    rng = np.random.default_rng(42)
    tissues = ["Brain_Cortex", "Brain_Hippocampus", "Brain_Cerebellum"]
    rows = []
    for tissue in tissues:
        for i in range(20):
            rows.append({
                "gene_ensembl_id": f"ENSG{i:011d}",
                "gene_symbol": f"GENE{i}",
                "gene_entrez_id": 1000 + i,
                "tissue": tissue,
                "zscore": rng.normal(0, 2),
                "pvalue": rng.uniform(0, 1),
                "effect_size": rng.normal(0, 0.5),
                "se": abs(rng.normal(0.1, 0.05)),
                "n_snps_used": rng.integers(5, 50),
                "n_snps_in_model": rng.integers(10, 100),
                "pred_perf_r2": rng.uniform(0, 0.5),
                "pred_perf_pval": rng.uniform(0, 0.1),
                "mhc_flag": i == 15,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_drug_signatures():
    """3 drugs with 15-gene signatures overlapping the disease genes."""
    rng = np.random.default_rng(123)
    rows = []
    for drug_idx in range(3):
        gene_ids = list(range(1000, 1015))
        z_scores = rng.normal(0, 2, size=15).tolist()
        rows.append({
            "drug_name": f"Drug_{drug_idx}",
            "drug_inchikey": f"INCHIKEY{drug_idx:03d}",
            "drug_chembl_id": f"CHEMBL{drug_idx}",
            "lincs_pert_id": f"BRD-K{drug_idx:08d}",
            "n_profiles_aggregated": 5,
            "cell_lines": ["A549", "MCF7"],
            "doses": ["10um"],
            "time_points": ["24h"],
            "gene_ids": gene_ids,
            "z_scores": z_scores,
            "match_confidence": "inchikey" if drug_idx < 2 else "name",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_lincs_gene_info(tmp_path):
    """Gene info TSV with 20 genes, 15 landmark."""
    rows = []
    for i in range(20):
        rows.append({
            "entrez_id": 1000 + i,
            "gene_symbol": f"GENE{i}",
            "is_landmark": i < 15,
            "is_bing": 15 <= i < 18,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "lincs_gene_info.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def synthetic_drug_targets():
    """DrugTargetRecord for enrichment tests."""
    return pd.DataFrame([
        {
            "drug_name": "Drug_0",
            "drug_chembl_id": "CHEMBL0",
            "drug_inchikey": "INCHIKEY000",
            "drug_pubchem_cid": "CID123",
            "gene_symbol": "SLC6A4",
            "gene_ensembl_id": "ENSG00000108576",
            "gene_uniprot_id": "P31645",
            "gene_entrez_id": 6532,
            "interaction_type": "inhibitor",
            "mechanism_of_action": "Serotonin reuptake inhibitor",
            "max_phase": 4,
            "atc_codes": ["N06AB03"],
            "indication_mesh": ["Depressive Disorder"],
            "molecule_type": "small_molecule",
            "source": "chembl",
            "confidence": "high",
        },
        {
            "drug_name": "Drug_0",
            "drug_chembl_id": "CHEMBL0",
            "drug_inchikey": "INCHIKEY000",
            "drug_pubchem_cid": "CID123",
            "gene_symbol": "HTR2A",
            "gene_ensembl_id": "ENSG00000102468",
            "gene_uniprot_id": "P28223",
            "gene_entrez_id": 3356,
            "interaction_type": "antagonist",
            "mechanism_of_action": None,
            "max_phase": 4,
            "atc_codes": ["N06AB03"],
            "indication_mesh": ["Anxiety Disorders"],
            "molecule_type": "small_molecule",
            "source": "chembl",
            "confidence": "high",
        },
    ])


@pytest.fixture
def minimal_pipeline_config(tmp_path, synthetic_lincs_gene_info):
    """Minimal PipelineConfig for integration tests."""
    return PipelineConfig(
        study={"name": "test_study", "gwas_input": str(tmp_path / "dummy.gwas")},
        negative_correlation={
            "min_overlapping_genes": 10,
            "lincs_gene_info_path": str(synthetic_lincs_gene_info),
            "xsum_top_n": 10,
        },
        output_dir=str(tmp_path / "results"),
    )


# ---------------------------------------------------------------------------
# Unit tests: load_lincs_gene_info
# ---------------------------------------------------------------------------


class TestLoadLincsGeneInfo:
    def test_landmark(self, synthetic_lincs_gene_info):
        result = load_lincs_gene_info(synthetic_lincs_gene_info, "landmark")
        assert len(result) == 15
        assert all(1000 <= g <= 1014 for g in result)

    def test_landmark_bing(self, synthetic_lincs_gene_info):
        result = load_lincs_gene_info(synthetic_lincs_gene_info, "landmark_bing")
        assert len(result) == 18
        assert 1015 in result
        assert 1017 in result

    def test_all(self, synthetic_lincs_gene_info):
        result = load_lincs_gene_info(synthetic_lincs_gene_info, "all")
        assert len(result) == 20


# ---------------------------------------------------------------------------
# Unit tests: filter_drug_signatures
# ---------------------------------------------------------------------------


class TestFilterDrugSignatures:
    def test_match_confidence_threshold(self, synthetic_drug_signatures):
        allowed = set(range(1000, 1015))
        dicts, df = filter_drug_signatures(
            synthetic_drug_signatures, allowed, "pubchem_cid", 1,
        )
        assert "BRD-K00000002" not in dicts
        assert "BRD-K00000000" in dicts
        assert "BRD-K00000001" in dicts

    def test_min_profiles(self, synthetic_drug_signatures):
        allowed = set(range(1000, 1015))
        dicts, df = filter_drug_signatures(
            synthetic_drug_signatures, allowed, "name", 10,
        )
        assert len(dicts) == 0

    def test_gene_set_filtering(self, synthetic_drug_signatures):
        allowed = {1000, 1001, 1002}
        dicts, df = filter_drug_signatures(
            synthetic_drug_signatures, allowed, "name", 1,
        )
        for pert_id, zdict in dicts.items():
            assert all(g in allowed for g in zdict.keys())


# ---------------------------------------------------------------------------
# Unit tests: prepare_disease_signatures
# ---------------------------------------------------------------------------


class TestPrepareDiseaseSignatures:
    def test_null_entrez_recovery(self, synthetic_disease_signature):
        df = synthetic_disease_signature.copy()
        df.loc[df["gene_entrez_id"] == 1005, "gene_entrez_id"] = pd.NA

        mock_converter = MagicMock()
        mock_converter.convert.return_value = {"ENSG00000000005": "1005"}

        result = prepare_disease_signatures(df, exclude_mhc=False, gene_id_converter=mock_converter)
        for tissue, zdict in result.items():
            assert 1005 in zdict

    def test_null_entrez_drop(self, synthetic_disease_signature):
        df = synthetic_disease_signature.copy()
        df.loc[df["gene_entrez_id"] == 1005, "gene_entrez_id"] = pd.NA

        result = prepare_disease_signatures(df, exclude_mhc=False, gene_id_converter=None)
        for tissue, zdict in result.items():
            assert 1005 not in zdict

    def test_mhc_exclusion(self, synthetic_disease_signature):
        result_with = prepare_disease_signatures(
            synthetic_disease_signature, exclude_mhc=True,
        )
        result_without = prepare_disease_signatures(
            synthetic_disease_signature, exclude_mhc=False,
        )
        for tissue in result_with:
            assert 1015 not in result_with[tissue]
            assert 1015 in result_without[tissue]


# ---------------------------------------------------------------------------
# Unit tests: compute_spearman
# ---------------------------------------------------------------------------


class TestComputeSpearman:
    def test_happy_path(self):
        disease = {i: float(i) for i in range(20)}
        drug = {i: float(i) * 0.5 + 1 for i in range(20)}
        result = compute_spearman(disease, drug, min_overlap=5)
        assert result is not None
        rho, pvalue, n_overlap, genes = result
        assert n_overlap == 20
        assert rho > 0.9

    def test_insufficient_overlap(self):
        disease = {1: 1.0, 2: 2.0}
        drug = {1: 0.5, 3: 1.5}
        result = compute_spearman(disease, drug, min_overlap=5)
        assert result is None

    def test_perfect_anticorrelation(self):
        disease = {i: float(i) for i in range(20)}
        drug = {i: -float(i) for i in range(20)}
        result = compute_spearman(disease, drug, min_overlap=5)
        assert result is not None
        rho = result[0]
        assert rho < -0.99

    def test_identical_signatures(self):
        disease = {i: float(i) for i in range(20)}
        drug = {i: float(i) for i in range(20)}
        result = compute_spearman(disease, drug, min_overlap=5)
        assert result is not None
        rho = result[0]
        assert rho > 0.99


# ---------------------------------------------------------------------------
# Unit tests: compute_xsum
# ---------------------------------------------------------------------------


class TestComputeXsum:
    def test_happy_path(self):
        disease = {i: float(i) - 5 for i in range(10)}
        drug = {i: float(i) for i in range(10)}
        overlapping = list(range(10))
        score, pval = compute_xsum(disease, drug, overlapping, top_n=5)
        assert score is not None
        assert isinstance(score, float)

    def test_insufficient_overlap(self):
        disease = {i: float(i) for i in range(5)}
        drug = {i: float(i) for i in range(5)}
        overlapping = list(range(5))
        score, pval = compute_xsum(disease, drug, overlapping, top_n=10)
        assert score is None
        assert pval is None

    def test_no_permutation(self):
        disease = {i: float(i) for i in range(10)}
        drug = {i: float(i) for i in range(10)}
        overlapping = list(range(10))
        score, pval = compute_xsum(disease, drug, overlapping, top_n=5, n_permutations=0)
        assert score is not None
        assert pval is None

    def test_with_permutation(self):
        rng = np.random.default_rng(42)
        disease = {i: rng.normal() for i in range(50)}
        drug = {i: rng.normal() for i in range(50)}
        overlapping = list(range(50))
        score, pval = compute_xsum(disease, drug, overlapping, top_n=20, n_permutations=100)
        assert score is not None
        assert pval is not None
        assert 0 <= pval <= 1


# ---------------------------------------------------------------------------
# Unit tests: aggregate_drug_metadata
# ---------------------------------------------------------------------------


class TestAggregateDrugMetadata:
    def test_basic_aggregation(self, synthetic_drug_targets):
        result = aggregate_drug_metadata(synthetic_drug_targets)
        assert "by_inchikey" in result
        assert "by_chembl_id" in result

        meta = result["by_chembl_id"]["CHEMBL0"]
        assert meta["clinical_phase"] == 4
        assert meta["mechanism_of_action"] == "Serotonin reuptake inhibitor"
        assert sorted(meta["known_targets"]) == ["HTR2A", "SLC6A4"]
        assert "Depressive Disorder" in meta["known_indications"]
        assert "Anxiety Disorders" in meta["known_indications"]
        assert "N06AB03" in meta["atc_codes"]

    def test_no_match(self):
        result = aggregate_drug_metadata(pd.DataFrame())
        assert result["by_inchikey"] == {}
        assert result["by_chembl_id"] == {}

    def test_ndarray_atc_and_indication(self):
        """Regression: Parquet round-trip stores lists as np.ndarray."""
        df = pd.DataFrame([
            {
                "drug_name": "TestDrug",
                "drug_chembl_id": "CHEMBL999",
                "drug_inchikey": "TESTKEY",
                "gene_symbol": "TP53",
                "max_phase": 3,
                "mechanism_of_action": "kinase inhibitor",
                "atc_codes": np.array(["L01XE01", "L01XE02"]),
                "indication_mesh": np.array(["Leukemia", "Lymphoma"]),
            },
        ])
        result = aggregate_drug_metadata(df)
        meta = result["by_chembl_id"]["CHEMBL999"]
        assert sorted(meta["atc_codes"]) == ["L01XE01", "L01XE02"]
        assert sorted(meta["known_indications"]) == ["Leukemia", "Lymphoma"]

    def test_mixed_containers_dedup(self):
        """Handles list, ndarray, tuple, and bare str; deduplicates."""
        df = pd.DataFrame([
            {
                "drug_name": "MixDrug",
                "drug_chembl_id": "CHEMBL888",
                "drug_inchikey": "MIXKEY",
                "gene_symbol": "BRCA1",
                "max_phase": 2,
                "mechanism_of_action": None,
                "atc_codes": ["A01AA01"],
                "indication_mesh": np.array(["Hypertension"]),
            },
            {
                "drug_name": "MixDrug",
                "drug_chembl_id": "CHEMBL888",
                "drug_inchikey": "MIXKEY",
                "gene_symbol": "BRCA2",
                "max_phase": 2,
                "mechanism_of_action": None,
                "atc_codes": np.array(["A01AA01", "B02BD03"]),
                "indication_mesh": "Hypertension",
            },
        ])
        result = aggregate_drug_metadata(df)
        meta = result["by_chembl_id"]["CHEMBL888"]
        assert sorted(meta["atc_codes"]) == ["A01AA01", "B02BD03"]
        assert meta["known_indications"] == ["Hypertension"]

    def test_parquet_roundtrip_preserves_metadata(self, tmp_path):
        """End-to-end: write drug_targets to Parquet, read back, aggregate."""
        df = pd.DataFrame([
            {
                "drug_name": "RoundTrip",
                "drug_chembl_id": "CHEMBL777",
                "drug_inchikey": "RTKEY",
                "gene_symbol": "EGFR",
                "max_phase": 4,
                "mechanism_of_action": "EGFR inhibitor",
                "atc_codes": ["L01XE03"],
                "indication_mesh": ["Lung Neoplasms", "Glioblastoma"],
            },
        ])
        pq_path = tmp_path / "dt.parquet"
        df.to_parquet(pq_path)
        reloaded = pd.read_parquet(pq_path)

        result = aggregate_drug_metadata(reloaded)
        meta = result["by_chembl_id"]["CHEMBL777"]
        assert "L01XE03" in meta["atc_codes"]
        assert "Lung Neoplasms" in meta["known_indications"]
        assert "Glioblastoma" in meta["known_indications"]


class TestCollectStringValues:
    """Unit tests for the _collect_string_values helper."""

    def test_ndarray_input(self):
        vals = [np.array(["A", "B"]), np.array(["C"])]
        assert _collect_string_values(vals) == ["A", "B", "C"]

    def test_list_input(self):
        vals = [["X", "Y"], ["Z"]]
        assert _collect_string_values(vals) == ["X", "Y", "Z"]

    def test_str_input(self):
        vals = ["alpha", "beta"]
        assert _collect_string_values(vals) == ["alpha", "beta"]

    def test_mixed_dedup(self):
        vals = [np.array(["A", "B"]), ["B", "C"], "A"]
        assert _collect_string_values(vals) == ["A", "B", "C"]

    def test_empty(self):
        assert _collect_string_values([]) == []


# ---------------------------------------------------------------------------
# Unit tests: compute_top_contributing_genes
# ---------------------------------------------------------------------------


class TestTopContributingGenes:
    def test_returns_correct_symbols(self, synthetic_disease_signature):
        disease = {1000 + i: float(i) * ((-1) ** i) for i in range(15)}
        drug = {1000 + i: float(i) * 0.5 for i in range(15)}
        overlapping = list(range(1000, 1015))

        result = compute_top_contributing_genes(
            disease, drug, overlapping,
            synthetic_disease_signature, "Brain_Cortex", n_top=5,
        )
        assert len(result) == 5
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRunNegativeCorrelation:
    def test_end_to_end(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info, minimal_pipeline_config,
    ):
        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        result = run_negative_correlation(
            config=minimal_pipeline_config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )

        assert len(result) > 0
        assert (output_dir / "per_tissue_results.parquet").exists()
        assert (output_dir / "per_tissue_results.csv").exists()
        assert (output_dir / "drug_summary.parquet").exists()
        assert (output_dir / "drug_summary.csv").exists()
        assert (output_dir / "metadata.json").exists()

        with open(output_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["module"] == "negative_correlation"
        assert meta["summary"]["n_drugs_tested"] > 0

    def test_mhc_sensitivity_produced(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info,
    ):
        config = PipelineConfig(
            study={"name": "test", "gwas_input": str(tmp_path / "d.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "exclude_mhc": False,
                "xsum_top_n": 10,
            },
            output_dir=str(tmp_path / "results"),
        )

        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        run_negative_correlation(
            config=config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )

        assert (output_dir / "sensitivity" / "mhc_excluded" / "per_tissue_results.parquet").exists()

    def test_mhc_sensitivity_skipped_when_excluded(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info,
    ):
        config = PipelineConfig(
            study={"name": "test", "gwas_input": str(tmp_path / "d.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "exclude_mhc": True,
                "xsum_top_n": 10,
            },
            output_dir=str(tmp_path / "results"),
        )

        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        run_negative_correlation(
            config=config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )

        assert not (output_dir / "sensitivity" / "mhc_excluded").exists()

    def test_gene_id_converter_fallback_recovers_null_entrez(
        self, tmp_path, synthetic_drug_signatures,
        synthetic_drug_targets, synthetic_lincs_gene_info,
    ):
        """Null Entrez IDs are recovered via GeneIDConverter when reference files exist."""
        rng = np.random.default_rng(42)
        tissues = ["Brain_Cortex", "Brain_Hippocampus"]
        rows = []
        for tissue in tissues:
            for i in range(20):
                entrez = 1000 + i
                if i == 5:
                    entrez = None
                rows.append({
                    "gene_ensembl_id": f"ENSG{i:011d}",
                    "gene_symbol": f"GENE{i}",
                    "gene_entrez_id": entrez,
                    "tissue": tissue,
                    "zscore": rng.normal(0, 2),
                    "pvalue": rng.uniform(0, 1),
                    "effect_size": rng.normal(0, 0.5),
                    "se": abs(rng.normal(0.1, 0.05)),
                    "n_snps_used": rng.integers(5, 50),
                    "n_snps_in_model": rng.integers(10, 100),
                    "pred_perf_r2": rng.uniform(0, 0.5),
                    "pred_perf_pval": rng.uniform(0, 0.1),
                    "mhc_flag": False,
                })
        disease_df = pd.DataFrame(rows)

        ensembl_to_name = tmp_path / "ensembl_to_name.tsv"
        name_to_ensembl = tmp_path / "name_to_ensembl.tsv"
        ensembl_to_name.write_text("ENSG00000000005\tGENE5\n")
        name_to_ensembl.write_text("GENE5\tENSG00000000005\n")

        ncbi_gene_info = tmp_path / "gene_info.gz"
        import gzip
        with gzip.open(ncbi_gene_info, "wt") as f:
            f.write("#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\tdbXrefs\tchromosome\tmap_location\tdescription\ttype_of_gene\tSymbol_from_nomenclature_authority\tFull_name_from_nomenclature_authority\tNomenclature_status\tOther_designations\tModification_date\tFeature_type\n")
            f.write(f"9606\t1005\tGENE5\t-\t-\tEnsembl:ENSG00000000005\t1\t1p1\tdesc\tprotein-coding\tGENE5\tGene Five\tO\t-\t20240101\t-\n")

        config = PipelineConfig(
            study={"name": "test", "gwas_input": str(tmp_path / "d.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "xsum_top_n": 10,
            },
            reference={
                "ensembl_to_name": str(ensembl_to_name),
                "name_to_ensembl": str(name_to_ensembl),
                "ncbi_gene_info": str(ncbi_gene_info),
            },
            output_dir=str(tmp_path / "results"),
        )

        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output"

        disease_df.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        result = run_negative_correlation(
            config=config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )
        assert len(result) > 0

    def test_empty_after_filters(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_targets, synthetic_lincs_gene_info,
    ):
        drug_sigs = pd.DataFrame([{
            "drug_name": "NoMatch",
            "drug_inchikey": "X",
            "drug_chembl_id": "CX",
            "lincs_pert_id": "BRD-X",
            "n_profiles_aggregated": 1,
            "cell_lines": [],
            "doses": [],
            "time_points": [],
            "gene_ids": [999999],
            "z_scores": [0.5],
            "match_confidence": "name",
        }])

        config = PipelineConfig(
            study={"name": "test", "gwas_input": str(tmp_path / "d.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "match_confidence_threshold": "inchikey",
                "xsum_top_n": 10,
            },
            output_dir=str(tmp_path / "results"),
        )

        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"

        synthetic_disease_signature.to_parquet(disease_path)
        drug_sigs.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        with pytest.raises(ValueError, match="No valid drug-tissue pairs"):
            run_negative_correlation(
                config=config,
                disease_signature_path=disease_path,
                drug_signatures_path=drug_sig_path,
                drug_targets_path=drug_targets_path,
                output_dir=tmp_path / "output",
            )

    def test_summary_contains_cell_lines(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info, minimal_pipeline_config,
    ):
        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        run_negative_correlation(
            config=minimal_pipeline_config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )

        summary = pd.read_parquet(output_dir / "drug_summary.parquet")
        assert "cell_lines" in summary.columns

    def test_parquet_preserves_native_list_columns(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info, minimal_pipeline_config,
    ):
        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        run_negative_correlation(
            config=minimal_pipeline_config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )

        results = pd.read_parquet(output_dir / "per_tissue_results.parquet")
        first_cell = results["cell_lines"].iloc[0]
        assert isinstance(first_cell, (list, np.ndarray)), (
            f"Parquet cell_lines should be a native list, got {type(first_cell)}"
        )


# ---------------------------------------------------------------------------
# FDR and cross-tissue aggregation tests
# ---------------------------------------------------------------------------


class TestFDRAndAggregation:
    def test_fdr_correction_global(self):
        """FDR is computed across all tissue×drug pairs, not per-tissue."""
        rows = []
        for tissue in ["T1", "T2"]:
            for drug_idx in range(5):
                p = 0.001 * (drug_idx + 1)
                rows.append({
                    "drug_name": f"D{drug_idx}",
                    "tissue": tissue,
                    "spearman_rho": -0.5,
                    "spearman_pvalue": p,
                    # Additive column - the production path
                    # always populates via _run_correlation_loop; test
                    # fixtures constructed by hand must provide it too.
                    "directional_pvalue": p / 2.0,
                    "fdr_global": np.nan,
                    "n_tissues_nominal": 0,
                    "lincs_pert_id": f"BRD{drug_idx}",
                    "n_overlapping_genes": 100,
                    "overlap_fraction_disease": 0.5,
                    "overlap_fraction_drug": 0.5,
                    "match_confidence": "inchikey",
                    "n_profiles_aggregated": 5,
                    "direction": "reversal",
                })
        df = pd.DataFrame(rows)
        result = _apply_fdr_and_aggregate(df, 0.05)
        assert not result["fdr_global"].isna().any()
        assert len(result) == 10

    def test_n_tissues_nominal_correct(self):
        rows = []
        for tissue in ["T1", "T2", "T3"]:
            p = 0.01 if tissue != "T3" else 0.1
            rows.append({
                "drug_name": "D0",
                "tissue": tissue,
                "spearman_rho": -0.5,
                "spearman_pvalue": p,
                # Additive column (see note above).
                "directional_pvalue": p / 2.0,
                "fdr_global": np.nan,
                "n_tissues_nominal": 0,
                "lincs_pert_id": "BRD0",
                "n_overlapping_genes": 100,
                "overlap_fraction_disease": 0.5,
                "overlap_fraction_drug": 0.5,
                "match_confidence": "inchikey",
                "n_profiles_aggregated": 5,
                "direction": "reversal",
            })
        df = pd.DataFrame(rows)
        result = _apply_fdr_and_aggregate(df, 0.05)
        assert (result["n_tissues_nominal"] == 2).all()


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_gene_set_mode_invalid(self):
        with pytest.raises(ValidationError):
            NegativeCorrelationConfig(gene_set_mode="invalid")

    def test_match_confidence_invalid(self):
        with pytest.raises(ValidationError):
            NegativeCorrelationConfig(match_confidence_threshold="exact")

    def test_aggregation_mode_invalid(self):
        with pytest.raises(ValidationError):
            NegativeCorrelationConfig(aggregation_mode="per_condition")

    def test_deprecated_fields_removed(self):
        assert "predixcan_models_dir" not in NegativeCorrelationConfig.model_fields
        assert "predixcan_covariances_dir" not in NegativeCorrelationConfig.model_fields

    def test_valid_defaults(self):
        nc = NegativeCorrelationConfig()
        assert nc.gene_set_mode == "landmark"
        assert nc.match_confidence_threshold == "pubchem_cid"
        assert nc.exclude_mhc is False
        assert nc.min_profiles_aggregated == 3
        assert nc.aggregation_mode == "consensus"
        assert nc.xsum_top_n == 200
        assert nc.xsum_permutations == 0

    def test_gene_set_mode_valid_values(self):
        for mode in ("landmark", "landmark_bing", "all"):
            nc = NegativeCorrelationConfig(gene_set_mode=mode)
            assert nc.gene_set_mode == mode

    def test_match_confidence_valid_values(self):
        for conf in ("inchikey", "pubchem_cid", "name"):
            nc = NegativeCorrelationConfig(match_confidence_threshold=conf)
            assert nc.match_confidence_threshold == conf

    def test_aggregation_mode_best_dose_rejected(self):
        with pytest.raises(ValidationError, match="not yet wired"):
            NegativeCorrelationConfig(aggregation_mode="best_dose")

    def test_correlation_method_non_spearman_rejected(self):
        with pytest.raises(ValidationError, match="spearman"):
            NegativeCorrelationConfig(correlation_method="pearson")

    def test_xsum_top_n_sweep_rejected(self):
        with pytest.raises(ValidationError, match="not yet implemented"):
            NegativeCorrelationConfig(xsum_top_n_sweep=[50, 100, 200])


class TestAutoResolvePath:
    def test_default_lincs_path_uses_canonical_location(self):
        nc = NegativeCorrelationConfig()
        assert nc.lincs_gene_info_path is None


# ---------------------------------------------------------------------------
# Directional inference, per-tissue FDR, calibration sidecar
# ---------------------------------------------------------------------------


class TestDirectionalPvalue:
    """One-sided reversal p-value transform."""

    def test_negative_rho_halves_two_sided(self) -> None:
        assert _directional_pvalue(-0.5, 0.04) == pytest.approx(0.02)

    def test_positive_rho_complement_halves(self) -> None:
        assert _directional_pvalue(0.5, 0.04) == pytest.approx(0.98)

    def test_zero_rho_returns_half(self) -> None:
        assert _directional_pvalue(0.0, 0.04) == 0.5

    def test_nan_rho_propagates(self) -> None:
        assert np.isnan(_directional_pvalue(float("nan"), 0.04))

    def test_nan_pvalue_propagates(self) -> None:
        assert np.isnan(_directional_pvalue(-0.5, float("nan")))


def _make_fdr_fixture_df() -> pd.DataFrame:
    """3 tissues × 10 drugs with controlled p-values for FDR tests."""
    rows = []
    for t_idx, tissue in enumerate(["T1", "T2", "T3"]):
        for d_idx in range(10):
            p = 0.001 * (d_idx + 1) + 0.0001 * t_idx
            rho = -0.3 if d_idx % 2 == 0 else 0.2
            rows.append({
                "drug_name": f"D{d_idx}",
                "tissue": tissue,
                "spearman_rho": rho,
                "spearman_pvalue": p,
                "directional_pvalue": _directional_pvalue(rho, p),
                "fdr_global": np.nan,
                "n_tissues_nominal": 0,
                "lincs_pert_id": f"BRD{d_idx}",
                "n_overlapping_genes": 100,
                "overlap_fraction_disease": 0.5,
                "overlap_fraction_drug": 0.5,
                "match_confidence": "inchikey",
                "n_profiles_aggregated": 5,
                "direction": "reversal" if rho < 0 else "mimicry",
            })
    return pd.DataFrame(rows)


class TestDirectionalFDR:
    """FDR columns and byte-identity lock on fdr_global."""

    def test_fdr_global_byte_identical_to_plain_bh(self) -> None:
        df = _make_fdr_fixture_df()
        _, plain_bh_q, _, _ = multipletests(
            df["spearman_pvalue"].values, method="fdr_bh",
        )
        result = _apply_fdr_and_aggregate(df.copy(), 0.05)
        np.testing.assert_array_equal(result["fdr_global"].values, plain_bh_q)

    def test_directional_fdr_global_matches_multipletests(self) -> None:
        df = _make_fdr_fixture_df()
        _, expected, _, _ = multipletests(
            df["directional_pvalue"].values, method="fdr_bh",
        )
        result = _apply_fdr_and_aggregate(df.copy(), 0.05)
        np.testing.assert_array_equal(
            result["directional_fdr_global"].values, expected,
        )

    def test_spearman_fdr_per_tissue_matches_hand_computed_bh(self) -> None:
        df = _make_fdr_fixture_df()
        result = _apply_fdr_and_aggregate(df.copy(), 0.05)
        for tissue, group in df.groupby("tissue"):
            _, expected, _, _ = multipletests(
                group["spearman_pvalue"].values, method="fdr_bh",
            )
            got = result.loc[result["tissue"] == tissue, "spearman_fdr_per_tissue"]
            np.testing.assert_array_equal(got.values, expected)

    def test_directional_fdr_per_tissue_populated(self) -> None:
        df = _make_fdr_fixture_df()
        result = _apply_fdr_and_aggregate(df.copy(), 0.05)
        assert not result["directional_fdr_per_tissue"].isna().any()

    def test_single_row_tissue_group_does_not_crash(self) -> None:
        df = _make_fdr_fixture_df().iloc[:1].copy()
        result = _apply_fdr_and_aggregate(df, 0.05)
        assert len(result) == 1
        assert not np.isnan(result["fdr_global"].iloc[0])

    def test_per_tissue_bh_masks_nan_pvalues(self) -> None:
        """A single NaN p-value in a
        tissue must not spoil the per-tissue FDR column for every other
        drug in that tissue.  ``multipletests([0.01, NaN, 0.2])`` returns
        all-NaN q-values, so ``_bh_fdr_per_group`` must mask NaN before
        calling BH.
        """
        df = _make_fdr_fixture_df()
        # Inject one NaN directional_pvalue in tissue T1 (drug BRD3).
        nan_mask = (df["tissue"] == "T1") & (df["lincs_pert_id"] == "BRD3")
        df.loc[nan_mask, "directional_pvalue"] = np.nan
        # Also inject one NaN spearman_pvalue in T2 to lock the two-sided
        # per-tissue path against the same regression.
        nan_mask_2 = (df["tissue"] == "T2") & (df["lincs_pert_id"] == "BRD7")
        df.loc[nan_mask_2, "spearman_pvalue"] = np.nan

        result = _apply_fdr_and_aggregate(df.copy(), 0.05)

        # The row with NaN directional p keeps NaN in the per-tissue q...
        t1_nan_row = result.loc[nan_mask, "directional_fdr_per_tissue"]
        assert t1_nan_row.isna().all()
        # ...but the OTHER drugs in T1 must have valid per-tissue q-values
        # (not all-NaN because of the single invalid row).
        t1_valid = result.loc[
            (result["tissue"] == "T1") & (~nan_mask), "directional_fdr_per_tissue"
        ]
        assert not t1_valid.isna().any()

        # Same for two-sided per-tissue BH in T2.
        t2_nan_row = result.loc[nan_mask_2, "spearman_fdr_per_tissue"]
        assert t2_nan_row.isna().all()
        t2_valid = result.loc[
            (result["tissue"] == "T2") & (~nan_mask_2), "spearman_fdr_per_tissue"
        ]
        assert not t2_valid.isna().any()

        # T3 has no NaNs - must be entirely valid on both columns.
        t3 = result[result["tissue"] == "T3"]
        assert not t3["spearman_fdr_per_tissue"].isna().any()
        assert not t3["directional_fdr_per_tissue"].isna().any()


class TestDirectionalSummary:
    """Additive summary columns."""

    def test_directional_counts_and_best_fields(self) -> None:
        df = _make_fdr_fixture_df()
        result = _apply_fdr_and_aggregate(df.copy(), 0.05)
        summary = _build_summary(result, 0.05)

        assert "n_tissues_directional_nominal" in summary.columns
        assert "n_tissues_directional_fdr_significant" in summary.columns
        assert "best_directional_pvalue" in summary.columns
        assert "best_directional_fdr_per_tissue" in summary.columns

        # Columns that predate the directional track are unchanged
        assert "n_tissues_fdr_significant" in summary.columns
        assert "n_tissues_nominal" in summary.columns
        assert "best_spearman_rho" in summary.columns

        drug0 = summary[summary["lincs_pert_id"] == "BRD0"].iloc[0]
        sub = result[result["lincs_pert_id"] == "BRD0"]
        expected_nominal = int((sub["directional_pvalue"] < 0.05).sum())
        assert drug0["n_tissues_directional_nominal"] == expected_nominal
        assert drug0["best_directional_pvalue"] == pytest.approx(
            sub["directional_pvalue"].min(),
        )


class TestLambdaGc:
    """Genomic-control lambda helper."""

    def test_uniform_null_lambda_near_one(self) -> None:
        rng = np.random.default_rng(0)
        pvals = rng.uniform(0.01, 0.99, size=5000)
        out = _lambda_gc(pvals)
        assert out["lambda_gc_theoretical"] == pytest.approx(1.0, abs=0.15)
        assert out["n_pvals_used"] == 5000

    def test_inflated_null_lambda_above_one(self) -> None:
        from scipy.stats import chi2 as chi2_dist
        rng = np.random.default_rng(1)
        chi2_vals = rng.chisquare(df=1, size=2000) * 1.5
        pvals = chi2_dist.sf(chi2_vals, df=1)
        out = _lambda_gc(pvals)
        assert out["lambda_gc_theoretical"] == pytest.approx(1.5, abs=0.2)

    def test_neglog10_ratio_differs_from_lambda_under_inflation(self) -> None:
        from scipy.stats import chi2 as chi2_dist
        rng = np.random.default_rng(2)
        chi2_obs = rng.chisquare(df=1, size=2000) * 1.5
        p_obs = chi2_dist.sf(chi2_obs, df=1)
        chi2_null = rng.chisquare(df=1, size=5000)
        p_null = chi2_dist.sf(chi2_null, df=1)
        out = _lambda_gc(p_obs, p_null)
        assert out["lambda_gc_empirical"] is not None
        assert out["neglog10_ratio_heuristic"] is not None
        assert out["lambda_gc_empirical"] != pytest.approx(
            out["neglog10_ratio_heuristic"], abs=0.05,
        )

    def test_nan_and_zero_pvalues_handled(self) -> None:
        pvals = np.array([0.0, float("nan"), 0.05, 0.1])
        out = _lambda_gc(pvals)
        assert out["n_pvals_used"] == 3
        assert out["n_pvals_dropped_nan"] == 1
        assert np.isfinite(out["lambda_gc_theoretical"])


class TestXsumSeedBackwardCompat:
    """The default xsum_seed=42 reproduces the original byte-for-byte behaviour."""

    def test_default_seed_is_deterministic(self) -> None:
        disease = {i: float(i) for i in range(1000, 1020)}
        drug = {i: float((-1) ** i) for i in range(1000, 1020)}
        overlap = list(range(1000, 1020))
        a = compute_xsum(disease, drug, overlap, top_n=10, n_permutations=50)
        b = compute_xsum(
            disease, drug, overlap, top_n=10,
            n_permutations=50, xsum_seed=42,
        )
        assert a == b


class TestCalibrationConfigDefaults:
    """Config surface defaults."""

    def test_xsum_seed_default_42(self) -> None:
        nc = NegativeCorrelationConfig()
        assert nc.xsum_seed == 42

    def test_permutation_calibration_disabled_by_default(self) -> None:
        nc = NegativeCorrelationConfig()
        assert nc.permutation_calibration.enabled is False
        assert nc.permutation_calibration.n_permutations == 100
        assert nc.permutation_calibration.seed == 42


class TestDirectionalIntegration:
    """End-to-end directional-track integration against the synthetic fixture."""

    def test_directional_columns_present_in_parquet(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info, minimal_pipeline_config,
    ) -> None:
        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        run_negative_correlation(
            config=minimal_pipeline_config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )

        results = pd.read_parquet(output_dir / "per_tissue_results.parquet")
        for col in (
            "directional_pvalue",
            "directional_fdr_global",
            "spearman_fdr_per_tissue",
            "directional_fdr_per_tissue",
        ):
            assert col in results.columns
            assert not results[col].isna().all()

        summary = pd.read_parquet(output_dir / "drug_summary.parquet")
        for col in (
            "n_tissues_directional_nominal",
            "n_tissues_directional_fdr_significant",
            "best_directional_pvalue",
            "best_directional_fdr_per_tissue",
        ):
            assert col in summary.columns

        with open(output_dir / "metadata.json") as fh:
            meta = json.load(fh)
        assert "xsum_provenance" in meta
        assert meta["xsum_provenance"]["xsum_seed"] == 42
        assert meta["xsum_provenance"]["permutation_calibration"]["enabled"] is False
        assert not (output_dir / "calibration.json").exists()

    def test_calibration_sidecar_written_when_enabled(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info,
    ) -> None:
        config = PipelineConfig(
            study={"name": "test_study", "gwas_input": str(tmp_path / "dummy.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "xsum_top_n": 10,
                "permutation_calibration": {
                    "enabled": True,
                    "n_permutations": 10,
                    "seed": 7,
                },
            },
            output_dir=str(tmp_path / "results"),
        )

        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output_cal"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        run_negative_correlation(
            config=config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )

        cal_path = output_dir / "calibration.json"
        assert cal_path.exists()
        with open(cal_path) as fh:
            cal = json.load(fh)
        for key in (
            "n_permutations", "seed", "lambda_gc_theoretical",
            "lambda_gc_empirical", "lambda_gc_per_tissue", "notes",
        ):
            assert key in cal
        assert cal["n_permutations"] == 10
        assert cal["seed"] == 7
        assert np.isfinite(cal["lambda_gc_theoretical"])

    def test_calibration_failure_leaves_no_partial_outputs(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info,
    ) -> None:
        config = PipelineConfig(
            study={"name": "test_study", "gwas_input": str(tmp_path / "dummy.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "xsum_top_n": 10,
                "permutation_calibration": {"enabled": True, "n_permutations": 10},
            },
            output_dir=str(tmp_path / "results"),
        )

        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output_fail"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        with patch(
            "repogen.analysis.negative_correlation._run_calibration",
            side_effect=RuntimeError("calibration boom"),
        ):
            with pytest.raises(RuntimeError, match="calibration boom"):
                run_negative_correlation(
                    config=config,
                    disease_signature_path=disease_path,
                    drug_signatures_path=drug_sig_path,
                    drug_targets_path=drug_targets_path,
                    output_dir=output_dir,
                )

        assert not (output_dir / "per_tissue_results.parquet").exists()
        assert not (output_dir / "calibration.json").exists()

    def test_atomic_write_json_cleans_up_on_failure(self, tmp_path) -> None:
        target = tmp_path / "calibration.json"
        with patch("json.dump", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                _atomic_write_json(target, {"ok": True})
        assert not target.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_calibration_write_failure_rolls_back_primary_outputs(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info,
    ) -> None:
        """If the calibration sidecar
        WRITE fails after primary parquets are already on disk, the run
        must roll back every file it wrote this invocation so the user
        never sees complete-looking primary outputs for a run whose
        requested calibration sidecar could not be produced.
        """
        config = PipelineConfig(
            study={"name": "test_study", "gwas_input": str(tmp_path / "dummy.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "xsum_top_n": 10,
                "permutation_calibration": {"enabled": True, "n_permutations": 10},
            },
            output_dir=str(tmp_path / "results"),
        )

        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output_write_fail"

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        # Patch _atomic_write_json at its call site inside the module so
        # only the calibration-sidecar write raises, leaving the primary
        # parquets already on disk before rollback runs.
        with patch(
            "repogen.analysis.negative_correlation._atomic_write_json",
            side_effect=OSError("simulated disk full"),
        ):
            with pytest.raises(OSError, match="simulated disk full"):
                run_negative_correlation(
                    config=config,
                    disease_signature_path=disease_path,
                    drug_signatures_path=drug_sig_path,
                    drug_targets_path=drug_targets_path,
                    output_dir=output_dir,
                )

        # Every file this invocation wrote must be rolled back.
        for name in (
            "per_tissue_results.parquet",
            "drug_summary.parquet",
            "per_tissue_results.csv",
            "drug_summary.csv",
            "metadata.json",
            "calibration.json",
        ):
            assert not (output_dir / name).exists(), (
                f"{name} was not rolled back after calibration write failure"
            )
        # Sensitivity subdir contents (if produced) also rolled back.
        sens_dir = output_dir / "sensitivity" / "mhc_excluded"
        if sens_dir.exists():
            assert not (sens_dir / "per_tissue_results.parquet").exists()
            assert not (sens_dir / "drug_summary.parquet").exists()

    def test_calibration_write_failure_does_not_touch_prior_run_outputs(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info,
    ) -> None:
        """Rollback must ONLY unlink files written by the current
        invocation.  Pre-existing archived outputs from a prior run
        (e.g. a stale ``per_tissue_results.parquet``) that happen to sit
        in the same output directory must never be touched.
        """
        config = PipelineConfig(
            study={"name": "test_study", "gwas_input": str(tmp_path / "dummy.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "xsum_top_n": 10,
                "permutation_calibration": {"enabled": True, "n_permutations": 10},
            },
            output_dir=str(tmp_path / "results"),
        )

        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output_preexisting"
        output_dir.mkdir(parents=True, exist_ok=True)

        synthetic_disease_signature.to_parquet(disease_path)
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        # Simulate a pre-existing artifact from an unrelated prior run.
        preexisting = output_dir / "unrelated_artifact.txt"
        preexisting.write_text("pre-existing content that must survive rollback")

        with patch(
            "repogen.analysis.negative_correlation._atomic_write_json",
            side_effect=OSError("simulated disk full"),
        ):
            with pytest.raises(OSError, match="simulated disk full"):
                run_negative_correlation(
                    config=config,
                    disease_signature_path=disease_path,
                    drug_signatures_path=drug_sig_path,
                    drug_targets_path=drug_targets_path,
                    output_dir=output_dir,
                )

        assert preexisting.exists()
        assert preexisting.read_text() == "pre-existing content that must survive rollback"


# ---------------------------------------------------------------------------
# Composition propagation from drug_signatures into NC parquets
# ---------------------------------------------------------------------------


class TestCompositionPropagation:
    """5 composition columns must appear in per_tissue_results
    AND drug_summary parquets when the upstream drug_signatures parquet
    carries them.
    """

    def test_composition_columns_propagate_to_negative_correlation_outputs(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info,
    ) -> None:
        # Extend the shared synthetic_drug_signatures fixture with the
        # composition columns (as if extract_drug_signatures had written
        # them in a real run).
        drug_sigs = synthetic_drug_signatures.copy()
        n = len(drug_sigs)
        drug_sigs["n_profiles_total"] = [5] * n
        drug_sigs["n_profiles_neural"] = [1, 0, 2]
        drug_sigs["n_profiles_non_neural"] = [4, 5, 3]
        drug_sigs["n_profiles_unknown_cell_line"] = [0, 0, 0]
        drug_sigs["neural_fraction"] = [0.2, 0.0, 0.4]
        drug_sigs["neural_weight_fraction"] = [0.4286, 0.0, 0.6667]
        drug_sigs["cell_line_weighting_mode"] = ["neural_priority"] * n
        drug_sigs["neural_weight"] = [3.0] * n

        config = PipelineConfig(
            study={"name": "test_study", "gwas_input": str(tmp_path / "dummy.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "xsum_top_n": 10,
            },
            output_dir=str(tmp_path / "results"),
        )
        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output_r4"

        synthetic_disease_signature.to_parquet(disease_path)
        drug_sigs.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        run_negative_correlation(
            config=config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )

        per_tissue = pd.read_parquet(output_dir / "per_tissue_results.parquet")
        summary = pd.read_parquet(output_dir / "drug_summary.parquet")
        expected_composition_cols = {
            "n_profiles_total", "n_profiles_neural", "neural_fraction",
            "neural_weight_fraction", "cell_line_weighting_mode",
        }
        assert expected_composition_cols.issubset(set(per_tissue.columns)), (
            f"Missing composition columns from per_tissue_results: "
            f"{expected_composition_cols - set(per_tissue.columns)}"
        )
        assert expected_composition_cols.issubset(set(summary.columns)), (
            f"Missing composition columns from drug_summary: "
            f"{expected_composition_cols - set(summary.columns)}"
        )
        # Spot-check a value round-tripped correctly for one drug.
        row = per_tissue[per_tissue["lincs_pert_id"] == "BRD-K00000000"].iloc[0]
        assert row["n_profiles_neural"] == 1
        assert row["neural_fraction"] == pytest.approx(0.2)
        assert row["cell_line_weighting_mode"] == "neural_priority"

    def test_backward_compat_signatures_without_composition_still_produce_valid_nc(
        self, tmp_path, synthetic_disease_signature,
        synthetic_drug_signatures, synthetic_drug_targets,
        synthetic_lincs_gene_info,
    ) -> None:
        """Archived drug_signatures.parquet files without the composition
        columns must still produce a valid negative-correlation output
        (columns present as NaN, no crash).
        """
        config = PipelineConfig(
            study={"name": "test_study", "gwas_input": str(tmp_path / "dummy.gwas")},
            negative_correlation={
                "min_overlapping_genes": 10,
                "lincs_gene_info_path": str(synthetic_lincs_gene_info),
                "xsum_top_n": 10,
            },
            output_dir=str(tmp_path / "results"),
        )
        disease_path = tmp_path / "disease.parquet"
        drug_sig_path = tmp_path / "drug_sigs.parquet"
        drug_targets_path = tmp_path / "drug_targets.parquet"
        output_dir = tmp_path / "output_bc"

        synthetic_disease_signature.to_parquet(disease_path)
        # Note: synthetic_drug_signatures fixture does NOT include composition columns.
        synthetic_drug_signatures.to_parquet(drug_sig_path)
        synthetic_drug_targets.to_parquet(drug_targets_path)

        run_negative_correlation(
            config=config,
            disease_signature_path=disease_path,
            drug_signatures_path=drug_sig_path,
            drug_targets_path=drug_targets_path,
            output_dir=output_dir,
        )
        per_tissue = pd.read_parquet(output_dir / "per_tissue_results.parquet")
        # Composition columns should be present (populated as NaN because upstream
        # did not supply them).
        for col in ("n_profiles_total", "n_profiles_neural", "neural_fraction",
                    "neural_weight_fraction", "cell_line_weighting_mode"):
            assert col in per_tissue.columns
            assert per_tissue[col].isna().all()
