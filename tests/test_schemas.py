"""Tests for repogen.data.schemas."""

from __future__ import annotations

import pandas as pd
import pytest

from repogen.data.schemas import (
    DiseaseSignatureMetaRow,
    DiseaseSignaturePerTissueRow,
    GWASMetadata,
    NegativeCorrelationResultRow,
    NegativeCorrelationSummaryRow,
    StandardizedGWASRow,
    DrugTargetRecordRow,
    validate_dataframe,
)


class TestStandardizedGWASRow:
    """Tests for the StandardizedGWAS Pydantic model."""

    def test_valid_row(self) -> None:
        row = StandardizedGWASRow(
            SNP="rs12345",
            VARIANT_ID="1:100000:A:G",
            CHR=1, POS=100000,
            A1="A", A2="G",
            BETA=0.05, SE=0.01,
            P=1e-8, N=50000,
        )
        assert row.CHR == 1
        assert row.SNP == "rs12345"

    def test_optional_fields_default_none(self) -> None:
        row = StandardizedGWASRow(
            VARIANT_ID="1:100000:A:G",
            CHR=1, POS=100000,
            A1="A", A2="G",
            BETA=0.05, SE=0.01,
            P=1e-8, N=50000,
        )
        assert row.MAF is None
        assert row.INFO is None


class TestGWASMetadata:
    """Tests for the GWASMetadata model."""

    def test_valid_metadata(self) -> None:
        meta = GWASMetadata(
            genome_build="GRCh37",
            trait="Depression",
            trait_type="case_control",
        )
        assert meta.genome_build == "GRCh37"

    def test_invalid_build_rejected(self) -> None:
        with pytest.raises(Exception):
            GWASMetadata(genome_build="hg19")


class TestDrugTargetRecordRow:
    """Tests for the DrugTargetRecord Pydantic model."""

    def test_valid_row(self) -> None:
        row = DrugTargetRecordRow(
            drug_name="Fluoxetine",
            drug_chembl_id="CHEMBL41",
            gene_symbol="SLC6A4",
            gene_ensembl_id="ENSG00000108576",
            gene_uniprot_id="P31645",
            gene_entrez_id=6532,
            interaction_type="inhibitor",
            max_phase=4,
            source="chembl",
            confidence="high",
        )
        assert row.drug_name == "Fluoxetine"
        assert row.max_phase == 4


class TestValidateDataframe:
    """Tests for the validate_dataframe function."""

    def test_valid_gwas_dataframe(self) -> None:
        df = pd.DataFrame({
            "SNP": ["rs1"],
            "VARIANT_ID": ["1:100:A:G"],
            "CHR": [1],
            "POS": [100],
            "A1": ["A"],
            "A2": ["G"],
            "BETA": [0.1],
            "SE": [0.05],
            "P": [0.001],
            "N": [1000],
        })
        errors = validate_dataframe(df, "StandardizedGWAS")
        assert errors == []

    def test_missing_required_column(self) -> None:
        df = pd.DataFrame({"SNP": ["rs1"], "CHR": [1]})
        errors = validate_dataframe(df, "StandardizedGWAS")
        assert len(errors) > 0
        assert any("VARIANT_ID" in e for e in errors)

    def test_wrong_dtype(self) -> None:
        df = pd.DataFrame({
            "SNP": ["rs1"],
            "VARIANT_ID": ["1:100:A:G"],
            "CHR": ["one"],
            "POS": [100],
            "A1": ["A"],
            "A2": ["G"],
            "BETA": [0.1],
            "SE": [0.05],
            "P": [0.001],
            "N": [1000],
        })
        errors = validate_dataframe(df, "StandardizedGWAS")
        assert any("CHR" in e for e in errors)

    def test_unknown_schema_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schema"):
            validate_dataframe(pd.DataFrame(), "FakeSchema")

    def test_disease_signature_per_tissue_valid(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG00000141510"],
            "gene_symbol": ["TP53"],
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

    def test_disease_signature_per_tissue_missing_column(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG1"],
            "gene_symbol": ["G1"],
        })
        errors = validate_dataframe(df, "DiseaseSignaturePerTissue")
        assert len(errors) > 0
        assert any("tissue" in e for e in errors)

    def test_disease_signature_meta_valid(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG00000141510"],
            "gene_symbol": ["TP53"],
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

    def test_disease_signature_meta_missing_column(self) -> None:
        df = pd.DataFrame({
            "gene_ensembl_id": ["ENSG1"],
            "meta_zscore": [3.5],
        })
        errors = validate_dataframe(df, "DiseaseSignatureMeta")
        assert len(errors) > 0
        assert any("meta_pvalue" in e for e in errors)

    def test_valid_drug_target_dataframe(self) -> None:
        df = pd.DataFrame({
            "drug_name": ["Aspirin"],
            "drug_chembl_id": ["CHEMBL25"],
            "gene_symbol": ["PTGS2"],
            "gene_ensembl_id": ["ENSG00000073756"],
            "gene_uniprot_id": ["P35354"],
            "gene_entrez_id": [5743],
            "interaction_type": ["inhibitor"],
            "max_phase": [4],
            "source": ["chembl"],
            "confidence": ["high"],
        })
        errors = validate_dataframe(df, "DrugTargetRecord")
        assert errors == []

    def test_negative_correlation_result_valid(self) -> None:
        df = pd.DataFrame({
            "drug_name": ["Fluoxetine"],
            "tissue": ["Brain_Cortex"],
            "spearman_rho": [-0.35],
            "spearman_pvalue": [0.001],
            "fdr_global": [0.02],
            "n_tissues_nominal": [3],
            "n_overlapping_genes": [250],
            "overlap_fraction_disease": [0.65],
            "overlap_fraction_drug": [0.82],
            "match_confidence": ["inchikey"],
            "lincs_pert_id": ["BRD-K12345678"],
            "n_profiles_aggregated": [5],
            "direction": ["reversal"],
        })
        errors = validate_dataframe(df, "NegativeCorrelationResult")
        assert errors == []

    def test_negative_correlation_result_missing_column(self) -> None:
        df = pd.DataFrame({
            "drug_name": ["Fluoxetine"],
            "tissue": ["Brain_Cortex"],
        })
        errors = validate_dataframe(df, "NegativeCorrelationResult")
        assert len(errors) > 0
        assert any("spearman_rho" in e for e in errors)

    def test_negative_correlation_summary_valid(self) -> None:
        df = pd.DataFrame({
            "drug_name": ["Fluoxetine"],
            "best_spearman_rho": [-0.42],
            "best_tissue": ["Brain_Cortex"],
            "n_tissues_fdr_significant": [2],
            "n_tissues_nominal": [5],
        })
        errors = validate_dataframe(df, "NegativeCorrelationSummary")
        assert errors == []

    def test_negative_correlation_summary_missing_column(self) -> None:
        df = pd.DataFrame({
            "drug_name": ["Fluoxetine"],
            "best_spearman_rho": [-0.42],
        })
        errors = validate_dataframe(df, "NegativeCorrelationSummary")
        assert len(errors) > 0
        assert any("best_tissue" in e for e in errors)
