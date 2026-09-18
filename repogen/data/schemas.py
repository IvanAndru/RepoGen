"""Canonical data schemas for the RepoGen pipeline.

Defines the "contracts" between data modules and analysis modules.
Each schema is represented two ways:

1. Pydantic models for single-row validation and documentation.
2. Column-spec dicts for bulk DataFrame validation at runtime.

Modules produce DataFrames matching these schemas so consumers never
need format-specific logic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Schema 1: StandardizedGWAS
# ---------------------------------------------------------------------------


class StandardizedGWASRow(BaseModel):
    """Single-row representation of a standardised GWAS variant."""

    SNP: Optional[str] = None
    VARIANT_ID: str
    CHR: int
    POS: int
    A1: str
    A2: str
    BETA: float
    SE: float
    P: float
    MAF: Optional[float] = None
    N: int
    N_CAS: Optional[int] = None
    N_CON: Optional[int] = None
    INFO: Optional[float] = None


STANDARDIZED_GWAS_REQUIRED: dict[str, type] = {
    "SNP": str,
    "VARIANT_ID": str,
    "CHR": int,
    "POS": int,
    "A1": str,
    "A2": str,
    "BETA": float,
    "SE": float,
    "P": float,
    "N": int,
}

STANDARDIZED_GWAS_OPTIONAL: dict[str, type] = {
    "MAF": float,
    "N_CAS": int,
    "N_CON": int,
    "INFO": float,
}


class GWASMetadata(BaseModel):
    """Metadata sidecar for a StandardizedGWAS DataFrame."""

    genome_build: str = Field(..., pattern=r"^GRCh(37|38)$")
    trait: str = ""
    trait_type: str = Field(default="unknown", pattern=r"^(case_control|quantitative|unknown)$")
    frequency_column_is: str = Field(default="MAF", pattern=r"^(MAF|EAF)$")
    source: str = "custom"
    original_n_variants: int = 0
    n_variants_after_qc: int = 0
    liftover_applied: bool = False
    liftover_source: Optional[str] = None
    effect_allele_column: Optional[str] = None
    # Case/control structure threaded to Branch C coloc/Steiger.
    # All optional + default None so archived sidecars remain valid and the
    # case/control calibration paths stay dormant unless explicitly populated.
    n_cases: Optional[int] = None
    n_controls: Optional[int] = None
    n_total_cases_controls: Optional[int] = None
    case_control_n_source: Optional[str] = None
    population_prevalence: Optional[float] = None


# ---------------------------------------------------------------------------
# Schema 2: DrugTargetRecord
# ---------------------------------------------------------------------------


class DrugTargetRecordRow(BaseModel):
    """Single-row representation of a unified drug-target interaction."""

    drug_name: str
    drug_chembl_id: str
    drug_inchikey: Optional[str] = None
    drug_pubchem_cid: Optional[str] = None
    drug_smiles: Optional[str] = None
    gene_symbol: str
    gene_ensembl_id: str
    gene_uniprot_id: str
    gene_entrez_id: int
    interaction_type: str
    action_type: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    pchembl_value: Optional[float] = None
    affinity_value: Optional[float] = None
    affinity_type: Optional[str] = None
    affinity_unit: Optional[str] = None
    max_phase: int
    atc_codes: Optional[list[str]] = None
    indication_mesh: Optional[list[str]] = None
    molecule_type: Optional[str] = None
    is_withdrawn: Optional[bool] = None
    source: str
    confidence: str
    source_pmids: Optional[list[str]] = None


DRUG_TARGET_REQUIRED: dict[str, type] = {
    "drug_name": str,
    "drug_chembl_id": str,
    "gene_symbol": str,
    "gene_ensembl_id": str,
    "gene_uniprot_id": str,
    "gene_entrez_id": int,
    "interaction_type": str,
    "max_phase": int,
    "source": str,
    "confidence": str,
}

DRUG_TARGET_OPTIONAL: dict[str, type] = {
    "drug_inchikey": str,
    "drug_pubchem_cid": str,
    "drug_smiles": str,
    "action_type": str,
    "mechanism_of_action": str,
    "pchembl_value": float,
    "affinity_value": float,
    "affinity_type": str,
    "affinity_unit": str,
    "atc_codes": object,
    "indication_mesh": object,
    "molecule_type": str,
    "is_withdrawn": bool,
    "source_pmids": object,
}


# ---------------------------------------------------------------------------
# Schema 3: GeneAnnotationRecord
# ---------------------------------------------------------------------------


class GeneAnnotationRecordRow(BaseModel):
    """Single-row representation of a gene annotation."""

    gene_symbol: str
    gene_ensembl_id: str
    gene_entrez_id: Optional[int] = None
    gene_uniprot_id: Optional[str] = None
    chr: int
    start: int
    end: int
    biotype: str
    description: Optional[str] = None


GENE_ANNOTATION_REQUIRED: dict[str, type] = {
    "gene_symbol": str,
    "gene_ensembl_id": str,
    "chr": int,
    "start": int,
    "end": int,
    "biotype": str,
}

GENE_ANNOTATION_OPTIONAL: dict[str, type] = {
    "gene_entrez_id": int,
    "gene_uniprot_id": str,
    "description": str,
}


# ---------------------------------------------------------------------------
# Schema 4: PathwayRecord
# ---------------------------------------------------------------------------


class PathwayRecordRow(BaseModel):
    """Single-row representation of a pathway / gene set."""

    pathway_id: str
    pathway_name: str
    source_db: str
    category: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    genes: list[str]
    n_genes: int


PATHWAY_RECORD_REQUIRED: dict[str, type] = {
    "pathway_id": str,
    "pathway_name": str,
    "source_db": str,
    "genes": object,
    "n_genes": int,
}

PATHWAY_RECORD_OPTIONAL: dict[str, type] = {
    "category": str,
    "description": str,
    "url": str,
}


# ---------------------------------------------------------------------------
# Schema 5: DrugSignatureRecord
# ---------------------------------------------------------------------------


class DrugSignatureRecordRow(BaseModel):
    """Single-row representation of an aggregated drug expression signature."""

    drug_name: str
    drug_inchikey: str
    drug_chembl_id: Optional[str] = None
    lincs_pert_id: str
    n_profiles_aggregated: int
    cell_lines: list[str]
    doses: list[str]
    time_points: list[str]
    gene_ids: list[int]
    z_scores: list[float]
    match_confidence: str
    # additive composition columns (Optional for legacy parquet
    # re-load).  Computed on the *composition set* - the profile pool after
    # ``best_dose``/``per_condition`` filtering but BEFORE any
    # ``cell_line_weighting`` filter - so ``neural_fraction`` in
    # ``neural_only`` mode reflects the drug's original LINCS composition,
    # not the trivially-1.0 post-filter fraction.
    #
    # Invariant (test-locked): n_profiles_total == n_profiles_neural +
    # n_profiles_non_neural + n_profiles_unknown_cell_line.
    #
    # ``neural_weight_fraction`` semantics, per mode:
    #   * ``uniform``:          equals ``neural_fraction`` (all weights == 1).
    #   * ``neural_priority``:  ``sum(w_neural) / sum(w_all)`` - the
    #                           actual weight allocation, which differs
    #                           from ``neural_fraction`` because neural
    #                           weights are up-weighted.
    #   * ``neural_only``:      1.0 for retained drugs - post-filter
    #                           contribution is entirely neural, while
    #                           ``neural_fraction`` still reports the
    #                           pre-filter composition (which can be
    #                           < 1.0 and typically is).
    n_profiles_total: Optional[int] = None
    n_profiles_neural: Optional[int] = None
    n_profiles_non_neural: Optional[int] = None
    n_profiles_unknown_cell_line: Optional[int] = None
    neural_fraction: Optional[float] = None
    neural_weight_fraction: Optional[float] = None
    cell_line_weighting_mode: Optional[str] = None
    neural_weight: Optional[float] = None


DRUG_SIGNATURE_REQUIRED: dict[str, type] = {
    "drug_name": str,
    "drug_inchikey": str,
    "lincs_pert_id": str,
    "n_profiles_aggregated": int,
    "cell_lines": object,
    "doses": object,
    "time_points": object,
    "gene_ids": object,
    "z_scores": object,
    "match_confidence": str,
}

DRUG_SIGNATURE_OPTIONAL: dict[str, type] = {
    "drug_chembl_id": str,
    # additive composition columns (Optional so archived
    # previous parquets still validate). Current pipeline always populates.
    "n_profiles_total": int,
    "n_profiles_neural": int,
    "n_profiles_non_neural": int,
    "n_profiles_unknown_cell_line": int,
    "neural_fraction": float,
    "neural_weight_fraction": float,
    "cell_line_weighting_mode": str,
    "neural_weight": float,
}


# ---------------------------------------------------------------------------
# Schema 6: DiseaseSignaturePerTissue
# ---------------------------------------------------------------------------


class DiseaseSignaturePerTissueRow(BaseModel):
    """Single row of per-tissue S-PrediXcan results."""

    gene_ensembl_id: str
    gene_symbol: str
    gene_entrez_id: Optional[int] = None
    tissue: str
    zscore: float
    pvalue: float
    effect_size: float
    se: float
    n_snps_used: int
    n_snps_in_model: int
    pred_perf_r2: Optional[float] = None
    pred_perf_pval: Optional[float] = None
    mhc_flag: bool


DISEASE_SIGNATURE_PER_TISSUE_REQUIRED: dict[str, type] = {
    "gene_ensembl_id": str,
    "gene_symbol": str,
    "tissue": str,
    "zscore": float,
    "pvalue": float,
    "effect_size": float,
    "se": float,
    "n_snps_used": int,
    "n_snps_in_model": int,
    "mhc_flag": bool,
}

DISEASE_SIGNATURE_PER_TISSUE_OPTIONAL: dict[str, type] = {
    "gene_entrez_id": int,
    "pred_perf_r2": float,
    "pred_perf_pval": float,
}


# ---------------------------------------------------------------------------
# Schema 7: DiseaseSignatureMeta
# ---------------------------------------------------------------------------


class DiseaseSignatureMetaRow(BaseModel):
    """Single row of IVW meta-analysis results across tissues."""

    gene_ensembl_id: str
    gene_symbol: str
    gene_entrez_id: Optional[int] = None
    meta_zscore: float
    meta_pvalue: float
    meta_beta: float
    meta_se: float
    n_tissues: int
    i_squared: float
    q_statistic: float
    q_pvalue: float
    best_tissue: str
    best_tissue_zscore: float
    mhc_flag: bool


DISEASE_SIGNATURE_META_REQUIRED: dict[str, type] = {
    "gene_ensembl_id": str,
    "gene_symbol": str,
    "meta_zscore": float,
    "meta_pvalue": float,
    "meta_beta": float,
    "meta_se": float,
    "n_tissues": int,
    "i_squared": float,
    "q_statistic": float,
    "q_pvalue": float,
    "best_tissue": str,
    "best_tissue_zscore": float,
    "mhc_flag": bool,
}

DISEASE_SIGNATURE_META_OPTIONAL: dict[str, type] = {
    "gene_entrez_id": int,
}


# ---------------------------------------------------------------------------
# Schema 8: NegativeCorrelationResult (per tissue-drug pair)
# ---------------------------------------------------------------------------


class NegativeCorrelationResultRow(BaseModel):
    """Single row of negative correlation results (one drug × tissue pair)."""

    drug_name: str
    drug_pubchem_cid: Optional[str] = None
    drug_chembl_id: Optional[str] = None
    drug_inchikey: Optional[str] = None
    atc_codes: Optional[list[str]] = None
    tissue: str
    spearman_rho: float
    spearman_pvalue: float
    xsum_score: Optional[float] = None
    xsum_pvalue: Optional[float] = None
    fdr_global: float
    n_tissues_nominal: int
    n_overlapping_genes: int
    # additive columns (Optional for legacy parquet re-load)
    directional_pvalue: Optional[float] = None
    directional_fdr_global: Optional[float] = None
    spearman_fdr_per_tissue: Optional[float] = None
    directional_fdr_per_tissue: Optional[float] = None
    overlap_fraction_disease: float
    overlap_fraction_drug: float
    match_confidence: str
    lincs_pert_id: str
    n_profiles_aggregated: int
    cell_lines: Optional[list[str]] = None
    clinical_phase: Optional[int] = None
    mechanism_of_action: Optional[str] = None
    known_targets: Optional[list[str]] = None
    known_indications: Optional[list[str]] = None
    direction: str
    top_contributing_genes: Optional[list[str]] = None
    # additive composition columns (Optional for legacy parquet
    # re-load; propagated from ``drug_signatures.parquet`` via drug_meta_df).
    n_profiles_total: Optional[int] = None
    n_profiles_neural: Optional[int] = None
    neural_fraction: Optional[float] = None
    neural_weight_fraction: Optional[float] = None
    cell_line_weighting_mode: Optional[str] = None


NEGATIVE_CORRELATION_RESULT_REQUIRED: dict[str, type] = {
    "drug_name": str,
    "tissue": str,
    "spearman_rho": float,
    "spearman_pvalue": float,
    "fdr_global": float,
    "n_tissues_nominal": int,
    "n_overlapping_genes": int,
    "overlap_fraction_disease": float,
    "overlap_fraction_drug": float,
    "match_confidence": str,
    "lincs_pert_id": str,
    "n_profiles_aggregated": int,
    "direction": str,
}

NEGATIVE_CORRELATION_RESULT_OPTIONAL: dict[str, type] = {
    "drug_pubchem_cid": str,
    "drug_chembl_id": str,
    "drug_inchikey": str,
    "atc_codes": object,
    "xsum_score": float,
    "xsum_pvalue": float,
    "cell_lines": object,
    "clinical_phase": int,
    "mechanism_of_action": str,
    "known_targets": object,
    "known_indications": object,
    "top_contributing_genes": object,
    # additive columns (Optional so archived previous parquets still validate).
    # The current pipeline always populates these.
    "directional_pvalue": float,
    "directional_fdr_global": float,
    "spearman_fdr_per_tissue": float,
    "directional_fdr_per_tissue": float,
    # additive composition columns (propagated from drug_signatures).
    "n_profiles_total": int,
    "n_profiles_neural": int,
    "neural_fraction": float,
    "neural_weight_fraction": float,
    "cell_line_weighting_mode": str,
}


# ---------------------------------------------------------------------------
# Schema 9: NegativeCorrelationSummary (per drug, collapsed across tissues)
# ---------------------------------------------------------------------------


class NegativeCorrelationSummaryRow(BaseModel):
    """Single row of per-drug negative correlation summary."""

    drug_name: str
    best_spearman_rho: float
    best_tissue: str
    n_tissues_fdr_significant: int
    n_tissues_nominal: int
    drug_pubchem_cid: Optional[str] = None
    drug_chembl_id: Optional[str] = None
    drug_inchikey: Optional[str] = None
    atc_codes: Optional[list[str]] = None
    match_confidence: Optional[str] = None
    lincs_pert_id: Optional[str] = None
    n_profiles_aggregated: Optional[int] = None
    cell_lines: Optional[list[str]] = None
    clinical_phase: Optional[int] = None
    mechanism_of_action: Optional[str] = None
    known_targets: Optional[list[str]] = None
    known_indications: Optional[list[str]] = None
    # additive summary columns.
    # IMPORTANT: these are *descriptive selected-tissue statistics*, NOT
    # drug-level calibrated FDR inference.  ``best_directional_pvalue``
    # is the minimum directional p-value across tissues for the drug;
    # ``best_directional_fdr_per_tissue`` is the minimum per-tissue-BH
    # adjusted directional p across tissues.  Both compound
    # cross-tissue selection with per-tissue calibration and should not
    # be interpreted as unbiased drug-level significance.  The
    # principled drug-level combined statistic (Cauchy / harmonic-mean
    # p / Fisher) is deferred to a future revision; until then the
    # per-drug correction is intentionally not applied.
    n_tissues_directional_nominal: Optional[int] = None
    n_tissues_directional_fdr_significant: Optional[int] = None
    best_directional_pvalue: Optional[float] = None
    best_directional_fdr_per_tissue: Optional[float] = None
    # additive composition columns (Optional; propagated from
    # drug_signatures via drug-level metadata).
    n_profiles_total: Optional[int] = None
    n_profiles_neural: Optional[int] = None
    neural_fraction: Optional[float] = None
    neural_weight_fraction: Optional[float] = None
    cell_line_weighting_mode: Optional[str] = None


NEGATIVE_CORRELATION_SUMMARY_REQUIRED: dict[str, type] = {
    "drug_name": str,
    "best_spearman_rho": float,
    "best_tissue": str,
    "n_tissues_fdr_significant": int,
    "n_tissues_nominal": int,
}

NEGATIVE_CORRELATION_SUMMARY_OPTIONAL: dict[str, type] = {
    "drug_pubchem_cid": str,
    "drug_chembl_id": str,
    "drug_inchikey": str,
    "atc_codes": object,
    "match_confidence": str,
    "lincs_pert_id": str,
    "n_profiles_aggregated": int,
    # additive summary columns (descriptive; see class docstring)
    "n_tissues_directional_nominal": int,
    "n_tissues_directional_fdr_significant": int,
    "best_directional_pvalue": float,
    "best_directional_fdr_per_tissue": float,
    "cell_lines": object,
    "clinical_phase": int,
    "mechanism_of_action": str,
    "known_targets": object,
    "known_indications": object,
    # additive composition columns.
    "n_profiles_total": int,
    "n_profiles_neural": int,
    "neural_fraction": float,
    "neural_weight_fraction": float,
    "cell_line_weighting_mode": str,
}


# ---------------------------------------------------------------------------
# Schema 10: MRResult (one row per gene × eQTL source)
# ---------------------------------------------------------------------------


class MRResultRow(BaseModel):
    """Single row of MR results for one gene × eQTL source pair."""

    gene_ensembl_id: str
    gene_symbol: str
    gene_entrez_id: Optional[int] = None
    gene_uniprot_id: Optional[str] = None
    eqtl_source: str
    n_instruments: int
    mr_method: str
    mr_beta: float
    mr_se: float
    mr_pval: float
    mr_significant: bool
    bonferroni_threshold: float
    mean_f_stat: float
    weak_instrument_excluded: bool
    steiger_pval: Optional[float] = None
    steiger_valid: Optional[bool] = None
    r2_exposure: Optional[float] = None
    r2_outcome: Optional[float] = None
    q_stat: Optional[float] = None
    q_pval: Optional[float] = None
    heterogeneity_warning: bool = False
    ivw_fe_beta: Optional[float] = None
    ivw_fe_pval: Optional[float] = None
    ivw_re_beta: Optional[float] = None
    ivw_re_pval: Optional[float] = None
    egger_intercept_pval: Optional[float] = None
    egger_slope: Optional[float] = None
    egger_se: Optional[float] = None
    wm_beta: Optional[float] = None
    wm_se: Optional[float] = None
    wm_pval: Optional[float] = None
    pp_h4: Optional[float] = None
    pp_h3: Optional[float] = None
    n_snps_coloc: Optional[int] = None
    coloc_supported: bool = False
    coloc_status: str = "not_run"
    cross_source_status: str = "unavailable"
    confidence_tier: str = "low"


MR_RESULT_REQUIRED: dict[str, type] = {
    "gene_ensembl_id": str,
    "gene_symbol": str,
    "eqtl_source": str,
    "n_instruments": int,
    "mr_method": str,
    "mr_beta": float,
    "mr_se": float,
    "mr_pval": float,
    "mr_significant": bool,
    "bonferroni_threshold": float,
    "mean_f_stat": float,
    "weak_instrument_excluded": bool,
    "heterogeneity_warning": bool,
    "coloc_supported": bool,
    "coloc_status": str,
    "cross_source_status": str,
    "confidence_tier": str,
}

MR_RESULT_OPTIONAL: dict[str, type] = {
    "gene_entrez_id": int,
    "gene_uniprot_id": str,
    "steiger_pval": float,
    "steiger_valid": bool,
    "r2_exposure": float,
    "r2_outcome": float,
    "q_stat": float,
    "q_pval": float,
    "ivw_fe_beta": float,
    "ivw_fe_pval": float,
    "ivw_re_beta": float,
    "ivw_re_pval": float,
    "egger_intercept_pval": float,
    "egger_slope": float,
    "egger_se": float,
    "wm_beta": float,
    "wm_se": float,
    "wm_pval": float,
    "pp_h4": float,
    "pp_h3": float,
    "n_snps_coloc": int,
    # Additive per-source BH-FDR sensitivity track. The primary
    # significance call (mr_significant / bonferroni_threshold) is unchanged;
    # these expose the discovery-regime view over finite MR p-values.
    "mr_fdr_bh_q": float,
    "mr_significant_fdr_bh": bool,
    "bonferroni_threshold_tested": float,
    "mr_significant_bonferroni_tested": bool,
    # Whether the Steiger liability transform fired.
    "steiger_binary_calibration": str,
    # Gene anchor (eQTL-loader coordinate; may be min-SNP-pos) and
    # additive MHC-region flag (Ensembl-membership derived; coordinate fallback
    # opt-in). mhc_flag_method records how the flag was computed.
    "gene_chr": str,
    "gene_start": int,
    "mhc_flag": bool,
    "mhc_flag_method": str,
}


# ---------------------------------------------------------------------------
# Schema 11: MRDrugMatch (one row per gene × drug × eQTL source)
# ---------------------------------------------------------------------------


class MRDrugMatchRow(BaseModel):
    """Single row of MR drug matches for genes passing drug-matching gating."""

    gene_ensembl_id: str
    gene_symbol: str
    gene_entrez_id: Optional[int] = None
    gene_uniprot_id: Optional[str] = None
    eqtl_source: str
    mr_beta: float
    mr_pval: float
    pp_h4: Optional[float] = None
    confidence_tier: str
    drug_chembl_id: str
    drug_name: str
    drug_inchikey: Optional[str] = None
    interaction_type: str
    direction_concordant: Optional[bool] = None
    interaction_direction_ambiguous: bool = False
    pchembl_value: Optional[float] = None
    max_phase: int = 0
    atc_codes: Optional[list[str]] = None
    # Additive drug-evidence provenance / ranking. Populated in both
    # legacy and strict match modes so mechanism-vs-binder ranking is auditable.
    action_type: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    has_mechanism_text: bool = False
    direction_inferable: bool = False
    drug_target_source: Optional[str] = None
    drug_target_confidence: Optional[str] = None
    match_via: Optional[str] = None
    drug_match_rank: Optional[int] = None


MR_DRUG_MATCH_REQUIRED: dict[str, type] = {
    "gene_ensembl_id": str,
    "gene_symbol": str,
    "eqtl_source": str,
    "mr_beta": float,
    "mr_pval": float,
    "confidence_tier": str,
    "drug_chembl_id": str,
    "drug_name": str,
    "interaction_type": str,
    "interaction_direction_ambiguous": bool,
    "max_phase": int,
}

MR_DRUG_MATCH_OPTIONAL: dict[str, type] = {
    "gene_entrez_id": int,
    "gene_uniprot_id": str,
    "pp_h4": float,
    "drug_inchikey": str,
    "direction_concordant": bool,
    "pchembl_value": float,
    "atc_codes": object,
    "action_type": str,
    "mechanism_of_action": str,
    "has_mechanism_text": bool,
    "direction_inferable": bool,
    "drug_target_source": str,
    "drug_target_confidence": str,
    "match_via": str,
    "drug_match_rank": int,
}


# ---------------------------------------------------------------------------
# Schema 11b: MRTargetVerdict (one row per drug-match-eligible gene × eQTL source)
# ---------------------------------------------------------------------------


class MRTargetVerdictRow(BaseModel):
    """Per-gene drug-actionability verdict for MR-significant, eligible genes.

    Emitted for *every* drug-match-eligible gene×source - including genes with no
    passing drug - so "causal gene, no actionable drug" is a recorded scientific
    result rather than a silent absence.
    """

    gene_ensembl_id: str
    gene_symbol: str
    gene_entrez_id: Optional[int] = None
    gene_uniprot_id: Optional[str] = None
    eqtl_source: str
    mr_beta: float
    mr_pval: float
    pp_h4: Optional[float] = None
    confidence_tier: str
    verdict_status: str  # actionable | binder_only | no_filtered_match | no_drug_record
    n_drug_records_raw: int
    n_filtered_matches: int
    n_direction_inferable: int
    n_direction_concordant: int
    best_max_phase: Optional[int] = None
    best_pchembl: Optional[float] = None
    best_match_via: Optional[str] = None
    best_drug_chembl_id: Optional[str] = None
    best_drug_name: Optional[str] = None
    druggable_tier: Optional[str] = None


MR_TARGET_VERDICT_REQUIRED: dict[str, type] = {
    "gene_ensembl_id": str,
    "gene_symbol": str,
    "eqtl_source": str,
    "mr_beta": float,
    "mr_pval": float,
    "confidence_tier": str,
    "verdict_status": str,
    "n_drug_records_raw": int,
    "n_filtered_matches": int,
    "n_direction_inferable": int,
    "n_direction_concordant": int,
}

MR_TARGET_VERDICT_OPTIONAL: dict[str, type] = {
    "gene_entrez_id": int,
    "gene_uniprot_id": str,
    "pp_h4": float,
    "best_max_phase": int,
    "best_pchembl": float,
    "best_match_via": str,
    "best_drug_chembl_id": str,
    "best_drug_name": str,
    "druggable_tier": str,
}


# ---------------------------------------------------------------------------
# Schema registry and DataFrame validator
# ---------------------------------------------------------------------------

_SCHEMA_REGISTRY: dict[str, dict[str, type]] = {
    "StandardizedGWAS": STANDARDIZED_GWAS_REQUIRED,
    "DrugTargetRecord": DRUG_TARGET_REQUIRED,
    "GeneAnnotationRecord": GENE_ANNOTATION_REQUIRED,
    "PathwayRecord": PATHWAY_RECORD_REQUIRED,
    "DrugSignatureRecord": DRUG_SIGNATURE_REQUIRED,
    "DiseaseSignaturePerTissue": DISEASE_SIGNATURE_PER_TISSUE_REQUIRED,
    "DiseaseSignatureMeta": DISEASE_SIGNATURE_META_REQUIRED,
    "NegativeCorrelationResult": NEGATIVE_CORRELATION_RESULT_REQUIRED,
    "NegativeCorrelationSummary": NEGATIVE_CORRELATION_SUMMARY_REQUIRED,
    "MRResult": MR_RESULT_REQUIRED,
    "MRDrugMatch": MR_DRUG_MATCH_REQUIRED,
    "MRTargetVerdict": MR_TARGET_VERDICT_REQUIRED,
}

_DTYPE_COMPAT: dict[type, list[str]] = {
    int: ["int", "Int", "uint"],
    float: ["float", "Float", "int", "Int"],
    str: ["object", "string", "str"],
    bool: ["bool", "Bool", "object"],
    object: ["object"],
}


def _dtype_compatible(col_dtype: np.dtype, expected_py_type: type) -> bool:
    """Check if a pandas dtype is compatible with the expected Python type."""
    dtype_str = str(col_dtype)
    for prefix in _DTYPE_COMPAT.get(expected_py_type, []):
        if dtype_str.startswith(prefix):
            return True
    if expected_py_type is object:
        return True
    return False


def validate_dataframe(df: pd.DataFrame, schema_name: str) -> list[str]:
    """Validate that *df* conforms to a named canonical schema.

    Checks that all required columns exist and have compatible dtypes.
    Returns a list of error strings (empty = valid).

    Args:
        df: The DataFrame to validate.
        schema_name: One of the registered schema names
            (``StandardizedGWAS``, ``DrugTargetRecord``,
            ``GeneAnnotationRecord``, ``PathwayRecord``,
            ``DrugSignatureRecord``).

    Returns:
        List of validation error messages.  Empty list means valid.

    Raises:
        ValueError: If *schema_name* is not in the registry.
    """
    if schema_name not in _SCHEMA_REGISTRY:
        raise ValueError(
            f"Unknown schema '{schema_name}'. "
            f"Available: {list(_SCHEMA_REGISTRY.keys())}"
        )

    required_cols = _SCHEMA_REGISTRY[schema_name]
    errors: list[str] = []

    for col_name, expected_type in required_cols.items():
        if col_name not in df.columns:
            errors.append(f"Missing required column: {col_name}")
            continue
        if not _dtype_compatible(df[col_name].dtype, expected_type):
            errors.append(
                f"Column '{col_name}' has dtype '{df[col_name].dtype}', "
                f"expected compatible with {expected_type.__name__}"
            )

    if errors:
        logger.warning(
            "Schema validation failed for %s: %d error(s)", schema_name, len(errors)
        )
        for err in errors:
            logger.warning("  %s", err)

    return errors
