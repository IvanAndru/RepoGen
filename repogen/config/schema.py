"""Pydantic models defining all RepoGen pipeline configuration.

Every configurable parameter flows through these models.  If validation
fails the pipeline refuses to start - no silent misconfiguration.

Updated to v2.1: added GWASPrepConfig, DrugSignaturesConfig,
OpenTargetsConfig; expanded StudyConfig, MagmaConfig, DrugEnrichmentConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class StudyConfig(BaseModel):
    """Metadata about the GWAS study being analysed."""

    name: str = Field(..., min_length=1, description="Short study identifier used in output file names")
    gwas_input: Path = Field(..., description="Path to the raw GWAS summary statistics file")
    sample_size: Optional[int] = Field(default=None, gt=0, description="Total sample size (None = read from file)")
    n_cases: Optional[int] = Field(default=None, gt=0, description="Number of cases (case-control studies)")
    n_controls: Optional[int] = Field(default=None, gt=0, description="Number of controls (case-control studies)")
    genome_build: Optional[str] = Field(
        default=None,
        pattern=r"^GRCh(37|38)$",
        description="Genome build (GRCh37 or GRCh38); None = auto-detect",
    )
    trait_type: Optional[str] = Field(
        default=None,
        pattern=r"^(case_control|quantitative)$",
        description="Trait type; None = auto-detect from effect column distribution",
    )
    population_prevalence: Optional[float] = Field(
        default=None,
        gt=0,
        lt=1,
        description=(
            "Population (lifetime) disease prevalence for a case/control trait, "
            "e.g. 0.01 for schizophrenia. Used ONLY for the Steiger liability-scale "
            "transform in Branch C. Must be the true population "
            "prevalence - never the sample case fraction. When None, Steiger stays "
            "on the observed/binary-approximation scale (annotate-only)."
        ),
    )
    description: str = Field(default="", description="Optional free-text description")


class GWASPrepConfig(BaseModel):
    """GWAS preprocessing / quality control settings."""

    info_threshold: float = Field(default=0.6, ge=0, le=1, description="Minimum imputation INFO score")
    maf_threshold: float = Field(default=0.01, ge=0, le=0.5, description="Minimum minor allele frequency")
    liftover_to: Optional[str] = Field(
        default=None,
        pattern=r"^GRCh(37|38)$",
        description="Target build for liftover; None = no liftover",
    )
    remove_mhc: bool = Field(default=False, description="Remove MHC region (chr6:25-35Mb)")


class ReferenceConfig(BaseModel):
    """Paths to reference data files."""

    genome_dir: Path = Field(default=Path("resources/reference"), description="Directory containing 1000G reference files")
    population: str = Field(default="EUR", description="Reference population (EUR, EAS, AFR, AMR, SAS)")
    bfile_prefix: str = Field(default="g1000_eur", description="PLINK bfile prefix within genome_dir")
    gene_loc_file: Optional[Path] = Field(default=None, description="NCBI gene location file (GRCh37) - primary source for Branch A MAGMA gene annotation")
    gene_loc_file_grch38: Optional[Path] = Field(
        default=None,
        description=(
            "NCBI/MAGMA gene location file (GRCh38) - primary MHC annotation "
            "source for Branch B S-PrediXcan. When unset, Branch B falls back "
            "to the GRCh37 file (gene_loc_file) projected via Entrez->Ensembl."
        ),
    )
    ensembl_to_name: Optional[Path] = Field(default=None, description="BioMart dico1 - Ensembl to gene name")
    name_to_ensembl: Optional[Path] = Field(default=None, description="BioMart dico2 - gene name to Ensembl")
    uniprot_to_ensembl: Optional[Path] = Field(default=None, description="BioMart dico3 - UniProt to Ensembl")
    liftover_chain: Optional[Path] = Field(default=None, description="Chain file for coordinate liftover")
    ncbi_gene_info: Optional[Path] = Field(default=None, description="NCBI gene_info.gz for Entrez ID mapping")
    ncbi_gene_history: Optional[Path] = Field(default=None, description="NCBI gene_history.gz for deprecated ID resolution")
    predixcan_model_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing PredictDB .db model files (e.g., mashr_Brain_Amygdala.db)",
    )
    predixcan_covariance_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing .txt.gz covariance files. If None, assumes same directory as model_dir.",
    )


class MagmaConfig(BaseModel):
    """MAGMA gene and pathway analysis settings."""

    binary_path: Optional[Path] = None

    annotation_mode: str = "proximity"
    custom_annot_file: Optional[Path] = None

    gene_model: str = "mean"

    window_upstream_kb: int = Field(default=35, ge=0)
    window_downstream_kb: int = Field(default=10, ge=0)

    exclude_mhc: bool = True
    biotype_filter: Optional[list[str]] = None

    memory_efficient: bool = False

    @field_validator("annotation_mode")
    @classmethod
    def validate_annotation_mode(cls, v: str) -> str:
        allowed = {"proximity", "hmagma_fetal_brain", "hmagma_adult_brain",
                    "hmagma_ipsc_neurons", "custom"}
        if v not in allowed:
            raise ValueError(f"annotation_mode must be one of {allowed}, got '{v}'")
        return v

    @field_validator("gene_model")
    @classmethod
    def validate_gene_model(cls, v: str) -> str:
        allowed = {"mean", "multi"}
        if v not in allowed:
            raise ValueError(f"gene_model must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def check_custom_annot(self) -> MagmaConfig:
        if self.annotation_mode == "custom" and self.custom_annot_file is None:
            raise ValueError("custom_annot_file must be set when annotation_mode is 'custom'")
        return self


class PathwayConfig(BaseModel):
    """Configuration for MAGMA pathway/gene-set enrichment analysis."""

    fdr_method: str = "fdr_bh"
    fdr_threshold: float = Field(default=0.05, gt=0, lt=1)

    min_set_size: int = Field(default=10, ge=2)
    max_set_size: int = Field(default=500, ge=10)
    sources: Optional[list[str]] = None

    driver_gene_p_threshold: float = Field(default=0.05, gt=0, lt=1)

    extra_gmt_files: list[Path] = Field(default_factory=list)

    @field_validator("fdr_method")
    @classmethod
    def validate_fdr_method(cls, v: str) -> str:
        allowed = {"fdr_bh", "fdr_by", "holm", "bonferroni", "fdr_tsbh", "fdr_tsbky"}
        if v not in allowed:
            raise ValueError(f"fdr_method must be one of {allowed}, got '{v}'")
        return v


class DrugEnrichmentConfig(BaseModel):
    """Drug-gene enrichment analysis settings."""

    sources: list[str] = Field(default=["chembl"], description='Drug databases to use (chembl, pdsp, dgidb)')
    chembl_scope: str = Field(
        default="mechanism_only",
        description="ChEMBL inclusion mode: 'mechanism_only' (curated MOA drugs, default) "
                    "or 'mechanism_or_affinity' (includes affinity-only compounds - "
                    "consider pairing with min_pchembl/max_phase_filter for practical runs)",
    )

    enable_expression_enrichment: bool = Field(
        default=False,
        description=(
            "Enable expression-perturbation drug-gene "
            "evidence (CREEDS, DSigDB) as a parallel Branch-A enrichment "
            "input.  When False (default), pipeline runs TARGETS-only and "
            "outputs are byte-identical to the previous behaviour. When True, an "
            "additional 'load_drug_targets_expr' loader produces "
            "drug_targets_expr.parquet (TARGETS+EXPR) which the Branch-A "
            "drug_enrichment rule consumes; Branch B and Branch C continue "
            "to read drug_targets.parquet (TARGETS-only) so their behaviour "
            "is unaffected."
        ),
    )
    expression_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Expression-perturbation drug-gene sources. "
            "Subset of {'creeds', 'dsigdb'}.  Strictly disjoint from "
            "'sources' (which is target-family only).  Must be non-empty "
            "when enable_expression_enrichment is True."
        ),
    )

    min_genes_per_drug: int = Field(default=3, ge=1, description="Minimum target genes for a drug to be tested at the headline level (drug-gene FDR + exports + plot)")
    atc_min_genes_per_drug: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Minimum target genes for a drug to enter the ATC "
            "enrichment universe.  When None (default), inherits "
            "min_genes_per_drug for byte-identical legacy behaviour.  Set "
            "<= min_genes_per_drug (typically 1) in configs/config.yaml to "
            "expand the ATC GLS universe while keeping headline drug-gene "
            "FDR/exports strict.  Sub-headline drugs receive valid magma_p "
            "but magma_fdr_q=NaN and are excluded from headline outputs."
        ),
    )
    min_pchembl: Optional[float] = Field(default=None, description="Potency cutoff in pChEMBL units (e.g. 6.0 = 1µM)")
    max_phase_filter: Optional[int] = Field(default=None, ge=0, le=4, description="Minimum clinical phase (e.g. 1 = Phase I+)")
    phase_filter_scope: Literal["global", "chembl_only"] = Field(
        default="global",
        description="Scope for max_phase_filter: 'global' applies to all sources, "
                    "'chembl_only' exempts PDSP/DGIdb (which are always phase 0)",
    )
    confidence_filter: Optional[str] = Field(default=None, description="Confidence level filter (high, medium, low, or None)")

    pdsp_dedup_mode: Literal["off", "exact_signature"] = Field(
        default="off",
        description="PDSP family dedup: 'off' = no dedup (default), "
                    "'exact_signature' = collapse PDSP-only drugs with identical target gene sets",
    )

    parent_salt_unification: bool = Field(
        default=True,
        description=(
            "Collapse child and salt ChEMBL IDs onto "
            "their parent CID before drug-gene dedup so the GLS ATC "
            "regression treats each pharmacological entity as a single "
            "observation.  Default True; set False as a rollback / "
            "debugging escape hatch."
        ),
    )

    fdr_method: str = Field(default="fdr_bh", description="FDR correction method")
    fdr_threshold: float = Field(default=0.05, gt=0, lt=1, description="FDR significance threshold")
    include_wilcoxon_auc: bool = Field(default=True, description="Compute Wilcoxon AUC as descriptive complement")

    permutation_test: bool = Field(default=False, description="Enable permutation-based enrichment test")
    n_permutations: int = Field(default=10000, ge=100, description="Permutations per drug (if permutation_test enabled)")
    permutation_seed: Optional[int] = Field(default=42, description="Random seed for permutation reproducibility")

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        allowed = {"chembl", "pdsp", "dgidb"}
        seen: set[str] = set()
        result: list[str] = []
        for s in v:
            s_lower = s.lower()
            if s_lower not in allowed:
                raise ValueError(
                    f"Unknown drug source '{s}'. Allowed: {sorted(allowed)}"
                )
            if s_lower not in seen:
                seen.add(s_lower)
                result.append(s_lower)
        if not result:
            raise ValueError("sources must contain at least one entry")
        return result

    @field_validator("chembl_scope")
    @classmethod
    def validate_chembl_scope(cls, v: str) -> str:
        v = v.lower().strip()
        allowed = {"mechanism_only", "mechanism_or_affinity"}
        if v not in allowed:
            raise ValueError(f"chembl_scope must be one of {allowed}, got '{v}'")
        return v

    @field_validator("fdr_method")
    @classmethod
    def validate_fdr_method(cls, v: str) -> str:
        allowed = {"fdr_bh", "fdr_by", "holm", "bonferroni", "fdr_tsbh", "fdr_tsbky"}
        if v not in allowed:
            raise ValueError(f"fdr_method must be one of {allowed}, got '{v}'")
        return v

    @field_validator("confidence_filter")
    @classmethod
    def validate_confidence_filter(cls, v: str | None) -> str | None:
        if v is not None and v not in {"high", "medium", "low"}:
            raise ValueError(f"confidence_filter must be 'high', 'medium', 'low', or None, got '{v}'")
        return v

    @field_validator("expression_sources")
    @classmethod
    def validate_expression_sources(cls, v: list[str]) -> list[str]:
        """Validate expression source list.

        Allowed values are the strict-subset of expression-perturbation
        gene-set sources. NOTE: this domain is intentionally disjoint
        from ``sources`` (which is target-family only). The disjoint-set
        invariant is enforced by the model validator below.
        """
        allowed = {"creeds", "dsigdb"}
        seen: set[str] = set()
        result: list[str] = []
        for s in v:
            s_lower = s.lower().strip()
            if s_lower not in allowed:
                raise ValueError(
                    f"Unknown expression source '{s}'. Allowed: {sorted(allowed)}"
                )
            if s_lower not in seen:
                seen.add(s_lower)
                result.append(s_lower)
        return result

    @model_validator(mode="after")
    def _resolve_atc_min_genes(self) -> "DrugEnrichmentConfig":
        """Resolve atc_min_genes_per_drug.

        - When None (default), inherit min_genes_per_drug so legacy code paths
          (programmatic ``DrugEnrichmentConfig(...)`` instantiations and tests
          that pre-date the two-tier threshold) get byte-identical behaviour.
        - When set explicitly, must be <= min_genes_per_drug (otherwise the
          headline drug-gene pool would be empty).
        """
        if self.atc_min_genes_per_drug is None:
            self.atc_min_genes_per_drug = self.min_genes_per_drug
        elif self.atc_min_genes_per_drug > self.min_genes_per_drug:
            raise ValueError(
                f"atc_min_genes_per_drug ({self.atc_min_genes_per_drug}) must be "
                f"<= min_genes_per_drug ({self.min_genes_per_drug})."
            )
        return self

    @model_validator(mode="after")
    def _validate_expression_enrichment_consistency(self) -> "DrugEnrichmentConfig":
        """Enforce expression-enrichment configuration invariants.

        - When ``enable_expression_enrichment`` is True, ``expression_sources``
          must be non-empty (otherwise the new loader has nothing to load).
        - ``sources`` and ``expression_sources`` must be strictly disjoint.
          The per-field validators already restrict each field to a non-
          overlapping domain, but this check provides defence-in-depth
          against future relaxations of either domain.
        """
        if self.enable_expression_enrichment and not self.expression_sources:
            raise ValueError(
                "enable_expression_enrichment=True requires non-empty "
                "expression_sources (subset of {'creeds', 'dsigdb'})."
            )
        overlap = set(self.sources) & set(self.expression_sources)
        if overlap:
            raise ValueError(
                "drug_enrichment.sources and drug_enrichment.expression_sources "
                f"must be disjoint; got overlap: {sorted(overlap)}."
            )
        return self


class CustomATCClass(BaseModel):
    """A curated drug-set tested as an extra ATC class, defined by explicit ATC codes.

    Curated classes are additive: they are tested alongside the standard
    prefix-derived classes and are independent of ``atc_levels`` (the ``level``
    field is display-only metadata for plotting/grouping, not a test gate).
    Membership is by ATC-code intersection (typically Level 5 codes); the same
    ``min_drugs_per_class`` floor applies as for standard classes.
    """

    code: str = Field(
        description="Result key / display code, e.g. 'N06AB_SSRI'. Should start with "
                    "the relevant ATC chapter prefix (e.g. 'N06') so the forest plot "
                    "groups it correctly. Must not match a standard ATC class code."
    )
    level: int = Field(
        default=4, ge=1, le=4,
        description="Display level for plotting/grouping only (does NOT gate testing)."
    )
    description: str = Field(default="", description="Human-readable class description.")
    atc_members: list[str] = Field(
        default_factory=list,
        description="ATC codes (typically Level 5) defining membership. A drug is "
                    "included if any of its atc_codes matches one of these."
    )

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("custom ATC class 'code' must be non-empty")
        return v

    @field_validator("atc_members")
    @classmethod
    def validate_members(cls, v: list[str]) -> list[str]:
        cleaned = [m.strip().upper() for m in v if m and m.strip()]
        if not cleaned:
            raise ValueError("custom ATC class 'atc_members' must contain at least one ATC code")
        return sorted(set(cleaned))


class ATCEnrichmentConfig(BaseModel):
    """Configuration for ATC drug class enrichment analysis (DRUGSETS-style GLS)."""

    atc_levels: list[int] = Field(default=[2, 3], description="ATC hierarchy levels to test (1-4)")
    min_drugs_per_class: int = Field(default=5, ge=2, description="Minimum drugs for a class to be tested")
    two_sided: bool = Field(default=False, description="Two-sided test (default: one-sided upper-tail for enrichment)")
    fdr_method: str = Field(default="fdr_bh", description="FDR correction method for statsmodels.multipletests")
    fdr_threshold: float = Field(default=0.05, gt=0, lt=1, description="FDR significance threshold")
    eigenvalue_threshold: float = Field(default=0.1, gt=0, description="Eigenvalue clipping threshold for matrix regularization")
    permutation_test: bool = Field(default=False, description="Run drug-label permutation test")
    n_permutations: int = Field(default=10000, ge=100, description="Number of permutations (if enabled)")
    permutation_seed: int | None = Field(default=42, description="Random seed for reproducibility")
    atc_universe_mode: str = Field(
        default="annotated_only",
        description="Drug universe for ATC class tests: 'annotated_only' (DRUGSETS-style, "
                    "only ATC-annotated drugs) or 'all_drugs' (all drugs in enrichment)."
    )
    custom_classes: list[CustomATCClass] = Field(
        default_factory=list,
        description="Curated drug-sets tested as additional ATC classes (additive; "
                    "independent of atc_levels). Empty = standard behaviour only."
    )

    @field_validator("atc_levels")
    @classmethod
    def validate_atc_levels(cls, v: list[int]) -> list[int]:
        for level in v:
            if level not in {1, 2, 3, 4}:
                raise ValueError(f"ATC level must be 1-4, got {level}")
        return sorted(set(v))

    @field_validator("fdr_method")
    @classmethod
    def validate_fdr_method(cls, v: str) -> str:
        allowed = {"fdr_bh", "fdr_by", "bonferroni", "holm", "hommel"}
        if v not in allowed:
            raise ValueError(f"fdr_method must be one of {allowed}, got '{v}'")
        return v

    @field_validator("atc_universe_mode")
    @classmethod
    def validate_atc_universe_mode(cls, v: str) -> str:
        allowed = {"annotated_only", "all_drugs"}
        if v not in allowed:
            raise ValueError(f"atc_universe_mode must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_custom_classes(self):
        codes = [c.code for c in self.custom_classes]
        dupes = sorted({c for c in codes if codes.count(c) > 1})
        if dupes:
            raise ValueError(f"Duplicate custom ATC class codes: {dupes}")
        return self


class SpredixcanConfig(BaseModel):
    """Configuration for S-PrediXcan disease signature analysis."""

    model_type: str = Field(
        default="mashr",
        description="Prediction model type: 'mashr', 'jti', or 'elastic_net'",
    )
    tissues: list[str] = Field(
        default_factory=list,
        description="GTEx tissue names. Empty = use default brain_13 preset.",
    )
    tissue_preset: str = Field(
        default="brain_13",
        description="Tissue preset: 'brain_13', 'brain_extended', 'all_gtex', or 'custom'",
    )
    extra_models: list[Path] = Field(
        default_factory=list,
        description="Paths to non-GTEx .db model files (e.g., PsychENCODE)",
    )
    gwas_imputation: bool = Field(
        default=False,
        description=(
            "DEFERRED (v2.0): GWAS summary imputation via hakyimlab/summary-gwas-imputation. "
            "Setting to True raises NotImplementedError in v1.0. "
            "Built-in variant harmonisation (~95% coverage) is the default mode."
        ),
    )
    min_snps_used_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of model SNPs that must be found in GWAS for a gene to be included.",
    )
    exclude_palindromic: bool = Field(
        default=True,
        description="Skip palindromic (A/T, C/G) SNPs during allele alignment.",
    )

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = {"mashr", "jti", "elastic_net"}
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("tissues", mode="before")
    @classmethod
    def resolve_tissues(cls, v: list[str]) -> list[str]:
        """Pass through; resolution from preset happens in orchestrator."""
        return v if v else v

    @field_validator("tissue_preset")
    @classmethod
    def validate_tissue_preset(cls, v: str) -> str:
        allowed = {"brain_13", "brain_extended", "all_gtex", "custom"}
        if v not in allowed:
            raise ValueError(f"tissue_preset must be one of {allowed}, got '{v}'")
        return v


class DrugSignaturesConfig(BaseModel):
    """LINCS L1000 drug signature extraction settings.

    adds cell-line-aware consensus weighting via
    ``cell_line_weighting``. Default ``"uniform"`` reproduces previous
    behaviour byte-for-byte (locked by regression test).

    Neural cell-line defaults are derived from a Phase-0 census of the
    LINCS L1000 Level 5 corpus (see
    ``scripts/lincs_cell_line_census.py``). Tokens with zero observed
    profiles in the shipped GCTX are excluded from defaults - see :data:`NEURAL_PRIMARY_CELL_LINES`
    and :data:`NEURAL_TUMOR_CELL_LINES` in
    :mod:`repogen.data.drug_signatures`.
    """

    aggregation: str = Field(
        default="consensus",
        pattern=r"^(consensus|best_dose|per_condition)$",
        description="Profile aggregation strategy",
    )
    min_profiles: int = Field(default=3, ge=1, description="Minimum profiles to aggregate per drug")
    preferred_dose: Optional[str] = Field(default="10 µM", description="Preferred dose for best_dose aggregation")
    preferred_time: Optional[str] = Field(default="24 h", description="Preferred time point for best_dose aggregation")

    # --- cell-line-aware consensus weighting --------------
    cell_line_weighting: Literal["uniform", "neural_priority", "neural_only"] = Field(
        default="uniform",
        description=(
            "Cell-line weighting mode for drug-signature aggregation. "
            "'uniform' (default) reproduces previous nanmedian behaviour "
            "byte-for-byte. 'neural_priority' upweights neural profiles by "
            "``neural_weight`` in a weighted nanmedian. 'neural_only' "
            "drops non-neural profiles before aggregation and requires "
            "``min_neural_profiles`` neural profiles per drug."
        ),
    )
    neural_cell_lines: Optional[list[str]] = Field(
        default=None,
        description=(
            "Exact uppercase LINCS cell-line tokens treated as neural. "
            "If None, resolved from ``include_neural_tumor_cell_lines`` "
            "via a model validator. User overrides are normalized "
            "(uppercased, stripped, deduplicated)."
        ),
    )
    include_neural_tumor_cell_lines: bool = Field(
        default=True,
        description=(
            "When True, the derived default ``neural_cell_lines`` includes "
            "LN229 (neural-lineage-derived cancer). When False, primary-"
            "neural only (NEU/NPC/ASC/FIBRNPC). Ignored if the user "
            "explicitly sets ``neural_cell_lines``."
        ),
    )
    neural_weight: float = Field(
        default=3.0, ge=1.0,
        description=(
            "Upweight factor applied to neural profiles in "
            "``neural_priority`` mode. 1.0 collapses to ``uniform`` "
            "aggregation. Ignored in ``uniform`` and ``neural_only`` modes."
        ),
    )
    min_neural_profiles: Optional[int] = Field(
        default=None, ge=1,
        description=(
            "Minimum number of neural profiles required in ``neural_only`` "
            "mode. If None (default), resolves to ``min_profiles`` for "
            "coherence with the extraction-stage threshold. Setting this "
            "below ``negative_correlation.min_profiles_aggregated`` will "
            "produce silent downstream drops - the extractor logs its own "
            "attrition; the NC layer logs its own."
        ),
    )

    @model_validator(mode="after")
    def _resolve_neural_cell_lines(self) -> "DrugSignaturesConfig":
        """Derive and normalise ``neural_cell_lines`` after full
        construction so we do not depend on field-declaration order.

        Delegates to :func:`repogen.data.drug_signatures.resolve_neural_cell_lines_from_yaml`
        so this class and the Snakemake ``extract_drug_signatures`` rule
        share a single source of truth. The workflow reads raw YAML and so
        bypasses Pydantic entirely; without a shared resolver it silently
        ignored ``include_neural_tumor_cell_lines``.
        """
        # Import here to avoid circular imports at module load time.
        from repogen.data.drug_signatures import resolve_neural_cell_lines_from_yaml

        self.neural_cell_lines = resolve_neural_cell_lines_from_yaml(
            {
                "neural_cell_lines": self.neural_cell_lines,
                "include_neural_tumor_cell_lines": self.include_neural_tumor_cell_lines,
            }
        )
        return self


class PermutationCalibrationConfig(BaseModel):
    """Permutation-based calibration diagnostic for Branch B.

    Produces ``negative_correlation/calibration.json`` as a diagnostic
    sidecar reporting genomic-control-style lambda (chi²_1-based, per
    Devlin & Roeder 1999) and a secondary -log10(p) ratio heuristic.
    It does not modify the primary p-values in ``per_tissue_results.parquet``.
    Off by default for backward compatibility.

    Scale note: ``n_permutations=100`` is a *diagnostic* scale
    appropriate for a single scalar lambda estimate. It is not
    sufficient for per-test empirical p-values (min p = 1/(R+1)
    ≈ 0.0099 is too coarse for FDR over thousands of tests); those
    are deferred to a future revision.
    """

    enabled: bool = Field(
        default=False,
        description="Enable permutation calibration sidecar. Default False (backward-compatible).",
    )
    n_permutations: int = Field(
        default=100, ge=10,
        description="Number of disease-vector permutations. 100 is a diagnostic scale "
                    "sufficient for a single scalar lambda estimate; NOT suitable for "
                    "per-test empirical p-values (min p = 1/(R+1)).",
    )
    seed: int = Field(
        default=42,
        description="RNG seed for the calibration permutations. Deterministic by default.",
    )


class NegativeCorrelationConfig(BaseModel):
    """Negative correlation (signature reversal) pipeline settings."""

    tissues: list[str] = Field(default_factory=list, validate_default=True, description="Brain tissues for S-PrediXcan")
    correlation_method: str = Field(default="spearman", description="Correlation method. Only 'spearman' is supported in v1.0.")
    min_overlapping_genes: int = Field(default=50, ge=10, description="Minimum overlapping genes for correlation")
    fdr_threshold: float = Field(default=0.05, gt=0, lt=1, description="FDR significance threshold")

    gene_set_mode: str = Field(
        default="landmark",
        description="Gene set for drug signature filtering: 'landmark', 'landmark_bing', 'all'",
    )
    lincs_gene_info_path: Optional[Path] = Field(
        default=None,
        description="Path to lincs_gene_info.tsv. None = auto-resolve from resources/",
    )
    match_confidence_threshold: str = Field(
        default="pubchem_cid",
        description="Min match confidence: 'inchikey', 'pubchem_cid', 'name'",
    )
    exclude_mhc: bool = Field(
        default=False,
        description="Exclude MHC region genes from correlation. Default False (trait-agnostic). "
                    "MHC sensitivity analysis always runs regardless of this setting.",
    )
    min_profiles_aggregated: int = Field(
        default=3, ge=1,
        description="Minimum LINCS profiles for drug signature inclusion",
    )
    aggregation_mode: str = Field(
        default="consensus",
        description="Which 8.6 aggregation mode to consume. Only 'consensus' is wired in v1.0.",
    )
    xsum_top_n: int = Field(
        default=200, ge=10,
        description="Top N genes for XSum computation. NA when overlap < this.",
    )
    xsum_top_n_sweep: Optional[list[int]] = Field(
        default=None,
        description="Reserved for future XSum sweep. Must be None in v1.0.",
    )
    xsum_permutations: int = Field(
        default=0, ge=0,
        description="Number of permutations for XSum p-value. 0 = no permutation.",
    )
    xsum_seed: int = Field(
        default=42,
        description="RNG seed for XSum permutation. Default 42 preserves current byte-for-byte "
                    "behaviour (compute_xsum previously hard-coded default_rng(42)). Only "
                    "consulted when xsum_permutations > 0. Broader RNG architecture (e.g. "
                    "fresh entropy, per-tissue streams) is deferred to a future revision.",
    )
    permutation_calibration: PermutationCalibrationConfig = Field(
        default_factory=PermutationCalibrationConfig,
        description="Opt-in permutation calibration diagnostic. "
                    "Produces negative_correlation/calibration.json when enabled; "
                    "does not modify primary p-values.",
    )

    @field_validator("tissues", mode="before")
    @classmethod
    def default_tissues(cls, v: list[str]) -> list[str]:
        """Use all brain tissues from constants if none specified."""
        if not v:
            from repogen.utils.constants import BRAIN_TISSUES
            return list(BRAIN_TISSUES)
        return v

    @field_validator("correlation_method")
    @classmethod
    def validate_correlation_method(cls, v: str) -> str:
        if v != "spearman":
            raise ValueError(
                f"correlation_method='{v}' is not supported. "
                f"Only 'spearman' is implemented in v1.0."
            )
        return v

    @field_validator("gene_set_mode")
    @classmethod
    def validate_gene_set_mode(cls, v: str) -> str:
        allowed = {"landmark", "landmark_bing", "all"}
        if v not in allowed:
            raise ValueError(f"gene_set_mode must be one of {allowed}, got '{v}'")
        return v

    @field_validator("match_confidence_threshold")
    @classmethod
    def validate_match_confidence(cls, v: str) -> str:
        allowed = {"inchikey", "pubchem_cid", "name"}
        if v not in allowed:
            raise ValueError(f"match_confidence_threshold must be one of {allowed}, got '{v}'")
        return v

    @field_validator("aggregation_mode")
    @classmethod
    def validate_aggregation_mode(cls, v: str) -> str:
        if v != "consensus":
            raise ValueError(
                f"aggregation_mode='{v}' is not yet wired - workflow/input "
                f"selection for alternative 8.6 outputs is not implemented. "
                f"Only 'consensus' is supported in v1.0."
            )
        return v

    @field_validator("xsum_top_n_sweep")
    @classmethod
    def validate_xsum_top_n_sweep(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None:
            raise ValueError(
                "xsum_top_n_sweep is not yet implemented - output contract "
                "for sweep results is undefined. Set to null/None for v1.0."
            )
        return v


class EQTLSourceConfig(BaseModel):
    """Configuration for a single eQTL data source."""

    source: str = Field(description="Source name: 'eqtlgen', 'metabrain_cortex', etc.")
    path: Path = Field(description="Path to eQTL summary statistics directory")
    required: bool = Field(
        default=False,
        description=(
            "If true, the MR run fails loudly when this source's Phase-2 yield "
            "(emitted MR rows / genes processed) is below min_result_fraction, "
            "or below the default floor when min_result_fraction is unset. "
            "Guards against silent source degradation."
        ),
    )
    min_result_fraction: Optional[float] = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "Minimum Phase-2 yield (emitted MR rows / genes processed) for this "
            "source. If set, the run fails when the observed yield is below it "
            "(regardless of 'required'). None disables the explicit threshold."
        ),
    )


class MRDrugMatchConfig(BaseModel):
    """Branch C drug-matching filters and gene-to-drug ID-join strategy.

    Every default reproduces the previous output: a legacy exact-Ensembl->loose-symbol
    join with no phase/potency/druggability filtering and all interaction types
    retained. Provenance columns are added additively regardless of settings.
    """

    match_mode: Literal["legacy", "strict"] = Field(
        default="legacy",
        description=(
            "Gene-to-drug ID join strategy. 'legacy' = exact Ensembl then a loose "
            "gene_symbol fallback (previous behaviour). 'strict' = exact Ensembl "
            "(version-stripped) -> Entrez -> UniProt -> unambiguous symbol (flagged "
            "and demoted); distinct Ensembl IDs are never merged by shared symbol."
        ),
    )
    min_pchembl: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Drop ChEMBL matches whose pchembl_value is below this cutoff "
            "(null-pchembl mechanism rows are retained). None disables the filter."
        ),
    )
    min_phase: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description=(
            "Keep only drugs with max_phase >= min_phase. Semantically equal to "
            "Branch A's misleadingly-named 'max_phase_filter'. None disables the filter."
        ),
    )
    phase_filter_scope: Literal["global", "chembl_only"] = Field(
        default="global",
        description=(
            "Scope of min_phase: 'global' applies to all sources; 'chembl_only' "
            "exempts PDSP/DGIdb (which are always phase 0), mirroring Branch A."
        ),
    )
    direction_policy: Literal["all", "prefer_inferable", "inferable_only"] = Field(
        default="all",
        description=(
            "Handling of direction-inferable interactions (inhibitor/agonist/etc.). "
            "'all' = keep every match; 'prefer_inferable' = keep all but rank "
            "direction-inferable matches first; 'inferable_only' = drop "
            "direction-ambiguous matches (e.g. interaction_type='other')."
        ),
    )
    allow_symbol_fallback: bool = Field(
        default=True,
        description=(
            "Allow gene_symbol as a match tier. In 'legacy' mode this is the loose "
            "fallback; in 'strict' mode it is the last, unambiguous-only, flagged "
            "tier. Set False to require ID-based (Ensembl/Entrez/UniProt) matches."
        ),
    )
    require_druggable: bool = Field(
        default=False,
        description=(
            "If True, keep only matches whose gene carries a druggable_tier "
            "annotation (requires druggable_genome_path). Default False = annotate "
            "only, never gate."
        ),
    )
    druggable_genome_path: Optional[Path] = Field(
        default=None,
        description=(
            "Optional TSV of the druggable genome (columns: gene_ensembl_id, "
            "druggable_tier). When set, adds a druggable_tier column to mr_results "
            "and enables the secondary druggable-restricted significance track "
            ". None = off (druggable_tier stays absent/NA)."
        ),
    )

    @model_validator(mode="after")
    def _validate_require_druggable(self) -> "MRDrugMatchConfig":
        # Fail loud at config time: gating on druggability without a source would
        # silently drop every drug match. Runtime also validates the
        # resource exists / has the right columns.
        if self.require_druggable and self.druggable_genome_path is None:
            raise ValueError(
                "mr.drug_match.require_druggable=True requires "
                "druggable_genome_path to be set (a druggable-genome TSV)."
            )
        return self


class MRMHCSensitivityConfig(BaseModel):
    """Branch C MHC-region sensitivity settings.

    The MHC (chr6, extended) is a long-range-LD region where cis-MR and
    single-variant coloc are notoriously unreliable. When enabled, Branch C
    annotates an additive ``mhc_flag`` on ``mr_results`` (by build-invariant
    Ensembl-ID membership, reusing the Branch B annotation) and emits an
    MHC-excluded sensitivity view under ``mr/sensitivity/mhc_excluded/``.
    Primary ``mr_results`` / ``mr_significant`` / coloc / drug matching are
    never changed. Default off preserves the current output schema.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Enable MHC annotation + MHC-excluded sensitivity output. Default "
            "False = no mhc_flag column and no sensitivity view (unchanged output)."
        ),
    )
    allow_mhc_coordinate_fallback: bool = Field(
        default=False,
        description=(
            "If True, and the canonical Ensembl-membership annotation is "
            "unavailable, fall back to a per-source build-aware coordinate flag "
            "derived from the (possibly min-SNP-position) gene anchor. This is "
            "approximate and clearly labelled mhc_flag_method="
            "'coordinate_fallback_approximate'. Default False = fail loud when "
            "Enabled but the annotation resource is missing."
        ),
    )


class MRConfig(BaseModel):
    """Configuration for Mendelian Randomisation analysis (module 8.8)."""

    eqtl_sources: list[EQTLSourceConfig] = Field(
        default_factory=lambda: [EQTLSourceConfig(source="eqtlgen", path=Path("resources/eqtl/eqtlgen/"))],
        description="List of eQTL sources to use. Default: eQTLGen only.",
    )

    cis_window_kb: int = Field(default=1000, ge=100, description="cis-window in kb from gene TSS (±)")
    instrument_pval: float = Field(default=5e-8, gt=0, lt=1, description="P-value threshold for instrument selection")
    clump_r2: float = Field(default=0.001, gt=0, lt=1, description="LD clumping r² threshold")
    f_stat_threshold: float = Field(default=10.0, ge=1, description="Minimum per-SNP F-statistic")

    coloc_enabled: bool = Field(default=True, description="Run coloc.abf for each significant gene-source pair")
    coloc_variance_mode: Literal["reported_se", "case_control_approx"] = Field(
        default="reported_se",
        description=(
            "coloc.abf outcome variance (V2) source. "
            "'reported_se' (default) uses the GWAS BETA/SE directly (coloc's "
            "beta/varbeta path) - faithful for case/control traits with proper "
            "log-OR SEs and byte-stable with previous output. 'case_control_approx' "
            "derives V2 from total-N (cases+controls, never Neff), MAF and the "
            "sample case fraction s2 - an opt-in sensitivity mode requiring "
            "n_cases/n_controls in the GWAS metadata."
        ),
    )
    coloc_pp_h4_threshold: float = Field(default=0.80, ge=0, le=1, description="PP.H4 threshold for coloc_supported")
    coloc_prior_p1: float = Field(default=1e-4, gt=0, description="Prior prob SNP associated with trait 1")
    coloc_prior_p2: float = Field(default=1e-4, gt=0, description="Prior prob SNP associated with trait 2")
    coloc_prior_p12: float = Field(default=1e-5, gt=0, description="Prior prob SNP associated with both traits")
    min_coloc_snps: int = Field(default=50, ge=10, description="Min shared SNPs for informative coloc")

    require_coloc: bool = Field(default=True, description="Gate drug matching on coloc PP.H4 ≥ threshold")
    require_steiger: bool = Field(default=False, description="Gate drug matching on Steiger directionality")

    n_workers: int = Field(default=1, ge=1, description="Number of parallel workers for per-gene MR")
    rule_threads: int | None = Field(
        default=None, ge=1,
        description="Snakemake rule threads / HPC CPU reservation. Defaults to n_workers if not set.",
    )

    drug_match: MRDrugMatchConfig = Field(
        default_factory=MRDrugMatchConfig,
        description=(
            "Branch C drug-matching filters and ID-join strategy. "
            "Defaults reproduce the previous output."
        ),
    )

    mhc_sensitivity: MRMHCSensitivityConfig = Field(
        default_factory=MRMHCSensitivityConfig,
        description=(
            "Branch C MHC-region sensitivity settings. "
            "Default off preserves the current output schema."
        ),
    )

    @field_validator("eqtl_sources")
    @classmethod
    def validate_eqtl_sources(cls, v: list[EQTLSourceConfig]) -> list[EQTLSourceConfig]:
        if not v:
            raise ValueError("At least one eQTL source must be configured")
        if len(v) > 2:
            raise ValueError(
                f"v1.0 supports at most 2 eQTL sources (got {len(v)}). "
                f"N-source concordance logic is deferred to v2.0."
            )
        names = [s.source for s in v]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate eQTL source names: {names}")
        return v


class OpenTargetsConfig(BaseModel):
    """Open Targets Platform post-analysis annotation settings."""

    enabled: bool = Field(default=True, description="Enable Open Targets annotation")
    disease_efo_id: Optional[str] = Field(default=None, description='EFO disease ID (e.g. "EFO_0003761" for MDD)')
    use_api: bool = Field(default=True, description="Use GraphQL API (vs bulk download)")
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory for API responses")


class ManhattanStyleConfig(BaseModel):
    chromosome_scale: Literal["rank", "bp"] = Field(
        default="rank",
        description="'rank' = equal-area chromosome bands (spacing proportional to gene count); "
                    "'bp' = bp-proportional (legacy behaviour).",
    )
    y_cap: Optional[float] = Field(
        default=20.0, gt=0,
        description="Cap -log10(p) at this value; clipped points shown via upward arrow.",
    )
    highlight_genes: list[str] = Field(
        default_factory=list,
        description="Gene symbols to spotlight (labelled + halo) regardless of FDR rank.",
    )
    chromosome_set: Literal["autosomes", "autosomes+X", "all"] = Field(
        default="autosomes+X",
        description="Which chromosomes to include.",
    )
    n_top_labels: int = Field(default=20, ge=0, le=100)
    locus_window_kb: int = Field(
        default=500, ge=0,
        description="Collapse labels within this window to one per locus (keeps lowest p).",
    )


class VolcanoStyleConfig(BaseModel):
    x_axis_mode: Literal["magma_z", "top_snp_beta"] = Field(
        default="magma_z",
        description="X-axis semantics. 'magma_z' (default) plots raw signed Z as "
                    "produced by MAGMA snp-wise=mean; sign reflects p<0.5 vs p>0.5, "
                    "not effect direction (coloured by FDR only). 'top_snp_beta' "
                    "multiplies |Z| by the sign of the lowest-p SNP's beta from the "
                    "prepared GWAS, enabling a four-quadrant coloured volcano.",
    )
    n_top_labels: int = Field(default=15, ge=0, le=50)
    y_cap: Optional[float] = Field(default=20.0, gt=0)


class DrugEnrichmentStyleConfig(BaseModel):
    size_by: Literal["wilcoxon_auc", "n_target_genes"] = Field(
        default="wilcoxon_auc",
        description="Which column maps to dot size in the drug enrichment plot.",
    )
    top_n: int = Field(default=20, ge=5, le=100)
    show_mechanism: bool = Field(
        default=True,
        description="Render an additional column with mechanism_of_action text.",
    )


class PlotStyleConfig(BaseModel):
    figure_formats: list[Literal["pdf", "svg", "png"]] = Field(
        default_factory=lambda: ["pdf", "svg", "png"],
        description="Which formats to emit for every figure.",
    )
    png_dpi: int = Field(default=600, ge=150, le=1200)
    pdf_dpi: int = Field(
        default=300, ge=150, le=1200,
        description="Raster DPI used for any rasterized layers inside PDF/SVG.",
    )
    max_label_len: int = Field(default=40, ge=10, le=120)
    provenance_footer: bool = Field(
        default=True,
        description="Stamp study/version/date in bottom-right of every figure.",
    )

    manhattan: ManhattanStyleConfig = Field(default_factory=ManhattanStyleConfig)
    volcano: VolcanoStyleConfig = Field(default_factory=VolcanoStyleConfig)
    drug_enrichment: DrugEnrichmentStyleConfig = Field(default_factory=DrugEnrichmentStyleConfig)


class OutputConfig(BaseModel):
    """Output format and visualisation toggles."""

    csv: bool = True
    excel: bool = True
    html: bool = True
    manhattan: bool = True
    qq: bool = True
    enrichment: bool = True
    plot_style: PlotStyleConfig = Field(default_factory=PlotStyleConfig)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration - everything flows from here."""

    study: StudyConfig
    gwas_prep: GWASPrepConfig = Field(default_factory=GWASPrepConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    magma: MagmaConfig = Field(default_factory=MagmaConfig)
    pathway: PathwayConfig = Field(default_factory=PathwayConfig)
    drug_enrichment: DrugEnrichmentConfig = Field(default_factory=DrugEnrichmentConfig)
    atc_enrichment: ATCEnrichmentConfig = Field(default_factory=ATCEnrichmentConfig)
    drug_signatures: DrugSignaturesConfig = Field(default_factory=DrugSignaturesConfig)
    spredixcan: SpredixcanConfig = Field(default_factory=SpredixcanConfig)
    negative_correlation: NegativeCorrelationConfig = Field(default_factory=NegativeCorrelationConfig)
    mr: MRConfig = Field(default_factory=MRConfig)
    open_targets: OpenTargetsConfig = Field(default_factory=OpenTargetsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    resource_dir: Path = Field(
        default=Path("resources"),
        description=(
            "Root directory holding downloaded reference data (drug databases, "
            "LINCS signatures, pathway GMTs, eQTL summary statistics). Set this "
            "when the data lives outside the working directory, e.g. on shared "
            "cluster storage. Corresponds to '<setup-resources --target-dir>/resources', "
            "because local_path entries in resources.yaml already carry the "
            "'resources/' prefix."
        ),
    )
    output_dir: Path = Field(default=Path("results"), description="Root directory for all pipeline outputs")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Optional log file path")
