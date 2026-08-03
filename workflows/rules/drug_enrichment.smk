def _configured_drug_sources():
    """Resolve drug source file paths based on config.drug_enrichment.sources.

    Returns dict mapping source name to file path (or None if not requested).
    Raises WorkflowError if a requested source file is missing.
    """
    sources = config.get("drug_enrichment", {}).get("sources", ["chembl"])
    paths = {
        "pdsp": f"{DRUGS_RES_DIR}/pdsp_ki.csv",
        "dgidb": f"{DRUGS_RES_DIR}/dgidb_interactions.tsv",
    }
    result = {}
    for name, path in paths.items():
        if name in sources:
            if not os.path.isfile(path):
                raise WorkflowError(
                    f"Drug source '{name}' is listed in drug_enrichment.sources "
                    f"but the required file is missing: {path}"
                )
            result[name] = path
        else:
            result[name] = None
    return result

_DRUG_SOURCES = _configured_drug_sources()


# Expression-perturbation source plumbing.
#
# Mode OFF (default): _EXPR_ENABLED is False; _EXPR_PATHS is {}.
#   Only the existing `load_drug_targets` rule contributes to the DAG.
#   Branch B/C and Branch A all read `drug_targets.parquet` (TARGETS-only).
#   DAG, output filenames, and downstream behaviour are byte-identical
# to the previous baseline.
#
# Mode ON: _EXPR_ENABLED is True; _EXPR_PATHS lists the resolved
#   resource paths for the requested expression sources.  An additional
#   `load_drug_targets_expr` rule produces drug_targets_expr.parquet.
#   The `drug_enrichment` rule's input.drug_targets path switches to
#   drug_targets_expr.parquet (see _drug_targets_input below); Branch
#   B/C continue to consume drug_targets.parquet so their behaviour is
#   strictly unaffected by this flag.
_EXPR_ENABLED = bool(
    config.get("drug_enrichment", {}).get(
        "enable_expression_enrichment", False
    )
)


def _configured_expression_sources():
    """Resolve expression source file paths based on config.

    Returns dict mapping expression source name -> resource file path.
    Returns {} when ``enable_expression_enrichment`` is False (mode OFF).
    Raises ``WorkflowError`` for unknown source names or missing files
    so misconfiguration is surfaced at config-load time, not at run
    time.
    """
    if not _EXPR_ENABLED:
        return {}
    sources = (
        config.get("drug_enrichment", {}).get("expression_sources", []) or []
    )
    paths = {
        "creeds": f"{PERTURBATION_RES_DIR}/creeds.json",
        "dsigdb": f"{PERTURBATION_RES_DIR}/dsigdb_d3.tsv",
    }
    result = {}
    for name in sources:
        if name not in paths:
            raise WorkflowError(
                f"Unknown expression source '{name}'. "
                f"Allowed: {sorted(paths)}."
            )
        if not os.path.isfile(paths[name]):
            raise WorkflowError(
                f"Expression source '{name}' is listed in "
                f"drug_enrichment.expression_sources but the required "
                f"file is missing: {paths[name]}.  Run: "
                f"`repogen setup-resources --pipeline-config <config.yaml>` "
                f"or `python -m repogen.data.resources --pipeline-config "
                f"<config.yaml>` to fetch it."
            )
        result[name] = paths[name]
    return result


_EXPR_PATHS = _configured_expression_sources()


def _drug_targets_input():
    """Mode-switch helper: which drug_targets parquet drug_enrichment reads.

    Mode OFF -> ``{DRUG_DIR}/drug_targets.parquet``        (TARGETS-only)
    Mode ON  -> ``{DRUG_DIR}/drug_targets_expr.parquet``   (TARGETS+EXPR)

    Output filenames downstream of `drug_enrichment` are byte-identical
    in both modes; only the *input* parquet differs.  Branch B and
    Branch C consumers ALWAYS read ``drug_targets.parquet`` (which is
    always TARGETS-only) so they are unaffected by this switch.
    """
    if _EXPR_ENABLED:
        return f"{DRUG_DIR}/drug_targets_expr.parquet"
    return f"{DRUG_DIR}/drug_targets.parquet"


rule load_drug_targets:
    input:
        chembl=f"{DRUGS_RES_DIR}/chembl_35.db",
    output:
        parquet=f"{DRUG_DIR}/drug_targets.parquet",
    params:
        biomart_dir=(
            str(Path(config["reference"]["ensembl_to_name"]).parent)
            if config.get("reference", {}).get("ensembl_to_name")
            else REFERENCE_RES_DIR
        ),
        gene_info_flag=opt_flag(
            "--gene-info", config.get("reference", {}).get("ncbi_gene_info")
        ),
        gene_history_flag=opt_flag(
            "--gene-history", config.get("reference", {}).get("ncbi_gene_history")
        ),
        pdsp_flag=opt_flag("--pdsp-csv", _DRUG_SOURCES["pdsp"]),
        dgidb_flag=opt_flag("--dgidb-tsv", _DRUG_SOURCES["dgidb"]),
        chembl_scope_flag=f"--chembl-scope {config.get('drug_enrichment', {}).get('chembl_scope', 'mechanism_only')}",
        unichem_flag=opt_flag(
            "--unichem-mapping",
            f"{DRUGS_RES_DIR}/src1src22.txt.gz"
            if os.path.isfile(f"{DRUGS_RES_DIR}/src1src22.txt.gz") else None,
        ),
        # Parent/salt unification toggle. Defaults
        # to enabled; set drug_enrichment.parent_salt_unification:
        # false in config.yaml to disable (rollback / debugging).
        psu_flag=(
            "--parent-salt-unification"
            if config.get("drug_enrichment", {}).get(
                "parent_salt_unification", True
            )
            else "--no-parent-salt-unification"
        ),
    threads: 1
    resources:
        runtime=120,
        mem_mb=32000,
    log:
        f"{LOG_DIR}/load_drug_targets.log",
    shell:
        """
        python -m repogen.data.drug_loader \
            --chembl-sqlite {input.chembl} \
            --output {output.parquet} \
            --biomart-dir {params.biomart_dir} \
            {params.gene_info_flag} \
            {params.gene_history_flag} \
            {params.pdsp_flag} \
            {params.dgidb_flag} \
            {params.chembl_scope_flag} \
            {params.unichem_flag} \
            {params.psu_flag} \
            2>&1 | tee {log}
        """


rule load_drug_targets_expr:
    """Load TARGETS+EXPR drug-target evidence for Branch A only.

    Only invoked when
    ``drug_enrichment.enable_expression_enrichment: true`` and at least
    one entry in ``drug_enrichment.expression_sources`` resolves to a
    present file (gated via ``_configured_expression_sources``).

    The output parquet is consumed *only* by ``drug_enrichment`` (see
    ``_drug_targets_input``); Branch B (extract_drug_signatures,
    negative_correlation) and Branch C (mendelian_randomisation)
    continue to read the regular ``drug_targets.parquet`` so they stay
    on TARGETS-only data regardless of this flag.
    """
    input:
        chembl=f"{DRUGS_RES_DIR}/chembl_35.db",
    output:
        parquet=f"{DRUG_DIR}/drug_targets_expr.parquet",
    params:
        biomart_dir=(
            str(Path(config["reference"]["ensembl_to_name"]).parent)
            if config.get("reference", {}).get("ensembl_to_name")
            else REFERENCE_RES_DIR
        ),
        gene_info_flag=opt_flag(
            "--gene-info", config.get("reference", {}).get("ncbi_gene_info")
        ),
        gene_history_flag=opt_flag(
            "--gene-history", config.get("reference", {}).get("ncbi_gene_history")
        ),
        pdsp_flag=opt_flag("--pdsp-csv", _DRUG_SOURCES["pdsp"]),
        dgidb_flag=opt_flag("--dgidb-tsv", _DRUG_SOURCES["dgidb"]),
        chembl_scope_flag=f"--chembl-scope {config.get('drug_enrichment', {}).get('chembl_scope', 'mechanism_only')}",
        unichem_flag=opt_flag(
            "--unichem-mapping",
            f"{DRUGS_RES_DIR}/src1src22.txt.gz"
            if os.path.isfile(f"{DRUGS_RES_DIR}/src1src22.txt.gz") else None,
        ),
        psu_flag=(
            "--parent-salt-unification"
            if config.get("drug_enrichment", {}).get(
                "parent_salt_unification", True
            )
            else "--no-parent-salt-unification"
        ),
        # Config-driven expression-source resolution.
        # The flag value is computed from _EXPR_PATHS so a config update
        # propagates cleanly; nothing is hardcoded in this shell block.
        expression_sources_flag=(
            "--expression-sources " + ",".join(sorted(_EXPR_PATHS.keys()))
            if _EXPR_PATHS else ""
        ),
        creeds_flag=opt_flag("--creeds-path", _EXPR_PATHS.get("creeds")),
        dsigdb_flag=opt_flag("--dsigdb-path", _EXPR_PATHS.get("dsigdb")),
    threads: 1
    resources:
        runtime=120,
        mem_mb=32000,
    log:
        f"{LOG_DIR}/load_drug_targets_expr.log",
    shell:
        """
        python -m repogen.data.drug_loader \
            --chembl-sqlite {input.chembl} \
            --output {output.parquet} \
            --biomart-dir {params.biomart_dir} \
            {params.gene_info_flag} \
            {params.gene_history_flag} \
            {params.pdsp_flag} \
            {params.dgidb_flag} \
            {params.chembl_scope_flag} \
            {params.unichem_flag} \
            {params.psu_flag} \
            {params.expression_sources_flag} \
            {params.creeds_flag} \
            {params.dsigdb_flag} \
            2>&1 | tee {log}
        """


rule drug_enrichment:
    input:
        genes_raw=f"{MAGMA_DIR}/{STUDY}.genes.raw",
        gene_results=f"{MAGMA_DIR}/{STUDY}_gene_results.parquet",
        # Mode-switch via helper. Mode OFF ->
        # drug_targets.parquet (TARGETS-only); mode ON ->
        # drug_targets_expr.parquet (TARGETS+EXPR).  Output filenames
        # downstream are unchanged in both modes.
        drug_targets=_drug_targets_input(),
    output:
        parquet=f"{DRUG_DIR}/{STUDY}_drug_enrichment.parquet",
        meta_json=f"{DRUG_DIR}/{STUDY}_drug_enrichment_metadata.json",
        genesets=f"{DRUG_DIR}/{STUDY}_drug_genesets.txt",
    params:
        config_path=CONFIG_PATH,
        study_name=STUDY,
        study_dir=STUDY_DIR,
    threads: 4
    resources:
        runtime=120,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/drug_enrichment.log",
    shell:
        """
        python -m repogen.analysis.drug_enrichment \
            --gene-results-raw {input.genes_raw} \
            --gene-results-parquet {input.gene_results} \
            --drug-targets {input.drug_targets} \
            --config {params.config_path} \
            --output-dir {params.study_dir} \
            --study-name {params.study_name} \
            2>&1 | tee {log}
        """


rule atc_enrichment:
    input:
        drug_results=f"{DRUG_DIR}/{STUDY}_drug_enrichment.parquet",
        genes_raw=f"{MAGMA_DIR}/{STUDY}.genes.raw",
        drug_geneset=f"{DRUG_DIR}/{STUDY}_drug_genesets.txt",
    output:
        parquet=f"{ATC_DIR}/atc_enrichment_results.parquet",
        meta_json=f"{ATC_DIR}/atc_enrichment_metadata.json",
    params:
        config_path=CONFIG_PATH,
        study_name=STUDY,
        output_root=OUTPUT_ROOT,
    threads: 1
    resources:
        runtime=120,
        mem_mb=32000,
    log:
        f"{LOG_DIR}/atc_enrichment.log",
    shell:
        """
        python -m repogen.analysis.atc_enrichment \
            --drug-results {input.drug_results} \
            --genes-raw {input.genes_raw} \
            --drug-geneset {input.drug_geneset} \
            --config {params.config_path} \
            --output-dir {params.output_root} \
            --study-name {params.study_name} \
            2>&1 | tee {log}
        """
