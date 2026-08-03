rule prepare_gwas_spredixcan:
    input:
        gwas=config["study"]["gwas_input"],
    output:
        parquet=f"{PREP_DIR}/gwas_spredixcan.parquet",
        meta=f"{PREP_DIR}/gwas_spredixcan.meta.json",
    params:
        info_threshold=config.get("gwas_prep", {}).get("info_threshold", 0.6),
        maf_threshold=config.get("gwas_prep", {}).get("maf_threshold", 0.01),
        genome_build_flag=opt_flag(
            "--genome-build", config.get("study", {}).get("genome_build")
        ),
        trait_type_flag=opt_flag(
            "--trait-type", config.get("study", {}).get("trait_type")
        ),
        sample_size_flag=opt_flag(
            "--sample-size", config.get("study", {}).get("sample_size")
        ),
        n_cases_flag=opt_flag(
            "--n-cases", config.get("study", {}).get("n_cases")
        ),
        n_controls_flag=opt_flag(
            "--n-controls", config.get("study", {}).get("n_controls")
        ),
        chain_flag=opt_flag(
            "--chain-file", config.get("reference", {}).get("liftover_chain")
        ),
        target_build="GRCh38",
    threads: 1
    resources:
        runtime=120,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/prepare_gwas_spredixcan.log",
    shell:
        """
        python -m repogen.data.gwas_prep \
            --spredixcan-prep \
            --input {input.gwas} \
            --output {output.parquet} \
            --metadata-out {output.meta} \
            --info-threshold {params.info_threshold} \
            --maf-threshold {params.maf_threshold} \
            --target-build {params.target_build} \
            {params.genome_build_flag} \
            {params.trait_type_flag} \
            {params.sample_size_flag} \
            {params.n_cases_flag} \
            {params.n_controls_flag} \
            {params.chain_flag} \
            2>&1 | tee {log}
        """


rule spredixcan:
    input:
        gwas=f"{PREP_DIR}/gwas_spredixcan.parquet",
    output:
        per_tissue=f"{SPX_DIR}/spredixcan_per_tissue.parquet",
        meta=f"{SPX_DIR}/spredixcan_meta_analysis.parquet",
        meta_json=f"{SPX_DIR}/spredixcan_metadata.json",
    params:
        config_path=CONFIG_PATH,
        spx_dir=SPX_DIR,
    threads: 4
    resources:
        runtime=360,
        mem_mb=16000,
        branch_b_heavy=1,
    log:
        f"{LOG_DIR}/spredixcan.log",
    benchmark:
        f"{BENCH_DIR}/spredixcan.tsv"
    shell:
        """
        python -m repogen.analysis.spredixcan \
            --gwas-path {input.gwas} \
            --config-path {params.config_path} \
            --output-dir {params.spx_dir} \
            2>&1 | tee {log}
        """


rule extract_drug_signatures:
    input:
        gctx=f"{SIGNATURE_RES_DIR}/level5_beta_trt_cp_n720216x12328.gctx",
        compound_meta=f"{SIGNATURE_RES_DIR}/repurposing_hub.csv",
        gene_meta=f"{SIGNATURE_RES_DIR}/geneinfo_beta.txt",
        drug_targets=f"{DRUG_DIR}/drug_targets.parquet",
    output:
        parquet=f"{NC_DIR}/drug_signatures.parquet",
    params:
        aggregation=config.get("drug_signatures", {}).get("aggregation", "consensus"),
        min_profiles=config.get("drug_signatures", {}).get("min_profiles", 3),
        # cell-line weighting knobs. Defaults preserve previous
        # behaviour byte-for-byte (uniform mode, no neural weighting).
        #
        # the raw-YAML path must honour
        # ``include_neural_tumor_cell_lines`` when ``neural_cell_lines`` is
        # unset.  ``r4_neural_cell_lines_flag`` (defined in common.smk)
        # delegates to ``resolve_neural_cell_lines_from_yaml`` - the same
        # helper that ``DrugSignaturesConfig`` uses - so this rule and the
        # Pydantic validator share a single source of truth.  For non-
        # uniform modes we ALWAYS pass the resolved list explicitly; the
        # CLI fallback is intentionally never reached from Snakemake.
        cell_line_weighting=config.get("drug_signatures", {}).get("cell_line_weighting", "uniform"),
        neural_cell_lines_flag=r4_neural_cell_lines_flag(
            config.get("drug_signatures", {})
        ),
        neural_weight=config.get("drug_signatures", {}).get("neural_weight", 3.0),
        min_neural_profiles_flag=(
            f"--min-neural-profiles {config['drug_signatures']['min_neural_profiles']}"
            if config.get("drug_signatures", {}).get("min_neural_profiles") is not None
            else ""
        ),
    threads: 4
    resources:
        runtime=480,
        mem_mb=32000,
        branch_b_heavy=1,
    log:
        f"{LOG_DIR}/extract_drug_signatures.log",
    benchmark:
        f"{BENCH_DIR}/extract_drug_signatures.tsv"
    shell:
        """
        python -m repogen.data.drug_signatures \
            --gctx {input.gctx} \
            --compound-metadata {input.compound_meta} \
            --gene-metadata {input.gene_meta} \
            --drug-targets {input.drug_targets} \
            --output {output.parquet} \
            --aggregation {params.aggregation} \
            --min-profiles {params.min_profiles} \
            --cell-line-weighting {params.cell_line_weighting} \
            --neural-weight {params.neural_weight} \
            {params.neural_cell_lines_flag} \
            {params.min_neural_profiles_flag} \
            2>&1 | tee {log}
        """


rule negative_correlation:
    input:
        disease_sig=f"{SPX_DIR}/spredixcan_per_tissue.parquet",
        drug_sigs=f"{NC_DIR}/drug_signatures.parquet",
        drug_targets=f"{DRUG_DIR}/drug_targets.parquet",
    output:
        per_tissue=f"{NC_DIR}/per_tissue_results.parquet",
        summary=f"{NC_DIR}/drug_summary.parquet",
        meta_json=f"{NC_DIR}/metadata.json",
    params:
        config_path=CONFIG_PATH,
        nc_dir=NC_DIR,
    threads: 4
    resources:
        runtime=120,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/negative_correlation.log",
    shell:
        """
        python -m repogen.analysis.negative_correlation \
            --disease-signature {input.disease_sig} \
            --drug-signatures {input.drug_sigs} \
            --drug-targets {input.drug_targets} \
            --config {params.config_path} \
            --output-dir {params.nc_dir} \
            2>&1 | tee {log}
        """
