rule prepare_gwas:
    input:
        gwas=config["study"]["gwas_input"],
    output:
        parquet=f"{PREP_DIR}/gwas_standardized.parquet",
        meta=f"{PREP_DIR}/gwas_standardized.meta.json",
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
        population_prevalence_flag=opt_flag(
            "--population-prevalence", config.get("study", {}).get("population_prevalence")
        ),
        liftover_flag=opt_flag(
            "--liftover-to", config.get("gwas_prep", {}).get("liftover_to")
        ),
        reference_bim_flag=opt_flag(
            "--reference-bim",
            str(
                Path(config.get("reference", {}).get("genome_dir", REFERENCE_RES_DIR))
                / (config.get("reference", {}).get("bfile_prefix", "g1000_eur") + ".bim")
            )
            if config.get("reference", {}).get("genome_dir")
            else None,
        ),
        chain_flag=opt_flag(
            "--chain-file", config.get("reference", {}).get("liftover_chain")
        ),
        remove_mhc_flag=(
            "--remove-mhc"
            if config.get("gwas_prep", {}).get("remove_mhc", False)
            else ""
        ),
    threads: 1
    resources:
        runtime=30,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/prepare_gwas.log",
    shell:
        """
        python -m repogen.data.gwas_prep \
            --input {input.gwas} \
            --output {output.parquet} \
            --metadata-out {output.meta} \
            --info-threshold {params.info_threshold} \
            --maf-threshold {params.maf_threshold} \
            {params.genome_build_flag} \
            {params.trait_type_flag} \
            {params.sample_size_flag} \
            {params.n_cases_flag} \
            {params.n_controls_flag} \
            {params.population_prevalence_flag} \
            {params.liftover_flag} \
            {params.reference_bim_flag} \
            {params.chain_flag} \
            {params.remove_mhc_flag} \
            2>&1 | tee {log}
        """
