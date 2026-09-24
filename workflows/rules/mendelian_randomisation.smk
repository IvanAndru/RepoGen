rule mendelian_randomisation:
    input:
        gwas=f"{PREP_DIR}/gwas_standardized.parquet",
        gwas_meta=f"{PREP_DIR}/gwas_standardized.meta.json",
        drug_targets=f"{DRUG_DIR}/drug_targets.parquet",
    output:
        results=f"{MR_DIR}/mr_results.parquet",
        drugs=f"{MR_DIR}/mr_drug_matches.parquet",
        verdicts=f"{MR_DIR}/mr_target_verdicts.parquet",
        meta_json=f"{MR_DIR}/mr_metadata.json",
    params:
        config_path=CONFIG_PATH,
        mr_dir=MR_DIR,
    threads: int(config.get("mr", {}).get("rule_threads") or config.get("mr", {}).get("n_workers", 4))
    resources:
        runtime=720,
        mem_mb=32000,
    log:
        f"{LOG_DIR}/mendelian_randomisation.log",
    shell:
        """
        python -m repogen.analysis.mendelian_randomisation \
            --config {params.config_path} \
            --gwas {input.gwas} \
            --gwas-metadata {input.gwas_meta} \
            --drug-targets {input.drug_targets} \
            --output-dir {params.mr_dir} \
            --threads {threads} \
            2>&1 | tee {log}
        """
