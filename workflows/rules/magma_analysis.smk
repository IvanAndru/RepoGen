rule magma_gene:
    input:
        gwas=f"{PREP_DIR}/gwas_standardized.parquet",
        gene_annot=f"{GENE_ANNOT_DIR}/gene_annotations.parquet",
    output:
        parquet=f"{MAGMA_DIR}/{STUDY}_gene_results.parquet",
        genes_raw=f"{MAGMA_DIR}/{STUDY}.genes.raw",
        genes_out=f"{MAGMA_DIR}/{STUDY}.genes.out",
        genes_annot=f"{MAGMA_DIR}/{STUDY}.genes.annot",
    params:
        reference_bfile=str(
            Path(config.get("reference", {}).get("genome_dir", REFERENCE_RES_DIR))
            / config.get("reference", {}).get("bfile_prefix", "g1000_eur")
        ),
        gene_loc=config.get("reference", {}).get("gene_loc_file", f"{REFERENCE_RES_DIR}/NCBI37.3.gene.loc"),
        resources_dir="resources",
        config_path=CONFIG_PATH,
        study_name=STUDY,
        output_root=OUTPUT_ROOT,
        source_parquet=f"{MAGMA_DIR}/{STUDY}.genes.parquet",
    threads: 4
    resources:
        runtime=240,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/magma_gene.log",
    shell:
        """
        python -m repogen.analysis.magma_gene \
            --gwas {input.gwas} \
            --gene-annotations {input.gene_annot} \
            --reference-bfile {params.reference_bfile} \
            --gene-loc {params.gene_loc} \
            --resources-dir {params.resources_dir} \
            --config {params.config_path} \
            --output-dir {params.output_root} \
            --study-name {params.study_name} \
            2>&1 | tee {log}

        # K2: move/rename MAGMA gene parsed parquet to canonical name
        if [ -f "{params.source_parquet}" ] && [ "{params.source_parquet}" != "{output.parquet}" ]; then
            mv "{params.source_parquet}" "{output.parquet}"
        fi

        # Fail-fast if canonical output is missing
        if [ ! -f "{output.parquet}" ]; then
            echo "ERROR: canonical gene results parquet not found at {output.parquet}" >&2
            exit 1
        fi
        """


rule magma_pathway:
    input:
        genes_raw=f"{MAGMA_DIR}/{STUDY}.genes.raw",
        gene_results=f"{MAGMA_DIR}/{STUDY}_gene_results.parquet",
        gene_sets=f"{GENE_ANNOT_DIR}/gene_sets.parquet",
    output:
        parquet=f"{MAGMA_DIR}/{STUDY}_pathway_results.parquet",
        meta_json=f"{MAGMA_DIR}/{STUDY}_pathway_results_meta.json",
    params:
        config_path=CONFIG_PATH,
        output_root=OUTPUT_ROOT,
        source_parquet=f"{OUTPUT_ROOT}/{STUDY}_pathway_results.parquet",
        source_meta=f"{OUTPUT_ROOT}/{STUDY}_pathway_results_meta.json",
    threads: 2
    resources:
        runtime=240,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/magma_pathway.log",
    shell:
        """
        python -m repogen.analysis.magma_pathway \
            --config {params.config_path} \
            --gene-results-raw {input.genes_raw} \
            --gene-results-parquet {input.gene_results} \
            --gene-sets-parquet {input.gene_sets} \
            --output-dir {params.output_root} \
            2>&1 | tee {log}

        # K2: move pathway parquet and metadata from output root into magma/
        if [ -f "{params.source_parquet}" ]; then
            mv "{params.source_parquet}" "{output.parquet}"
        fi
        if [ -f "{params.source_meta}" ]; then
            mv "{params.source_meta}" "{output.meta_json}"
        fi

        # Fail-fast: canonical destinations must exist
        if [ ! -f "{output.parquet}" ]; then
            echo "ERROR: pathway parquet not found at {output.parquet}" >&2
            exit 1
        fi
        if [ ! -f "{output.meta_json}" ]; then
            echo "ERROR: pathway metadata JSON not found at {output.meta_json}" >&2
            exit 1
        fi

        # Root-level files must not remain after move
        if [ -f "{params.source_parquet}" ]; then
            echo "ERROR: root-level pathway parquet still exists after move" >&2
            exit 1
        fi
        """
