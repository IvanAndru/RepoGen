def _get_gene_loc_file(wildcards):
    """Lazy resolver for gene location file - defers config lookup to DAG time."""
    gene_loc = config.get("reference", {}).get("gene_loc_file")
    if not gene_loc:
        raise WorkflowError(
            "reference.gene_loc_file is not set in the config. "
            "Provide it in your config YAML or reference.yaml."
        )
    return gene_loc


rule annotate_genes:
    input:
        gene_loc=_get_gene_loc_file,
    output:
        parquet=f"{GENE_ANNOT_DIR}/gene_annotations.parquet",
    params:
        window_kb=config.get("magma", {}).get("window_upstream_kb", 35),
        mapping_dir_flag=opt_flag(
            "--mapping-dir",
            str(Path(config["reference"]["ensembl_to_name"]).parent)
            if config.get("reference", {}).get("ensembl_to_name")
            else None,
        ),
        gene_info_flag=opt_flag(
            "--gene-info", config.get("reference", {}).get("ncbi_gene_info")
        ),
        gene_history_flag=opt_flag(
            "--gene-history", config.get("reference", {}).get("ncbi_gene_history")
        ),
        biotype_flag=(
            " ".join(
                ["--biotype-filter"]
                + config.get("magma", {}).get("biotype_filter", [])
            )
            if config.get("magma", {}).get("biotype_filter")
            else ""
        ),
    threads: 1
    resources:
        runtime=60,
        mem_mb=16000,
    log:
        f"{LOG_DIR}/annotate_genes.log",
    shell:
        """
        python -m repogen.data.gene_annotation \
            --gene-loc {input.gene_loc} \
            --output {output.parquet} \
            --window-kb {params.window_kb} \
            {params.mapping_dir_flag} \
            {params.gene_info_flag} \
            {params.gene_history_flag} \
            {params.biotype_flag} \
            2>&1 | tee {log}
        """


def _resolve_default_gmts():
    """Return default GMT file paths that exist on disk."""
    defaults = [
        f"{PATHWAY_RES_DIR}/c5.all.v2024.1.Hs.symbols.gmt",
        f"{PATHWAY_RES_DIR}/c2.cp.v2024.1.Hs.symbols.gmt",
    ]
    return [g for g in defaults if os.path.isfile(g)]


def _resolve_all_gmts():
    """Return all GMT files (defaults + extra from config)."""
    gmts = _resolve_default_gmts()
    extras = config.get("pathway", {}).get("extra_gmt_files", [])
    if extras:
        gmts.extend([str(e) for e in extras])
    if not gmts:
        raise WorkflowError(
            "No GMT files found: default MSigDB files are missing and no "
            "extra_gmt_files configured. Run `repogen setup-resources` first."
        )
    return gmts


rule prepare_gene_sets:
    input:
        gmts=lambda wildcards: _resolve_all_gmts(),
    output:
        parquet=f"{GENE_ANNOT_DIR}/gene_sets.parquet",
    params:
        min_size=config.get("pathway", {}).get("min_set_size", 10),
        max_size=config.get("pathway", {}).get("max_set_size", 500),
        sources_flag=(
            " ".join(
                ["--sources"]
                + config.get("pathway", {}).get("sources", [])
            )
            if config.get("pathway", {}).get("sources")
            else ""
        ),
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/prepare_gene_sets.log",
    shell:
        """
        python -m repogen.data.gene_sets \
            --gmt {input.gmts} \
            --output {output.parquet} \
            --min-size {params.min_size} \
            --max-size {params.max_size} \
            {params.sources_flag} \
            2>&1 | tee {log}
        """
