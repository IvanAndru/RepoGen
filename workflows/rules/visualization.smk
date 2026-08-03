"""Figure, export, aggregation and report rules.

Each rule shells out to a CLI entry point in the package
(``repogen.plotting.cli`` / ``repogen.reporting.cli``) rather than embedding
Python here. Snakemake can only apply ``conda:``/``container:`` directives to
rules that shell out, so this is what allows the whole workflow to run inside
a container; it also keeps the orchestration importable and testable.

Optional figures declare only their status marker as an output; the figure
path is passed as a parameter. That is deliberate - a figure that cannot be
drawn (empty table, missing column, too few branches) must not become a
missing-output error, and the marker records why it was skipped.
"""


# ---------------------------------------------------------------------------
# 11.1  MAGMA group
# ---------------------------------------------------------------------------

rule plot_magma_gene:
    input:
        parquet=f"{MAGMA_DIR}/{STUDY}_gene_results.parquet",
        annot=f"{MAGMA_DIR}/{STUDY}.genes.annot",
        prepared_gwas=f"{PREP_DIR}/gwas_standardized.parquet",
    output:
        manhattan=f"{PLOTS_DIR}/manhattan.png",
        qq=f"{PLOTS_DIR}/qq.png",
        volcano=f"{PLOTS_DIR}/volcano.png",
        status=f"{PLOT_STATUS_DIR}/plot_magma_gene.json",
    params:
        config_path=CONFIG_PATH,
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_magma_gene.log",
    shell:
        """
        python -m repogen.plotting.cli magma-gene \
            --config {params.config_path} \
            --gene-results {input.parquet} \
            --annot {input.annot} \
            --prepared-gwas {input.prepared_gwas} \
            --manhattan {output.manhattan} \
            --qq {output.qq} \
            --volcano {output.volcano} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule plot_magma_pathway:
    input:
        parquet=f"{MAGMA_DIR}/{STUDY}_pathway_results.parquet",
    output:
        figure=f"{PLOTS_DIR}/pathway_enrichment.png",
        status=f"{PLOT_STATUS_DIR}/plot_magma_pathway.json",
    params:
        config_path=CONFIG_PATH,
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_magma_pathway.log",
    shell:
        """
        python -m repogen.plotting.cli magma-pathway \
            --config {params.config_path} \
            --pathway-results {input.parquet} \
            --figure {output.figure} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule export_magma:
    input:
        gene=f"{MAGMA_DIR}/{STUDY}_gene_results.parquet",
        pathway=f"{MAGMA_DIR}/{STUDY}_pathway_results.parquet",
    output:
        gene_csv=f"{EXPORT_DIR}/magma/{STUDY}_gene.csv",
        gene_json=f"{EXPORT_DIR}/magma/{STUDY}_gene.json",
        gene_xlsx=f"{EXPORT_DIR}/magma/{STUDY}_gene.xlsx",
        pathway_csv=f"{EXPORT_DIR}/magma/{STUDY}_pathway.csv",
        pathway_json=f"{EXPORT_DIR}/magma/{STUDY}_pathway.json",
        pathway_xlsx=f"{EXPORT_DIR}/magma/{STUDY}_pathway.xlsx",
    params:
        config_path=CONFIG_PATH,
        out_dir=f"{EXPORT_DIR}/magma",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/export_magma.log",
    shell:
        """
        python -m repogen.reporting.cli export-magma \
            --config {params.config_path} \
            --gene {input.gene} \
            --pathway {input.pathway} \
            --output-dir {params.out_dir} \
            2>&1 | tee {log}
        """


# ---------------------------------------------------------------------------
# 11.2  Drug group
# ---------------------------------------------------------------------------

rule plot_drug_enrichment:
    input:
        parquet=f"{DRUG_DIR}/{STUDY}_drug_enrichment.parquet",
        meta_json=f"{DRUG_DIR}/{STUDY}_drug_enrichment_metadata.json",
    output:
        figure=f"{PLOTS_DIR}/drug_enrichment.png",
        status=f"{PLOT_STATUS_DIR}/plot_drug_enrichment.json",
    params:
        config_path=CONFIG_PATH,
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_drug_enrichment.log",
    shell:
        """
        python -m repogen.plotting.cli drug-enrichment \
            --config {params.config_path} \
            --drug-results {input.parquet} \
            --metadata {input.meta_json} \
            --figure {output.figure} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule plot_atc_enrichment:
    input:
        parquet=f"{ATC_DIR}/atc_enrichment_results.parquet",
    output:
        status=f"{PLOT_STATUS_DIR}/plot_atc_enrichment.json",
    params:
        config_path=CONFIG_PATH,
        figure=f"{PLOTS_DIR}/atc_enrichment.png",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_atc_enrichment.log",
    shell:
        """
        python -m repogen.plotting.cli atc-enrichment \
            --config {params.config_path} \
            --atc-results {input.parquet} \
            --figure {params.figure} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule export_drug:
    input:
        drug=f"{DRUG_DIR}/{STUDY}_drug_enrichment.parquet",
        atc=f"{ATC_DIR}/atc_enrichment_results.parquet",
    output:
        drug_csv=f"{EXPORT_DIR}/drug/{STUDY}_drug.csv",
        drug_json=f"{EXPORT_DIR}/drug/{STUDY}_drug.json",
        drug_xlsx=f"{EXPORT_DIR}/drug/{STUDY}_drug.xlsx",
        atc_csv=f"{EXPORT_DIR}/drug/{STUDY}_atc.csv",
        atc_json=f"{EXPORT_DIR}/drug/{STUDY}_atc.json",
        atc_xlsx=f"{EXPORT_DIR}/drug/{STUDY}_atc.xlsx",
    params:
        config_path=CONFIG_PATH,
        out_dir=f"{EXPORT_DIR}/drug",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/export_drug.log",
    shell:
        """
        python -m repogen.reporting.cli export-drug \
            --config {params.config_path} \
            --drug {input.drug} \
            --atc {input.atc} \
            --output-dir {params.out_dir} \
            2>&1 | tee {log}
        """


# ---------------------------------------------------------------------------
# 11.3  Correlation group
# ---------------------------------------------------------------------------

rule plot_tissue_signature:
    input:
        parquet=f"{SPX_DIR}/spredixcan_per_tissue.parquet",
    output:
        figure=f"{PLOTS_DIR}/tissue_signature.png",
        status=f"{PLOT_STATUS_DIR}/plot_tissue_signature.json",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_tissue_signature.log",
    shell:
        """
        python -m repogen.plotting.cli tissue-signature \
            --per-tissue {input.parquet} \
            --figure {output.figure} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule plot_correlation:
    input:
        parquet=f"{NC_DIR}/per_tissue_results.parquet",
    output:
        scatter=f"{PLOTS_DIR}/correlation_scatter.png",
        heatmap=f"{PLOTS_DIR}/correlation_heatmap.png",
        status=f"{PLOT_STATUS_DIR}/plot_correlation.json",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_correlation.log",
    shell:
        """
        python -m repogen.plotting.cli correlation \
            --per-tissue {input.parquet} \
            --scatter {output.scatter} \
            --heatmap {output.heatmap} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule export_correlation:
    input:
        spx_meta=f"{SPX_DIR}/spredixcan_meta_analysis.parquet",
        spx_tissue=f"{SPX_DIR}/spredixcan_per_tissue.parquet",
        nc_summary=f"{NC_DIR}/drug_summary.parquet",
        nc_tissue=f"{NC_DIR}/per_tissue_results.parquet",
    output:
        spx_csv=f"{EXPORT_DIR}/correlation/{STUDY}_spredixcan.csv",
        spx_json=f"{EXPORT_DIR}/correlation/{STUDY}_spredixcan.json",
        spx_xlsx=f"{EXPORT_DIR}/correlation/{STUDY}_spredixcan.xlsx",
        spx_tissue_csv=f"{EXPORT_DIR}/correlation/{STUDY}_spredixcan_per_tissue.csv",
        spx_tissue_json=f"{EXPORT_DIR}/correlation/{STUDY}_spredixcan_per_tissue.json",
        spx_tissue_xlsx=f"{EXPORT_DIR}/correlation/{STUDY}_spredixcan_per_tissue.xlsx",
        nc_csv=f"{EXPORT_DIR}/correlation/{STUDY}_correlation.csv",
        nc_json=f"{EXPORT_DIR}/correlation/{STUDY}_correlation.json",
        nc_xlsx=f"{EXPORT_DIR}/correlation/{STUDY}_correlation.xlsx",
        nc_tissue_csv=f"{EXPORT_DIR}/correlation/{STUDY}_correlation_per_tissue.csv",
        nc_tissue_json=f"{EXPORT_DIR}/correlation/{STUDY}_correlation_per_tissue.json",
        nc_tissue_xlsx=f"{EXPORT_DIR}/correlation/{STUDY}_correlation_per_tissue.xlsx",
    params:
        config_path=CONFIG_PATH,
        out_dir=f"{EXPORT_DIR}/correlation",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/export_correlation.log",
    shell:
        """
        python -m repogen.reporting.cli export-correlation \
            --config {params.config_path} \
            --spx-meta {input.spx_meta} \
            --spx-tissue {input.spx_tissue} \
            --nc-summary {input.nc_summary} \
            --nc-tissue {input.nc_tissue} \
            --output-dir {params.out_dir} \
            2>&1 | tee {log}
        """


# ---------------------------------------------------------------------------
# 11.4  MR group
# ---------------------------------------------------------------------------

rule plot_mr_forest:
    input:
        parquet=f"{MR_DIR}/mr_results.parquet",
    output:
        figure=f"{PLOTS_DIR}/mr_forest.png",
        status=f"{PLOT_STATUS_DIR}/plot_mr_forest.json",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_mr_forest.log",
    shell:
        """
        python -m repogen.plotting.cli mr-forest \
            --mr-results {input.parquet} \
            --figure {output.figure} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule plot_mr_coloc:
    input:
        parquet=f"{MR_DIR}/mr_results.parquet",
    output:
        status=f"{PLOT_STATUS_DIR}/plot_mr_coloc.json",
    params:
        figure=f"{PLOTS_DIR}/coloc_posteriors.png",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_mr_coloc.log",
    shell:
        """
        python -m repogen.plotting.cli mr-coloc \
            --mr-results {input.parquet} \
            --figure {params.figure} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule plot_mr_drug_summary:
    input:
        parquet=f"{MR_DIR}/mr_drug_matches.parquet",
    output:
        status=f"{PLOT_STATUS_DIR}/plot_mr_drug_summary.json",
    params:
        figure=f"{PLOTS_DIR}/mr_drug_summary.png",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_mr_drug_summary.log",
    shell:
        """
        python -m repogen.plotting.cli mr-drug-summary \
            --mr-drug-matches {input.parquet} \
            --figure {params.figure} \
            --status {output.status} \
            2>&1 | tee {log}
        """


rule export_mr:
    input:
        results=f"{MR_DIR}/mr_results.parquet",
        drugs=f"{MR_DIR}/mr_drug_matches.parquet",
    output:
        mr_csv=f"{EXPORT_DIR}/mr/{STUDY}_mr.csv",
        mr_json=f"{EXPORT_DIR}/mr/{STUDY}_mr.json",
        mr_xlsx=f"{EXPORT_DIR}/mr/{STUDY}_mr.xlsx",
        drugs_csv=f"{EXPORT_DIR}/mr/{STUDY}_mr_drugs.csv",
        drugs_json=f"{EXPORT_DIR}/mr/{STUDY}_mr_drugs.json",
        drugs_xlsx=f"{EXPORT_DIR}/mr/{STUDY}_mr_drugs.xlsx",
    params:
        config_path=CONFIG_PATH,
        out_dir=f"{EXPORT_DIR}/mr",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/export_mr.log",
    shell:
        """
        python -m repogen.reporting.cli export-mr \
            --config {params.config_path} \
            --mr-results {input.results} \
            --mr-drugs {input.drugs} \
            --output-dir {params.out_dir} \
            2>&1 | tee {log}
        """


# ---------------------------------------------------------------------------
# 11.5  Combined / report group
# ---------------------------------------------------------------------------

rule plot_convergence:
    input:
        drug=f"{DRUG_DIR}/{STUDY}_drug_enrichment.parquet",
        nc=f"{NC_DIR}/drug_summary.parquet",
        mr=f"{MR_DIR}/mr_drug_matches.parquet",
    output:
        status=f"{PLOT_STATUS_DIR}/plot_convergence.json",
    params:
        figure=f"{PLOTS_DIR}/convergence.png",
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/plot_convergence.log",
    shell:
        """
        python -m repogen.plotting.cli convergence \
            --drug-results {input.drug} \
            --nc-summary {input.nc} \
            --mr-drug-matches {input.mr} \
            --figure {params.figure} \
            --status {output.status} \
            2>&1 | tee {log}
        """


# --- Strict full-report group ---------------------------------------------

rule combine_results_full:
    input:
        gene=f"{MAGMA_DIR}/{STUDY}_gene_results.parquet",
        pathway=f"{MAGMA_DIR}/{STUDY}_pathway_results.parquet",
        drug=f"{DRUG_DIR}/{STUDY}_drug_enrichment.parquet",
        atc=f"{ATC_DIR}/atc_enrichment_results.parquet",
        spx_meta=f"{SPX_DIR}/spredixcan_meta_analysis.parquet",
        nc_summary=f"{NC_DIR}/drug_summary.parquet",
        mr_results=f"{MR_DIR}/mr_results.parquet",
        mr_drugs=f"{MR_DIR}/mr_drug_matches.parquet",
    output:
        meta_json=f"{REPORT_FULL_DIR}/combined_metadata.json",
    params:
        config_path=CONFIG_PATH,
        results_dir=OUTPUT_ROOT,
    threads: 1
    resources:
        runtime=60,
        mem_mb=8000,
    log:
        f"{LOG_DIR}/combine_results_full.log",
    shell:
        """
        python -m repogen.reporting.cli combine \
            --config {params.config_path} \
            --results-dir {params.results_dir} \
            --mode full \
            --output {output.meta_json} \
            2>&1 | tee {log}
        """


rule export_combined_full:
    input:
        meta_json=f"{REPORT_FULL_DIR}/combined_metadata.json",
    output:
        combined_json=f"{REPORT_FULL_DIR}/{STUDY}_combined.json",
        combined_xlsx=f"{REPORT_FULL_DIR}/{STUDY}_combined.xlsx",
    params:
        config_path=CONFIG_PATH,
        results_dir=OUTPUT_ROOT,
        out_dir=REPORT_FULL_DIR,
    threads: 1
    resources:
        runtime=30,
        mem_mb=8000,
    log:
        f"{LOG_DIR}/export_combined_full.log",
    shell:
        """
        python -m repogen.reporting.cli export-combined \
            --config {params.config_path} \
            --results-dir {params.results_dir} \
            --output-dir {params.out_dir} \
            2>&1 | tee {log}
        """


rule html_report_full:
    input:
        combined_json=f"{REPORT_FULL_DIR}/{STUDY}_combined.json",
        combined_xlsx=f"{REPORT_FULL_DIR}/{STUDY}_combined.xlsx",
        status_magma_gene=f"{PLOT_STATUS_DIR}/plot_magma_gene.json",
        status_magma_pathway=f"{PLOT_STATUS_DIR}/plot_magma_pathway.json",
        status_drug=f"{PLOT_STATUS_DIR}/plot_drug_enrichment.json",
        status_tissue=f"{PLOT_STATUS_DIR}/plot_tissue_signature.json",
        status_corr=f"{PLOT_STATUS_DIR}/plot_correlation.json",
        status_mr_forest=f"{PLOT_STATUS_DIR}/plot_mr_forest.json",
        status_atc=f"{PLOT_STATUS_DIR}/plot_atc_enrichment.json",
        status_mr_coloc=f"{PLOT_STATUS_DIR}/plot_mr_coloc.json",
        status_mr_drug=f"{PLOT_STATUS_DIR}/plot_mr_drug_summary.json",
        status_convergence=f"{PLOT_STATUS_DIR}/plot_convergence.json",
    output:
        html=f"{REPORT_FULL_DIR}/report.html",
    params:
        config_path=CONFIG_PATH,
        results_dir=OUTPUT_ROOT,
        plots_dir=PLOTS_DIR,
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/html_report_full.log",
    shell:
        """
        python -m repogen.reporting.cli html-report \
            --config {params.config_path} \
            --results-dir {params.results_dir} \
            --plots-dir {params.plots_dir} \
            --output {output.html} \
            2>&1 | tee {log}
        """


# --- Partial available-report group ---------------------------------------

rule combine_results_available:
    input:
        existing_k4_available_inputs,
    output:
        meta_json=f"{REPORT_AVAILABLE_DIR}/combined_metadata.json",
    params:
        config_path=CONFIG_PATH,
        results_dir=OUTPUT_ROOT,
    threads: 1
    resources:
        runtime=60,
        mem_mb=8000,
    log:
        f"{LOG_DIR}/combine_results_available.log",
    shell:
        """
        python -m repogen.reporting.cli combine \
            --config {params.config_path} \
            --results-dir {params.results_dir} \
            --mode available \
            --output {output.meta_json} \
            2>&1 | tee {log}
        """


rule export_combined_available:
    input:
        meta_json=f"{REPORT_AVAILABLE_DIR}/combined_metadata.json",
    output:
        combined_json=f"{REPORT_AVAILABLE_DIR}/{STUDY}_combined.json",
        combined_xlsx=f"{REPORT_AVAILABLE_DIR}/{STUDY}_combined.xlsx",
    params:
        config_path=CONFIG_PATH,
        results_dir=OUTPUT_ROOT,
        out_dir=REPORT_AVAILABLE_DIR,
    threads: 1
    resources:
        runtime=30,
        mem_mb=8000,
    log:
        f"{LOG_DIR}/export_combined_available.log",
    shell:
        """
        python -m repogen.reporting.cli export-combined \
            --config {params.config_path} \
            --results-dir {params.results_dir} \
            --output-dir {params.out_dir} \
            2>&1 | tee {log}
        """


rule html_report_available:
    input:
        combined_json=f"{REPORT_AVAILABLE_DIR}/{STUDY}_combined.json",
        combined_xlsx=f"{REPORT_AVAILABLE_DIR}/{STUDY}_combined.xlsx",
    output:
        html=f"{REPORT_AVAILABLE_DIR}/report.html",
    params:
        config_path=CONFIG_PATH,
        results_dir=OUTPUT_ROOT,
        plots_dir=PLOTS_DIR,
    threads: 1
    resources:
        runtime=30,
        mem_mb=4000,
    log:
        f"{LOG_DIR}/html_report_available.log",
    shell:
        """
        python -m repogen.reporting.cli html-report \
            --config {params.config_path} \
            --results-dir {params.results_dir} \
            --plots-dir {params.plots_dir} \
            --output {output.html} \
            2>&1 | tee {log}
        """
