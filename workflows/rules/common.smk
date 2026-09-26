"""Shared constants and helpers for the RepoGen Snakemake workflow.

Resource budgets
~~~~~~~~~~~~~~~~
Each rule declares its own ``threads``, ``runtime`` and ``mem_mb`` so the
budgets travel with the pipeline rather than living in a site profile. The
values are drawn from full PGC3 schizophrenia runs (7.66M variants; 17,189
genes, 18,501 in the latest Branch C run) on KCL CREATE:

    rule                       observed        budget
    prepare_gwas                2m29           30m
    prepare_gwas_mr             1m07           30m
    prepare_gwas_spredixcan     2m10           30m
    spredixcan                  3m43, 2.4 GB   60m, 8 GB
    extract_drug_signatures     9m14           120m
    magma_gene                  18-21m         90m
    magma_pathway               <1m            60m
    drug_enrichment             3m12           60m
    atc_enrichment              <1m            60m
    negative_correlation        19m56, 5.2 GB  60m, 16 GB
    load_drug_targets           37m            120m
    mendelian_randomisation     1h18, 22 GB    6h, 32 GB

Runtimes carry roughly 3-6x headroom; a timeout is recoverable because
Snakemake resubmits once. Memory is deliberately less aggressive, since an
out-of-memory kill loses the whole job, and it was only reduced where a peak
was actually measured. Over-requesting is not free either: a large request
queues longer on a busy scheduler, which is the usual reason a run appears
stuck in PENDING.

Scale these for a larger GWAS or a wider tissue set. Only
``mendelian_randomisation`` is genuinely long; everything else is minutes.
"""

# Os, Path and WorkflowError are used by the other rule files too: Snakemake
# evaluates all included files in one namespace, so these imports serve the
# whole workflow.
import os
from pathlib import Path

from snakemake.exceptions import WorkflowError


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

if "study" not in config or "name" not in config.get("study", {}):
    raise WorkflowError("config must contain study.name")
if "gwas_input" not in config.get("study", {}):
    raise WorkflowError("config must contain study.gwas_input")

STUDY = config["study"]["name"]
OUTPUT_ROOT = config.get("output_dir", "results")
STUDY_DIR = f"{OUTPUT_ROOT}/{STUDY}"

# Root of the downloaded reference data.  Defaults to "resources" beside the
# working directory; override via `resource_dir` in the config when the data
# lives on shared storage (e.g. cluster scratch).  Every hardcoded data path
# below is expressed relative to this so no site needs to edit workflow code.
RESOURCE_DIR = str(config.get("resource_dir", "resources")).rstrip("/\\")

DRUGS_RES_DIR = f"{RESOURCE_DIR}/drugs"
PATHWAY_RES_DIR = f"{RESOURCE_DIR}/pathways"
SIGNATURE_RES_DIR = f"{RESOURCE_DIR}/drug_signatures"
REFERENCE_RES_DIR = f"{RESOURCE_DIR}/reference"
PERTURBATION_RES_DIR = f"{RESOURCE_DIR}/drug_perturbations"

PREP_DIR = f"{STUDY_DIR}/prepare_data"
GENE_ANNOT_DIR = f"{STUDY_DIR}/gene_annotation"
MAGMA_DIR = f"{STUDY_DIR}/magma"
DRUG_DIR = f"{STUDY_DIR}/drug_enrichment"
ATC_DIR = f"{STUDY_DIR}/atc_enrichment"
SPX_DIR = f"{STUDY_DIR}/spredixcan"
NC_DIR = f"{STUDY_DIR}/negative_correlation"
MR_DIR = f"{STUDY_DIR}/mr"
PLOTS_DIR = f"{STUDY_DIR}/plots"
PLOT_STATUS_DIR = f"{STUDY_DIR}/plots/_status"
EXPORT_DIR = f"{STUDY_DIR}/exports"
REPORT_FULL_DIR = f"{STUDY_DIR}/report/full"
REPORT_AVAILABLE_DIR = f"{STUDY_DIR}/report/available"
LOG_DIR = f"{STUDY_DIR}/logs"
BENCH_DIR = f"{STUDY_DIR}/benchmarks"


# ---------------------------------------------------------------------------
# CONFIG_PATH helper
# ---------------------------------------------------------------------------

if workflow.configfiles:
    CONFIG_PATH = workflow.configfiles[0]
else:
    CONFIG_PATH = "configs/config.yaml"

# Typed pipeline config (used by Branch A plot rules for PlotStyleConfig access)
from repogen.config.schema import PipelineConfig as _PipelineConfig

PIPELINE_CONFIG = _PipelineConfig(**config)
PLOT_STYLE = PIPELINE_CONFIG.output.plot_style


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# Plot status markers and stale-figure cleanup live in
# repogen.plotting.status, which the plotting CLI calls directly. Keeping one
# implementation in the package means the workflow and the CLI cannot drift.


def opt_flag(flag, value):
    """Return a CLI flag string if *value* is truthy, else empty string."""
    if value is None or value == "" or value == []:
        return ""
    return f"{flag} {value}"


# --- shared neural-cell-line resolver ------------------------
# the Snakemake path reads raw YAML and
# bypasses ``DrugSignaturesConfig`` Pydantic validation, so the workflow
# must use the same resolver as the Pydantic validator to honour
# ``include_neural_tumor_cell_lines`` when ``neural_cell_lines`` is unset.
from repogen.data.drug_signatures import resolve_neural_cell_lines_from_yaml as _resolve_neural_cell_lines


def neural_cell_lines_flag_for(ds_cfg):
    """Return ``--neural-cell-lines <csv>`` iff a non-uniform mode is
    requested; otherwise empty string.  Delegates to
    :func:`repogen.data.drug_signatures.resolve_neural_cell_lines_from_yaml`.
    """
    if (ds_cfg or {}).get("cell_line_weighting", "uniform") == "uniform":
        return ""
    tokens = _resolve_neural_cell_lines(ds_cfg)
    return f"--neural-cell-lines {','.join(tokens)}" if tokens else ""


# ---------------------------------------------------------------------------
# Availability helpers - canonical branch result files
# ---------------------------------------------------------------------------

AVAILABLE_RESULT_FILES = [
    f"{MAGMA_DIR}/{STUDY}_gene_results.parquet",
    f"{MAGMA_DIR}/{STUDY}_pathway_results.parquet",
    f"{DRUG_DIR}/{STUDY}_drug_enrichment.parquet",
    f"{ATC_DIR}/atc_enrichment_results.parquet",
    f"{SPX_DIR}/spredixcan_meta_analysis.parquet",
    f"{NC_DIR}/drug_summary.parquet",
    f"{MR_DIR}/mr_results.parquet",
    f"{MR_DIR}/mr_drug_matches.parquet",
]


def existing_available_inputs(wildcards):
    """Return the canonical branch result files that currently exist on disk.

    The order matches ``AVAILABLE_RESULT_FILES`` (deterministic).
    Accepts *wildcards* (unused) to satisfy the Snakemake input-function contract.
    """
    return [f for f in AVAILABLE_RESULT_FILES if os.path.isfile(f)]
