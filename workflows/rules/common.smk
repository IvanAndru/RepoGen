"""Shared constants and helpers for the RepoGen Snakemake workflow."""

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


def r4_neural_cell_lines_flag(ds_cfg):
    """Return ``--neural-cell-lines <csv>`` iff a non-uniform mode is
    requested; otherwise empty string.  Delegates to
    :func:`repogen.data.drug_signatures.resolve_neural_cell_lines_from_yaml`.
    """
    if (ds_cfg or {}).get("cell_line_weighting", "uniform") == "uniform":
        return ""
    tokens = _resolve_neural_cell_lines(ds_cfg)
    return f"--neural-cell-lines {','.join(tokens)}" if tokens else ""


# ---------------------------------------------------------------------------
# K4 availability helpers - canonical branch result files
# ---------------------------------------------------------------------------

K4_AVAILABLE_RESULT_FILES = [
    f"{MAGMA_DIR}/{STUDY}_gene_results.parquet",
    f"{MAGMA_DIR}/{STUDY}_pathway_results.parquet",
    f"{DRUG_DIR}/{STUDY}_drug_enrichment.parquet",
    f"{ATC_DIR}/atc_enrichment_results.parquet",
    f"{SPX_DIR}/spredixcan_meta_analysis.parquet",
    f"{NC_DIR}/drug_summary.parquet",
    f"{MR_DIR}/mr_results.parquet",
    f"{MR_DIR}/mr_drug_matches.parquet",
]


def existing_k4_available_inputs(wildcards):
    """Return canonical K4 result files that currently exist on disk.

    The order matches ``K4_AVAILABLE_RESULT_FILES`` (deterministic).
    Accepts *wildcards* (unused) to satisfy the Snakemake input-function contract.
    """
    return [f for f in K4_AVAILABLE_RESULT_FILES if os.path.isfile(f)]
