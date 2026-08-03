"""MAGMA gene-level association analysis.

Runs MAGMA v1.10 to aggregate SNP-level GWAS p-values into gene-level
association statistics (Z-scores and p-values). Supports proximity-based
(standard) and Hi-C-based (H-MAGMA) SNP-to-gene annotation.

Output feeds downstream modules: magma_pathway.py, drug_enrichment.py,
atc_enrichment.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from repogen.config.schema import MagmaConfig, PipelineConfig
from repogen.utils.constants import (
    HMAGMA_ANNOTATION_FILES,
    MHC_CHR,
    MHC_END,
    MHC_START,
)
from repogen.utils.io import check_file_exists, ensure_directory
from repogen.utils.logging import setup_logging
from repogen.utils.provenance import magma_provenance

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# MAGMA subprocess execution
# ---------------------------------------------------------------------------


def _run_magma_command(
    magma_binary: Path, args: list[str], description: str
) -> None:
    """Execute a MAGMA command with standardised error handling and logging.

    Args:
        magma_binary: Path to MAGMA executable.
        args: List of command-line arguments (without the binary path).
        description: Human-readable description for logging
            (e.g., "annotation step").

    Raises:
        subprocess.CalledProcessError: If MAGMA exits with non-zero code.
    """
    cmd = [str(magma_binary)] + args
    logger.info("Running MAGMA %s: %s", description, " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error(
            "MAGMA %s failed (exit code %d)", description, result.returncode
        )
        logger.error("MAGMA stderr:\n%s", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    logger.debug("MAGMA %s stdout:\n%s", description, result.stdout)


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------


def detect_magma_binary(
    config_path: Optional[Path] = None,
    resource_dir: Optional[Path] = None,
) -> Path:
    """Locate the MAGMA binary.

    Resolution order:
        1. *config_path* (if provided) - an explicit choice always wins.
        2. ``<resource_dir>/bin/magma``, installed by ``repogen setup-resources``.
        3. ``shutil.which("magma")``.
        4. Raise ``FileNotFoundError`` with install instructions.

    The resource copy is preferred over ``PATH`` deliberately. MAGMA's output
    is build-dependent - the dynamic ``v1.10 (linux)`` and static
    ``v1.10 (linux/s)`` builds agree on which genes are significant but differ
    in the last digits of the Z-scores - so results are only comparable across
    machines when everyone runs the same build. The resource copy is the one
    pinned by the manifest and fetched identically everywhere; whatever
    happens to be on ``PATH`` varies per machine. Sites that manage MAGMA
    themselves can still force their own build via ``magma.binary_path``.

    MAGMA is not available from conda: the name ``magma`` on conda-forge is an
    unrelated GPU linear-algebra library, and bioconda has no package for the
    gene-analysis tool. Its licence also forbids redistribution, so it cannot
    be bundled into a shared container image.

    Args:
        config_path: Explicit path from ``MagmaConfig.binary_path``.
        resource_dir: Root of the downloaded reference data
            (``PipelineConfig.resource_dir``).

    Returns:
        Path to the MAGMA binary.

    Raises:
        FileNotFoundError: If MAGMA is not found anywhere.
    """
    if config_path is not None:
        p = Path(config_path)
        if p.is_file():
            logger.info("Using MAGMA binary from config: %s", p)
            return p
        raise FileNotFoundError(
            f"MAGMA binary specified in config not found: {p}"
        )

    if resource_dir is not None:
        candidate = Path(resource_dir) / "bin" / "magma"
        if candidate.is_file():
            logger.info(
                "Using the MAGMA binary installed by setup-resources: %s", candidate
            )
            return candidate

    which_result = shutil.which("magma")
    if which_result is not None:
        p = Path(which_result)
        logger.info(
            "Using MAGMA from PATH: %s (no copy under the resource directory; "
            "run `repogen setup-resources` for a build pinned by the manifest)",
            p,
        )
        return p

    raise FileNotFoundError(
        "MAGMA binary not found. Fetch it with `repogen setup-resources` "
        "(downloads the official static binary to <resource_dir>/bin/magma), "
        "or set `magma.binary_path` in your config, or put `magma` on PATH."
    )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def run_magma_annotation(
    magma_binary: Path,
    snp_loc: Path,
    gene_loc: Path,
    output_prefix: Path,
    window_upstream_kb: int = 35,
    window_downstream_kb: int = 10,
) -> Path:
    """Run MAGMA's native ``--annotate`` step to create a ``.genes.annot`` file.

    MAGMA command::

        magma --annotate window={up},{down}
              --snp-loc {snp_loc}
              --gene-loc {gene_loc}
              --out {output_prefix}

    Args:
        magma_binary: Path to MAGMA executable.
        snp_loc: SNP location file.  In practice, use
            ``{reference_bfile}.bim`` - the BIM file from the same
            1000 Genomes reference panel used in the gene analysis step.
            MAGMA accepts BIM format directly.
        gene_loc: Gene location file in MAGMA format
            (GENE_ID CHR START END STRAND SYMBOL).  This is the NCBI
            gene location file from the MAGMA website
            (e.g., ``NCBI37.3.gene.loc``), NOT the output of
            ``gene_annotation.py``.
        output_prefix: Output path prefix; MAGMA appends ``.genes.annot``.
        window_upstream_kb: Upstream window in kb.
        window_downstream_kb: Downstream window in kb.

    Returns:
        Path to the generated ``.genes.annot`` file.

    Raises:
        subprocess.CalledProcessError: If MAGMA annotation fails.
        FileNotFoundError: If output ``.genes.annot`` was not created.
    """
    check_file_exists(snp_loc, label="SNP location file (BIM)")
    check_file_exists(gene_loc, label="Gene location file")

    args = [
        "--annotate",
        f"window={window_upstream_kb},{window_downstream_kb}",
        "--snp-loc",
        str(snp_loc),
        "--gene-loc",
        str(gene_loc),
        "--out",
        str(output_prefix),
    ]

    _run_magma_command(magma_binary, args, "annotation step")

    annot_file = Path(f"{output_prefix}.genes.annot")
    if not annot_file.is_file():
        raise FileNotFoundError(
            f"MAGMA annotation step did not produce expected output: {annot_file}"
        )

    logger.info("MAGMA annotation file created: %s", annot_file)
    return annot_file


def resolve_annotation_file(
    config: MagmaConfig,
    resources_dir: Path,
    reference_bfile: Optional[Path] = None,
    gene_loc: Optional[Path] = None,
    magma_binary: Optional[Path] = None,
    output_prefix: Optional[Path] = None,
) -> Path:
    """Determine which ``.genes.annot`` file to use based on annotation_mode.

    - ``"proximity"``: Run MAGMA ``--annotate`` to create a new annotation
      file.  Uses ``{reference_bfile}.bim`` as the SNP location file.
      Requires *reference_bfile*, *gene_loc*, *magma_binary*,
      *output_prefix*.
    - ``"hmagma_*"``: Look up the pre-downloaded file in
      ``resources_dir/hmagma/``.
    - ``"custom"``: Return ``config.custom_annot_file``.

    Args:
        config: MAGMA configuration.
        resources_dir: Base resources directory.
        reference_bfile: PLINK bfile prefix (needed for proximity mode).
        gene_loc: NCBI gene location file (needed for proximity mode).
        magma_binary: Path to MAGMA binary (needed for proximity mode).
        output_prefix: Output prefix for annotation (needed for
            proximity mode).

    Returns:
        Path to the ``.genes.annot`` file.

    Raises:
        FileNotFoundError: If the required annotation file does not exist.
        ValueError: If proximity mode but required args are None.
    """
    mode = config.annotation_mode

    if mode == "proximity":
        if any(v is None for v in (reference_bfile, gene_loc, magma_binary, output_prefix)):
            raise ValueError(
                "Proximity annotation requires reference_bfile, gene_loc, "
                "magma_binary, and output_prefix"
            )
        snp_loc = Path(f"{reference_bfile}.bim")
        return run_magma_annotation(
            magma_binary=magma_binary,  # type: ignore[arg-type]
            snp_loc=snp_loc,
            gene_loc=gene_loc,  # type: ignore[arg-type]
            output_prefix=output_prefix,  # type: ignore[arg-type]
            window_upstream_kb=config.window_upstream_kb,
            window_downstream_kb=config.window_downstream_kb,
        )

    if mode == "custom":
        assert config.custom_annot_file is not None, (
            "custom_annot_file should never be None here - "
            "MagmaConfig validator enforces this at construction time"
        )
        return check_file_exists(
            config.custom_annot_file,
            label="Custom MAGMA annotation file",
        )

    # H-MAGMA presets
    filename = HMAGMA_ANNOTATION_FILES.get(mode)
    if filename is None:
        raise ValueError(f"Unknown annotation mode: '{mode}'")

    annot_path = resources_dir / "hmagma" / filename
    if not annot_path.is_file():
        raise FileNotFoundError(
            f"H-MAGMA annotation file not found for mode '{mode}'. "
            f"Expected at: {annot_path}. "
            "Run `repogen setup-resources` or set `annotation_mode: proximity`"
        )

    logger.info("Using H-MAGMA annotation: %s", annot_path)
    return annot_path


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def prepare_magma_input(
    gwas_df: pd.DataFrame,
    output_dir: Path,
    study_name: str,
) -> tuple[Path, Optional[int]]:
    """Write a MAGMA-compatible p-value input file from a StandardizedGWAS DataFrame.

    MAGMA expects: ``SNP  P  N`` (whitespace-delimited, no index).

    If N is constant across all SNPs, return it separately for the
    ``N=`` command-line argument (more efficient than per-SNP N column).
    If N varies per SNP, include it in the file and return ``None``.

    Args:
        gwas_df: StandardizedGWAS DataFrame (must have SNP, P, N columns).
        output_dir: Directory to write the p-value file into.
        study_name: Study identifier for the output filename.

    Returns:
        Tuple of (path_to_pval_file, constant_n_or_none).
    """
    ensure_directory(output_dir)
    pval_path = output_dir / f"{study_name}_magma_pval.txt"

    for required in ("SNP", "P", "N"):
        if required not in gwas_df.columns:
            raise ValueError(
                f"MAGMA pval input requires column '{required}', "
                f"but it is missing. Available: {sorted(gwas_df.columns.tolist())}"
            )
    n_null = gwas_df["N"].isna().sum()
    if n_null > 0:
        examples = gwas_df.loc[gwas_df["N"].isna(), "SNP"].head(5).tolist()
        raise ValueError(
            f"MAGMA requires non-null N for all rows, but {n_null} row(s) "
            f"have missing N (e.g. {examples}). This indicates incomplete "
            f"GWAS QC - check quality_control() upstream."
        )

    n_values = gwas_df["N"].dropna().unique()
    constant_n: Optional[int] = None

    if len(n_values) == 1:
        constant_n = int(n_values[0])
        out_df = gwas_df[["SNP", "P"]].copy()
    else:
        out_df = gwas_df[["SNP", "P", "N"]].copy()

    out_df.to_csv(pval_path, sep="\t", index=False)
    logger.info(
        "MAGMA pval file written: %s (%d SNPs, N=%s)",
        pval_path,
        len(out_df),
        str(constant_n) if constant_n else "per-SNP",
    )
    return pval_path, constant_n


# ---------------------------------------------------------------------------
# Gene analysis
# ---------------------------------------------------------------------------


def run_magma_gene_analysis(
    magma_binary: Path,
    reference_bfile: Path,
    pval_file: Path,
    gene_annot_file: Path,
    output_prefix: Path,
    sample_size: Optional[int] = None,
    gene_model: str = "mean",
    memory_efficient: bool = False,
) -> tuple[Path, Path]:
    """Run MAGMA gene-level analysis.

    MAGMA command::

        magma --bfile {reference_bfile}
              --pval {pval_file} N={sample_size}
              --gene-annot {gene_annot_file}
              [--gene-model snp-wise=mean]
              [--batch-size 50]
              --out {output_prefix}

    Args:
        magma_binary: Path to MAGMA executable.
        reference_bfile: PLINK BIM/BED/FAM prefix for LD reference.
        pval_file: MAGMA-format p-value file (from :func:`prepare_magma_input`).
        gene_annot_file: Gene annotation file (``.genes.annot``).
        output_prefix: Output path prefix.
        sample_size: Constant sample size (if None, per-SNP N from file).
        gene_model: ``"mean"`` or ``"multi"``.
        memory_efficient: If True, use ``--batch-size 50``.

    Returns:
        Tuple of (path_to_genes_out, path_to_genes_raw).

    Raises:
        subprocess.CalledProcessError: If MAGMA fails.
        FileNotFoundError: If expected output files were not created.
    """
    args = [
        "--bfile",
        str(reference_bfile),
        "--pval",
        str(pval_file),
    ]
    if sample_size is not None:
        args.append(f"N={sample_size}")
    else:
        args.append("ncol=N")
    args.extend([
        "--gene-annot",
        str(gene_annot_file),
    ])

    model_flag = "snp-wise=mean" if gene_model == "mean" else "snp-wise=multi"
    args.extend(["--gene-model", model_flag])

    if memory_efficient:
        args.extend(["--batch-size", "50"])

    args.extend(["--out", str(output_prefix)])

    _run_magma_command(magma_binary, args, "gene analysis")

    genes_out = Path(f"{output_prefix}.genes.out")
    genes_raw = Path(f"{output_prefix}.genes.raw")

    if not genes_out.is_file():
        raise FileNotFoundError(
            f"MAGMA gene analysis did not produce .genes.out: {genes_out}"
        )
    if not genes_raw.is_file():
        raise FileNotFoundError(
            f"MAGMA gene analysis did not produce .genes.raw: {genes_raw}"
        )

    logger.info("MAGMA gene analysis complete: %s", genes_out)
    return genes_out, genes_raw


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_magma_results(
    genes_out_path: Path,
    gene_annotations: pd.DataFrame,
    exclude_mhc: bool = True,
    annotation_mode: str = "proximity",
) -> pd.DataFrame:
    """Parse MAGMA ``.genes.out`` file and enrich with gene annotations.

    Steps:
        1. Read ``.genes.out`` (whitespace-delimited).
        2. Map Entrez IDs to gene symbols, Ensembl IDs, biotype via
           *gene_annotations* (GeneAnnotationRecord DataFrame).
        3. Add ``in_mhc`` boolean column (chr6:25-34Mb).
        4. Compute FDR q-values (BH correction, MHC-aware).
        5. Add ``annotation_mode`` column for provenance.
        6. Sort by ``magma_p`` ascending.

    Args:
        genes_out_path: Path to MAGMA ``.genes.out`` file.
        gene_annotations: GeneAnnotationRecord DataFrame with columns
            ``gene_entrez_id``, ``gene_symbol``, ``gene_ensembl_id``,
            ``biotype``.
        exclude_mhc: If True, compute FDR on non-MHC genes only;
            MHC genes get ``fdr_q=NaN``.
        annotation_mode: Annotation mode string for provenance tracking.

    Returns:
        DataFrame with all genes (including MHC), sorted by ``magma_p``.

    Raises:
        RuntimeError: If the ``.genes.out`` file is empty or unparseable.
    """
    magma_df = pd.read_csv(
        genes_out_path,
        sep=r"\s+",
        comment="#",
        engine="python",
    )

    if magma_df.empty:
        raise RuntimeError(
            "MAGMA produced no gene results. "
            "Check GWAS input and reference panel."
        )

    magma_df = magma_df.rename(
        columns={
            "GENE": "gene_entrez_id",
            "CHR": "chr",
            "START": "start",
            "STOP": "end",
            "NSNPS": "n_snps",
            "NPARAM": "n_param",
            "N": "n_samples",
            "ZSTAT": "magma_z",
            "P": "magma_p",
        }
    )

    magma_df["gene_entrez_id"] = magma_df["gene_entrez_id"].astype(int)

    annot_cols = gene_annotations[
        ["gene_entrez_id", "gene_symbol", "gene_ensembl_id", "biotype"]
    ].copy()
    annot_cols = annot_cols.dropna(subset=["gene_entrez_id"])
    annot_cols["gene_entrez_id"] = annot_cols["gene_entrez_id"].astype(int)
    annot_cols = annot_cols.drop_duplicates(subset=["gene_entrez_id"])

    results = magma_df.merge(annot_cols, on="gene_entrez_id", how="left")

    n_mapped = results["gene_symbol"].notna().sum()
    n_unmapped = results["gene_symbol"].isna().sum()
    logger.info(
        "Gene ID mapping: %d mapped, %d unmapped (of %d total)",
        n_mapped,
        n_unmapped,
        len(results),
    )
    if n_unmapped > 0 and n_mapped == 0:
        logger.warning(
            "No genes mapped to annotations - results will have null "
            "symbol/ensembl/biotype fields"
        )

    chr_num = pd.to_numeric(results["chr"], errors="coerce")
    results["in_mhc"] = (
        (chr_num == MHC_CHR)
        & (results["start"] <= MHC_END)
        & (results["end"] >= MHC_START)
    )

    pvalues = results["magma_p"].values.copy()
    fdr_q = np.full(len(results), np.nan)

    if exclude_mhc:
        non_mhc_mask = ~results["in_mhc"].values
        if non_mhc_mask.any():
            _, qvals, _, _ = multipletests(
                pvalues[non_mhc_mask], method="fdr_bh"
            )
            fdr_q[non_mhc_mask] = qvals
    else:
        _, qvals, _, _ = multipletests(pvalues, method="fdr_bh")
        fdr_q = qvals

    results["fdr_q"] = fdr_q

    results["annotation_mode"] = annotation_mode

    output_cols = [
        "gene_entrez_id",
        "gene_symbol",
        "gene_ensembl_id",
        "chr",
        "start",
        "end",
        "biotype",
        "n_snps",
        "n_param",
        "n_samples",
        "magma_z",
        "magma_p",
        "fdr_q",
        "in_mhc",
        "annotation_mode",
    ]
    results = results[output_cols].sort_values("magma_p").reset_index(drop=True)

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_gene_analysis(
    config: PipelineConfig,
    gwas_df: pd.DataFrame,
    gene_annotations: pd.DataFrame,
    reference_bfile: Path,
    gene_loc_file: Path,
    resources_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Main entry point.  Orchestrates the full MAGMA gene analysis pipeline.

    Steps:
        1. :func:`detect_magma_binary`
        2. :func:`resolve_annotation_file` - runs MAGMA ``--annotate``
           if proximity mode
        3. :func:`prepare_magma_input` - write p-value file
        4. :func:`run_magma_gene_analysis` - run MAGMA gene analysis
        5. :func:`parse_magma_results` - parse, annotate, FDR
        6. Save results to Parquet + copy ``.genes.raw`` for downstream use
        7. Log summary statistics

    Args:
        config: Full pipeline config.  Uses ``config.magma`` for MAGMA
            settings and ``config.study.name`` for output file naming.
        gwas_df: StandardizedGWAS DataFrame.
        gene_annotations: GeneAnnotationRecord DataFrame.
        reference_bfile: PLINK BIM/BED/FAM prefix for 1000 Genomes
            reference.
        gene_loc_file: NCBI gene location file in MAGMA format
            (e.g., ``NCBI37.3.gene.loc`` from ``resources.yaml``
            ``magma_gene_loc``).  NOT the output of
            ``gene_annotation.py``.
        resources_dir: Base resources directory (for resolving H-MAGMA
            files).
        output_dir: Output directory.  Results saved under
            ``{output_dir}/{study_name}/magma/``.

    Returns:
        Gene-level results DataFrame.

    Raises:
        FileNotFoundError: If MAGMA binary, reference, or annotation
            files are missing.
        RuntimeError: If MAGMA produces no gene results.
    """
    magma_cfg = config.magma
    study_name = config.study.name

    magma_out_dir = output_dir / study_name / "magma"
    ensure_directory(magma_out_dir)
    output_prefix = magma_out_dir / study_name

    logger.info(
        "Starting MAGMA gene analysis for study '%s' "
        "(annotation_mode=%s, gene_model=%s)",
        study_name,
        magma_cfg.annotation_mode,
        magma_cfg.gene_model,
    )

    # 1. Detect binary
    magma_binary = detect_magma_binary(
        magma_cfg.binary_path, resource_dir=config.resource_dir
    )

    # 2. Resolve annotation
    gene_annot_file = resolve_annotation_file(
        config=magma_cfg,
        resources_dir=resources_dir,
        reference_bfile=reference_bfile,
        gene_loc=gene_loc_file,
        magma_binary=magma_binary,
        output_prefix=output_prefix,
    )

    # 3. Prepare input
    pval_file, constant_n = prepare_magma_input(
        gwas_df=gwas_df,
        output_dir=magma_out_dir,
        study_name=study_name,
    )

    # 4. Run gene analysis
    genes_out, genes_raw = run_magma_gene_analysis(
        magma_binary=magma_binary,
        reference_bfile=reference_bfile,
        pval_file=pval_file,
        gene_annot_file=gene_annot_file,
        output_prefix=output_prefix,
        sample_size=constant_n,
        gene_model=magma_cfg.gene_model,
        memory_efficient=magma_cfg.memory_efficient,
    )

    # 5. Parse results
    results = parse_magma_results(
        genes_out_path=genes_out,
        gene_annotations=gene_annotations,
        exclude_mhc=magma_cfg.exclude_mhc,
        annotation_mode=magma_cfg.annotation_mode,
    )

    # 6. Save outputs
    parquet_path = magma_out_dir / f"{study_name}.genes.parquet"
    results.to_parquet(parquet_path, engine="pyarrow", index=False)
    logger.info("Gene results saved: %s", parquet_path)

    raw_dest = magma_out_dir / f"{study_name}.genes.raw"
    if genes_raw != raw_dest:
        shutil.copy2(genes_raw, raw_dest)

    out_dest = magma_out_dir / f"{study_name}.genes.out"
    if genes_out != out_dest:
        shutil.copy2(genes_out, out_dest)

    annot_dest = magma_out_dir / f"{study_name}.genes.annot"
    if gene_annot_file != annot_dest:
        shutil.copy2(gene_annot_file, annot_dest)

    # 6b. Provenance sidecar.
    #
    # MAGMA's numerical output is build-dependent: two compilations of the
    # same release (the dynamic "v1.10 (linux)" and static "v1.10 (linux/s)"
    # builds) agree on which genes are significant but differ in the last
    # digits of the Z-scores. Recording which binary ran makes that
    # immediately diagnosable instead of requiring a forensic comparison.
    #
    # Written as a side product rather than a declared rule output, so adding
    # it does not invalidate existing results or force a re-run.
    metadata = {
        "study_name": study_name,
        "analysis": "magma_gene",
        **magma_provenance(magma_binary),
        "annotation_mode": magma_cfg.annotation_mode,
        "gene_model": magma_cfg.gene_model,
        "window_upstream_kb": magma_cfg.window_upstream_kb,
        "window_downstream_kb": magma_cfg.window_downstream_kb,
        "exclude_mhc_from_fdr": magma_cfg.exclude_mhc,
        "reference_bfile": str(reference_bfile),
        "gene_loc_file": str(gene_loc_file),
        "n_genes": int(len(results)),
        "n_genes_in_mhc": int(results["in_mhc"].sum()),
        "n_significant_fdr05": int((results["fdr_q"].dropna() < 0.05).sum()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = magma_out_dir / f"{study_name}_gene_results_meta.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Gene analysis metadata saved: %s", metadata_path)

    # 7. Log summary
    n_sig_fdr = (results["fdr_q"].dropna() < 0.05).sum()
    n_mhc = results["in_mhc"].sum()
    logger.info(
        "Gene analysis complete: %d genes (%d in MHC), "
        "%d significant at FDR < 0.05",
        len(results),
        n_mhc,
        n_sig_fdr,
    )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MAGMA gene-level association analysis"
    )
    parser.add_argument(
        "--gwas", type=Path, required=True,
        help="StandardizedGWAS parquet file",
    )
    parser.add_argument(
        "--gene-annotations", type=Path, required=True,
        help="GeneAnnotationRecord parquet file",
    )
    parser.add_argument(
        "--reference-bfile", type=Path, required=True,
        help="PLINK BIM/BED/FAM prefix for LD reference",
    )
    parser.add_argument(
        "--gene-loc", type=Path, required=True,
        help="NCBI gene location file in MAGMA format (e.g., NCBI37.3.gene.loc)",
    )
    parser.add_argument(
        "--resources-dir", type=Path, default=Path("resources"),
        help="Resources directory (for H-MAGMA files)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Pipeline config YAML (optional; uses defaults if not provided)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--study-name", type=str, default="study",
        help="Study name for output file naming",
    )

    args = parser.parse_args()

    if args.config:
        from repogen.config.loader import load_config

        pipeline_config = load_config(args.config)
    else:
        from repogen.config.schema import StudyConfig

        pipeline_config = PipelineConfig(
            study=StudyConfig(name=args.study_name, gwas_input=args.gwas)
        )

    gwas_data = pd.read_parquet(args.gwas)
    gene_annot_data = pd.read_parquet(args.gene_annotations)

    gene_results = run_gene_analysis(
        config=pipeline_config,
        gwas_df=gwas_data,
        gene_annotations=gene_annot_data,
        reference_bfile=args.reference_bfile,
        gene_loc_file=args.gene_loc,
        resources_dir=args.resources_dir,
        output_dir=args.output_dir,
    )

    logger.info(
        "Gene analysis complete: %d genes, %d significant at FDR < 0.05",
        len(gene_results),
        (gene_results["fdr_q"] < 0.05).sum(),
    )
