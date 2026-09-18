"""MAGMA pathway/gene-set enrichment analysis.

Runs MAGMA's competitive gene-set test and extracts driving genes
for each significant pathway.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from statsmodels.stats.multitest import multipletests

from repogen.analysis.magma_gene import detect_magma_binary
from repogen.config.schema import PipelineConfig
from repogen.utils.io import check_file_exists, ensure_directory
from repogen.utils.logging import setup_logging
from repogen.utils.provenance import magma_version_string

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Gene-set file creation
# ---------------------------------------------------------------------------


def create_geneset_file(
    gene_sets_df: pd.DataFrame,
    gene_results_df: pd.DataFrame,
    output_path: Path,
    min_set_size: int = 10,
    max_set_size: int = 500,
) -> tuple[Path, pd.DataFrame]:
    """Create a MAGMA-compatible gene-set definition file.

    MAGMA expects a whitespace-delimited file where each line is:
        SET_NAME  GENE1  GENE2  GENE3  ...

    Gene IDs must match the IDs in the MAGMA gene results file (.genes.raw).
    MAGMA gene results use Entrez IDs (from the NCBI gene location file),
    so this function maps HGNC symbols from PathwayRecord.genes to Entrez IDs
    using the gene_results_df (which has both gene_entrez_id and gene_symbol).

    Args:
        gene_sets_df: PathwayRecord DataFrame with columns: pathway_id, genes
            (list[str] of symbols).
        gene_results_df: Parsed MAGMA gene results with columns:
            gene_entrez_id, gene_symbol.
        output_path: Where to write the MAGMA set-annot file.
        min_set_size: Minimum genes (after ID mapping) to keep a set.
        max_set_size: Maximum genes to keep a set.

    Returns:
        Tuple of (path to written file, filtered gene_sets_df with added
        'entrez_ids' column and 'n_genes_mapped' column).

    Raises:
        RuntimeError: If all gene sets are empty after mapping and filtering.
    """
    symbol_to_entrez = dict(zip(
        gene_results_df["gene_symbol"].dropna(),
        gene_results_df["gene_entrez_id"].dropna().astype(str),
    ))

    mapped = gene_sets_df.copy()
    mapped["entrez_ids"] = mapped["genes"].apply(
        lambda genes: [symbol_to_entrez[g] for g in genes if g in symbol_to_entrez]
    )
    mapped["n_genes_mapped"] = mapped["entrez_ids"].apply(len)

    total_genes_in = mapped["genes"].apply(len).sum()
    total_mapped = mapped["n_genes_mapped"].sum()
    n_unmapped = total_genes_in - total_mapped
    if n_unmapped > 0:
        logger.warning(
            "Gene ID mapping: %d of %d gene-pathway memberships unmapped "
            "(genes not in MAGMA gene results)",
            n_unmapped,
            total_genes_in,
        )

    filtered = mapped[
        (mapped["n_genes_mapped"] >= min_set_size)
        & (mapped["n_genes_mapped"] <= max_set_size)
    ].copy()

    n_dropped = len(mapped) - len(filtered)
    if n_dropped > 0:
        logger.info(
            "Gene-set size filter: %d of %d sets dropped "
            "(outside %d-%d mapped genes)",
            n_dropped,
            len(mapped),
            min_set_size,
            max_set_size,
        )

    if filtered.empty:
        raise RuntimeError(
            f"All {len(mapped)} gene sets were empty or outside size range "
            f"({min_set_size}-{max_set_size}) after mapping symbols to "
            "Entrez IDs. Check that gene_results_df has gene_symbol column "
            "and that gene set symbols match."
        )

    ensure_directory(output_path.parent)
    with open(output_path, "w") as fh:
        for row in filtered.itertuples(index=False):
            line = row.pathway_id + "\t" + "\t".join(row.entrez_ids)
            fh.write(line + "\n")

    logger.info(
        "MAGMA gene-set file: %d sets written (%d-%d genes each) -> %s",
        len(filtered),
        filtered["n_genes_mapped"].min(),
        filtered["n_genes_mapped"].max(),
        output_path,
    )

    return output_path, filtered


# ---------------------------------------------------------------------------
# MAGMA subprocess execution
# ---------------------------------------------------------------------------


def _preflight_geneset_file(geneset_file: Path) -> None:
    """Validate geneset file structure before MAGMA execution.

    Checks that the file exists, is non-empty, and each line contains a
    set ID followed by at least one gene ID (tab-separated wide format).
    Logs an aggregate summary of set/gene counts.

    Raises:
        RuntimeError: If the file is missing, empty, or malformed.
    """
    if not geneset_file.is_file():
        raise RuntimeError(f"Geneset file does not exist: {geneset_file}")

    genes_per_set: list[int] = []
    all_genes: set[str] = set()
    malformed_lines: list[int] = []

    with open(geneset_file, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.rstrip("\n\r")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) < 2 or not fields[0]:
                malformed_lines.append(line_num)
                continue
            gene_ids = fields[1:]
            genes_per_set.append(len(gene_ids))
            all_genes.update(gene_ids)

    if malformed_lines:
        preview = malformed_lines[:5]
        raise RuntimeError(
            f"Geneset file has {len(malformed_lines)} malformed line(s) "
            f"(need SET_ID<TAB>GENE1[<TAB>GENE2...]): "
            f"line(s) {preview} in {geneset_file}"
        )

    if not genes_per_set:
        raise RuntimeError(
            f"Geneset file is empty or contains no valid sets: {geneset_file}"
        )

    logger.info(
        "Geneset preflight OK: %d sets, %d unique genes, "
        "genes/set min=%d median=%d max=%d",
        len(genes_per_set),
        len(all_genes),
        min(genes_per_set),
        int(statistics.median(genes_per_set)),
        max(genes_per_set),
    )


def run_magma_geneset_analysis(
    magma_binary: Path,
    gene_results_raw: Path,
    geneset_file: Path,
    output_prefix: Path,
) -> Path:
    """Execute MAGMA gene-set analysis subprocess.

    Runs: magma --gene-results <gene_results_raw>
                --set-annot <geneset_file>
                --out <output_prefix>

    The geneset file uses MAGMA's default wide format
    (SET_NAME<TAB>GENE1<TAB>GENE2...), so no ``col=`` modifier is needed.
    A preflight check validates the file before invoking MAGMA.

    Args:
        magma_binary: Path to MAGMA executable.
        gene_results_raw: Path to .genes.raw file from magma_gene.py.
        geneset_file: Path to gene-set annotation file from
            create_geneset_file().
        output_prefix: Output file prefix (MAGMA appends .gsa.out).

    Returns:
        Path to the MAGMA .gsa.out output file.

    Raises:
        RuntimeError: If geneset file is malformed or MAGMA exits non-zero.
        FileNotFoundError: If MAGMA output file is not created.
    """
    _preflight_geneset_file(geneset_file)

    cmd = [
        str(magma_binary),
        "--gene-results", str(gene_results_raw),
        "--set-annot", str(geneset_file),
        "--out", str(output_prefix),
    ]

    logger.info("Running MAGMA gene-set analysis: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error(
            "MAGMA gene-set analysis failed (exit code %d)",
            result.returncode,
        )
        diag_parts = [
            f"MAGMA gene-set analysis failed (exit code {result.returncode})."
        ]

        stderr_text = (result.stderr or "").strip()
        if stderr_text:
            logger.error("MAGMA stderr:\n%s", stderr_text)
            diag_parts.append(f"stderr: {stderr_text}")
        else:
            diag_parts.append("stderr: <empty>")

        stdout_text = (result.stdout or "").strip()
        if stdout_text:
            stdout_tail = "\n".join(stdout_text.splitlines()[-20:])
            logger.error("MAGMA stdout (last 20 lines):\n%s", stdout_tail)
            diag_parts.append(f"stdout tail: {stdout_tail}")

        magma_log = Path(f"{output_prefix}.log")
        if magma_log.is_file():
            log_text = magma_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-30:]
            log_tail_str = "\n".join(log_text)
            logger.error(
                "MAGMA log tail (%s):\n%s", magma_log, log_tail_str
            )
            diag_parts.append(f"MAGMA log tail ({magma_log}): {log_tail_str}")
        else:
            diag_parts.append(f"MAGMA log: <missing at {magma_log}>")

        raise RuntimeError("\n".join(diag_parts))

    logger.debug("MAGMA gene-set analysis stdout:\n%s", result.stdout)

    gsa_out = Path(f"{output_prefix}.gsa.out")
    if not gsa_out.is_file():
        raise FileNotFoundError(
            f"MAGMA gene-set analysis did not produce expected output: "
            f"{gsa_out}"
        )

    logger.info("MAGMA gene-set analysis complete: %s", gsa_out)
    return gsa_out


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_magma_geneset_results(
    gsa_out_path: Path,
) -> pd.DataFrame:
    """Parse MAGMA .gsa.out file into a DataFrame.

    The .gsa.out file is whitespace-delimited with columns:
        VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P  [FULL_NAME]

    MAGMA truncates long pathway IDs in the ``VARIABLE`` column (~35 chars).
    When ``FULL_NAME`` is present it contains the untruncated ID and is used
    as the canonical ``pathway_id`` for downstream joins.

    Args:
        gsa_out_path: Path to .gsa.out file.

    Returns:
        DataFrame with columns: pathway_id, n_genes_tested, beta,
        beta_std_error, p_value.

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError: If file is empty or unparseable.
    """
    check_file_exists(gsa_out_path, label="MAGMA .gsa.out file")

    df = pd.read_csv(
        gsa_out_path,
        comment="#",
        sep=r"\s+",
        engine="python",
    )

    if df.empty:
        raise RuntimeError(
            f"MAGMA .gsa.out file is empty (no data rows): {gsa_out_path}"
        )

    expected = {"VARIABLE", "TYPE", "NGENES", "BETA", "SE", "P"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(
            f"MAGMA .gsa.out missing expected columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    if "FULL_NAME" in df.columns:
        pathway_ids = df["FULL_NAME"]
        n_recovered = int((df["VARIABLE"] != df["FULL_NAME"]).sum())
        if n_recovered > 0:
            logger.info(
                "Using FULL_NAME as pathway_id (%d / %d had truncated VARIABLE)",
                n_recovered, len(df),
            )
    else:
        pathway_ids = df["VARIABLE"]

    result = pd.DataFrame({
        "pathway_id": pathway_ids,
        "n_genes_tested": df["NGENES"].astype(int),
        "beta": df["BETA"].astype(float),
        "beta_std_error": df["SE"].astype(float),
        "p_value": df["P"].astype(float),
    })

    logger.info(
        "Parsed %d gene-set results from %s",
        len(result),
        gsa_out_path,
    )

    return result


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------


def apply_fdr_correction(
    results_df: pd.DataFrame,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Apply global FDR correction across all pathway p-values.

    Uses statsmodels.stats.multitest.multipletests. This must be called
    ONCE on the full set of p-values from all collections combined --
    NOT per-collection.

    Args:
        results_df: DataFrame with 'p_value' column.
        method: statsmodels method string (default "fdr_bh").

    Returns:
        Same DataFrame with added 'fdr_q' column.
    """
    _, qvalues, _, _ = multipletests(
        results_df["p_value"].values, method=method
    )
    results_df = results_df.copy()
    results_df["fdr_q"] = qvalues
    return results_df


# ---------------------------------------------------------------------------
# Driving gene extraction
# ---------------------------------------------------------------------------


def _compute_driving_genes(
    genes_list: list[str],
    gene_lookup: dict[str, dict],
    p_threshold: float,
) -> tuple[list[dict], list[dict]]:
    """Process a single pathway's gene list. Called via DataFrame.apply().

    Args:
        genes_list: List of gene symbols in this pathway.
        gene_lookup: Dict mapping gene_symbol to
            {gene_entrez_id, z_score, p_value}.
        p_threshold: Threshold for flagging driving genes.

    Returns:
        Tuple of (driving_genes, all_gene_z_scores).
    """
    gene_stats = []
    for symbol in genes_list:
        if symbol in gene_lookup:
            stats = gene_lookup[symbol]
            gene_stats.append({
                "gene_symbol": symbol,
                "gene_entrez_id": str(stats["gene_entrez_id"]),
                "z_score": stats["z_score"],
                "p_value": stats["p_value"],
            })
    gene_stats.sort(key=lambda x: x["z_score"], reverse=True)
    driving = [g for g in gene_stats if g["p_value"] < p_threshold]
    return driving, gene_stats


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def assemble_pathway_results(
    magma_results_df: pd.DataFrame,
    gene_sets_df: pd.DataFrame,
    gene_results_df: pd.DataFrame,
    mapped_sets: pd.DataFrame,
    fdr_threshold: float = 0.05,
    driver_gene_p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Merge MAGMA gene-set results with pathway metadata and driving genes.

    Steps:
    1. Merge magma_results_df with gene_sets_df on pathway_id to get metadata.
    2. FDR correction should already be applied (fdr_q column present).
    3. Add 'significant' boolean column.
    4. For each pathway, compute driving genes and all_gene_z_scores.

    Args:
        magma_results_df: Output of parse_magma_geneset_results() +
            apply_fdr_correction(). Must have fdr_q column.
        gene_sets_df: Original PathwayRecord DataFrame.
        gene_results_df: Parsed MAGMA gene results.
        mapped_sets: Output from create_geneset_file() with Entrez mappings.
        fdr_threshold: Significance threshold for FDR.
        driver_gene_p_threshold: p-value threshold for driving genes.

    Returns:
        Final output DataFrame matching PathwayEnrichmentResult schema.
    """
    metadata_cols = gene_sets_df[
        ["pathway_id", "pathway_name", "source_db", "n_genes"]
    ].copy()
    metadata_cols = metadata_cols.rename(columns={"n_genes": "n_genes_in_set"})

    results = magma_results_df.merge(metadata_cols, on="pathway_id", how="left")

    results["significant"] = results["fdr_q"] < fdr_threshold

    gene_lookup = (
        gene_results_df[["gene_symbol", "gene_entrez_id", "magma_z", "magma_p"]]
        .dropna(subset=["gene_symbol"])
        .set_index("gene_symbol")
        .rename(columns={"magma_z": "z_score", "magma_p": "p_value"})
        .to_dict("index")
    )

    genes_by_pathway = mapped_sets.set_index("pathway_id")["genes"].to_dict()

    def _apply_driving(pathway_id: str) -> tuple[list[dict], list[dict]]:
        genes_list = genes_by_pathway.get(pathway_id, [])
        return _compute_driving_genes(
            genes_list, gene_lookup, driver_gene_p_threshold
        )

    driving_data = results["pathway_id"].apply(
        lambda pid: pd.Series(_apply_driving(pid))
    )
    results["driving_genes"] = driving_data[0]
    results["all_gene_z_scores"] = driving_data[1]
    results["n_driving_genes"] = results["driving_genes"].apply(len)

    output_cols = [
        "pathway_id",
        "pathway_name",
        "source_db",
        "n_genes_in_set",
        "n_genes_tested",
        "beta",
        "beta_std_error",
        "p_value",
        "fdr_q",
        "significant",
        "n_driving_genes",
        "driving_genes",
        "all_gene_z_scores",
    ]

    for col in output_cols:
        if col not in results.columns:
            results[col] = pd.NA

    results = results[output_cols].sort_values("p_value").reset_index(drop=True)

    n_sig = results["significant"].sum()
    logger.info(
        "Pathway results assembled: %d pathways, %d significant (FDR < %.2f)",
        len(results),
        n_sig,
        fdr_threshold,
    )

    return results


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


# One implementation of the MAGMA version probe lives in
# repogen.utils.provenance; this alias keeps the historic private name that
# drug_enrichment imports and the tests patch.
_get_magma_version = magma_version_string


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pathway_analysis(
    config: PipelineConfig,
    gene_results_raw: Path,
    gene_results_df: pd.DataFrame,
    gene_sets_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Full orchestration: create set file, run MAGMA, parse, FDR, assemble.

    This is the main entry point called by the pipeline workflow.

    Args:
        config: Full pipeline config (uses config.magma and config.pathway
            sections).
        gene_results_raw: Path to .genes.raw from magma_gene.py.
        gene_results_df: Parsed gene results DataFrame from magma_gene.py.
        gene_sets_df: PathwayRecord DataFrame from gene_sets.py.
        output_dir: Directory for output files.

    Returns:
        Final PathwayEnrichmentResult DataFrame.

    Raises:
        RuntimeError: If MAGMA fails or produces no results.
        FileNotFoundError: If gene_results_raw does not exist.
    """
    check_file_exists(gene_results_raw, label="MAGMA gene results (.genes.raw)")

    pathway_cfg = config.pathway
    study_name = config.study.name

    magma_out_dir = output_dir / study_name / "magma"
    ensure_directory(magma_out_dir)

    logger.info(
        "Starting pathway analysis for study '%s' "
        "(FDR method=%s, threshold=%.2f, driver p threshold=%.3f)",
        study_name,
        pathway_cfg.fdr_method,
        pathway_cfg.fdr_threshold,
        pathway_cfg.driver_gene_p_threshold,
    )

    # Load extra GMT files if configured
    if pathway_cfg.extra_gmt_files:
        from repogen.data.gene_sets import load_gene_sets

        extra_df = load_gene_sets(
            gmt_files=pathway_cfg.extra_gmt_files,
            min_size=pathway_cfg.min_set_size,
            max_size=pathway_cfg.max_set_size,
        )
        gene_sets_df = pd.concat(
            [gene_sets_df, extra_df], ignore_index=True
        )
        logger.info(
            "Loaded %d extra gene sets from %d GMT files",
            len(extra_df),
            len(pathway_cfg.extra_gmt_files),
        )

    # Source filtering
    if pathway_cfg.sources is not None:
        before = len(gene_sets_df)
        gene_sets_df = gene_sets_df[
            gene_sets_df["source_db"].isin(pathway_cfg.sources)
        ].copy()
        logger.info(
            "Source filter %s: %d -> %d gene sets",
            pathway_cfg.sources,
            before,
            len(gene_sets_df),
        )

    # 1. Create gene-set file
    geneset_path = magma_out_dir / f"{study_name}.geneset"
    geneset_file, mapped_sets = create_geneset_file(
        gene_sets_df=gene_sets_df,
        gene_results_df=gene_results_df,
        output_path=geneset_path,
        min_set_size=pathway_cfg.min_set_size,
        max_set_size=pathway_cfg.max_set_size,
    )

    # 2. Run MAGMA gene-set analysis
    magma_binary = detect_magma_binary(
        config.magma.binary_path, resource_dir=config.resource_dir
    )
    output_prefix = magma_out_dir / study_name

    gsa_out = run_magma_geneset_analysis(
        magma_binary=magma_binary,
        gene_results_raw=gene_results_raw,
        geneset_file=geneset_file,
        output_prefix=output_prefix,
    )

    # 3. Parse results
    magma_results = parse_magma_geneset_results(gsa_out)

    # 4. Apply FDR
    magma_results = apply_fdr_correction(
        magma_results, method=pathway_cfg.fdr_method
    )

    # 5. Assemble final results
    results = assemble_pathway_results(
        magma_results_df=magma_results,
        gene_sets_df=gene_sets_df,
        gene_results_df=gene_results_df,
        mapped_sets=mapped_sets,
        fdr_threshold=pathway_cfg.fdr_threshold,
        driver_gene_p_threshold=pathway_cfg.driver_gene_p_threshold,
    )

    # 6. Save outputs
    output_parquet = output_dir / f"{study_name}_pathway_results.parquet"
    output_json = output_dir / f"{study_name}_pathway_results_meta.json"

    results.to_parquet(output_parquet, index=False)

    metadata = {
        "study_name": study_name,
        "analysis": "magma_pathway",
        "magma_version": _get_magma_version(magma_binary),
        "test_mode": "competitive",
        "gene_results_file": str(gene_results_raw),
        "n_gene_sets_tested": len(results),
        "n_significant_fdr05": int(results["significant"].sum()),
        "fdr_method": pathway_cfg.fdr_method,
        "fdr_threshold": pathway_cfg.fdr_threshold,
        "driver_gene_p_threshold": pathway_cfg.driver_gene_p_threshold,
        "gene_set_sources": sorted(results["source_db"].dropna().unique().tolist()),
        "extra_gmt_files": [str(p) for p in pathway_cfg.extra_gmt_files],
        "annotation_mode": config.magma.annotation_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Pathway analysis complete: %d pathways tested, %d significant "
        "(FDR < %.2f). Output: %s",
        len(results),
        int(results["significant"].sum()),
        pathway_cfg.fdr_threshold,
        output_parquet,
    )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAGMA pathway analysis")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gene-results-raw", type=Path, required=True)
    parser.add_argument("--gene-results-parquet", type=Path, required=True)
    parser.add_argument("--gene-sets-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from repogen.config.loader import load_config

    cfg = load_config(args.config)
    gene_results_df = pd.read_parquet(args.gene_results_parquet)
    gene_sets_df = pd.read_parquet(args.gene_sets_parquet)

    results = run_pathway_analysis(
        config=cfg,
        gene_results_raw=args.gene_results_raw,
        gene_results_df=gene_results_df,
        gene_sets_df=gene_sets_df,
        output_dir=args.output_dir,
    )
    logger.info(
        "Pathway analysis complete. %d significant pathways (FDR < %.2f)",
        results["significant"].sum(),
        cfg.pathway.fdr_threshold,
    )
