"""Preflight checks for configured inputs, reference data and tools.

Answers the question a user has before submitting anything: *which branches
can actually run here?* Missing data is reported per branch, because a Branch A
run has no use for LINCS signatures and should not be blocked by their absence.

Used by ``repogen validate``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from repogen.config.schema import PipelineConfig

__all__ = ["ResourceCheck", "check_resources", "summarise_by_branch"]

# Branch labels used throughout the report.
SHARED = "shared"
BRANCH_A = "A (MAGMA / drug / ATC)"
BRANCH_B = "B (S-PrediXcan / signature reversal)"
BRANCH_C = "C (Mendelian randomisation)"


@dataclass(frozen=True)
class ResourceCheck:
    """One preflight check and its outcome."""

    name: str
    branch: str
    ok: bool
    location: str
    detail: str = ""


def _file_check(name: str, branch: str, path: Path | None, *, detail: str = "") -> ResourceCheck:
    if path is None:
        return ResourceCheck(name, branch, False, "(not configured)", detail)
    p = Path(path)
    return ResourceCheck(name, branch, p.is_file(), str(p), detail)


def _dir_check(name: str, branch: str, path: Path | None, *, detail: str = "") -> ResourceCheck:
    if path is None:
        return ResourceCheck(name, branch, False, "(not configured)", detail)
    p = Path(path)
    return ResourceCheck(name, branch, p.is_dir(), str(p), detail)


def _tool_check(name: str, branch: str, candidates: list[str], *,
                preferred_path: Path | None = None, detail: str = "") -> ResourceCheck:
    """Locate an executable, reporting the one the pipeline will actually use.

    *preferred_path* is checked before ``PATH`` so this mirrors
    :func:`repogen.analysis.magma_gene.detect_magma_binary`: reporting a
    different binary from the one that will run defeats the purpose of the
    check.
    """
    if preferred_path is not None and Path(preferred_path).is_file():
        return ResourceCheck(name, branch, True, str(preferred_path), detail)
    for exe in candidates:
        found = shutil.which(exe)
        if found:
            return ResourceCheck(name, branch, True, found, detail)
    return ResourceCheck(name, branch, False, "(not found on PATH)", detail)


def check_resources(config: PipelineConfig) -> list[ResourceCheck]:
    """Run every preflight check for *config* and return the results."""
    res = Path(config.resource_dir)
    ref = config.reference
    checks: list[ResourceCheck] = []

    # --- Shared: needed by any run --------------------------------------
    checks.append(_file_check("GWAS summary statistics", SHARED, config.study.gwas_input))

    bfile = Path(ref.genome_dir) / ref.bfile_prefix
    for suffix in (".bed", ".bim", ".fam"):
        checks.append(
            _file_check(
                f"Reference genotypes ({suffix})", SHARED,
                Path(str(bfile) + suffix),
                detail="1000 Genomes panel used for LD",
            )
        )

    # --- Branch A -------------------------------------------------------
    checks.append(
        _tool_check(
            "MAGMA binary", BRANCH_A, ["magma"],
            preferred_path=(
                Path(config.magma.binary_path) if config.magma.binary_path
                else res / "bin" / "magma"
            ),
            detail="install with `repogen setup-resources`",
        )
    )
    checks.append(_file_check("Gene locations (GRCh37)", BRANCH_A, ref.gene_loc_file))
    checks.append(
        _file_check("ChEMBL database", BRANCH_A, res / "drugs" / "chembl_35.db")
    )

    gmt_dir = res / "pathways"
    gmts = sorted(gmt_dir.glob("*.gmt")) if gmt_dir.is_dir() else []
    checks.append(
        ResourceCheck(
            "Pathway gene sets (GMT)", BRANCH_A, bool(gmts),
            f"{gmt_dir} ({len(gmts)} file(s))" if gmts else str(gmt_dir),
        )
    )

    # Optional drug sources are only required when the config asks for them.
    source_files = {
        "pdsp": ("PDSP Ki database", res / "drugs" / "pdsp_ki.csv"),
        "dgidb": ("DGIdb interactions", res / "drugs" / "dgidb_interactions.tsv"),
    }
    for source, (label, path) in source_files.items():
        if source in config.drug_enrichment.sources:
            checks.append(
                _file_check(label, BRANCH_A, path, detail=f"listed in drug_enrichment.sources")
            )

    # --- Branch B -------------------------------------------------------
    sig_dir = res / "drug_signatures"
    gctx = sorted(sig_dir.glob("*.gctx")) if sig_dir.is_dir() else []
    checks.append(
        ResourceCheck(
            "LINCS L1000 signatures (GCTX)", BRANCH_B, bool(gctx),
            str(gctx[0]) if gctx else str(sig_dir),
            detail="manual download (~35 GB)",
        )
    )
    checks.append(
        _file_check("LINCS compound metadata", BRANCH_B, sig_dir / "repurposing_hub.csv")
    )
    checks.append(
        _file_check("LINCS gene metadata", BRANCH_B, sig_dir / "geneinfo_beta.txt")
    )
    checks.append(
        _dir_check("PredictDB models", BRANCH_B, ref.predixcan_model_dir)
    )
    checks.append(
        _file_check(
            "Liftover chain (GRCh37 to GRCh38)", BRANCH_B, ref.liftover_chain,
            detail="required when the GWAS is not already GRCh38",
        )
    )

    # --- Branch C -------------------------------------------------------
    checks.append(
        _tool_check(
            "PLINK 1.9 binary", BRANCH_C, ["plink", "plink1.9"],
            detail="used for LD clumping",
        )
    )
    # EQTL locations come from mr.eqtl_sources rather than resource_dir, and a
    # relative path there resolves against the working directory - exactly as
    # the MR module resolves it at run time. Setting resource_dir alone does
    # not move these, so the reported location is the one that will be read.
    for source in config.mr.eqtl_sources:
        checks.append(
            _dir_check(
                f"eQTL source '{source.source}'", BRANCH_C, source.path,
                detail="path taken from mr.eqtl_sources, not resource_dir",
            )
        )

    return checks


def summarise_by_branch(checks: list[ResourceCheck]) -> dict[str, tuple[int, int]]:
    """Return ``{branch: (n_ok, n_total)}`` preserving first-seen order."""
    summary: dict[str, tuple[int, int]] = {}
    for check in checks:
        ok, total = summary.get(check.branch, (0, 0))
        summary[check.branch] = (ok + int(check.ok), total + 1)
    return summary
