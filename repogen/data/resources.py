"""Download, verify, and manage external data dependencies.

Reads ``configs/resources.yaml`` to determine which external files are
needed, downloads missing ones, and verifies checksums.  Resources
marked ``manual: true`` are not downloaded automatically - the user
is shown instructions instead.

Usage::

    repogen setup-resources --config configs/resources.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import gzip
import shutil
import stat
from pathlib import Path
from typing import Optional

import yaml
import requests
from tqdm import tqdm

from repogen.utils.io import ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

_CHUNK_SIZE = 8192
_REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


#: Branch labels accepted by ``setup_resources(branches=...)``.
#: "shared" is implicit - those resources are fetched for every selection.
VALID_BRANCHES = frozenset({"a", "b", "c"})


def _wanted_by_branches(spec: dict, branches: Optional[set[str]]) -> bool:
    """Return True when *spec* is needed by any of the requested *branches*.

    Entries tagged ``shared`` are always needed: the drug-target inputs and
    gene-ID dictionaries feed ``drug_targets.parquet``, which Branch B's
    signature extraction and Branch C's Mendelian randomisation both consume.
    Fetching only the obviously branch-specific files would leave those
    branches unable to run.

    An entry with no ``branches:`` key is treated as shared, so an
    un-annotated manifest keeps its current behaviour.
    """
    if branches is None:
        return True
    tags = spec.get("branches")
    if not tags:
        return True
    tags = {str(t).strip().lower() for t in tags}
    return "shared" in tags or bool(tags & branches)


def setup_resources(
    resources_config: Path,
    target_dir: Optional[Path] = None,
    verify_only: bool = False,
    pipeline_config: Optional[Path] = None,
    branches: Optional[set[str]] = None,
) -> dict[str, str]:
    """Download and verify all external data dependencies.

    Args:
        resources_config: Path to ``resources.yaml`` configuration.
        target_dir: Root directory for resources.  Paths in the config
            are relative to this.  If ``None``, uses the parent of
            *resources_config*.
        verify_only: If ``True``, only check existing files (no
            downloads).
        pipeline_config: Optional path to a pipeline ``config.yaml``.
            When provided, resources whose definition carries the new
            ``enabled_by:`` field are gated against the dotted flag
            path resolved against this config - only included when the
            flag resolves to a truthy value.
        branches: Restrict the fetch to what a subset of branches needs,
            e.g. ``{"a"}`` to skip the 33 GB LINCS signatures and the eQTL
            files.  Resources tagged ``shared`` are always included, since
            the drug-target inputs feed all three branches.  ``None``
            (default) processes every entry.

    Returns:
        Status dict mapping resource name to one of:

        - ``"ok"`` - file present (and checksum verified if configured).
        - ``"downloaded"`` - freshly downloaded this run.
        - ``"generated"`` - derived resource generated from its source.
        - ``"manual_required"`` - manual download needed by user.
        - ``"waiting_for_source"`` - derived resource whose source is
          not yet available (e.g. manual parent not downloaded).
        - ``"waiting_for_generation"`` - verify-only mode; source is
          ready but generation was skipped.
        - ``"blocked_by_source_error"`` - derived resource whose source
          has an error status.
        - ``"missing"`` - file not found and no URL configured.
        - ``"checksum_mismatch"`` - file present but checksum wrong.
        - ``"error"`` - download or generation failed.
        - ``"gated_skipped"`` - entry carries ``enabled_by:`` but the
          gate is closed (default-deny when *pipeline_config* is None,
          or explicit-false when the resolved flag is falsy).
    """
    resources_config = Path(resources_config)
    if not resources_config.exists():
        raise FileNotFoundError(
            f"Resources config not found: {resources_config}"
        )

    with open(resources_config) as fh:
        config = yaml.safe_load(fh)

    resource_defs = config.get("resources", {})
    if not resource_defs:
        logger.warning("No resources defined in %s", resources_config)
        return {}

    if target_dir is None:
        target_dir = resources_config.parent.parent

    # Load the pipeline config once if provided so the
    # gating loop can resolve dotted ``enabled_by:`` paths cheaply.  When
    # *pipeline_config* is None, the dict stays empty and every gated
    # entry default-denies (status "gated_skipped").
    pipeline_cfg_dict: dict = {}
    if pipeline_config is not None:
        pipeline_config = Path(pipeline_config)
        if not pipeline_config.exists():
            raise FileNotFoundError(
                f"Pipeline config not found: {pipeline_config}"
            )
        with open(pipeline_config) as fh:
            pipeline_cfg_dict = yaml.safe_load(fh) or {}

    status: dict[str, str] = {}
    manual_resources: list[dict] = []

    if branches is not None:
        branches = {b.strip().lower() for b in branches}
        unknown = branches - VALID_BRANCHES
        if unknown:
            raise ValueError(
                f"Unknown branch(es): {sorted(unknown)}. "
                f"Valid branches: {sorted(VALID_BRANCHES)}"
            )
        logger.info(
            "Restricting resource setup to branch(es): %s (plus shared)",
            ", ".join(sorted(branches)),
        )

    for name, spec in resource_defs.items():
        if not isinstance(spec, dict):
            continue

        if not _wanted_by_branches(spec, branches):
            status[name] = "skipped_other_branch"
            continue

        local_path = target_dir / spec.get("local_path", name)
        is_manual = spec.get("manual", False)
        url = spec.get("url")
        checksum = spec.get("checksum")
        description = spec.get("description", name)

        # Gating contract for entries that carry the
        # new ``enabled_by:`` field.  Default-deny semantics: any such
        # entry skips unless *pipeline_config* is provided AND the
        # gating flag resolves to a truthy value.  Strictly additive -
        # entries without ``enabled_by:`` are unaffected (H-MAGMA still
        # auto-downloads as today).
        gate_path = spec.get("enabled_by")
        if gate_path is not None:
            if pipeline_config is None:
                logger.info(
                    "[GATED] %s: skipped (gating field 'enabled_by: %s' set; "
                    "pass --pipeline-config to enable)",
                    name, gate_path,
                )
                status[name] = "gated_skipped"
                continue
            gate_value = _resolve_dotted_path(pipeline_cfg_dict, gate_path)
            if not gate_value:
                logger.info(
                    "[GATED] %s: skipped (gated by '%s'=%r in %s)",
                    name, gate_path, gate_value, pipeline_config,
                )
                status[name] = "gated_skipped"
                continue

        if is_manual:
            if local_path.exists():
                logger.info("[OK] %s (manual, present): %s", name, local_path)
                status[name] = "ok"
            else:
                manual_resources.append({
                    "name": name,
                    "description": description,
                    "local_path": str(local_path),
                })
                status[name] = "manual_required"
            continue

        if local_path.exists():
            if checksum:
                if verify_checksum(local_path, checksum):
                    logger.info("[OK] %s: checksum verified", name)
                    status[name] = "ok"
                else:
                    logger.warning(
                        "[MISMATCH] %s: checksum mismatch at %s",
                        name, local_path,
                    )
                    status[name] = "checksum_mismatch"
            else:
                logger.info("[OK] %s: present (no checksum)", name)
                status[name] = "ok"
            continue

        if spec.get("derived_from"):
            status[name] = "derived_pending"
            continue

        if verify_only:
            logger.warning("[MISSING] %s: %s", name, local_path)
            status[name] = "missing"
            continue

        if not url:
            logger.warning(
                "[MISSING] %s: no URL provided and file not found at %s",
                name, local_path,
            )
            status[name] = "missing"
            continue

        postprocess = spec.get("postprocess")
        archive_member = spec.get("archive_member")
        try:
            if postprocess:
                raw_path = local_path.parent / f"{local_path.stem}_raw.txt"
                download_resource(
                    url=url,
                    local_path=raw_path,
                    description=description,
                    checksum=checksum,
                )
                _run_postprocess(postprocess, raw_path, local_path)
            else:
                download_resource(
                    url=url,
                    local_path=local_path,
                    description=description,
                    checksum=checksum,
                    archive_member=archive_member,
                )
            status[name] = "downloaded"
        except (requests.exceptions.RequestException, OSError, ValueError) as exc:
            logger.error("[ERROR] %s: download failed - %s", name, exc)
            status[name] = "error"

    if manual_resources:
        logger.info("")
        logger.info("=" * 60)
        logger.info("MANUAL DOWNLOADS REQUIRED")
        logger.info("=" * 60)
        for item in manual_resources:
            logger.info("")
            logger.info("  Resource: %s", item["name"])
            logger.info("  Description: %s", item["description"])
            logger.info("  Place at: %s", item["local_path"])
        logger.info("")
        logger.info("=" * 60)

    _generate_derived_resources(resource_defs, target_dir, status, verify_only)

    for name, st in list(status.items()):
        if st == "derived_pending":
            logger.error(
                "[ERROR] %s: derived resource was not processed "
                "(no registered generator)",
                name,
            )
            status[name] = "error"

    _SOURCE_READY = {"ok", "downloaded", "generated"}
    _ISSUES = {"missing", "checksum_mismatch", "error", "blocked_by_source_error"}
    _WAITING = {"waiting_for_source", "waiting_for_generation"}

    n_ok = sum(1 for v in status.values() if v in _SOURCE_READY)
    n_total = len(status)
    n_manual = sum(1 for v in status.values() if v == "manual_required")
    n_waiting = sum(1 for v in status.values() if v in _WAITING)
    n_issues = sum(1 for v in status.values() if v in _ISSUES)
    # gated_skipped is an *expected* outcome for entries
    # whose gating flag is closed; report it separately and never count it
    # as an issue.
    n_gated = sum(1 for v in status.values() if v == "gated_skipped")

    parts = [f"{n_ok} / {n_total} OK"]
    if n_manual:
        parts.append(f"{n_manual} manual")
    if n_waiting:
        parts.append(f"{n_waiting} waiting")
    if n_gated:
        parts.append(f"{n_gated} gated")
    if n_issues:
        parts.append(f"{n_issues} issues")
    logger.info("Resource check: %s", ", ".join(parts))

    return status


def _resolve_dotted_path(d: dict, dotted: str):
    """Resolve a dotted key path against a nested dict.

    Returns ``None`` if any key along the path is missing rather than
    raising, so absent keys behave the same as explicit ``False``
    values in the gating contract.

    Examples:
        >>> _resolve_dotted_path({"a": {"b": True}}, "a.b")
        True
        >>> _resolve_dotted_path({"a": {}}, "a.b") is None
        True
        >>> _resolve_dotted_path({}, "a.b.c") is None
        True
    """
    cur = d
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------------
# Post-download processing
# ---------------------------------------------------------------------------


_POSTPROCESSORS: dict[str, callable] = {}


def _run_postprocess(name: str, raw_path: Path, final_path: Path) -> None:
    """Dispatch to a named post-processing function."""
    func = _POSTPROCESSORS.get(name)
    if func is None:
        raise ValueError(f"Unknown postprocess handler: '{name}'")
    func(raw_path, final_path)


def _postprocess_repurposing_hub(raw_path: Path, final_path: Path) -> None:
    """Convert Broad Repurposing Hub TSV to CSV.

    The source file (``repurposing_samples_*.txt``) has ``!``-prefixed
    metadata lines followed by tab-separated data.  This strips the
    metadata, parses the TSV, adds a compound-level ``pert_id`` column
    (BRD prefix from ``broad_id``), cleans malformed InChIKey values,
    and writes a standard CSV that ``drug_signatures.py`` can read.
    """
    import pandas as pd
    from io import StringIO

    with open(raw_path, encoding="utf-8") as fh:
        data_lines = [line for line in fh if not line.startswith("!")]

    df = pd.read_csv(StringIO("".join(data_lines)), sep="\t")

    # Extract compound-level BRD prefix as pert_id (e.g. BRD-K76022557
    # from BRD-K76022557-003-28-9).  Falls back to full broad_id if the
    # expected format doesn't match.
    df["pert_id"] = df["broad_id"].astype(str).str.extract(
        r"^(BRD-[A-Z]\d{8})"
    )[0]
    missing = df["pert_id"].isna()
    if missing.any():
        logger.warning(
            "Could not derive BRD prefix for %d rows; falling back to broad_id",
            missing.sum(),
        )
        df.loc[missing, "pert_id"] = df.loc[missing, "broad_id"].astype(str)

    # Nullify malformed InChIKey values (full InChI strings stored in the
    # InChIKey column) so matching falls through to PubChem CID / name.
    if "InChIKey" in df.columns:
        bad_ik = df["InChIKey"].astype(str).str.startswith("InChI=")
        if bad_ik.any():
            logger.info(
                "Cleared %d malformed InChIKey values (full InChI strings)",
                bad_ik.sum(),
            )
            df.loc[bad_ik, "InChIKey"] = None

    df.to_csv(final_path, index=False)
    raw_path.unlink(missing_ok=True)
    logger.info(
        "Post-processed repurposing hub: %d rows, %d columns -> %s",
        len(df), len(df.columns), final_path,
    )


_POSTPROCESSORS["repurposing_hub_tsv_to_csv"] = _postprocess_repurposing_hub


def _postprocess_plink_zip(raw_path: Path, final_path: Path) -> None:
    """Extract PLINK trio (.bed/.bim/.fam) from a ZIP archive.

    *final_path* should point to the ``.bed`` file; the ``.bim`` and
    ``.fam`` siblings are extracted alongside it.
    """
    import zipfile

    if not zipfile.is_zipfile(raw_path):
        raise ValueError(
            f"Expected a ZIP archive at {raw_path}, "
            f"but file does not have ZIP magic bytes"
        )

    prefix = final_path.stem
    required_suffixes = (".bed", ".bim", ".fam")
    target_dir = final_path.parent
    ensure_directory(target_dir)

    with zipfile.ZipFile(raw_path, "r") as zf:
        for suffix in required_suffixes:
            member_name = f"{prefix}{suffix}"
            candidates = [n for n in zf.namelist() if n.endswith(member_name)]
            if not candidates:
                raise FileNotFoundError(
                    f"ZIP does not contain '*{member_name}'. "
                    f"Available: {zf.namelist()}"
                )
            member = candidates[0]
            out_path = target_dir / member_name
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    for suffix in required_suffixes:
        out_path = target_dir / f"{prefix}{suffix}"
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(
                f"PLINK file missing or empty after extraction: {out_path}"
            )

    raw_path.unlink(missing_ok=True)
    sizes = {s: (target_dir / f"{prefix}{s}").stat().st_size / 1e6
             for s in required_suffixes}
    logger.info(
        "Extracted PLINK trio: %s (%.1f MB .bed, %.1f MB .bim, %.1f MB .fam)",
        prefix, sizes[".bed"], sizes[".bim"], sizes[".fam"],
    )


_POSTPROCESSORS["extract_plink_zip"] = _postprocess_plink_zip


def _postprocess_extract_dsigdb_d3(raw_path: Path, final_path: Path) -> None:
    """Extract DSigDB D3 (drug-perturbation transcriptomic signatures) from
    ``DSigDB_All.zip``.

    DSigDB packages four signature classes:
      D1 - FDA-approved drugs (CMap-derived)
      D2 - kinase inhibitors
      D3 - drug-perturbation transcriptomic signatures (the class
            appropriate for expression-based enrichment)
      D4 - computational predictions

    RepoGen only consumes D3. This handler walks the ZIP for
    any member whose stem contains "D3" and writes it as the final
    target file.  When several D3-named members exist, the largest
    (most informative) is selected.
    """
    import zipfile

    if not zipfile.is_zipfile(raw_path):
        raise ValueError(
            f"Expected a ZIP archive at {raw_path}, "
            f"but file does not have ZIP magic bytes"
        )

    target_dir = final_path.parent
    ensure_directory(target_dir)

    with zipfile.ZipFile(raw_path, "r") as zf:
        # Accept any member whose filename contains "D3" and has a text
        # extension.  Naming varies across DSigDB releases (e.g.
        # "DSigDB_D3.txt", "D3_perturbations.gmt"), so we match
        # leniently.
        members = [
            n for n in zf.namelist()
            if "D3" in Path(n).stem
            and n.lower().endswith((".txt", ".gmt", ".tsv"))
        ]
        if not members:
            raise FileNotFoundError(
                f"ZIP does not contain a D3 (drug-perturbation) file. "
                f"Available members (first 20): {zf.namelist()[:20]}"
            )

        # Largest first - prefer the most informative D3 file when the
        # archive happens to ship more than one.
        members.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        chosen = members[0]
        with zf.open(chosen) as src, open(final_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError(
            f"DSigDB D3 file missing or empty after extraction: {final_path}"
        )

    raw_path.unlink(missing_ok=True)
    logger.info(
        "Extracted DSigDB D3: %s (%.1f MB) -> %s",
        chosen, final_path.stat().st_size / 1e6, final_path,
    )


_POSTPROCESSORS["extract_dsigdb_d3"] = _postprocess_extract_dsigdb_d3


def _postprocess_extract_ncbi38_gene_loc(raw_path: Path, final_path: Path) -> None:
    """Extract ``NCBI38.gene.loc`` from the MAGMA ``NCBI38.zip`` archive.

    The MAGMA GRCh38 gene-location distribution ships as a small ZIP
    containing ``REPORT``, ``README``, and ``NCBI38.gene.loc`` (~720 KB
    six-column tab-separated file: ``gene_id  chr  start  end  strand  name``,
    keyed on Entrez gene IDs).  This handler picks the gene-loc member,
    rejects path-traversal attempts, writes it to *final_path*, and emits
    a drift-detection log line carrying the observed SHA256 against the
    expected literal - a benign upstream version bump (e.g. additional
    bytes in REPORT shifting the ZIP SHA256) does not break the pipeline;
    it only surfaces the mismatch for human review.

    Used by ``ncbi_gene_loc_grch38`` in
    ``configs/resources.yaml``.  Models its safety checks on the existing
    ``_postprocess_plink_zip`` and ``_postprocess_extract_dsigdb_d3``
    handlers.
    """
    import zipfile

    if not zipfile.is_zipfile(raw_path):
        raise ValueError(
            f"Expected a ZIP archive at {raw_path}, "
            f"but file does not have ZIP magic bytes (likely an HTML 404 page)"
        )

    target_member_suffix = "NCBI38.gene.loc"
    ensure_directory(final_path.parent)

    with zipfile.ZipFile(raw_path, "r") as zf:
        # Reject directories, absolute paths, and traversal segments;
        # accept any member whose path ends with NCBI38.gene.loc.
        candidates = [
            m for m in zf.namelist()
            if m.endswith(target_member_suffix)
            and not m.endswith("/")
            and not m.startswith(("/", "\\"))
            and ".." not in Path(m).parts
        ]
        if not candidates:
            raise FileNotFoundError(
                f"ZIP does not contain a 'NCBI38.gene.loc' member. "
                f"Available: {zf.namelist()}"
            )
        # Prefer the shortest path (top-level member) if multiple match.
        member = sorted(candidates, key=len)[0]
        with zf.open(member) as src, open(final_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError(
            f"NCBI38.gene.loc empty after extraction: {final_path}"
        )

    # Drift-detection logging (non-enforcing): records the
    # expected SHA256 of the upstream file at first verification; any
    # change here surfaces immediately in setup-resources logs.
    extracted_sha = hashlib.sha256(final_path.read_bytes()).hexdigest()
    expected_sha = (
        "251fa802c60bd42802b0ea62b62e03ffbf0494982a8f82984a0b709bed082290"
    )
    drift = "" if extracted_sha == expected_sha else " (DRIFT vs expected)"
    logger.info(
        "Extracted NCBI38.gene.loc: %d bytes, SHA256=%s; expected %s%s",
        final_path.stat().st_size,
        extracted_sha,
        expected_sha,
        drift,
    )

    raw_path.unlink(missing_ok=True)


_POSTPROCESSORS["extract_ncbi38_gene_loc"] = _postprocess_extract_ncbi38_gene_loc


def _postprocess_extract_magma_binary(raw_path: Path, final_path: Path) -> None:
    """Extract the ``magma`` executable from the official distribution ZIP.

    MAGMA cannot be installed from conda: ``magma`` on conda-forge is an
    unrelated GPU linear-algebra library, and bioconda ships no package for
    the gene-analysis tool.  Its licence additionally forbids redistribution,
    so the binary cannot be baked into a shared container image - every
    installation fetches it from the official VU/CNCR distribution instead.

    The archive contains the ``magma`` executable alongside auxiliary files.
    This handler selects the executable, rejects path-traversal members, and
    sets the owner-execute bit (ZIP archives do not reliably carry POSIX
    permissions, and the file is useless without it).
    """
    import zipfile

    if not zipfile.is_zipfile(raw_path):
        raise ValueError(
            f"Expected a ZIP archive at {raw_path}, but the file lacks ZIP "
            f"magic bytes (an HTML error page is the usual cause)"
        )

    ensure_directory(final_path.parent)

    with zipfile.ZipFile(raw_path, "r") as zf:
        candidates = [
            m for m in zf.namelist()
            if Path(m).name == "magma"
            and not m.endswith("/")
            and not m.startswith(("/", "\\"))
            and ".." not in Path(m).parts
        ]
        if not candidates:
            raise FileNotFoundError(
                f"ZIP does not contain a 'magma' executable. "
                f"Available members: {zf.namelist()}"
            )
        member = sorted(candidates, key=len)[0]
        with zf.open(member) as src, open(final_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError(f"MAGMA binary empty after extraction: {final_path}")

    final_path.chmod(final_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    logger.info(
        "Extracted MAGMA binary: %s (%.1f MB, executable)",
        final_path,
        final_path.stat().st_size / 1e6,
    )

    raw_path.unlink(missing_ok=True)


_POSTPROCESSORS["extract_magma_binary"] = _postprocess_extract_magma_binary


# ---------------------------------------------------------------------------
# Derived resource generation
# ---------------------------------------------------------------------------

_SOURCE_READY_STATUSES: set[str] = {"ok", "downloaded", "generated"}
_SOURCE_ERROR_STATUSES: set[str] = {"missing", "checksum_mismatch", "error"}

_DERIVED_GENERATORS: dict[str, callable] = {}


def _generate_derived_resources(
    resource_defs: dict,
    target_dir: Path,
    status: dict[str, str],
    verify_only: bool = False,
) -> None:
    """Process all derived resources after the main download loop.

    For each resource with ``derived_from`` and status
    ``"derived_pending"``, checks the source status and either generates
    the output or sets an appropriate waiting/blocked status.
    """
    for name, spec in resource_defs.items():
        if not isinstance(spec, dict):
            continue
        if status.get(name) != "derived_pending":
            continue

        source_name = spec["derived_from"]
        source_status = status.get(source_name)
        output_path = target_dir / spec["local_path"]

        if source_status in _SOURCE_ERROR_STATUSES:
            logger.warning(
                "[BLOCKED] %s: source '%s' has status '%s'",
                name, source_name, source_status,
            )
            status[name] = "blocked_by_source_error"
            continue

        if source_status not in _SOURCE_READY_STATUSES:
            logger.info(
                "[WAITING] %s: source '%s' not yet available (status: %s)",
                name, source_name, source_status or "unknown",
            )
            status[name] = "waiting_for_source"
            continue

        if verify_only:
            logger.info(
                "[PENDING] %s: source ready, run without --verify-only to generate",
                name,
            )
            status[name] = "waiting_for_generation"
            continue

        gen_func = _DERIVED_GENERATORS.get(name)
        if gen_func is None:
            logger.error(
                "[ERROR] %s: no registered generator function", name,
            )
            status[name] = "error"
            continue

        source_spec = resource_defs.get(source_name, {})
        source_path = target_dir / source_spec.get("local_path", "")

        try:
            gen_func(source_path, output_path)
            status[name] = "generated"
        except Exception as exc:
            logger.error("[ERROR] %s: generation failed - %s", name, exc)
            status[name] = "error"


def generate_lincs_gene_info(
    geneinfo_beta_path: Path,
    output_path: Path,
) -> Path:
    """Generate ``lincs_gene_info.tsv`` from LINCS ``geneinfo_beta.txt``.

    Reads the LINCS gene metadata file and produces a simplified TSV
    with columns: entrez_id, gene_symbol, is_landmark, is_bing.

    Supports two source-column naming contracts for gene IDs:

    * ``pr_gene_id`` / ``pr_gene_symbol`` - older LINCS metadata format.
    * ``gene_id`` / ``gene_symbol`` - CMap 2020 ``geneinfo_beta.txt``.

    And three contracts for gene-set classification (tried in order):

    1. ``pr_is_lm`` / ``pr_is_bing`` (binary ``"1"``/``"0"``).
    2. ``pr_gene_space`` (``"landmark"`` / ``"best inferred"`` /
       ``"inferred"``).
    3. ``feature_space`` - same semantics as ``pr_gene_space``, used in
       CMap 2020 ``geneinfo_beta.txt``.

    Args:
        geneinfo_beta_path: Path to ``geneinfo_beta.txt``.
        output_path: Where to write the derived TSV.

    Returns:
        Path to the generated file.
    """
    import pandas as pd

    geneinfo_beta_path = Path(geneinfo_beta_path)
    output_path = Path(output_path)

    df = pd.read_csv(geneinfo_beta_path, sep="\t")

    if "pr_gene_id" in df.columns and "pr_gene_symbol" in df.columns:
        col_id, col_sym = "pr_gene_id", "pr_gene_symbol"
    elif "gene_id" in df.columns and "gene_symbol" in df.columns:
        col_id, col_sym = "gene_id", "gene_symbol"
    else:
        raise ValueError(
            f"Expected gene ID columns ('pr_gene_id'/'pr_gene_symbol' or "
            f"'gene_id'/'gene_symbol') in {geneinfo_beta_path}, "
            f"got: {list(df.columns)}"
        )

    if "pr_is_lm" in df.columns and "pr_is_bing" in df.columns:
        is_landmark = df["pr_is_lm"].astype(str).str.strip() == "1"
        is_bing = df["pr_is_bing"].astype(str).str.strip() == "1"
    elif "pr_gene_space" in df.columns:
        space = df["pr_gene_space"].str.strip().str.lower()
        is_landmark = space == "landmark"
        is_bing = space == "best inferred"
    elif "feature_space" in df.columns:
        space = df["feature_space"].str.strip().str.lower()
        is_landmark = space == "landmark"
        is_bing = space == "best inferred"
    else:
        raise ValueError(
            f"Cannot determine gene-set classification from {geneinfo_beta_path}. "
            f"Expected 'pr_is_lm'+'pr_is_bing', 'pr_gene_space', or "
            f"'feature_space', got columns: {list(df.columns)}"
        )

    result = pd.DataFrame({
        "entrez_id": df[col_id].astype(int),
        "gene_symbol": df[col_sym].astype(str),
        "is_landmark": is_landmark,
        "is_bing": is_bing,
    })

    ensure_directory(output_path.parent)
    result.to_csv(output_path, sep="\t", index=False)
    logger.info("Generated lincs_gene_info.tsv: %d genes at %s", len(result), output_path)
    return output_path


_DERIVED_GENERATORS["lincs_gene_info"] = generate_lincs_gene_info


def generate_metabrain_normalized(
    metabrain_dir: Path,
    output_path: Path,
) -> Path:
    """Combine and normalize raw MetaBrain per-chromosome eQTL files.

    Reads ``*cortex*chr*.txt.gz`` files from *metabrain_dir* in numeric
    chromosome order, renames columns to the MR-canonical schema, derives
    the other allele from ``SNPAlleles``, and writes a single compressed
    TSV.

    Output columns (superset of MR-required):

    ======= =============== ==========================================
    Output  Raw source      Notes
    ======= =============== ==========================================
    gene    Gene            Ensembl gene ID (versioned)
    SNP     SNP             Variant identifier
    chr     SNPChr          Chromosome (int)
    pos     SNPPos          Base-pair position (int)
    a1      SNPEffectAllele Effect allele
    a2      (derived)       Other allele from SNPAlleles
    beta    MetaBeta        Meta-analysis effect size
    se      MetaSE          Standard error
    pval    MetaP           Meta-analysis p-value
    n       MetaPN          Per-variant sample size (fallback: 2970)
    gene_chr    GeneChr     Gene chromosome
    gene_pos    GenePos     Gene TSS position
    gene_symbol GeneSymbol  HGNC symbol
    eaf     SNPEffectAlleleFreq  Effect allele frequency
    gene_strand GeneStrand  Strand (+/-)
    ======= =============== ==========================================
    """
    import gzip
    import re

    import pandas as pd

    metabrain_dir = Path(metabrain_dir)
    output_path = Path(output_path)

    chr_files = sorted(
        metabrain_dir.glob("*cortex*chr*.txt.gz"),
        key=lambda p: int(re.search(r"chr(\d+)", p.name).group(1)),
    )
    if not chr_files:
        raise FileNotFoundError(
            f"No MetaBrain chromosome files (*cortex*chr*.txt.gz) "
            f"found in {metabrain_dir}"
        )

    _METABRAIN_DEFAULT_N = 2970

    rename_map = {
        "Gene": "gene",
        "SNP": "SNP",
        "SNPChr": "chr",
        "SNPPos": "pos",
        "SNPEffectAllele": "a1",
        "MetaBeta": "beta",
        "MetaSE": "se",
        "MetaP": "pval",
        "MetaPN": "n",
        "GeneChr": "gene_chr",
        "GenePos": "gene_pos",
        "GeneSymbol": "gene_symbol",
        "SNPEffectAlleleFreq": "eaf",
        "GeneStrand": "gene_strand",
    }
    out_cols = [
        "gene", "SNP", "chr", "pos", "a1", "a2",
        "beta", "se", "pval", "n",
        "gene_chr", "gene_pos", "gene_symbol", "eaf", "gene_strand",
    ]

    ensure_directory(output_path.parent)
    total_rows = 0
    header_written = False

    with gzip.open(output_path, "wt", compresslevel=6, newline="") as out_fh:
        for chr_file in chr_files:
            logger.info("Processing %s ...", chr_file.name)
            df = pd.read_csv(chr_file, sep="\t", dtype=str)

            available_renames = {k: v for k, v in rename_map.items()
                                if k in df.columns}
            df = df.rename(columns=available_renames)

            if "gene" in df.columns:
                df["gene"] = df["gene"].str.replace(
                    r"\.\d+$", "", regex=True,
                )

            if "SNPAlleles" in df.columns and "a1" in df.columns:
                parts = df["SNPAlleles"].str.split("/", n=1)
                allele_a = parts.str[0].str.strip()
                allele_b = parts.str[1].str.strip()
                a1 = df["a1"].str.strip()
                df["a2"] = allele_b.where(
                    a1.str.upper() == allele_a.str.upper(), allele_a,
                )
            else:
                df["a2"] = pd.NA

            if "n" in df.columns:
                df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(
                    _METABRAIN_DEFAULT_N
                ).astype(int)
            else:
                df["n"] = _METABRAIN_DEFAULT_N

            for col in out_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            chunk = df[out_cols]
            chunk.to_csv(
                out_fh, sep="\t", index=False,
                header=(not header_written),
                lineterminator="\n",
            )
            header_written = True
            total_rows += len(chunk)
            logger.info(
                "  %s: %d rows (running total: %d)",
                chr_file.name, len(chunk), total_rows,
            )

    size_mb = output_path.stat().st_size / 1e6
    logger.info(
        "MetaBrain normalized: %d total rows, %d chr files -> %s (%.1f MB)",
        total_rows, len(chr_files), output_path.name, size_mb,
    )
    return output_path


_DERIVED_GENERATORS["metabrain_cortex_normalized"] = generate_metabrain_normalized


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_resource(
    url: str,
    local_path: Path,
    description: str = "",
    checksum: Optional[str] = None,
    archive_member: Optional[str] = None,
) -> Path:
    """Download a file with progress bar and optional checksum verification.

    Handles ``.tar.gz`` archives by extracting the contents.  When
    *archive_member* is given, only that member is extracted directly
    to *local_path* (useful for archives with nested directories).

    Args:
        url: Download URL.
        local_path: Where to save the file.
        description: Human-readable label for the progress bar.
        checksum: Expected checksum in ``algorithm:hex`` format
            (e.g. ``sha256:abc123...``).
        archive_member: Path within a ``.tar.gz`` archive to extract
            (e.g. ``"chembl_35/chembl_35_sqlite/chembl_35.db"``).

    Returns:
        Path to the downloaded (or extracted) file.

    Raises:
        requests.exceptions.RequestException: On download failure.
        ValueError: If checksum verification fails.
    """
    local_path = Path(local_path)
    ensure_directory(local_path.parent)

    label = description or local_path.name
    logger.info("Downloading %s from %s", label, url)

    is_tar_gz = url.endswith(".tar.gz") or url.endswith(".tgz")
    download_target = (
        local_path.parent / f"{local_path.stem}_download.tar.gz"
        if is_tar_gz else local_path
    )

    resp = requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))

    with (
        open(download_target, "wb") as fh,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=label[:40],
            disable=total_size == 0,
        ) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
            fh.write(chunk)
            pbar.update(len(chunk))

    logger.info("Downloaded %s (%.1f MB)", label, download_target.stat().st_size / 1e6)

    if is_tar_gz:
        if archive_member:
            _extract_tar_member(download_target, archive_member, local_path)
        else:
            _extract_tar_gz(download_target, local_path)
        download_target.unlink(missing_ok=True)

    if checksum:
        if not verify_checksum(local_path, checksum):
            raise ValueError(
                f"Checksum verification failed for {local_path}. "
                f"Expected: {checksum}"
            )
        logger.info("Checksum verified: %s", label)

    return local_path


def _extract_tar_member(
    archive_path: Path,
    member_name: str,
    target_path: Path,
) -> None:
    """Extract a single member from a .tar.gz archive to *target_path*."""
    logger.info("Extracting member '%s' from %s", member_name, archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError:
            available = [m.name for m in tar.getmembers() if not m.isdir()]
            raise FileNotFoundError(
                f"Archive member '{member_name}' not found in "
                f"{archive_path}. Available files: {available}"
            )
        source = tar.extractfile(member)
        if source is None:
            raise ValueError(
                f"Cannot read archive member '{member_name}' "
                f"(is it a directory?)"
            )
        ensure_directory(target_path.parent)
        with open(target_path, "wb") as dest:
            shutil.copyfileobj(source, dest)
    logger.info("Extracted '%s' -> %s", member_name, target_path)


def _extract_tar_gz(archive_path: Path, target_path: Path) -> None:
    """Extract a .tar.gz archive.

    If the archive contains a single file, extract it directly to
    *target_path*.  If it contains a directory, extract to the parent
    of *target_path*.
    """
    logger.info("Extracting %s", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()

        non_dir_members = [m for m in members if not m.isdir()]
        if len(non_dir_members) == 1:
            member = non_dir_members[0]
            source = tar.extractfile(member)
            if source:
                with open(target_path, "wb") as dest:
                    shutil.copyfileobj(source, dest)
                logger.info("Extracted single file to %s", target_path)
                return

        extract_dir = target_path if target_path.suffix == "" else target_path.parent
        ensure_directory(extract_dir)

        safe_members = []
        for member in members:
            if member.name.startswith("/") or ".." in member.name:
                logger.warning("Skipping unsafe path: %s", member.name)
                continue
            safe_members.append(member)

        tar.extractall(path=extract_dir, members=safe_members)
        logger.info("Extracted %d files to %s", len(safe_members), extract_dir)


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------


def verify_checksum(filepath: Path, expected: str) -> bool:
    """Verify a file's checksum.

    Args:
        filepath: Path to the file.
        expected: Checksum in ``algorithm:hex`` format
            (e.g. ``sha256:abc123...``).  If the algorithm prefix
            is missing, SHA-256 is assumed.

    Returns:
        ``True`` if the checksum matches.
    """
    if ":" in expected:
        algorithm, expected_hex = expected.split(":", 1)
    else:
        algorithm = "sha256"
        expected_hex = expected

    if expected_hex in ("...", "abc123...", "def456...", ""):
        logger.info("Skipping placeholder checksum for %s", filepath.name)
        return True

    try:
        h = hashlib.new(algorithm)
    except ValueError:
        logger.warning("Unknown hash algorithm: %s", algorithm)
        return False

    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)

    actual_hex = h.hexdigest()
    matches = actual_hex == expected_hex
    if not matches:
        logger.warning(
            "Checksum mismatch for %s: expected %s:%s, got %s:%s",
            filepath.name, algorithm, expected_hex[:16] + "...",
            algorithm, actual_hex[:16] + "...",
        )
    return matches


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and verify external data dependencies"
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/resources.yaml"),
        help="Path to resources.yaml",
    )
    parser.add_argument(
        "--target-dir", type=Path, default=None,
        help="Root directory for resources (default: parent of config)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing files, do not download",
    )
    parser.add_argument(
        "--pipeline-config", type=Path, default=None,
        help=(
            "Optional pipeline config.yaml path used to evaluate the "
            "'enabled_by:' gates on optional resources. "
            "When omitted, every gated resource skips with "
            "status 'gated_skipped' (default-deny).  Resources without "
            "an 'enabled_by:' field are unaffected by this flag."
        ),
    )
    parser.add_argument(
        "--branch", action="append", default=None,
        help=(
            "Fetch only what these branches need: a (MAGMA/drug/ATC), "
            "b (S-PrediXcan/signature reversal), c (Mendelian randomisation). "
            "Repeatable or comma-separated.  Shared inputs - reference panel, "
            "gene-ID dictionaries, drug-target databases - are always "
            "included because all three branches consume them.  Omit to "
            "fetch everything."
        ),
    )
    args = parser.parse_args()

    selected_branches = None
    if args.branch:
        selected_branches = {
            part.strip().lower()
            for value in args.branch
            for part in value.split(",")
            if part.strip()
        }

    result_status = setup_resources(
        resources_config=args.config,
        target_dir=args.target_dir,
        verify_only=args.verify_only,
        pipeline_config=args.pipeline_config,
        branches=selected_branches,
    )

    for name, stat in result_status.items():
        logger.info("  %s: %s", name, stat)
