"""Capture the versions of external tools used to produce a result.

MAGMA and PLINK are third-party binaries whose output is build-dependent:
two compilations of the same MAGMA release - the dynamic ``v1.10 (linux)``
and static ``v1.10 (linux/s)`` builds - agree on which genes are significant
but differ in the last digits of the gene Z-scores, because floating-point
arithmetic differs between builds. Recording which binary ran turns "why do
these two tables differ?" from a forensic exercise into a lookup.

Nothing here fails a run: an unreadable or missing binary yields a
descriptive string, because provenance is metadata, not a precondition for
analysis.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

__all__ = [
    "magma_version_string",
    "tool_version",
    "magma_provenance",
    "plink_provenance",
]

#: Seconds to wait for a ``--version`` call before giving up.
_VERSION_TIMEOUT = 15


def _version_lines(binary: str | Path) -> list[str]:
    """Run ``<binary> --version`` and return its output lines.

    MAGMA prints its banner to stderr, PLINK to stdout, so both are read.
    Returns an empty list when the binary cannot be executed.
    """
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True, text=True, timeout=_VERSION_TIMEOUT, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Could not read version of %s: %r", binary, exc)
        return []

    combined = (result.stderr or "").strip() or (result.stdout or "").strip()
    return [ln.strip() for ln in combined.splitlines() if ln.strip()]


def magma_version_string(magma_binary: str | Path) -> str:
    """Return MAGMA's version banner, or ``"unknown"``.

    The full banner is kept rather than just the number: the dynamic and
    static builds of v1.10 are distinguished only by the ``(linux)`` versus
    ``(linux/s)`` suffix, and that distinction is the whole point of
    recording it.
    """
    lines = _version_lines(magma_binary)
    for line in lines:
        if "version" in line.lower():
            return line
    return lines[0] if lines else "unknown"


def tool_version(binary: str | Path | None) -> str:
    """Return the first line of ``<binary> --version``, or a status string."""
    if binary is None:
        return "not configured"
    lines = _version_lines(binary)
    return lines[0] if lines else "unknown"


def _provenance(binary: str | Path | None, label: str) -> dict[str, str]:
    if binary is None:
        return {
            f"{label}_binary": "not configured",
            f"{label}_version": "not configured",
        }
    return {f"{label}_binary": str(binary), f"{label}_version": tool_version(binary)}


def magma_provenance(binary: str | Path | None) -> dict[str, str]:
    """Return ``{magma_binary, magma_version}`` for a metadata sidecar."""
    if binary is None:
        return {"magma_binary": "not configured", "magma_version": "not configured"}
    return {"magma_binary": str(binary), "magma_version": magma_version_string(binary)}


def plink_provenance(binary: str | Path | None) -> dict[str, str]:
    """Return ``{plink_binary, plink_version}`` for a metadata sidecar."""
    return _provenance(binary, "plink")
