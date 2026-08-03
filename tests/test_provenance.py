"""Tests for external-tool provenance capture.

Two builds of MAGMA v1.10 (dynamic and static) produce identical conclusions
but numerically different gene Z-scores. Recording which binary ran is what
makes such a difference diagnosable, so these tests pin the behaviour that
matters: the build suffix survives, and a missing tool never fails a run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from repogen.utils.provenance import (
    magma_provenance,
    magma_version_string,
    plink_provenance,
    tool_version,
)


def _completed(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


class TestMagmaVersionString:
    def test_reads_banner_from_stderr(self) -> None:
        """MAGMA prints its banner to stderr."""
        with patch("repogen.utils.provenance.subprocess.run",
                   return_value=_completed(stderr="MAGMA version: v1.10 (linux)")):
            assert magma_version_string("magma") == "MAGMA version: v1.10 (linux)"

    def test_preserves_build_suffix(self) -> None:
        """The (linux) vs (linux/s) distinction is the entire point."""
        with patch("repogen.utils.provenance.subprocess.run",
                   return_value=_completed(stderr="MAGMA version: v1.10 (linux/s)")):
            assert magma_version_string("magma").endswith("(linux/s)")

    def test_picks_the_version_line_among_several(self) -> None:
        banner = "Welcome to MAGMA\nMAGMA version: v1.10 (linux)\nusage: ..."
        with patch("repogen.utils.provenance.subprocess.run",
                   return_value=_completed(stderr=banner)):
            assert magma_version_string("magma") == "MAGMA version: v1.10 (linux)"

    def test_missing_binary_returns_unknown(self) -> None:
        with patch("repogen.utils.provenance.subprocess.run",
                   side_effect=FileNotFoundError):
            assert magma_version_string("/nonexistent/magma") == "unknown"

    def test_timeout_returns_unknown(self) -> None:
        with patch("repogen.utils.provenance.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="magma", timeout=15)):
            assert magma_version_string("magma") == "unknown"

    def test_empty_output_returns_unknown(self) -> None:
        with patch("repogen.utils.provenance.subprocess.run", return_value=_completed()):
            assert magma_version_string("magma") == "unknown"


class TestToolVersion:
    def test_reads_stdout_when_stderr_empty(self) -> None:
        """PLINK prints to stdout."""
        with patch("repogen.utils.provenance.subprocess.run",
                   return_value=_completed(stdout="PLINK v1.90b7.7 64-bit")):
            assert tool_version("plink").startswith("PLINK v1.90")

    def test_none_binary_is_reported_not_raised(self) -> None:
        assert tool_version(None) == "not configured"


class TestProvenanceDicts:
    def test_magma_provenance_records_binary_and_version(self, tmp_path: Path) -> None:
        binary = tmp_path / "magma"
        with patch("repogen.utils.provenance.subprocess.run",
                   return_value=_completed(stderr="MAGMA version: v1.10 (linux/s)")):
            prov = magma_provenance(binary)
        assert prov["magma_binary"] == str(binary)
        assert prov["magma_version"] == "MAGMA version: v1.10 (linux/s)"

    def test_plink_provenance_records_binary_and_version(self, tmp_path: Path) -> None:
        with patch("repogen.utils.provenance.subprocess.run",
                   return_value=_completed(stdout="PLINK v1.90b7.7 64-bit")):
            prov = plink_provenance(tmp_path / "plink")
        assert set(prov) == {"plink_binary", "plink_version"}
        assert "PLINK" in prov["plink_version"]

    def test_unconfigured_tool_does_not_raise(self) -> None:
        assert magma_provenance(None)["magma_version"] == "not configured"
        assert plink_provenance(None)["plink_version"] == "not configured"

    def test_broken_binary_never_raises(self, tmp_path: Path) -> None:
        """Provenance is metadata; it must not be able to fail an analysis."""
        with patch("repogen.utils.provenance.subprocess.run", side_effect=OSError("boom")):
            prov = magma_provenance(tmp_path / "magma")
        assert prov["magma_version"] == "unknown"


class TestPathwayAliasPreserved:
    """drug_enrichment imports this private name and the tests patch it."""

    def test_alias_points_at_the_shared_implementation(self) -> None:
        from repogen.analysis.magma_pathway import _get_magma_version
        assert _get_magma_version is magma_version_string

    def test_drug_enrichment_still_imports_it(self) -> None:
        from repogen.analysis import drug_enrichment
        assert callable(drug_enrichment._get_magma_version)
