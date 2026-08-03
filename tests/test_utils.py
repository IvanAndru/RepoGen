"""Tests for repogen.utils modules (logging, io, constants)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from repogen.utils.constants import (
    ATC_DESCRIPTIONS,
    BRAIN_TISSUES,
    GWAS_COLUMN_ALIASES,
    INTERACTION_TYPE_STANDARDISATION,
    PDSP_TARGET_MAP,
)
from repogen.utils.io import (
    check_file_exists,
    detect_delimiter,
    ensure_directory,
)
from repogen.utils.logging import setup_logging


class TestSetupLogging:
    """Tests for the centralised logging setup."""

    def test_returns_logger(self) -> None:
        log = setup_logging("test_returns_logger")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_returns_logger"

    def test_duplicate_call_no_extra_handlers(self) -> None:
        name = "test_dup_handlers"
        log1 = setup_logging(name)
        n_handlers = len(log1.handlers)
        log2 = setup_logging(name)
        assert log1 is log2
        assert len(log2.handlers) == n_handlers

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging("test_bad_level", level="INVALID")

    def test_file_handler(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        log = setup_logging("test_file_handler", log_file=log_file)
        log.info("test message")
        assert log_file.exists()


class TestDetectDelimiter:
    """Tests for delimiter detection."""

    def test_tab_delimited(self, tmp_path: Path) -> None:
        f = tmp_path / "tab.txt"
        f.write_text("SNP\tCHR\tPOS\n")
        assert detect_delimiter(f) == "\t"

    def test_comma_delimited(self, tmp_path: Path) -> None:
        f = tmp_path / "csv.txt"
        f.write_text("SNP,CHR,POS\n")
        assert detect_delimiter(f) == ","

    def test_space_delimited(self, tmp_path: Path) -> None:
        f = tmp_path / "space.txt"
        f.write_text("SNP CHR POS\n")
        assert detect_delimiter(f) == r"\s+"

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            detect_delimiter(f)


class TestCheckFileExists:
    """Tests for file existence checking."""

    def test_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "exists.txt"
        f.write_text("data")
        result = check_file_exists(f)
        assert result == f

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            check_file_exists(tmp_path / "missing.txt")

    def test_label_in_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="GWAS"):
            check_file_exists(tmp_path / "missing.txt", label="GWAS")


class TestEnsureDirectory:
    """Tests for directory creation."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_directory(new_dir)
        assert new_dir.exists()
        assert result == new_dir

    def test_existing_directory_ok(self, tmp_path: Path) -> None:
        result = ensure_directory(tmp_path)
        assert result == tmp_path


class TestConstants:
    """Tests for constant definitions."""

    def test_atc_has_level1_codes(self) -> None:
        for code in ("A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"):
            assert code in ATC_DESCRIPTIONS

    def test_atc_coverage_floor(self) -> None:
        assert len(ATC_DESCRIPTIONS) >= 1255

    def test_atc_level_distribution(self) -> None:
        by_len: dict[int, int] = {}
        for code in ATC_DESCRIPTIONS:
            by_len[len(code)] = by_len.get(len(code), 0) + 1
        assert by_len.get(1, 0) == 14
        assert by_len.get(3, 0) >= 93
        assert by_len.get(4, 0) >= 268
        assert by_len.get(5, 0) >= 880

    def test_atc_no_empty_descriptions(self) -> None:
        for code, desc in ATC_DESCRIPTIONS.items():
            assert isinstance(desc, str) and desc.strip(), f"Empty description for {code}"

    def test_atc_dermatological_sentinels_present(self) -> None:
        for code in ("D02", "D02A", "D03", "D04", "D08", "D11"):
            assert code in ATC_DESCRIPTIONS, f"Sentinel ATC code {code} missing"

    def test_brain_tissues_count(self) -> None:
        assert len(BRAIN_TISSUES) == 13

    def test_gwas_aliases_has_core_columns(self) -> None:
        for col in ("SNP", "CHR", "POS", "P", "BETA", "SE", "N"):
            assert col in GWAS_COLUMN_ALIASES

    def test_n_cas_not_aliased_as_n(self) -> None:
        n_aliases = [a.upper() for a in GWAS_COLUMN_ALIASES["N"]]
        assert "N_CAS" not in n_aliases
        assert "NCASE" not in n_aliases

    def test_pdsp_target_map_coverage(self) -> None:
        assert PDSP_TARGET_MAP["5-HT2A"] == "HTR2A"
        assert PDSP_TARGET_MAP["D2"] == "DRD2"
        assert PDSP_TARGET_MAP["SERT"] == "SLC6A4"
        assert len(PDSP_TARGET_MAP) >= 60

    def test_interaction_type_standardisation(self) -> None:
        assert INTERACTION_TYPE_STANDARDISATION["INHIBITOR"] == "inhibitor"
        assert INTERACTION_TYPE_STANDARDISATION["FULL AGONIST"] == "agonist"
        assert INTERACTION_TYPE_STANDARDISATION["channel blocker"] == "blocker"


class TestMHCInterval:
    """Build-aware MHC interval helper + GRCh37 alias preservation."""

    def test_grch37_interval(self) -> None:
        from repogen.utils.constants import (
            mhc_interval,
            MHC_CHR,
            MHC_START_GRCH37,
            MHC_END_GRCH37,
        )
        assert mhc_interval("GRCh37") == (MHC_CHR, MHC_START_GRCH37, MHC_END_GRCH37)
        assert mhc_interval("GRCh37") == (6, 25_000_000, 34_000_000)

    def test_grch38_interval(self) -> None:
        from repogen.utils.constants import (
            mhc_interval,
            MHC_CHR,
            MHC_START_GRCH38,
            MHC_END_GRCH38,
        )
        assert mhc_interval("GRCh38") == (MHC_CHR, MHC_START_GRCH38, MHC_END_GRCH38)
        assert mhc_interval("GRCh38") == (6, 25_726_063, 33_400_644)

    def test_invalid_build_raises(self) -> None:
        from repogen.utils.constants import mhc_interval
        with pytest.raises(ValueError, match="GRCh37"):
            mhc_interval("hg19")
        with pytest.raises(ValueError, match="GRCh37"):
            mhc_interval("")

    def test_grch37_aliases_preserved_for_branch_a(self) -> None:
        # Branch A (repogen.analysis.magma_gene) consumes MHC_START / MHC_END
        # directly as GRCh37 constants.  Changing them would silently shift
        # the MAGMA MHC exclusion window - must remain stable.
        from repogen.utils.constants import (
            MHC_START,
            MHC_END,
            MHC_START_GRCH37,
            MHC_END_GRCH37,
        )
        assert MHC_START == MHC_START_GRCH37 == 25_000_000
        assert MHC_END == MHC_END_GRCH37 == 34_000_000
