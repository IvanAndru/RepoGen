"""Shared pytest fixtures and test-data path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture()
def test_data_dir() -> Path:
    """Return the path to the tests/data directory."""
    return TEST_DATA_DIR
