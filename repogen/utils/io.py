"""File I/O utilities for RepoGen.

Provides helpers for reading GWAS data, detecting delimiters, and
managing file/directory paths.  All path handling uses ``pathlib.Path``.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional

import pandas as pd


def detect_delimiter(filepath: Path, n_lines: int = 5) -> str:
    """Sniff the column delimiter from the first *n_lines* of a file.

    Checks for tab, comma, and whitespace (in that priority order).

    Args:
        filepath: Path to a delimited text file (may be gzipped).
        n_lines: Number of header lines to inspect.

    Returns:
        The detected delimiter string (a literal character or the
        regex ``r'\\s+'`` for whitespace-separated files).

    Raises:
        ValueError: If the file is empty or the delimiter cannot be
            determined.
    """
    filepath = Path(filepath)
    opener = gzip.open if filepath.suffix in (".gz", ".bgz") else open

    lines: list[str] = []
    with opener(filepath, "rt") as fh:
        for _ in range(n_lines):
            line = fh.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))

    if not lines:
        raise ValueError(f"File is empty or unreadable: {filepath}")

    header = lines[0]
    if "\t" in header:
        return "\t"
    if "," in header:
        return ","
    return r"\s+"


def read_gwas(
    filepath: Path,
    delimiter: Optional[str] = None,
    usecols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read GWAS summary statistics into a DataFrame.

    Handles plain text and ``.gz`` compressed files.  If *delimiter* is
    ``None`` it is auto-detected via :func:`detect_delimiter`.

    Args:
        filepath: Path to the GWAS summary statistics file.
        delimiter: Explicit delimiter character.  Auto-detected if
            ``None``.
        usecols: Optional subset of columns to read.

    Returns:
        A ``pandas.DataFrame`` with the GWAS data.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        RuntimeError: If the file cannot be parsed.
    """
    filepath = check_file_exists(filepath, label="GWAS summary statistics")

    if delimiter is None:
        delimiter = detect_delimiter(filepath)

    is_regex_sep = delimiter is not None and len(delimiter) > 1
    try:
        return pd.read_csv(
            filepath,
            sep=delimiter,
            usecols=usecols,
            comment="#",
            engine="python" if is_regex_sep else "c",
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Failed to read GWAS file {filepath}: {exc}"
        ) from exc


def check_file_exists(filepath: Path, label: str = "") -> Path:
    """Assert that *filepath* exists and return it as a resolved ``Path``.

    Args:
        filepath: Path to verify.
        label: Human-readable description used in the error message.

    Returns:
        The resolved ``Path`` object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        tag = f" ({label})" if label else ""
        raise FileNotFoundError(f"Required file not found{tag}: {filepath}")
    return filepath


def ensure_directory(dirpath: Path) -> Path:
    """Create *dirpath* (and parents) if it does not exist.

    Args:
        dirpath: Directory path to ensure.

    Returns:
        The resolved directory ``Path``.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath
