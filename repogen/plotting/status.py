"""Status markers for optional figures.

Some figures cannot be drawn for every study - an ATC table may be empty, a
colocalisation column may be absent, convergence needs at least two branches.
Rather than emitting a placeholder image (which downstream consumers cannot
distinguish from a real result), each plot rule writes a small JSON marker
recording what happened:

``rendered``                  the figure was produced
``skipped_empty``             the input held no rows
``skipped_not_applicable``    the input lacked the columns the figure needs

The marker is the rule's declared output, so the workflow stays reproducible
while the figure itself remains optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repogen.utils.io import ensure_directory

__all__ = ["write_plot_status", "remove_if_exists"]

VALID_STATUSES = frozenset({"rendered", "skipped_empty", "skipped_not_applicable"})


def write_plot_status(
    path: str | Path,
    status: str,
    *,
    figure: str | Path | None = None,
    reason: str | None = None,
    n_rows: int | None = None,
    **extra: Any,
) -> None:
    """Write a JSON status marker for a plot rule.

    Args:
        path: Destination for the status JSON file.
        status: One of ``rendered``, ``skipped_empty``,
            ``skipped_not_applicable``.
        figure: Path to the rendered figure, when one was produced.
        reason: Human-readable explanation for a skip.
        n_rows: Number of input rows considered.
        **extra: Additional metadata fields (n_significant, formats, ...).

    Raises:
        ValueError: If *status* is not one of the three recognised values.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid plot status {status!r}; expected one of {sorted(VALID_STATUSES)}"
        )

    ensure_directory(Path(path).parent)
    payload: dict[str, Any] = {"status": status}
    if figure is not None:
        payload["figure"] = str(figure)
    if reason is not None:
        payload["reason"] = reason
    if n_rows is not None:
        payload["n_rows"] = int(n_rows)
    payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def remove_if_exists(path: str | Path) -> None:
    """Delete *path* if present.

    Used when a figure is skipped on a rerun: a stale image from an earlier
    run would otherwise contradict the status marker beside it.
    """
    Path(path).unlink(missing_ok=True)
