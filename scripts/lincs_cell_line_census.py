"""Enumerate the cell-line tokens present in a LINCS L1000 GCTX corpus.

Reads only ``0/META/COL/id`` from the GCTX (no expression matrix loaded),
parses column IDs via ``_parse_gctx_col_id``, and reports:
  1. Top-30 observed cell-line tokens by profile count.
  2. All tokens matching the neural-relevance regex (exploratory
     candidate discovery, not a validation gate).
  3. Presence check for the shipped defaults
     (``NEURAL_PRIMARY_CELL_LINES`` and ``NEURAL_TUMOR_CELL_LINES``
     imported from :mod:`repogen.data.drug_signatures`).  This is the
     validation gate; any default with zero observed profiles is a
     blocker and returns exit code 2.
  4. Diagnostic-only presence check for tokens that were evaluated and
     rejected (``MNEU``, ``SHSY5Y``, both with zero observed profiles at
     the time).  These do not gate on exit code; they exist so that a
     future LINCS release adding them can be spotted immediately.

The defaults are imported rather than restated here.  An earlier version
hard-coded a candidate token list, which meant a re-run against the
current corpus could report the shipped defaults as broken when they
were not.  Importing them keeps the check honest: it fails only when the
defaults actually are inconsistent with the corpus.

Usage:
    python scripts/lincs_cell_line_census.py \
        --gctx resources/drug_signatures/level5_beta_trt_cp_n720216x12328.gctx
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import h5py

from repogen.data.drug_signatures import (
    NEURAL_PRIMARY_CELL_LINES,
    NEURAL_TUMOR_CELL_LINES,
    _parse_gctx_col_id,
)


# Production defaults are the single source of truth for the gate check.
PRODUCTION_PRIMARY = tuple(NEURAL_PRIMARY_CELL_LINES)
PRODUCTION_TUMOR = tuple(NEURAL_TUMOR_CELL_LINES)
PRODUCTION_ALL = PRODUCTION_PRIMARY + PRODUCTION_TUMOR

# Historically evaluated but excluded from the defaults. Diagnostic only:
# a re-run against a future LINCS release should spot these tokens if they
# appear and prompt a re-evaluation.
HISTORICAL_REJECTED = ("MNEU", "SHSY5Y")

NEURAL_REGEX = re.compile(
    r"NEU|NPC|ASC|GLIA|GLIO|BRAIN|CORT|OLIGO|NEUR|MNEU|SHSY|SKMEL|"
    r"NEURON|IPSC|FIBRNPC|SH-SY|LN22|LN18|LN229|U87|U251|U118|U373|"
    r"T98G|SHSY|SH_SY|SK-N|SKN|IMR32|SFXN|A172"
)


def census(gctx_path: Path) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    with h5py.File(str(gctx_path), "r") as f:
        col_ids_raw = f["0/META/COL/id"][:]
    n_scanned = 0
    for raw_id in col_ids_raw:
        n_scanned += 1
        col_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
        parsed = _parse_gctx_col_id(col_id)
        cl = parsed["cell_line"].upper().strip()
        counts[cl] += 1
    return counts, n_scanned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gctx", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    print(f"[census] Scanning {args.gctx} ...", flush=True)
    counts, n_scanned = census(args.gctx)
    print(f"[census] Scanned {n_scanned:,} GCTX columns", flush=True)
    print(f"[census] Distinct cell-line tokens: {len(counts):,}", flush=True)
    print()
    print(f"Top-{args.top_n} cell-line tokens by profile count:")
    for tok, n in counts.most_common(args.top_n):
        print(f"  {tok:<20s} {n:>8,}")
    print()

    print("All tokens matching neural-relevance regex (exploratory):")
    hits = [(tok, n) for tok, n in counts.most_common() if NEURAL_REGEX.search(tok)]
    if not hits:
        print("  (none)")
    else:
        for tok, n in hits:
            print(f"  {tok:<20s} {n:>8,}")
    print()

    print("Production default-token presence check (exact uppercase match):")
    print("  (imported from repogen.data.drug_signatures - this is the gate)")
    blockers: list[str] = []
    for tok in PRODUCTION_ALL:
        n = counts.get(tok, 0)
        marker = "OK " if n > 0 else "!! BLOCKER"
        category = "primary" if tok in PRODUCTION_PRIMARY else "tumor  "
        print(f"  {marker}  {tok:<10s} [{category}]  {n:>8,} profiles")
        if n == 0:
            blockers.append(tok)
    print()

    print("Historically rejected tokens (diagnostic only, not a gate):")
    for tok in HISTORICAL_REJECTED:
        n = counts.get(tok, 0)
        note = (
            "still absent - keep excluded"
            if n == 0
            else "PRESENT NOW - consider re-evaluating"
        )
        print(f"  --  {tok:<10s}  {n:>8,} profiles  ({note})")

    if blockers:
        print()
        print(
            f"[census] BLOCKERS: {blockers} not observed in GCTX. "
            f"Production defaults are inconsistent with the shipped "
            f"corpus. Adjust NEURAL_PRIMARY_CELL_LINES / "
            f"NEURAL_TUMOR_CELL_LINES in repogen/data/drug_signatures.py "
            f"and re-run this script.",
            file=sys.stderr,
        )
        return 2
    print()
    print("[census] All production defaults observed. Constants approved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
