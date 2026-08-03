"""Shared MHC gene-annotation utilities.

Build-aware MHC-region gene annotation, extracted from Branch B's
``spredixcan.py`` so that Branch B (S-PrediXcan) and Branch C (Mendelian
randomisation) can share the same, already-validated machinery
without cross-branch imports.

Two public helpers:

- :func:`load_mhc_gene_annotation` - resolve the set of MHC-region genes as
  build-invariant Ensembl IDs (GRCh38 primary -> GRCh37 projection fallback ->
  no-annotation), with a fail-loud path when a configured GRCh38 gene-loc is
  missing.
- :func:`add_mhc_flag` - add an ``mhc_flag`` column to any DataFrame carrying
  ``gene_ensembl_id`` by membership against a (build-consistent) annotation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from repogen.utils.constants import mhc_interval
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


def load_mhc_gene_annotation(
    reference: Any,
    gene_id_converter: Any | None,
    *,
    require_mhc_annotation: bool = False,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load a build-consistent MHC gene annotation.

    Implements the following decision tree:

    1. Configured-but-missing -> fail loud. If
       ``reference.gene_loc_file_grch38`` is set but the file does not
       exist, raise ``FileNotFoundError`` with a "run setup-resources"
       hint.  This is an operator error, not a license to silently
       degrade to the GRCh37 fallback.
    2. Primary (GRCh38, recommended): if ``reference.gene_loc_file_grch38``
       is set and the file exists, read it (Entrez-keyed), filter to the
       GRCh38 MHC interval, then convert Entrez -> Ensembl via
       *gene_id_converter*.  Strategy label: ``"grch38_gene_loc"``.
    3. Fallback (GRCh37 projection): ONLY entered when
       ``reference.gene_loc_file_grch38`` is genuinely unset (None).
       Reads ``reference.gene_loc_file`` (Entrez or Ensembl-keyed),
       filters to the GRCh37 MHC interval, converts Entrez -> Ensembl when
       needed.  The Ensembl IDs survive the GRCh37->GRCh38 coordinate
       shift unchanged because Ensembl IDs are build-invariant for
       stable genes.  Strategy label: ``"grch37_gene_loc_ensembl_projection"``.
    4. No annotation: return ``(None, metadata)``.  ``metadata`` always
       carries ``mhc_annotation_source`` / ``_build`` / ``_strategy`` /
       ``_coordinate_mode`` so the metadata.json is fully self-describing.

    The fallback path requires *gene_id_converter* to do anything useful
    (Entrez -> Ensembl); without it the MHC annotation is degraded to
    "no annotation" rather than producing a silently empty filter.

    Coordinate-mode semantics.  The synthesised annotation frame
    carries *uniform* placeholder coordinates (the GRCh38 MHC interval
    repeated on every row), not per-gene positions.  ``add_mhc_flag``
    re-applies the GRCh38 interval over these placeholders, which makes
    the flag check a pure membership test on ``gene_ensembl_id``.  This
    is intentional and documented in the metadata field
    ``mhc_annotation_coordinate_mode = "membership_interval_placeholder"``.

    Args:
        reference: ``ReferenceConfig`` instance (duck-typed for tests).
        gene_id_converter: Optional ``GeneIDConverter`` instance; required
            for Entrez -> Ensembl conversion of MHC genes.
        require_mhc_annotation: If True, raise when no MHC annotation
            can be loaded.  Callers should set this when a silent
            "no annotation" would be dangerous.

    Returns:
        Tuple ``(annotation_df, metadata)`` where *annotation_df* is a
        DataFrame with columns ``gene_ensembl_id, chr, start, end`` (one
        row per MHC gene with a resolvable Ensembl ID) or ``None`` when
        no annotation could be loaded.

    Raises:
        FileNotFoundError: If ``reference.gene_loc_file_grch38`` is
            configured but the file does not exist.
        RuntimeError: If *require_mhc_annotation* is True but no
            annotation can be loaded (and no configured-but-missing
            file path triggered ``FileNotFoundError`` first).
    """
    from repogen.data.gene_annotation import _read_gene_location_file

    metadata: dict[str, Any] = {
        "mhc_annotation_source": None,
        "mhc_annotation_build": None,
        "mhc_annotation_strategy": "no_annotation",
        "mhc_annotation_coordinate_mode": "membership_interval_placeholder",
        "mhc_source_interval": None,
        "mhc_flag_interval": list(mhc_interval("GRCh38")),
        "n_unique_mhc_entrez_ids_in_gene_loc": 0,
        "n_unique_mhc_ensembl_ids_after_conversion": 0,
    }

    grch38_path = getattr(reference, "gene_loc_file_grch38", None)
    grch37_path = getattr(reference, "gene_loc_file", None)

    # If the user explicitly
    # configured a GRCh38 gene-loc but the file is missing, that is an
    # operator error - not a license to silently degrade to the GRCh37
    # fallback.  Only enter the fallback path when *gene_loc_file_grch38*
    # is genuinely unset (None).
    if grch38_path is not None and not Path(grch38_path).exists():
        raise FileNotFoundError(
            f"reference.gene_loc_file_grch38 is configured ({grch38_path!r}) "
            "but the file does not exist. Run `repogen setup-resources "
            "--config configs/resources.yaml` to fetch the official GRCh38 "
            "MAGMA gene-location file (ncbi_gene_loc_grch38), or unset the "
            "config field to opt into the GRCh37 + Ensembl-projection fallback."
        )

    def _filter_to_mhc(df: pd.DataFrame, build: str) -> pd.DataFrame:
        chr_, start, end = mhc_interval(build)
        chr_match = df["chr"].astype("Int64") == chr_
        start_col = pd.to_numeric(df.get("start"), errors="coerce")
        end_col = pd.to_numeric(df.get("end"), errors="coerce")
        # Interval-overlap (start <= MHC_end AND end >= MHC_start).
        overlap = (start_col <= end) & (end_col >= start)
        return df.loc[chr_match & overlap].copy()

    def _entrez_to_ensembl(entrez_ids: pd.Series) -> dict[int, str]:
        """Return ``{entrez_id: ensembl_id}`` for resolvable entries."""
        mapping: dict[int, str] = {}
        if gene_id_converter is None:
            return mapping
        for eid in entrez_ids.dropna().astype(int).unique().tolist():
            try:
                record = gene_id_converter.get_full_record(str(eid), "entrez")
            except (KeyError, ValueError):
                continue
            if record and record.get("ensembl"):
                mapping[eid] = str(record["ensembl"])
        return mapping

    # --- Primary path: GRCh38 gene-loc ------------------------------------
    if grch38_path is not None:
        try:
            loc_df = _read_gene_location_file(Path(grch38_path))
        except RuntimeError as exc:
            logger.warning(
                "Failed to read GRCh38 gene-loc file %s: %s; falling through",
                grch38_path, exc,
            )
        else:
            mhc_rows = _filter_to_mhc(loc_df, "GRCh38")
            entrez_ids = mhc_rows["gene_entrez_id"].dropna().astype(int)
            n_entrez = int(entrez_ids.nunique())
            metadata["n_unique_mhc_entrez_ids_in_gene_loc"] = n_entrez

            existing_ens = mhc_rows.get("gene_ensembl_id")
            ens_direct = (
                existing_ens.dropna().astype(str).tolist()
                if existing_ens is not None else []
            )

            entrez_to_ens = _entrez_to_ensembl(entrez_ids)
            ens_via_converter = list(entrez_to_ens.values())

            all_ensembl = sorted(set(ens_direct) | set(ens_via_converter))
            metadata["n_unique_mhc_ensembl_ids_after_conversion"] = len(all_ensembl)

            if all_ensembl:
                chr_, start, end = mhc_interval("GRCh38")
                annotation = pd.DataFrame({
                    "gene_ensembl_id": all_ensembl,
                    "chr": chr_,
                    "start": start,
                    "end": end,
                })
                metadata.update({
                    "mhc_annotation_source": str(grch38_path),
                    "mhc_annotation_build": "GRCh38",
                    "mhc_annotation_strategy": "grch38_gene_loc",
                    "mhc_source_interval": [chr_, start, end],
                })
                logger.info(
                    "MHC annotation loaded: GRCh38 gene-loc %s (%d MHC Entrez -> %d Ensembl)",
                    grch38_path, n_entrez, len(all_ensembl),
                )
                return annotation, metadata

            logger.warning(
                "GRCh38 gene-loc %s yielded zero MHC genes after Entrez->Ensembl "
                "conversion; falling back",
                grch38_path,
            )

    # --- Fallback: GRCh37 gene-loc with Ensembl projection ----------------
    if grch37_path is not None and Path(grch37_path).exists():
        try:
            loc_df = _read_gene_location_file(Path(grch37_path))
        except RuntimeError as exc:
            logger.warning(
                "Failed to read GRCh37 gene-loc file %s: %s",
                grch37_path, exc,
            )
        else:
            mhc_rows = _filter_to_mhc(loc_df, "GRCh37")
            entrez_ids = mhc_rows["gene_entrez_id"].dropna().astype(int)
            n_entrez = int(entrez_ids.nunique())
            metadata["n_unique_mhc_entrez_ids_in_gene_loc"] = n_entrez

            existing_ens = mhc_rows.get("gene_ensembl_id")
            ens_direct = (
                existing_ens.dropna().astype(str).tolist()
                if existing_ens is not None else []
            )

            entrez_to_ens = _entrez_to_ensembl(entrez_ids)
            ens_via_converter = list(entrez_to_ens.values())

            all_ensembl = sorted(set(ens_direct) | set(ens_via_converter))
            metadata["n_unique_mhc_ensembl_ids_after_conversion"] = len(all_ensembl)

            if all_ensembl:
                src_chr, src_start, src_end = mhc_interval("GRCh37")
                chr_, start, end = mhc_interval("GRCh38")
                annotation = pd.DataFrame({
                    "gene_ensembl_id": all_ensembl,
                    "chr": chr_,
                    "start": start,
                    "end": end,
                })
                metadata.update({
                    "mhc_annotation_source": str(grch37_path),
                    "mhc_annotation_build": "GRCh37",
                    "mhc_annotation_strategy": "grch37_gene_loc_ensembl_projection",
                    "mhc_source_interval": [src_chr, src_start, src_end],
                })
                logger.warning(
                    "MHC annotation falling back to GRCh37 gene-loc %s "
                    "(%d MHC Entrez -> %d Ensembl); add gene_loc_file_grch38 "
                    "to ReferenceConfig for the GRCh38-native path.",
                    grch37_path, n_entrez, len(all_ensembl),
                )
                return annotation, metadata

    # --- No annotation -----------------------------------------------------
    if require_mhc_annotation:
        raise RuntimeError(
            "No MHC gene annotation could be loaded but it is required. "
            "Run `repogen setup-resources --config configs/resources.yaml` to "
            "fetch the GRCh38 gene-loc file, or set "
            "reference.gene_loc_file_grch38 / reference.gene_loc_file in the "
            "pipeline config."
        )
    logger.warning(
        "No MHC gene annotation could be loaded "
        "(reference.gene_loc_file_grch38=%r, reference.gene_loc_file=%r, "
        "gene_id_converter=%s). MHC flagging will be inert. "
        "Run `repogen setup-resources` to populate the GRCh38 gene-loc.",
        grch38_path, grch37_path,
        "yes" if gene_id_converter is not None else "no",
    )
    return None, metadata


def add_mhc_flag(
    df: pd.DataFrame,
    gene_annotation: pd.DataFrame | None = None,
    build: str = "GRCh38",
) -> pd.DataFrame:
    """Add ``mhc_flag`` column based on gene coordinates.

    contract is that the *gene_annotation* coordinate frame
    and the *build* argument must refer to the same genome assembly.  The
    function does not translate between builds; the caller is
    responsible for picking a build-consistent ``(annotation, bounds)``
    pair.  The output ``mhc_flag`` column is keyed on
    ``gene_ensembl_id`` - which is build-invariant for stable Ensembl
    genes - so a fallback caller can legitimately compute the flag on
    GRCh37 coordinates and join it back to GRCh38 output.

    Args:
        df: DataFrame with ``gene_ensembl_id`` column.
        gene_annotation: Optional ``GeneAnnotationRecord``-shaped
            DataFrame with ``gene_ensembl_id``, ``chr``, ``start``,
            ``end`` columns.  When ``None`` or empty, every row receives
            ``mhc_flag = False`` (the historical behaviour preserved for
            minimal fixtures).
        build: Genome build of *gene_annotation* coordinates, either
            ``"GRCh38"`` or ``"GRCh37"``.

    Returns:
        DataFrame with ``mhc_flag`` column added.

    Raises:
        ValueError: If *build* is not one of the two supported builds
            (propagated from ``mhc_interval``).
    """
    df = df.copy()
    df["mhc_flag"] = False

    if gene_annotation is not None and not gene_annotation.empty:
        mhc_chr, mhc_start, mhc_end = mhc_interval(build)
        ann = gene_annotation[["gene_ensembl_id", "chr", "start", "end"]].drop_duplicates(
            subset="gene_ensembl_id"
        )
        ann_map = ann.set_index("gene_ensembl_id")

        matched = df["gene_ensembl_id"].isin(ann_map.index)
        if matched.any():
            matched_ids = df.loc[matched, "gene_ensembl_id"]
            coords = ann_map.loc[matched_ids.values]
            in_mhc = (
                (coords["chr"].values == mhc_chr)
                & (coords["start"].values <= mhc_end)
                & (coords["end"].values >= mhc_start)
            )
            df.loc[matched, "mhc_flag"] = in_mhc

    return df
