"""Multi-source drug-target integration module.

Queries ChEMBL v35 SQLite directly for full drug-target records with
mechanism, affinity, indications, and ATC codes.  Optionally supplements
with PDSP Ki data and DGIdb interactions.  Outputs the
:class:`DrugTargetRecord` schema.

This is a complete redesign from the old ``drug_gene_enrichment.py``
which loaded pre-processed CSVs and lost most of ChEMBL's richness.

Sources
-------
* ChEMBL (primary) - relational tables via SQLite.
* PDSP (optional) - binding affinity specialist, CNS focus.
* DGIdb (optional) - meta-aggregator for confidence scoring.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from repogen.data.gene_id_converter import GeneIDConverter
from repogen.data.schemas import validate_dataframe
from repogen.utils.constants import (
    INTERACTION_TYPE_STANDARDISATION,
    PDSP_TARGET_MAP,
)
from repogen.utils.io import check_file_exists
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

_NON_SLUG = re.compile(r"[^A-Z0-9_-]+")
_MULTI_UNDER = re.compile(r"_{2,}")
_CHEMBL_TOKEN = re.compile(r"CHEMBL\d+", re.IGNORECASE)


def _extract_chembl_id(value: str) -> str | None:
    """Extract a canonical ``CHEMBL\\d+`` token from a concept-ID string.

    Handles bare IDs (``CHEMBL12345``), namespaced IDs
    (``chembl:CHEMBL12345``), and mixed case.  Returns ``None`` if no
    token is found or if the string contains multiple distinct tokens
    (ambiguous).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    matches = _CHEMBL_TOKEN.findall(value)
    if not matches:
        return None
    unique = set(m.upper() for m in matches)
    if len(unique) > 1:
        return None
    return unique.pop()


def _sanitize_placeholder_id(prefix: str, name: pd.Series) -> pd.Series:
    """Create ASCII-safe placeholder IDs from drug names.

    Normalizes to NFKD, strips non-ASCII, replaces non-alphanumeric
    chars with ``_``, and collapses runs of underscores.  NA-safe:
    missing values produce ``prefix`` with an empty slug.

    When distinct original names produce the same slug (e.g. enantiomer
    suffixes stripped by ASCII folding), an 8-char MD5 hash of the
    original name is appended to disambiguate.
    """
    safe_name = name.astype(object).fillna("").astype(str)
    upper = safe_name.str.upper().str.strip()
    ascii_safe = upper.apply(
        lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        if s else ""
    )
    slugs = ascii_safe.str.replace(_NON_SLUG, "_", regex=True)
    slugs = slugs.str.replace(_MULTI_UNDER, "_", regex=True)
    slugs = slugs.str.strip("_")
    base_ids = prefix + slugs

    deduped = pd.DataFrame({"slug": base_ids, "orig": safe_name}).drop_duplicates()
    names_per_slug = deduped.groupby("slug")["orig"].nunique()
    colliding_slugs = set(names_per_slug[names_per_slug > 1].index)

    if colliding_slugs:
        logger.warning(
            "%d placeholder-ID slugs map to >1 distinct drug name; "
            "appending hash suffix to disambiguate",
            len(colliding_slugs),
        )
        needs_hash = base_ids.isin(colliding_slugs)
        hash_suffix = safe_name.apply(
            lambda s: "_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
            if s else ""
        )
        base_ids = base_ids.where(~needs_hash, base_ids + hash_suffix)

    return base_ids


_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _normalize_drug_name(name: object) -> str:
    """Normalize a drug name to a stable lookup key.

    Pipeline:
      1. NFKD decomposition (e.g. ``"étodolac" -> "etodolac"``,
         ``"½" -> "1/2"``).
      2. ASCII-only encode (drop combining marks and non-ASCII).
      3. Casefold (Unicode-aware lower-case).
      4. Strip everything that is not ``[a-z0-9]``.

    The NFKD+ASCII step matches the existing :func:`_sanitize_placeholder_id`
    convention so name-normalization is consistent across the loader.

    Returns an empty string for ``None``, NaN, or non-stringifiable input,
    which acts as a "do-not-match" sentinel for the synonym index.
    """
    if name is None:
        return ""
    try:
        if isinstance(name, float) and pd.isna(name):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(name)
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub("", ascii_only.casefold())


# ---------------------------------------------------------------------------
# UniChem PubChem CID enrichment
# ---------------------------------------------------------------------------

_CHEMBL_ID_RE = re.compile(r"^CHEMBL\d+$")


def _parse_unichem_mapping(path: Path) -> dict[str, list[str]]:
    """Parse UniChem src1->src22 mapping into {ChEMBL_ID: [PubChem_CIDs]}.

    Skips non-data lines (header, blank). Accepts only rows where col1
    matches ``CHEMBL\\d+`` and col2 is a positive integer.
    """
    mapping: dict[str, list[str]] = defaultdict(list)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[call-overload]
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            chembl_id, cid = parts
            if not _CHEMBL_ID_RE.match(chembl_id):
                continue
            if not cid.isdigit():
                continue
            mapping[chembl_id].append(cid)
    logger.info(
        "UniChem mapping loaded: %d ChEMBL IDs, %d total CID pairs",
        len(mapping), sum(len(v) for v in mapping.values()),
    )
    return dict(mapping)


def _enrich_pubchem_cid(
    df: pd.DataFrame,
    mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """Populate ``drug_pubchem_cid`` and ``_pubchem_cid_all`` from UniChem.

    For the scalar ``drug_pubchem_cid``, the smallest numeric CID is
    chosen as a deterministic canonical.  ``_pubchem_cid_all`` stores every
    mapped CID (sorted ascending) as a list, used for expanded matching.
    Only fills rows where ``drug_pubchem_cid`` is currently null or empty.
    """
    if "drug_chembl_id" not in df.columns:
        return df

    df = df.copy()
    if "drug_pubchem_cid" not in df.columns:
        df["drug_pubchem_cid"] = pd.NA

    needs_cid = df["drug_pubchem_cid"].isna() | (df["drug_pubchem_cid"].astype(str).str.strip() == "")

    canonical = df["drug_chembl_id"].map(
        lambda cid: min(mapping.get(cid, []), key=int, default=None)
    )
    canonical = canonical.where(canonical.notna(), pd.NA).astype(str)
    canonical = canonical.replace({"None": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    df.loc[needs_cid, "drug_pubchem_cid"] = canonical[needs_cid]

    all_cids = df["drug_chembl_id"].map(
        lambda cid: sorted(mapping.get(cid, []), key=int) or None
    )
    df["_pubchem_cid_all"] = all_cids

    n_enriched = df["drug_pubchem_cid"].notna().sum() - (~needs_cid).sum()
    n_drugs = df["drug_chembl_id"].nunique()
    n_with_cid = df.loc[df["drug_pubchem_cid"].notna(), "drug_chembl_id"].nunique()
    logger.info(
        "PubChem CID enrichment: %d/%d unique drugs now have CIDs "
        "(%d rows enriched in this step)",
        n_with_cid, n_drugs, n_enriched,
    )
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_drug_targets(
    chembl_sqlite: Optional[Path] = None,
    pdsp_csv: Optional[Path] = None,
    dgidb_tsv: Optional[Path] = None,
    gene_id_converter: Optional[GeneIDConverter] = None,
    sources: list[str] | None = None,
    min_pchembl: Optional[float] = None,
    max_phase_filter: Optional[int] = None,
    target_type_filter: list[str] | None = None,
    chembl_scope: str = "mechanism_only",
    unichem_mapping: Optional[Path] = None,
    parent_salt_unification: bool = True,
    *,
    expression_sources: Optional[list[str]] = None,
    creeds_path: Optional[Path] = None,
    dsigdb_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and integrate drug-target interactions from multiple sources.

    Args:
        chembl_sqlite: Path to ChEMBL v35 SQLite database.
        pdsp_csv: Path to PDSP Ki database CSV export.
        dgidb_tsv: Path to DGIdb interactions TSV download.
        gene_id_converter: Initialised converter for gene ID mapping.
        sources: Which target-family sources to load (subset of
            ``{"chembl", "pdsp", "dgidb"}``).  Defaults to
            ``["chembl"]``.
        min_pchembl: Minimum pChEMBL value for ChEMBL filtering.
        max_phase_filter: Minimum clinical phase for ChEMBL filtering.
        target_type_filter: ChEMBL target_type whitelist.
        chembl_scope: ``"mechanism_only"`` (default) or
            ``"mechanism_or_affinity"``.
        unichem_mapping: Path to UniChem src1->src22 (ChEMBL->PubChem CID)
            mapping file.  If provided, populates ``drug_pubchem_cid``.
        parent_salt_unification: When ``True`` (default) and ChEMBL is
            in ``sources``, child / salt CIDs are collapsed onto their
            parent CID before drug-gene dedup. See.
        expression_sources: Optional list of expression-perturbation
            sources to additionally load.  Subset of
            ``{"creeds", "dsigdb"}``.  Strictly disjoint from
            *sources* - kwarg-only to make accidental cross-mixing
            obvious at call sites.
        creeds_path: Path to ``single_drug_perturbations-v1.0.json``
            (required iff ``"creeds" in expression_sources``).
        dsigdb_path: Path to the DSigDB D3 file (required iff
            ``"dsigdb" in expression_sources``).

    Returns:
        Unified :class:`DrugTargetRecord` DataFrame.

    Raises:
        FileNotFoundError: If a required database file is missing.
        RuntimeError: If no records survive filtering.
        ValueError: If *sources* and *expression_sources* overlap.
    """
    if sources is None:
        sources = ["chembl"]
    if target_type_filter is None:
        target_type_filter = ["SINGLE PROTEIN"]

    # Defence-in-depth disjoint-set check at the
    # function boundary.  The Pydantic schema already enforces this at
    # config-load time, but programmatic callers (tests, ad-hoc
    # scripts) may bypass the schema.
    if expression_sources is None:
        expression_sources = []
    overlap = set(sources) & set(expression_sources)
    if overlap:
        raise ValueError(
            f"sources and expression_sources must be disjoint; "
            f"got overlap: {sorted(overlap)}"
        )

    dataframes: dict[str, pd.DataFrame] = {}

    if "chembl" in sources:
        if chembl_sqlite is None:
            raise FileNotFoundError(
                "ChEMBL SQLite path required when 'chembl' is in sources"
            )
        chembl_sqlite = check_file_exists(chembl_sqlite, label="ChEMBL SQLite")
        dataframes["chembl"] = load_chembl(
            sqlite_path=chembl_sqlite,
            gene_id_converter=gene_id_converter,
            min_pchembl=min_pchembl,
            max_phase_filter=max_phase_filter,
            target_type_filter=target_type_filter,
            chembl_scope=chembl_scope,
        )

    if "pdsp" in sources:
        if pdsp_csv is None:
            raise FileNotFoundError(
                "PDSP CSV path required when 'pdsp' is in sources"
            )
        pdsp_csv = check_file_exists(pdsp_csv, label="PDSP Ki database")
        dataframes["pdsp"] = load_pdsp(
            pdsp_csv=pdsp_csv,
            gene_id_converter=gene_id_converter,
        )

    if "dgidb" in sources:
        if dgidb_tsv is None:
            raise FileNotFoundError(
                "DGIdb TSV path required when 'dgidb' is in sources"
            )
        dgidb_tsv = check_file_exists(dgidb_tsv, label="DGIdb interactions")
        dataframes["dgidb"] = load_dgidb(
            interactions_tsv=dgidb_tsv,
            gene_id_converter=gene_id_converter,
        )

    # Expression-perturbation sources. These are
    # opt-in (gated by DrugEnrichmentConfig.enable_expression_enrichment
    # at the workflow level) and produce TARGETS+EXPR data only when
    # the caller passes a non-empty *expression_sources* list.
    if "creeds" in expression_sources:
        if creeds_path is None:
            raise FileNotFoundError(
                "creeds_path required when 'creeds' is in expression_sources"
            )
        creeds_path = check_file_exists(creeds_path, label="CREEDS perturbations")
        dataframes["creeds"] = load_creeds(
            creeds_json=creeds_path,
            gene_id_converter=gene_id_converter,
        )

    if "dsigdb" in expression_sources:
        if dsigdb_path is None:
            raise FileNotFoundError(
                "dsigdb_path required when 'dsigdb' is in expression_sources"
            )
        dsigdb_path = check_file_exists(dsigdb_path, label="DSigDB D3")
        dataframes["dsigdb"] = load_dsigdb(
            dsigdb_path=dsigdb_path,
            gene_id_converter=gene_id_converter,
        )

    if not dataframes:
        raise RuntimeError("No drug-target sources loaded")

    for name, df in dataframes.items():
        logger.info("Source '%s': %d records loaded", name, len(df))

    # Build the ChEMBL synonym index once (only when ChEMBL is in the
    # source set).  This adds a small SQLite reopen + scan (~6 s on
    # ChEMBL 35) that we accept rather than mutating ``load_chembl``'s
    # public return type, which has broad blast radius across tests,
    # CLI mocks, and Snakemake invocations.  Passes the loaded chembl
    # CIDs as ``extra_cids`` so the parent map covers all drug
    # rows it will canonicalize, not just the candidate-name subset
    # (~75% of redundancy contributors live outside the candidate set
    # on real ChEMBL 35).
    chembl_name_index: Optional[ChemblNameIndex] = None
    if "chembl" in sources and chembl_sqlite is not None:
        try:
            loaded_chembl_cids: Optional[set[str]] = None
            if "chembl" in dataframes and not dataframes["chembl"].empty:
                loaded_chembl_cids = set(
                    dataframes["chembl"]["drug_chembl_id"]
                    .dropna().astype(str).unique()
                )
            chembl_name_index = load_chembl_synonym_index(
                chembl_sqlite,
                extra_cids=loaded_chembl_cids,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to build ChEMBL synonym index (%s); "
                "DGIdb name fallback, ATC propagation, and parent/salt "
                "unification will be no-ops",
                exc,
            )
            chembl_name_index = None

    result = merge_sources(
        dataframes,
        gene_id_converter=gene_id_converter,
        chembl_name_index=chembl_name_index,
        parent_salt_unification=parent_salt_unification,
    )

    if result.empty:
        raise RuntimeError(
            "No drug-target records survived integration. "
            "Check source files and filter parameters."
        )

    if unichem_mapping is not None:
        uc_path = Path(unichem_mapping)
        if uc_path.exists():
            uc_map = _parse_unichem_mapping(uc_path)
            result = _enrich_pubchem_cid(result, uc_map)
        else:
            logger.warning(
                "UniChem mapping file not found (%s); "
                "drug_pubchem_cid will remain empty",
                uc_path,
            )

    errors = validate_dataframe(result, "DrugTargetRecord")
    if errors:
        logger.warning(
            "DrugTargetRecord validation found %d issue(s); "
            "proceeding with available data",
            len(errors),
        )

    logger.info(
        "Drug-target integration complete: %d records, %d unique drugs, "
        "%d unique genes, sources=%s",
        len(result),
        result["drug_chembl_id"].nunique(),
        result["gene_symbol"].nunique(),
        list(result["source"].unique()),
    )
    return result


# ---------------------------------------------------------------------------
# ChEMBL source (primary)
# ---------------------------------------------------------------------------

_CHEMBL_QUERY_MECHANISMS = """\
SELECT
    md.molregno             AS _molregno,
    md.chembl_id            AS drug_chembl_id,
    md.pref_name            AS drug_name,
    md.max_phase,
    md.molecule_type,
    cs.standard_inchi_key   AS drug_inchikey,
    cs.canonical_smiles     AS drug_smiles,
    dm.mechanism_of_action,
    dm.action_type          AS interaction_type,
    td.pref_name            AS target_name,
    td.target_type,
    cseq.accession          AS uniprot_id,
    dw.warning_type         AS withdrawal_reason,
    atc_agg.atc_codes_raw,
    di_agg.indications_raw
FROM molecule_dictionary md
JOIN molecule_hierarchy mh ON md.molregno = mh.molregno
LEFT JOIN compound_structures cs ON mh.parent_molregno = cs.molregno
LEFT JOIN drug_mechanism dm ON md.molregno = dm.molregno
LEFT JOIN target_dictionary td ON dm.tid = td.tid
LEFT JOIN target_components tc ON td.tid = tc.tid
LEFT JOIN component_sequences cseq ON tc.component_id = cseq.component_id
LEFT JOIN (
    SELECT d.molregno, GROUP_CONCAT(d.level5, '|') AS atc_codes_raw
    FROM (
        SELECT DISTINCT molregno, level5
        FROM molecule_atc_classification
        WHERE level5 IS NOT NULL
    ) d
    GROUP BY d.molregno
) atc_agg ON md.molregno = atc_agg.molregno
LEFT JOIN (
    SELECT d.molregno, GROUP_CONCAT(d.mesh_heading, '|') AS indications_raw
    FROM (
        SELECT DISTINCT molregno, mesh_heading
        FROM drug_indication
        WHERE mesh_heading IS NOT NULL
    ) d
    GROUP BY d.molregno
) di_agg ON md.molregno = di_agg.molregno
LEFT JOIN drug_warning dw ON md.molregno = dw.molregno
WHERE td.target_type IN ({target_types})
  AND td.organism = 'Homo sapiens'
GROUP BY md.chembl_id, dm.mec_id, cseq.accession
"""

_CHEMBL_QUERY_PARENT_ATC = """\
SELECT d.molregno, GROUP_CONCAT(d.level5, '|') AS parent_atc_raw
FROM (
    SELECT DISTINCT mh.molregno, mac.level5
    FROM molecule_hierarchy mh
    JOIN molecule_atc_classification mac ON mh.parent_molregno = mac.molregno
    WHERE mac.level5 IS NOT NULL
      AND mh.molregno != mh.parent_molregno
) d
GROUP BY d.molregno
"""

_CHEMBL_QUERY_PARENT_INDICATIONS = """\
SELECT d.molregno, GROUP_CONCAT(d.mesh_heading, '|') AS parent_indications_raw
FROM (
    SELECT DISTINCT mh.molregno, di.mesh_heading
    FROM molecule_hierarchy mh
    JOIN drug_indication di ON mh.parent_molregno = di.molregno
    WHERE di.mesh_heading IS NOT NULL
      AND mh.molregno != mh.parent_molregno
) d
GROUP BY d.molregno
"""

_CHEMBL_QUERY_AFFINITIES = """\
SELECT
    md.chembl_id,
    md.pref_name             AS drug_name,
    md.max_phase,
    cseq.accession           AS uniprot_id,
    act.standard_type        AS affinity_type,
    act.standard_value       AS affinity_value,
    act.standard_units       AS affinity_unit,
    act.pchembl_value
FROM activities act
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
JOIN target_components tc ON td.tid = tc.tid
JOIN component_sequences cseq ON tc.component_id = cseq.component_id
JOIN molecule_dictionary md ON act.molregno = md.molregno
WHERE act.standard_type IN ('Ki', 'IC50', 'EC50', 'Kd')
  AND act.standard_units = 'nM'
  AND act.pchembl_value IS NOT NULL
  AND act.data_validity_comment IS NULL
  AND td.target_type IN ({target_types})
  AND td.organism = 'Homo sapiens'
ORDER BY md.chembl_id, cseq.accession, act.pchembl_value DESC
"""


def _build_chembl_annotation_dicts(
    conn: sqlite3.Connection,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build drug-level (chembl_id-keyed) ATC and indication dictionaries.

    Self+parent semantics: for each ``chembl_id`` the value is the union of
    annotations attached to (a) the drug's own ``molregno`` and (b) its
    ``parent_molregno`` via ``molecule_hierarchy`` (when present).  This
    matches the existing row-level direct + parent-aware logic but is
    applied at the drug level so it reaches affinity-only rows that lose
    ``_molregno`` during the outer merge in :func:`_join_mechanism_affinity`.

    Drugs with no annotation in either anchor are simply omitted from the
    returned dict so ``dict.get(chembl_id)`` returns ``None``, preserving
    the ``Optional[list[str]]`` schema contract on ``DrugTargetRecord``.

    Sibling propagation (and reverse child->parent propagation) is
    intentionally out of scope here; the inverted cases in ChEMBL 35
    are rare and exclusively non-psychiatric.
    """
    atc_df = pd.read_sql_query(
        "SELECT molregno, level5 FROM molecule_atc_classification "
        "WHERE level5 IS NOT NULL",
        conn,
    )
    if atc_df.empty:
        direct_atc: dict[int, set[str]] = {}
    else:
        direct_atc = (
            atc_df.groupby("molregno")["level5"].agg(set).to_dict()
        )

    ind_df = pd.read_sql_query(
        "SELECT molregno, mesh_heading FROM drug_indication "
        "WHERE mesh_heading IS NOT NULL",
        conn,
    )
    if ind_df.empty:
        direct_ind: dict[int, set[str]] = {}
    else:
        direct_ind = (
            ind_df.groupby("molregno")["mesh_heading"].agg(set).to_dict()
        )

    if not direct_atc and not direct_ind:
        return {}, {}

    map_df = pd.read_sql_query(
        """
        SELECT md.molregno, md.chembl_id, mh.parent_molregno
        FROM molecule_dictionary md
        LEFT JOIN molecule_hierarchy mh ON md.molregno = mh.molregno
        """,
        conn,
    )

    relevant_mols = set(direct_atc) | set(direct_ind)
    if relevant_mols:
        mask = (
            map_df["molregno"].isin(relevant_mols)
            | map_df["parent_molregno"].isin(relevant_mols)
        )
        map_df = map_df[mask]

    chembl_to_atc: dict[str, list[str]] = {}
    chembl_to_ind: dict[str, list[str]] = {}

    for cid, m, p in zip(
        map_df["chembl_id"].to_numpy(),
        map_df["molregno"].to_numpy(),
        map_df["parent_molregno"].to_numpy(),
    ):
        if cid is None or pd.isna(cid):
            continue
        if pd.isna(m):
            continue
        anchors: set[int] = {int(m)}
        if pd.notna(p):
            anchors.add(int(p))

        atc_codes: set[str] = set()
        ind_terms: set[str] = set()
        for k in anchors:
            atc_codes |= direct_atc.get(k, set())
            ind_terms |= direct_ind.get(k, set())

        if atc_codes:
            chembl_to_atc[str(cid)] = sorted(atc_codes)
        if ind_terms:
            chembl_to_ind[str(cid)] = sorted(ind_terms)

    logger.info(
        "ChEMBL annotation dicts built: %d drugs with ATC, %d with indications",
        len(chembl_to_atc), len(chembl_to_ind),
    )

    return chembl_to_atc, chembl_to_ind


# ---------------------------------------------------------------------------
# Synonym index for cross-source drug-ID resolution
# ---------------------------------------------------------------------------


_TIER_PREF = "pref"
_TIER_SYN = "syn"
_TIER_WHO = "who"
_TIERS = (_TIER_PREF, _TIER_SYN, _TIER_WHO)


@dataclass(frozen=True)
class ChemblNameIndex:
    """Tiered ChEMBL name index for cross-source drug-ID resolution.

    Built once per loader run and consumed by
    :func:`_match_dgidb_to_chembl` and
    :func:`_propagate_chembl_atc_to_remapped`.  The three tiers
    correspond to ChEMBL tables in descending precedence:
    ``molecule_dictionary.pref_name``,
    ``molecule_synonyms.synonyms``, and
    ``atc_classification.who_name``.

    Each tier carries:

    * ``tier_resolved[tier]``: normalized key -> unique ``chembl_id``.
      For ambiguous keys, intra-tier parent-collapse may resolve them;
      otherwise the key is omitted from this map.
    * ``tier_candidates[tier]``: normalized key -> set of candidate
      ``chembl_id`` values (preserves the raw ambiguity so the
      resolver can apply parent-consistent rescue across tiers).

    Plus shared fields:

    * ``parent_of_cid``: ``chembl_id`` -> ``parent_molregno`` (or
      ``None`` if the molecule has no hierarchy entry).
    * ``parent_chembl_id_of_cid``: ``chembl_id`` -> parent ``chembl_id``
      (with self-parent as fallback when no hierarchy edge exists).
      Used by parent/salt unification to canonicalize child
      and salt CIDs to their parent compound's CID before drug-gene
      dedup.  Built over the union of (a) name-index candidate CIDs
      and (b) any ``extra_cids`` passed to
      :func:`load_chembl_synonym_index` (typically the loaded
      pharmacology rows from :func:`load_chembl`), so coverage scales
      with what is actually being canonicalized rather than what is
      name-annotated. See.
    * ``chembl_to_atc`` / ``chembl_to_ind``: drug-level dicts from
      :func:`_build_chembl_annotation_dicts`, used to propagate ATC
      and indications to remapped DGIdb (and any future) rows whose
      ``drug_chembl_id`` resolves to a real ChEMBL drug but whose
      pharmacology was not loaded under the current scope.
    """

    tier_resolved: dict[str, dict[str, str]]
    tier_candidates: dict[str, dict[str, set[str]]]
    parent_of_cid: dict[str, Optional[int]]
    parent_chembl_id_of_cid: dict[str, str]
    chembl_to_atc: dict[str, list[str]]
    chembl_to_ind: dict[str, list[str]]


def _collapse_to_parent(
    cids: set[str],
    parent_of_cid: dict[str, Optional[int]],
    molregno_to_cid: dict[int, str],
) -> Optional[str]:
    """Pure dict-lookup parent-collapse logic.

    If ``cids`` has a single element, return it.  Otherwise, try to
    fold them onto a single ``parent_molregno`` whose ``chembl_id`` is
    known, and return that parent's ``chembl_id``.  Returns ``None`` if
    any candidate has no parent or the candidates span multiple
    parents.
    """
    if not cids:
        return None
    if len(cids) == 1:
        return next(iter(cids))
    parents: set[int] = set()
    for c in cids:
        p = parent_of_cid.get(c)
        if p is None:
            return None
        parents.add(int(p))
    if len(parents) != 1:
        return None
    parent = next(iter(parents))
    return molregno_to_cid.get(parent)


def _query_in_chunks(
    conn: sqlite3.Connection,
    query_template: str,
    params_list: list,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Run a ``SELECT ... WHERE col IN ({placeholders})`` query in safe chunks.

    SQLite has a per-statement variable limit
    (``SQLITE_MAX_VARIABLE_NUMBER``: default 999 on ≤3.31, 32766 on
    ≥3.32).  ChEMBL 35's candidate molregno set is ~100k entries
    (48,688 pref + 89,341 syn + 3,498 who, with overlap), which
    exceeds the default limit on every modern SQLite build and
    raises ``sqlite3.OperationalError: too many SQL variables``.

    This helper splits ``params_list`` into batches no larger than
    ``chunk_size``, executes the query for each batch, and concatenates
    the results.  ``chunk_size=500`` is well below all default limits
    in the wild and adds <100 ms of total overhead on a ~100k-entry
    list (sub-millisecond per chunk × ~200 chunks).

    The input is deduplicated once (order-preserving via
    ``dict.fromkeys``) before chunking.  This is semantically harmless
    because ``IN`` is set-semantic, and avoids wasted queries +
    duplicate rows if a caller passes a non-deduped list.

    ``query_template`` must contain a single ``{placeholders}`` token
    (formatted as ``"?,?,?"``); the helper substitutes the per-chunk
    placeholder string before execution.

    Returns an empty :class:`pandas.DataFrame` when ``params_list`` is
    empty (no SQLite call issued).
    """
    if not params_list:
        return pd.DataFrame()
    deduped = list(dict.fromkeys(params_list))
    chunks: list[pd.DataFrame] = []
    for i in range(0, len(deduped), chunk_size):
        batch = deduped[i:i + chunk_size]
        sql = query_template.format(placeholders=",".join("?" * len(batch)))
        chunks.append(pd.read_sql_query(sql, conn, params=batch))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _build_chembl_synonym_index(
    conn: sqlite3.Connection,
    extra_cids: Optional[Iterable[str]] = None,
) -> ChemblNameIndex:
    """Build a tiered name -> ``chembl_id`` index from a ChEMBL connection.

    Pulls three tiers (``pref_name``, ``synonyms``, ``who_name``) and
    builds, for each tier independently:

    * a candidate map (key -> ``set[chembl_id]``) keeping every raw
      candidate per normalized key, and
    * a resolved map (key -> ``chembl_id``) where multi-candidate keys
      are reduced via :func:`_collapse_to_parent`.

    Also builds:

    * a ``parent_of_cid`` dict scoped to the union of all
      tier-candidate molregnos plus the parents they point to (so the
      ``molecule_hierarchy`` scan stays small even though the table
      itself is millions of rows in ChEMBL 35),
    * a derived ``parent_chembl_id_of_cid`` dict mapping each known
      ``chembl_id`` to its parent's ``chembl_id`` (self-parent
      fallback when no hierarchy edge exists), used by parent/salt
      unification to
      canonicalize salt/child CIDs before drug-gene dedup, and
    * the same drug-level ``chembl_to_atc`` / ``chembl_to_ind`` dicts
      used by the mechanism+affinity merge (rebuilt here from the same
      connection so the
      index is self-contained).

    Parameters
    ----------
    extra_cids :
        Optional iterable of additional ``chembl_id`` strings to
        extend the parent mapping over.  The candidate-scoped
        coverage above only includes molecules with ``pref_name``,
        ``synonyms``, or ``who_name`` (~100k of ~2.5M molecules in
        ChEMBL 35), which misses the bulk of unlabeled bioassay
        compounds loaded by :func:`load_chembl`.  Pass the loaded
        chembl ``drug_chembl_id`` set here to ensure the parent/salt
        parent/salt canonicalization covers all rows it will
        actually rewrite.  Missing CIDs degrade gracefully to
        self-parent (the canonicalization helper treats unknown
        CIDs as already-canonical). See.
    """
    pref = pd.read_sql_query(
        "SELECT molregno, chembl_id, pref_name FROM molecule_dictionary "
        "WHERE pref_name IS NOT NULL AND chembl_id IS NOT NULL",
        conn,
    )
    syn = pd.read_sql_query(
        "SELECT md.molregno, md.chembl_id, ms.synonyms "
        "FROM molecule_synonyms ms "
        "JOIN molecule_dictionary md ON md.molregno = ms.molregno "
        "WHERE ms.synonyms IS NOT NULL AND md.chembl_id IS NOT NULL",
        conn,
    )
    # WHO names live on atc_classification rows, joined to molecules
    # via molecule_atc_classification.
    who = pd.read_sql_query(
        "SELECT md.molregno, md.chembl_id, ac.who_name "
        "FROM atc_classification ac "
        "JOIN molecule_atc_classification mac ON mac.level5 = ac.level5 "
        "JOIN molecule_dictionary md ON md.molregno = mac.molregno "
        "WHERE ac.who_name IS NOT NULL AND md.chembl_id IS NOT NULL",
        conn,
    )

    candidate_molregnos: set[int] = (
        set(pref["molregno"].astype(int).tolist())
        | set(syn["molregno"].astype(int).tolist())
        | set(who["molregno"].astype(int).tolist())
    )

    # Build candidate-scoped parent_molregno map.
    parent_of_molregno: dict[int, Optional[int]] = {}
    if candidate_molregnos:
        # Pull only hierarchy rows whose molregno is in our candidate
        # set.  This is much cheaper than scanning the full table
        # (millions of rows in ChEMBL 35).  Chunked via
        # ``_query_in_chunks`` to avoid SQLite's per-statement
        # ``SQLITE_MAX_VARIABLE_NUMBER`` limit (32766 on 3.32+, 999 on
        # older builds), which the candidate set blows past on real
        # ChEMBL 35 data.
        hier = _query_in_chunks(
            conn,
            "SELECT molregno, parent_molregno FROM molecule_hierarchy "
            "WHERE molregno IN ({placeholders})",
            list(candidate_molregnos),
        )
        if not hier.empty:
            for m, p in zip(
                hier["molregno"].to_numpy(),
                hier["parent_molregno"].to_numpy(),
            ):
                if pd.isna(m):
                    continue
                parent_of_molregno[int(m)] = None if pd.isna(p) else int(p)

    # We need molregno -> chembl_id for every parent_molregno that
    # appears as a parent of any candidate (plus the candidates
    # themselves).  Gather chembl_ids for parent molregnos that are
    # not already known.
    molregno_to_cid: dict[int, str] = {}
    for df in (pref, syn, who):
        for m, c in zip(df["molregno"].to_numpy(), df["chembl_id"].to_numpy()):
            if pd.isna(m) or c is None:
                continue
            molregno_to_cid.setdefault(int(m), str(c))

    parents_needed = {
        p for p in parent_of_molregno.values() if p is not None
    }
    missing_parents = parents_needed - set(molregno_to_cid)
    if missing_parents:
        # Chunked for the same SQL-variable-limit reason as the
        # hierarchy query above.
        extra = _query_in_chunks(
            conn,
            "SELECT molregno, chembl_id FROM molecule_dictionary "
            "WHERE molregno IN ({placeholders})",
            list(missing_parents),
        )
        if not extra.empty:
            for m, c in zip(
                extra["molregno"].to_numpy(),
                extra["chembl_id"].to_numpy(),
            ):
                if pd.isna(m) or c is None:
                    continue
                molregno_to_cid.setdefault(int(m), str(c))

    parent_of_cid: dict[str, Optional[int]] = {}
    for m, c in molregno_to_cid.items():
        # parent_of_cid maps cid -> parent_molregno (could be self,
        # could be different molregno, could be None).  Default to
        # self when no hierarchy row exists (matches ChEMBL's implicit
        # "parent is itself" assumption for terminal molecules).
        p = parent_of_molregno.get(m, m)
        parent_of_cid[c] = p

    # Extend parent_of_cid + molregno_to_cid for any CIDs the
    # caller passes in extra_cids that are not already covered by the
    # candidate set.  Without this extension, parent/salt
    # canonicalization in _canonicalize_to_parent_chembl_id would
    # silently miss ~70% of the redundancy contributors in real
    # chembl-source loads.
    n_extra_requested = 0
    n_extra_resolved = 0
    if extra_cids is not None:
        extras_to_resolve = {
            str(c) for c in extra_cids if c
        } - set(parent_of_cid)
        n_extra_requested = len(extras_to_resolve)
        if extras_to_resolve:
            extra_md = _query_in_chunks(
                conn,
                "SELECT molregno, chembl_id FROM molecule_dictionary "
                "WHERE chembl_id IN ({placeholders})",
                list(extras_to_resolve),
            )
            extra_cid_to_mol: dict[str, int] = {}
            if not extra_md.empty:
                for m, c in zip(
                    extra_md["molregno"].to_numpy(),
                    extra_md["chembl_id"].to_numpy(),
                ):
                    if pd.isna(m) or c is None:
                        continue
                    extra_cid_to_mol[str(c)] = int(m)
                    molregno_to_cid.setdefault(int(m), str(c))

            extra_mols = list(extra_cid_to_mol.values())
            if extra_mols:
                extra_hier = _query_in_chunks(
                    conn,
                    "SELECT molregno, parent_molregno FROM molecule_hierarchy "
                    "WHERE molregno IN ({placeholders})",
                    extra_mols,
                )
                if not extra_hier.empty:
                    for m, p in zip(
                        extra_hier["molregno"].to_numpy(),
                        extra_hier["parent_molregno"].to_numpy(),
                    ):
                        if pd.isna(m):
                            continue
                        parent_of_molregno[int(m)] = (
                            None if pd.isna(p) else int(p)
                        )

                # Resolve any new parent molregnos to chembl_ids.
                new_parents_needed = {
                    p for p in parent_of_molregno.values()
                    if p is not None
                } - set(molregno_to_cid)
                if new_parents_needed:
                    extra_parent_md = _query_in_chunks(
                        conn,
                        "SELECT molregno, chembl_id FROM molecule_dictionary "
                        "WHERE molregno IN ({placeholders})",
                        list(new_parents_needed),
                    )
                    if not extra_parent_md.empty:
                        for m, c in zip(
                            extra_parent_md["molregno"].to_numpy(),
                            extra_parent_md["chembl_id"].to_numpy(),
                        ):
                            if pd.isna(m) or c is None:
                                continue
                            molregno_to_cid.setdefault(int(m), str(c))

            # Add extras to parent_of_cid using the same self-parent
            # fallback as the candidate-set semantics above.
            for cid, mol in extra_cid_to_mol.items():
                p = parent_of_molregno.get(mol, mol)
                parent_of_cid[cid] = p
            n_extra_resolved = len(extra_cid_to_mol)

    # Derive a direct cid -> parent_cid map for fast vectorized
    # lookup in the canonicalization helper.  Self-parent fallback is
    # applied wherever the parent molregno is None or has no
    # chembl_id resolution; this matches the semantic that a
    # molecule with no hierarchy edge is its own parent.
    parent_chembl_id_of_cid: dict[str, str] = {}
    for cid, p_mol in parent_of_cid.items():
        if p_mol is None:
            parent_chembl_id_of_cid[cid] = cid
            continue
        p_cid = molregno_to_cid.get(int(p_mol))
        parent_chembl_id_of_cid[cid] = p_cid if p_cid else cid

    def _build_tier(
        df: pd.DataFrame, name_col: str
    ) -> tuple[dict[str, set[str]], dict[str, str]]:
        candidates: dict[str, set[str]] = defaultdict(set)
        for cid, n in zip(df["chembl_id"].to_numpy(), df[name_col].to_numpy()):
            if cid is None or n is None:
                continue
            try:
                if isinstance(n, float) and pd.isna(n):
                    continue
            except (TypeError, ValueError):
                pass
            k = _normalize_drug_name(n)
            if not k:
                continue
            candidates[k].add(str(cid))
        resolved: dict[str, str] = {}
        for k, cs in candidates.items():
            r = _collapse_to_parent(cs, parent_of_cid, molregno_to_cid)
            if r is not None:
                resolved[k] = r
        return dict(candidates), resolved

    cand_pref, res_pref = _build_tier(pref, "pref_name")
    cand_syn, res_syn = _build_tier(syn, "synonyms")
    cand_who, res_who = _build_tier(who, "who_name")

    # Reuse the Item-1 helper to populate the drug-level annotation
    # dicts, so the index ships with everything
    # ``_propagate_chembl_atc_to_remapped`` needs.
    chembl_to_atc, chembl_to_ind = _build_chembl_annotation_dicts(conn)

    logger.info(
        "ChEMBL synonym index built: pref=%d (resolved=%d), syn=%d (resolved=%d), "
        "who=%d (resolved=%d); parent_of_cid=%d entries "
        "(extra_cids requested=%d, resolved=%d); "
        "parent_chembl_id_of_cid=%d entries; chembl_to_atc=%d",
        len(cand_pref), len(res_pref),
        len(cand_syn), len(res_syn),
        len(cand_who), len(res_who),
        len(parent_of_cid),
        n_extra_requested,
        n_extra_resolved,
        len(parent_chembl_id_of_cid),
        len(chembl_to_atc),
    )

    return ChemblNameIndex(
        tier_resolved={
            _TIER_PREF: res_pref,
            _TIER_SYN: res_syn,
            _TIER_WHO: res_who,
        },
        tier_candidates={
            _TIER_PREF: cand_pref,
            _TIER_SYN: cand_syn,
            _TIER_WHO: cand_who,
        },
        parent_of_cid=parent_of_cid,
        parent_chembl_id_of_cid=parent_chembl_id_of_cid,
        chembl_to_atc=chembl_to_atc,
        chembl_to_ind=chembl_to_ind,
    )


def load_chembl_synonym_index(
    sqlite_path: Path,
    extra_cids: Optional[Iterable[str]] = None,
) -> ChemblNameIndex:
    """Public wrapper around :func:`_build_chembl_synonym_index`.

    Opens its own SQLite connection (the build is read-only) so it
    can be called independently of :func:`load_chembl` without
    affecting that function's contract.

    Parameters
    ----------
    sqlite_path :
        Path to the ChEMBL SQLite database.
    extra_cids :
        Optional iterable of additional ``chembl_id`` strings whose
        parent mapping should be populated (in addition to the
        candidate-scoped molecules).  Pass the loaded chembl
        ``drug_chembl_id`` set here for full parent/salt coverage; see
        :func:`_build_chembl_synonym_index` for details.
        ``None`` (default) preserves previous behavior exactly.
    """
    sqlite_path = check_file_exists(sqlite_path, label="ChEMBL SQLite")
    conn = sqlite3.connect(str(sqlite_path))
    try:
        return _build_chembl_synonym_index(conn, extra_cids=extra_cids)
    finally:
        conn.close()


def _resolve_synonym_index(
    name: object,
    index: ChemblNameIndex,
) -> tuple[Optional[str], str]:
    """Resolve a free-text drug name to a ``chembl_id`` via the index.

    Implements the following decision tree:

    1. Empty / non-stringifiable input -> ``(None, "empty")``.
    2. Tier-1 unique resolution -> ``(cid, "pref")``.
    3. If tier-1 has candidates but did not collapse, build
       ``T1_parents`` (the set of candidate parent_molregnos):

       * If ``T1_parents`` is empty (candidates exist but none have a
         usable parent) -> ``(None, "ambig-no-parents")``; the resolver
         does not fall through to lower tiers.
       * Tier-2 resolved + parent-consistent -> ``(cid, "syn-rescue")``.
       * Tier-2 resolved + parent-inconsistent -> fall through to tier-3
         (a tier-3 hit must still be parent-consistent).
       * Tier-2 ambiguous (candidates exist, no resolved hit) -> block
         tier-3 fallthrough; ``(None, "pref-ambig-blocked")``.
         Symmetric with the tier-1-miss branch's tier-2-ambig rule.
       * Tier-3 resolved + parent-consistent -> ``(cid, "who-rescue")``.
       * Otherwise -> ``(None, "pref-ambig-blocked")``.

    4. If ``k`` was a true tier-1 miss (key not in ``cand_pref``):
       tier 2 unique -> ``(cid, "syn")``; tier-2 ambiguous ->
       ``(None, "syn-ambig-blocked")``; tier 3 unique -> ``(cid, "who")``;
       otherwise ``(None, "miss")``.
    """
    k = _normalize_drug_name(name)
    if not k:
        return None, "empty"

    res_pref = index.tier_resolved[_TIER_PREF]
    res_syn = index.tier_resolved[_TIER_SYN]
    res_who = index.tier_resolved[_TIER_WHO]
    cand_pref = index.tier_candidates[_TIER_PREF]
    cand_syn = index.tier_candidates[_TIER_SYN]
    cand_who = index.tier_candidates[_TIER_WHO]

    if k in res_pref:
        return res_pref[k], "pref"

    # Tier-1 candidates exist but did not collapse: classify the
    # ambiguity.
    if k in cand_pref:
        t1_parents: set[int] = set()
        for c in cand_pref[k]:
            p = index.parent_of_cid.get(c)
            if p is not None:
                t1_parents.add(int(p))
        if not t1_parents:
            # Ambiguous with no usable parents - explicit reject.
            return None, "ambig-no-parents"
        # Tier-2 evaluation:
        #   * resolved hit -> try parent-consistent rescue; if
        #     parent-inconsistent, fall through to tier-3 (lenient,
        #     documented).
        #   * ambiguous candidates (in cand_syn but not res_syn) ->
        #     block tier-3 fallthrough.  Symmetric with the
        #     tier-1-miss branch's "syn-ambig-blocked" rule: if the
        #     synonym tier itself registered an ambiguity on this
        #     name, a tier-3 hit cannot resolve it.
        if k in res_syn:
            cid = res_syn[k]
            psyn = index.parent_of_cid.get(cid)
            if psyn is not None and int(psyn) in t1_parents:
                return cid, "syn-rescue"
            # Parent-inconsistent: fall through to tier-3 below.
        elif k in cand_syn:
            return None, "pref-ambig-blocked"
        # Try tier-3 rescue (parent-consistent).  Reached only when
        # tier-2 truly missed OR tier-2 resolved-but-parent-inconsistent.
        if k in res_who:
            cid = res_who[k]
            pwho = index.parent_of_cid.get(cid)
            if pwho is not None and int(pwho) in t1_parents:
                return cid, "who-rescue"
        return None, "pref-ambig-blocked"

    # True tier-1 miss -> lower tiers may resolve freely.
    if k in res_syn:
        return res_syn[k], "syn"
    if k in cand_syn:
        # Tier-2 ambiguous; do not fall through to tier 3.
        return None, "syn-ambig-blocked"
    if k in res_who:
        return res_who[k], "who"
    return None, "miss"


def load_chembl(
    sqlite_path: Path,
    gene_id_converter: Optional[GeneIDConverter] = None,
    min_pchembl: Optional[float] = None,
    max_phase_filter: Optional[int] = None,
    target_type_filter: list[str] | None = None,
    chembl_scope: str = "mechanism_only",
) -> pd.DataFrame:
    """Query ChEMBL SQLite for drug-target interactions.

    Executes two queries (mechanisms + affinities), joins them, maps
    UniProt accessions to all four gene IDs via *gene_id_converter*.

    Args:
        sqlite_path: Path to ChEMBL SQLite database file.
        gene_id_converter: For UniProt -> Ensembl/Symbol/Entrez mapping.
        min_pchembl: Potency cutoff (e.g. 5.0 = 10 µM).
        max_phase_filter: Minimum clinical phase (0-4).
        target_type_filter: Target type whitelist.
        chembl_scope: ``"mechanism_only"`` (default) or
            ``"mechanism_or_affinity"``.

    Returns:
        DataFrame with ChEMBL drug-target records.
    """
    if target_type_filter is None:
        target_type_filter = ["SINGLE PROTEIN"]

    placeholders = ", ".join(f"'{t}'" for t in target_type_filter)

    logger.info("Querying ChEMBL SQLite: %s", sqlite_path)
    conn = sqlite3.connect(str(sqlite_path))

    try:
        q1 = _CHEMBL_QUERY_MECHANISMS.format(target_types=placeholders)
        logger.info("Running ChEMBL mechanism query...")
        df_mech = pd.read_sql_query(q1, conn)
        logger.info("Mechanism query: %d rows", len(df_mech))

        q2 = _CHEMBL_QUERY_AFFINITIES.format(target_types=placeholders)
        logger.info("Running ChEMBL affinity query...")
        df_aff = pd.read_sql_query(q2, conn)
        logger.info("Affinity query: %d rows", len(df_aff))

        logger.info("Running parent ATC/indication queries...")
        df_parent_atc = pd.read_sql_query(_CHEMBL_QUERY_PARENT_ATC, conn)
        df_parent_ind = pd.read_sql_query(_CHEMBL_QUERY_PARENT_INDICATIONS, conn)
        logger.info(
            "Parent lookups: %d ATC, %d indication rows",
            len(df_parent_atc), len(df_parent_ind),
        )

        chembl_to_atc, chembl_to_ind = _build_chembl_annotation_dicts(conn)
    finally:
        conn.close()

    df_aff_best = _best_affinity_per_pair(df_aff)

    df = _join_mechanism_affinity(df_mech, df_aff_best)

    if "_molregno" in df.columns:
        if not df_parent_atc.empty:
            df = df.merge(df_parent_atc, left_on="_molregno", right_on="molregno", how="left")
            df.drop(columns=["molregno"], inplace=True, errors="ignore")
        if not df_parent_ind.empty:
            df = df.merge(df_parent_ind, left_on="_molregno", right_on="molregno", how="left")
            df.drop(columns=["molregno"], inplace=True, errors="ignore")

    df = _postprocess_chembl(
        df,
        gene_id_converter=gene_id_converter,
        min_pchembl=min_pchembl,
        max_phase_filter=max_phase_filter,
        chembl_scope=chembl_scope,
        chembl_to_atc=chembl_to_atc,
        chembl_to_ind=chembl_to_ind,
    )

    logger.info("ChEMBL loading complete: %d records", len(df))
    return df


def _best_affinity_per_pair(df_aff: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-potency affinity measurement per drug-target pair.

    The affinity query is pre-sorted by pchembl_value DESC, so the first
    row per group is the best.
    """
    if df_aff.empty:
        return df_aff
    return (
        df_aff
        .sort_values("pchembl_value", ascending=False)
        .drop_duplicates(subset=["chembl_id", "uniprot_id"], keep="first")
        .rename(columns={"chembl_id": "drug_chembl_id"})
    )


def _join_mechanism_affinity(
    df_mech: pd.DataFrame,
    df_aff: pd.DataFrame,
) -> pd.DataFrame:
    """Full outer join of mechanism and affinity data on (drug, target).

    Tags mechanism-provenance rows with ``_from_mechanism=True`` before the
    join so that downstream filtering can distinguish mechanism-only records
    from affinity-only records without relying on field-content heuristics.
    """
    if df_mech.empty and df_aff.empty:
        return pd.DataFrame()

    if df_mech.empty:
        return df_aff.assign(
            mechanism_of_action=pd.NA,
            interaction_type=pd.NA,
            _from_mechanism=False,
        )

    df_mech = df_mech.copy()
    df_mech["_from_mechanism"] = True

    if df_aff.empty:
        return df_mech.assign(
            affinity_type=pd.NA,
            affinity_value=pd.NA,
            affinity_unit=pd.NA,
            pchembl_value=pd.NA,
        )

    aff_cols = ["drug_chembl_id", "uniprot_id", "drug_name", "max_phase",
                "affinity_type", "affinity_value", "affinity_unit",
                "pchembl_value"]
    aff_subset = df_aff[
        [c for c in aff_cols if c in df_aff.columns]
    ].copy()

    merged = df_mech.merge(
        aff_subset,
        on=["drug_chembl_id", "uniprot_id"],
        how="outer",
        suffixes=("", "_aff"),
    )

    for col in ("drug_name", "max_phase"):
        aff_col = f"{col}_aff"
        if aff_col in merged.columns:
            merged[col] = merged[col].fillna(merged[aff_col])
            merged = merged.drop(columns=[aff_col])

    return merged


def _postprocess_chembl(
    df: pd.DataFrame,
    gene_id_converter: Optional[GeneIDConverter] = None,
    min_pchembl: Optional[float] = None,
    max_phase_filter: Optional[int] = None,
    chembl_scope: str = "mechanism_only",
    chembl_to_atc: Optional[dict[str, list[str]]] = None,
    chembl_to_ind: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Clean, filter, and enrich ChEMBL records.

    Args:
        chembl_scope: ``"mechanism_only"`` keeps only mechanism-provenance
            rows (default, backward-compatible).  ``"mechanism_or_affinity"``
            also retains affinity-only compounds.
        chembl_to_atc: Optional drug-level (``chembl_id`` keyed) ATC dict
            built by :func:`_build_chembl_annotation_dicts`.  When provided,
            takes precedence over the legacy ``atc_codes_raw`` /
            ``parent_atc_raw`` columns; this is required to reach
            affinity-only rows that lose ``_molregno`` during the outer
            mechanism+affinity merge.
        chembl_to_ind: Optional drug-level indications dict, same semantics.
    """
    if df.empty:
        return df

    if "_from_mechanism" in df.columns:
        df["_from_mechanism"] = df["_from_mechanism"].fillna(False)
    else:
        df["_from_mechanism"] = True

    if chembl_scope == "mechanism_only":
        before = len(df)
        df = df[df["_from_mechanism"]].copy()
        logger.info(
            "chembl_scope=mechanism_only: kept %d/%d mechanism-provenance rows",
            len(df), before,
        )
    else:
        n_mech = int(df["_from_mechanism"].sum())
        n_aff_only = len(df) - n_mech
        logger.info(
            "chembl_scope=mechanism_or_affinity: %d mechanism + %d affinity-only rows",
            n_mech, n_aff_only,
        )

    if "max_phase" in df.columns:
        df["max_phase"] = pd.to_numeric(df["max_phase"], errors="coerce").fillna(0).astype(int)
    else:
        df["max_phase"] = 0

    if max_phase_filter is not None:
        before = len(df)
        df = df[df["max_phase"] >= max_phase_filter].copy()
        logger.info("Phase filter (>=%d): %d -> %d", max_phase_filter, before, len(df))

    if min_pchembl is not None and "pchembl_value" in df.columns:
        has_pchembl = df["pchembl_value"].notna()
        before = len(df)
        df = df[~has_pchembl | (df["pchembl_value"] >= min_pchembl)].copy()
        logger.info("pChEMBL filter (>=%.1f): %d -> %d", min_pchembl, before, len(df))

    df["interaction_type"] = df.get("interaction_type", pd.Series(dtype=str)).apply(
        _standardize_interaction_type
    )

    if chembl_to_atc is not None and "drug_chembl_id" in df.columns:
        df["atc_codes"] = df["drug_chembl_id"].map(
            lambda c: list(chembl_to_atc.get(c, [])) or None
        )
        atc_source = "drug-level dict"
    else:
        direct_atc = df["atc_codes_raw"].apply(_parse_pipe_list) if "atc_codes_raw" in df.columns else pd.Series([None] * len(df), index=df.index)
        parent_atc = df["parent_atc_raw"].apply(_parse_pipe_list) if "parent_atc_raw" in df.columns else pd.Series([None] * len(df), index=df.index)
        df["atc_codes"] = [
            sorted(set((d or []) + (p or []))) or None
            for d, p in zip(direct_atc, parent_atc)
        ]
        atc_source = "raw-column fallback"

    if chembl_to_ind is not None and "drug_chembl_id" in df.columns:
        df["indication_mesh"] = df["drug_chembl_id"].map(
            lambda c: list(chembl_to_ind.get(c, [])) or None
        )
    else:
        direct_ind = df["indications_raw"].apply(_parse_pipe_list) if "indications_raw" in df.columns else pd.Series([None] * len(df), index=df.index)
        parent_ind = df["parent_indications_raw"].apply(_parse_pipe_list) if "parent_indications_raw" in df.columns else pd.Series([None] * len(df), index=df.index)
        df["indication_mesh"] = [
            sorted(set((d or []) + (p or []))) or None
            for d, p in zip(direct_ind, parent_ind)
        ]

    n_drugs = df["drug_chembl_id"].nunique() if "drug_chembl_id" in df.columns else 0
    n_with_atc = df.loc[df["atc_codes"].apply(lambda x: isinstance(x, list) and len(x) > 0), "drug_chembl_id"].nunique() if n_drugs else 0
    n_with_ind = df.loc[df["indication_mesh"].apply(lambda x: isinstance(x, list) and len(x) > 0), "drug_chembl_id"].nunique() if n_drugs else 0
    logger.info(
        "ChEMBL annotation coverage (%s): %d/%d drugs with ATC, %d/%d with indications",
        atc_source, n_with_atc, n_drugs, n_with_ind, n_drugs,
    )

    df["is_withdrawn"] = df.get("withdrawal_reason", pd.Series(dtype=str)).notna()

    df["source"] = "chembl"

    if gene_id_converter is not None and "uniprot_id" in df.columns:
        df = _annotate_genes_from_uniprot(df, gene_id_converter)
    else:
        for col in ("gene_symbol", "gene_ensembl_id", "gene_uniprot_id", "gene_entrez_id"):
            if col not in df.columns:
                df[col] = pd.NA
        if "uniprot_id" in df.columns and "gene_uniprot_id" in df.columns:
            df["gene_uniprot_id"] = df["gene_uniprot_id"].fillna(df["uniprot_id"])

    has_mech = df["mechanism_of_action"].notna() if "mechanism_of_action" in df.columns else pd.Series(False, index=df.index)
    has_aff = df["pchembl_value"].notna() if "pchembl_value" in df.columns else pd.Series(False, index=df.index)
    df["confidence"] = np.select(
        [has_mech & has_aff, has_mech | has_aff],
        ["high", "medium"],
        default="low",
    )

    drop_cols = [
        "atc_codes_raw", "indications_raw", "withdrawal_reason",
        "parent_atc_raw", "parent_indications_raw",
        "target_name", "target_type", "uniprot_id", "_molregno",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "drug_name" in df.columns:
        df["drug_name"] = df["drug_name"].fillna("").str.strip()
        if chembl_scope == "mechanism_or_affinity" and "_from_mechanism" in df.columns:
            aff_only_blank = ~df["_from_mechanism"] & (df["drug_name"] == "")
            df.loc[aff_only_blank, "drug_name"] = df.loc[aff_only_blank, "drug_chembl_id"]
        df = df[df["drug_name"] != ""].copy()

    df = df.drop(columns=["_from_mechanism"], errors="ignore")

    if "drug_chembl_id" in df.columns:
        df = df[df["drug_chembl_id"].notna()].copy()

    return df.reset_index(drop=True)


def _annotate_genes_from_uniprot(
    df: pd.DataFrame,
    converter: GeneIDConverter,
) -> pd.DataFrame:
    """Map UniProt accessions to all four gene IDs."""
    df = df.copy()
    unique_uniprots = df["uniprot_id"].dropna().unique().tolist()
    cache: dict[str, Optional[dict]] = {}
    for uid in unique_uniprots:
        cache[uid] = converter.get_full_record(uid, "uniprot")

    symbols, ensembls, uniprots, entrezs = [], [], [], []
    for uid in df["uniprot_id"]:
        rec = cache.get(uid) if pd.notna(uid) else None
        if rec is None:
            symbols.append(None)
            ensembls.append(None)
            uniprots.append(uid if pd.notna(uid) else None)
            entrezs.append(None)
        else:
            symbols.append(rec.get("symbol"))
            ensembls.append(rec.get("ensembl"))
            uniprots.append(uid)
            entrezs.append(rec.get("entrez"))

    df["gene_symbol"] = symbols
    df["gene_ensembl_id"] = ensembls
    df["gene_uniprot_id"] = uniprots
    df["gene_entrez_id"] = pd.array(entrezs, dtype=pd.Int64Dtype())

    n_resolved = sum(1 for s in symbols if s is not None)
    n_total = df["uniprot_id"].notna().sum()
    logger.info(
        "ChEMBL gene ID mapping: resolved %d / %d UniProt IDs (%.1f%%)",
        n_resolved, n_total,
        100 * n_resolved / max(n_total, 1),
    )

    before = len(df)
    df = df[df["gene_symbol"].notna()].copy()
    if len(df) < before:
        logger.info(
            "Dropped %d records with unmappable UniProt IDs", before - len(df)
        )

    return df


# ---------------------------------------------------------------------------
# PDSP source (optional)
# ---------------------------------------------------------------------------


def load_pdsp(
    pdsp_csv: Path,
    gene_id_converter: Optional[GeneIDConverter] = None,
) -> pd.DataFrame:
    """Load PDSP Ki database CSV export.

    Maps pharmacological target names to HGNC gene symbols using
    :data:`PDSP_TARGET_MAP`.  Converts Ki to pChEMBL-equivalent:
    ``pchembl = 9 - log10(Ki_nM)``.

    Args:
        pdsp_csv: Path to PDSP CSV file.
        gene_id_converter: For gene ID enrichment.

    Returns:
        DataFrame with PDSP drug-target records.
    """
    logger.info("Loading PDSP Ki database: %s", pdsp_csv)

    df = pd.read_csv(pdsp_csv, low_memory=False)
    logger.info("PDSP raw: %d rows, columns: %s", len(df), list(df.columns))

    col_map = _detect_pdsp_columns(df)
    if col_map is None:
        logger.warning("Could not detect PDSP column layout; returning empty")
        return pd.DataFrame()

    drug_col = col_map["drug_name"]
    target_col = col_map["target"]
    ki_col = col_map.get("ki")
    species_col = col_map.get("species")

    if species_col and species_col in df.columns:
        before = len(df)
        df = df[
            df[species_col].str.lower().str.contains("human", na=False)
        ].copy()
        logger.info("PDSP species filter (human): %d -> %d", before, len(df))

    df["gene_symbol"] = df[target_col].map(PDSP_TARGET_MAP)
    before = len(df)
    df = df[df["gene_symbol"].notna()].copy()
    logger.info(
        "PDSP target mapping: %d -> %d (%.0f%% mapped)",
        before, len(df), 100 * len(df) / max(before, 1),
    )

    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=[drug_col]).copy()
    df["_drug_name_clean"] = df[drug_col].astype(str).str.strip()
    df = df[df["_drug_name_clean"] != ""].copy()

    ki_values = pd.to_numeric(df[ki_col], errors="coerce") if ki_col else pd.Series(np.nan, index=df.index)
    df["pchembl_value"] = np.where(ki_values > 0, 9.0 - np.log10(ki_values), np.nan)
    df["affinity_value"] = np.where(ki_values > 0, ki_values, np.nan)

    result = pd.DataFrame({
        "drug_name": df["_drug_name_clean"].values,
        "drug_chembl_id": _sanitize_placeholder_id("PDSP_", df["_drug_name_clean"]),
        "gene_symbol": df["gene_symbol"].values,
        "interaction_type": "other",
        "pchembl_value": df["pchembl_value"].values,
        "affinity_value": df["affinity_value"].values,
        "affinity_type": "Ki",
        "affinity_unit": "nM",
        "max_phase": 0,
        "source": "pdsp",
        "confidence": np.where(df["pchembl_value"].notna(), "high", "medium"),
    })

    if gene_id_converter is not None and not result.empty:
        result = gene_id_converter.batch_annotate(
            result, id_column="gene_symbol", id_type="symbol"
        )

    logger.info("PDSP loading complete: %d records", len(result))
    return result


def _detect_pdsp_columns(df: pd.DataFrame) -> Optional[dict[str, str]]:
    """Heuristically detect PDSP column names."""
    cols_lower = {c.lower().strip(): c for c in df.columns}

    drug_candidates = ["drug name", "drug_name", "ligand name",
                       "ligand_name", "compound", "drug"]
    target_candidates = ["target", "receptor", "target name",
                         "target_name", "receptor name"]
    ki_candidates = ["ki (nm)", "ki_nm", "ki (nanomolar)", "ki",
                     "ki_nM", "ki_value"]
    species_candidates = ["species", "organism", "test species"]

    drug_col = _first_match(cols_lower, drug_candidates)
    target_col = _first_match(cols_lower, target_candidates)

    if drug_col is None or target_col is None:
        return None

    return {
        "drug_name": drug_col,
        "target": target_col,
        "ki": _first_match(cols_lower, ki_candidates),
        "species": _first_match(cols_lower, species_candidates),
    }


def _first_match(
    cols_lower: dict[str, str],
    candidates: list[str],
) -> Optional[str]:
    """Return the original column name for the first matching candidate."""
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


# ---------------------------------------------------------------------------
# DGIdb source (optional)
# ---------------------------------------------------------------------------


def load_dgidb(
    interactions_tsv: Path,
    gene_id_converter: Optional[GeneIDConverter] = None,
) -> pd.DataFrame:
    """Load DGIdb bulk interaction download.

    DGIdb lacks chemical identifiers (no InChIKey/PubChem CID) and
    binding affinity data.  Used for interaction confirmation and source
    diversity.

    Args:
        interactions_tsv: Path to DGIdb interactions TSV file.
        gene_id_converter: For gene ID enrichment.

    Returns:
        DataFrame with DGIdb drug-target records.
    """
    logger.info("Loading DGIdb interactions: %s", interactions_tsv)

    df = pd.read_csv(interactions_tsv, sep="\t", low_memory=False, comment="#")
    logger.info("DGIdb raw: %d rows, columns: %s", len(df), list(df.columns))

    col_map = _detect_dgidb_columns(df)
    if col_map is None:
        logger.warning("Could not detect DGIdb column layout; returning empty")
        return pd.DataFrame()

    drug_col = col_map["drug_name"]
    drug_claim_col = col_map.get("drug_claim_name")
    gene_col = col_map["gene_name"]
    itype_col = col_map.get("interaction_type")
    pmid_col = col_map.get("pmids")
    drug_chembl_col = col_map.get("drug_chembl_id")
    source_col = col_map.get("source_db")

    df = df.dropna(subset=[drug_col, gene_col]).copy()
    df["_drug_clean"] = df[drug_col].astype(str).str.strip()
    df["_gene_clean"] = df[gene_col].astype(str).str.strip()
    df = df[(df["_drug_clean"] != "") & (df["_gene_clean"] != "")].copy()

    if df.empty:
        return pd.DataFrame()

    # Carry the drug_claim_name through as a leading-underscore
    # transient column so ``_match_dgidb_to_chembl`` can fall back on
    # it when the canonical drug_name does not resolve.  The
    # underscore prefix signals "internal" and is stripped both at the
    # matcher boundary and (defensively) at ``_finalize_schema``.
    if drug_claim_col and drug_claim_col in df.columns and drug_claim_col != drug_col:
        df["_drug_claim_name"] = df[drug_claim_col].astype(str).str.strip()
    else:
        df["_drug_claim_name"] = ""

    itype_series = df[itype_col].astype(str) if itype_col and itype_col in df.columns else pd.Series("", index=df.index)
    std_itypes = itype_series.apply(_standardize_interaction_type)

    chembl_ids = pd.Series(pd.NA, index=df.index, dtype=object)
    if drug_chembl_col and drug_chembl_col in df.columns:
        raw_cids = df[drug_chembl_col].fillna("")
        chembl_ids = raw_cids.apply(_extract_chembl_id)
        chembl_ids = chembl_ids.where(chembl_ids.notna(), other=pd.NA)
        n_concept = raw_cids.astype(bool).sum()
        n_extracted = chembl_ids.notna().sum()
        logger.info(
            "DGIdb concept-ID parsing: %d non-empty concept IDs, %d extracted ChEMBL IDs, %d fallback to placeholder",
            n_concept, n_extracted, n_concept - n_extracted,
        )

    placeholder_ids = _sanitize_placeholder_id("DGIDB_", df["_drug_clean"])
    chembl_ids = chembl_ids.fillna(placeholder_ids)

    has_pmids = pd.Series(False, index=df.index)
    pmid_lists = pd.Series([None] * len(df), index=df.index, dtype=object)
    if pmid_col and pmid_col in df.columns:
        notna_mask = df[pmid_col].notna()
        raw_pmids = df.loc[notna_mask, pmid_col].astype(str)
        parsed = raw_pmids.apply(lambda x: [p.strip() for p in x.split(",") if p.strip()] or None)
        pmid_lists.loc[notna_mask] = parsed
        has_pmids.loc[notna_mask] = parsed.apply(lambda x: x is not None and len(x) > 0)

    result = pd.DataFrame({
        "drug_name": df["_drug_clean"].values,
        "drug_chembl_id": chembl_ids.values,
        "gene_symbol": df["_gene_clean"].values,
        "interaction_type": std_itypes.values,
        "max_phase": 0,
        "source": "dgidb",
        "confidence": np.where(has_pmids, "medium", "low"),
        "source_pmids": pmid_lists.values,
        "_drug_claim_name": df["_drug_claim_name"].values,
    })

    if gene_id_converter is not None and not result.empty:
        result = gene_id_converter.batch_annotate(
            result, id_column="gene_symbol", id_type="symbol"
        )

    logger.info("DGIdb loading complete: %d records", len(result))
    return result


def _detect_dgidb_columns(df: pd.DataFrame) -> Optional[dict[str, str]]:
    """Heuristically detect DGIdb column names."""
    cols_lower = {c.lower().strip(): c for c in df.columns}

    drug_candidates = ["drug_name", "drug name", "compound_name"]
    drug_claim_candidates = ["drug_claim_name", "drug claim name",
                             "drug_claim_primary_name"]
    gene_candidates = ["gene_name", "gene name", "gene_claim_name",
                       "gene"]
    itype_candidates = ["interaction_types", "interaction_type",
                        "interaction_claim_source"]
    pmid_candidates = ["pmids", "pmid", "pubmed_ids"]
    chembl_candidates = ["drug_chembl_id", "drug_concept_id", "drug concept id", "concept_id"]
    source_candidates = ["source_db_name", "source", "interaction_group_score"]

    drug_col = _first_match(cols_lower, drug_candidates)
    # If the canonical drug-name column is missing, fall back to the
    # claim name as the primary drug column (preserves prior behavior).
    if drug_col is None:
        drug_col = _first_match(cols_lower, drug_claim_candidates)
    gene_col = _first_match(cols_lower, gene_candidates)

    if drug_col is None or gene_col is None:
        return None

    return {
        "drug_name": drug_col,
        "drug_claim_name": _first_match(cols_lower, drug_claim_candidates),
        "gene_name": gene_col,
        "interaction_type": _first_match(cols_lower, itype_candidates),
        "pmids": _first_match(cols_lower, pmid_candidates),
        "drug_chembl_id": _first_match(cols_lower, chembl_candidates),
        "source_db": _first_match(cols_lower, source_candidates),
    }


# ---------------------------------------------------------------------------
# Expression-perturbation sources
# ---------------------------------------------------------------------------
#
# CREEDS and DSigDB encode drug-induced gene expression *changes*, not
# direct target binding.  They are loaded into the same DrugTargetRecord
# schema so the existing dedup / matcher / Branch-A enrichment plumbing
# can consume them without bespoke code paths, but they are kept in a
# strictly disjoint ``expression_sources`` field at the config level
# (see DrugEnrichmentConfig in repogen/config/schema.py) so the
# pipeline never silently mixes target-binding and expression-based
# evidence into the same upstream contract.
#
# Outputs from these loaders satisfy the same contract as load_pdsp /
# load_dgidb: placeholder ``CREEDS_<name>`` / ``DSIGDB_<name>``
# ``drug_chembl_id`` values are emitted, and
# ``_match_creeds_to_chembl`` / ``_match_dsigdb_to_chembl`` upgrade
# them to real CHEMBL IDs via the synonym index (when available).


def load_creeds(
    creeds_json: Path,
    gene_id_converter: Optional[GeneIDConverter] = None,
) -> pd.DataFrame:
    """Load CREEDS single-drug perturbation up/down gene signatures.

    CREEDS (Wang et al. 2016, *Nature Communications*) ships a JSON
    list of curated GEO drug-treatment vs control comparisons.  Each
    entry has ``drug_name``, ``up_genes``, and ``down_genes`` keys;
    the gene lists are ``[symbol, score]`` pairs.

    The loader emits one ``DrugTargetRecord`` row per (drug, gene)
    pair, with ``interaction_type`` set to ``"creeds_up"`` or
    ``"creeds_down"`` so the standardiser at
    :func:`_standardize_interaction_type` maps it to
    ``"expression_up"`` / ``"expression_down"``.  ``confidence`` is
    set to ``"low"`` because individual GEO comparisons carry no
    independent statistical replication.

    Args:
        creeds_json: Path to ``single_drug_perturbations-v1.0.json``.
        gene_id_converter: Optional gene-ID converter for symbol->Entrez
            annotation (mirrors the load_dgidb contract).

    Returns:
        DataFrame with the standardised drug-target record columns.
        Empty DataFrame if the input is missing keys or yields no
        valid (drug, gene) pairs.
    """
    import json

    logger.info("Loading CREEDS drug perturbations: %s", creeds_json)
    with open(creeds_json, encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        logger.warning(
            "CREEDS JSON is not a list (got %s); returning empty",
            type(records).__name__,
        )
        return pd.DataFrame()

    rows: list[dict] = []
    for entry in records:
        if not isinstance(entry, dict):
            continue
        drug_name = (entry.get("drug_name") or "").strip()
        if not drug_name:
            continue

        for direction, key in (("creeds_up", "up_genes"),
                               ("creeds_down", "down_genes")):
            genes = entry.get(key) or []
            for item in genes:
                # Tolerate both [symbol, score] pairs and bare symbols.
                if isinstance(item, (list, tuple)) and item:
                    gene_symbol = str(item[0]).strip()
                else:
                    gene_symbol = str(item).strip()
                if not gene_symbol:
                    continue
                rows.append({
                    "drug_name": drug_name,
                    "gene_symbol": gene_symbol,
                    "interaction_type": direction,
                })

    if not rows:
        logger.warning("CREEDS: no (drug, gene) pairs extracted")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["drug_chembl_id"] = _sanitize_placeholder_id("CREEDS_", df["drug_name"])
    df["max_phase"] = 0
    df["source"] = "creeds"
    df["confidence"] = "low"
    df["source_pmids"] = pd.Series([None] * len(df), dtype=object)
    df["interaction_type"] = df["interaction_type"].apply(
        _standardize_interaction_type
    )

    if gene_id_converter is not None and not df.empty:
        df = gene_id_converter.batch_annotate(
            df, id_column="gene_symbol", id_type="symbol"
        )

    logger.info(
        "CREEDS loading complete: %d records, %d unique drugs, %d unique genes",
        len(df),
        df["drug_name"].nunique(),
        df["gene_symbol"].nunique(),
    )
    return df


def load_dsigdb(
    dsigdb_path: Path,
    gene_id_converter: Optional[GeneIDConverter] = None,
) -> pd.DataFrame:
    """Load DSigDB D3 (drug-perturbation transcriptomic signatures).

    DSigDB (Yoo et al. 2015) D3 ships a GMT-like file: each line is a
    tab-separated record with the gene-set name in the first field, a
    description in the second field, and gene symbols from the third
    field onward.  The set name typically encodes the drug name plus
    perturbation metadata (cell line, concentration, GEO accession);
    we treat the leading segment (``"_"``-delimited) as the drug
    name.

    Output schema and confidence/max_phase semantics mirror
    :func:`load_creeds`.  ``interaction_type`` is set to ``"dsigdb"``
    which the standardiser maps to ``"expression_perturbation"``.

    Args:
        dsigdb_path: Path to the DSigDB D3 file (extracted by the
            ``"extract_dsigdb_d3"`` postprocessor in
            :mod:`repogen.data.resources`).
        gene_id_converter: Optional gene-ID converter for symbol->Entrez
            annotation.

    Returns:
        DataFrame with the standardised drug-target record columns.
    """
    logger.info("Loading DSigDB D3: %s", dsigdb_path)

    rows: list[dict] = []
    with open(dsigdb_path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            set_name = parts[0].strip()
            if not set_name:
                continue
            # The leading segment of the set name is the drug.  Some
            # DSigDB releases use spaces instead of underscores; we
            # split on the first whitespace or underscore so the rule
            # holds in both cases.
            drug_token = set_name
            for sep in ("_", " "):
                if sep in drug_token:
                    drug_token = drug_token.split(sep, 1)[0].strip()
                    break
            if not drug_token:
                drug_token = set_name
            for sym in parts[2:]:
                gene_symbol = sym.strip()
                if not gene_symbol:
                    continue
                rows.append({
                    "drug_name": drug_token,
                    "gene_symbol": gene_symbol,
                })

    if not rows:
        logger.warning("DSigDB D3: no (drug, gene) pairs extracted")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["drug_chembl_id"] = _sanitize_placeholder_id("DSIGDB_", df["drug_name"])
    df["interaction_type"] = _standardize_interaction_type("dsigdb")
    df["max_phase"] = 0
    df["source"] = "dsigdb"
    df["confidence"] = "low"
    df["source_pmids"] = pd.Series([None] * len(df), dtype=object)

    if gene_id_converter is not None and not df.empty:
        df = gene_id_converter.batch_annotate(
            df, id_column="gene_symbol", id_type="symbol"
        )

    logger.info(
        "DSigDB D3 loading complete: %d records, %d unique drugs, %d unique genes",
        len(df),
        df["drug_name"].nunique(),
        df["gene_symbol"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Multi-source integration
# ---------------------------------------------------------------------------


def _canonicalize_to_parent_chembl_id(
    df: pd.DataFrame,
    index: ChemblNameIndex,
) -> pd.DataFrame:
    """Rewrite child / salt ``drug_chembl_id`` values to their parent CID.

    Parent/salt unification: collapse different ChEMBL IDs
    that belong to the same pharmacological entity (parent compound +
    its salt forms) onto a single canonical ``drug_chembl_id`` so the
    downstream :func:`_deduplicate_records` step merges their target
    rows and ATC enrichment treats each entity as a single
    observation.

    The rewrite is a vectorized lookup against
    ``index.parent_chembl_id_of_cid``:

    * CIDs whose parent ≠ self get rewritten (typical salt -> parent).
    * CIDs whose parent == self are no-ops.
    * CIDs not in the index (e.g. external placeholders, typos) are
      left as-is via ``.fillna(...)``.

    Only ``drug_chembl_id`` is rewritten.  ``drug_name`` and other
    metadata are left untouched; the dedup step will pick whichever
    surviving row is richer. See for the rationale of keeping
    the MVP minimal.

    Emits coverage observability metrics at INFO so log consumers can
    detect regressions:

    * ``unique_cids_seen``: number of distinct ``drug_chembl_id``
      values in ``df``.
    * ``parent_map_hits``: rows whose CID was found in the parent
      map.
    * ``rewritten``: rows where the new CID differs from the input
      (i.e. an actual collapse).
    * ``unmapped_kept_self``: rows whose CID was absent from the
      map (kept as-is).
    """
    if df.empty or "drug_chembl_id" not in df.columns:
        return df

    pmap = index.parent_chembl_id_of_cid
    if not pmap:
        return df

    cids = df["drug_chembl_id"].astype(object).astype(str)
    n_seen = int(cids.nunique())

    new_cids = cids.map(pmap)
    n_parent_map_hits = int(new_cids.notna().sum())
    n_unmapped_kept_self = int(new_cids.isna().sum())

    new_cids = new_cids.fillna(cids)
    n_rewritten = int((new_cids != cids).sum())

    logger.info(
        "Parent/salt canonicalization: rows=%d, unique_cids_seen=%d, "
        "parent_map_hits=%d (%.1f%% of rows), rewritten=%d, "
        "unmapped_kept_self=%d",
        len(df),
        n_seen,
        n_parent_map_hits,
        100 * n_parent_map_hits / max(len(df), 1),
        n_rewritten,
        n_unmapped_kept_self,
    )

    out = df.copy()
    out["drug_chembl_id"] = new_cids.values
    return out


def merge_sources(
    dataframes: dict[str, pd.DataFrame],
    gene_id_converter: Optional[GeneIDConverter] = None,
    chembl_name_index: Optional[ChemblNameIndex] = None,
    parent_salt_unification: bool = True,
) -> pd.DataFrame:
    """Merge drug-target records from multiple sources.

    ChEMBL is the primary source.  PDSP records are matched to ChEMBL
    drugs by name (PDSP lacks InChIKey).  DGIdb records are matched by
    gene symbol + drug name; when ``chembl_name_index`` is provided
    they are additionally remapped via the tiered synonym index
    and any remaining real-``CHEMBL`` rows have their
    ``atc_codes`` / ``indication_mesh`` populated from the
    drug-level ChEMBL annotation dicts.  For the same drug-gene pair
    from multiple sources, the record with the richest data is kept
    and source fields are merged.

    Args:
        dataframes: Mapping of source name to DataFrame.
        gene_id_converter: For filling missing gene IDs.
        chembl_name_index: Optional :class:`ChemblNameIndex`.  When
            provided, DGIdb placeholder IDs are resolved through it,
            ATC propagation is applied post-dedup, and (with
            ``parent_salt_unification=True``) parent/salt CIDs are
            canonicalized before dedup.  No-op when ``None`` or when
            DGIdb is not in ``dataframes``.
        parent_salt_unification: When ``True`` (default) and
            ``chembl_name_index`` is provided, child / salt CIDs are
            rewritten to their parent CID via
            :func:`_canonicalize_to_parent_chembl_id` so the dedup
            step collapses parent + salt rows of the same compound
            into a single entity.  Set ``False`` to disable as a
            rollback / debugging escape hatch. See.

    Returns:
        Unified DataFrame conforming to DrugTargetRecord schema.

    Notes:
        Single-source inputs are routed through the same unified
        path as multi-source inputs (no special-case early return).
        This is required so parent/salt canonicalization +
        post-canonicalization dedup also fires on default
        ``sources=["chembl"]`` runs.  Matchers and propagation are
        gated and become no-ops when their inputs are empty or
        ``chembl_name_index`` is ``None``, so the behavior of
        single-source non-chembl loads is unchanged byte-for-byte.
    """
    chembl_df = dataframes.get("chembl", pd.DataFrame())
    pdsp_df = dataframes.get("pdsp", pd.DataFrame())
    dgidb_df = dataframes.get("dgidb", pd.DataFrame())
    creeds_df = dataframes.get("creeds", pd.DataFrame())
    dsigdb_df = dataframes.get("dsigdb", pd.DataFrame())

    if not chembl_df.empty and not pdsp_df.empty:
        pdsp_df = _match_pdsp_to_chembl(pdsp_df, chembl_df)

    if chembl_name_index is not None and not dgidb_df.empty:
        dgidb_df = _match_dgidb_to_chembl(dgidb_df, chembl_name_index)
    elif not dgidb_df.empty and "_drug_claim_name" in dgidb_df.columns:
        # Even without an index, strip the transient column so it
        # never reaches the merged DataFrame.
        dgidb_df = dgidb_df.drop(columns=["_drug_claim_name"])

    # Same name-based placeholder->CHEMBL upgrade as
    # DGIdb, but driven by the simpler matcher (no _drug_claim_name
    # fallback).  Skipped silently when ``chembl_name_index`` is None
    # (e.g. when the caller did not load ChEMBL).
    if chembl_name_index is not None:
        if not creeds_df.empty:
            creeds_df = _match_expression_source_to_chembl(
                creeds_df, chembl_name_index, source_label="creeds",
            )
        if not dsigdb_df.empty:
            dsigdb_df = _match_expression_source_to_chembl(
                dsigdb_df, chembl_name_index, source_label="dsigdb",
            )

    all_dfs = [
        df for df in [chembl_df, pdsp_df, dgidb_df, creeds_df, dsigdb_df]
        if not df.empty
    ]
    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True, sort=False)

    # Parent/salt canonicalization runs AFTER all matchers
    # (so PDSP / DGIdb placeholder IDs already resolved to real
    # CHEMBL IDs are also canonicalized to their parent) and BEFORE
    # _deduplicate_records (so parent + salt rows on the same
    # gene_symbol get collapsed into a single (drug, gene) record).
    if parent_salt_unification and chembl_name_index is not None:
        combined = _canonicalize_to_parent_chembl_id(
            combined, chembl_name_index,
        )

    combined = _deduplicate_records(combined)

    if chembl_name_index is not None:
        loaded_ids: Optional[set[str]] = None
        if not chembl_df.empty and "drug_chembl_id" in chembl_df.columns:
            loaded_ids = set(
                chembl_df["drug_chembl_id"].dropna().astype(str).unique()
            )
        combined = _propagate_chembl_atc_to_remapped(
            combined, chembl_name_index, loaded_chembl_ids=loaded_ids,
        )

    return _finalize_schema(combined)


def _match_pdsp_to_chembl(
    pdsp_df: pd.DataFrame,
    chembl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Upgrade PDSP placeholder IDs to real ChEMBL IDs where possible."""
    if pdsp_df.empty or chembl_df.empty:
        return pdsp_df

    chembl_dedup = chembl_df.drop_duplicates("drug_chembl_id").copy()
    chembl_dedup["_name_upper"] = chembl_dedup["drug_name"].astype(str).str.upper().str.strip()
    chembl_dedup = chembl_dedup[
        chembl_dedup["drug_chembl_id"].astype(str).str.startswith("CHEMBL")
        & (chembl_dedup["_name_upper"] != "")
    ]

    name_to_chembl_id = (
        chembl_dedup
        .drop_duplicates(subset=["_name_upper"])
        .set_index("_name_upper")["drug_chembl_id"]
        .to_dict()
    )
    name_to_inchikey = (
        chembl_dedup
        .dropna(subset=["drug_inchikey"])
        .drop_duplicates(subset=["_name_upper"])
        .set_index("_name_upper")["drug_inchikey"]
        .to_dict()
    ) if "drug_inchikey" in chembl_dedup.columns else {}

    pdsp_df = pdsp_df.copy()
    pdsp_name_upper = pdsp_df["drug_name"].astype(str).str.upper().str.strip()
    mapped_ids = pdsp_name_upper.map(name_to_chembl_id)
    pdsp_df["drug_chembl_id"] = mapped_ids.fillna(pdsp_df["drug_chembl_id"])

    mapped_inchikeys = pdsp_name_upper.map(name_to_inchikey)
    if "drug_inchikey" not in pdsp_df.columns:
        pdsp_df["drug_inchikey"] = mapped_inchikeys
    else:
        pdsp_df["drug_inchikey"] = pdsp_df["drug_inchikey"].fillna(mapped_inchikeys)

    n_matched = pdsp_df["drug_chembl_id"].astype(str).str.startswith("CHEMBL").sum()
    logger.info(
        "PDSP->ChEMBL name matching: %d / %d drugs matched (%.0f%%)",
        n_matched, len(pdsp_df),
        100 * n_matched / max(len(pdsp_df), 1),
    )
    return pdsp_df


def _match_dgidb_to_chembl(
    dgidb_df: pd.DataFrame,
    index: ChemblNameIndex,
) -> pd.DataFrame:
    """Upgrade DGIdb placeholder IDs to real ChEMBL IDs via the synonym index.

    For each row whose ``drug_chembl_id`` is not already a real
    ``CHEMBL\\d+`` token, attempt :func:`_resolve_synonym_index` on
    the row's ``drug_name``; if that misses, fall back to the
    ``_drug_claim_name`` carried from :func:`load_dgidb`.

    This function is a pure ``drug_chembl_id`` rewriter.  It does
    not mutate ``confidence`` - multi-source confidence upgrade is
    centralized in :func:`_deduplicate_records` and pre-mutating here
    would bias downstream ``confidence_filter`` behavior.

    The transient ``_drug_claim_name`` column is always dropped
    from the returned DataFrame (primary mitigation against schema
    leakage; :func:`_finalize_schema` provides a defensive net).
    """
    if dgidb_df.empty:
        return dgidb_df

    out = dgidb_df.copy()

    has_real_id = out["drug_chembl_id"].astype(str).str.startswith("CHEMBL")
    needs_resolve = ~has_real_id

    counters: dict[str, int] = defaultdict(int)
    new_ids = out["drug_chembl_id"].copy()

    drug_names = out["drug_name"].astype(object)
    if "_drug_claim_name" in out.columns:
        claim_names = out["_drug_claim_name"].astype(object)
    else:
        claim_names = pd.Series([""] * len(out), index=out.index, dtype=object)

    for idx in out.index[needs_resolve]:
        n = drug_names.loc[idx]
        cid, origin = _resolve_synonym_index(n, index)
        if cid is None:
            cn = claim_names.loc[idx]
            if cn and (not isinstance(n, str) or _normalize_drug_name(cn) != _normalize_drug_name(n)):
                cid_c, origin_c = _resolve_synonym_index(cn, index)
                if cid_c is not None:
                    new_ids.loc[idx] = cid_c
                    counters[f"claim:{origin_c}"] += 1
                    continue
            counters[f"name:{origin}"] += 1
        else:
            new_ids.loc[idx] = cid
            counters[f"name:{origin}"] += 1

    out["drug_chembl_id"] = new_ids

    n_total = int(needs_resolve.sum())
    n_resolved_now = int(
        out.loc[needs_resolve, "drug_chembl_id"]
        .astype(str)
        .str.startswith("CHEMBL")
        .sum()
    )
    logger.info(
        "DGIdb->ChEMBL synonym matching: %d / %d placeholder rows upgraded "
        "to real ChEMBL IDs (%.1f%%)",
        n_resolved_now, n_total,
        100.0 * n_resolved_now / max(n_total, 1),
    )
    if counters:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counters.items()))
        logger.info("  resolution breakdown: %s", breakdown)

    if "_drug_claim_name" in out.columns:
        out = out.drop(columns=["_drug_claim_name"])

    return out


def _match_expression_source_to_chembl(
    df: pd.DataFrame,
    index: ChemblNameIndex,
    source_label: str,
) -> pd.DataFrame:
    """Upgrade expression-source placeholder IDs to real ChEMBL IDs.

    Mirrors :func:`_match_dgidb_to_chembl` for
    expression-perturbation sources (CREEDS, DSigDB) which carry only a
    single ``drug_name`` column (no ``_drug_claim_name`` fallback).
    Placeholder IDs (e.g. ``CREEDS_<name>``, ``DSIGDB_<name>``) are
    resolved through :func:`_resolve_synonym_index` so the dedup step
    can collapse the same compound across target-family and
    expression sources.

    Like :func:`_match_dgidb_to_chembl` this is a **pure
    ``drug_chembl_id`` rewriter** - confidence-upgrade is centralised
    in :func:`_deduplicate_records` (where the tokenised
    target-family rule means an expression-only co-occurrence does
    NOT trigger the upgrade).
    """
    if df.empty:
        return df

    out = df.copy()
    has_real_id = out["drug_chembl_id"].astype(str).str.startswith("CHEMBL")
    needs_resolve = ~has_real_id

    counters: dict[str, int] = defaultdict(int)
    new_ids = out["drug_chembl_id"].copy()
    drug_names = out["drug_name"].astype(object)

    for idx in out.index[needs_resolve]:
        cid, origin = _resolve_synonym_index(drug_names.loc[idx], index)
        if cid is not None:
            new_ids.loc[idx] = cid
            counters[f"name:{origin}"] += 1
        else:
            counters[f"name:{origin}"] += 1

    out["drug_chembl_id"] = new_ids

    n_total = int(needs_resolve.sum())
    n_resolved_now = int(
        out.loc[needs_resolve, "drug_chembl_id"]
        .astype(str)
        .str.startswith("CHEMBL")
        .sum()
    )
    logger.info(
        "%s->ChEMBL synonym matching: %d / %d placeholder rows upgraded "
        "to real ChEMBL IDs (%.1f%%)",
        source_label.upper(), n_resolved_now, n_total,
        100.0 * n_resolved_now / max(n_total, 1),
    )
    if counters:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counters.items()))
        logger.info("  resolution breakdown: %s", breakdown)

    return out


def _propagate_chembl_atc_to_remapped(
    combined_df: pd.DataFrame,
    index: ChemblNameIndex,
    loaded_chembl_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Strictly-additive ATC/indication propagation for remapped rows.

    For every row whose ``drug_chembl_id`` is a real ``CHEMBL\\d+``
    token, fill ``atc_codes`` from ``index.chembl_to_atc.get(cid)``
    when the column is empty/null, and similarly for
    ``indication_mesh`` from ``index.chembl_to_ind``.

    Idempotent for rows already populated by the ChEMBL merge.
    Logs a coverage summary partitioned by whether the cid is in
    ``loaded_chembl_ids`` (post-merge cross-source) versus
    "ghost-but-ATC-coded" (real ChEMBL drug whose pharmacology was
    not loaded under the current scope but whose ATC is still
    propagated).
    """
    if combined_df.empty:
        return combined_df

    if "drug_chembl_id" not in combined_df.columns:
        return combined_df

    cid_str = combined_df["drug_chembl_id"].astype(object).astype(str)
    is_real = cid_str.str.startswith("CHEMBL")

    if not is_real.any():
        return combined_df

    atc_dict = index.chembl_to_atc
    ind_dict = index.chembl_to_ind

    n_atc_filled = 0
    n_ind_filled = 0
    n_in_loaded = 0
    n_ghost_with_atc = 0
    n_no_atc = 0

    df = combined_df.copy()
    if "atc_codes" not in df.columns:
        df["atc_codes"] = None
    if "indication_mesh" not in df.columns:
        df["indication_mesh"] = None

    atc_col = df["atc_codes"]
    ind_col = df["indication_mesh"]

    for idx in df.index[is_real]:
        cid = cid_str.loc[idx]
        if loaded_chembl_ids is not None:
            if cid in loaded_chembl_ids:
                n_in_loaded += 1
            elif cid in atc_dict:
                n_ghost_with_atc += 1
            else:
                n_no_atc += 1
        if _is_missing(atc_col.loc[idx]):
            atc_val = atc_dict.get(cid)
            if atc_val:
                atc_col.loc[idx] = list(atc_val)
                n_atc_filled += 1
        if _is_missing(ind_col.loc[idx]):
            ind_val = ind_dict.get(cid)
            if ind_val:
                ind_col.loc[idx] = list(ind_val)
                n_ind_filled += 1

    df["atc_codes"] = atc_col
    df["indication_mesh"] = ind_col

    if loaded_chembl_ids is not None:
        logger.info(
            "ChEMBL ATC propagation to remapped rows: %d filled (atc), %d filled (indication); "
            "of remapped real-CHEMBL rows: %d in loaded ChEMBL, %d ghost-but-ATC-coded, %d no-ATC",
            n_atc_filled, n_ind_filled,
            n_in_loaded, n_ghost_with_atc, n_no_atc,
        )
    else:
        logger.info(
            "ChEMBL ATC propagation to remapped rows: %d filled (atc), %d filled (indication)",
            n_atc_filled, n_ind_filled,
        )

    return df


_LIST_LIKE_COLS = frozenset({"atc_codes", "indication_mesh", "source_pmids"})


def _is_missing(value: object) -> bool:
    """Check if a dedup cell value is missing, handling list-like fields safely."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except (ValueError, TypeError):
        return False


def _merge_list_values(series: pd.Series) -> list | None:
    """Merge list-like column values across duplicate rows (unique, order-preserved)."""
    seen: set = set()
    merged: list = []
    for val in series:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        items = val if isinstance(val, (list, tuple, set)) else [val]
        for item in items:
            if item is not None and item not in seen:
                seen.add(item)
                merged.append(item)
    return merged if merged else None


_TARGET_SOURCE_FAMILY: frozenset[str] = frozenset({"chembl", "pdsp", "dgidb"})


def _deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate drug-gene pairs, keeping the richest record.

    When the same drug-gene pair appears from multiple sources, keep
    the record with the most populated fields.  Merge the ``source``
    column (comma-separated) and upgrade confidence if multiple
    independent sources agree.  List-like columns are merged across
    duplicates (unique values, order-preserved).

    Confidence upgrade:
        Only target-family source agreement (≥2 distinct of
        ``chembl``, ``pdsp``, ``dgidb``) triggers an upgrade to
        ``"high"``.  Mixed target+expression evidence (e.g. ChEMBL +
        CREEDS) does not upgrade because direct binding and
        downstream expression perturbation are not independent
        confirmations of the same biological claim.

        The check tokenises every row's ``source`` string by comma so
        compound tokens (``"chembl,dgidb"`` already merged from a
        prior dedup pass) are counted correctly.
    """
    if df.empty:
        return df

    key_cols = ["drug_chembl_id", "gene_symbol"]
    if not all(c in df.columns for c in key_cols):
        return df

    df = df.copy()
    df["_richness"] = df.notna().sum(axis=1)

    groups = []
    for _, group in df.groupby(key_cols, sort=False):
        if len(group) == 1:
            groups.append(group.iloc[0])
            continue

        best = group.sort_values("_richness", ascending=False).iloc[0].copy()

        # Tokenise per-row so compound source
        # strings (e.g. "chembl,dgidb" merged in an earlier dedup
        # pass) are unioned at the atomic-token level, not treated
        # as opaque list elements.
        all_tokens: set[str] = set()
        for s in group["source"].dropna():
            for tok in str(s).split(","):
                tok = tok.strip().lower()
                if tok:
                    all_tokens.add(tok)
        best["source"] = ",".join(sorted(all_tokens))

        target_tokens = all_tokens & _TARGET_SOURCE_FAMILY
        if len(target_tokens) >= 2 and best.get("confidence") != "high":
            best["confidence"] = "high"

        for col in group.columns:
            if col in _LIST_LIKE_COLS:
                merged = _merge_list_values(group[col])
                if merged is not None:
                    best[col] = merged
            elif _is_missing(best.get(col)):
                non_null = group[col].dropna()
                if not non_null.empty:
                    best[col] = non_null.iloc[0]

        groups.append(best)

    result = pd.DataFrame(groups).reset_index(drop=True)
    result = result.drop(columns=["_richness"], errors="ignore")

    logger.info(
        "Deduplication: %d -> %d records", len(df), len(result),
    )
    return result


def _finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all DrugTargetRecord columns exist with correct types."""
    if df.empty:
        return df

    schema_cols = {
        "drug_name": str,
        "drug_chembl_id": str,
        "drug_inchikey": str,
        "drug_pubchem_cid": str,
        "drug_smiles": str,
        "gene_symbol": str,
        "gene_ensembl_id": str,
        "gene_uniprot_id": str,
        "gene_entrez_id": "Int64",
        "interaction_type": str,
        "action_type": str,
        "mechanism_of_action": str,
        "pchembl_value": float,
        "affinity_value": float,
        "affinity_type": str,
        "affinity_unit": str,
        "max_phase": int,
        "atc_codes": object,
        "indication_mesh": object,
        "molecule_type": str,
        "is_withdrawn": bool,
        "source": str,
        "confidence": str,
        "source_pmids": object,
    }

    for col, dtype in schema_cols.items():
        if col not in df.columns:
            if dtype == "Int64":
                df[col] = pd.array([pd.NA] * len(df), dtype="Int64")
            elif dtype is object:
                df[col] = None
            elif dtype is bool:
                df[col] = False
            elif dtype is int:
                df[col] = 0
            elif dtype is float:
                df[col] = float("nan")
            else:
                df[col] = pd.NA

    if "gene_entrez_id" in df.columns and df["gene_entrez_id"].dtype != pd.Int64Dtype():
        df["gene_entrez_id"] = pd.to_numeric(
            df["gene_entrez_id"], errors="coerce"
        ).astype("Int64")

    if "gene_entrez_id" in df.columns:
        n_total = len(df)
        n_null = int(df["gene_entrez_id"].isna().sum())
        if n_null == n_total:
            logger.warning(
                "gene_entrez_id is null for ALL %d records. "
                "Check that reference.ncbi_gene_info and "
                "reference.ncbi_gene_history are set in configs/reference.yaml.",
                n_total,
            )
        elif n_null > 0:
            logger.info(
                "Entrez ID coverage: %d / %d records (%.1f%%) have gene_entrez_id",
                n_total - n_null,
                n_total,
                100.0 * (n_total - n_null) / n_total,
            )

    desired_order = [c for c in schema_cols if c in df.columns]
    extra_cols = [c for c in df.columns if c not in schema_cols]
    if extra_cols:
        # Defensive net: any transient or helper column that escaped
        # earlier dedup/strip steps is dropped here so the canonical
        # schema is the strict superset of output columns.  This is a
        # behavior tightening (not a contract change) and guards
        # against future leakage of leading-underscore columns.
        logger.debug(
            "_finalize_schema: dropping %d non-schema column(s): %s",
            len(extra_cols), extra_cols,
        )
    df = df[desired_order]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _standardize_interaction_type(raw: object) -> str:
    """Map a raw action_type string to a canonical interaction type."""
    if pd.isna(raw) or not str(raw).strip():
        return "other"
    raw_str = str(raw).strip()
    return INTERACTION_TYPE_STANDARDISATION.get(
        raw_str,
        INTERACTION_TYPE_STANDARDISATION.get(raw_str.upper(), "other"),
    )


def _parse_pipe_list(value: object) -> Optional[list[str]]:
    """Parse a pipe-delimited string into a list, or return None."""
    if pd.isna(value) or not str(value).strip():
        return None
    items = [v.strip() for v in str(value).split("|") if v.strip()]
    return items if items else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load drug-target interactions from ChEMBL/PDSP/DGIdb"
    )
    parser.add_argument(
        "--chembl-sqlite", type=Path, required=True,
        help="Path to ChEMBL SQLite database",
    )
    parser.add_argument(
        "--pdsp-csv", type=Path, default=None,
        help="Path to PDSP Ki database CSV",
    )
    parser.add_argument(
        "--dgidb-tsv", type=Path, default=None,
        help="Path to DGIdb interactions TSV",
    )
    parser.add_argument(
        "--biomart-dir", type=Path,
        default=Path("resources/reference"),
        help="Directory containing BioMart dictionary files",
    )
    parser.add_argument(
        "--gene-info", type=Path, default=None,
        help="Path to NCBI gene_info.gz",
    )
    parser.add_argument(
        "--gene-history", type=Path, default=None,
        help="Path to NCBI gene_history.gz",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--min-pchembl", type=float, default=None,
        help="Minimum pChEMBL value",
    )
    parser.add_argument(
        "--max-phase", type=int, default=None,
        help="Minimum clinical phase filter",
    )
    parser.add_argument(
        "--chembl-scope", type=str, default="mechanism_only",
        choices=["mechanism_only", "mechanism_or_affinity"],
        help="ChEMBL inclusion mode (default: mechanism_only)",
    )
    parser.add_argument(
        "--unichem-mapping", type=Path, default=None,
        help="UniChem src1src22.txt.gz (ChEMBL->PubChem CID mapping)",
    )
    parser.add_argument(
        "--parent-salt-unification",
        dest="parent_salt_unification",
        action="store_true",
        default=True,
        help="Collapse child/salt CHEMBL IDs onto parent CID "
             "before drug-gene dedup (default: enabled).",
    )
    parser.add_argument(
        "--no-parent-salt-unification",
        dest="parent_salt_unification",
        action="store_false",
        help="Disable parent/salt unification (rollback / debug).",
    )
    parser.add_argument(
        "--expression-sources", type=str, default="",
        help=(
            "comma-separated expression-perturbation "
            "sources (subset of 'creeds,dsigdb').  Strictly disjoint "
            "from target-family sources.  Empty (default) means "
            "TARGETS-only (byte-identical to the previous behaviour)."
        ),
    )
    parser.add_argument(
        "--creeds-path", type=Path, default=None,
        help=(
            "Path to CREEDS single_drug_perturbations JSON.  "
            "Required iff 'creeds' is in --expression-sources."
        ),
    )
    parser.add_argument(
        "--dsigdb-path", type=Path, default=None,
        help=(
            "Path to DSigDB D3 GMT-like file.  Required iff "
            "'dsigdb' is in --expression-sources."
        ),
    )
    args = parser.parse_args()

    sources = ["chembl"]
    if args.pdsp_csv:
        sources.append("pdsp")
    if args.dgidb_tsv:
        sources.append("dgidb")

    expression_sources: list[str] = []
    if args.expression_sources:
        expression_sources = [
            s.strip().lower()
            for s in args.expression_sources.split(",")
            if s.strip()
        ]

    biomart = {
        "ensembl_to_name": args.biomart_dir / "biomart_dico1",
        "name_to_ensembl": args.biomart_dir / "biomart_dico2",
        "uniprot_to_ensembl": args.biomart_dir / "biomart_dico3",
    }
    converter = GeneIDConverter(
        biomart_dicts=biomart,
        entrez_mapping_file=args.gene_info,
        gene_history_file=args.gene_history,
    )

    result_df = load_drug_targets(
        chembl_sqlite=args.chembl_sqlite,
        pdsp_csv=args.pdsp_csv,
        dgidb_tsv=args.dgidb_tsv,
        gene_id_converter=converter,
        sources=sources,
        min_pchembl=args.min_pchembl,
        max_phase_filter=args.max_phase,
        chembl_scope=args.chembl_scope,
        unichem_mapping=args.unichem_mapping,
        parent_salt_unification=args.parent_salt_unification,
        expression_sources=expression_sources,
        creeds_path=args.creeds_path,
        dsigdb_path=args.dsigdb_path,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Saved %d records to %s", len(result_df), output_path)
