"""Robust bidirectional gene ID conversion.

Supports all four ID types used by the pipeline:

* Ensembl - used by S-PrediXcan models and gene annotation.
* HGNC Symbol - used by MAGMA, pathways, human-readable output.
* UniProt - used by ChEMBL target lookup.
* Entrez - used by LINCS L1000 gene matching.

Primary data sources are BioMart dictionary files (local, fast) with
optional NCBI ``gene_info`` for Entrez mapping and ``gene_history``
for resolving deprecated / retired gene IDs.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Optional

import pandas as pd

from repogen.utils.io import check_file_exists
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


class GeneIDConverter:
    """Bidirectional gene ID converter.

    Initialised with BioMart dictionary files and optionally NCBI
    gene_info / gene_history.  Provides conversion between all four ID
    types and batch annotation of DataFrames.

    Args:
        biomart_dicts: Mapping of logical names to file paths.
            Expected keys: ``"ensembl_to_name"`` (dico1),
            ``"name_to_ensembl"`` (dico2), ``"uniprot_to_ensembl"``
            (dico3).  Missing keys are tolerated (the corresponding
            lookup will return ``None``).
        entrez_mapping_file: Path to NCBI ``gene_info.gz`` (tab-
            separated, ``#tax_id`` header).  Filtered to *Homo sapiens*
            (tax_id 9606).
        gene_history_file: Path to NCBI ``gene_history.gz`` for
            resolving deprecated gene IDs.
        cache_dir: Optional directory for caching processed lookups.
    """

    def __init__(
        self,
        biomart_dicts: dict[str, Path],
        entrez_mapping_file: Optional[Path] = None,
        gene_history_file: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None

        self._ensembl_to_symbol: dict[str, str] = {}
        self._symbol_to_ensembl: dict[str, list[str]] = {}
        self._uniprot_to_ensembl: dict[str, str] = {}
        self._ensembl_to_uniprot: dict[str, str] = {}

        self._symbol_to_entrez: dict[str, int] = {}
        self._entrez_to_symbol: dict[int, str] = {}
        self._ensembl_to_entrez: dict[str, int] = {}
        self._entrez_to_ensembl: dict[int, str] = {}

        self._deprecated_entrez: dict[int, int] = {}

        self._load_biomart(biomart_dicts)

        if entrez_mapping_file is not None:
            self._load_ncbi_gene_info(Path(entrez_mapping_file))

        if gene_history_file is not None:
            self._load_gene_history(Path(gene_history_file))

        logger.info(
            "GeneIDConverter ready: %d Ensembl↔Symbol, %d UniProt↔Ensembl, "
            "%d Symbol↔Entrez, %d deprecated IDs tracked",
            len(self._ensembl_to_symbol),
            len(self._uniprot_to_ensembl),
            len(self._symbol_to_entrez),
            len(self._deprecated_entrez),
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_biomart(self, dicts: dict[str, Path]) -> None:
        """Load BioMart tab-separated dictionary files."""
        if "ensembl_to_name" in dicts:
            self._ensembl_to_symbol = self._read_tsv_pair(
                dicts["ensembl_to_name"], "ensembl_to_name"
            )

        if "name_to_ensembl" in dicts:
            raw = self._read_tsv_pair(
                dicts["name_to_ensembl"], "name_to_ensembl"
            )
            for symbol, ensembl_id in raw.items():
                self._symbol_to_ensembl.setdefault(symbol, []).append(ensembl_id)

        if "uniprot_to_ensembl" in dicts:
            self._uniprot_to_ensembl = self._read_tsv_pair(
                dicts["uniprot_to_ensembl"], "uniprot_to_ensembl"
            )
            for uniprot, ensembl in self._uniprot_to_ensembl.items():
                self._ensembl_to_uniprot.setdefault(ensembl, uniprot)

    @staticmethod
    def _read_tsv_pair(path: Path, label: str) -> dict[str, str]:
        """Read a two-column TSV into a dict (first-seen wins)."""
        path = Path(path)
        if not path.exists():
            logger.warning("BioMart file not found (%s): %s", label, path)
            return {}
        mapping: dict[str, str] = {}
        opener = gzip.open if path.suffix in (".gz", ".bgz") else open
        with opener(path, "rt") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                key, value = parts[0].strip(), parts[1].strip()
                if key and value and key not in mapping:
                    mapping[key] = value
        logger.info("Loaded %d entries from %s (%s)", len(mapping), path.name, label)
        return mapping

    def _load_ncbi_gene_info(self, path: Path) -> None:
        """Load NCBI gene_info.gz to build Entrez↔Symbol↔Ensembl maps.

        Expected columns (tab-separated, ``#`` header):
        ``#tax_id  GeneID  Symbol  ...  dbXrefs  ...``

        ``dbXrefs`` (column index 5) contains ``Ensembl:ENSG...`` entries.
        """
        path = check_file_exists(path, label="NCBI gene_info")
        opener = gzip.open if path.suffix == ".gz" else open
        n_loaded = 0
        with opener(path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                tax_id = parts[0]
                if tax_id != "9606":
                    continue

                try:
                    entrez_id = int(parts[1])
                except ValueError:
                    continue
                symbol = parts[2]
                dbxrefs = parts[5]

                if symbol and symbol != "-":
                    self._symbol_to_entrez.setdefault(symbol, entrez_id)
                    self._entrez_to_symbol.setdefault(entrez_id, symbol)

                if "Ensembl:" in dbxrefs:
                    for token in dbxrefs.split("|"):
                        if token.startswith("Ensembl:"):
                            ensembl_id = token.split(":", 1)[1]
                            self._ensembl_to_entrez.setdefault(ensembl_id, entrez_id)
                            self._entrez_to_ensembl.setdefault(entrez_id, ensembl_id)
                            break

                n_loaded += 1

        logger.info(
            "NCBI gene_info: %d human genes loaded (%d with Ensembl xref)",
            n_loaded,
            len(self._ensembl_to_entrez),
        )

    def _load_gene_history(self, path: Path) -> None:
        """Load NCBI gene_history.gz for deprecated-ID resolution.

        Maps discontinued GeneID -> replacement GeneID.
        """
        path = check_file_exists(path, label="NCBI gene_history")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[0] != "9606":
                    continue
                try:
                    old_id = int(parts[2])
                except ValueError:
                    continue
                replacement_raw = parts[1]
                if replacement_raw == "-":
                    continue
                try:
                    new_id = int(replacement_raw)
                except ValueError:
                    continue
                self._deprecated_entrez[old_id] = new_id

        logger.info(
            "NCBI gene_history: %d deprecated Entrez IDs loaded",
            len(self._deprecated_entrez),
        )

    # ------------------------------------------------------------------
    # Public conversion API
    # ------------------------------------------------------------------

    def convert(
        self,
        ids: list[str],
        from_type: str,
        to_type: str,
    ) -> dict[str, Optional[str]]:
        """Convert a list of gene IDs from one type to another.

        Unresolved IDs map to ``None`` (not silently dropped).

        Args:
            ids: Gene identifiers to convert.
            from_type: Source type (``"symbol"``, ``"ensembl"``,
                ``"uniprot"``, ``"entrez"``).
            to_type: Target type (same options).

        Returns:
            Mapping of input ID -> output ID (or ``None``).
        """
        result: dict[str, Optional[str]] = {}
        for gid in ids:
            record = self.get_full_record(gid, from_type)
            if record is None:
                result[gid] = None
                continue
            target = record.get(to_type)
            result[gid] = str(target) if target is not None else None
        return result

    def get_full_record(self, gene_id: str, id_type: str) -> Optional[dict]:
        """Return all known IDs for a gene.

        Args:
            gene_id: The gene identifier.
            id_type: Type of *gene_id* (``"symbol"``, ``"ensembl"``,
                ``"uniprot"``, ``"entrez"``).

        Returns:
            Dict with keys ``symbol``, ``ensembl``, ``uniprot``,
            ``entrez`` (values may be ``None``), or ``None`` if the
            gene cannot be found at all.
        """
        symbol: Optional[str] = None
        ensembl: Optional[str] = None
        uniprot: Optional[str] = None
        entrez: Optional[int] = None

        if id_type == "ensembl":
            ensembl = gene_id
            symbol = self._ensembl_to_symbol.get(gene_id)
            uniprot = self._ensembl_to_uniprot.get(gene_id)
            entrez = self._ensembl_to_entrez.get(gene_id)
            if symbol is None and entrez is not None:
                symbol = self._entrez_to_symbol.get(entrez)
        elif id_type == "symbol":
            symbol = gene_id
            ensembl_list = self._symbol_to_ensembl.get(gene_id, [])
            ensembl = ensembl_list[0] if ensembl_list else None
            entrez = self._symbol_to_entrez.get(gene_id)
            if ensembl:
                uniprot = self._ensembl_to_uniprot.get(ensembl)
        elif id_type == "uniprot":
            uniprot = gene_id
            ensembl = self._uniprot_to_ensembl.get(gene_id)
            if ensembl:
                symbol = self._ensembl_to_symbol.get(ensembl)
                entrez = self._ensembl_to_entrez.get(ensembl)
        elif id_type == "entrez":
            try:
                eid = int(gene_id)
            except ValueError:
                return None
            entrez = eid
            symbol = self._entrez_to_symbol.get(eid)
            ens_from_entrez = self._entrez_to_ensembl.get(eid)
            if ens_from_entrez:
                ensembl = ens_from_entrez
            elif symbol:
                ens_list = self._symbol_to_ensembl.get(symbol, [])
                ensembl = ens_list[0] if ens_list else None
            if ensembl:
                uniprot = self._ensembl_to_uniprot.get(ensembl)
        else:
            raise ValueError(
                f"Unknown id_type '{id_type}'. "
                "Must be 'symbol', 'ensembl', 'uniprot', or 'entrez'."
            )

        if symbol is None and ensembl is None and uniprot is None and entrez is None:
            return None

        return {
            "symbol": symbol,
            "ensembl": ensembl,
            "uniprot": uniprot,
            "entrez": entrez,
        }

    def resolve_deprecated(self, gene_id: str, id_type: str) -> Optional[str]:
        """Attempt to resolve a deprecated gene ID to its current version.

        Currently supports Entrez IDs via NCBI gene_history.

        Args:
            gene_id: The potentially deprecated identifier.
            id_type: Type of *gene_id*.

        Returns:
            The current ID as a string, or ``None`` if unresolvable.
        """
        if id_type == "entrez":
            try:
                old_eid = int(gene_id)
            except ValueError:
                return None
            new_eid = self._deprecated_entrez.get(old_eid)
            if new_eid is not None:
                chain_limit = 10
                visited: set[int] = {old_eid}
                current = new_eid
                for _ in range(chain_limit):
                    if current not in self._deprecated_entrez:
                        break
                    nxt = self._deprecated_entrez[current]
                    if nxt in visited:
                        break
                    visited.add(current)
                    current = nxt
                return str(current)
        return None

    def batch_annotate(
        self,
        df: pd.DataFrame,
        id_column: str,
        id_type: str,
    ) -> pd.DataFrame:
        """Add all four gene ID columns to *df* based on one input column.

        New columns added: ``gene_symbol``, ``gene_ensembl_id``,
        ``gene_uniprot_id``, ``gene_entrez_id``.  Rows where the input
        ID cannot be resolved get ``None`` in the new columns.

        Args:
            df: Input DataFrame.
            id_column: Name of the column containing gene identifiers.
            id_type: Type of IDs in *id_column*.

        Returns:
            A copy of *df* with the four new columns appended.
        """
        df = df.copy()
        symbols, ensembls, uniprots, entrezs = [], [], [], []

        unique_ids = df[id_column].dropna().unique()
        cache: dict[str, Optional[dict]] = {}
        for gid in unique_ids:
            cache[str(gid)] = self.get_full_record(str(gid), id_type)

        for raw_id in df[id_column]:
            rec = cache.get(str(raw_id)) if pd.notna(raw_id) else None
            if rec is None:
                symbols.append(None)
                ensembls.append(None)
                uniprots.append(None)
                entrezs.append(None)
            else:
                symbols.append(rec["symbol"])
                ensembls.append(rec["ensembl"])
                uniprots.append(rec["uniprot"])
                entrezs.append(rec["entrez"])

        new_cols = {
            "gene_symbol": symbols,
            "gene_ensembl_id": ensembls,
            "gene_uniprot_id": uniprots,
            "gene_entrez_id": pd.array(entrezs, dtype=pd.Int64Dtype()),
        }
        for col, new_values in new_cols.items():
            new_series = pd.Series(new_values, index=df.index)
            if col in df.columns:
                existing = df[col]
                df[col] = existing.where(existing.notna(), new_series)
            else:
                df[col] = new_series

        n_resolved = sum(1 for s in symbols if s is not None)
        logger.info(
            "batch_annotate: resolved %d / %d IDs (%.1f%%)",
            n_resolved,
            len(df),
            100 * n_resolved / max(len(df), 1),
        )
        return df

    def get_stats(self) -> dict[str, int]:
        """Return counts of loaded mappings."""
        return {
            "ensembl_to_symbol": len(self._ensembl_to_symbol),
            "symbol_to_ensembl": len(self._symbol_to_ensembl),
            "uniprot_to_ensembl": len(self._uniprot_to_ensembl),
            "symbol_to_entrez": len(self._symbol_to_entrez),
            "ensembl_to_entrez": len(self._ensembl_to_entrez),
            "deprecated_entrez": len(self._deprecated_entrez),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gene ID conversion utility"
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=Path("resources/reference"),
        help="Directory containing biomart_dico1/2/3 files",
    )
    parser.add_argument(
        "--gene-info",
        type=Path,
        default=None,
        help="Path to NCBI gene_info.gz",
    )
    parser.add_argument(
        "--gene-history",
        type=Path,
        default=None,
        help="Path to NCBI gene_history.gz",
    )
    parser.add_argument(
        "--test-id",
        type=str,
        default="HTR2A",
        help="Gene ID to test conversion",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default="symbol",
        choices=["symbol", "ensembl", "uniprot", "entrez"],
        help="Type of the test ID",
    )
    args = parser.parse_args()

    biomart = {
        "ensembl_to_name": args.mapping_dir / "biomart_dico1",
        "name_to_ensembl": args.mapping_dir / "biomart_dico2",
        "uniprot_to_ensembl": args.mapping_dir / "biomart_dico3",
    }

    converter = GeneIDConverter(
        biomart_dicts=biomart,
        entrez_mapping_file=args.gene_info,
        gene_history_file=args.gene_history,
    )

    record = converter.get_full_record(args.test_id, args.test_type)
    logger.info("%s=%s -> %s", args.test_type, args.test_id, record)

    stats = converter.get_stats()
    logger.info("Mapping statistics:")
    for key, value in stats.items():
        logger.info("  %s: %s", key, f"{value:,}")
