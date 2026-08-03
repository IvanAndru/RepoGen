"""MSigDB GMT parsing with rich metadata extraction.

Parses ``.gmt`` files into the ``PathwayRecord`` canonical schema,
extracting source database, human-readable pathway name, and optional
URL from MSigDB naming conventions.

Must use MSigDB v2024.1+, not the legacy v5.2 files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from repogen.data.schemas import validate_dataframe
from repogen.utils.io import check_file_exists, ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# MSigDB name -> source_db prefix mapping
# ---------------------------------------------------------------------------

_PREFIX_TO_SOURCE: dict[str, str] = {
    "GOBP_": "GO_BP",
    "GOCC_": "GO_CC",
    "GOMF_": "GO_MF",
    "KEGG_": "KEGG",
    "REACTOME_": "REACTOME",
    "WP_": "WIKIPATHWAYS",
    "BIOCARTA_": "BIOCARTA",
    "PID_": "PID",
    "HALLMARK_": "HALLMARK",
    "HP_": "HPO",
}

_SOURCE_URLS: dict[str, str] = {
    "GO_BP": "https://amigo.geneontology.org/amigo/term/{go_id}",
    "GO_CC": "https://amigo.geneontology.org/amigo/term/{go_id}",
    "GO_MF": "https://amigo.geneontology.org/amigo/term/{go_id}",
    "KEGG": "https://www.kegg.jp/kegg-bin/show_pathway?{pathway_id}",
    "REACTOME": "https://reactome.org/content/detail/{pathway_id}",
    "WIKIPATHWAYS": "https://www.wikipathways.org/index.php/Pathway:{pathway_id}",
}


# ---------------------------------------------------------------------------
# GMT parsing
# ---------------------------------------------------------------------------


def load_gene_sets(
    gmt_files: list[Path],
    gene_id_type: str = "symbol",
    min_size: int = 10,
    max_size: int = 500,
    sources: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Parse GMT files and return a ``PathwayRecord`` DataFrame.

    Args:
        gmt_files: List of paths to ``.gmt`` files.
        gene_id_type: Type of gene IDs in the GMT
            (``"symbol"``, ``"ensembl"``, ``"entrez"``).
        min_size: Minimum number of genes per pathway.
        max_size: Maximum number of genes per pathway.
        sources: Optional filter - only keep pathways from these
            databases (e.g. ``["GO_BP", "KEGG", "REACTOME"]``).

    Returns:
        ``PathwayRecord``-conforming DataFrame.
    """
    all_records: list[dict] = []

    for gmt_path in gmt_files:
        gmt_path = check_file_exists(Path(gmt_path), label="GMT file")
        records = _parse_single_gmt(gmt_path)
        all_records.extend(records)
        logger.info("Parsed %d gene sets from %s", len(records), gmt_path.name)

    if not all_records:
        logger.warning("No gene sets loaded from any GMT file")
        return pd.DataFrame(columns=[
            "pathway_id", "pathway_name", "source_db", "category",
            "description", "url", "genes", "n_genes",
        ])

    df = pd.DataFrame(all_records)

    if sources:
        sources_upper = [s.upper() for s in sources]
        before = len(df)
        df = df[df["source_db"].str.upper().isin(sources_upper)]
        logger.info("Source filter %s: %d -> %d gene sets", sources, before, len(df))

    before = len(df)
    df = df[(df["n_genes"] >= min_size) & (df["n_genes"] <= max_size)]
    logger.info(
        "Size filter [%d, %d]: %d -> %d gene sets", min_size, max_size, before, len(df)
    )

    df = df.drop_duplicates(subset="pathway_id", keep="first")

    errors = validate_dataframe(df, "PathwayRecord")
    if errors:
        logger.warning("Schema validation warnings: %s", "; ".join(errors))

    logger.info(
        "Gene sets loaded: %d total, sources: %s",
        len(df),
        sorted(df["source_db"].unique()),
    )
    return df.reset_index(drop=True)


def _parse_single_gmt(gmt_path: Path) -> list[dict]:
    """Parse a single GMT file into a list of record dicts."""
    records: list[dict] = []

    with open(gmt_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            set_name = parts[0].strip()
            description_field = parts[1].strip()
            genes = [g.strip() for g in parts[2:] if g.strip()]

            if not genes:
                continue

            metadata = parse_gmt_metadata(set_name)

            desc = description_field
            if desc in ("na", "NA", "", set_name):
                desc = metadata.get("description") or ""

            url = metadata.get("url") or _build_url(metadata.get("source_db", ""), set_name, description_field)

            records.append({
                "pathway_id": set_name,
                "pathway_name": metadata.get("name", set_name),
                "source_db": metadata.get("source_db", "UNKNOWN"),
                "category": metadata.get("category"),
                "description": desc if desc else None,
                "url": url,
                "genes": genes,
                "n_genes": len(genes),
            })

    return records


# ---------------------------------------------------------------------------
# Metadata extraction from MSigDB naming conventions
# ---------------------------------------------------------------------------


def parse_gmt_metadata(set_name: str) -> dict:
    """Extract source_db and human-readable name from an MSigDB set name.

    Examples::

        'GOBP_SYNAPTIC_SIGNALING'     -> source_db='GO_BP', name='Synaptic Signaling'
        'KEGG_MAPK_SIGNALING_PATHWAY' -> source_db='KEGG', name='MAPK Signaling Pathway'
        'REACTOME_SIGNALING_BY_GPCRS' -> source_db='REACTOME', name='Signaling By GPCRs'
        'WP_ALZHEIMERS_DISEASE'       -> source_db='WIKIPATHWAYS', name='Alzheimers Disease'
        'HALLMARK_TNFA_SIGNALING_VIA_NFKB' -> source_db='HALLMARK', name='TNFA Signaling Via NFKB'

    Args:
        set_name: The raw gene set name from the GMT file.

    Returns:
        Dict with keys ``source_db``, ``name``, and optionally
        ``category``, ``description``, ``url``.
    """
    for prefix, source_db in _PREFIX_TO_SOURCE.items():
        if set_name.startswith(prefix):
            remainder = set_name[len(prefix):]
            human_name = _underscores_to_title(remainder)
            return {
                "source_db": source_db,
                "name": human_name,
                "category": _infer_category(source_db),
            }

    if set_name.startswith("GO:"):
        return {
            "source_db": "GO",
            "name": set_name,
        }

    return {
        "source_db": _guess_source(set_name),
        "name": _underscores_to_title(set_name),
    }


def _underscores_to_title(s: str) -> str:
    """Convert ``MAPK_SIGNALING_PATHWAY`` -> ``MAPK Signaling Pathway``."""
    words = s.split("_")
    result = []
    for word in words:
        if len(word) <= 3 and word.isupper():
            result.append(word)
        else:
            result.append(word.capitalize())
    return " ".join(result)


def _infer_category(source_db: str) -> Optional[str]:
    """Map source_db to a broad category."""
    category_map = {
        "GO_BP": "biological_process",
        "GO_CC": "cellular_component",
        "GO_MF": "molecular_function",
        "KEGG": "canonical_pathways",
        "REACTOME": "canonical_pathways",
        "WIKIPATHWAYS": "canonical_pathways",
        "BIOCARTA": "canonical_pathways",
        "PID": "canonical_pathways",
        "HALLMARK": "hallmark",
        "HPO": "phenotype",
    }
    return category_map.get(source_db)


def _guess_source(set_name: str) -> str:
    """Best-effort source detection for non-prefixed names."""
    upper = set_name.upper()
    if "REACTOME" in upper:
        return "REACTOME"
    if "KEGG" in upper:
        return "KEGG"
    if "BIOCARTA" in upper:
        return "BIOCARTA"
    return "OTHER"


def _build_url(source_db: str, set_name: str, desc: str) -> Optional[str]:
    """Build a URL to the source database entry if possible."""
    go_match = re.search(r"(GO:\d+)", desc)
    if go_match and source_db.startswith("GO"):
        template = _SOURCE_URLS.get(source_db)
        if template:
            return template.format(go_id=go_match.group(1))

    return None


# ---------------------------------------------------------------------------
# MAGMA pathway file output
# ---------------------------------------------------------------------------


def create_magma_pathway_file(
    gene_sets: pd.DataFrame,
    gene_annotations: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write a MAGMA-compatible pathway definition file.

    Format: one line per pathway, ``PATHWAY_ID  GENE1 GENE2 ...``
    where gene IDs match those used in the MAGMA annotation file
    (Ensembl IDs).

    Args:
        gene_sets: ``PathwayRecord`` DataFrame with a ``genes`` column
            containing HGNC symbols.
        gene_annotations: ``GeneAnnotationRecord`` DataFrame for
            symbol -> Ensembl ID mapping.
        output_path: Where to write the MAGMA pathway file.

    Returns:
        Path to the written file.
    """
    symbol_to_ensembl = (
        gene_annotations
        .dropna(subset=["gene_symbol", "gene_ensembl_id"])
        .drop_duplicates(subset=["gene_symbol"])
        .set_index("gene_symbol")["gene_ensembl_id"]
        .to_dict()
    )

    lines: list[str] = []
    n_skipped = 0
    for row in gene_sets.itertuples(index=False):
        genes = row.genes
        if not isinstance(genes, list):
            continue

        mapped = [symbol_to_ensembl[g] for g in genes if g in symbol_to_ensembl]
        if len(mapped) >= 2:
            lines.append(f"{row.pathway_id}\t{' '.join(mapped)}")
        else:
            n_skipped += 1

    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n")

    logger.info(
        "MAGMA pathway file: %d pathways written, %d skipped (< 2 mapped genes) -> %s",
        len(lines), n_skipped, output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse MSigDB GMT files for RepoGen")
    parser.add_argument("--gmt", nargs="+", required=True, type=Path, help="GMT file(s)")
    parser.add_argument("--output", required=True, type=Path, help="Output Parquet file")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum genes per pathway")
    parser.add_argument("--max-size", type=int, default=500, help="Maximum genes per pathway")
    parser.add_argument("--sources", nargs="*", default=None, help="Source filter (GO_BP, KEGG, ...)")
    args = parser.parse_args()

    result = load_gene_sets(
        gmt_files=args.gmt,
        min_size=args.min_size,
        max_size=args.max_size,
        sources=args.sources,
    )

    ensure_directory(args.output.parent)
    result.to_parquet(args.output, engine="pyarrow", index=False)
    logger.info("Wrote %d gene sets to %s", len(result), args.output)
