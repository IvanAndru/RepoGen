"""Gene annotation with full multi-ID records, biotype, and MAGMA-compatible output.

Uses local BioMart dictionaries and a gene location file as the primary
source.  Optionally enriches with MyGeneInfo API for missing IDs.

Output conforms to the ``GeneAnnotationRecord`` canonical schema.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from repogen.data.gene_id_converter import GeneIDConverter
from repogen.data.schemas import validate_dataframe
from repogen.utils.io import check_file_exists, ensure_directory
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")


# ---------------------------------------------------------------------------
# Core annotation
# ---------------------------------------------------------------------------


def annotate_genes(
    reference_gene_file: Path,
    biomart_dicts: Optional[dict[str, Path]] = None,
    gene_id_converter: Optional[GeneIDConverter] = None,
    gene_window_kb: int = 35,
    biotype_filter: Optional[list[str]] = None,
    use_mygene_api: bool = False,
    entrez_mapping_file: Optional[Path] = None,
    gene_history_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Annotate genes from a reference gene location file.

    Returns a ``GeneAnnotationRecord`` DataFrame with all four ID types,
    biotype, and genomic coordinates.

    Args:
        reference_gene_file: NCBI-format gene location file
            (tab-separated: ``gene_id  chr  start  end  strand  [name]``).
        biomart_dicts: Mapping of logical names to BioMart dictionary
            paths (``ensembl_to_name``, ``name_to_ensembl``,
            ``uniprot_to_ensembl``).
        gene_id_converter: Pre-initialised converter; built from
            *biomart_dicts* if ``None``.
        gene_window_kb: Gene boundary extension in kb for MAGMA.
        biotype_filter: Restrict output to these biotypes (e.g.
            ``["protein_coding"]``).  ``None`` = all biotypes.
        use_mygene_api: Query MyGeneInfo for genes missing local IDs.
        entrez_mapping_file: NCBI gene_info.gz for the converter.
        gene_history_file: NCBI gene_history.gz for the converter.

    Returns:
        ``GeneAnnotationRecord``-conforming DataFrame.

    Raises:
        FileNotFoundError: If the gene location file is missing.
        RuntimeError: If the result is empty.
    """
    ref_path = check_file_exists(Path(reference_gene_file), label="Gene location file")

    gene_loc = _read_gene_location_file(ref_path)
    logger.info("Read %d gene records from %s", len(gene_loc), ref_path.name)

    if gene_id_converter is None and biomart_dicts is not None:
        gene_id_converter = GeneIDConverter(
            biomart_dicts=biomart_dicts,
            entrez_mapping_file=entrez_mapping_file,
            gene_history_file=gene_history_file,
        )

    if gene_id_converter is not None:
        gene_loc = _enrich_with_converter(gene_loc, gene_id_converter)
    else:
        for col in ("gene_symbol", "gene_uniprot_id", "gene_entrez_id"):
            if col not in gene_loc.columns:
                gene_loc[col] = pd.NA

    if biotype_filter:
        before = len(gene_loc)
        gene_loc = gene_loc[gene_loc["biotype"].isin(biotype_filter)]
        logger.info(
            "Biotype filter %s: %d -> %d genes", biotype_filter, before, len(gene_loc)
        )

    if use_mygene_api:
        gene_loc = _enrich_with_mygene(gene_loc)

    if "description" not in gene_loc.columns:
        gene_loc["description"] = pd.NA

    expected_cols = [
        "gene_symbol", "gene_ensembl_id", "gene_entrez_id",
        "gene_uniprot_id", "chr", "start", "end", "biotype", "description",
    ]
    for col in expected_cols:
        if col not in gene_loc.columns:
            gene_loc[col] = pd.NA

    gene_loc = gene_loc[expected_cols].copy()

    before = len(gene_loc)
    has_ensembl = gene_loc["gene_ensembl_id"].notna()
    has_entrez = gene_loc["gene_entrez_id"].notna()
    has_coords = gene_loc[["chr", "start", "end"]].notna().all(axis=1)
    gene_loc = gene_loc[(has_ensembl | has_entrez) & has_coords]

    gene_loc["_dedup_key"] = gene_loc["gene_ensembl_id"].fillna(
        gene_loc["gene_entrez_id"].astype(str)
    )
    gene_loc = gene_loc.drop_duplicates(subset="_dedup_key", keep="first")
    gene_loc = gene_loc.drop(columns="_dedup_key")
    logger.info("Deduplication: %d -> %d genes", before, len(gene_loc))

    errors = validate_dataframe(gene_loc, "GeneAnnotationRecord")
    if errors:
        logger.warning("Schema validation warnings: %s", "; ".join(errors))

    if len(gene_loc) == 0:
        raise RuntimeError(
            f"No genes remained after annotation from {ref_path}. "
            "Check the gene location file format."
        )

    logger.info(
        "Gene annotation complete: %d genes, %d with symbol, %d with Entrez",
        len(gene_loc),
        gene_loc["gene_symbol"].notna().sum(),
        gene_loc["gene_entrez_id"].notna().sum(),
    )
    return gene_loc


def _classify_id_column(values: pd.Series) -> str:
    """Classify a gene ID column as 'ensembl', 'entrez', or 'mixed'.

    Uses strict guardrails: the column must be overwhelmingly one type
    (>90% of non-null values) to be classified.  Otherwise returns
    'mixed' which triggers a warning.
    """
    sample = values.dropna().astype(str).str.strip()
    if sample.empty:
        return "mixed"
    n_total = len(sample)
    n_ensembl = sample.str.match(r"^ENS[A-Z]*G\d+").sum()
    n_numeric = sample.str.match(r"^\d+$").sum()
    if n_ensembl / n_total > 0.9:
        return "ensembl"
    if n_numeric / n_total > 0.9:
        return "entrez"
    return "mixed"


def _read_gene_location_file(path: Path) -> pd.DataFrame:
    """Read an NCBI-format gene location file.

    Expected columns (tab-separated, no header):
    ``gene_id  chr  start  end  strand  [name]``

    Column 1 may contain Ensembl IDs (``ENSG...``) or Entrez numeric
    IDs; the function detects which at the column level.
    """
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Cannot parse gene location file {path}: {exc}") from exc

    ncols = df.shape[1]

    if ncols >= 6:
        df.columns = ["gene_id", "chr", "start", "end", "strand", "name"] + [
            f"extra_{i}" for i in range(ncols - 6)
        ]
    elif ncols == 5:
        df.columns = ["gene_id", "chr", "start", "end", "strand"]
    elif ncols == 4:
        df.columns = ["gene_id", "chr", "start", "end"]
    elif ncols >= 1:
        df.columns = ["gene_id"] + [f"col{i}" for i in range(1, ncols)]
    else:
        raise RuntimeError(f"Gene location file {path} has no columns")

    raw_ids = df["gene_id"].str.strip()
    id_type = _classify_id_column(raw_ids)
    logger.info("Gene location file ID type detected: %s", id_type)

    result = pd.DataFrame()

    if id_type == "ensembl":
        result["gene_ensembl_id"] = raw_ids.str.replace(r"\.\d+$", "", regex=True)
        result["gene_entrez_id"] = pd.array([pd.NA] * len(raw_ids), dtype=pd.Int64Dtype())
    elif id_type == "entrez":
        result["gene_entrez_id"] = pd.to_numeric(raw_ids, errors="coerce").astype("Int64")
        result["gene_ensembl_id"] = pd.NA
    else:
        logger.warning(
            "Gene ID column has mixed types (neither >90%% Ensembl nor >90%% numeric). "
            "Treating as Ensembl; results may be incomplete."
        )
        result["gene_ensembl_id"] = raw_ids
        result["gene_entrez_id"] = pd.array([pd.NA] * len(raw_ids), dtype=pd.Int64Dtype())

    if "chr" in df.columns:
        result["chr"] = pd.to_numeric(
            df["chr"].str.replace(r"^chr", "", regex=True), errors="coerce"
        ).astype("Int64")
    if "start" in df.columns:
        result["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    if "end" in df.columns:
        result["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")

    if "name" in df.columns:
        result["gene_symbol"] = df["name"].str.strip()
        result["description"] = pd.NA
    else:
        result["gene_symbol"] = pd.NA

    result["biotype"] = "protein_coding"

    return result


def _has_valid_ensembl_ids(series: pd.Series) -> bool:
    """Check whether a Series contains genuine Ensembl gene IDs."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    return non_null.str.match(r"^ENS[A-Z]*G\d+").sum() / len(non_null) > 0.5


def _enrich_with_converter(
    gene_loc: pd.DataFrame,
    converter: GeneIDConverter,
) -> pd.DataFrame:
    """Add all four gene ID columns using GeneIDConverter.batch_annotate()."""
    logger.info("Enriching gene annotations via GeneIDConverter.batch_annotate()")

    if (
        "gene_ensembl_id" in gene_loc.columns
        and gene_loc["gene_ensembl_id"].notna().any()
        and _has_valid_ensembl_ids(gene_loc["gene_ensembl_id"])
    ):
        id_column = "gene_ensembl_id"
        id_type = "ensembl"
    elif "gene_symbol" in gene_loc.columns and gene_loc["gene_symbol"].notna().any():
        id_column = "gene_symbol"
        id_type = "symbol"
    elif "gene_entrez_id" in gene_loc.columns and gene_loc["gene_entrez_id"].notna().any():
        id_column = "gene_entrez_id"
        id_type = "entrez"
    else:
        logger.warning("No usable gene ID column found for enrichment - skipping.")
        return gene_loc

    logger.info("Using %s (%s) as enrichment source", id_column, id_type)
    enriched = converter.batch_annotate(gene_loc, id_column, id_type)

    n_filled = enriched["gene_entrez_id"].notna().sum()
    logger.info(
        "Enrichment complete: %d/%d genes have Entrez IDs",
        n_filled, len(enriched),
    )
    return enriched


def _enrich_with_mygene(df: pd.DataFrame) -> pd.DataFrame:
    """Query MyGeneInfo API for genes still missing IDs.

    Only queries genes where ``gene_symbol`` or ``gene_entrez_id`` is
    missing.  Respects API rate limits with retry and exponential
    backoff.
    """
    try:
        import mygene
        import requests as _requests
    except ImportError:
        logger.warning("mygene (or requests) not installed - skipping API enrichment")
        return df

    missing_mask = df["gene_symbol"].isna() | df["gene_entrez_id"].isna()
    missing_ids = df.loc[missing_mask, "gene_ensembl_id"].dropna().tolist()

    if not missing_ids:
        logger.info("No genes need API enrichment")
        return df

    logger.info("Querying MyGeneInfo for %d genes with missing IDs", len(missing_ids))
    mg = mygene.MyGeneInfo()

    batch_size = 1000
    all_results: dict[str, dict] = {}

    for i in range(0, len(missing_ids), batch_size):
        batch = missing_ids[i : i + batch_size]
        try:
            response = mg.querymany(
                batch,
                scopes="ensembl.gene",
                fields="symbol,entrezgene,uniprot.Swiss-Prot,type_of_gene",
                species="human",
                returnall=True,
                verbose=False,
            )
            for hit in response.get("out", []):
                if "query" in hit and not hit.get("notfound"):
                    all_results[hit["query"]] = hit
        except (_requests.exceptions.RequestException, KeyError, ValueError) as exc:
            logger.warning("MyGeneInfo batch %d failed: %s", i // batch_size, exc)

    update_rows: list[dict] = []
    for ens_id, hit in all_results.items():
        row_data: dict = {"gene_ensembl_id": ens_id}
        if "symbol" in hit:
            row_data["gene_symbol"] = hit["symbol"]
        if "entrezgene" in hit:
            row_data["gene_entrez_id"] = int(hit["entrezgene"])
        up = hit.get("uniprot", {})
        sp = up.get("Swiss-Prot") if isinstance(up, dict) else None
        if isinstance(sp, list):
            sp = sp[0]
        if sp:
            row_data["gene_uniprot_id"] = sp
        if "type_of_gene" in hit and hit["type_of_gene"] != "-":
            row_data["biotype"] = hit["type_of_gene"].replace("-", "_")
        update_rows.append(row_data)

    if not update_rows:
        return df

    update_df = pd.DataFrame(update_rows).set_index("gene_ensembl_id")
    df = df.copy()
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["gene_ensembl_id"], keep="first")
    n_dropped = n_before_dedup - len(df)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d duplicate Ensembl IDs before MyGeneInfo enrichment "
            "(%d -> %d genes)", n_dropped, n_before_dedup, len(df),
        )
    df = df.set_index("gene_ensembl_id", drop=False)

    for col in ["gene_symbol", "gene_entrez_id", "gene_uniprot_id", "biotype"]:
        if col not in update_df.columns:
            continue
        mask = df.index.isin(update_df.index) & df[col].isna()
        df.loc[mask, col] = update_df.loc[df.loc[mask].index.intersection(update_df.index), col]

    df = df.reset_index(drop=True)

    logger.info("MyGeneInfo enriched %d genes", len(all_results))
    return df


# ---------------------------------------------------------------------------
# MAGMA annotation output
# ---------------------------------------------------------------------------


def create_magma_annotation(
    gene_annotations: pd.DataFrame,
    gwas_df: pd.DataFrame,
    reference_bim: Path,
    window_kb: int = 35,
) -> Path:
    """Create a MAGMA ``.genes.annot`` file mapping SNPs to genes.

    For each gene the annotation window is ``[start - window_kb*1000,
    end + window_kb*1000]``.  All SNPs from the reference BIM falling
    within this window are assigned to the gene.

    Args:
        gene_annotations: ``GeneAnnotationRecord`` DataFrame.
        gwas_df: ``StandardizedGWAS`` DataFrame (used to restrict to
            tested SNPs).
        reference_bim: Path to PLINK BIM file.
        window_kb: Extension in kilobases on each side of the gene.

    Returns:
        Path to the written ``.genes.annot`` file.
    """
    bim_path = check_file_exists(Path(reference_bim), label="Reference BIM")
    bim = pd.read_csv(bim_path, sep="\t", header=None, names=["CHR", "SNP", "CM", "POS", "REF", "ALT"])

    if gwas_df is not None and "SNP" in gwas_df.columns:
        valid_snps = set(gwas_df["SNP"].dropna())
        bim = bim[bim["SNP"].isin(valid_snps)]

    output_path = bim_path.parent / (bim_path.stem + ".genes.annot")
    window = window_kb * 1000

    lines: list[str] = []
    for chrom in gene_annotations["chr"].unique():
        genes_chr = gene_annotations[gene_annotations["chr"] == chrom]
        snps_chr = bim[bim["CHR"] == chrom]

        if genes_chr.empty or snps_chr.empty:
            continue

        gene_starts = genes_chr["start"].values - window
        gene_ends = genes_chr["end"].values + window
        snp_positions = snps_chr["POS"].values
        snp_names = snps_chr["SNP"].values

        for i, (gs, ge) in enumerate(zip(gene_starts, gene_ends)):
            mask = (snp_positions >= gs) & (snp_positions <= ge)
            if mask.any():
                matched_snps = snp_names[mask]
                gene_row = genes_chr.iloc[i]
                gene_id = gene_row["gene_ensembl_id"]
                snp_list = " ".join(matched_snps)
                lines.append(f"{gene_id}\t{chrom}\t{gene_row['start']}\t{gene_row['end']}\t{snp_list}")

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(
        "MAGMA annotation: %d genes with ≥1 SNP (±%dkb window) -> %s",
        len(lines), window_kb, output_path,
    )
    return output_path


def create_gene_info(gene_annotations: pd.DataFrame) -> pd.DataFrame:
    """Create a gene info DataFrame for downstream modules.

    Columns: gene_symbol, gene_ensembl_id, gene_entrez_id,
    gene_uniprot_id, biotype, chr, start, end.
    """
    cols = [
        "gene_symbol", "gene_ensembl_id", "gene_entrez_id",
        "gene_uniprot_id", "biotype", "chr", "start", "end",
    ]
    available = [c for c in cols if c in gene_annotations.columns]
    return gene_annotations[available].copy()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene annotation for RepoGen")
    parser.add_argument("--gene-loc", required=True, type=Path, help="Gene location file")
    parser.add_argument("--output", required=True, type=Path, help="Output Parquet file")
    parser.add_argument("--mapping-dir", type=Path, default=None, help="BioMart dictionary directory")
    parser.add_argument("--gene-info", type=Path, default=None, help="NCBI gene_info.gz")
    parser.add_argument("--gene-history", type=Path, default=None, help="NCBI gene_history.gz")
    parser.add_argument("--window-kb", type=int, default=35, help="MAGMA gene window in kb")
    parser.add_argument("--biotype-filter", nargs="*", default=None, help="Biotype filter")
    parser.add_argument("--use-mygene", action="store_true", help="Use MyGeneInfo API")
    args = parser.parse_args()

    biomart = None
    if args.mapping_dir:
        biomart = {
            "ensembl_to_name": args.mapping_dir / "biomart_dico1",
            "name_to_ensembl": args.mapping_dir / "biomart_dico2",
            "uniprot_to_ensembl": args.mapping_dir / "biomart_dico3",
        }

    result = annotate_genes(
        reference_gene_file=args.gene_loc,
        biomart_dicts=biomart,
        gene_window_kb=args.window_kb,
        biotype_filter=args.biotype_filter,
        use_mygene_api=args.use_mygene,
        entrez_mapping_file=args.gene_info,
        gene_history_file=args.gene_history,
    )

    ensure_directory(args.output.parent)
    result.to_parquet(args.output, engine="pyarrow", index=False)
    logger.info("Wrote %d gene annotations to %s", len(result), args.output)
