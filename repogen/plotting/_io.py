"""I/O helpers for plotting - sign-lookup loader for volcano plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)


def load_gene_sign_lookup(
    annot_file: Path,
    prepared_gwas: Path,
) -> dict[int, float]:
    """Return {gene_entrez_id: +1.0 | -1.0} using the top-p SNP's beta sign per gene.

    Args:
        annot_file:    MAGMA .genes.annot - each non-comment line is
                       "ENTREZ\\tCHR:START:STOP\\trs1\\trs2\\t...".
        prepared_gwas: prepare_data/gwas_standardized.parquet - must contain
                       SNP rsID column (named 'SNP' or 'rsID'), an effect column
                       ('BETA' or log-transformed 'OR'), and a 'P' column.

    Tolerates missing columns and empty GWAS - returns {} rather than raising.
    """
    annot_file = Path(annot_file)
    prepared_gwas = Path(prepared_gwas)

    if not annot_file.is_file():
        logger.warning("Annot file missing: %s", annot_file)
        return {}
    if not prepared_gwas.is_file():
        logger.warning("Prepared GWAS file missing: %s", prepared_gwas)
        return {}

    # Parse annot
    rows: list[tuple[int, str]] = []
    with open(annot_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                entrez = int(parts[0])
            except ValueError:
                continue
            for rsid in parts[2:]:
                rsid = rsid.strip()
                if rsid:
                    rows.append((entrez, rsid))

    if not rows:
        logger.info("No gene-SNP pairs parsed from annot file")
        return {}

    annot_df = pd.DataFrame(rows, columns=["gene_entrez_id", "rsid"])

    # Load GWAS
    try:
        gwas = pd.read_parquet(prepared_gwas)
    except Exception as exc:
        logger.warning("Cannot read prepared GWAS: %s", exc)
        return {}

    snp_col = None
    for candidate in ("SNP", "rsID", "rsid", "snp"):
        if candidate in gwas.columns:
            snp_col = candidate
            break
    if snp_col is None:
        logger.info("No SNP/rsID column in prepared GWAS")
        return {}

    beta_col = None
    for candidate in ("BETA", "beta", "Beta"):
        if candidate in gwas.columns:
            beta_col = candidate
            break
    if beta_col is None:
        if "OR" in gwas.columns:
            gwas["_beta"] = np.log(gwas["OR"].astype(float))
            beta_col = "_beta"
        else:
            logger.info("No BETA/OR column in prepared GWAS")
            return {}

    p_col = None
    for candidate in ("P", "p", "p_value", "pval"):
        if candidate in gwas.columns:
            p_col = candidate
            break
    if p_col is None:
        logger.info("No P column in prepared GWAS")
        return {}

    gwas_slim = gwas[[snp_col, beta_col, p_col]].rename(
        columns={snp_col: "rsid", beta_col: "_beta", p_col: "_p"}
    ).dropna()

    merged = annot_df.merge(gwas_slim, on="rsid", how="inner")
    if merged.empty:
        logger.info("No overlap between annot SNPs and prepared GWAS")
        return {}

    # Per gene, pick the row with lowest p
    idx_min_p = merged.groupby("gene_entrez_id")["_p"].idxmin()
    best = merged.loc[idx_min_p]

    result: dict[int, float] = {}
    for row in best.itertuples():
        beta_val = float(row._beta)
        sign = float(np.sign(beta_val)) if beta_val != 0 else 1.0
        result[int(row.gene_entrez_id)] = sign

    logger.info("Sign lookup: %d genes with sign data from prepared GWAS", len(result))
    return result
