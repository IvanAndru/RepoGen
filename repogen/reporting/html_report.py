"""RepoGen reporting - HTML report with conditional sections."""

from __future__ import annotations

import argparse
import base64
import html as html_mod
import importlib.metadata
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2
import pandas as pd

from repogen.reporting.combine_results import CombinedResults
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

_FIGURE_FILENAMES: dict[str, list[str]] = {
    "gene": ["manhattan.png", "qq.png", "volcano.png"],
    "pathway": ["pathway_enrichment.png"],
    "drug": ["drug_enrichment.png"],
    "atc": ["atc_enrichment.png"],
    "spredixcan": ["tissue_signature.png"],
    "correlation": ["correlation_scatter.png", "correlation_heatmap.png"],
    "mr": ["mr_forest.png", "coloc_posteriors.png", "mr_drug_summary.png"],
    "convergence": ["convergence.png"],
}


def generate_html_report(
    combined: CombinedResults,
    plot_dir: Path,
    output_path: Path,
    template_dir: Path | None = None,
) -> Path:
    """Render HTML report with conditional sections based on available results.

    Args:
        combined: CombinedResults from combine_results.py.
        plot_dir: Directory containing pre-rendered PNG figures.
        output_path: Path to write the HTML report file.
        template_dir: Optional custom Jinja2 template directory.
            If None, uses the built-in template.

    Returns:
        Path to the written HTML report.

    Raises:
        FileNotFoundError: If *plot_dir* does not exist.
    """
    plot_dir = Path(plot_dir)
    if not plot_dir.is_dir():
        raise FileNotFoundError(f"Plot directory not found: {plot_dir}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    context = _build_context(combined, plot_dir)

    if template_dir is not None:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        template = env.get_template("report.html")
    else:
        template = jinja2.Template(_REPORT_TEMPLATE, autoescape=True)

    html = template.render(**context)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)
    return output_path


def _filter_drug_to_headline(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Filter the drug-enrichment dataframe to headline rows.

    Idempotent and graceful: if the column is absent (e.g. legacy parquet),
    returns ``df`` unchanged.  Mirrors ``export._filter_to_headline``.
    """
    if df is None or "passes_headline_min_genes" not in df.columns:
        return df
    mask = df["passes_headline_min_genes"].fillna(False).astype(bool)
    return df.loc[mask].copy()


def _build_context(combined: CombinedResults, plot_dir: Path) -> dict[str, Any]:
    """Prepare the template context dict."""
    drug_enrichment_headline = _filter_drug_to_headline(combined.drug_enrichment)

    gene_table = (
        _render_table(
            combined.gene_results,
            columns=["gene_symbol", "chr", "magma_z", "magma_p", "fdr_q", "biotype"],
            sort_by="magma_p",
            ascending=True,
        )
        if combined.gene_results is not None
        else None
    )
    pathway_table = (
        _render_table(
            combined.pathway_results,
            columns=["pathway_name", "source_db", "p_value", "fdr_q", "n_genes_in_set"],
            sort_by="fdr_q",
            ascending=True,
        )
        if combined.pathway_results is not None
        else None
    )
    drug_table = (
        _render_table(
            drug_enrichment_headline,
            columns=["drug_name", "magma_fdr_q", "max_phase", "magma_beta"],
            sort_by="magma_fdr_q",
            ascending=True,
        )
        if drug_enrichment_headline is not None
        else None
    )
    atc_table = (
        _render_table(
            combined.atc_enrichment,
            columns=["atc_description", "atc_level", "gls_fdr", "n_drugs"],
            sort_by="gls_fdr",
            ascending=True,
        )
        if combined.atc_enrichment is not None
        else None
    )
    spredixcan_table = (
        _render_table(
            combined.spredixcan_meta,
            columns=["gene_symbol", "meta_zscore", "meta_pvalue"],
            sort_by="meta_pvalue",
            ascending=True,
        )
        if combined.spredixcan_meta is not None
        else None
    )
    neg_corr_table = (
        _render_table(
            combined.neg_correlation_summary,
            columns=[
                "drug_name",
                "best_spearman_rho",
                "best_tissue",
                "n_tissues_fdr_significant",
            ],
            sort_by="best_spearman_rho",
            ascending=True,
        )
        if combined.neg_correlation_summary is not None
        else None
    )
    mr_table = (
        _render_table(
            combined.mr_results,
            columns=[
                "gene_symbol",
                "eqtl_source",
                "mr_beta",
                "mr_pval",
                "confidence_tier",
                "n_instruments",
            ],
            sort_by="mr_pval",
            ascending=True,
        )
        if combined.mr_results is not None
        else None
    )
    mr_drugs_table = (
        _render_table(
            combined.mr_drug_matches,
            columns=[
                "gene_symbol",
                "drug_name",
                "confidence_tier",
                "direction_concordant",
            ],
        )
        if combined.mr_drug_matches is not None
        else None
    )
    drug_overlap_table = (
        _render_table(
            combined.drug_overlap,
            max_rows=0,
            sort_by="n_branches",
            ascending=False,
        )
        if combined.drug_overlap is not None and len(combined.drug_overlap) > 0
        else None
    )

    gene_count = _count_summary(combined.gene_results, "fdr_q", "lt", 0.05) if combined.gene_results is not None else None
    pathway_count = _count_summary(combined.pathway_results, "fdr_q", "lt", 0.05) if combined.pathway_results is not None else None
    drug_count = _count_summary(drug_enrichment_headline, "magma_fdr_q", "lt", 0.05) if drug_enrichment_headline is not None else None

    return {
        "study_name": combined.study_name,
        "branches_csv": ",".join(combined.branches_present),
        "branches_present": combined.branches_present,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "metadata": combined.metadata,
        "gene_results": combined.gene_results,
        "pathway_results": combined.pathway_results,
        "drug_enrichment": drug_enrichment_headline,
        "atc_enrichment": combined.atc_enrichment,
        "spredixcan_meta": combined.spredixcan_meta,
        "neg_correlation_summary": combined.neg_correlation_summary,
        "mr_results": combined.mr_results,
        "mr_drug_matches": combined.mr_drug_matches,
        "drug_overlap": combined.drug_overlap,
        "gene_table": gene_table,
        "pathway_table": pathway_table,
        "drug_table": drug_table,
        "atc_table": atc_table,
        "spredixcan_table": spredixcan_table,
        "neg_corr_table": neg_corr_table,
        "mr_table": mr_table,
        "mr_drugs_table": mr_drugs_table,
        "drug_overlap_table": drug_overlap_table,
        "gene_count": gene_count,
        "pathway_count": pathway_count,
        "drug_count": drug_count,
        "fig_manhattan": _embed_figure(plot_dir, "manhattan.png"),
        "fig_qq": _embed_figure(plot_dir, "qq.png"),
        "fig_volcano": _embed_figure(plot_dir, "volcano.png"),
        "fig_pathway_enrichment": _embed_figure(plot_dir, "pathway_enrichment.png"),
        "fig_drug_enrichment": _embed_figure(plot_dir, "drug_enrichment.png"),
        "fig_atc_enrichment": _embed_figure(plot_dir, "atc_enrichment.png"),
        "fig_tissue_signature": _embed_figure(plot_dir, "tissue_signature.png"),
        "fig_correlation_scatter": _embed_figure(plot_dir, "correlation_scatter.png"),
        "fig_correlation_heatmap": _embed_figure(plot_dir, "correlation_heatmap.png"),
        "fig_mr_forest": _embed_figure(plot_dir, "mr_forest.png"),
        "fig_coloc_posteriors": _embed_figure(plot_dir, "coloc_posteriors.png"),
        "fig_mr_drug_summary": _embed_figure(plot_dir, "mr_drug_summary.png"),
        "fig_convergence": _embed_figure(plot_dir, "convergence.png"),
        "repogen_version": _get_version(),
    }


def _count_summary(
    df: pd.DataFrame, col: str, op: str, val: float
) -> dict[str, int] | None:
    """Return {n_tested, n_significant} counts."""
    if col not in df.columns:
        return None
    n = len(df)
    if op == "lt":
        n_sig = int((df[col] < val).sum())
    elif op == "gt":
        n_sig = int((df[col] > val).sum())
    else:
        n_sig = 0
    return {"n_tested": n, "n_significant": n_sig}


_MAX_SVG_BYTES = 2 * 1024 * 1024


def _embed_figure(plot_dir: Path, filename: str) -> str | None:
    """Read a figure and return a base64 data URI string.

    Prefers SVG if a same-stem .svg exists and is <= 2 MB (vector output renders
    crisply at any zoom). Falls back to the existing PNG path.

    Returns None when neither format exists.

    Note: text inside <img src="data:image/svg+xml;..."> is not user-selectable
    (browser rasterises it). The standalone .svg file on disk still has selectable
    text for Illustrator/Inkscape use.
    """
    png_path = plot_dir / filename
    svg_path = png_path.with_suffix(".svg")

    if svg_path.is_file() and svg_path.stat().st_size <= _MAX_SVG_BYTES:
        data = svg_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"

    if not png_path.is_file():
        logger.debug("Figure not found, skipping: %s", png_path)
        return None
    data = png_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_table(
    df: pd.DataFrame,
    max_rows: int = 20,
    columns: list[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
) -> str:
    """Render a DataFrame as an HTML table string.

    Optionally sorts by *sort_by* column before taking the top
    *max_rows* rows.  Set *max_rows* to 0 to render all rows.
    Formats p-values in scientific notation and rounds other floats
    to 4 decimal places.
    """
    work = df
    if sort_by is not None and sort_by in df.columns:
        work = df.sort_values(sort_by, ascending=ascending, na_position="last")

    if columns is not None:
        avail = [c for c in columns if c in work.columns]
        sub = work[avail]
    else:
        sub = work

    if max_rows > 0:
        sub = sub.head(max_rows)

    p_keywords = ("_p", "_pval", "pvalue", "fdr", "p_value")

    rows_html: list[str] = []

    header_cells = "".join(f"<th>{_format_col_name(c)}</th>" for c in sub.columns)
    rows_html.append(f"<thead><tr>{header_cells}</tr></thead>")

    rows_html.append("<tbody>")
    for row in sub.itertuples(index=False):
        cells = []
        for col_name, val in zip(sub.columns, row):
            cells.append(f"<td>{_format_cell(val, col_name, p_keywords)}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    rows_html.append("</tbody>")

    return f'<table class="data-table">{"".join(rows_html)}</table>'


def _format_col_name(col: str) -> str:
    """Human-readable column name."""
    return html_mod.escape(col.replace("_", " ").title())


def _format_cell(val: Any, col_name: str, p_keywords: tuple[str, ...]) -> str:
    """Format a single cell value for HTML display.

    Dynamic data values are HTML-escaped to prevent markup injection
    when tables are inserted via Jinja2 ``|safe``.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "&mdash;"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, (list, tuple)):
        return ", ".join(html_mod.escape(str(v), quote=True) for v in val)
    if isinstance(val, float):
        is_p = any(kw in col_name.lower() for kw in p_keywords)
        if is_p:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return html_mod.escape(str(val), quote=True)


def _get_version() -> str:
    """Get RepoGen package version."""
    try:
        return importlib.metadata.version("repogen")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Inline Jinja2 template
# ---------------------------------------------------------------------------

_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RepoGen Report &mdash; {{ study_name }}</title>
    <style>
        :root {
            --branch-a: #E69F00;
            --branch-b: #56B4E9;
            --branch-c: #009E73;
            --text-primary: #1a1a2e;
            --text-secondary: #555;
            --bg-main: #ffffff;
            --bg-alt: #f8f9fa;
            --border: #dee2e6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Arial, Helvetica, sans-serif;
            color: var(--text-primary);
            background: var(--bg-main);
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }
        h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
        h2 {
            font-size: 1.4rem;
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid var(--border);
        }
        h3 { font-size: 1.1rem; margin-top: 1.2rem; margin-bottom: 0.6rem; }
        p { margin-bottom: 0.8rem; }
        .meta { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem; }
        .badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 3px;
            font-size: 0.8rem;
            font-weight: bold;
            color: #fff;
            margin-right: 0.4rem;
        }
        .badge-magma { background: var(--branch-a); }
        .badge-neg_correlation { background: var(--branch-b); }
        .badge-mr { background: var(--branch-c); }
        .figure-container {
            margin: 1rem 0;
            text-align: center;
        }
        .figure-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border);
            border-radius: 4px;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.85rem;
        }
        .data-table th {
            background: var(--bg-alt);
            border-bottom: 2px solid var(--border);
            padding: 0.5rem 0.6rem;
            text-align: left;
            position: sticky;
            top: 0;
        }
        .data-table td {
            padding: 0.4rem 0.6rem;
            border-bottom: 1px solid var(--border);
        }
        .data-table tbody tr:nth-child(even) { background: var(--bg-alt); }
        .data-table tbody tr:hover { background: #e8f4f8; }
        .summary-stat {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        section { margin-bottom: 2rem; }
        footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.8rem;
        }
        footer p { margin-bottom: 0.3rem; }
        @media print {
            body { max-width: 100%; padding: 1rem; }
            .data-table th { position: static; }
            section { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div id="repogen-report" data-study="{{ study_name }}" data-branches="{{ branches_csv }}">

        <section id="header">
            <h1>RepoGen Report &mdash; {{ study_name }}</h1>
            <div class="meta">
                <p>Generated: {{ timestamp }}</p>
                <p>Branches:
                {% for branch in branches_present %}
                    <span class="badge badge-{{ branch }}">{{ branch }}</span>
                {% endfor %}
                </p>
            </div>
        </section>

        {% if gene_results is not none %}
        <section id="magma-genes" data-result-type="gene">
            <h2>MAGMA Gene Results</h2>
            {% if gene_count %}
            <p class="summary-stat">{{ gene_count.n_tested }} genes tested, {{ gene_count.n_significant }} significant at FDR &lt; 0.05</p>
            {% endif %}
            {% if fig_manhattan %}
            <div class="figure-container"><img src="{{ fig_manhattan }}" alt="Manhattan plot"></div>
            {% endif %}
            {% if fig_qq %}
            <div class="figure-container"><img src="{{ fig_qq }}" alt="QQ plot"></div>
            {% endif %}
            {% if fig_volcano %}
            <div class="figure-container"><img src="{{ fig_volcano }}" alt="Volcano plot"></div>
            {% endif %}
            {% if gene_table %}
            <h3>Top Genes</h3>
            {{ gene_table | safe }}
            {% endif %}
        </section>
        {% endif %}

        {% if pathway_results is not none %}
        <section id="pathways" data-result-type="pathway">
            <h2>Pathway Enrichment</h2>
            {% if pathway_count %}
            <p class="summary-stat">{{ pathway_count.n_tested }} pathways tested, {{ pathway_count.n_significant }} significant at FDR &lt; 0.05</p>
            {% endif %}
            {% if fig_pathway_enrichment %}
            <div class="figure-container"><img src="{{ fig_pathway_enrichment }}" alt="Pathway enrichment"></div>
            {% endif %}
            {% if pathway_table %}
            <h3>Top Pathways</h3>
            {{ pathway_table | safe }}
            {% endif %}
        </section>
        {% endif %}

        {% if drug_enrichment is not none %}
        <section id="drug-enrichment" data-result-type="drug">
            <h2>Drug Enrichment</h2>
            {% if drug_count %}
            <p class="summary-stat">{{ drug_count.n_tested }} drugs tested, {{ drug_count.n_significant }} significant at FDR &lt; 0.05</p>
            {% endif %}
            {% if fig_drug_enrichment %}
            <div class="figure-container"><img src="{{ fig_drug_enrichment }}" alt="Drug enrichment"></div>
            {% endif %}
            {% if drug_table %}
            <h3>Top Drugs</h3>
            {{ drug_table | safe }}
            {% endif %}
        </section>
        {% endif %}

        {% if atc_enrichment is not none %}
        <section id="atc-enrichment" data-result-type="atc">
            <h2>ATC Class Enrichment</h2>
            {% if fig_atc_enrichment %}
            <div class="figure-container"><img src="{{ fig_atc_enrichment }}" alt="ATC enrichment"></div>
            {% endif %}
            {% if atc_table %}
            <h3>Top ATC Classes</h3>
            {{ atc_table | safe }}
            {% endif %}
        </section>
        {% endif %}

        {% if spredixcan_meta is not none %}
        <section id="spredixcan" data-result-type="spredixcan">
            <h2>S-PrediXcan</h2>
            {% if fig_tissue_signature %}
            <div class="figure-container"><img src="{{ fig_tissue_signature }}" alt="Tissue signature"></div>
            {% endif %}
            {% if spredixcan_table %}
            <h3>Top Genes (Meta-Analysis)</h3>
            {{ spredixcan_table | safe }}
            {% endif %}
        </section>
        {% endif %}

        {% if neg_correlation_summary is not none %}
        <section id="neg-correlation" data-result-type="correlation">
            <h2>Negative Correlation (Drug-Disease Signature Matching)</h2>
            {% if fig_correlation_scatter %}
            <div class="figure-container"><img src="{{ fig_correlation_scatter }}" alt="Correlation scatter"></div>
            {% endif %}
            {% if fig_correlation_heatmap %}
            <div class="figure-container"><img src="{{ fig_correlation_heatmap }}" alt="Correlation heatmap"></div>
            {% endif %}
            {% if neg_corr_table %}
            <h3>Top Candidates</h3>
            {{ neg_corr_table | safe }}
            {% endif %}
        </section>
        {% endif %}

        {% if mr_results is not none %}
        <section id="mendelian-randomisation" data-result-type="mr">
            <h2>Mendelian Randomisation</h2>
            {% if fig_mr_forest %}
            <div class="figure-container"><img src="{{ fig_mr_forest }}" alt="MR forest plot"></div>
            {% endif %}
            {% if fig_coloc_posteriors %}
            <div class="figure-container"><img src="{{ fig_coloc_posteriors }}" alt="Colocalisation posteriors"></div>
            {% endif %}
            {% if fig_mr_drug_summary %}
            <div class="figure-container"><img src="{{ fig_mr_drug_summary }}" alt="MR drug summary"></div>
            {% endif %}
            {% if mr_table %}
            <h3>Top MR Results</h3>
            {{ mr_table | safe }}
            {% endif %}
            {% if mr_drugs_table %}
            <h3>Drug Matches</h3>
            {{ mr_drugs_table | safe }}
            {% endif %}
        </section>
        {% endif %}

        {% if drug_overlap is not none and drug_overlap|length > 0 %}
        <section id="convergence" data-result-type="convergence">
            <h2>Cross-Branch Convergence</h2>
            {% if fig_convergence %}
            <div class="figure-container"><img src="{{ fig_convergence }}" alt="Convergence UpSet plot"></div>
            {% endif %}
            {% if drug_overlap_table %}
            <h3>Convergent Drugs</h3>
            {{ drug_overlap_table | safe }}
            {% endif %}
        </section>
        {% endif %}

        <section id="footer">
            <footer>
                <p><strong>RepoGen</strong> v{{ repogen_version }}</p>
                <p>Methods: MAGMA (de Leeuw et al. 2015), S-PrediXcan (Barbeira et al. 2018), LINCS L1000 (Subramanian et al. 2017), Mendelian Randomisation (Davey Smith &amp; Hemani 2014).</p>
                <p>Generated by RepoGen &mdash; {{ timestamp }}</p>
            </footer>
        </section>

    </div>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RepoGen HTML report.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--plot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()

    from repogen.reporting.combine_results import combine_results

    combined = combine_results(
        results_dir=args.results_dir,
        study_name=args.study_name,
    )
    generate_html_report(
        combined=combined,
        plot_dir=args.plot_dir,
        output_path=args.output,
    )
