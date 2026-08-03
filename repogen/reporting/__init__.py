"""Result aggregation and export modules."""

from repogen.reporting.combine_results import (
    RESULT_FILE_MAP,
    CombinedResults,
    combine_results,
)
from repogen.reporting.export import export_combined, export_results
from repogen.reporting.html_report import generate_html_report

__all__ = [
    "CombinedResults",
    "RESULT_FILE_MAP",
    "combine_results",
    "export_combined",
    "export_results",
    "generate_html_report",
]
