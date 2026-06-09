"""Analysis utilities for diagnostics and intersurvey comparisons."""

from .diagnostics import (
    analyze_intersurvey_correlations,
    prepare_mock_similarity_data,
    compute_mock_similarity_matrix,
    plot_covariance_correlation_matrix,
    _top_matrix_pairs,
    compute_chi2_diagnostic_tables,
    print_statistics_summary,
    print_physical_interpretation,
)

__all__ = [
    "analyze_intersurvey_correlations",
    "prepare_mock_similarity_data",
    "compute_mock_similarity_matrix",
    "plot_covariance_correlation_matrix",
    "_top_matrix_pairs",
    "compute_chi2_diagnostic_tables",
    "print_statistics_summary",
    "print_physical_interpretation",
]
