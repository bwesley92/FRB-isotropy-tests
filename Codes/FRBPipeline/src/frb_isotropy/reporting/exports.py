
from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.types import FloatArray
from ..core.config import RuntimeContext

from ..core.models import (
    JackknifeResult,
    TestStatistics,
)

from ..analysis.diagnostics import (
    _top_matrix_pairs,
    compute_chi2_diagnostic_tables,
)




def _first_not_none(*values: object) -> object | None:
    """Return the first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None

def export_covariance_eigenvalues(
    cov: FloatArray,
    output: str | Path,
) -> None:
    """
    Export positive covariance eigenvalues.

    Parameters
    ----------
    cov : FloatArray
        Covariance matrix.

    output : str | Path
        CSV output file.
    """

    cov = np.asarray(
        cov,
        dtype=float,
    )

    if cov.ndim != 2:

        raise ValueError(
            "cov must be 2-dimensional."
        )

    if cov.shape[0] != cov.shape[1]:

        raise ValueError(
            "cov must be square."
        )

    eigvals = np.linalg.eigvalsh(
        cov
    )

    eigvals_pos = eigvals[
        eigvals > 1e-15
    ]

    if len(eigvals_pos) == 0:

        raise ValueError(
            "No positive covariance "
            "eigenvalues found."
        )

    eigenvalues_sorted = np.sort(
        eigvals_pos
    )[::-1]

    pd.DataFrame(
        {
            "eigenvalue":
                eigenvalues_sorted
        }
    ).to_csv(
        Path(output),
        index=False,
    )


# ==============================================================================
# Reporting and export
# ==============================================================================


def _format_report_value(
    value: object,
    precision: int = 6,
) -> str:
    """
    Format values for human-readable reports.
    """

    # ------------------------------------------------------------------
    # NumPy scalars
    # ------------------------------------------------------------------

    if isinstance(
        value,
        np.generic,
    ):

        value = value.item()

    # ------------------------------------------------------------------
    # Boolean
    # IMPORTANT:
    # bool is subclass of int -> check BEFORE int
    # ------------------------------------------------------------------

    if isinstance(value, bool):

        return str(value)

    # ------------------------------------------------------------------
    # Floating-point
    # ------------------------------------------------------------------

    if isinstance(
        value,
        float,
    ):

        if np.isfinite(value):

            return (
                f"{value:.{precision}g}"
            )

        return str(value)

    # ------------------------------------------------------------------
    # Integer
    # ------------------------------------------------------------------

    if isinstance(
        value,
        int,
    ):

        return str(value)

    # ------------------------------------------------------------------
    # Arrays
    # ------------------------------------------------------------------

    if isinstance(
        value,
        np.ndarray,
    ):

        return np.array2string(
            value,
            precision=precision,
            threshold=10,
        )

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    return str(value)


def _append_key_values(
    lines: list[str],
    title: str,
    values: dict[str, object],
    precision: int = 6,
) -> None:
    """
    Append formatted key-value diagnostics block.
    """

    if len(values) == 0:

        return

    lines.append(
        f"## {title}"
    )

    lines.append("")

    max_key = max(
        len(str(key))
        for key in values
    )

    lines.append(
        "```text"
    )

    for key, value in values.items():

        formatted = _format_report_value(
            value,
            precision=precision,
        )

        lines.append(
            f"{str(key):<{max_key}} : {formatted}"
        )

    lines.append(
        "```"
    )

    lines.append("")


# ---------------------------------------------------------------------------
# Internal data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _ReportData:
    """Validated and normalized inputs extracted from the results dict."""

    stats:      TestStatistics
    jackknife:  JackknifeResult
    df_data:    pd.DataFrame
    all_w_h0:   FloatArray
    all_abs_h0: FloatArray

    # Optional diagnostics
    chi2_bin_diagnostics: pd.DataFrame | None
    chi2_svd_modes:       pd.DataFrame | None
    sf_validation:        pd.DataFrame | None
    corr_df:              pd.DataFrame | None
    overlap_df:           pd.DataFrame | None


# ---------------------------------------------------------------------------
# Report header
# ---------------------------------------------------------------------------

def _append_report_header(lines: list[str]) -> None:
    """Append report title and description."""
    lines.extend([
        "# FRB Isotropy Analysis Report",
        "",
        "Consolidated output of the isotropy pipeline.",
        "",
    ])


# ---------------------------------------------------------------------------
# Step 1 — extraction and validation
# ---------------------------------------------------------------------------

def _extract_report_data(
    results: dict[str, object],
) -> _ReportData:
    """Validate required keys and extract all entries from results."""

    required_keys = [
        "stats",
        "jackknife",
        "df_data",
        "all_w_h0",
        "all_abs_h0",
    ]
    missing = [key for key in required_keys if key not in results]
    if missing:
        raise KeyError(f"Missing required report entries: {missing}")

    return _ReportData(
        stats     = results["stats"],
        jackknife = results["jackknife"],
        df_data   = results["df_data"],
        all_w_h0  = np.asarray(results["all_w_h0"],   dtype=float),
        all_abs_h0= np.asarray(results["all_abs_h0"], dtype=float),
        chi2_bin_diagnostics = results.get("chi2_bin_diagnostics"),
        chi2_svd_modes       = _first_not_none(
            results.get("svd_mode_contributions"),
            results.get("chi2_svd_modes"),
        ),
        sf_validation        = results.get("sf_validation"),
        corr_df              = _first_not_none(
            results.get("corr_df"),
            results.get("intersurvey_corr"),
        ),
        overlap_df           = _first_not_none(
            results.get("overlap_df"),
            results.get("intersurvey_overlap"),
        ),
    )


# ---------------------------------------------------------------------------
# Step 2 — content builders
# ---------------------------------------------------------------------------

def _build_report_lines(
    context: RuntimeContext,
    data:    _ReportData,
    paths:   dict[str, str] | None,
) -> list[str]:
    """Assemble and return all markdown lines for the report."""

    lines: list[str] = []

    _append_report_header(lines)
    _append_run_config(lines, context, data)
    _append_full_covariance(lines, data)
    _append_svd_statistics(lines, data)
    _append_covariance_diagnostics(lines, data)
    _append_nonparametric_tests(lines, data)
    _append_absolute_anisotropy(lines, data)
    _append_jackknife_summary(lines, data)
    _append_optional_dataframe(
        lines, data.chi2_bin_diagnostics,
        "Chi-square Bin Audit", _CHI2_BIN_COLUMNS,
    )
    _append_optional_dataframe(
        lines, data.chi2_svd_modes,
        "Chi-square SVD Mode Audit", _SVD_MODE_COLUMNS,
    )
    _append_optional_dataframe(
        lines, data.sf_validation,
        "Selection Function Validation", None,
        sort_by="weight",
    )
    _append_optional_corr_matrix(
        lines, data.corr_df,
        "Intersurvey Correlations", absolute=True,
    )
    _append_optional_corr_matrix(
        lines, data.overlap_df,
        "Intersurvey Overlap", absolute=False,
    )
    _append_saved_files(lines, paths)

    return lines


# -- Section builders --------------------------------------------------------

def _append_run_config(
    lines:   list[str],
    context: RuntimeContext,
    data:    _ReportData,
) -> None:
    config = context.config
    stats  = data.stats
    _append_key_values(lines, "Run Configuration", {
        "run_tag":                config.run_tag,
        "catalog_path":           str(config.catalog_path),
        "catalog_size":           int(len(data.df_data)),
        "use_gal_mask":           bool(config.use_gal_mask),
        "use_selection_function": bool(config.use_sel_func),
        "gal_cut_deg":            config.gal_cut,
        "bin_size_deg":           config.bin_size,
        "n_bins":                 stats.n_bins,
        "n_rand_factor":          config.n_rand_factor,
        "n_mocks":                stats.n_mocks,
        "n_ensemble":             config.n_ensemble,
        "n_mocks_per_ensemble":   config.n_mocks_per_ensemble,
        "nside_sf":               config.nside_sf,
        "smooth_sigma_deg":       config.smooth_sigma,
        "perturbation_scale":     config.perturbation_scale,
        "nside_jackknife":        config.nside_jackknife,
        "svd_eigenvalue_cut":     stats.covariance.svd_eigenvalue_cut,
    })


def _append_full_covariance(lines: list[str], data: _ReportData) -> None:
    s = data.stats
    _append_key_values(lines, "Full Covariance Diagnostic", {
        "chi2":                     s.chi2.chi2,
        "chi2_red":                 s.chi2.chi2_red,
        "p_chi2_analytic":          s.chi2.p_chi2,
        "p_chi2_empirical":         s.chi2.p_empirical,
        "p_chi2_empirical_floor":   s.chi2.p_empirical_floor,
        "sigma_equiv_empirical":    s.chi2.sigma_equiv,
        "hartlap_factor":           s.covariance.hartlap_factor,
        "rms_normalized_deviation": s.absolute.global_tension,
    })


def _append_svd_statistics(lines: list[str], data: _ReportData) -> None:
    s = data.stats
    _append_key_values(lines, "Primary Isotropy Statistic (SVD-Regularized)", {
        "chi2_svd":               s.svd.chi2_svd,
        "chi2_svd_red":           s.svd.chi2_svd_red,
        "p_chi2_svd_analytic":    s.svd.p_chi2_svd,
        "p_chi2_svd_empirical":   s.svd.p_svd_empirical,
        "sigma_svd_empirical":    s.svd.sigma_svd_equiv,
        "svd_modes_kept":         f"{s.covariance.svd_modes_kept}/{s.n_bins}",
        "svd_retained_condition": s.covariance.svd_condition,
    })


def _append_covariance_diagnostics(lines: list[str], data: _ReportData) -> None:
    s = data.stats
    _append_key_values(lines, "Covariance Diagnostics", {
        "mocks":                s.n_mocks,
        "bins":                 s.n_bins,
        "covariance_rank":      f"{s.covariance.covariance_rank}/{s.n_bins}",
        "covariance_condition": s.covariance.covariance_condition,
        "effective_modes":      s.covariance.n_eff,
        "h0_w_shape":           tuple(data.all_w_h0.shape),
        "h0_abs_shape":         tuple(data.all_abs_h0.shape),
    })


def _append_nonparametric_tests(lines: list[str], data: _ReportData) -> None:
    np_ = data.stats.nonparametric
    _append_key_values(lines, "Non-parametric Profile Tests", {
        "w_ks_stat":          np_.ks_w_stat,
        "w_ks_pvalue":        np_.ks_w_pvalue,
        "w_ks_empirical_p":   np_.ks_w_empirical_p,
        "w_ad_stat":          np_.ad_w_stat,
        "w_ad_pvalue":        np_.ad_w_pvalue,
        "w_ad_empirical_p":   np_.ad_w_empirical_p,
        "abs_ks_stat":        np_.ks_abs_stat,
        "abs_ks_pvalue":      np_.ks_abs_pvalue,
        "abs_ks_empirical_p": np_.ks_abs_empirical_p,
        "abs_ad_stat":        np_.ad_abs_stat,
        "abs_ad_pvalue":      np_.ad_abs_pvalue,
        "abs_ad_empirical_p": np_.ad_abs_empirical_p,
    })


def _append_absolute_anisotropy(lines: list[str], data: _ReportData) -> None:
    _append_key_values(lines, "Absolute Anisotropy Amplitude", {
        "observed_rms_absolute_amplitude": data.stats.absolute.abs_observed_stat,
        "empirical_pvalue":                data.stats.absolute.abs_empirical_p,
    })


def _append_jackknife_summary(lines: list[str], data: _ReportData) -> None:
    jk = data.jackknife
    _append_key_values(lines, "Jackknife Summary", {
        "n_regions":        int(len(jk.regions)),
        "median_sigma_w":   float(np.nanmedian(jk.w_err)),
        "min_sigma_w":      float(np.nanmin(jk.w_err)),
        "max_sigma_w":      float(np.nanmax(jk.w_err)),
        "median_sigma_abs": float(np.nanmedian(jk.abs_err)),
        "min_sigma_abs":    float(np.nanmin(jk.abs_err)),
        "max_sigma_abs":    float(np.nanmax(jk.abs_err)),
    })


# -- Optional section helpers ------------------------------------------------

_CHI2_BIN_COLUMNS = [
    "bin_index", "theta_deg", "w_obs", "h0_mean", "h0_std",
    "delta", "pull", "diagonal_chi2_contribution",
]

_SVD_MODE_COLUMNS = [
    "mode_index", "eigenvalue", "relative_eigenvalue", "kept_by_svd_cut",
    "delta_projection", "raw_chi2_contribution", "svd_chi2_contribution",
    "fractional_svd_chi2_contribution", "dominant_theta_deg", "dominant_loading",
]


def _append_optional_dataframe(
    lines:   list[str],
    df:      pd.DataFrame | None,
    heading: str,
    columns: list[str] | None,
    *,
    sort_by: str | None = None,
    n:       int = 12,
) -> None:
    """Append a fenced code block for df if it is a non-empty DataFrame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]

    lines += [f"## {heading}", "", "```text",
              df.head(n).to_string(index=False), "```", ""]


def _append_optional_corr_matrix(
    lines:    list[str],
    df:       pd.DataFrame | None,
    heading:  str,
    *,
    absolute: bool,
    n:        int = 12,
) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    top = _top_matrix_pairs(df, n=n, absolute=absolute)
    lines += [f"## {heading}", "", "```text",
              top.to_string(index=False), "```", ""]


def _append_saved_files(
    lines: list[str],
    paths: dict[str, str] | None,
) -> None:
    if not paths:
        return
    lines += ["## Saved Files", "", "```text"]
    lines += [f"{name}: {path}" for name, path in paths.items()]
    lines += ["```", ""]


# ---------------------------------------------------------------------------
# Step 3 — write to disk
# ---------------------------------------------------------------------------

def _write_report(output_path: Path, lines: list[str]) -> None:
    """Create parent directories and write the report file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"\nSaved report: {output_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_report(
    context:     RuntimeContext,
    results:     dict[str, object],
    paths:       dict[str, str] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Save a consolidated human-readable analysis report."""
    output_path = Path(output_path or context.outputs.report_dir / "summary.md")
    data        = _extract_report_data(results)
    lines       = _build_report_lines(context, data, paths)
    _write_report(output_path, lines)
    return str(output_path)


# ===========================================================================
# save_tables — internal helpers
# ===========================================================================

# ---------------------------------------------------------------------------
# Internal data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _TableData:
    """Validated and normalized inputs extracted from the results dict."""

    stats:    TestStatistics
    jackknife: JackknifeResult
    theta:    FloatArray
    w_obs:    FloatArray
    abs_obs:  FloatArray
    all_w_h0: FloatArray

    # Optional
    sf_validation:     pd.DataFrame | None
    intersurvey_corr:  pd.DataFrame | None
    intersurvey_overlap: pd.DataFrame | None
    covariance_matrix: FloatArray | None


# ---------------------------------------------------------------------------
# Step 1 — resolve paths
# ---------------------------------------------------------------------------

def _resolve_table_paths(
    context: RuntimeContext,
    prefix:  str,
) -> dict[str, Path]:
    """Create output directories and return the full path mapping."""

    tables = context.outputs.tables_dir
    report = context.outputs.report_dir

    tables.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)

    return {
        "sf_validation":        tables / f"{prefix}_sf_validation.csv",
        "intersurvey_corr":     tables / f"{prefix}_intersurvey_corr.csv",
        "intersurvey_overlap":  tables / f"{prefix}_intersurvey_overlap.csv",
        "stats_summary":        tables / f"{prefix}_stats_summary.csv",
        "jackknife_w_errors":   tables / f"{prefix}_jackknife_w_errors.csv",
        "jackknife_abs_errors": tables / f"{prefix}_jackknife_abs_errors.csv",
        "covariance_matrix":    tables / f"{prefix}_covariance_matrix.csv",
        "chi2_bin_diagnostics": tables / f"{prefix}_chi2_bin_diagnostics.csv",
        "svd_mode_contributions": tables / f"{prefix}_svd_mode_contributions.csv",
        "report":               report / f"{prefix}_summary.md",
    }


# ---------------------------------------------------------------------------
# Step 2 — extraction and validation
# ---------------------------------------------------------------------------

def _extract_table_data(
    results: dict[str, object],
) -> _TableData:
    """Validate required keys and extract all entries from results."""

    required_keys = ["stats", "jackknife", "theta", "w_obs", "abs_obs", "all_w_h0"]
    missing = [k for k in required_keys if k not in results]
    if missing:
        raise KeyError(f"Missing required results entries: {missing}")

    cov = results.get("covariance_matrix")

    return _TableData(
        stats     = results["stats"],
        jackknife = results["jackknife"],
        theta     = np.asarray(results["theta"],    dtype=float),
        w_obs     = np.asarray(results["w_obs"],    dtype=float),
        abs_obs   = np.asarray(results["abs_obs"],  dtype=float),
        all_w_h0  = np.asarray(results["all_w_h0"], dtype=float),
        sf_validation      = results.get("sf_validation"),
        intersurvey_corr   = _first_not_none(
            results.get("corr_df"),
            results.get("intersurvey_corr"),
        ),
        intersurvey_overlap= _first_not_none(
            results.get("overlap_df"),
            results.get("intersurvey_overlap"),
        ),
        covariance_matrix  = np.asarray(cov, dtype=float) if cov is not None else None,
    )


# ---------------------------------------------------------------------------
# Step 3 — individual table savers
# ---------------------------------------------------------------------------

def _save_optional_csv(
    df:   pd.DataFrame | None,
    path: Path,
    *,
    index: bool = False,
) -> None:
    """Save df to CSV only if it is a non-empty DataFrame."""
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(path, index=index)


def _save_stats_summary(
    stats: TestStatistics,
    path:  Path,
) -> None:
    rows = [
        ("chi2",       "chi2",             stats.chi2.chi2),
        ("chi2",       "chi2_red",         stats.chi2.chi2_red),
        ("chi2",       "p_empirical",      stats.chi2.p_empirical),
        ("svd",        "chi2_svd",         stats.svd.chi2_svd),
        ("svd",        "chi2_svd_red",     stats.svd.chi2_svd_red),
        ("svd",        "p_svd_empirical",  stats.svd.p_svd_empirical),
        ("covariance", "effective_modes",  stats.covariance.n_eff),
        ("covariance", "condition",        stats.covariance.covariance_condition),
        ("anisotropy", "abs_stat",         stats.absolute.abs_observed_stat),
        ("anisotropy", "abs_p",            stats.absolute.abs_empirical_p),
    ]
    pd.DataFrame(rows, columns=["category", "statistic", "value"]).to_csv(
        path, index=False,
    )


def _save_jackknife_tables(
    context: RuntimeContext,
    data:    _TableData,
    paths:   dict[str, Path],
) -> None:
    jk = data.jackknife

    pd.DataFrame({
        "theta_deg":          data.theta,
        "w_obs":              data.w_obs,
        "w_jackknife_mean":   np.asarray(jk.w_mean,  dtype=float),
        "w_jackknife_err":    np.asarray(jk.w_err,   dtype=float),
    }).to_csv(paths["jackknife_w_errors"], index=False)

    pd.DataFrame({
        "theta_center_deg":   np.asarray(context.config.coarse_centers, dtype=float),
        "abs_obs":            data.abs_obs,
        "abs_jackknife_mean": np.asarray(jk.abs_mean, dtype=float),
        "abs_jackknife_err":  np.asarray(jk.abs_err,  dtype=float),
    }).to_csv(paths["jackknife_abs_errors"], index=False)


def _save_chi2_diagnostics(
    context: RuntimeContext,
    data:    _TableData,
    paths:   dict[str, Path],
    results: dict[str, object],
) -> None:
    """Compute, persist back into results, and save chi2 diagnostic tables."""
    chi2_bin_diagnostics, svd_mode_contributions = compute_chi2_diagnostic_tables(
        context=context,
        theta=data.theta,
        w_obs=data.w_obs,
        all_w_h0=data.all_w_h0,
        eigenvalue_cut=context.config.svd_eigenvalue_cut,
    )

    results["chi2_bin_diagnostics"]    = chi2_bin_diagnostics
    results["svd_mode_contributions"]  = svd_mode_contributions

    chi2_bin_diagnostics.to_csv(paths["chi2_bin_diagnostics"],    index=False)
    svd_mode_contributions.to_csv(paths["svd_mode_contributions"], index=False)


def _save_covariance_matrix(
    cov:  FloatArray | None,
    path: Path,
) -> None:
    if cov is not None:
        pd.DataFrame(cov).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Step 4 — orchestrate all table saves
# ---------------------------------------------------------------------------

def _save_all_tables(
    context: RuntimeContext,
    data:    _TableData,
    paths:   dict[str, Path],
    results: dict[str, object],
) -> None:
    _save_optional_csv(data.sf_validation,      paths["sf_validation"])
    _save_optional_csv(data.intersurvey_corr,   paths["intersurvey_corr"],  index=True)
    _save_optional_csv(data.intersurvey_overlap, paths["intersurvey_overlap"], index=True)
    _save_stats_summary(data.stats,             paths["stats_summary"])
    _save_jackknife_tables(context, data,        paths)
    _save_chi2_diagnostics(context, data,        paths, results)
    _save_covariance_matrix(data.covariance_matrix, paths["covariance_matrix"])


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_table_summary(paths: dict[str, Path]) -> None:
    print("\n--- Saved analysis tables ---")
    for name, path in paths.items():
        print(f"{name}: {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_tables(
    context: RuntimeContext,
    results: dict[str, object],
    prefix:  str | None = None,
) -> dict[str, str]:
    """Save analysis tables, diagnostics, and markdown report."""
    prefix = str(prefix or context.config.run_tag)
    paths  = _resolve_table_paths(context, prefix)
    data   = _extract_table_data(results)

    _save_all_tables(context, data, paths, results)

    save_report(
        context=context,
        results=results,
        paths={k: str(v) for k, v in paths.items()},
        output_path=paths["report"],
    )

    _print_table_summary(paths)

    return {k: str(v) for k, v in paths.items()}


