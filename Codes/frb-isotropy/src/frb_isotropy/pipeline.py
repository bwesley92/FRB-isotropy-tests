from __future__ import annotations

import numpy as np

from .core.config import (
    AnalysisConfig,
    create_runtime_context,
)

from .core.runtime_logging import (
    log_runtime_configuration,
)

from .catalog.catalog import load_catalog, apply_mask

from .selection.selection import (
    split_by_survey,
    build_survey_selection_functions,
    SelectionFunctionValidator,
)

from .analysis.diagnostics import analyze_intersurvey_correlations
from .simulations.mocks import generate_random_catalog, run_ensemble_mocks
from .statistics.estimators import compute_2pacf, get_absolute_sum
from .simulations.resampling import run_jackknife_errors, run_bootstrap_errors
from .statistics.inference import compute_statistics

from .visualization.plotting import (
    plot_top_survey_maps,
    plot_results_with_jackknife,
    plot_results_with_bootstrap,
    plot_results_jk_vs_bootstrap,
)

from .reporting.exports import save_tables
from .core.types import FloatArray


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    """Print a visible pipeline section header."""
    print(f"\n{'='*50}\n{title}\n{'='*50}")


def _filter_valid_h0(
    all_w_h0:   FloatArray,
    all_abs_h0: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Drop mock rows that contain non-finite values in either array."""
    valid = (
        np.all(np.isfinite(all_w_h0),   axis=1)
        & np.all(np.isfinite(all_abs_h0), axis=1)
    )
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"\nWARNING: dropping {n_dropped} invalid H0 mock(s).")
    return all_w_h0[valid], all_abs_h0[valid]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def main(
    config:       AnalysisConfig,
    save_outputs: bool = True,
) -> dict[str, object]:
    """
    Execute the complete FRB isotropy analysis pipeline.

    Notes
    -----
    Physical H0 modes:

    use_sel_func=False
        Pure isotropic sky.

    use_sel_func=True
        Isotropy convolved with survey selection functions.
    """

    # ------------------------------------------------------------------
    # Runtime context
    # ------------------------------------------------------------------

    context = create_runtime_context(config)
    log_runtime_configuration(context)
    outputs = context.outputs

    _section("FRB ISOTROPY ANALYSIS PIPELINE")

    # ------------------------------------------------------------------
    # Load observed catalog
    # ------------------------------------------------------------------

    _section("LOAD OBSERVED CATALOG")

    df_data = load_catalog(context)
    df_data = apply_mask(context, df_data)

    print(f"Masked catalog size: {len(df_data)}")

    if len(df_data) < 10:
        raise RuntimeError("Catalog too small after masking.")

    # ------------------------------------------------------------------
    # Selection functions
    # ------------------------------------------------------------------

    _section("SELECTION FUNCTIONS")

    selection_functions = build_survey_selection_functions(
        context=context,
        df_data=df_data,
        nside=config.nside_sf,
        smooth_sigma=config.smooth_sigma,
    )

    # ------------------------------------------------------------------
    # Selection-function validation
    # ------------------------------------------------------------------

    sf_validation = None

    if config.use_sel_func:
        validator = SelectionFunctionValidator(
            context=context,
            sf_set=selection_functions,
        )
        sf_validation = validator.check_coverage()
        validator.plot_sf(
            save_prefix=outputs.figures_dir / "sf_validation",
        )
    else:
        print("\nSelection-function validation skipped.")

    # ------------------------------------------------------------------
    # Optional survey maps
    # ------------------------------------------------------------------

    survey_map_summary = None

    if config.run_top_survey_maps and config.use_sel_func:
        survey_map_summary = plot_top_survey_maps(
            context=context,
            df_data=df_data,
            sf_set=selection_functions,
            max_surveys=12,
            output_prefix=outputs.figures_dir / "survey_maps",
        )

    # ------------------------------------------------------------------
    # Intersurvey diagnostics
    # ------------------------------------------------------------------

    _section("INTERSURVEY DIAGNOSTICS")

    surveys      = split_by_survey(context, df_data, verbose=False)
    intersurvey  = analyze_intersurvey_correlations(context, surveys)
    corr_df      = intersurvey.correlation_matrix
    overlap_df   = intersurvey.overlap_matrix

    # ------------------------------------------------------------------
    # Observed random catalog
    # ------------------------------------------------------------------

    _section("OBSERVED RANDOM CATALOG")

    rng_obs    = np.random.default_rng(config.random_seed)
    df_rand_obs = generate_random_catalog(
        context=context,
        n_observed=len(df_data) * config.n_rand_factor,
        sf_set=selection_functions,
        jitter_pixels=True,
        use_poisson=False,
        rng=rng_obs,
    )

    print(f"Observed random catalog size = {len(df_rand_obs)}")

    # ------------------------------------------------------------------
    # Observed angular statistics
    # ------------------------------------------------------------------

    _section("OBSERVED ANGULAR STATISTICS")

    theta, w_obs = compute_2pacf(
        context=context,
        df_data=df_data,
        df_rand=df_rand_obs,
    )

    theta = np.asarray(theta, dtype=float)
    w_obs = np.nan_to_num(
        np.asarray(w_obs, dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    abs_obs = np.asarray(
        get_absolute_sum(context=context, theta=theta, w=w_obs),
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Jackknife
    # ------------------------------------------------------------------

    _section("JACKKNIFE UNCERTAINTIES")

    jackknife = run_jackknife_errors(
        context=context,
        df_data=df_data,
        sf_set=selection_functions,
        nside_jackknife=config.nside_jackknife,
        min_regions=config.min_jackknife_regions,
    )

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    _section("BOOTSTRAP UNCERTAINTIES")

    bootstrap = run_bootstrap_errors(
        context=context,
        df_data=df_data,
        sf_set=selection_functions,
        n_bootstrap=config.n_bootstrap,
        n_jobs=config.n_jobs,
    )

    # ------------------------------------------------------------------
    # H0 ensemble
    # ------------------------------------------------------------------

    _section("H0 ENSEMBLE")

    mock_ensemble = run_ensemble_mocks(
        context=context,
        df_data=df_data,
        sf_set=selection_functions if config.use_sel_func else None,
        n_ensemble=config.n_ensemble,
        n_mocks_per=config.n_mocks_per_ensemble,
        perturbation_scale=config.perturbation_scale,
        n_rand_factor=config.n_rand_factor,
        n_jobs=config.n_jobs,
    )

    all_w_h0   = np.asarray(mock_ensemble.w_theta,       dtype=float)
    all_abs_h0 = np.asarray(mock_ensemble.abs_statistics, dtype=float)
    all_w_h0, all_abs_h0 = _filter_valid_h0(all_w_h0, all_abs_h0)

    # ------------------------------------------------------------------
    # Statistical inference
    # ------------------------------------------------------------------

    _section("STATISTICAL INFERENCE")

    stats = compute_statistics(
        context=context,
        w_obs=w_obs,
        abs_obs=abs_obs,
        all_w_h0=all_w_h0,
        all_abs_h0=all_abs_h0,
    )

    # ------------------------------------------------------------------
    # Main visualizations
    # ------------------------------------------------------------------

    _section("MAIN VISUALIZATIONS")

    plot_results_with_jackknife(
        context=context,
        theta=theta,
        w_obs=w_obs,
        abs_obs=abs_obs,
        all_w_h0=all_w_h0,
        all_abs_h0=all_abs_h0,
        jackknife=jackknife,
    )

    plot_results_with_bootstrap(
        context=context,
        theta=theta,
        w_obs=w_obs,
        abs_obs=abs_obs,
        all_w_h0=all_w_h0,
        all_abs_h0=all_abs_h0,
        bootstrap=bootstrap,
    )

    plot_results_jk_vs_bootstrap(
        context=context,
        theta=theta,
        w_obs=w_obs,
        abs_obs=abs_obs,
        all_w_h0=all_w_h0,
        all_abs_h0=all_abs_h0,
        jackknife=jackknife,
        bootstrap=bootstrap,
    )

    # ------------------------------------------------------------------
    # Results container
    # ------------------------------------------------------------------

    results = {
        "context":              context,
        "config":               config,
        "df_data":              df_data,
        "theta":                theta,
        "w_obs":                w_obs,
        "abs_obs":              abs_obs,
        "selection_functions":  selection_functions,
        "sf_validation":        sf_validation,
        "survey_map_summary":   survey_map_summary,
        "corr_df":              corr_df,
        "overlap_df":           overlap_df,
        "jackknife":            jackknife,
        "bootstrap":            bootstrap,
        "all_w_h0":             all_w_h0,
        "all_abs_h0":           all_abs_h0,
        "stats":                stats,
        "chi2_bin_diagnostics": None,        
        "svd_mode_contributions": None,      
        "covariance_matrix":    getattr(context, "last_covariance_matrix", None),
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    if save_outputs:
        print("\n--- Saving outputs ---")
        results.update(
            save_tables(
                context=context,
                results=results,
                prefix=config.run_tag,
            )
        )

    _section("PIPELINE FINISHED SUCCESSFULLY")

    return results