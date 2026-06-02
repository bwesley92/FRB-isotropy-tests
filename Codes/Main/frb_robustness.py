from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import replace
from pathlib import Path

from tqdm.auto import tqdm

import frb_isotropy as iso


# ==============================================================================
# Robustness parameter grid
# ==============================================================================

GAL_CUT_RANGE = [10.0, 15.0, 20.0]

BIN_SIZE_RANGE = [5.0, 10.0, 15.0]

NSIDE_SF_RANGE = [32, 64, 128]

SMOOTH_SIGMA_RANGE = [3.0, 5.0, 7.5, 10.0]


# ==============================================================================
# Robustness Monte Carlo configuration
# ==============================================================================

ROBUSTNESS_N_ENSEMBLE = 10

ROBUSTNESS_N_MOCKS_PER_ENSEMBLE = 20

ROBUSTNESS_BASE_CONFIG: iso.AnalysisConfig | None = None


# ==============================================================================
# Robustness configuration utilities
# ==============================================================================

def _require_active_config() -> iso.AnalysisConfig:

    if iso.ACTIVE_CONFIG is None:

        raise RuntimeError(
            "No active AnalysisConfig found. "
            "Define the full configuration in the notebook and call "
            "iso.apply_analysis_config(config) before running robustness."
        )

    return iso.ACTIVE_CONFIG


def _robustness_name(
    gal_cut: float,
    bin_size: float,
    nside_sf: int,
    smooth_sigma: float,
) -> str:

    def fmt(value: float | int) -> str:
        text = f"{value:g}" if isinstance(value, float) else str(value)
        return text.replace(".", "p")

    return (
        f"robust_"
        f"gal{fmt(gal_cut)}_"
        f"bin{fmt(bin_size)}_"
        f"nside{nside_sf}_"
        f"smooth{fmt(smooth_sigma)}"
    )


def build_robustness_config(
    gal_cut: float,
    bin_size: float,
    nside_sf: int,
    smooth_sigma: float,
    perturbation_scale: float | None = None,
) -> iso.AnalysisConfig:

    base_config = _require_active_config()

    coarse_step = 2.0 * float(bin_size)

    coarse_bins = tuple(
        np.arange(
            0,
            181,
            coarse_step,
        ).astype(float)
    )

    return replace(
        base_config,
        name="robust",
        gal_cut=float(gal_cut),
        bin_size=float(bin_size),
        coarse_bins=coarse_bins,
        nside_sf=int(nside_sf),
        smooth_sigma=float(smooth_sigma),
        perturbation_scale=(
            base_config.perturbation_scale
            if perturbation_scale is None
            else float(perturbation_scale)
        ),
        n_ensemble=ROBUSTNESS_N_ENSEMBLE,
        n_mocks_per_ensemble=ROBUSTNESS_N_MOCKS_PER_ENSEMBLE,
    )


def apply_robustness_configuration(
    gal_cut: float,
    bin_size: float,
    nside_sf: int,
    smooth_sigma: float,
    perturbation_scale: float | None = None,
) -> dict[str, float]:

    global ROBUSTNESS_BASE_CONFIG

    active_config = _require_active_config()

    if not active_config.name.startswith("robust_"):
        ROBUSTNESS_BASE_CONFIG = active_config

    config = build_robustness_config(
        gal_cut=gal_cut,
        bin_size=bin_size,
        nside_sf=nside_sf,
        smooth_sigma=smooth_sigma,
        perturbation_scale=perturbation_scale,
    )

    iso.apply_analysis_config(config)

    return {
        "gal_cut": iso.GAL_CUT,
        "min_sep": iso.MIN_SEP,
        "max_sep": iso.MAX_SEP,
        "bin_size": iso.BIN_SIZE,
        "n_bins": iso.N_BINS,
        "coarse_bin_count": len(iso.COARSE_BINS) - 1,
        "nside_sf": iso.NSIDE_SF,
        "smooth_sigma": iso.SMOOTH_SIGMA,
        "perturbation_scale": iso.PERTURBATION_SCALE,
        "n_rand_factor": iso.N_RAND_FACTOR,
        "n_ensemble": iso.N_ENSEMBLE,
        "n_mocks_per_ensemble": iso.N_MOCKS_PER_ENSEMBLE,
        "n_mocks": iso.N_MOCKS,
        "run_tag": iso.RUN_TAG,
    }


def restore_fiducial_runtime_state() -> None:

    if ROBUSTNESS_BASE_CONFIG is None:

        raise RuntimeError(
            "No stored base AnalysisConfig found for restoration."
        )

    iso.apply_analysis_config(ROBUSTNESS_BASE_CONFIG)

    print(
        "\nBase AnalysisConfig restored."
    )


# ==============================================================================
# Single robustness realization
# ==============================================================================

def run_single_robustness_case(
    gal_cut: float,
    bin_size: float,
    nside_sf: int,
    smooth_sigma: float,
    perturbation_scale: float | None = None,
    n_jobs: int | None = None,
) -> dict[str, float]:

    """
    Run one complete robustness configuration.
    """

    if n_jobs is None:

        n_jobs = iso.N_JOBS

    # ----------------------------------------------------------
    # Apply runtime configuration
    # ----------------------------------------------------------

    config_dict = (
        apply_robustness_configuration(
            gal_cut=gal_cut,
            bin_size=bin_size,
            nside_sf=nside_sf,
            smooth_sigma=smooth_sigma,
            perturbation_scale=(
                perturbation_scale
            ),
        )
    )

    # ----------------------------------------------------------
    # Load observed catalog
    # ----------------------------------------------------------

    print("\n--- Loading catalog ---")

    df_data = iso.apply_mask(
        iso.load_catalog(
            iso.ALL_FRB_PATH
        ),
        gal_cut=iso.GAL_CUT,
    )

    print(
        f"Catalog size = {len(df_data)}"
    )

    # ----------------------------------------------------------
    # Build selection functions
    # ----------------------------------------------------------

    print(
        "\n--- Building selection functions ---"
    )

    (
        sf_dict,
        sf_cov_diag_dict,
        survey_weights,
        sf_nside,
    ) = (
        iso.build_survey_selection_functions_improved(
            df_data,
            nside=iso.NSIDE_SF,
            smooth_sigma=(
                iso.SMOOTH_SIGMA
            ),
            gal_cut=iso.GAL_CUT,
        )
    )

    # ----------------------------------------------------------
    # Observed random catalog
    # ----------------------------------------------------------

    print(
        "\n--- Generating observed random catalog ---"
    )

    df_rand_obs = (
        iso.generate_mixture_catalog_improved(
            n_observed=(
                len(df_data)
                * iso.N_RAND_FACTOR
            ),
            sf_dict=sf_dict,
            weights=survey_weights,
            nside=sf_nside,
            seed=2000,
            use_poisson=False,
        )
    )

    # ----------------------------------------------------------
    # Observed statistics
    # ----------------------------------------------------------

    print(
        "\n--- Computing observed statistics ---"
    )

    theta, w_obs = iso.compute_2pacf(
        df_data,
        df_rand_obs,
    )

    theta = np.asarray(
        theta,
        dtype=float,
    )

    w_obs = np.asarray(
        w_obs,
        dtype=float,
    )

    abs_obs = np.asarray(
        iso.get_absolute_sum(
            theta,
            w_obs,
        ),
        dtype=float,
    )

    # ----------------------------------------------------------
    # H0 ensemble
    # ----------------------------------------------------------

    print(
        "\n--- Running H0 ensemble ---"
    )

    (
        all_w_h0,
        all_abs_h0,
    ) = iso.run_ensemble_mocks(
        df_data,
        sf_dict,
        sf_cov_diag_dict,
        survey_weights,
        sf_nside,
        n_ensemble=(ROBUSTNESS_N_ENSEMBLE),
        n_mocks_per=(ROBUSTNESS_N_MOCKS_PER_ENSEMBLE),
        perturbation_scale=(
            iso.PERTURBATION_SCALE
        ),
        n_jobs=n_jobs,
    )

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    all_abs_h0 = np.asarray(
        all_abs_h0,
        dtype=float,
    )

    # ----------------------------------------------------------
    # Main statistics
    # ----------------------------------------------------------

    print(
        "\n--- Computing statistics ---"
    )

    stats = iso.compute_statistics(
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
        label="robustness",
    )

    # ----------------------------------------------------------
    # Return summary row
    # ----------------------------------------------------------

    return {

        # ======================================================
        # Configuration
        # ======================================================

        **config_dict,

        # ======================================================
        # Full Covariance Diagnostic
        # ======================================================

        "chi2":
            stats.chi2,

        "chi2_red":
            stats.chi2_red,

        "p_chi2_analytic":
            stats.p_chi2,

        "p_chi2_empirical":
            stats.p_empirical,

        "p_chi2_empirical_floor":
            stats.p_empirical_floor,

        "sigma_equiv_empirical":
            stats.sigma_equiv,

        "hartlap_factor":
            stats.hartlap_factor,

        "rms_normalized_deviation":
            stats.global_tension,

        # ======================================================
        # Primary Isotropy Statistic (SVD-Regularized)
        # ======================================================

        "chi2_svd":
            stats.chi2_svd,

        "chi2_svd_red":
            stats.chi2_svd_red,

        "p_chi2_svd_analytic":
            stats.p_chi2_svd,

        "p_chi2_svd_empirical":
            stats.p_svd_empirical,

        "sigma_svd_empirical":
            stats.sigma_svd_equiv,

        "svd_modes_kept":
            stats.svd_modes_kept,

        "svd_retained_condition":
            stats.svd_condition,

        # ======================================================
        # Covariance Diagnostics
        # ======================================================

        "covariance_condition":
            stats.covariance_condition,

        "effective_modes":
            stats.n_eff,

        # ======================================================
        # Non-parametric Profile Tests
        # ======================================================

        "w_ks_stat":
            stats.ks_w_stat,

        "w_ks_pvalue":
            stats.ks_w_pvalue,

        "w_ks_empirical_p":
            stats.ks_w_empirical_p,

        "w_ad_stat":
            stats.ad_w_stat,

        "w_ad_pvalue":
            stats.ad_w_pvalue,

        "w_ad_empirical_p":
            stats.ad_w_empirical_p,

        "abs_ks_stat":
            stats.ks_abs_stat,

        "abs_ks_pvalue":
            stats.ks_abs_pvalue,

        "abs_ks_empirical_p":
            stats.ks_abs_empirical_p,

        "abs_ad_stat":
            stats.ad_abs_stat,

        "abs_ad_pvalue":
            stats.ad_abs_pvalue,

        "abs_ad_empirical_p":
            stats.ad_abs_empirical_p,

        # ======================================================
        # Absolute Anisotropy Amplitude
        # ======================================================

        "observed_rms_absolute_amplitude":
            stats.abs_observed_stat,

        "empirical_pvalue":
            stats.abs_empirical_p,
    }


# ==============================================================================
# Full robustness scan
# ==============================================================================

# ==============================================================================
# Full robustness scan
# ==============================================================================

def run_full_robustness_scan(
    gal_cut_range: list[float] = (
        GAL_CUT_RANGE
    ),
    bin_size_range: list[float] = (
        BIN_SIZE_RANGE
    ),
    nside_sf_range: list[int] = (
        NSIDE_SF_RANGE
    ),
    smooth_sigma_range: list[float] = (
        SMOOTH_SIGMA_RANGE
    ),
    perturbation_scale: float | None = None,
    n_jobs: int | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:

    """
    Run the complete robustness grid scan.
    """

    if n_jobs is None:

        n_jobs = iso.N_JOBS

    # ----------------------------------------------------------
    # Total configurations
    # ----------------------------------------------------------

    total_configs = (

        len(gal_cut_range)

        * len(bin_size_range)

        * len(nside_sf_range)

        * len(smooth_sigma_range)
    )

    print("\n==================================================")

    print("FULL ROBUSTNESS SCAN")

    print("==================================================")

    print(
        f"Total configurations = "
        f"{total_configs}"
    )

    print("==================================================\n")

    # ----------------------------------------------------------
    # Prepare output path
    # ----------------------------------------------------------

    output_path = None

    if output_csv is not None:

        output_path = Path(output_csv)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # overwrite previous file immediately
        if output_path.exists():

            output_path.unlink()

    # ----------------------------------------------------------
    # Results accumulator
    # ----------------------------------------------------------

    rows = []

    config_counter = 0

    global ROBUSTNESS_BASE_CONFIG

    base_config = _require_active_config()
    ROBUSTNESS_BASE_CONFIG = base_config

    try:

        # ------------------------------------------------------
        # Parameter scan
        # ------------------------------------------------------

        for gal_cut in gal_cut_range:

            for bin_size in bin_size_range:

                for nside_sf in nside_sf_range:

                    for smooth_sigma in (
                        smooth_sigma_range
                    ):

                        config_counter += 1

                        print(
                            "\n--------------------------------------------------"
                        )

                        print(
                            f"Configuration "
                            f"{config_counter}"
                            f"/{total_configs}"
                        )

                        print(
                            "--------------------------------------------------"
                        )

                        try:

                            row = (
                                run_single_robustness_case(
                                    gal_cut=gal_cut,
                                    bin_size=bin_size,
                                    nside_sf=nside_sf,
                                    smooth_sigma=(
                                        smooth_sigma
                                    ),
                                    perturbation_scale=(
                                        perturbation_scale
                                    ),
                                    n_jobs=n_jobs,
                                )
                            )

                            row["status"] = "success"

                            rows.append(row)

                            print(
                                "\nConfiguration completed successfully."
                            )

                            print(
                                f"p_chi2_empirical = "
                                f"{row['p_chi2_empirical']:.4e}"
                            )

                            print(
                                f"p_chi2_svd_empirical = "
                                f"{row['p_chi2_svd_empirical']:.4e}"
                            )

                            print(
                                f"empirical_pvalue = "
                                f"{row['empirical_pvalue']:.4e}"
                            )

                        except Exception as error:

                            print(
                                "\nERROR during robustness run:"
                            )

                            print(error)

                            rows.append({

                                "gal_cut":
                                    gal_cut,

                                "bin_size":
                                    bin_size,

                                "nside_sf":
                                    nside_sf,

                                "smooth_sigma":
                                    smooth_sigma,

                                "status":
                                    "failed",

                                "error":
                                    str(error),
                            })

                        # --------------------------------------------------
                        # Incremental save
                        # --------------------------------------------------

                        if output_path is not None:

                            partial = pd.DataFrame(rows)

                            partial.to_csv(
                                output_path,
                                index=False,
                            )

    finally:

        # ------------------------------------------------------
        # Restore fiducial runtime state
        # ------------------------------------------------------

        iso.apply_analysis_config(base_config)

        print(
            "\nBase AnalysisConfig restored after robustness scan."
        )

    # ----------------------------------------------------------
    # Consolidated dataframe
    # ----------------------------------------------------------

    results = pd.DataFrame(rows)

    # ----------------------------------------------------------
    # Final save
    # ----------------------------------------------------------

    if output_path is not None:

        results.to_csv(
            output_path,
            index=False,
        )

        print(
            f"\nSaved robustness table:"
        )

        print(output_path)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n==================================================")

    print("ROBUSTNESS SCAN FINISHED")

    print("==================================================")

    n_failed = np.sum(
        results["status"] == "failed"
    )

    n_success = np.sum(
        results["status"] == "success"
    )

    print(
        f"Successful runs = "
        f"{n_success}"
    )

    print(
        f"Failed runs = "
        f"{n_failed}"
    )

    print("==================================================\n")

    return results