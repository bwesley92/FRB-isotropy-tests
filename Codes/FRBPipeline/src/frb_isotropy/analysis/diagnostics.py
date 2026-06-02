from ..core.config import RuntimeContext

import astropy.units as u

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.types import FloatArray

import healpy as hp
from astropy.coordinates import SkyCoord

from ..core.models import IntersurveyAnalysis

from ..statistics.covariance import (
    build_shrinkage_covariance,
    compute_covariance_diagnostics,
    compute_covariance_comparison_metrics,
)

from ..core.models import (
    AbsoluteStatistics,
    ChiSquareStatistics,
    CovarianceDiagnostics,
    SVDStatistics,
)


# ==============================================================================
# Intersurvey correlation and overlap analysis
# ==============================================================================

def analyze_intersurvey_correlations(
    context: RuntimeContext,
    surveys: dict[str, pd.DataFrame],
    radius_deg: float | None = None,
    nside: int | None = None,
) -> IntersurveyAnalysis:
    """
    Quantify spatial correlations and overlap between surveys.

    Diagnostics
    -----------
    1) HEALPix Pearson correlation
       between survey sky maps.

    2) Fraction of events with nearest-neighbor
       counterpart within radius_deg.

    Notes
    -----
    This analysis is only meaningful when
    selection-function modeling is enabled.

    Parameters
    ----------
    surveys : dict[str, pd.DataFrame]
        Dictionary containing one DataFrame
        per survey.

    radius_deg : float
        Angular radius used to define nearest-neighbor
        overlap between surveys.

    nside : int
        HEALPix resolution used to pixelize survey
        footprints.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if radius_deg is None:

        radius_deg = (
            config.overlap_radius_deg
        )

    if nside is None:

        nside = (
            config.overlap_nside
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if radius_deg <= 0:

        raise ValueError(
            "radius_deg must be positive."
        )

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    # ------------------------------------------------------------------
    # Disabled mode
    # ------------------------------------------------------------------

    if config.use_sel_func is False:

        print(
            "\n--- Intersurvey analysis skipped ---"
        )

        print(
            "use_sel_func=False "
            "→ no survey partition exists."
        )

        empty = pd.DataFrame()

        return IntersurveyAnalysis(

            correlation_matrix=empty,

            overlap_matrix=empty,

            mean_correlation=np.nan,

            max_correlation=np.nan,

            mean_overlap=np.nan,

            max_overlap=np.nan,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if len(surveys) == 0:

        raise ValueError(
            "No surveys available for "
            "intersurvey analysis."
        )

    survey_names = list(
        surveys.keys()
    )

    n_surveys = len(
        survey_names
    )

    eps = (
        config.numerical_zero_tolerance
    )

    print("\n==================================================")
    print("INTERSURVEY CORRELATION ANALYSIS")
    print("==================================================")

    print(
        f"\nUnique surveys: "
        f"{n_surveys}"
    )

    print(
        f"HEALPix nside: "
        f"{nside}"
    )

    print(
        f"Overlap radius: "
        f"{radius_deg:.1f} deg"
    )

    # ------------------------------------------------------------------
    # Storage arrays
    # ------------------------------------------------------------------

    corr_matrix = np.zeros(
        (n_surveys, n_surveys),
        dtype=float,
    )

    overlap_fractions = np.zeros(
        (n_surveys, n_surveys),
        dtype=float,
    )

    npix = hp.nside2npix(
        nside
    )

    # ------------------------------------------------------------------
    # SkyCoord cache
    # ------------------------------------------------------------------

    coords_by_survey: dict[
        str,
        SkyCoord,
    ] = {

        name: SkyCoord(

            ra=(
                subdf["RA"]
                .to_numpy(float)
                * u.degree
            ),

            dec=(
                subdf["DEC"]
                .to_numpy(float)
                * u.degree
            ),

            frame="icrs",
        )

        for name, subdf
        in surveys.items()
    }

    # ------------------------------------------------------------------
    # HEALPix histogram cache
    # ------------------------------------------------------------------

    histograms: dict[
        str,
        FloatArray,
    ] = {}

    for name, subdf in surveys.items():

        theta = np.radians(
            90.0
            - subdf["DEC"]
            .to_numpy(float)
        )

        phi = np.radians(
            subdf["RA"]
            .to_numpy(float)
        )

        pix = hp.ang2pix(
            nside,
            theta,
            phi,
        )

        hist = np.bincount(
            pix,
            minlength=npix,
        ).astype(float)

        hist_sum = hist.sum()

        if hist_sum > 0:

            hist /= hist_sum

        histograms[name] = hist

    # ------------------------------------------------------------------
    # Pairwise survey analysis
    # ------------------------------------------------------------------

    for i, survey1 in enumerate(
        survey_names
    ):

        for j, survey2 in enumerate(
            survey_names
        ):

            # ----------------------------------------------------------
            # Diagonal elements
            # ----------------------------------------------------------

            if i == j:

                corr_matrix[i, j] = 1.0

                overlap_fractions[i, j] = 1.0

                continue

            # ----------------------------------------------------------
            # Pearson correlation between sky maps
            # ----------------------------------------------------------

            hist1 = histograms[survey1]

            hist2 = histograms[survey2]

            if (
                np.std(hist1) < eps
                or np.std(hist2) < eps
            ):

                corr = 0.0

            else:

                with np.errstate(
                    invalid="ignore",
                    divide="ignore",
                ):

                    corr = np.corrcoef(
                        hist1,
                        hist2,
                    )[0, 1]

            if not np.isfinite(corr):

                corr = 0.0

            corr_matrix[i, j] = float(
                corr
            )

            # ----------------------------------------------------------
            # Nearest-neighbor overlap
            # ----------------------------------------------------------

            coords1 = (
                coords_by_survey[survey1]
            )

            coords2 = (
                coords_by_survey[survey2]
            )

            if (
                len(coords1) == 0
                or len(coords2) == 0
            ):

                overlap = 0.0

            else:

                _, sep2d, _ = (

                    coords1.match_to_catalog_sky(
                        coords2
                    )
                )

                overlap = float(
                    np.mean(
                        sep2d.degree
                        < radius_deg
                    )
                )

            overlap_fractions[i, j] = overlap

    # ------------------------------------------------------------------
    # Convert to DataFrames
    # ------------------------------------------------------------------

    corr_df = pd.DataFrame(
        corr_matrix,
        index=survey_names,
        columns=survey_names,
    )

    overlap_df = pd.DataFrame(
        100.0 * overlap_fractions,
        index=survey_names,
        columns=survey_names,
    )

    # ------------------------------------------------------------------
    # Global diagnostics
    # ------------------------------------------------------------------

    offdiag = ~np.eye(
        n_surveys,
        dtype=bool,
    )

    mean_corr = float(
        np.mean(
            corr_matrix[offdiag]
        )
    )

    max_corr = float(
        np.max(
            corr_matrix[offdiag]
        )
    )

    mean_overlap = float(
        np.mean(
            overlap_fractions[offdiag]
        )
    )

    max_overlap = float(
        np.max(
            overlap_fractions[offdiag]
        )
    )

    print(
        "\n--- Spatial correlation summary ---"
    )

    print(
        f"Mean survey correlation: "
        f"{mean_corr:.4f}"
    )

    print(
        f"Max survey correlation: "
        f"{max_corr:.4f}"
    )

    print(
        f"Mean overlap fraction: "
        f"{100.0 * mean_overlap:.2f}%"
    )

    print(
        f"Max overlap fraction: "
        f"{100.0 * max_overlap:.2f}%"
    )

    print(
        "\nHigh overlap/correlation pairs "
        "may indicate non-independent "
        "observational footprints."
    )

    return IntersurveyAnalysis(

        correlation_matrix=corr_df,

        overlap_matrix=overlap_df,

        mean_correlation=mean_corr,

        max_correlation=max_corr,

        mean_overlap=mean_overlap,

        max_overlap=max_overlap,
    )


def print_statistics_summary(
    label: str,
    chi2_stats: ChiSquareStatistics,
    svd_stats: SVDStatistics,
    covariance_diag: CovarianceDiagnostics,
    absolute_stats: AbsoluteStatistics,
    n_mocks: int,
    n_bins: int,
) -> None:
    """
    Print the main statistical summary.
    """

    print(
        f"\n--- Full covariance diagnostic "
        f"({label}) ---"
    )

    print(
        f"Chi2 = "
        f"{chi2_stats.chi2:.3f}"
    )

    print(
        f"Chi2/dof = "
        f"{chi2_stats.chi2_red:.3f}"
    )

    print(
        f"p-value (Chi2 analytic) = "
        f"{chi2_stats.p_chi2:.4e}"
    )

    print(
        f"p-value (Chi2 empirical) = "
        f"{chi2_stats.p_empirical:.4e}"
    )

    if (
        chi2_stats.p_empirical
        <= chi2_stats.p_empirical_floor
    ):

        print(
            "p-value (Chi2 empirical) "
            "is at the Monte Carlo floor: "
            f"p <= "
            f"{chi2_stats.p_empirical_floor:.4e}"
        )

    print(
        f"Sigma-equivalent "
        f"(from empirical p) = "
        f"{chi2_stats.sigma_equiv:.2f}"
    )

    print(
        "\n--- Primary isotropy statistic "
        "(SVD-regularized) ---"
    )

    print(
        f"Eigenvalue cut = "
        f"{svd_stats.svd_eigenvalue_cut:.1e}"
    )

    print(
        f"SVD modes kept = "
        f"{svd_stats.modes_kept}/{n_bins}"
    )

    print(
        f"Chi2 SVD = "
        f"{svd_stats.chi2_svd:.3f}"
    )

    print(
        f"Chi2 SVD/dof = "
        f"{svd_stats.chi2_svd_red:.3f}"
    )

    print(
        f"p-value SVD "
        f"(Chi2 analytic) = "
        f"{svd_stats.p_chi2_svd:.4e}"
    )

    print(
        f"p-value SVD "
        f"(empirical) = "
        f"{svd_stats.p_svd_empirical:.4e}"
    )

    print(
        f"Sigma-equivalent SVD "
        f"(from empirical p) = "
        f"{svd_stats.sigma_svd_equiv:.2f}"
    )

    print(
        "\n--- Covariance diagnostics ---"
    )

    print(
        f"Mocks = {n_mocks}"
    )

    print(
        f"Bins = {n_bins}"
    )

    print(
        f"Covariance rank = "
        f"{covariance_diag.covariance_rank}/{n_bins}"
    )

    print(
        f"Covariance condition number = "
        f"{covariance_diag.covariance_condition:.4e}"
    )

    print(
        f"SVD retained condition number = "
        f"{covariance_diag.svd_condition:.4e}"
    )

    print(
        f"Effective modes = "
        f"{covariance_diag.n_eff:.2f}"
    )

    print(
        f"Hartlap factor = "
        f"{covariance_diag.hartlap_factor:.3f}"
    )

    print(
        "\n--- Absolute-sum statistic ---"
    )

    print(
        f"Observed total = "
        f"{absolute_stats.abs_observed_stat:.4e}"
    )

    print(
        f"Empirical p-value = "
        f"{absolute_stats.abs_empirical_p:.4e}"
    )

    print(
        f"RMS normalized deviation = "
        f"{absolute_stats.global_tension:.3f}"
    )


def print_physical_interpretation(
    context: RuntimeContext,
    chi2_stats: ChiSquareStatistics,
) -> None:
    """
    Print a human-readable interpretation
    of the isotropy test.
    """

    config = context.config

    print(
        "\n=================================================="
    )

    print(
        "PHYSICAL INTERPRETATION"
    )

    print(
        "=================================================="
    )

    print(
        "H0 hypothesis:"
    )

    if config.use_sel_func:

        print(
            "  isotropy + survey selection effects"
        )

    else:

        print(
            "  perfect isotropy"
        )

    if chi2_stats.p_empirical > 0.05:

        print(
            "\nResult:"
        )

        print(
            "  ✓ Data are statistically "
            "compatible with H0."
        )

    else:

        print(
            "\nResult:"
        )

        print(
            "  ⚠ Possible tension with H0 detected."
        )


def _build_bin_diagnostic_table(
    context: RuntimeContext,
    theta: FloatArray,
    w_obs: FloatArray,
    all_w_h0: FloatArray,
    hartlap_factor: float,
) -> tuple[
    pd.DataFrame,
    FloatArray,
]:
    """
    Build bin-level chi-square diagnostics.
    """

    config = context.config

    h0_mean = np.mean(
        all_w_h0,
        axis=0,
    )

    h0_std = np.nanstd(
        all_w_h0,
        axis=0,
        ddof=1,
    )

    h0_std = np.asarray(
        h0_std,
        dtype=float,
    )

    h0_std[
        ~np.isfinite(h0_std)
    ] = 0.0

    delta = (
        w_obs
        - h0_mean
    )

    pull = delta / (
        h0_std + 1e-12
    )

    diagonal_chi2 = (

        hartlap_factor

        * delta**2

        / (
            h0_std**2
            + 1e-24
        )
    )

    bin_df = pd.DataFrame(
        {

            "run_tag":
                config.run_tag,

            "bin_index":
                np.arange(
                    len(theta)
                ),

            "theta_deg":
                theta,

            "w_obs":
                w_obs,

            "h0_mean":
                h0_mean,

            "h0_std":
                h0_std,

            "delta":
                delta,

            "pull":
                pull,

            "abs_pull":
                np.abs(pull),

            "diagonal_chi2_contribution":
                diagonal_chi2,
        }
    )

    bin_df = (
        bin_df
        .sort_values(
            "abs_pull",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        bin_df,
        delta,
    )


def _build_mode_diagnostic_table(
    context: RuntimeContext,
    theta: FloatArray,
    delta: FloatArray,
    cov: FloatArray,
    hartlap_factor: float,
    eigenvalue_cut: float,
) -> tuple[
    pd.DataFrame,
    FloatArray,
    FloatArray,
]:
    """
    Build covariance eigenmode diagnostics.
    """

    config = context.config

    eigvals, eigvecs = np.linalg.eigh(
        cov
    )

    order = np.argsort(
        eigvals
    )[::-1]

    eigvals = eigvals[
        order
    ]

    eigvecs = eigvecs[
        :,
        order,
    ]

    positive = (
        eigvals > 1e-15
    )

    if not np.any(
        positive
    ):

        raise ValueError(
            "Covariance matrix has no stable "
            "positive eigenvalues."
        )

    eigvals_positive = eigvals[
        positive
    ]

    max_eig = float(
        eigvals_positive[0]
    )

    keep = (

        positive

        &

        (
            eigvals
            >= eigenvalue_cut * max_eig
        )
    )

    if not np.any(
        keep
    ):

        keep[
            np.flatnonzero(
                positive
            )[0]
        ] = True

    delta_modes = np.asarray(
        eigvecs.T @ delta,
        dtype=float,
    )

    raw_mode_chi2 = np.zeros(
        len(eigvals),
        dtype=float,
    )

    raw_mode_chi2[
        positive
    ] = (

        hartlap_factor

        * delta_modes[
            positive
        ]**2

        / eigvals[
            positive
        ]
    )

    svd_mode_chi2 = np.where(
        keep,
        raw_mode_chi2,
        0.0,
    )

    max_loading_idx = np.argmax(
        np.abs(eigvecs),
        axis=0,
    )

    dominant_theta = theta[
        max_loading_idx
    ]

    dominant_loading = eigvecs[
        max_loading_idx,
        np.arange(
            len(eigvals)
        ),
    ]

    mode_df = pd.DataFrame(
        {

            "run_tag":
                config.run_tag,

            "mode_index":
                np.arange(
                    len(eigvals)
                ),

            "eigenvalue":
                eigvals,

            "relative_eigenvalue":
                eigvals / max(
                    max_eig,
                    np.finfo(float).tiny,
                ),

            "kept_by_svd_cut":
                keep,

            "delta_projection":
                delta_modes,

            "abs_delta_projection":
                np.abs(
                    delta_modes
                ),

            "raw_chi2_contribution":
                raw_mode_chi2,

            "svd_chi2_contribution":
                svd_mode_chi2,

            "dominant_theta_deg":
                dominant_theta,

            "dominant_loading":
                dominant_loading,
        }
    )

    mode_df = (
        mode_df
        .sort_values(
            "raw_chi2_contribution",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    mode_df[
        "cumulative_raw_chi2_contribution"
    ] = (
        mode_df[
            "raw_chi2_contribution"
        ].cumsum()
    )

    total_raw_mode_chi2 = float(
        mode_df[
            "raw_chi2_contribution"
        ].sum()
    )

    if total_raw_mode_chi2 > 0:

        mode_df[
            "fractional_raw_chi2_contribution"
        ] = (
            mode_df[
                "raw_chi2_contribution"
            ]
            / total_raw_mode_chi2
        )

    else:

        mode_df[
            "fractional_raw_chi2_contribution"
        ] = 0.0

    total_svd_mode_chi2 = float(
        mode_df[
            "svd_chi2_contribution"
        ].sum()
    )

    if total_svd_mode_chi2 > 0:

        mode_df[
            "fractional_svd_chi2_contribution"
        ] = (
            mode_df[
                "svd_chi2_contribution"
            ]
            / total_svd_mode_chi2
        )

    else:

        mode_df[
            "fractional_svd_chi2_contribution"
        ] = 0.0

    return (
        mode_df,
        keep,
        eigvals_positive,
    )


def _print_chi2_diagnostic_summary(
    context: RuntimeContext,
    n_mocks: int,
    n_bins: int,
    hartlap_factor: float,
    keep: FloatArray,
    eigenvalue_cut: float,
    eigvals_positive: FloatArray,
) -> None:
    """
    Print chi-square diagnostic summary.
    """

    config = context.config

    eigvals_safe = eigvals_positive[
        eigvals_positive > 1e-12
    ]

    if len(eigvals_safe) > 0:

        smallest_safe = max(
            float(
                np.min(
                    eigvals_safe
                )
            ),
            np.finfo(float).tiny,
        )

        covariance_condition = float(
            np.max(
                eigvals_safe
            )
            / smallest_safe
        )

        smallest_retained = float(
            np.min(
                eigvals_safe
            )
        )

    else:

        covariance_condition = np.inf

        smallest_retained = np.nan

    print("\n==================================================")
    print("CHI-SQUARE DIAGNOSTICS")
    print("==================================================")

    print(
        f"\nRun tag: "
        f"{config.run_tag}"
    )

    print(
        f"\nMocks: "
        f"{n_mocks}"
    )

    print(
        f"Bins: "
        f"{n_bins}"
    )

    print(
        f"Hartlap factor: "
        f"{hartlap_factor:.5f}"
    )

    print(
        f"SVD modes kept: "
        f"{np.sum(keep)} / {len(keep)}"
    )

    print(
        f"Eigenvalue cut: "
        f"{eigenvalue_cut:.3e}"
    )

    print(
        f"Stable covariance condition number: "
        f"{covariance_condition:.3e}"
    )

    print(
        f"Largest eigenvalue: "
        f"{np.max(eigvals_positive):.3e}"
    )

    print(
        f"Smallest retained eigenvalue: "
        f"{smallest_retained:.3e}"
    )


def validate_chi2_diagnostic_inputs(
    theta: FloatArray,
    w_obs: FloatArray,
    all_w_h0: FloatArray,
    eigenvalue_cut: float,
) -> None:
    """
    Validate chi-square diagnostic inputs.
    """

    if theta.ndim != 1:

        raise ValueError(
            "theta must be 1-dimensional."
        )

    if w_obs.ndim != 1:

        raise ValueError(
            "w_obs must be 1-dimensional."
        )

    if all_w_h0.ndim != 2:

        raise ValueError(
            "all_w_h0 must be 2-dimensional."
        )

    if len(theta) != len(w_obs):

        raise ValueError(
            "theta and w_obs "
            "must have identical lengths."
        )

    if all_w_h0.shape[1] != len(theta):

        raise ValueError(
            "all_w_h0 and theta "
            "have incompatible shapes."
        )

    if eigenvalue_cut <= 0:

        raise ValueError(
            "eigenvalue_cut must be positive."
        )
 

def remove_invalid_chi2_bins(
    theta: FloatArray,
    w_obs: FloatArray,
    all_w_h0: FloatArray,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """
    Remove bins that contain invalid values.
    """

    valid_bins = (

        np.isfinite(theta)

        &

        np.isfinite(w_obs)

        &

        np.all(
            np.isfinite(all_w_h0),
            axis=0,
        )
    )

    if np.sum(valid_bins) < 2:

        raise ValueError(
            "Not enough valid bins available "
            "for chi-square diagnostics."
        )

    return (
        theta[valid_bins],
        w_obs[valid_bins],
        all_w_h0[:, valid_bins],
    )


def compute_chi2_diagnostic_tables(
    context: RuntimeContext,
    theta: FloatArray,
    w_obs: FloatArray,
    all_w_h0: FloatArray,
    eigenvalue_cut: float | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build diagnostic tables for covariance-aware chi-square analysis.
    """

    config = context.config

    if eigenvalue_cut is None:

        eigenvalue_cut = (
            config.svd_eigenvalue_cut
        )

    theta = np.asarray(
        theta,
        dtype=float,
    )

    w_obs = np.asarray(
        w_obs,
        dtype=float,
    )

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    validate_chi2_diagnostic_inputs(
        theta=theta,
        w_obs=w_obs,
        all_w_h0=all_w_h0,
        eigenvalue_cut=eigenvalue_cut,
    )

    theta, w_obs, all_w_h0 = (
        remove_invalid_chi2_bins(
            theta=theta,
            w_obs=w_obs,
            all_w_h0=all_w_h0,
        )
    )

    n_mocks, n_bins = (
        all_w_h0.shape
    )

    cov = build_shrinkage_covariance(
        all_w_h0
    )

    covariance_diag = (
        compute_covariance_diagnostics(
            cov=cov,
            n_mocks=n_mocks,
            n_bins=n_bins,
        )
    )

    hartlap_factor = (
        covariance_diag.hartlap_factor
    )

    bin_df, delta = (
        _build_bin_diagnostic_table(
            context=context,
            theta=theta,
            w_obs=w_obs,
            all_w_h0=all_w_h0,
            hartlap_factor=hartlap_factor,
        )
    )

    mode_df, keep, eigvals_positive = (
        _build_mode_diagnostic_table(
            context=context,
            theta=theta,
            delta=delta,
            cov=cov,
            hartlap_factor=hartlap_factor,
            eigenvalue_cut=eigenvalue_cut,
        )
    )

    _print_chi2_diagnostic_summary(
        context=context,
        n_mocks=n_mocks,
        n_bins=n_bins,
        hartlap_factor=hartlap_factor,
        keep=keep,
        eigenvalue_cut=eigenvalue_cut,
        eigvals_positive=eigvals_positive,
    )

    return (
        bin_df,
        mode_df,
    )


def validate_covariance_comparison_input(
    w_array: FloatArray,
    label: str,
) -> None:
    """
    Validate covariance-comparison mock ensemble.
    """

    if w_array.ndim != 2:

        raise ValueError(
            "Input mock arrays must be "
            "2-dimensional."
        )

    if w_array.shape[0] < 2:

        raise ValueError(
            f"{label}: "
            "at least two mocks are required."
        )

    if w_array.shape[1] < 2:

        raise ValueError(
            f"{label}: "
            "at least two angular bins "
            "are required."
        )
    

def remove_invalid_covariance_bins(
    w_array: FloatArray,
    label: str,
) -> FloatArray:
    """
    Remove angular bins containing
    non-finite values across the mock ensemble.
    """

    valid_bins = np.all(
        np.isfinite(w_array),
        axis=0,
    )

    if np.sum(valid_bins) < 2:

        raise ValueError(
            f"Not enough valid bins for "
            f"{label} covariance comparison."
        )

    return w_array[
        :,
        valid_bins,
    ]


def print_covariance_comparison_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print covariance-comparison diagnostics.
    """

    print("\n==================================================")
    print("COVARIANCE COMPARISON DIAGNOSTICS")
    print("==================================================")

    print(
        df.round(4)
    )

    if len(df) != 2:

        return

    print(
        "\n--- Interpretation ---"
    )

    cond_old = float(
        df.loc[
            0,
            "condition_number",
        ]
    )

    cond_new = float(
        df.loc[
            1,
            "condition_number",
        ]
    )

    corr_old = float(
        df.loc[
            0,
            "mean_mock_corr",
        ]
    )

    corr_new = float(
        df.loc[
            1,
            "mean_mock_corr",
        ]
    )

    neff_old = float(
        df.loc[
            0,
            "effective_modes",
        ]
    )

    neff_new = float(
        df.loc[
            1,
            "effective_modes",
        ]
    )

    print(
        f"Mean mock correlation: "
        f"{corr_old:.4f} → {corr_new:.4f}"
    )

    print(
        f"Effective modes: "
        f"{neff_old:.2f} → {neff_new:.2f}"
    )

    print(
        f"Condition number: "
        f"{cond_old:.3e} → {cond_new:.3e}"
    )

    if (
        np.isfinite(corr_new)
        and np.isfinite(corr_old)
        and corr_new < corr_old
    ):

        print(
            "\n✓ Mock independence improved."
        )

    if (
        np.isfinite(neff_new)
        and np.isfinite(neff_old)
        and neff_new > neff_old
    ):

        print(
            "✓ Covariance dimensionality improved."
        )

    if (
        np.isfinite(cond_new)
        and np.isfinite(cond_old)
        and cond_new < cond_old
    ):

        print(
            "✓ Covariance conditioning improved."
        )


def compare_covariance_statistics(
    all_w_h0_old: FloatArray,
    all_w_h0_new: FloatArray,
) -> pd.DataFrame:

    comparisons = [

        (
            "Before (correlated)",
            np.asarray(
                all_w_h0_old,
                dtype=float,
            ),
        ),

        (
            "After (fixed)",
            np.asarray(
                all_w_h0_new,
                dtype=float,
            ),
        ),
    ]

    stats = []

    for label, w_array in comparisons:

        validate_covariance_comparison_input(
            w_array=w_array,
            label=label,
        )

        w_array = (
            remove_invalid_covariance_bins(
                w_array=w_array,
                label=label,
            )
        )

        stats.append(
            compute_covariance_comparison_metrics(
                w_array=w_array,
                label=label,
            )
        )

    df = pd.DataFrame(
        stats
    )

    print_covariance_comparison_summary(
        df
    )

    return df


# ==============================================================================
# Visualization
# ==============================================================================

_CORRELATION_EPS = 1e-30


def plot_covariance_correlation_matrix(
    context: RuntimeContext,
    cov: FloatArray,
    title: str = "Correlation Matrix",
    output: str | Path = "correlation_matrix.png",
) -> None:
    """
    Visualize covariance-derived correlation matrix.
    """

    outputs = context.outputs

    cov = np.asarray(
        cov,
        dtype=float,
    )

    output = Path(
        output
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if cov.ndim != 2:

        raise ValueError(
            "Covariance matrix must be 2D."
        )

    if cov.shape[0] != cov.shape[1]:

        raise ValueError(
            "Covariance matrix must be square."
        )

    if not np.all(
        np.isfinite(cov)
    ):

        raise ValueError(
            "Covariance matrix contains "
            "non-finite values."
        )

    if not np.allclose(
        cov,
        cov.T,
        atol=1e-12,
        rtol=1e-10,
    ):

        raise ValueError(
            "Covariance matrix is not symmetric."
        )

    # ------------------------------------------------------------------
    # Numerical stabilization
    # ------------------------------------------------------------------

    diag = np.diag(
        cov
    )

    sigma = np.sqrt(
        np.clip(
            diag,
            _CORRELATION_EPS,
            None,
        )
    )

    corr = cov / np.outer(
        sigma,
        sigma,
    )

    corr = np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    corr = np.clip(
        corr,
        -1.0,
        1.0,
    )

    # ------------------------------------------------------------------
    # Off-diagonal diagnostics
    # ------------------------------------------------------------------

    mask = ~np.eye(
        len(corr),
        dtype=bool,
    )

    offdiag = corr[
        mask
    ]

    offdiag = offdiag[
        np.isfinite(
            offdiag
        )
    ]

    print(
        "\n--- Covariance correlation diagnostics ---"
    )

    if len(offdiag) > 0:

        print(
            f"Mean off-diagonal corr: "
            f"{np.mean(offdiag):.4f}"
        )

        print(
            f"Median off-diagonal corr: "
            f"{np.median(offdiag):.4f}"
        )

        print(
            f"95th percentile corr: "
            f"{np.percentile(offdiag, 95):.4f}"
        )

        print(
            f"Max off-diagonal corr: "
            f"{np.max(offdiag):.4f}"
        )

    else:

        print(
            "No off-diagonal elements available."
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    im = ax.imshow(
        corr,
        origin="lower",
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    cbar = plt.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Correlation"
    )

    ax.set_xlabel(
        "Angular bin index"
    )

    ax.set_ylabel(
        "Angular bin index"
    )

    ax.set_title(
        f"{title}\n"
        f"({context.config.run_tag})"
    )

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------

    if not output.is_absolute():

        output = (
            outputs.figures_dir
            / output
        )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        "\nSaved correlation matrix:"
    )

    print(
        f"  {output}"
    )


def prepare_mock_similarity_data(
    all_w_h0: FloatArray,
    n_compare: int = 100,
) -> FloatArray:
    """
    Validate, clean, subsample and normalize
    mock realizations for similarity analysis.
    """

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if all_w_h0.ndim != 2:

        raise ValueError(
            "Mock array must be 2D."
        )

    if n_compare < 2:

        raise ValueError(
            "n_compare must be at least 2."
        )

    # ------------------------------------------------------------------
    # Remove invalid mocks
    # ------------------------------------------------------------------

    valid = np.all(
        np.isfinite(
            all_w_h0
        ),
        axis=1,
    )

    if not np.all(valid):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid)} invalid mocks "
            f"from similarity analysis."
        )

        all_w_h0 = all_w_h0[
            valid
        ]

    if len(all_w_h0) < 2:

        raise ValueError(
            "Not enough valid mocks "
            "for similarity analysis."
        )

    # ------------------------------------------------------------------
    # Subsample
    # ------------------------------------------------------------------

    subset = all_w_h0[
        :min(
            n_compare,
            len(all_w_h0),
        )
    ]

    # ------------------------------------------------------------------
    # Normalize angular bins
    # ------------------------------------------------------------------

    mock_mean = np.mean(
        subset,
        axis=0,
    )

    mock_std = np.std(
        subset,
        axis=0,
        ddof=1,
    )

    normalized = (
        subset
        - mock_mean
    ) / (
        mock_std
        + 1e-12
    )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return np.asarray(
        normalized,
        dtype=float,
    )


def compute_mock_similarity_matrix(
    normalized: FloatArray,
) -> tuple[
    FloatArray,
    FloatArray,
]:
    """
    Compute mock-to-mock correlation matrix
    and off-diagonal correlations.
    """

    corr = np.corrcoef(
        normalized
    )

    corr = np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    corr = np.clip(
        corr,
        -1.0,
        1.0,
    )

    mask = ~np.eye(
        len(corr),
        dtype=bool,
    )

    offdiag = corr[
        mask
    ]

    offdiag = offdiag[
        np.isfinite(
            offdiag
        )
    ]

    return (
        np.asarray(
            corr,
            dtype=float,
        ),
        np.asarray(
            offdiag,
            dtype=float,
        ),
    )


def _top_matrix_pairs(
    matrix: pd.DataFrame,
    n: int = 10,
    absolute: bool = True,
) -> pd.DataFrame:
    """
    Extract strongest pairwise survey relations.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not isinstance(
        matrix,
        pd.DataFrame,
    ):

        raise TypeError(
            "matrix must be a pandas DataFrame."
        )

    if n <= 0:

        raise ValueError(
            "n must be positive."
        )

    if matrix.empty:

        return pd.DataFrame(
            columns=[
                "survey_1",
                "survey_2",
                "value",
            ]
        )

    if matrix.shape[0] != matrix.shape[1]:

        raise ValueError(
            "matrix must be square."
        )

    # ------------------------------------------------------------------
    # Extract upper triangle
    # ------------------------------------------------------------------

    rows: list[
        dict[str, object]
    ] = []

    columns = list(
        matrix.columns
    )

    for i, row_name in enumerate(
        matrix.index
    ):

        for j, col_name in enumerate(
            columns
        ):

            # ----------------------------------------------------------
            # Upper triangle only
            # ----------------------------------------------------------

            if j <= i:

                continue

            value = matrix.iloc[
                i,
                j,
            ]

            if not np.isfinite(value):

                continue

            value = float(value)

            rows.append(
                {
                    "survey_1":
                        str(row_name),

                    "survey_2":
                        str(col_name),

                    "value":
                        value,

                    "abs_value":
                        abs(value),
                }
            )

    # ------------------------------------------------------------------
    # Empty output
    # ------------------------------------------------------------------

    if len(rows) == 0:

        return pd.DataFrame(
            columns=[
                "survey_1",
                "survey_2",
                "value",
            ]
        )

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    df_pairs = pd.DataFrame(
        rows
    )

    sort_col = (
        "abs_value"
        if absolute
        else "value"
    )

    df_pairs = (
        df_pairs
        .sort_values(
            sort_col,
            ascending=False,
            kind="stable",
        )
        .head(n)
        [
            [
                "survey_1",
                "survey_2",
                "value",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    return df_pairs