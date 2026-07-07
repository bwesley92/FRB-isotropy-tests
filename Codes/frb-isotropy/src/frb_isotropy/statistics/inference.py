from __future__ import annotations

import warnings

import numpy as np

from scipy.stats import (
    anderson_ksamp,
    chi2 as chi2_dist,
    ks_2samp,
    norm,
)

from ..core.config import RuntimeContext

from ..core.types import FloatArray

from ..visualization.plotting import (
    plot_covariance_diagnostics,
)

# reporting exports imported at runtime to avoid circular imports

from ..statistics.covariance import (
    build_shrinkage_covariance,
    compute_covariance_diagnostics,
)

from ..core.models import (
    SVDStatistics,
    ProfileNonParametricStatistics,
    NonParametricStatistics,
    AbsoluteStatistics,
    ChiSquareStatistics,
    CovarianceResult,
    TestStatistics,
)


# ==============================================================================
# Statistical inference and diagnostics
# ==============================================================================

def sigma_equivalent_from_p(
    p_value: float,
) -> float:
    """
    Convert a p-value into a one-sided
    Gaussian-equivalent significance.

    Parameters
    ----------
    p_value : float
        Probability value.

    Returns
    -------
    float
        One-sided Gaussian significance.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not np.isfinite(
        p_value
    ):

        raise ValueError(
            "p_value must be finite."
        )

    if (
        p_value < 0.0
        or p_value > 1.0
    ):

        raise ValueError(
            "p_value must lie within "
            "[0, 1]."
        )

    # ------------------------------------------------------------------
    # Exact limits
    # ------------------------------------------------------------------

    if p_value == 1.0:

        return float(
            -np.inf
        )

    if p_value == 0.0:

        return float(
            np.inf
        )

    # ------------------------------------------------------------------
    # Numerical safeguard
    # ------------------------------------------------------------------

    p_clipped = np.clip(
        p_value,
        1e-300,
        1.0 - 1e-16,
    )

    sigma = norm.isf(
        p_clipped
    )

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    if not np.isfinite(
        sigma
    ):

        raise RuntimeError(
            "Failed to compute finite "
            "Gaussian-equivalent significance."
        )

    return float(
        sigma
    )


def compute_svd_regularized_chi2(
    context: RuntimeContext,
    delta: FloatArray,
    mock_delta: FloatArray,
    cov: FloatArray,
    hartlap_factor: float,
    eigenvalue_cut: float | None = None,
    verbose: bool = True,
) -> SVDStatistics:
    """
    Compute SVD-regularized chi-square statistics.

    Strategy
    --------
    - Diagonalize covariance matrix
    - Remove poorly constrained eigenmodes
    - Compute chi-square in the stable subspace

    This mitigates catastrophic amplification
    of noisy covariance modes.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if eigenvalue_cut is None:

        eigenvalue_cut = (
            config.svd_eigenvalue_cut
        )

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    delta = np.asarray(
        delta,
        dtype=float,
    )

    mock_delta = np.asarray(
        mock_delta,
        dtype=float,
    )

    cov = np.asarray(
        cov,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if cov.ndim != 2:

        raise ValueError(
            "Covariance matrix must be "
            "2-dimensional."
        )

    if cov.shape[0] != cov.shape[1]:

        raise ValueError(
            "Covariance matrix must be square."
        )

    if delta.ndim != 1:

        raise ValueError(
            "delta must be 1-dimensional."
        )

    if mock_delta.ndim != 2:

        raise ValueError(
            "mock_delta must be 2-dimensional."
        )

    if delta.shape[0] != cov.shape[0]:

        raise ValueError(
            "delta and covariance dimensions "
            "do not match."
        )

    if mock_delta.shape[1] != cov.shape[0]:

        raise ValueError(
            "mock_delta and covariance "
            "dimensions do not match."
        )

    if not np.all(
        np.isfinite(delta)
    ):

        raise ValueError(
            "delta contains non-finite values."
        )

    if not np.all(
        np.isfinite(mock_delta)
    ):

        raise ValueError(
            "mock_delta contains non-finite values."
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

    if hartlap_factor <= 0:

        raise ValueError(
            "hartlap_factor must be positive."
        )

    if eigenvalue_cut <= 0:

        raise ValueError(
            "eigenvalue_cut must be positive."
        )

    # ------------------------------------------------------------------
    # Hartlap consistency
    # ------------------------------------------------------------------

    n_mocks = int(
        mock_delta.shape[0]
    )

    n_dim = int(
        cov.shape[0]
    )

    if n_mocks <= n_dim + 2:

        raise ValueError(
            "Number of mocks must satisfy "
            "N_mocks > N_bins + 2 "
            "for Hartlap correction."
        )

    # ------------------------------------------------------------------
    # Eigen decomposition
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Remove numerically unstable eigenmodes
    # ------------------------------------------------------------------

    finite = np.isfinite(
        eigvals
    )

    positive = (
        eigvals
        > config.numerical_eigenvalue_floor
    )

    valid = (
        finite
        & positive
    )

    if not np.any(valid):

        raise ValueError(
            "Covariance matrix has no valid "
            "positive eigenvalues."
        )

    eigvals_valid = eigvals[
        valid
    ]

    eigvecs_valid = eigvecs[
        :,
        valid,
    ]

    # ------------------------------------------------------------------
    # Relative eigenvalue threshold
    # ------------------------------------------------------------------

    max_eig = float(
        eigvals_valid[0]
    )

    keep = (
        eigvals_valid
        >= eigenvalue_cut * max_eig
    )

    # ------------------------------------------------------------------
    # Always keep at least one mode
    # ------------------------------------------------------------------

    if not np.any(keep):

        keep[0] = True

    kept_eigvals = eigvals_valid[
        keep
    ]

    kept_eigvecs = eigvecs_valid[
        :,
        keep,
    ]

    modes_kept = int(
        len(kept_eigvals)
    )

    # ------------------------------------------------------------------
    # Hartlap factor for the truncated eigenbasis
    #
    # The Hartlap et al. (2007) debiasing factor depends on the
    # dimensionality of the space in which the precision matrix is
    # estimated/inverted. The `hartlap_factor` argument is derived
    # from the full covariance dimension (n_bins) and is correct for
    # the full-space chi-square, but chi2_svd inverts only the
    # `modes_kept`-dimensional stable eigenbasis, so it must be
    # rederived here using modes_kept instead of n_bins.
    # modes_kept <= n_dim is guaranteed by construction, and the
    # caller already enforces n_mocks > n_dim + 2, so this is always
    # well-defined and positive.
    # ------------------------------------------------------------------

    hartlap_factor_svd = float(
        (n_mocks - modes_kept - 2)
        / (n_mocks - 1)
    )

    # ------------------------------------------------------------------
    # Stable inverse eigenvalues
    # ------------------------------------------------------------------

    inv_eigvals = 1.0 / np.clip(
        kept_eigvals,
        config.numerical_eigenvalue_floor,
        None,
    )

    # ------------------------------------------------------------------
    # Projection onto stable eigenbasis
    # ------------------------------------------------------------------

    delta_modes = (
        kept_eigvecs.T
        @ delta
    )

    mock_modes = (
        mock_delta
        @ kept_eigvecs
    )

    # ------------------------------------------------------------------
    # SVD-regularized chi-square
    # ------------------------------------------------------------------

    chi2_svd = float(

        hartlap_factor_svd

        * np.sum(
            delta_modes**2
            * inv_eigvals
        )
    )

    chi2_svd_mocks = (

        hartlap_factor_svd

        * np.sum(
            mock_modes**2
            * inv_eigvals,
            axis=1,
        )
    )

    chi2_svd_mocks = np.asarray(
        chi2_svd_mocks,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not np.isfinite(
        chi2_svd
    ):

        raise RuntimeError(
            "Non-finite SVD chi-square."
        )

    if not np.all(
        np.isfinite(
            chi2_svd_mocks
        )
    ):

        raise RuntimeError(
            "Non-finite mock SVD chi-square values."
        )

    # ------------------------------------------------------------------
    # Empirical significance
    # ------------------------------------------------------------------

    n_extreme = int(
        np.sum(
            chi2_svd_mocks
            >= chi2_svd
        )
    )

    p_empirical_floor = (
        1.0
        / (n_mocks + 1)
    )

    p_empirical = float(
        (n_extreme + 1)
        / (n_mocks + 1)
    )

    # ------------------------------------------------------------------
    # Analytic significance
    # ------------------------------------------------------------------

    p_analytic = float(
        chi2_dist.sf(
            chi2_svd,
            modes_kept,
        )
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    smallest_eig = max(
        float(
            np.min(
                kept_eigvals
            )
        ),
        config.numerical_eigenvalue_floor,
    )

    svd_condition = float(
        kept_eigvals[0]
        / smallest_eig
    )

    total_variance = np.sum(
        eigvals_valid
    )

    if total_variance <= 0:

        explained_fraction = 0.0

    else:

        explained_fraction = float(
            np.sum(kept_eigvals)
            / total_variance
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    if verbose:

        print(
            "\n--- SVD covariance regularization ---"
        )

        print(
            f"Modes kept: "
            f"{modes_kept}/"
            f"{len(eigvals_valid)}"
        )

        print(
            f"Eigenvalue threshold: "
            f"{eigenvalue_cut:.2e}"
        )

        print(
            f"Explained variance kept: "
            f"{100.0 * explained_fraction:.2f}%"
        )

        print(
            f"SVD condition number: "
            f"{svd_condition:.2e}"
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return SVDStatistics(

        chi2_svd=chi2_svd,

        chi2_svd_red=(
            chi2_svd
            / modes_kept
        ),

        p_chi2_svd=p_analytic,

        p_svd_empirical=p_empirical,

        p_svd_empirical_floor=(
            p_empirical_floor
        ),

        sigma_svd_equiv=(

            sigma_equivalent_from_p(
                p_empirical
            )
        ),

        modes_kept=(
            modes_kept
        ),

        svd_condition=(
            svd_condition
        ),

        svd_eigenvalue_cut=(
            float(
                eigenvalue_cut
            )
        ),

        explained_variance_fraction=(
            explained_fraction
        ),
    )


def _anderson_ksamp_stat_p(
    sample_a: FloatArray,
    sample_b: FloatArray,
) -> tuple[float, float]:
    """
    Anderson-Darling k-sample test with
    numerical protection.

    Parameters
    ----------
    sample_a : FloatArray
        First sample.

    sample_b : FloatArray
        Second sample.

    Returns
    -------
    tuple[float, float]
        Anderson-Darling statistic and p-value.
    """

    sample_a = np.asarray(
        sample_a,
        dtype=float,
    )

    sample_b = np.asarray(
        sample_b,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if sample_a.ndim != 1:

        raise ValueError(
            "sample_a must be 1-dimensional."
        )

    if sample_b.ndim != 1:

        raise ValueError(
            "sample_b must be 1-dimensional."
        )

    # ------------------------------------------------------------------
    # Remove invalid values
    # ------------------------------------------------------------------

    sample_a = sample_a[
        np.isfinite(sample_a)
    ]

    sample_b = sample_b[
        np.isfinite(sample_b)
    ]

    # ------------------------------------------------------------------
    # Minimum sample size
    # ------------------------------------------------------------------

    if (
        len(sample_a) < 2
        or len(sample_b) < 2
    ):

        return (
            np.nan,
            np.nan,
        )

    # ------------------------------------------------------------------
    # Degenerate samples
    # ------------------------------------------------------------------

    if np.allclose(
        sample_a,
        sample_a[0],
    ) and np.allclose(
        sample_b,
        sample_b[0],
    ):

        return (
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Nearly degenerate variance
    # ------------------------------------------------------------------

    if (
        np.std(sample_a)
        < 1e-15
        and np.std(sample_b)
        < 1e-15
    ):

        return (
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Anderson-Darling test
    # ------------------------------------------------------------------

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        try:

            result = anderson_ksamp(
                [
                    sample_a,
                    sample_b,
                ]
            )

            stat = float(
                result.statistic
            )

            pvalue = float(
                result.pvalue
            )

        except Exception:

            stat = np.nan

            pvalue = np.nan

    # ------------------------------------------------------------------
    # Numerical cleanup
    # ------------------------------------------------------------------

    if not np.isfinite(stat):

        stat = np.nan

    if not np.isfinite(pvalue):

        pvalue = np.nan

    else:

        pvalue = float(
            np.clip(
                pvalue,
                0.0,
                1.0,
            )
        )

    return (
        stat,
        pvalue,
    )


def _profile_nonparametric_stats(
    observed: FloatArray,
    mocks: FloatArray,
    reference: FloatArray,
    verbose: bool = True,
) -> ProfileNonParametricStatistics:
    """
    Nonparametric comparison between observed
    angular profiles and isotropic benchmark profiles.

    Methodology
    -----------
    Following:

        Andrade et al. (2019)
        "Revisiting the statistical isotropy
        of GRB sky distribution"

    using:
        - Kolmogorov-Smirnov test
        - Anderson-Darling test
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    mocks = np.asarray(
        mocks,
        dtype=float,
    )

    reference = np.asarray(
        reference,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if observed.ndim != 1:

        raise ValueError(
            "observed must be 1-dimensional."
        )

    if reference.ndim != 1:

        raise ValueError(
            "reference must be 1-dimensional."
        )

    if mocks.ndim != 2:

        raise ValueError(
            "mocks must be 2-dimensional."
        )

    # ------------------------------------------------------------------
    # Remove invalid values
    # ------------------------------------------------------------------

    observed = observed[
        np.isfinite(observed)
    ]

    reference = reference[
        np.isfinite(reference)
    ]

    if (
        len(observed) < 2
        or len(reference) < 2
    ):

        raise ValueError(
            "Observed and reference samples "
            "must contain at least two "
            "finite values."
        )

    # ------------------------------------------------------------------
    # Degenerate protection
    # ------------------------------------------------------------------

    if np.std(observed) < 1e-15:

        observed = observed + np.random.default_rng(
            0
        ).normal(
            0.0,
            1e-12,
            size=len(observed),
        )

    if np.std(reference) < 1e-15:

        reference = reference + np.random.default_rng(
            1
        ).normal(
            0.0,
            1e-12,
            size=len(reference),
        )

    # ------------------------------------------------------------------
    # Observed vs benchmark
    # ------------------------------------------------------------------

    ks_result = ks_2samp(
        observed,
        reference,
        alternative="two-sided",
        mode="auto",
    )

    ks_stat = float(
        ks_result.statistic
    )

    ks_pvalue = float(
        ks_result.pvalue
    )

    ad_stat, ad_pvalue = (
        _anderson_ksamp_stat_p(
            observed,
            reference,
        )
    )

    # ------------------------------------------------------------------
    # Empirical calibration against H0 mocks
    # ------------------------------------------------------------------

    ks_mock_stats: list[float] = []

    ad_mock_stats: list[float] = []

    for mock_idx, mock in enumerate(mocks):

        mock = np.asarray(
            mock,
            dtype=float,
        )

        mock = mock[
            np.isfinite(mock)
        ]

        if len(mock) < 2:

            continue

        # --------------------------------------------------------------
        # Degenerate protection
        #
        # Seed varies per mock so that distinct degenerate mocks
        # receive independent jitter instead of an identical
        # perturbation (which would make them byte-for-byte equal
        # and distort the empirical KS/AD calibration).
        # --------------------------------------------------------------

        if np.std(mock) < 1e-15:

            mock = mock + np.random.default_rng(
                1000 + mock_idx
            ).normal(
                0.0,
                1e-12,
                size=len(mock),
            )

        # --------------------------------------------------------------
        # Kolmogorov-Smirnov
        # --------------------------------------------------------------

        ks_mock = ks_2samp(
            mock,
            reference,
            alternative="two-sided",
            mode="auto",
        ).statistic

        if np.isfinite(ks_mock):

            ks_mock_stats.append(
                float(ks_mock)
            )

        # --------------------------------------------------------------
        # Anderson-Darling
        # --------------------------------------------------------------

        ad_mock, _ = (
            _anderson_ksamp_stat_p(
                mock,
                reference,
            )
        )

        if np.isfinite(ad_mock):

            ad_mock_stats.append(
                float(ad_mock)
            )

    ks_mock_stats = np.asarray(
        ks_mock_stats,
        dtype=float,
    )

    ad_mock_stats = np.asarray(
        ad_mock_stats,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Remove invalid realizations
    # ------------------------------------------------------------------

    ks_mock_stats = ks_mock_stats[
        np.isfinite(
            ks_mock_stats
        )
    ]

    ad_mock_stats = ad_mock_stats[
        np.isfinite(
            ad_mock_stats
        )
    ]

    if len(ks_mock_stats) == 0:

        raise ValueError(
            "No valid KS mock realizations found."
        )

    if len(ad_mock_stats) == 0:

        raise ValueError(
            "No valid AD mock realizations found."
        )

    # ------------------------------------------------------------------
    # Empirical p-values
    # ------------------------------------------------------------------

    ks_n_extreme = int(
        np.sum(
            ks_mock_stats
            >= ks_stat
        )
    )

    ks_empirical_p = float(
        (ks_n_extreme + 1)
        / (len(ks_mock_stats) + 1)
    )

    if np.isfinite(ad_stat):

        ad_n_extreme = int(
            np.sum(
                ad_mock_stats
                >= ad_stat
            )
        )

        ad_empirical_p = float(
            (ad_n_extreme + 1)
            / (len(ad_mock_stats) + 1)
        )

    else:

        ad_n_extreme = 0

        ad_empirical_p = np.nan

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    if verbose:

        print(
            "\n--- Heuristic KS / AD profile diagnostics ---"
        )

        print(
            f"KS statistic   = "
            f"{ks_stat:.5f}"
        )

        print(
            f"KS p-value [heuristic] = "
            f"{ks_pvalue:.5f}"
        )

        print(
            f"KS empirical p = "
            f"{ks_empirical_p:.5f}"
        )

        print(
            f"AD statistic   = "
            f"{ad_stat:.5f}"
        )

        print(
            f"AD p-value [heuristic] = "
            f"{ad_pvalue:.5f}"
        )

        print(
            f"AD empirical p = "
            f"{ad_empirical_p:.5f}"
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return ProfileNonParametricStatistics(

        ks_statistic=ks_stat,

        ks_pvalue=ks_pvalue,

        ks_empirical_p=ks_empirical_p,

        ks_n_extreme_mocks=ks_n_extreme,

        ad_statistic=ad_stat,

        ad_pvalue=ad_pvalue,

        ad_empirical_p=ad_empirical_p,

        ad_n_extreme_mocks=ad_n_extreme,
    )



def compute_nonparametric_tests(
    w_obs: FloatArray,
    abs_obs: FloatArray,
    all_w_h0: FloatArray,
    all_abs_h0: FloatArray,
    verbose: bool = True,
) -> NonParametricStatistics:
    """
    Heuristic nonparametric isotropy diagnostics.

    Methodology
    -----------
    Following the approach commonly adopted in the
    GRB isotropy literature using:

        - Kolmogorov-Smirnov test
        - Anderson-Darling test

    comparing:

        observed angular profiles
            vs
        isotropic benchmark profiles.

    Notes
    -----
    These tests are complementary qualitative
    diagnostics and do NOT replace the
    covariance-aware inference.

    Angular bins are strongly correlated.
    Therefore:

        - KS/AD p-values should NOT be interpreted
          as rigorous frequentist probabilities;

        - these statistics are best interpreted
          as heuristic profile diagnostics.

    Empirical p-values are calibrated directly
    from the H0 mock ensemble.
    """

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    w_obs = np.asarray(
        w_obs,
        dtype=float,
    )

    abs_obs = np.asarray(
        abs_obs,
        dtype=float,
    )

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    all_abs_h0 = np.asarray(
        all_abs_h0,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if w_obs.ndim != 1:

        raise ValueError(
            "w_obs must be 1-dimensional."
        )

    if abs_obs.ndim != 1:

        raise ValueError(
            "abs_obs must be 1-dimensional."
        )

    if all_w_h0.ndim != 2:

        raise ValueError(
            "all_w_h0 must be 2-dimensional."
        )

    if all_abs_h0.ndim != 2:

        raise ValueError(
            "all_abs_h0 must be 2-dimensional."
        )

    if all_w_h0.shape[1] != len(w_obs):

        raise ValueError(
            "all_w_h0 and w_obs "
            "have incompatible shapes."
        )

    if all_abs_h0.shape[1] != len(abs_obs):

        raise ValueError(
            "all_abs_h0 and abs_obs "
            "have incompatible shapes."
        )

    if not np.any(
        np.isfinite(w_obs)
    ):

        raise ValueError(
            "w_obs contains no finite values."
        )

    if not np.any(
        np.isfinite(abs_obs)
    ):

        raise ValueError(
            "abs_obs contains no finite values."
        )

    # ------------------------------------------------------------------
    # Benchmark reference profiles
    # ------------------------------------------------------------------

    w_reference = np.nanmean(
        all_w_h0,
        axis=0,
    )

    abs_reference = np.nanmean(
        all_abs_h0,
        axis=0,
    )

    # ------------------------------------------------------------------
    # Valid bins
    # ------------------------------------------------------------------

    valid_w = (

        np.isfinite(w_obs)

        &

        np.isfinite(w_reference)

        &

        np.all(
            np.isfinite(all_w_h0),
            axis=0,
        )
    )

    valid_abs = (

        np.isfinite(abs_obs)

        &

        np.isfinite(abs_reference)

        &

        np.all(
            np.isfinite(all_abs_h0),
            axis=0,
        )
    )

    if np.sum(valid_w) < 2:

        raise ValueError(
            "Not enough valid bins for "
            "w(theta) nonparametric tests."
        )

    if np.sum(valid_abs) < 2:

        raise ValueError(
            "Not enough valid bins for "
            "absolute-sum nonparametric tests."
        )

    # ------------------------------------------------------------------
    # Profile tests
    # ------------------------------------------------------------------

    w_stats = (
        _profile_nonparametric_stats(

            observed=w_obs[
                valid_w
            ],

            mocks=all_w_h0[
                :,
                valid_w,
            ],

            reference=w_reference[
                valid_w
            ],

            verbose=False,
        )
    )

    abs_stats = (
        _profile_nonparametric_stats(

            observed=abs_obs[
                valid_abs
            ],

            mocks=all_abs_h0[
                :,
                valid_abs,
            ],

            reference=abs_reference[
                valid_abs
            ],

            verbose=False,
        )
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    if verbose:

        print("\n==================================================")
        print("NONPARAMETRIC ISOTROPY DIAGNOSTICS")
        print("==================================================")

        print(
            "\nNOTE: KS/AD tests are heuristic "
            "diagnostics only because "
            "angular bins are correlated."
        )

        # ==============================================================
        # w(theta)
        # ==============================================================

        print("\n[w(theta)]")

        print(
            f"KS statistic        = "
            f"{w_stats.ks_statistic:.5f}"
        )

        print(
            f"KS p-value [heuristic] = "
            f"{w_stats.ks_pvalue:.5f}"
        )

        print(
            f"KS empirical p      = "
            f"{w_stats.ks_empirical_p:.5f}"
        )

        print(
            f"AD statistic        = "
            f"{w_stats.ad_statistic:.5f}"
        )

        print(
            f"AD p-value [heuristic] = "
            f"{w_stats.ad_pvalue:.5f}"
        )

        print(
            f"AD empirical p      = "
            f"{w_stats.ad_empirical_p:.5f}"
        )

        # ==============================================================
        # Absolute-sum statistic
        # ==============================================================

        print("\n[Absolute sum]")

        print(
            f"KS statistic        = "
            f"{abs_stats.ks_statistic:.5f}"
        )

        print(
            f"KS p-value [heuristic] = "
            f"{abs_stats.ks_pvalue:.5f}"
        )

        print(
            f"KS empirical p      = "
            f"{abs_stats.ks_empirical_p:.5f}"
        )

        print(
            f"AD statistic        = "
            f"{abs_stats.ad_statistic:.5f}"
        )

        print(
            f"AD p-value [heuristic] = "
            f"{abs_stats.ad_pvalue:.5f}"
        )

        print(
            f"AD empirical p      = "
            f"{abs_stats.ad_empirical_p:.5f}"
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return NonParametricStatistics(

        # --------------------------------------------------------------
        # w(theta)
        # --------------------------------------------------------------

        ks_w_stat=(
            w_stats.ks_statistic
        ),

        ks_w_pvalue=(
            w_stats.ks_pvalue
        ),

        ks_w_empirical_p=(
            w_stats.ks_empirical_p
        ),

        ks_w_n_extreme_mocks=(
            w_stats.ks_n_extreme_mocks
        ),

        ad_w_stat=(
            w_stats.ad_statistic
        ),

        ad_w_pvalue=(
            w_stats.ad_pvalue
        ),

        ad_w_empirical_p=(
            w_stats.ad_empirical_p
        ),

        ad_w_n_extreme_mocks=(
            w_stats.ad_n_extreme_mocks
        ),

        # --------------------------------------------------------------
        # Absolute-sum statistic
        # --------------------------------------------------------------

        ks_abs_stat=(
            abs_stats.ks_statistic
        ),

        ks_abs_pvalue=(
            abs_stats.ks_pvalue
        ),

        ks_abs_empirical_p=(
            abs_stats.ks_empirical_p
        ),

        ks_abs_n_extreme_mocks=(
            abs_stats.ks_n_extreme_mocks
        ),

        ad_abs_stat=(
            abs_stats.ad_statistic
        ),

        ad_abs_pvalue=(
            abs_stats.ad_pvalue
        ),

        ad_abs_empirical_p=(
            abs_stats.ad_empirical_p
        ),

        ad_abs_n_extreme_mocks=(
            abs_stats.ad_n_extreme_mocks
        ),
    )


def compute_chi2_statistics(
    delta: FloatArray,
    mock_delta: FloatArray,
    cov: FloatArray,
    hartlap_factor: float,
    dof: int,
) -> ChiSquareStatistics:
    """
    Compute covariance-aware chi-square statistics.
    """

    delta = np.asarray(
        delta,
        dtype=float,
    )

    mock_delta = np.asarray(
        mock_delta,
        dtype=float,
    )

    cov = np.asarray(
        cov,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Inverse covariance
    # --------------------------------------------------------------

    inv_cov = (
        hartlap_factor
        * np.linalg.pinv(cov)
    )

    # --------------------------------------------------------------
    # Observed chi-square
    # --------------------------------------------------------------

    chi2 = float(
        delta
        @ inv_cov
        @ delta
    )

    chi2_red = float(
        chi2 / dof
    )

    p_chi2 = float(
        chi2_dist.sf(
            chi2,
            dof,
        )
    )

    # --------------------------------------------------------------
    # Mock chi-square distribution
    # --------------------------------------------------------------

    chi2_mocks = np.einsum(
        "ij,jk,ik->i",
        mock_delta,
        inv_cov,
        mock_delta,
    )

    chi2_mocks = np.asarray(
        chi2_mocks,
        dtype=float,
    )

    chi2_mocks = chi2_mocks[
        np.isfinite(
            chi2_mocks
        )
    ]

    if len(chi2_mocks) == 0:

        raise ValueError(
            "No valid chi-square mock realizations."
        )

    # --------------------------------------------------------------
    # Empirical significance
    # --------------------------------------------------------------

    n_extreme = int(
        np.sum(
            chi2_mocks >= chi2
        )
    )

    p_empirical_floor = float(
        1.0
        / (
            len(chi2_mocks)
            + 1
        )
    )

    p_empirical = float(
        (
            n_extreme + 1
        )
        / (
            len(chi2_mocks) + 1
        )
    )

    sigma_equiv = (
        sigma_equivalent_from_p(
            p_empirical
        )
    )

    return ChiSquareStatistics(

        chi2=chi2,

        chi2_red=chi2_red,

        p_chi2=p_chi2,

        p_empirical=p_empirical,

        p_empirical_floor=(
            p_empirical_floor
        ),

        sigma_equiv=sigma_equiv,
    )


def compute_absolute_statistics(
    abs_obs: FloatArray,
    all_abs_h0: FloatArray,
) -> AbsoluteStatistics:
    """
    Compute empirical significance of the
    tomographic absolute-sum statistic.
    """

    abs_obs = np.asarray(
        abs_obs,
        dtype=float,
    )

    all_abs_h0 = np.asarray(
        all_abs_h0,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Observed statistic
    # --------------------------------------------------------------

    abs_obs_valid = abs_obs[
        np.isfinite(abs_obs)
    ]

    if len(abs_obs_valid) == 0:

        raise ValueError(
            "abs_obs contains no finite values."
        )

    abs_observed_total = float(
        np.sum(
            abs_obs_valid
        )
    )

    # --------------------------------------------------------------
    # Mock statistics
    # --------------------------------------------------------------

    abs_mock_totals = np.asarray(
        [
            np.sum(
                row[
                    np.isfinite(row)
                ]
            )
            for row in all_abs_h0
        ],
        dtype=float,
    )

    abs_mock_totals = abs_mock_totals[
        np.isfinite(
            abs_mock_totals
        )
    ]

    if len(abs_mock_totals) == 0:

        raise ValueError(
            "No valid absolute-statistic "
            "mock realizations."
        )

    # --------------------------------------------------------------
    # Empirical significance
    # --------------------------------------------------------------

    abs_empirical_p = float(
        (
            np.sum(
                abs_mock_totals
                >= abs_observed_total
            )
            + 1
        )
        / (
            len(abs_mock_totals)
            + 1
        )
    )

    # --------------------------------------------------------------
    # Normalized global deviation
    # --------------------------------------------------------------

    abs_mock_mean = float(
        np.mean(
            abs_mock_totals
        )
    )

    abs_mock_std = float(
        np.std(
            abs_mock_totals,
            ddof=1,
        )
    ) if len(abs_mock_totals) > 1 else 0.0

    if (
        np.isfinite(abs_mock_std)
        and abs_mock_std > 0.0
    ):

        global_tension = float(
            abs(
                abs_observed_total
                - abs_mock_mean
            )
            / abs_mock_std
        )

    elif np.isclose(
        abs_observed_total,
        abs_mock_mean,
    ):

        global_tension = 0.0

    else:

        global_tension = np.inf

    return AbsoluteStatistics(

        abs_observed_stat=(
            abs_observed_total
        ),

        abs_empirical_p=(
            abs_empirical_p
        ),

        global_tension=global_tension,
    )


def compute_statistics(
    context: RuntimeContext,
    w_obs: FloatArray,
    abs_obs: FloatArray,
    all_w_h0: FloatArray,
    all_abs_h0: FloatArray,
    label: str | None = None,
) -> TestStatistics:
    """
    Perform the full covariance-aware isotropy analysis
    against the H0 ensemble.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if label is None:

        label = (
            f"FRB isotropy "
            f"({config.run_tag})"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("GLOBAL ISOTROPY INFERENCE")
    print("==================================================")

    print(
        f"RUN_TAG = "
        f"{config.run_tag}"
    )

    print(
        f"USE_GAL_MASK = "
        f"{config.use_gal_mask}"
    )

    print(
        f"USE_SEL_FUNC = "
        f"{config.use_sel_func}"
    )

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    w_obs = np.asarray(
        w_obs,
        dtype=float,
    )

    abs_obs = np.asarray(
        abs_obs,
        dtype=float,
    )

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    all_abs_h0 = np.asarray(
        all_abs_h0,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if w_obs.ndim != 1:

        raise ValueError(
            "w_obs must be 1-dimensional."
        )

    if abs_obs.ndim != 1:

        raise ValueError(
            "abs_obs must be 1-dimensional."
        )

    if all_w_h0.ndim != 2:

        raise ValueError(
            "all_w_h0 must be 2-dimensional."
        )

    if all_abs_h0.ndim != 2:

        raise ValueError(
            "all_abs_h0 must be 2-dimensional."
        )

    if all_w_h0.shape[1] != len(w_obs):

        raise ValueError(
            "all_w_h0 and w_obs "
            "have incompatible shapes."
        )

    if all_abs_h0.shape[1] != len(abs_obs):

        raise ValueError(
            "all_abs_h0 and abs_obs "
            "have incompatible shapes."
        )

    # ------------------------------------------------------------------
    # Valid covariance bins
    # ------------------------------------------------------------------

    valid_bins = (

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
            "for covariance inference."
        )

    w_obs_valid = w_obs[
        valid_bins
    ]

    all_w_h0_valid = all_w_h0[
        :,
        valid_bins,
    ]

    # ------------------------------------------------------------------
    # H0 reference profile
    # ------------------------------------------------------------------

    h0_mean = np.mean(
        all_w_h0_valid,
        axis=0,
    )

    print(
        f"\nMax |H0 mean| = "
        f"{np.nanmax(np.abs(h0_mean)):.4e}"
    )

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------

    cov = build_shrinkage_covariance(
        all_w_h0_valid
    )

    plot_covariance_diagnostics(
        context=context,
        cov=cov,
        all_w_h0=all_w_h0_valid,
    )

    from ..reporting.exports import (
        export_covariance_eigenvalues,
    )

    export_covariance_eigenvalues(
        cov=cov,
        output=(
            context.outputs.tables_dir
            / f"{config.run_tag}_covariance_eigenvalues.csv"
        ),
    )

    covariance_diag = (
        compute_covariance_diagnostics(
            cov=cov,
            n_mocks=len(
                all_w_h0_valid
            ),
            n_bins=(
                all_w_h0_valid.shape[1]
            ),
        )
    )

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    delta = (
        w_obs_valid
        - h0_mean
    )

    mock_delta = (
        all_w_h0_valid
        - h0_mean
    )

    # ------------------------------------------------------------------
    # SVD statistic
    # ------------------------------------------------------------------

    svd_stats = (
        compute_svd_regularized_chi2(
            context=context,
            delta=delta,
            mock_delta=mock_delta,
            cov=cov,
            hartlap_factor=(
                covariance_diag.hartlap_factor
            ),
            eigenvalue_cut=(
                config.svd_eigenvalue_cut
            ),
        )
    )

    covariance_diag = compute_covariance_diagnostics(
        cov=cov,
        n_mocks=len(
            all_w_h0_valid
        ),
        n_bins=all_w_h0_valid.shape[1],
        svd_modes_kept=svd_stats.modes_kept,
        svd_eigenvalue_cut=svd_stats.svd_eigenvalue_cut,
        svd_condition=svd_stats.svd_condition,
    )

    # ------------------------------------------------------------------
    # Full chi-square
    # ------------------------------------------------------------------

    chi2_stats = (
        compute_chi2_statistics(
            delta=delta,
            mock_delta=mock_delta,
            cov=cov,
            hartlap_factor=covariance_diag.hartlap_factor,
            dof=int(
                all_w_h0_valid.shape[1]
            ),
        )
    )

    # ------------------------------------------------------------------
    # Absolute statistic
    # ------------------------------------------------------------------

    absolute_stats = (
        compute_absolute_statistics(
            abs_obs=abs_obs,
            all_abs_h0=all_abs_h0,
        )
    )

    # ------------------------------------------------------------------
    # Nonparametric tests
    # ------------------------------------------------------------------

    nonparam_stats = (
        compute_nonparametric_tests(
            w_obs=w_obs,
            abs_obs=abs_obs,
            all_w_h0=all_w_h0,
            all_abs_h0=all_abs_h0,
            verbose=False,
        )
    )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    from ..analysis.diagnostics import (
        print_statistics_summary,
        print_physical_interpretation,
    )

    print_statistics_summary(
        label=label,
        chi2_stats=chi2_stats,
        svd_stats=svd_stats,
        covariance_diag=covariance_diag,
        absolute_stats=absolute_stats,
        n_mocks=(
            all_w_h0_valid.shape[0]
        ),
        n_bins=(
            all_w_h0_valid.shape[1]
        ),
    )

    print_physical_interpretation(
        context=context,
        chi2_stats=chi2_stats,
    )

    covariance_result = CovarianceResult(
        matrix=np.asarray(
            cov,
            dtype=float,
        ).copy(),
        mean_profile=np.asarray(
            h0_mean,
            dtype=float,
        ).copy(),
        valid_bins=np.asarray(
            valid_bins,
            dtype=bool,
        ).copy(),
        diagnostics=covariance_diag,
        estimator="LedoitWolf",
        n_mocks=int(
            all_w_h0_valid.shape[0]
        ),
        n_bins=int(
            all_w_h0_valid.shape[1]
        ),
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return TestStatistics(

        chi2=chi2_stats,

        svd=svd_stats,

        nonparametric=(
            nonparam_stats
        ),

        absolute=(
            absolute_stats
        ),


        n_mocks=(
            all_w_h0_valid.shape[0]
        ),

        n_bins=(
            all_w_h0_valid.shape[1]
        ),

        covariance_result=covariance_result,
    )
