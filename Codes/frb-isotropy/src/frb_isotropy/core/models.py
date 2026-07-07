from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import healpy as hp

from .types import FloatArray, BoolArray


# ==============================================================================
# Selection-function container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class SelectionFunctionSet:
    """
    Immutable container holding the complete
    survey selection-function model.
    """

    sf_dict: dict[
        str,
        FloatArray,
    ]

    sf_cov_diag_dict: dict[
        str,
        FloatArray,
    ]

    survey_weights: dict[
        str,
        float,
    ]

    nside: int

    # ==================================================================
    # Automatic validation
    # ==================================================================

    def __post_init__(
        self,
    ) -> None:

        self.validate()

    # ==================================================================
    # Derived properties
    # ==================================================================

    @property
    def surveys(
        self,
    ) -> tuple[str, ...]:

        return tuple(
            self.sf_dict.keys()
        )

    @property
    def n_surveys(
        self,
    ) -> int:

        return len(
            self.sf_dict
        )

    @property
    def npix(
        self,
    ) -> int:

        return hp.nside2npix(
            self.nside
        )

    # ==================================================================
    # Validation
    # ==================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate internal consistency of the
        selection-function model.
        """

        # --------------------------------------------------------------
        # HEALPix validation
        # --------------------------------------------------------------

        if not hp.isnsideok(
            self.nside
        ):

            raise ValueError(
                f"Invalid HEALPix nside: "
                f"{self.nside}"
            )

        # --------------------------------------------------------------
        # Survey consistency
        # --------------------------------------------------------------

        surveys_sf = set(
            self.sf_dict.keys()
        )

        surveys_cov = set(
            self.sf_cov_diag_dict.keys()
        )

        surveys_weights = set(
            self.survey_weights.keys()
        )

        if (
            surveys_sf
            != surveys_cov
        ):

            raise ValueError(
                "Mismatch between sf_dict and "
                "sf_cov_diag_dict survey sets."
            )

        if (
            surveys_sf
            != surveys_weights
        ):

            raise ValueError(
                "Mismatch between sf_dict and "
                "survey_weights survey sets."
            )

        # --------------------------------------------------------------
        # Weight normalization
        # --------------------------------------------------------------

        total_weight = sum(
            self.survey_weights.values()
        )

        if not np.isfinite(
            total_weight
        ):

            raise ValueError(
                "Non-finite survey weights."
            )

        if total_weight <= 0:

            raise ValueError(
                "Survey weights sum to zero."
            )

        # --------------------------------------------------------------
        # Per-survey validation
        # --------------------------------------------------------------

        npix_expected = self.npix

        for survey in self.surveys:

            sf = np.asarray(
                self.sf_dict[survey],
                dtype=float,
            )

            cov = np.asarray(
                self.sf_cov_diag_dict[survey],
                dtype=float,
            )

            # ----------------------------------------------------------
            # Shape checks
            # ----------------------------------------------------------

            if len(sf) != npix_expected:

                raise ValueError(
                    f"Invalid map size for survey "
                    f"{survey}."
                )

            if sf.shape != cov.shape:

                raise ValueError(
                    f"Shape mismatch for survey "
                    f"{survey}."
                )

            # ----------------------------------------------------------
            # Numerical checks
            # ----------------------------------------------------------

            if not np.all(
                np.isfinite(sf)
            ):

                raise ValueError(
                    f"Non-finite SF values for "
                    f"survey {survey}."
                )

            if not np.all(
                np.isfinite(cov)
            ):

                raise ValueError(
                    f"Non-finite covariance values "
                    f"for survey {survey}."
                )

            if np.any(sf < 0):

                raise ValueError(
                    f"Negative probabilities in "
                    f"survey {survey}."
                )

            if np.any(cov < 0):

                raise ValueError(
                    f"Negative covariance entries "
                    f"in survey {survey}."
                )

            # ----------------------------------------------------------
            # Probability normalization
            # ----------------------------------------------------------

            sf_sum = sf.sum()

            if not np.isclose(
                sf_sum,
                1.0,
                rtol=0.0,
                atol=1e-8,
            ):

                raise ValueError(
                    f"Selection function for "
                    f"survey {survey} is not "
                    f"normalized."
                )
            

# ==============================================================================
# Intersurvey correlations and overlap analysis container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class IntersurveyAnalysis:

    correlation_matrix: pd.DataFrame

    overlap_matrix: pd.DataFrame

    mean_correlation: float

    max_correlation: float

    mean_overlap: float

    max_overlap: float


# ==============================================================================
# Mock Ensemble container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class MockEnsemble:

    w_theta: FloatArray

    abs_statistics: FloatArray

    metadata: dict[
        str,
        int | float | str | bool
    ]


# ==============================================================================
# SVD statistics container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class SVDStatistics:

    chi2_svd: float

    chi2_svd_red: float

    p_chi2_svd: float

    p_svd_empirical: float

    p_svd_empirical_floor: float

    sigma_svd_equiv: float

    modes_kept: int

    svd_condition: float

    svd_eigenvalue_cut: float

    explained_variance_fraction: float


# ==============================================================================
# Nonparametric statistics container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class ProfileNonParametricStatistics:
    """
    Nonparametric statistics for a single profile.
    """

    ks_statistic: float

    ks_pvalue: float

    ks_empirical_p: float

    ks_n_extreme_mocks: int

    ad_statistic: float

    ad_pvalue: float

    ad_empirical_p: float

    ad_n_extreme_mocks: int


@dataclass(
    frozen=True,
    slots=True,
)
class NonParametricStatistics:
    """
    Nonparametric goodness-of-fit statistics.
    """

    # ==========================================================
    # w(theta)
    # ==========================================================

    ks_w_stat: float

    ks_w_pvalue: float

    ks_w_empirical_p: float

    ks_w_n_extreme_mocks: int

    ad_w_stat: float

    ad_w_pvalue: float

    ad_w_empirical_p: float

    ad_w_n_extreme_mocks: int

    # ==========================================================
    # Absolute-sum statistic
    # ==========================================================

    ks_abs_stat: float

    ks_abs_pvalue: float

    ks_abs_empirical_p: float

    ks_abs_n_extreme_mocks: int

    ad_abs_stat: float

    ad_abs_pvalue: float

    ad_abs_empirical_p: float

    ad_abs_n_extreme_mocks: int


# ==============================================================================
# Chi-square statistics container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class ChiSquareStatistics:

    chi2: float

    chi2_red: float

    p_chi2: float

    p_empirical: float

    p_empirical_floor: float

    sigma_equiv: float


# ==============================================================================
# Absolute sum test statistics container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class AbsoluteStatistics:

    abs_observed_stat: float

    abs_empirical_p: float

    global_tension: float


# ==============================================================================
# Covariance diagnostics container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class CovarianceDiagnostics:

    hartlap_factor: float

    covariance_rank: int

    covariance_condition: float

    n_eff: float

    svd_modes_kept: int

    svd_eigenvalue_cut: float

    svd_condition: float


# ==============================================================================
# Covariance inference result container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class CovarianceResult:

    matrix: FloatArray

    mean_profile: FloatArray

    valid_bins: BoolArray

    diagnostics: CovarianceDiagnostics

    estimator: str

    n_mocks: int

    n_bins: int


# ==============================================================================
# Full statistical inference container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class TestStatistics:

    chi2: ChiSquareStatistics

    svd: SVDStatistics

    nonparametric: NonParametricStatistics

    absolute: AbsoluteStatistics

    covariance_result: CovarianceResult

    n_mocks: int

    n_bins: int

# ==============================================================================
# Jackknife container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True)
class JackknifeResult:
    """
    Container for jackknife uncertainty estimation.
    """

    regions: np.ndarray

    theta: np.ndarray

    w_samples: np.ndarray

    w_mean: np.ndarray

    w_cov: np.ndarray

    w_err: np.ndarray

    abs_samples: np.ndarray

    abs_mean: np.ndarray

    abs_cov: np.ndarray

    abs_err: np.ndarray


# ==============================================================================
# Bootstrap container
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True)
class BootstrapResult:
    """
    Container for bootstrap uncertainty estimation.
    """

    w_mean: np.ndarray

    w_err: np.ndarray

    w_cov: np.ndarray

    abs_mean: np.ndarray

    abs_err: np.ndarray

    abs_cov: np.ndarray

    w_realizations: np.ndarray

    abs_realizations: np.ndarray


@dataclass(frozen=True, slots=True)
class InjectionResult:
    """Result of one injection-recovery amplitude."""

    epsilon_injected: float
    multipole: int

    signal_no_sf: float
    signal_fixed_sf: float
    signal_rebuilt_sf: float

    signal_no_sf_std: float
    signal_fixed_sf_std: float
    signal_rebuilt_sf_std: float

    retention_fixed_vs_no_sf: float
    retention_rebuilt_vs_fixed: float
    absorption_fraction: float

    raw_no_sf: float
    raw_fixed_sf: float
    raw_rebuilt_sf: float

    baseline_no_sf: float
    baseline_fixed_sf: float
    baseline_rebuilt_sf: float

    w_obs_no_sf: FloatArray
    w_obs_fixed_sf: FloatArray
    w_obs_rebuilt_sf: FloatArray
    theta: FloatArray

    signal_fixed_instrumental_sf: float = float("nan")
    signal_fixed_instrumental_sf_std: float = float("nan")
    retention_instrumental_vs_no_sf: float = float("nan")
    retention_instrumental_vs_fixed: float = float("nan")
    absorption_instrumental_vs_no_sf: float = float("nan")
    raw_fixed_instrumental_sf: float = float("nan")
    baseline_fixed_instrumental_sf: float = float("nan")
    w_obs_fixed_instrumental_sf: FloatArray | None = None

    @property
    def epsilon_recovered_nosf(self) -> float:
        """Backward-compatible alias for older notebooks."""

        return self.signal_no_sf

    @property
    def epsilon_recovered_sf(self) -> float:
        """Backward-compatible alias for the rebuilt empirical-SF mode."""

        return self.signal_rebuilt_sf

    @property
    def suppression_factor(self) -> float:
        """Backward-compatible alias: rebuilt-SF retention relative to fixed SF."""

        return self.retention_rebuilt_vs_fixed

@dataclass(frozen=True, slots=True)
class InjectionSuite:
    """Set of results for multiple injected amplitudes."""
    
    multipole: int
    epsilons: FloatArray
    suppression_factors: FloatArray
    results: list[InjectionResult]
