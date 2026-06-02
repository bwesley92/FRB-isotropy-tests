# ==============================================================================
# FRB isotropy analysis pipeline
# ==============================================================================
#
# Fully stateless scientific pipeline architecture
#
# Key design principles:
#
#   - No mutable global state
#   - No hidden configuration
#   - No implicit filesystem dependencies
#   - No module-level runtime configuration
#   - Fully reproducible execution
#   - HPC-safe
#   - Notebook-controlled execution
#
# ==============================================================================


# ==============================================================================
# Future annotations
# ==============================================================================

from __future__ import annotations


# ==============================================================================
# Standard library
# ==============================================================================

import os
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias


# ==============================================================================
# HPC thread control
# ==============================================================================

_HPC_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

for _var, _value in _HPC_ENV_VARS.items():

    os.environ.setdefault(
        _var,
        _value,
    )


# ==============================================================================
# Third-party imports
# ==============================================================================

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import treecorr

from astropy.coordinates import SkyCoord

from joblib import Parallel, delayed

from matplotlib.patches import Patch

from numpy.random import Generator
from numpy.typing import NDArray

from scipy.stats import anderson_ksamp
from scipy.stats import chi2 as chi2_dist
from scipy.stats import ks_2samp
from scipy.stats import norm

from sklearn.covariance import LedoitWolf

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib


# ==============================================================================
# Array typing aliases
# ==============================================================================

FloatArray: TypeAlias = NDArray[np.float64]

IntArray: TypeAlias = NDArray[np.int_]

BoolArray: TypeAlias = NDArray[np.bool_]


# ==============================================================================
# Warning suppression
# ==============================================================================

def suppress_library_warnings() -> None:
    """Suppress known library warnings for cleaner output."""
    warnings.filterwarnings(
        "ignore",
        message='.*"verbose" was deprecated.*',
        category=Warning,
    )

# ==============================================================================
# Utilities
# ==============================================================================

def _format_tag_value(
    value: float | int | str,
) -> str:
    """
    Format configuration values into filesystem-safe tags.
    """

    text = (
        f"{value:g}"
        if isinstance(value, float)
        else str(value)
    )

    return (
        text
        .replace("-", "m")
        .replace(".", "p")
    )


# ==============================================================================
# Output paths
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class OutputPaths:
    """
    Filesystem layout for one analysis run.
    """

    run_tag: str

    run_dir: Path

    figures_dir: Path
    tables_dir: Path
    report_dir: Path

    summary_csv: Path
    diagnostics_xlsx: Path
    report_md: Path
    results_pkl: Path


# ==============================================================================
# Analysis configuration
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class AnalysisConfig:
    """
    Immutable configuration describing one isotropy-analysis run.

    Entire pipeline behavior is controlled explicitly through
    this configuration object.

    No implicit global defaults are allowed.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    name: str

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------

    project_root: Path

    outputs_root: Path

    catalog_path: Path

    # ------------------------------------------------------------------
    # Pipeline modes
    # ------------------------------------------------------------------

    use_gal_mask: bool
    use_sel_func: bool

    # ------------------------------------------------------------------
    # Parallel processing
    # ------------------------------------------------------------------

    n_jobs: int

    # ------------------------------------------------------------------
    # Galactic mask
    # ------------------------------------------------------------------

    gal_cut: float

    # ------------------------------------------------------------------
    # 2pACF configuration
    # ------------------------------------------------------------------

    min_sep: float
    max_sep: float
    bin_size: float

    # ------------------------------------------------------------------
    # Absolute anisotropy statistic
    # ------------------------------------------------------------------

    coarse_bins: tuple[float, ...]

    # ------------------------------------------------------------------
    # Selection-function modeling
    # ------------------------------------------------------------------

    nside_sf: int
    smooth_sigma: float
    perturbation_scale: float

    # ------------------------------------------------------------------
    # Random catalogs
    # ------------------------------------------------------------------

    n_rand_factor: int

    # ------------------------------------------------------------------
    # Mock generation
    # ------------------------------------------------------------------

    n_ensemble: int
    n_mocks_per_ensemble: int

    # ------------------------------------------------------------------
    # RNG seeds
    # ------------------------------------------------------------------

    random_seed: int = 12345

    mock_seed_base: int = 100_000

    random_catalog_seed_base: int = 200_000

    sf_perturbation_seed_base: int = 300_000

    jackknife_seed_base: int = 400_000

    bootstrap_seed_base: int = 500_000

    # ------------------------------------------------------------------
    # Numerical stability
    # ------------------------------------------------------------------

    selection_function_floor: float = 1e-12

    numerical_eigenvalue_floor: float = 1e-15

    stable_eigenvalue_floor: float = 1e-12

    numerical_zero_tolerance: float = 1e-30

    # ------------------------------------------------------------------
    # Covariance regularization
    # ------------------------------------------------------------------

    svd_eigenvalue_cut: float = 1e-2

    # ------------------------------------------------------------------
    # Jackknife
    # ------------------------------------------------------------------

    nside_jackknife: int = 4

    min_jackknife_regions: int = 20

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    n_bootstrap: int = 500

    # ------------------------------------------------------------------
    # Overlap analysis
    # ------------------------------------------------------------------

    overlap_radius_deg: float = 5.0

    overlap_nside: int = 32

    # ==================================================================
    # Derived properties
    # ==================================================================

    @property
    def mask_tag(
        self,
    ) -> str:

        return (
            "mask"
            if self.use_gal_mask
            else "nomask"
        )

    @property
    def sf_tag(
        self,
    ) -> str:

        return (
            "sf"
            if self.use_sel_func
            else "nosf"
        )

    @property
    def run_tag(
        self,
    ) -> str:

        return (
            f"{self.name}_"
            f"{self.mask_tag}_"
            f"{self.sf_tag}_"
            f"gal{_format_tag_value(self.gal_cut)}_"
            f"bin{_format_tag_value(self.bin_size)}_"
            f"nside{self.nside_sf}_"
            f"smooth{_format_tag_value(self.smooth_sigma)}"
        )

    @property
    def angular_range(
        self,
    ) -> float:

        return (
            self.max_sep
            - self.min_sep
        )

    @property
    def n_bins(
        self,
    ) -> int:

        n_bins_float = (
            self.angular_range
            / self.bin_size
        )

        n_bins_int = int(
            round(n_bins_float)
        )

        if not np.isclose(
            n_bins_float,
            n_bins_int,
            rtol=0.0,
            atol=1e-10,
        ):

            raise ValueError(
                "Angular binning is inconsistent."
            )

        if n_bins_int < 1:

            raise ValueError(
                "n_bins < 1."
            )

        return n_bins_int

    @property
    def coarse_bins_array(
        self,
    ) -> FloatArray:

        return np.asarray(
            self.coarse_bins,
            dtype=float,
        )

    @property
    def coarse_centers(
        self,
    ) -> FloatArray:

        bins = self.coarse_bins_array

        return 0.5 * (
            bins[:-1]
            + bins[1:]
        )

    @property
    def n_mocks(
        self,
    ) -> int:

        return (
            self.n_ensemble
            * self.n_mocks_per_ensemble
        )


# ==============================================================================
# Runtime context
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class RuntimeContext:
    """
    Immutable execution context.

    Every pipeline function receives this object explicitly.

    No hidden module state exists anywhere.
    """

    config: AnalysisConfig

    outputs: OutputPaths


# ==============================================================================
# Output path construction
# ==============================================================================

def build_output_paths(
    config: AnalysisConfig,
) -> OutputPaths:
    """
    Construct all filesystem paths for one analysis run.
    """

    run_dir = (
        config.outputs_root
        / config.run_tag
    )

    figures_dir = (
        run_dir
        / "figures"
    )

    tables_dir = (
        run_dir
        / "tables"
    )

    report_dir = (
        run_dir
        / "report"
    )

    return OutputPaths(

        run_tag=config.run_tag,

        run_dir=run_dir,

        figures_dir=figures_dir,
        tables_dir=tables_dir,
        report_dir=report_dir,

        summary_csv=(
            tables_dir
            / "summary.csv"
        ),

        diagnostics_xlsx=(
            tables_dir
            / "diagnostics.xlsx"
        ),

        report_md=(
            report_dir
            / "summary.md"
        ),

        results_pkl=(
            run_dir
            / "results.pkl"
        ),
    )


# ==============================================================================
# Output directory initialization
# ==============================================================================

def initialize_output_directories(
    outputs: OutputPaths,
) -> None:
    """
    Create output directories.
    """

    directories = (
        outputs.run_dir,
        outputs.figures_dir,
        outputs.tables_dir,
        outputs.report_dir,
    )

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ==============================================================================
# Intersurvey analysis container
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

        npix_expected = hp.nside2npix(
            self.nside
        )

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
# Absolute anisotropy statistics container
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

    covariance: CovarianceDiagnostics

    n_mocks: int

    n_bins: int


@dataclass(slots=True)
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


@dataclass(slots=True)
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




# ==============================================================================
# Configuration validation
# ==============================================================================

def validate_analysis_config(
    config: AnalysisConfig,
) -> None:
    """
    Validate configuration consistency.
    """

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------

    if not config.project_root.exists():

        raise FileNotFoundError(
            f"project_root does not exist:\n"
            f"{config.project_root}"
        )

    if not config.catalog_path.exists():

        raise FileNotFoundError(
            f"catalog_path does not exist:\n"
            f"{config.catalog_path}"
        )

    # ------------------------------------------------------------------
    # Angular configuration
    # ------------------------------------------------------------------

    if config.max_sep <= config.min_sep:

        raise ValueError(
            "max_sep must exceed min_sep."
        )

    if config.bin_size <= 0:

        raise ValueError(
            "bin_size must be positive."
        )

    _ = config.n_bins

    # ------------------------------------------------------------------
    # Numerical stability
    # ------------------------------------------------------------------

    if config.selection_function_floor <= 0:

        raise ValueError(
            "selection_function_floor "
            "must be positive."
        )

    if config.numerical_eigenvalue_floor <= 0:

        raise ValueError(
            "numerical_eigenvalue_floor "
            "must be positive."
        )

    if config.stable_eigenvalue_floor <= 0:

        raise ValueError(
            "stable_eigenvalue_floor "
            "must be positive."
        )

    # ------------------------------------------------------------------
    # RNG validation
    # ------------------------------------------------------------------

    if config.random_seed < 0:

        raise ValueError(
            "random_seed must be non-negative."
        )

    # ------------------------------------------------------------------
    # HEALPix
    # ------------------------------------------------------------------

    if not hp.isnsideok(config.nside_sf):

        raise ValueError(
            "Invalid nside_sf."
        )

    if not hp.isnsideok(
        config.nside_jackknife
    ):

        raise ValueError(
            "Invalid nside_jackknife."
        )

    if not hp.isnsideok(
        config.overlap_nside
    ):

        raise ValueError(
            "Invalid overlap_nside."
        )

    # ------------------------------------------------------------------
    # Positive parameters
    # ------------------------------------------------------------------

    positive_parameters = [

        config.n_jobs,

        config.n_rand_factor,

        config.n_ensemble,

        config.n_mocks_per_ensemble,

        config.n_bootstrap,

        config.overlap_radius_deg,

        config.svd_eigenvalue_cut,
    ]

    if any(x <= 0 for x in positive_parameters):

        raise ValueError(
            "All positive-definite parameters "
            "must be > 0."
        )


# ==============================================================================
# Runtime context creation
# ==============================================================================

def create_runtime_context(
    config: AnalysisConfig,
) -> RuntimeContext:
    """
    Create immutable runtime context.
    """

    validate_analysis_config(
        config
    )

    outputs = build_output_paths(
        config
    )

    initialize_output_directories(
        outputs
    )

    return RuntimeContext(
        config=config,
        outputs=outputs,
    )


# ==============================================================================
# Logging
# ==============================================================================

def log_runtime_configuration(
    context: RuntimeContext,
) -> None:
    """
    Print complete runtime configuration.
    """

    config = context.config

    print("\n==================================================")
    print("FRB ISOTROPY PIPELINE")
    print("==================================================")

    # ------------------------------------------------------
    # Core paths
    # ------------------------------------------------------

    print(
        f"RUN TAG                  : "
        f"{config.run_tag}"
    )

    print(
        f"PROJECT ROOT             : "
        f"{config.project_root}"
    )

    print(
        f"CATALOG PATH             : "
        f"{config.catalog_path}"
    )

    print(
        f"OUTPUT ROOT              : "
        f"{config.outputs_root}"
    )

    # ------------------------------------------------------
    # Physical model
    # ------------------------------------------------------

    print(
        f"USE GAL MASK             : "
        f"{config.use_gal_mask}"
    )

    print(
        f"USE SEL FUNC             : "
        f"{config.use_sel_func}"
    )

    # ------------------------------------------------------
    # Parallelization
    # ------------------------------------------------------

    print(
        f"N JOBS                   : "
        f"{config.n_jobs}"
    )

    # ------------------------------------------------------
    # Galactic masking
    # ------------------------------------------------------

    print(
        f"GAL CUT                  : "
        f"{config.gal_cut}"
    )

    # ------------------------------------------------------
    # Angular statistics
    # ------------------------------------------------------

    print(
        f"MIN SEP                  : "
        f"{config.min_sep}"
    )

    print(
        f"MAX SEP                  : "
        f"{config.max_sep}"
    )

    print(
        f"BIN SIZE                 : "
        f"{config.bin_size}"
    )

    print(
        f"N BINS                   : "
        f"{config.n_bins}"
    )

    print(
        f"COARSE BINS              : "
        f"{config.coarse_bins}"
    )

    # ------------------------------------------------------
    # Selection functions
    # ------------------------------------------------------

    print(
        f"NSIDE SF                 : "
        f"{config.nside_sf}"
    )

    print(
        f"SMOOTH SIGMA             : "
        f"{config.smooth_sigma}"
    )

    # ------------------------------------------------------
    # H0 perturbations
    # ------------------------------------------------------

    print(
        f"PERTURBATION SCALE       : "
        f"{config.perturbation_scale}"
    )

    # ------------------------------------------------------
    # Random catalogs
    # ------------------------------------------------------

    print(
        f"N RAND FACTOR            : "
        f"{config.n_rand_factor}"
    )

    # ------------------------------------------------------
    # H0 ensemble
    # ------------------------------------------------------

    print(
        f"N ENSEMBLE               : "
        f"{config.n_ensemble}"
    )

    print(
        f"N MOCKS PER ENSEMBLE     : "
        f"{config.n_mocks_per_ensemble}"
    )

    print(
        f"TOTAL H0 MOCKS           : "
        f"{config.n_mocks}"
    )

    # ------------------------------------------------------
    # Covariance regularization
    # ------------------------------------------------------

    print(
        f"SVD EIGENVALUE CUT       : "
        f"{config.svd_eigenvalue_cut}"
    )

    # ------------------------------------------------------
    # Jackknife
    # ------------------------------------------------------

    print(
        f"NSIDE JACKKNIFE          : "
        f"{config.nside_jackknife}"
    )

    print(
        f"MIN JK REGIONS           : "
        f"{config.min_jackknife_regions}"
    )

    # ------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------

    print(
        f"N BOOTSTRAP              : "
        f"{config.n_bootstrap}"
    )

    # ------------------------------------------------------
    # Overlap analysis
    # ------------------------------------------------------

    print(
        f"OVERLAP RADIUS           : "
        f"{config.overlap_radius_deg}"
    )

    print(
        f"OVERLAP NSIDE            : "
        f"{config.overlap_nside}"
    )

    print("==================================================")


# ==============================================================================
# Catalog loading and sky geometry
# ==============================================================================

_REQUIRED_CATALOG_COLUMNS = (
    "RA",
    "DEC",
    "Reporting_Group_s",
)


def load_catalog(
    context: RuntimeContext,
) -> pd.DataFrame:
    """
    Load FRB catalog and standardize column names.
    """

    path = context.config.catalog_path

    df = pd.read_csv(path)

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    missing = [

        col
        for col
        in _REQUIRED_CATALOG_COLUMNS

        if col not in df.columns
    ]

    if len(missing) > 0:

        raise ValueError(
            f"Missing catalog columns:\n"
            f"{missing}"
        )

    df = (
        df[
            [
                "RA",
                "DEC",
                "Reporting_Group_s",
            ]
        ]
        .dropna()
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Coordinate validation
    # ------------------------------------------------------------------

    df["RA"] = pd.to_numeric(
        df["RA"],
        errors="coerce",
    )

    df["DEC"] = pd.to_numeric(
        df["DEC"],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[
                "RA",
                "DEC",
            ]
        )
        .reset_index(drop=True)
    )

    valid = (

        (df["RA"] >= 0.0)
        &
        (df["RA"] < 360.0)
        &
        (df["DEC"] >= -90.0)
        &
        (df["DEC"] <= 90.0)
    )

    df = (
        df.loc[valid]
        .reset_index(drop=True)
    )

    if len(df) == 0:

        raise ValueError(
            "Catalog is empty after validation."
        )

    print("\n--- Catalog loaded ---")

    print(
        f"Objects: {len(df)}"
    )

    return df


def apply_mask(
    context: RuntimeContext,
    df: pd.DataFrame,
    gal_cut: float | None = None,
    use_mask: bool | None = None,
) -> pd.DataFrame:
    """
    Apply Galactic latitude mask to observed catalog.

    If the active configuration disables the mask,
    return the original catalog unchanged.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if use_mask is None:

        use_mask = (
            config.use_gal_mask
        )

    if gal_cut is None:

        gal_cut = (
            config.gal_cut
        )

    # ------------------------------------------------------------------
    # Disabled mode
    # ------------------------------------------------------------------

    if use_mask is False:

        print(
            "\n--- Galactic mask disabled ---"
        )

        return (
            df
            .reset_index(drop=True)
            .copy()
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if gal_cut < 0:

        raise ValueError(
            "gal_cut must be non-negative."
        )

    # ------------------------------------------------------------------
    # Sky coordinates
    # ------------------------------------------------------------------

    coords = SkyCoord(

        ra=(
            df["RA"].to_numpy(float)
            * u.degree
        ),

        dec=(
            df["DEC"].to_numpy(float)
            * u.degree
        ),

        frame="icrs",
    )

    gal_b = np.asarray(

        coords
        .galactic
        .b
        .degree,

        dtype=float,
    )

    # ------------------------------------------------------------------
    # Galactic mask
    # ------------------------------------------------------------------

    mask = (
        np.abs(gal_b)
        > gal_cut
    )

    df_masked = (
        df.loc[mask]
        .reset_index(drop=True)
        .copy()
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print(
        "\n--- Galactic mask applied ---"
    )

    print(
        f"gal_cut = ±{gal_cut:.1f} deg"
    )

    print(
        f"Remaining objects: "
        f"{len(df_masked)}"
    )

    print(
        f"Removed objects: "
        f"{len(df) - len(df_masked)}"
    )

    return df_masked


def healpix_galactic_mask(
    context: RuntimeContext,
    nside: int,
    gal_cut: float | None = None,
    use_mask: bool | None = None,
) -> BoolArray:
    """
    Build Galactic mask in HEALPix space.

    Returns
    -------
    BoolArray
        Boolean mask:
            True  -> usable pixel
            False -> masked pixel
    """

    config = context.config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if use_mask is None:

        use_mask = (
            config.use_gal_mask
        )

    if gal_cut is None:

        gal_cut = (
            config.gal_cut
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if gal_cut < 0:

        raise ValueError(
            "gal_cut must be non-negative."
        )

    npix = hp.nside2npix(
        nside
    )

    # ------------------------------------------------------------------
    # No Galactic masking
    # ------------------------------------------------------------------

    if use_mask is False:

        return np.ones(
            npix,
            dtype=bool,
        )

    # ------------------------------------------------------------------
    # Standard Galactic mask
    # ------------------------------------------------------------------

    theta, phi = hp.pix2ang(
        nside,
        np.arange(npix),
    )

    coords = SkyCoord(

        ra=(
            np.degrees(phi)
            * u.degree
        ),

        dec=(
            90.0
            - np.degrees(theta)
        ) * u.degree,

        frame="icrs",
    )

    gal_b = np.asarray(

        coords
        .galactic
        .b
        .degree,

        dtype=float,
    )

    return (
        np.abs(gal_b)
        > gal_cut
    )


def rotate_healpix_map_to_galactic(
    hmap: FloatArray,
    coord_in: str = "C",
    coord_out: str = "G",
) -> FloatArray:
    """
    Rotate a HEALPix map between coordinate systems.

    Parameters
    ----------
    hmap : FloatArray
        Input HEALPix map.

    coord_in : str
        Input coordinate system:
            "C" = Equatorial / ICRS
            "G" = Galactic

    coord_out : str
        Output coordinate system.

    Returns
    -------
    FloatArray
        Rotated HEALPix map.
    """

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    hmap = np.asarray(
        hmap,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if hmap.ndim != 1:

        raise ValueError(
            "Input HEALPix map must be "
            "1-dimensional."
        )

    valid_coords = {
        "C",
        "G",
        "E",
    }

    if coord_in not in valid_coords:

        raise ValueError(
            f"Invalid coord_in: {coord_in}"
        )

    if coord_out not in valid_coords:

        raise ValueError(
            f"Invalid coord_out: {coord_out}"
        )

    if not np.all(
        np.isfinite(hmap)
    ):

        raise ValueError(
            "Input HEALPix map contains "
            "non-finite values."
        )

    nside = hp.get_nside(
        hmap
    )

    npix_expected = hp.nside2npix(
        nside
    )

    if len(hmap) != npix_expected:

        raise ValueError(
            "Input map has inconsistent "
            "HEALPix size."
        )

    # ------------------------------------------------------------------
    # Coordinate rotation
    # ------------------------------------------------------------------

    rotator = hp.Rotator(
        coord=[
            coord_out,
            coord_in,
        ]
    )

    theta, phi = hp.pix2ang(
        nside,
        np.arange(npix_expected),
    )

    theta_rot, phi_rot = rotator(
        theta,
        phi,
    )

    pix_rot = hp.ang2pix(
        nside,
        theta_rot,
        phi_rot,
    )

    return np.asarray(
        hmap[pix_rot],
        dtype=float,
    )


# ==============================================================================
# Survey modeling and selection functions
# ==============================================================================

def split_by_survey(
    context: RuntimeContext,
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Partition catalog into observational surveys.

    Behavior depends on the active configuration.

    use_sel_func = True
        -> split by survey

    use_sel_func = False
        -> treat entire catalog as one population
    """

    config = context.config

    # ------------------------------------------------------------------
    # No selection function
    # ------------------------------------------------------------------

    if config.use_sel_func is False:

        print(
            "\n--- Survey partitioning disabled ---"
        )

        print(
            "use_sel_func=False "
            "→ pure isotropic mode"
        )

        return {}

    # ------------------------------------------------------------------
    # Catalog normalization
    # ------------------------------------------------------------------

    df = df.copy()

    df["Reporting_Group_s"] = (

        df["Reporting_Group_s"]

        .fillna("UNKNOWN")

        .astype(str)
    )

    surveys: dict[
        str,
        list[
            tuple[float, float]
        ],
    ] = {}

    # ------------------------------------------------------------------
    # Survey extraction
    # ------------------------------------------------------------------

    for row in df.itertuples(index=False):

        groups = list({

            g.strip()

            for g in str(
                row.Reporting_Group_s
            ).split(",")

            if g.strip()
        })

        for group in groups:

            if group == "":

                group = "UNKNOWN"

            surveys.setdefault(
                group,
                [],
            ).append(
                (
                    row.RA,
                    row.DEC,
                )
            )

    # ------------------------------------------------------------------
    # Survey DataFrames
    # ------------------------------------------------------------------

    survey_dict = {

        name: pd.DataFrame(
            values,
            columns=[
                "RA",
                "DEC",
            ],
        )

        for name, values
        in surveys.items()
    }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    counts = {

        name: len(subdf)

        for name, subdf
        in survey_dict.items()
    }

    sorted_counts = sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        "\n--- Survey partitioning ---"
    )

    print(
        f"Unique surveys: "
        f"{len(survey_dict)}"
    )

    print(
        "\nSurvey populations:"
    )

    for name, count in sorted_counts:

        frac = (
            100.0
            * count
            / len(df)
        )

        print(
            f"{name:>15s} | "
            f"N={count:4d} | "
            f"{frac:6.2f}%"
        )

    return survey_dict



def build_selection_function_improved(
    context: RuntimeContext,
    subdf: pd.DataFrame,
    nside: int | None = None,
    smooth_sigma: float | None = None,
    gal_cut: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """
    Build probabilistic sky selection function for one survey.

    Returns
    -------
    sf : FloatArray
        Normalized sky probability distribution.

    sf_cov_diag : FloatArray
        Approximate diagonal covariance.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    if len(subdf) == 0:

        raise ValueError(
            "Cannot build selection function "
            "from empty survey."
        )

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if nside is None:

        nside = (
            config.nside_sf
        )

    if smooth_sigma is None:

        smooth_sigma = (
            config.smooth_sigma
        )

    if gal_cut is None:

        gal_cut = (
            config.gal_cut
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    if smooth_sigma < 0:

        raise ValueError(
            "smooth_sigma must be "
            "non-negative."
        )

    if gal_cut < 0:

        raise ValueError(
            "gal_cut must be "
            "non-negative."
        )

    npix = hp.nside2npix(
        nside
    )

    # ------------------------------------------------------------------
    # Pixelized survey map
    # ------------------------------------------------------------------

    theta = np.radians(
        90.0
        - subdf["DEC"].to_numpy(float)
    )

    phi = np.radians(
        subdf["RA"].to_numpy(float)
    )

    pix = hp.ang2pix(
        nside,
        theta,
        phi,
    )

    counts = np.bincount(
        pix,
        minlength=npix,
    ).astype(float)

    # ------------------------------------------------------------------
    # Small numerical floor
    # ------------------------------------------------------------------

    counts += (
        config.selection_function_floor
    )

    total_counts = counts.sum()

    if total_counts <= 0:

        raise ValueError(
            "Selection function has "
            "zero counts."
        )

    # ------------------------------------------------------------------
    # Raw selection function
    # ------------------------------------------------------------------

    sf = counts / total_counts

    # ------------------------------------------------------------------
    # Poisson uncertainty
    # ------------------------------------------------------------------

    sf_err_counts = np.sqrt(
        counts
    )

    sf_cov_diag = (
        sf_err_counts
        / total_counts
    ) ** 2

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    if smooth_sigma > 0:

        sf = hp.smoothing(
            sf,
            sigma=np.radians(
                smooth_sigma
            ),
            verbose=False,
        )

        sf = np.clip(
            sf,
            0.0,
            None,
        )

    # ------------------------------------------------------------------
    # Galactic mask
    # ------------------------------------------------------------------

    gal_mask = healpix_galactic_mask(
        context,
        nside,
        gal_cut=gal_cut,
        use_mask=config.use_gal_mask,
    )

    sf[~gal_mask] = 0.0

    # ------------------------------------------------------------------
    # Final normalization
    # ------------------------------------------------------------------

    sf_sum = sf.sum()

    if sf_sum <= 0:

        raise ValueError(
            "Selection function vanished "
            "after masking."
        )

    sf /= sf_sum

    return (
        np.asarray(
            sf,
            dtype=float,
        ),
        np.asarray(
            sf_cov_diag,
            dtype=float,
        ),
    )


# ==============================================================================
# Multi-survey selection functions
# ==============================================================================

def build_survey_selection_functions_improved(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    nside: int | None = None,
    smooth_sigma: float | None = None,
    gal_cut: float | None = None,
    use_adaptive_smoothing: bool = False,
) -> SelectionFunctionSet | None:
    """
    Build survey-based sky probability maps.

    Behavior depends on the active configuration.

    use_sel_func = True
        -> one selection function per survey

    use_sel_func = False
        -> no selection-function modeling
    """

    config = context.config

    # ----------------------------------------------------------
    # Pure isotropic mode
    # ----------------------------------------------------------

    if config.use_sel_func is False:

        print(
            "\n--- Selection-function modeling disabled ---"
        )

        print(
            "use_sel_func=False "
            "→ no survey selection functions "
            "will be built"
        )

        return None

    # ----------------------------------------------------------
    # Runtime defaults
    # ----------------------------------------------------------

    if nside is None:

        nside = (
            config.nside_sf
        )

    if smooth_sigma is None:

        smooth_sigma = (
            config.smooth_sigma
        )

    if gal_cut is None:

        gal_cut = (
            config.gal_cut
        )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    if smooth_sigma < 0:

        raise ValueError(
            "smooth_sigma must be "
            "non-negative."
        )

    if gal_cut < 0:

        raise ValueError(
            "gal_cut must be "
            "non-negative."
        )

    # ----------------------------------------------------------
    # Survey partitioning
    # ----------------------------------------------------------

    surveys = split_by_survey(
        context,
        df_data,
    )

    total_survey_memberships = sum(

        len(subdf)

        for subdf
        in surveys.values()
    )

    if total_survey_memberships <= 0:

        raise ValueError(
            "No survey memberships available "
            "for selection-function modeling."
        )

    sf_dict: dict[
        str,
        FloatArray,
    ] = {}

    sf_cov_diag_dict: dict[
        str,
        FloatArray,
    ] = {}

    survey_weights: dict[
        str,
        float,
    ] = {}

    print(
        "\n--- Building selection functions ---"
    )

    print(
        f"Unique surveys: "
        f"{len(surveys)}"
    )

    print(
        f"Adaptive smoothing: "
        f"{use_adaptive_smoothing}"
    )

    # ----------------------------------------------------------
    # Survey loop
    # ----------------------------------------------------------

    for name, subdf in surveys.items():

        # ------------------------------------------------------
        # Optional adaptive smoothing
        # ------------------------------------------------------

        effective_sigma = smooth_sigma

        if use_adaptive_smoothing:

            if len(subdf) < 10:

                effective_sigma = max(
                    smooth_sigma,
                    8.0,
                )

            elif len(subdf) < 30:

                effective_sigma = max(
                    smooth_sigma,
                    5.0,
                )

        # ------------------------------------------------------
        # Build survey selection function
        # ------------------------------------------------------

        sf, sf_cov_diag = (

            build_selection_function_improved(
                context,
                subdf,
                nside=nside,
                smooth_sigma=effective_sigma,
                gal_cut=gal_cut,
            )
        )

        sf_dict[name] = sf

        sf_cov_diag_dict[name] = (
            sf_cov_diag
        )

        survey_weights[name] = (

            len(subdf)
            / total_survey_memberships
        )

        print(
            f"{name:>15s} | "
            f"N={len(subdf):4d} | "
            f"sigma={effective_sigma:4.1f} deg | "
            f"weight={survey_weights[name]:.4f}"
        )

    print(
        "\nSelection functions "
        "successfully built."
    )

    # ----------------------------------------------------------
    # Build immutable validated container
    # ----------------------------------------------------------

    sf_set = SelectionFunctionSet(
        sf_dict=sf_dict,
        sf_cov_diag_dict=sf_cov_diag_dict,
        survey_weights=survey_weights,
        nside=nside,
    )

    sf_set.validate()

    return sf_set


# ==============================================================================
# Selection-function perturbations
# ==============================================================================

def generate_sf_variant(
    context: RuntimeContext,
    sf_set: SelectionFunctionSet,
    perturbation_scale: float | None = None,
    rng: Generator | None = None,
) -> dict[str, FloatArray]:
    
    """
    Generate perturbed realizations of survey selection functions.

    Notes
    -----
    This function is only physically relevant when
    the active configuration enables selection-function modeling.

    When selection functions are disabled,
    no perturbation is generated.

    IMPORTANT
    ---------
    A Generator instance must be passed explicitly
    to guarantee statistically independent realizations
    across mocks and parallel workers.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if perturbation_scale is None:

        perturbation_scale = (
            config.perturbation_scale
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if perturbation_scale < 0:

        raise ValueError(
            "perturbation_scale must be "
            "non-negative."
        )

    # ------------------------------------------------------------------
    # Pure isotropic mode
    # ------------------------------------------------------------------

    if config.use_sel_func is False:

        return {}

    # ------------------------------------------------------------------
    # RNG validation
    # ------------------------------------------------------------------

    if rng is None:

        raise ValueError(
            "A numpy.random.Generator instance "
            "must be provided explicitly."
        )

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------

    sf_dict_perturbed: dict[
        str,
        FloatArray,
    ] = {}

    # ------------------------------------------------------------------
    # Survey perturbation loop
    # ------------------------------------------------------------------

    for survey_name in sf_set.surveys:

        sf = np.asarray(
            sf_set.sf_dict[
                survey_name
            ],
            dtype=float,
        )

        sf_cov_diag = np.asarray(
            sf_set.sf_cov_diag_dict[
                survey_name
            ],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Shape validation
        # --------------------------------------------------------------

        if sf.shape != sf_cov_diag.shape:

            raise ValueError(
                f"Shape mismatch for survey "
                f"{survey_name}: "
                f"{sf.shape} != "
                f"{sf_cov_diag.shape}"
            )

        # --------------------------------------------------------------
        # Covariance validation
        # --------------------------------------------------------------

        if np.any(sf_cov_diag < 0):

            raise ValueError(
                f"Negative covariance entries "
                f"detected for survey: "
                f"{survey_name}"
            )

        # --------------------------------------------------------------
        # Gaussian perturbation
        # --------------------------------------------------------------

        sigma = np.sqrt(
            sf_cov_diag
        )

        noise = rng.normal(
            loc=0.0,
            scale=(
                perturbation_scale
                * sigma
            ),
            size=sf.shape,
        )

        sf_pert = np.clip(
            sf + noise,
            0.0,
            None,
        )

        # --------------------------------------------------------------
        # Numerical safeguard
        # --------------------------------------------------------------

        sf_sum = sf_pert.sum()

        if sf_sum <= 0:

            sf_pert = sf.copy()

        else:

            sf_pert /= sf_sum

        sf_dict_perturbed[
            survey_name
        ] = np.asarray(
            sf_pert,
            dtype=float,
        )

    return sf_dict_perturbed


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


# ==============================================================================
# Selection-function validation
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class SelectionFunctionValidator:
    """
    Numerical and visual validation of survey
    selection functions.

    This validator is only physically meaningful when:

        use_sel_func = True
    """

    context: RuntimeContext

    sf_set: SelectionFunctionSet

    # ==========================================================
    # Numerical validation
    # ==========================================================

    def check_coverage(
        self,
    ) -> pd.DataFrame:

        config = self.context.config

        # ------------------------------------------------------
        # Disabled case
        # ------------------------------------------------------

        if config.use_sel_func is False:

            print(
                "\n--- Selection-function validation skipped ---"
            )

            print(
                "use_sel_func=False "
                "→ no instrumental selection model."
            )

            return pd.DataFrame()

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        self.sf_set.validate()

        if not hp.isnsideok(
            self.sf_set.nside
        ):

            raise ValueError(
                f"Invalid HEALPix nside: "
                f"{self.sf_set.nside}"
            )

        print("\n==================================================")
        print("SELECTION FUNCTION VALIDATION")
        print("==================================================")

        rows: list[
            dict[
                str,
                float | str,
            ]
        ] = []

        # ------------------------------------------------------
        # Survey ordering
        # ------------------------------------------------------

        ordered = sorted(

            self.sf_set.surveys,

            key=lambda name: (
                self.sf_set.survey_weights[name]
            ),

            reverse=True,
        )


        # ------------------------------------------------------
        # Survey loop
        # ------------------------------------------------------

        for survey_name in ordered:

            sf = np.asarray(
                self.sf_set.sf_dict[
                    survey_name
                ],
                dtype=float,
            )

            # --------------------------------------------------
            # Numerical validation
            # --------------------------------------------------

            if not np.all(
                np.isfinite(sf)
            ):

                raise ValueError(
                    f"Selection function "
                    f"{survey_name} contains "
                    f"non-finite values."
                )

            # --------------------------------------------------
            # Peak position
            # --------------------------------------------------

            max_pix = int(
                np.argmax(sf)
            )

            theta_max, phi_max = hp.pix2ang(
                self.sf_set.nside,
                max_pix,
            )

            dec_max = (
                90.0
                - np.degrees(theta_max)
            )

            ra_max = np.degrees(
                phi_max
            )

            # --------------------------------------------------
            # Coverage
            # --------------------------------------------------

            threshold = (
                np.percentile(sf, 99)
                * 1e-2
            )

            active = sf > threshold

            coverage = float(
                np.mean(active)
                * 100.0
            )

            # --------------------------------------------------
            # Entropy
            # --------------------------------------------------

            positive = sf > 0.0

            entropy = float(
                -np.sum(
                    sf[positive]
                    * np.log2(
                        sf[positive]
                    )
                )
            )

            # --------------------------------------------------
            # Effective sky area
            # --------------------------------------------------

            npix_active = int(
                np.sum(active)
            )

            sky_fraction = float(
                npix_active
                / len(sf)
            )

            # --------------------------------------------------
            # Store row
            # --------------------------------------------------

            rows.append(
                {

                    "run_tag":
                        config.run_tag,

                    "survey":
                        survey_name,

                    "weight":
                        self.sf_set.survey_weights[
                            survey_name
                        ],

                    "peak_ra":
                        ra_max,

                    "peak_dec":
                        dec_max,

                    "active_coverage_pct":
                        coverage,

                    "sky_fraction":
                        sky_fraction,

                    "entropy_bits":
                        entropy,
                }
            )

            # --------------------------------------------------
            # Diagnostics
            # --------------------------------------------------

            print(
                f"\n{survey_name}:"
            )

            print(
                f"  Weight: "
                f"{self.sf_set.survey_weights[survey_name]:.2%}"
            )

            print(
                f"  Peak density: "
                f"RA={ra_max:.1f} deg, "
                f"DEC={dec_max:.1f} deg"
            )

            print(
                f"  Active coverage: "
                f"{coverage:.1f}%"
            )

            print(
                f"  Entropy: "
                f"{entropy:.2f} bits"
            )

            # --------------------------------------------------
            # Physical sanity checks
            # --------------------------------------------------

            survey_upper = (
                survey_name.upper()
            )

            if (
                "CHIME" in survey_upper
                and dec_max < 20.0
            ):

                print(
                    "  WARNING: "
                    "CHIME peak unexpectedly far south."
                )

            if (
                "PARKES" in survey_upper
                and dec_max > 5.0
            ):

                print(
                    "  WARNING: "
                    "PARKES peak unexpectedly far north."
                )

        validation_df = pd.DataFrame(
            rows
        )

        return validation_df

    # ==========================================================
    # Visual validation
    # ==========================================================

    def plot_sf(
        self,
        save_prefix: str | Path = "SF_validation",
        max_surveys: int = 12,
        cmap: str = "viridis",
        log_scale: bool = False,
    ) -> None:

        config = self.context.config

        floor = (
            config.selection_function_floor
        )

        save_prefix = Path(
            save_prefix
        )

        # ------------------------------------------------------
        # Disabled case
        # ------------------------------------------------------

        if config.use_sel_func is False:

            print(
                "\n--- SF plots skipped ---"
            )

            print(
                "use_sel_func=False "
                "→ no selection functions to visualize."
            )

            return

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        self.sf_set.validate()

        if max_surveys < 1:

            raise ValueError(
                "max_surveys must be positive."
            )

        # ------------------------------------------------------
        # Survey ordering
        # ------------------------------------------------------

        ordered = sorted(

            self.sf_set.surveys,

            key=lambda name: (
                self.sf_set.survey_weights[name]
            ),

            reverse=True,
        )[:max_surveys]

        # ------------------------------------------------------
        # Safety check
        # ------------------------------------------------------

        if len(ordered) == 0:

            print(
                "\nNo selection functions "
                "available for plotting."
            )

            return

        ncols = min(
            4,
            len(ordered),
        )

        nrows = int(
            np.ceil(
                len(ordered) / ncols
            )
        )

        scale_label = (
            "log10"
            if log_scale
            else "linear"
        )

        # ======================================================
        # EQUATORIAL MAPS
        # ======================================================

        plt.figure(
            figsize=(
                5 * ncols,
                3.8 * nrows,
            )
        )

        for i, name in enumerate(
            ordered,
            start=1,
        ):

            sf_plot = np.asarray(
                self.sf_set.sf_dict[name],
                dtype=float,
            )

            if log_scale:

                sf_plot = np.log10(
                    np.clip(
                        sf_plot,
                        floor,
                        None,
                    )
                )

            hp.mollview(
                sf_plot,
                title=(
                    f"{name}\n"
                    f"(w="
                    f"{self.sf_set.survey_weights[name]:.1%}, "
                    f"{scale_label})"
                ),
                sub=(
                    nrows,
                    ncols,
                    i,
                ),
                cbar=True,
                cmap=cmap,
            )

        equatorial_path = (
            save_prefix
            .with_name(
                save_prefix.name
                + "_equatorial.png"
            )
        )

        plt.savefig(
            equatorial_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()

        plt.close()

        # ======================================================
        # GALACTIC MAPS
        # ======================================================

        plt.figure(
            figsize=(
                5 * ncols,
                3.8 * nrows,
            )
        )

        for i, name in enumerate(
            ordered,
            start=1,
        ):

            sf_gal = (
                rotate_healpix_map_to_galactic(
                    self.sf_set.sf_dict[name],
                    coord_in="C",
                    coord_out="G",
                )
            )

            if log_scale:

                sf_gal = np.log10(
                    np.clip(
                        sf_gal,
                        floor,
                        None,
                    )
                )

            hp.mollview(
                sf_gal,
                title=(
                    f"{name}\n"
                    f"(w="
                    f"{self.sf_set.survey_weights[name]:.1%}, "
                    f"{scale_label})"
                ),
                sub=(
                    nrows,
                    ncols,
                    i,
                ),
                cbar=True,
                cmap=cmap,
            )

        galactic_path = (
            save_prefix
            .with_name(
                save_prefix.name
                + "_galactic.png"
            )
        )

        plt.savefig(
            galactic_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()

        plt.close()

        print(
            "\nSelection-function maps saved:"
        )

        print(
            f"  {equatorial_path}"
        )

        print(
            f"  {galactic_path}"
        )


# ==============================================================================
# Mock catalog generation
# ==============================================================================

def _jitter_pixel_centers(
    context: RuntimeContext,
    ra: FloatArray,
    dec: FloatArray,
    nside: int,
    rng: Generator,
) -> tuple[FloatArray, FloatArray]:
    """
    Randomize positions inside HEALPix pixels.

    This suppresses artificial angular quantization
    produced by finite HEALPix resolution.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    if not isinstance(
        rng,
        Generator,
    ):

        raise TypeError(
            "rng must be a numpy.random.Generator."
        )

    if len(ra) != len(dec):

        raise ValueError(
            "ra and dec must have "
            "the same length."
        )

    if not np.all(
        np.isfinite(ra)
    ):

        raise ValueError(
            "ra contains non-finite values."
        )

    if not np.all(
        np.isfinite(dec)
    ):

        raise ValueError(
            "dec contains non-finite values."
        )

    # ------------------------------------------------------------------
    # HEALPix angular scale
    # ------------------------------------------------------------------

    pix_radius_deg = np.degrees(
        hp.max_pixrad(nside)
    )

    # ------------------------------------------------------------------
    # Declination jitter
    # ------------------------------------------------------------------

    dec_jitter = rng.uniform(
        -pix_radius_deg,
        pix_radius_deg,
        size=len(dec),
    )

    # ------------------------------------------------------------------
    # Right ascension jitter
    # ------------------------------------------------------------------

    cos_dec = np.clip(
        np.cos(
            np.radians(dec)
        ),
        config.numerical_zero_tolerance,
        None,
    )

    ra_jitter = (

        rng.uniform(
            -pix_radius_deg,
            pix_radius_deg,
            size=len(ra),
        )

        / cos_dec
    )

    # ------------------------------------------------------------------
    # Final coordinates
    # ------------------------------------------------------------------

    ra_out = (
        ra + ra_jitter
    ) % 360.0

    dec_out = np.clip(
        dec + dec_jitter,
        -89.999,
        89.999,
    )

    return (
        np.asarray(
            ra_out,
            dtype=float,
        ),
        np.asarray(
            dec_out,
            dtype=float,
        ),
    )


# ==============================================================================
# Isotropic mock catalog generation
# ==============================================================================

def generate_mixture_catalog_improved(
    context: RuntimeContext,
    n_observed: int,
    sf_set: SelectionFunctionSet | None = None,
    jitter_pixels: bool = True,
    use_poisson: bool = False,
    rng: Generator | None = None,
) -> pd.DataFrame:
    """
    Generate isotropic mock catalogs.

    Modes
    -----
    sf_set is None
        Pure isotropic sky realization.

    sf_set exists
        Isotropic realization filtered through
        survey selection functions.
    """

    config = context.config

    # ------------------------------------------------------------------
    # RNG validation
    # ------------------------------------------------------------------

    if rng is None:

        raise ValueError(
            "rng must be explicitly provided."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if n_observed <= 0:

        return pd.DataFrame(
            columns=[
                "RA",
                "DEC",
            ]
        )

    # ------------------------------------------------------------------
    # Catalog size
    # ------------------------------------------------------------------

    if use_poisson:

        n = int(
            rng.poisson(
                n_observed
            )
        )

    else:

        n = int(
            n_observed
        )

    if n <= 0:

        return pd.DataFrame(
            columns=[
                "RA",
                "DEC",
            ]
        )

    # ==========================================================================
    # PURE ISOTROPIC SKY
    # ==========================================================================

    if sf_set is None:

        ra_all: list[float] = []

        dec_all: list[float] = []

        while len(ra_all) < n:

            n_remaining = (
                n
                - len(ra_all)
            )

            # --------------------------------------------------------------
            # Uniform isotropic sphere
            # --------------------------------------------------------------

            ra = (
                360.0
                * rng.random(
                    n_remaining
                )
            )

            dec = np.degrees(
                np.arcsin(
                    2.0
                    * rng.random(
                        n_remaining
                    )
                    - 1.0
                )
            )

            # --------------------------------------------------------------
            # Galactic masking
            # --------------------------------------------------------------

            if config.use_gal_mask:

                coords = SkyCoord(

                    ra=(
                        ra
                        * u.degree
                    ),

                    dec=(
                        dec
                        * u.degree
                    ),

                    frame="icrs",
                )

                gal_b = np.asarray(

                    coords
                    .galactic
                    .b
                    .degree,

                    dtype=float,
                )

                keep = (
                    np.abs(gal_b)
                    > config.gal_cut
                )

                ra = ra[keep]

                dec = dec[keep]

            ra_all.extend(
                ra.tolist()
            )

            dec_all.extend(
                dec.tolist()
            )

        return pd.DataFrame(
            {
                "RA": np.asarray(
                    ra_all[:n],
                    dtype=float,
                ),

                "DEC": np.asarray(
                    dec_all[:n],
                    dtype=float,
                ),
            }
        )

    # ==========================================================================
    # SELECTION-FUNCTION SKY
    # ==========================================================================

    surveys = np.asarray(
        list(
            sf_set.sf_dict.keys()
        )
    )

    survey_probs = np.asarray(
        [
            sf_set.survey_weights[s]
            for s in surveys
        ],
        dtype=float,
    )

    prob_sum = survey_probs.sum()

    if prob_sum <= 0:

        raise ValueError(
            "Survey weights sum to zero."
        )

    survey_probs /= prob_sum

    chosen_surveys = rng.choice(
        surveys,
        size=n,
        p=survey_probs,
    )

    ra = np.empty(
        n,
        dtype=float,
    )

    dec = np.empty(
        n,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Survey realizations
    # ------------------------------------------------------------------

    for survey in surveys:

        mask = (
            chosen_surveys
            == survey
        )

        n_survey = int(
            np.sum(mask)
        )

        if n_survey == 0:

            continue

        sf = np.asarray(
            sf_set.sf_dict[
                survey
            ],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if not np.all(
            np.isfinite(sf)
        ):

            raise ValueError(
                f"Non-finite values detected "
                f"in survey SF: {survey}"
            )

        sf = np.clip(
            sf,
            0.0,
            None,
        )

        sf_sum = sf.sum()

        if sf_sum <= 0:

            raise ValueError(
                f"Selection function vanished "
                f"for survey: {survey}"
            )

        sf /= sf_sum

        # --------------------------------------------------------------
        # HEALPix sampling
        # --------------------------------------------------------------

        pix = rng.choice(
            len(sf),
            size=n_survey,
            p=sf,
        )

        theta, phi = hp.pix2ang(
            sf_set.nside,
            pix,
        )

        ra_survey = np.degrees(
            phi
        )

        dec_survey = (
            90.0
            - np.degrees(theta)
        )

        # --------------------------------------------------------------
        # Sub-pixel randomization
        # --------------------------------------------------------------

        if jitter_pixels:

            ra_survey, dec_survey = (

                _jitter_pixel_centers(

                    context=context,

                    ra=np.asarray(
                        ra_survey,
                        dtype=float,
                    ),

                    dec=np.asarray(
                        dec_survey,
                        dtype=float,
                    ),

                    nside=sf_set.nside,

                    rng=rng,
                )
            )

        ra[mask] = ra_survey

        dec[mask] = dec_survey

    return pd.DataFrame(
        {
            "RA": np.asarray(
                ra,
                dtype=float,
            ),

            "DEC": np.asarray(
                dec,
                dtype=float,
            ),
        }
    )


# ==============================================================================
# Independent mock realization
# ==============================================================================

def _run_one_independent_mock(
    context: RuntimeContext,
    sf_set: SelectionFunctionSet | None,
    seed_data: int,
    seed_rand: int,
    n_data: int,
    n_rand_factor: int,
) -> tuple[
    FloatArray,
    FloatArray,
]:
    """
    Generate one statistically independent
    isotropic mock realization.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if n_data <= 0:

        raise ValueError(
            "n_data must be positive."
        )

    if n_rand_factor <= 0:

        raise ValueError(
            "n_rand_factor must be positive."
        )

    # ------------------------------------------------------------------
    # Independent RNG streams
    # ------------------------------------------------------------------

    seed_sequence = np.random.SeedSequence(
        [
            seed_data,
            seed_rand,
        ]
    )

    child_sequences = seed_sequence.spawn(2)

    rng_data = np.random.default_rng(
        child_sequences[0]
    )

    rng_rand = np.random.default_rng(
        child_sequences[1]
    )

    # ------------------------------------------------------------------
    # Data catalog
    # ------------------------------------------------------------------

    d_iso = (
        generate_mixture_catalog_improved(
            context=context,

            n_observed=n_data,

            sf_set=sf_set,

            jitter_pixels=True,

            use_poisson=False,

            rng=rng_data,
        )
    )

    # ------------------------------------------------------------------
    # Random catalog
    # ------------------------------------------------------------------

    r_iso = (
        generate_mixture_catalog_improved(
            context=context,

            n_observed=(
                n_data
                * n_rand_factor
            ),

            sf_set=sf_set,

            jitter_pixels=True,

            use_poisson=False,

            rng=rng_rand,
        )
    )

    # ------------------------------------------------------------------
    # 2pACF
    # ------------------------------------------------------------------

    theta_h0, w_h0 = (
        compute_2pacf(
            context,
            d_iso,
            r_iso,
        )
    )

    # ------------------------------------------------------------------
    # Tomographic absolute anisotropy estimator
    # ------------------------------------------------------------------

    abs_h0 = get_absolute_sum(
        context,
        theta_h0,
        w_h0,
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return (

        np.asarray(
            w_h0,
            dtype=float,
        ),

        np.asarray(
            abs_h0,
            dtype=float,
        ),
    )


# ==============================================================================
# Ensemble mock generation
# ==============================================================================

def run_ensemble_mocks(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    n_ensemble: int | None = None,
    n_mocks_per: int | None = None,
    perturbation_scale: float | None = None,
    n_rand_factor: int | None = None,
    n_jobs: int | None = None,
) -> MockEnsemble:
    """
    Generate isotropic H0 mock ensembles.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if n_jobs is None:

        n_jobs = config.n_jobs

    if n_ensemble is None:

        n_ensemble = (
            config.n_ensemble
        )

    if n_mocks_per is None:

        n_mocks_per = (
            config.n_mocks_per_ensemble
        )

    if perturbation_scale is None:

        perturbation_scale = (
            config.perturbation_scale
        )

    if n_rand_factor is None:

        n_rand_factor = (
            config.n_rand_factor
        )

    n_jobs = max(
        1,
        int(n_jobs),
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if n_ensemble <= 0:

        raise ValueError(
            "n_ensemble must be positive."
        )

    if n_mocks_per <= 0:

        raise ValueError(
            "n_mocks_per must be positive."
        )

    if n_rand_factor <= 0:

        raise ValueError(
            "n_rand_factor must be positive."
        )

    if perturbation_scale < 0:

        raise ValueError(
            "perturbation_scale must be "
            "non-negative."
        )

    if len(df_data) <= 0:

        raise ValueError(
            "Observed catalog is empty."
        )

    # ------------------------------------------------------------------
    # Master seed sequence
    # ------------------------------------------------------------------

    master_seed_sequence = (
        np.random.SeedSequence(
            config.random_seed
        )
    )

    total_mocks = (
        n_ensemble
        * n_mocks_per
    )

    mock_seed_sequences = (
        master_seed_sequence.spawn(
            total_mocks
        )
    )

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    tasks: list[
        tuple[
            SelectionFunctionSet | None,
            int,
            int,
        ]
    ] = []

    # ==========================================================================
    # SELECTION-FUNCTION MODE
    # ==========================================================================

    if config.use_sel_func:

        if sf_set is None:

            raise ValueError(
                "sf_set cannot be None "
                "when use_sel_func=True."
            )

        print(
            "\n--- Generating H0 ensemble mocks "
            "(with selection functions) ---"
        )

        task_index = 0

        for i_ens in range(
            n_ensemble
        ):

            # --------------------------------------------------------------
            # Independent ensemble RNG
            # --------------------------------------------------------------

            ensemble_rng = (
                np.random.default_rng(
                    mock_seed_sequences[
                        task_index
                    ]
                )
            )

            # --------------------------------------------------------------
            # Shared SF realization
            # --------------------------------------------------------------

            sf_variant_dict = (
                generate_sf_variant(

                    context=context,

                    sf_set=sf_set,

                    perturbation_scale=(
                        perturbation_scale
                    ),

                    rng=ensemble_rng,
                )
            )

            # --------------------------------------------------------------
            # Rebuild immutable SF container
            # --------------------------------------------------------------

            sf_variant_set = (
                SelectionFunctionSet(

                    sf_dict=(
                        sf_variant_dict
                    ),

                    sf_cov_diag_dict=(
                        sf_set.sf_cov_diag_dict
                    ),

                    survey_weights=(
                        sf_set.survey_weights
                    ),

                    nside=(
                        sf_set.nside
                    ),
                )
            )

            # --------------------------------------------------------------
            # Mock realizations
            # --------------------------------------------------------------

            for i_mock in range(
                n_mocks_per
            ):

                mock_ss = (
                    mock_seed_sequences[
                        task_index
                    ]
                )

                child_ss = (
                    mock_ss.spawn(2)
                )

                seed_data = int(
                    child_ss[0]
                    .generate_state(1)[0]
                )

                seed_rand = int(
                    child_ss[1]
                    .generate_state(1)[0]
                )

                tasks.append(
                    (
                        sf_variant_set,
                        seed_data,
                        seed_rand,
                    )
                )

                task_index += 1

    # ==========================================================================
    # PURE ISOTROPIC SKY
    # ==========================================================================

    else:

        print(
            "\n--- Generating pure isotropic "
            "H0 mocks ---"
        )

        for i_mock in range(
            total_mocks
        ):

            mock_ss = (
                mock_seed_sequences[
                    i_mock
                ]
            )

            child_ss = (
                mock_ss.spawn(2)
            )

            seed_data = int(
                child_ss[0]
                .generate_state(1)[0]
            )

            seed_rand = int(
                child_ss[1]
                .generate_state(1)[0]
            )

            tasks.append(
                (
                    None,
                    seed_data,
                    seed_rand,
                )
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    n_total = len(
        tasks
    )

    if n_total == 0:

        raise RuntimeError(
            "No mock tasks were generated."
        )

    print(
        f"Total H0 mocks: {n_total}"
    )

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    with tqdm_joblib(
        tqdm(
            desc="H0 ensemble mocks",
            total=n_total,
        )
    ):

        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size="auto",
        )(
            delayed(
                _run_one_independent_mock
            )(
                context=context,

                sf_set=task_sf_set,

                seed_data=seed_data,

                seed_rand=seed_rand,

                n_data=len(df_data),

                n_rand_factor=(
                    n_rand_factor
                ),
            )

            for (
                task_sf_set,
                seed_data,
                seed_rand,
            ) in tasks
        )

    if len(results) == 0:

        raise RuntimeError(
            "No mock realizations generated."
        )

    # ------------------------------------------------------------------
    # Stack outputs
    # ------------------------------------------------------------------

    all_w_h0, all_abs_h0 = zip(
        *results
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

    if all_w_h0.ndim != 2:

        raise RuntimeError(
            "all_w_h0 must be 2-dimensional."
        )

    if all_abs_h0.ndim != 2:

        raise RuntimeError(
            "all_abs_h0 must be 2-dimensional."
        )

    if not np.all(
        np.isfinite(all_w_h0)
    ):

        raise RuntimeError(
            "Non-finite values detected "
            "in all_w_h0."
        )

    if not np.all(
        np.isfinite(all_abs_h0)
    ):

        raise RuntimeError(
            "Non-finite values detected "
            "in all_abs_h0."
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print(
        "\nGenerated mocks:"
    )

    print(
        f"  w(theta): "
        f"{all_w_h0.shape}"
    )

    print(
        f"  |<w>| tomography: "
        f"{all_abs_h0.shape}"
    )

    # ------------------------------------------------------------------
    # Final structured output
    # ------------------------------------------------------------------

    return MockEnsemble(

        w_theta=all_w_h0,

        abs_statistics=all_abs_h0,

        metadata={

            "n_mocks": int(
                len(all_w_h0)
            ),

            "n_ensemble": int(
                n_ensemble
            ),

            "n_mocks_per_ensemble": int(
                n_mocks_per
            ),

            "selection_function_mode": bool(
                config.use_sel_func
            ),

            "random_seed": int(
                config.random_seed
            ),
        },
    )


# ==============================================================================
# Physical estimators
# ==============================================================================

def compute_2pacf(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    df_rand: pd.DataFrame,
) -> tuple[
    FloatArray,
    FloatArray,
]:
    """
    Compute the angular two-point correlation function.

    Uses
    ----
    - Linear angular bins
    - Landy-Szalay estimator
    - TreeCorr pair counting

    Notes
    -----
    Fixed nominal angular bin centers are used
    to guarantee consistency across mock realizations.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    required_cols = {
        "RA",
        "DEC",
    }

    if not required_cols.issubset(
        df_data.columns
    ):

        raise ValueError(
            "df_data must contain "
            "RA and DEC columns."
        )

    if not required_cols.issubset(
        df_rand.columns
    ):

        raise ValueError(
            "df_rand must contain "
            "RA and DEC columns."
        )

    if len(df_data) == 0:

        raise ValueError(
            "df_data is empty."
        )

    if len(df_rand) == 0:

        raise ValueError(
            "df_rand is empty."
        )

    # ------------------------------------------------------------------
    # Coordinate arrays
    # ------------------------------------------------------------------

    ra_data = np.asarray(
        df_data["RA"],
        dtype=float,
    )

    dec_data = np.asarray(
        df_data["DEC"],
        dtype=float,
    )

    ra_rand = np.asarray(
        df_rand["RA"],
        dtype=float,
    )

    dec_rand = np.asarray(
        df_rand["DEC"],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Numerical validation
    # ------------------------------------------------------------------

    arrays_to_check = {

        "ra_data": ra_data,
        "dec_data": dec_data,
        "ra_rand": ra_rand,
        "dec_rand": dec_rand,
    }

    for name, arr in arrays_to_check.items():

        if not np.all(
            np.isfinite(arr)
        ):

            raise ValueError(
                f"{name} contains "
                f"non-finite values."
            )

    # ------------------------------------------------------------------
    # Physical coordinate validation
    # ------------------------------------------------------------------

    if np.any(
        (ra_data < 0.0)
        | (ra_data >= 360.0)
    ):

        raise ValueError(
            "ra_data outside [0, 360)."
        )

    if np.any(
        (ra_rand < 0.0)
        | (ra_rand >= 360.0)
    ):

        raise ValueError(
            "ra_rand outside [0, 360)."
        )

    if np.any(
        (dec_data < -90.0)
        | (dec_data > 90.0)
    ):

        raise ValueError(
            "dec_data outside [-90, 90]."
        )

    if np.any(
        (dec_rand < -90.0)
        | (dec_rand > 90.0)
    ):

        raise ValueError(
            "dec_rand outside [-90, 90]."
        )

    # ------------------------------------------------------------------
    # TreeCorr catalogs
    # ------------------------------------------------------------------

    cat_data = treecorr.Catalog(

        ra=ra_data,

        dec=dec_data,

        ra_units="deg",

        dec_units="deg",
    )

    cat_rand = treecorr.Catalog(

        ra=ra_rand,

        dec=dec_rand,

        ra_units="deg",

        dec_units="deg",
    )

    # ------------------------------------------------------------------
    # Pair-counting configuration
    # ------------------------------------------------------------------

    corr_config = dict(

        min_sep=(
            config.min_sep
        ),

        max_sep=(
            config.max_sep
        ),

        nbins=(
            config.n_bins
        ),

        sep_units="deg",

        metric="Arc",

        bin_type="Linear",

        num_threads=1,
    )

    dd = treecorr.NNCorrelation(
        **corr_config
    )

    rr = treecorr.NNCorrelation(
        **corr_config
    )

    dr = treecorr.NNCorrelation(
        **corr_config
    )

    # ------------------------------------------------------------------
    # Pair counts
    # ------------------------------------------------------------------

    dd.process(
        cat_data
    )

    rr.process(
        cat_rand
    )

    dr.process(
        cat_data,
        cat_rand,
    )

    # ------------------------------------------------------------------
    # Landy-Szalay estimator
    # ------------------------------------------------------------------

    dd.calculateXi(
        rr=rr,
        dr=dr,
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    theta = np.asarray(
        dd.rnom,
        dtype=float,
    )

    w_theta = np.asarray(
        dd.xi,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Numerical cleanup
    # ------------------------------------------------------------------

    bad = ~np.isfinite(
        w_theta
    )

    if np.any(bad):

        warnings.warn(
            f"{np.sum(bad)} invalid "
            "2pACF bins detected.",
            RuntimeWarning,
        )

        w_theta[bad] = np.nan

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    if theta.shape != w_theta.shape:

        raise RuntimeError(
            "theta and w_theta have "
            "incompatible shapes."
        )

    if len(theta) != config.n_bins:

        raise RuntimeError(
            "Unexpected number of "
            "2pACF bins returned."
        )

    return (
        theta,
        w_theta,
    )


# ==============================================================================
# Absolute anisotropy estimator
# ==============================================================================

def get_absolute_sum(
    context: RuntimeContext,
    theta: FloatArray,
    w: FloatArray,
) -> FloatArray:
    """
    Compute the tomographic absolute-sum anisotropy estimator.

    This follows Eq. (4) of:

        Andrade et al. (2019)
        "Revisiting the statistical isotropy
        of GRB sky distribution"

    For each coarse angular interval Δθ:

        Absolute sum = Σ |w(θ)|

    where the summation is performed over all
    fine 2pACF bins inside that interval.

    Notes
    -----
    - Positive and negative oscillations
      do not cancel.

    - The estimator is sensitive to residual
      angular structure.

    - The estimator intentionally depends
      on the adopted angular binning.
    """

    config = context.config

    coarse_bins = (
        config.coarse_bins_array
    )

    theta = np.asarray(
        theta,
        dtype=float,
    )

    w = np.asarray(
        w,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if theta.shape != w.shape:

        raise ValueError(
            "theta and w must have "
            "identical shapes."
        )

    if len(coarse_bins) < 2:

        raise ValueError(
            "At least two coarse angular "
            "bin edges are required."
        )

    if not np.all(
        np.isfinite(theta)
    ):

        raise ValueError(
            "theta contains non-finite values."
        )

    if not np.any(
        np.isfinite(w)
    ):

        raise ValueError(
            "w contains no finite values."
        )

    if np.any(
        np.diff(coarse_bins) <= 0
    ):

        raise ValueError(
            "coarse_bins must be strictly "
            "increasing."
        )

    if coarse_bins[0] < 0:

        raise ValueError(
            "coarse_bins must be non-negative."
        )

    if coarse_bins[-1] <= coarse_bins[0]:

        raise ValueError(
            "Invalid coarse_bins range."
        )

    vals: list[float] = []

    # ------------------------------------------------------------------
    # Coarse angular tomography
    # ------------------------------------------------------------------

    for i in range(
        len(coarse_bins) - 1
    ):

        theta_min = (
            coarse_bins[i]
        )

        theta_max = (
            coarse_bins[i + 1]
        )

        mask = (

            (theta >= theta_min)

            &

            (theta < theta_max)
        )

        # --------------------------------------------------------------
        # Ignore invalid bins
        # --------------------------------------------------------------

        mask &= np.isfinite(
            w
        )

        if np.any(mask):

            val = float(
                np.sum(
                    np.abs(
                        w[mask]
                    )
                )
            )

        else:

            val = 0.0

        vals.append(
            val
        )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    return np.asarray(
        vals,
        dtype=float,
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

        hartlap_factor

        * np.sum(
            delta_modes**2
            * inv_eigvals
        )
    )

    chi2_svd_mocks = (

        hartlap_factor

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

    for mock in mocks:

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
        # --------------------------------------------------------------

        if np.std(mock) < 1e-15:

            mock = mock + np.random.default_rng(
                2
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
            "\n--- KS / AD profile tests ---"
        )

        print(
            f"KS statistic   = "
            f"{ks_stat:.5f}"
        )

        print(
            f"KS p-value     = "
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
            f"AD p-value     = "
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

        ks_n_extreme_mocks=float(
            ks_n_extreme
        ),

        ad_statistic=ad_stat,

        ad_pvalue=ad_pvalue,

        ad_empirical_p=ad_empirical_p,

        ad_n_extreme_mocks=float(
            ad_n_extreme
        ),
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
            f"KS p-value          = "
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
            f"AD p-value          = "
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
            f"KS p-value          = "
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
            f"AD p-value          = "
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

    outputs = context.outputs

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
    # Remove globally invalid bins
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
    # H0 mean profile
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
    # Shrinkage covariance (Ledoit-Wolf)
    # ------------------------------------------------------------------

    lw = LedoitWolf()

    lw.fit(
        all_w_h0_valid
    )

    cov = np.asarray(
        lw.covariance_,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Diagnostics plots
    # ------------------------------------------------------------------

    plot_covariance_correlation_matrix(

        context=context,

        cov=cov,

        title="H0 Correlation Matrix",

        output=(

            context.outputs.figures_dir
            / "h0_correlation_matrix.png"
        ),
    )

    plot_mock_similarity(

        context=context,

        all_w_h0=all_w_h0_valid,

        n_compare=min(
            100,
            len(all_w_h0_valid),
        ),

        output=(
            outputs.figures_dir
            / "h0_mock_similarity.png"
        ),
    )

    # ------------------------------------------------------------------
    # Eigenvalue spectrum
    # ------------------------------------------------------------------

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
        outputs.tables_dir
        / "covariance_eigenvalues.csv",
        index=False,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.semilogy(
        eigenvalues_sorted,
        marker="o",
    )

    plt.xlabel(
        "Mode"
    )

    plt.ylabel(
        "Eigenvalue"
    )

    plt.title(
        "Covariance Eigenvalue Spectrum\n"
        f"({config.run_tag})"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        outputs.figures_dir
        / "covariance_eigenspectrum.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------------
    # Effective number of modes
    # ------------------------------------------------------------------

    eigvals_sq_sum = np.sum(
        eigvals_pos**2
    )

    if eigvals_sq_sum <= 0:

        n_eff = np.nan

    else:

        n_eff = float(
            (
                eigvals_pos.sum() ** 2
            )
            / eigvals_sq_sum
        )

    print(
        f"\nEffective number of modes = "
        f"{n_eff:.2f}"
    )

    # ------------------------------------------------------------------
    # Covariance diagnostics
    # ------------------------------------------------------------------

    n_mocks, n_bins = (
        all_w_h0_valid.shape
    )

    covariance_rank = int(
        np.linalg.matrix_rank(cov)
    )

    eigvals_safe = eigvals_pos[
        eigvals_pos > 1e-12
    ]

    if len(eigvals_safe) == 0:

        raise ValueError(
            "No stable covariance "
            "eigenvalues found."
        )

    smallest_eig = max(
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
        / smallest_eig
    )

    # ------------------------------------------------------------------
    # Hartlap correction
    # ------------------------------------------------------------------

    hartlap_factor = float(
        (
            n_mocks
            - n_bins
            - 2
        )
        / (
            n_mocks
            - 1
        )
    )

    if hartlap_factor <= 0:

        raise ValueError(
            "Hartlap correction invalid: "
            "number of mocks too small "
            "relative to covariance dimension."
        )

    # ------------------------------------------------------------------
    # Inverse covariance
    # ------------------------------------------------------------------

    inv_cov = (
        hartlap_factor
        * np.linalg.pinv(cov)
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
    # SVD-regularized chi-square
    # ------------------------------------------------------------------

    svd_stats = (
        compute_svd_regularized_chi2(
            context=context,
            delta=delta,
            mock_delta=mock_delta,
            cov=cov,
            hartlap_factor=hartlap_factor,
            eigenvalue_cut=(
                config.svd_eigenvalue_cut
            ),
        )
    )

    dof = int(
        svd_stats.modes_kept
    )

    # ------------------------------------------------------------------
    # Full chi-square
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Empirical chi-square p-value
    # ------------------------------------------------------------------

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

    n_chi2_extreme = int(
        np.sum(
            chi2_mocks
            >= chi2
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
            n_chi2_extreme
            + 1
        )
        / (
            len(chi2_mocks)
            + 1
        )
    )

    sigma_equiv = (
        sigma_equivalent_from_p(
            p_empirical
        )
    )

    # ------------------------------------------------------------------
    # Global normalized tension
    # ------------------------------------------------------------------

    diag_sigma = (
        np.sqrt(
            np.diag(cov)
        )
        + 1e-12
    )

    global_tension = float(
        np.sqrt(
            np.mean(
                (
                    delta
                    / diag_sigma
                ) ** 2
            )
        )
    )

    # ------------------------------------------------------------------
    # Absolute-sum empirical significance
    # ------------------------------------------------------------------

    abs_obs_valid = abs_obs[
        np.isfinite(abs_obs)
    ]

    if len(abs_obs_valid) == 0:

        raise ValueError(
            "abs_obs contains no finite values."
        )

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
            "No valid absolute-statistic mock realizations."
        )

    abs_observed_total = float(
        np.sum(abs_obs_valid)
    )

    abs_empirical_p = float(
        (
            np.sum(
                abs_mock_totals
                >= abs_observed_total
            ) + 1
        )
        / (
            len(abs_mock_totals)
            + 1
        )
    )

    # ------------------------------------------------------------------
    # Nonparametric tests
    # ------------------------------------------------------------------

    nonparam_stats = (
        compute_nonparametric_tests(
            w_obs=w_obs,
            abs_obs=abs_obs_valid,
            all_w_h0=all_w_h0,
            all_abs_h0=all_abs_h0,
            verbose=False,
        )
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(
        f"\n--- Full covariance diagnostic "
        f"({label}) ---"
    )

    print(
        f"Chi2 = "
        f"{chi2:.3f}"
    )

    print(
        f"Chi2/dof = "
        f"{chi2_red:.3f}"
    )

    print(
        f"p-value (Chi2 analytic) = "
        f"{p_chi2:.4e}"
    )

    print(
        f"p-value (Chi2 empirical) = "
        f"{p_empirical:.4e}"
    )

    if n_chi2_extreme == 0:

        print(
            "p-value (Chi2 empirical) "
            "is at the Monte Carlo floor: "
            f"p <= "
            f"{p_empirical_floor:.4e}"
        )

    print(
        f"Sigma-equivalent "
        f"(from empirical p) = "
        f"{sigma_equiv:.2f}"
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

    print("\n--- Covariance diagnostics ---")

    print(
        f"Mocks = {n_mocks}"
    )

    print(
        f"Bins = {n_bins}"
    )

    print(
        f"Covariance rank = "
        f"{covariance_rank}/{n_bins}"
    )

    print(
        f"Covariance condition number = "
        f"{covariance_condition:.4e}"
    )

    print(
        f"SVD retained condition number = "
        f"{svd_stats.svd_condition:.4e}"
    )

    print(
        f"Effective modes = "
        f"{n_eff:.2f}"
    )

    print(
        f"RMS normalized deviation = "
        f"{global_tension:.3f}"
    )

    print(
        f"Hartlap factor = "
        f"{hartlap_factor:.3f}"
    )

    print(
        "\n--- Absolute-sum statistic ---"
    )

    print(
        f"Observed total = "
        f"{abs_observed_total:.4e}"
    )

    print(
        f"Empirical p-value = "
        f"{abs_empirical_p:.4e}"
    )

    # ------------------------------------------------------------------
    # Physical interpretation
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("PHYSICAL INTERPRETATION")
    print("==================================================")

    if config.use_sel_func:

        print("H0 hypothesis:")

        print(
            "  isotropy + survey selection effects"
        )

    else:

        print("H0 hypothesis:")

        print(
            "  perfect isotropy"
        )

    if p_empirical > 0.05:

        print("\nResult:")

        print(
            "  ✓ Data are statistically "
            "compatible with H0."
        )

    else:

        print("\nResult:")

        print(
            "  ⚠ Possible tension with H0 detected."
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return TestStatistics(

        chi2=ChiSquareStatistics(

            chi2=chi2,

            chi2_red=chi2_red,

            p_chi2=p_chi2,

            p_empirical=p_empirical,

            p_empirical_floor=(
                p_empirical_floor
            ),

            sigma_equiv=sigma_equiv,
        ),

        svd=svd_stats,

        nonparametric=(
            nonparam_stats
        ),

        absolute=AbsoluteStatistics(

            abs_observed_stat=(
                abs_observed_total
            ),

            abs_empirical_p=(
                abs_empirical_p
            ),

            global_tension=(
                global_tension
            ),
        ),

        covariance=(
            CovarianceDiagnostics(

                hartlap_factor=(
                    hartlap_factor
                ),

                covariance_rank=(
                    covariance_rank
                ),

                covariance_condition=(
                    covariance_condition
                ),

                n_eff=n_eff,

                svd_modes_kept=(
                    svd_stats.modes_kept
                ),

                svd_eigenvalue_cut=(
                    svd_stats.svd_eigenvalue_cut
                ),

                svd_condition=(
                    svd_stats.svd_condition
                ),
            )
        ),

        n_mocks=n_mocks,

        n_bins=n_bins,
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

    Returns
    -------
    bin_df : pd.DataFrame
        Bin-level diagnostics.

    mode_df : pd.DataFrame
        Covariance eigenmode diagnostics.

    Notes
    -----
    The diagonal bin contributions are only approximate because
    the full covariance contains strong off-diagonal structure.

    The exact decomposition of the covariance-aware chi-square
    is given by the covariance eigenmodes.
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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Remove invalid bins globally
    # ------------------------------------------------------------------

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

    theta = theta[
        valid_bins
    ]

    w_obs = w_obs[
        valid_bins
    ]

    all_w_h0 = all_w_h0[
        :,
        valid_bins,
    ]

    # ------------------------------------------------------------------
    # H0 statistics
    # ------------------------------------------------------------------

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

    n_mocks, n_bins = (
        all_w_h0.shape
    )

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    lw = LedoitWolf()

    lw.fit(
        all_w_h0
    )

    cov = np.asarray(
        lw.covariance_,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Hartlap correction
    # ------------------------------------------------------------------

    hartlap_factor = float(
        (
            n_mocks
            - n_bins
            - 2
        )
        / (
            n_mocks
            - 1
        )
    )

    if hartlap_factor <= 0:

        raise ValueError(
            "Hartlap factor <= 0. "
            "Increase the number of mocks "
            "or reduce the covariance dimension."
        )

    # ------------------------------------------------------------------
    # Residual vector
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Bin-level diagnostics
    # ------------------------------------------------------------------

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
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Covariance eigendecomposition
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
    # Positive eigenmodes
    # ------------------------------------------------------------------

    positive = (
        eigvals > 1e-15
    )

    if not np.any(positive):

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

    # ------------------------------------------------------------------
    # SVD mode selection
    # ------------------------------------------------------------------

    keep = (

        positive

        &

        (
            eigvals
            >= eigenvalue_cut * max_eig
        )
    )

    # ------------------------------------------------------------------
    # Safety fallback
    # ------------------------------------------------------------------

    if not np.any(keep):

        keep[
            np.flatnonzero(
                positive
            )[0]
        ] = True

    # ------------------------------------------------------------------
    # Projection into covariance modes
    # ------------------------------------------------------------------

    delta_modes = np.asarray(
        eigvecs.T @ delta,
        dtype=float,
    )

    raw_mode_chi2 = np.zeros(
        len(eigvals),
        dtype=float,
    )

    raw_mode_chi2[positive] = (

        hartlap_factor

        * delta_modes[positive]**2

        / eigvals[positive]
    )

    svd_mode_chi2 = np.where(
        keep,
        raw_mode_chi2,
        0.0,
    )

    # ------------------------------------------------------------------
    # Dominant angular scale per mode
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Mode-level diagnostics
    # ------------------------------------------------------------------

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
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Cumulative contributions
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Stable covariance condition number
    # ------------------------------------------------------------------

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
            np.min(eigvals_safe)
        )

    else:

        covariance_condition = np.inf

        smallest_retained = np.nan

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

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

    return (
        bin_df,
        mode_df,
    )


def compare_covariance_statistics(
    all_w_h0_old: FloatArray,
    all_w_h0_new: FloatArray,
) -> pd.DataFrame:
    """
    Compare covariance properties before/after pipeline fixes.

    Useful for diagnosing:

        - mock correlations
        - covariance conditioning
        - effective dimensionality
        - rank deficiencies
        - ensemble independence
    """

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

    stats: list[
        dict[
            str,
            float | int
        ]
    ] = []

    for label, w_array in comparisons:

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Remove invalid bins globally
        # ------------------------------------------------------------------

        valid_bins = np.all(
            np.isfinite(w_array),
            axis=0,
        )

        if np.sum(valid_bins) < 2:

            raise ValueError(
                f"Not enough valid bins for "
                f"{label} covariance comparison."
            )

        w_array = w_array[
            :,
            valid_bins,
        ]

        # ------------------------------------------------------------------
        # Covariance estimation
        # ------------------------------------------------------------------

        lw = LedoitWolf()

        lw.fit(
            w_array
        )

        cov = np.asarray(
            lw.covariance_,
            dtype=float,
        )

        # ------------------------------------------------------------------
        # Covariance validation
        # ------------------------------------------------------------------

        if not np.allclose(
            cov,
            cov.T,
            atol=1e-12,
            rtol=1e-10,
        ):

            raise ValueError(
                f"{label}: covariance matrix "
                "is not symmetric."
            )

        # ------------------------------------------------------------------
        # Eigenstructure
        # ------------------------------------------------------------------

        eigvals = np.linalg.eigvalsh(
            cov
        )

        eigvals = np.asarray(
            eigvals,
            dtype=float,
        )

        eigvals_pos = eigvals[
            eigvals > 1e-15
        ]

        if len(eigvals_pos) == 0:

            raise ValueError(
                f"{label}: covariance matrix "
                "has no stable positive eigenvalues."
            )

        # ------------------------------------------------------------------
        # Effective dimensionality
        # ------------------------------------------------------------------

        eig_sum = float(
            np.sum(
                eigvals_pos
            )
        )

        eig_sq_sum = float(
            np.sum(
                eigvals_pos**2
            )
        )

        if eig_sq_sum <= 0:

            n_eff = np.nan

        else:

            n_eff = float(
                eig_sum**2
                / eig_sq_sum
            )

        # ------------------------------------------------------------------
        # Stable condition number
        # ------------------------------------------------------------------

        eigvals_safe = eigvals_pos[
            eigvals_pos > 1e-12
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

            largest_safe = float(
                np.max(
                    eigvals_safe
                )
            )

            condition = float(
                largest_safe
                / smallest_safe
            )

            min_safe_eig = smallest_safe

        else:

            condition = np.nan

            min_safe_eig = np.nan

        # ------------------------------------------------------------------
        # Matrix rank
        # ------------------------------------------------------------------

        rank = int(
            np.linalg.matrix_rank(
                cov
            )
        )

        # ------------------------------------------------------------------
        # Mock similarity diagnostics
        # ------------------------------------------------------------------

        corr = np.corrcoef(
            w_array
        )

        corr = np.asarray(
            corr,
            dtype=float,
        )

        corr[
            ~np.isfinite(corr)
        ] = np.nan

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

        if len(offdiag) == 0:

            raise ValueError(
                f"{label}: unable to compute "
                "off-diagonal mock correlations."
            )

        # ------------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------------

        stats.append(
            {

                "version":
                    label,

                "n_mocks":
                    int(
                        w_array.shape[0]
                    ),

                "n_bins":
                    int(
                        w_array.shape[1]
                    ),

                "covariance_rank":
                    rank,

                "effective_modes":
                    float(n_eff),

                "condition_number":
                    float(condition),

                "max_eigenvalue":
                    float(
                        np.max(
                            eigvals_pos
                        )
                    ),

                "min_positive_eigenvalue":
                    float(
                        np.min(
                            eigvals_pos
                        )
                    ),

                "min_safe_eigenvalue":
                    float(
                        min_safe_eig
                    ),

                "mean_mock_corr":
                    float(
                        np.mean(
                            offdiag
                        )
                    ),

                "median_mock_corr":
                    float(
                        np.median(
                            offdiag
                        )
                    ),

                "std_mock_corr":
                    float(
                        np.std(
                            offdiag
                        )
                    ),

                "max_mock_corr":
                    float(
                        np.max(
                            offdiag
                        )
                    ),

                "min_mock_corr":
                    float(
                        np.min(
                            offdiag
                        )
                    ),
            }
        )

    # ------------------------------------------------------------------
    # Build DataFrame
    # ------------------------------------------------------------------

    df = pd.DataFrame(
        stats
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("COVARIANCE COMPARISON DIAGNOSTICS")
    print("==================================================")

    print(
        df.round(4)
    )

    # ------------------------------------------------------------------
    # Physical interpretation
    # ------------------------------------------------------------------

    if len(df) == 2:

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

    return df


# ==============================================================================
# Observational uncertainty estimation
# ==============================================================================


def assign_jackknife_regions(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    nside_jackknife: int | None = None,
    min_regions: int | None = None,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    """
    Assign observed FRBs to coarse HEALPix jackknife regions.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if nside_jackknife is None:

        nside_jackknife = (
            config.nside_jackknife
        )

    if min_regions is None:

        min_regions = (
            config.min_jackknife_regions
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not hp.isnsideok(
        nside_jackknife
    ):

        raise ValueError(
            f"Invalid HEALPix nside: "
            f"{nside_jackknife}"
        )

    if min_regions <= 0:

        raise ValueError(
            "min_regions must be positive."
        )

    required_cols = {
        "RA",
        "DEC",
    }

    if not required_cols.issubset(
        df_data.columns
    ):

        raise ValueError(
            "df_data must contain "
            "RA and DEC columns."
        )

    # ------------------------------------------------------------------
    # Copy catalog
    # ------------------------------------------------------------------

    df = df_data.copy()

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    theta = np.radians(
        90.0
        - df["DEC"].to_numpy(float)
    )

    phi = np.radians(
        df["RA"].to_numpy(float)
    )

    # ------------------------------------------------------------------
    # HEALPix regions
    # ------------------------------------------------------------------

    region_pix = hp.ang2pix(
        nside_jackknife,
        theta,
        phi,
    )

    unique_regions, region_ids = np.unique(
        region_pix,
        return_inverse=True,
    )

    n_regions = int(
        len(unique_regions)
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if n_regions < min_regions:

        raise ValueError(
            f"Only {n_regions} populated "
            f"jackknife regions.\n"
            f"Decrease min_jackknife_regions "
            f"or reduce nside_jackknife."
        )

    # ------------------------------------------------------------------
    # Store region labels
    # ------------------------------------------------------------------

    df["jackknife_region"] = (
        region_ids
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print(
        "\n--- Jackknife region assignment ---"
    )

    print(
        f"NSIDE_JACKKNIFE = "
        f"{nside_jackknife}"
    )

    print(
        f"Populated regions = "
        f"{n_regions}"
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return (
        df,
        np.arange(
            n_regions,
            dtype=int,
        ),
    )


def _run_one_jackknife_region(
    context: RuntimeContext,
    region: int,
    df_data_with_regions: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    seed_base: int,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    int,
]:
    """
    Compute one leave-one-region-out jackknife realization.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if "jackknife_region" not in df_data_with_regions.columns:

        raise ValueError(
            "df_data_with_regions must contain "
            "'jackknife_region'."
        )

    # ------------------------------------------------------------------
    # Remove one region
    # ------------------------------------------------------------------

    df_jk = (
        df_data_with_regions[
            df_data_with_regions[
                "jackknife_region"
            ] != region
        ]
        .reset_index(drop=True)
    )

    if len(df_jk) == 0:

        raise ValueError(
            f"Jackknife realization "
            f"{region} produced an empty catalog."
        )

    # ------------------------------------------------------------------
    # Independent RNG
    # ------------------------------------------------------------------

    rng = np.random.default_rng(
        seed_base
        + int(region)
    )

    # ------------------------------------------------------------------
    # Random catalog
    # ------------------------------------------------------------------

    df_rand_jk = (
        generate_mixture_catalog_improved(
            context=context,

            n_observed=(
                len(df_jk)
                * config.n_rand_factor
            ),

            sf_set=sf_set,

            jitter_pixels=True,

            use_poisson=False,

            rng=rng,
        )
    )

    # ------------------------------------------------------------------
    # 2pACF
    # ------------------------------------------------------------------

    theta_jk, w_jk = (
        compute_2pacf(
            context,
            df_jk,
            df_rand_jk,
        )
    )

    # ------------------------------------------------------------------
    # Numerical protection
    # ------------------------------------------------------------------

    w_jk = np.nan_to_num(
        w_jk,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ------------------------------------------------------------------
    # Absolute anisotropy statistic
    # ------------------------------------------------------------------

    abs_jk = get_absolute_sum(
        context,
        theta_jk,
        w_jk,
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return (
        np.asarray(
            theta_jk,
            dtype=float,
        ),

        np.asarray(
            w_jk,
            dtype=float,
        ),

        np.asarray(
            abs_jk,
            dtype=float,
        ),

        int(
            len(df_jk)
        ),
    )


def jackknife_covariance(
    samples: FloatArray,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """
    Compute jackknife mean profile, covariance matrix,
    and 1-sigma uncertainties.

    Parameters
    ----------
    samples : FloatArray
        Array of shape:

            (n_regions, n_bins)

        containing leave-one-region-out
        jackknife realizations.

    Returns
    -------
    mean : FloatArray
        Jackknife mean profile.

    cov : FloatArray
        Jackknife covariance matrix.

    err : FloatArray
        1-sigma jackknife uncertainties.
    """

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    samples = np.asarray(
        samples,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if samples.ndim != 2:

        raise ValueError(
            "Jackknife samples must be 2-dimensional."
        )

    if not np.all(
        np.isfinite(samples)
    ):

        raise ValueError(
            "Jackknife samples contain "
            "non-finite values."
        )

    n_regions, n_bins = (
        samples.shape
    )

    if n_regions < 2:

        raise ValueError(
            "At least two jackknife "
            "samples are required."
        )

    if n_bins < 1:

        raise ValueError(
            "Jackknife samples contain "
            "zero bins."
        )

    # ------------------------------------------------------------------
    # Mean profile
    # ------------------------------------------------------------------

    mean = np.mean(
        samples,
        axis=0,
    )

    # ------------------------------------------------------------------
    # Centered realizations
    # ------------------------------------------------------------------

    centered = (
        samples
        - mean
    )

    # ------------------------------------------------------------------
    # Jackknife covariance
    # ------------------------------------------------------------------

    cov = (
        (n_regions - 1)
        / n_regions
    ) * (
        centered.T
        @ centered
    )

    cov = np.asarray(
        cov,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Numerical cleanup
    # ------------------------------------------------------------------

    cov = 0.5 * (
        cov
        + cov.T
    )

    # ------------------------------------------------------------------
    # 1-sigma uncertainties
    # ------------------------------------------------------------------

    err = np.sqrt(
        np.clip(
            np.diag(cov),
            0.0,
            None,
        )
    )

    err = np.asarray(
        err,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return (
        np.asarray(
            mean,
            dtype=float,
        ),

        cov,

        err,
    )


def run_jackknife_errors(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    nside_jackknife: int | None = None,
    min_regions: int | None = None,
    n_jobs: int | None = None,
    seed_base: int = 9000,
) -> JackknifeResult:
    """
    Estimate statistical uncertainties using
    leave-one-region-out jackknife resampling.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if n_jobs is None:

        n_jobs = config.n_jobs

    if nside_jackknife is None:

        nside_jackknife = (
            config.nside_jackknife
        )

    if min_regions is None:

        min_regions = (
            config.min_jackknife_regions
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    n_jobs = max(
        1,
        int(n_jobs),
    )

    if seed_base < 0:

        raise ValueError(
            "seed_base must be non-negative."
        )

    if (
        config.use_sel_func
        and sf_set is None
    ):

        raise ValueError(
            "Selection-function mode requires "
            "a valid SelectionFunctionSet."
        )

    if sf_set is not None:

        sf_set.validate()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("JACKKNIFE UNCERTAINTY ESTIMATION")
    print("==================================================")

    print(
        f"RUN_TAG = "
        f"{config.run_tag}"
    )

    # ------------------------------------------------------------------
    # Region assignment
    # ------------------------------------------------------------------

    df_regions, regions = (
        assign_jackknife_regions(
            context,
            df_data,
            nside_jackknife=(
                nside_jackknife
            ),
            min_regions=(
                min_regions
            ),
        )
    )

    counts = (
        df_regions[
            "jackknife_region"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\n--- Jackknife region statistics ---"
    )

    print(
        f"Objects per region:\n"
        f"min    = {counts.min()}\n"
        f"median = {counts.median():.1f}\n"
        f"max    = {counts.max()}"
    )

    # ------------------------------------------------------------------
    # Parallel leave-one-out realizations
    # ------------------------------------------------------------------

    with tqdm_joblib(
        tqdm(
            desc="Jackknife regions",
            total=len(regions),
        )
    ):

        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size="auto",
        )(
            delayed(
                _run_one_jackknife_region
            )(
                context=context,

                region=int(region),

                df_data_with_regions=(
                    df_regions
                ),

                sf_set=sf_set,

                seed_base=seed_base,
            )

            for region in regions
        )

    # ------------------------------------------------------------------
    # Collect realizations
    # ------------------------------------------------------------------

    (
        theta_samples,
        w_samples,
        abs_samples,
        n_kept,
    ) = zip(*results)

    theta = np.asarray(
        theta_samples[0],
        dtype=float,
    )

    w_samples = np.asarray(
        w_samples,
        dtype=float,
    )

    abs_samples = np.asarray(
        abs_samples,
        dtype=float,
    )

    n_kept = np.asarray(
        n_kept,
        dtype=int,
    )

    # ------------------------------------------------------------------
    # Remove invalid realizations
    # ------------------------------------------------------------------

    valid_w = np.all(
        np.isfinite(w_samples),
        axis=1,
    )

    valid_abs = np.all(
        np.isfinite(abs_samples),
        axis=1,
    )

    valid = (
        valid_w
        & valid_abs
    )

    if not np.all(valid):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid)} invalid "
            f"jackknife realizations."
        )

        w_samples = w_samples[
            valid
        ]

        abs_samples = abs_samples[
            valid
        ]

        n_kept = n_kept[
            valid
        ]

    if len(w_samples) < 2:

        raise RuntimeError(
            "Too few valid jackknife realizations "
            "remain after filtering."
        )

    # ------------------------------------------------------------------
    # Covariances
    # ------------------------------------------------------------------

    (
        w_mean,
        w_cov,
        w_err,
    ) = jackknife_covariance(
        w_samples
    )

    (
        abs_mean,
        abs_cov,
        abs_err,
    ) = jackknife_covariance(
        abs_samples
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print(
        "\n--- Jackknife summary ---"
    )

    print(
        f"Leave-one-region samples = "
        f"{len(w_samples)}"
    )

    print(
        f"Objects kept per sample:\n"
        f"min = {np.min(n_kept)}\n"
        f"max = {np.max(n_kept)}"
    )

    print(
        f"Median sigma[w(theta)] = "
        f"{np.median(w_err):.4e}"
    )

    print(
        f"Median sigma[|<w>|] = "
        f"{np.median(abs_err):.4e}"
    )

    # ------------------------------------------------------------------
    # Save covariance diagnostics
    # ------------------------------------------------------------------

    plot_covariance_correlation_matrix(

        context=context,

        cov=w_cov,

        title=(
            "Jackknife Correlation Matrix\n"
            f"({context.config.run_tag})"
        ),

        output="jackknife_correlation_matrix",
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return JackknifeResult(

        regions=np.asarray(
            regions,
            dtype=int,
        ),

        theta=theta,

        w_samples=w_samples,

        w_mean=w_mean,

        w_cov=w_cov,

        w_err=w_err,

        abs_samples=abs_samples,

        abs_mean=abs_mean,

        abs_cov=abs_cov,

        abs_err=abs_err,
    )


def _single_bootstrap_realization(
    context: RuntimeContext,
    seed: int,
    df_data: pd.DataFrame,
    df_rand_fixed: pd.DataFrame,
) -> tuple[
    FloatArray,
    FloatArray,
]:
    """
    Generate one bootstrap realization.

    IMPORTANT
    ---------
    - Resamples ONLY observed FRBs
    - Random catalog remains FIXED
    - Measures observational sampling uncertainty
    - Avoids artificial Monte Carlo covariance inflation
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if seed < 0:

        raise ValueError(
            "seed must be non-negative."
        )

    # ------------------------------------------------------------------
    # Independent RNG
    # ------------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    # ------------------------------------------------------------------
    # Bootstrap resampling with replacement
    # ------------------------------------------------------------------

    indices = rng.choice(
        len(df_data),
        size=len(df_data),
        replace=True,
    )

    df_boot = (
        df_data.iloc[indices]
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 2pACF
    # ------------------------------------------------------------------

    theta_boot, w_boot = (
        compute_2pacf(
            context,
            df_boot,
            df_rand_fixed,
        )
    )

    # ------------------------------------------------------------------
    # Numerical protection
    # ------------------------------------------------------------------

    w_boot = np.nan_to_num(
        w_boot,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ------------------------------------------------------------------
    # Absolute anisotropy estimator
    # ------------------------------------------------------------------

    abs_boot = get_absolute_sum(
        context,
        theta_boot,
        w_boot,
    )

    return (

        np.asarray(
            w_boot,
            dtype=float,
        ),

        np.asarray(
            abs_boot,
            dtype=float,
        ),
    )


def run_bootstrap_errors(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    n_bootstrap: int | None = None,
    n_jobs: int | None = None,
) -> BootstrapResult:
    """
    Estimate uncertainties using bootstrap resampling.

    IMPORTANT
    ---------
    Bootstrap resamples ONLY observed FRBs.

    The random catalog remains FIXED for all realizations,
    preventing Monte Carlo noise from contaminating the
    covariance estimate.
    """

    config = context.config

    outputs = context.outputs

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if n_jobs is None:

        n_jobs = config.n_jobs

    if n_bootstrap is None:

        n_bootstrap = (
            config.n_bootstrap
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    n_jobs = max(
        1,
        int(n_jobs),
    )

    if n_bootstrap < 2:

        raise ValueError(
            "At least two bootstrap realizations "
            "are required."
        )

    if (
        config.use_sel_func
        and sf_set is None
    ):

        raise ValueError(
            "Selection-function mode requires "
            "a valid SelectionFunctionSet."
        )

    if sf_set is not None:

        sf_set.validate()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("BOOTSTRAP UNCERTAINTY ESTIMATION")
    print("==================================================")

    print(
        f"RUN_TAG = "
        f"{config.run_tag}"
    )

    print(
        f"\nBootstrap realizations = "
        f"{n_bootstrap}"
    )

    # ------------------------------------------------------------------
    # Fixed random catalog
    # ------------------------------------------------------------------

    print(
        "\nGenerating fixed random catalog..."
    )

    df_rand_fixed = (
        generate_mixture_catalog_improved(
            context=context,

            n_observed=(
                len(df_data)
                * config.n_rand_factor
            ),

            sf_set=sf_set,

            jitter_pixels=True,

            use_poisson=False,

            rng=np.random.default_rng(
                123456
            ),
        )
    )

    print(
        f"Random catalog size = "
        f"{len(df_rand_fixed)}"
    )

    # ------------------------------------------------------------------
    # Parallel bootstrap realizations
    # ------------------------------------------------------------------

    with tqdm_joblib(
        tqdm(
            desc="Bootstrap realizations",
            total=n_bootstrap,
        )
    ):

        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size="auto",
        )(
            delayed(
                _single_bootstrap_realization
            )(
                context=context,

                seed=i,

                df_data=df_data,

                df_rand_fixed=(
                    df_rand_fixed
                ),
            )

            for i in range(
                n_bootstrap
            )
        )

    # ------------------------------------------------------------------
    # Stack realizations
    # ------------------------------------------------------------------

    w_realizations = np.asarray(
        [r[0] for r in results],
        dtype=float,
    )

    abs_realizations = np.asarray(
        [r[1] for r in results],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Remove invalid realizations
    # ------------------------------------------------------------------

    valid_w = np.all(
        np.isfinite(
            w_realizations
        ),
        axis=1,
    )

    valid_abs = np.all(
        np.isfinite(
            abs_realizations
        ),
        axis=1,
    )

    valid = (
        valid_w
        & valid_abs
    )

    if not np.all(valid):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid)} invalid "
            f"bootstrap realizations."
        )

        w_realizations = (
            w_realizations[
                valid
            ]
        )

        abs_realizations = (
            abs_realizations[
                valid
            ]
        )

    if len(w_realizations) < 2:

        raise RuntimeError(
            "Too few valid bootstrap realizations "
            "remain after filtering."
        )

    # ------------------------------------------------------------------
    # Mean profiles
    # ------------------------------------------------------------------

    w_mean = np.mean(
        w_realizations,
        axis=0,
    )

    abs_mean = np.mean(
        abs_realizations,
        axis=0,
    )

    # ------------------------------------------------------------------
    # Shrinkage covariance matrices
    # ------------------------------------------------------------------

    lw_w = LedoitWolf()

    lw_w.fit(
        w_realizations
    )

    w_cov = np.asarray(
        lw_w.covariance_,
        dtype=float,
    )

    lw_abs = LedoitWolf()

    lw_abs.fit(
        abs_realizations
    )

    abs_cov = np.asarray(
        lw_abs.covariance_,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 1-sigma uncertainties
    # ------------------------------------------------------------------

    w_err = np.sqrt(
        np.clip(
            np.diag(w_cov),
            0.0,
            None,
        )
    )

    abs_err = np.sqrt(
        np.clip(
            np.diag(abs_cov),
            0.0,
            None,
        )
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print(
        "\n--- Bootstrap summary ---"
    )

    print(
        f"Median sigma[w(theta)] = "
        f"{np.median(w_err):.4e}"
    )

    print(
        f"Median sigma[|<w>|] = "
        f"{np.median(abs_err):.4e}"
    )

    print(
        f"Max sigma[w(theta)] = "
        f"{np.max(w_err):.4e}"
    )

    print(
        f"Max sigma[|<w>|] = "
        f"{np.max(abs_err):.4e}"
    )

    # ------------------------------------------------------------------
    # Correlation matrix diagnostics
    # ------------------------------------------------------------------

    plot_covariance_correlation_matrix(

        context,

        w_cov,

        title=(
            "Bootstrap Correlation Matrix\n"
            f"({context.config.run_tag})"
        ),

        output="bootstrap_correlation_matrix",
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return BootstrapResult(

        w_mean=w_mean,

        w_err=w_err,

        w_cov=w_cov,

        abs_mean=abs_mean,

        abs_err=abs_err,

        abs_cov=abs_cov,

        w_realizations=w_realizations,

        abs_realizations=abs_realizations,
    )


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


def plot_mock_similarity(
    context: RuntimeContext,
    all_w_h0: FloatArray,
    n_compare: int = 100,
    output: str | Path = "mock_similarity.png",
) -> None:
    """
    Diagnose statistical similarity between H0 mocks.

    IMPORTANT
    ---------
    Each angular bin is normalized independently before
    computing mock-to-mock correlations.

    Otherwise:
        - common mean structures dominate;
        - correlations become artificially inflated.
    """

    outputs = context.outputs

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    output = Path(
        output
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

        print(
            "\nWARNING: not enough valid mocks "
            "for similarity analysis."
        )

        return

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

    # ------------------------------------------------------------------
    # Mock-to-mock correlation matrix
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("MOCK SIMILARITY DIAGNOSTICS")
    print("==================================================")

    print(
        f"Mocks analyzed: "
        f"{len(subset)}"
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
            f"99th percentile corr: "
            f"{np.percentile(offdiag, 99):.4f}"
        )

        print(
            f"Max off-diagonal corr: "
            f"{np.max(offdiag):.4f}"
        )

    # ------------------------------------------------------------------
    # Physical interpretation
    # ------------------------------------------------------------------

    median_corr = float(
        np.median(
            np.abs(offdiag)
        )
    )

    print(
        "\nInterpretation:"
    )

    if median_corr < 0.05:

        print(
            "  ✓ Mock ensemble appears "
            "statistically independent."
        )

    elif median_corr < 0.15:

        print(
            "  ⚠ Mild residual correlations detected."
        )

    else:

        print(
            "  ⚠ Strong mock correlations detected."
        )

        print(
            "  Covariance estimation may be biased."
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
        "Mock-to-mock correlation"
    )

    ax.set_title(
        "Normalized Mock Similarity Matrix\n"
        f"({context.config.run_tag})"
    )

    ax.set_xlabel(
        "Mock index"
    )

    ax.set_ylabel(
        "Mock index"
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
        "\nSaved mock similarity matrix:"
    )

    print(
        f"  {output}"
    )


def plot_top_survey_maps(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet,
    max_surveys: int = 12,
    output_prefix: str | Path | None = None,
    cmap: str = "viridis",
) -> pd.DataFrame:
    """
    Generate survey-footprint visualizations with
    observed FRBs overlaid.

    Outputs
    -------
    Survey selection functions with observed FRB
    positions overlaid, shown in:

        - Equatorial coordinates
        - Galactic coordinates

    Notes
    -----
    Pure selection-function maps are already produced by:

        SelectionFunctionValidator.plot_sf()

    Therefore this routine focuses exclusively on the
    combined visualization:

        selection function + observed FRBs
    """

    config = context.config

    outputs = context.outputs

    # ------------------------------------------------------------------
    # Skip if SF disabled
    # ------------------------------------------------------------------

    if not config.use_sel_func:

        print(
            "\n--- Survey map visualization skipped ---"
        )

        print(
            "use_sel_func=False "
            "→ no survey selection functions available."
        )

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if max_surveys < 1:

        raise ValueError(
            "max_surveys must be positive."
        )

    sf_set.validate()

    # ------------------------------------------------------------------
    # Default output prefix
    # ------------------------------------------------------------------

    if output_prefix is None:

        output_prefix = (
            outputs.figures_dir
            / "survey_maps"
        )

    output_prefix = Path(
        output_prefix
    )

    # ------------------------------------------------------------------
    # Survey partitioning
    # ------------------------------------------------------------------

    surveys = split_by_survey(
        context,
        df_data,
    )

    ordered = sorted(
        sf_set.sf_dict.keys(),
        key=lambda name: (
            sf_set.survey_weights[name]
        ),
        reverse=True,
    )[:max_surveys]

    if len(ordered) == 0:

        print(
            "\nWARNING: no surveys available for plotting."
        )

        return pd.DataFrame()

    rows: list[
        dict[str, object]
    ] = []

    # ------------------------------------------------------------------
    # Coordinate systems
    # ------------------------------------------------------------------

    for coord_system in (
        "equatorial",
        "galactic",
    ):

        ncols = min(
            4,
            len(ordered),
        )

        nrows = int(
            np.ceil(
                len(ordered)
                / ncols
            )
        )

        fig = plt.figure(
            figsize=(
                5.0 * ncols,
                3.8 * nrows,
            )
        )

        # ==============================================================
        # Survey loop
        # ==============================================================

        for idx, survey_name in enumerate(
            ordered,
            start=1,
        ):

            if survey_name not in surveys:

                print(
                    f"\nWARNING: survey "
                    f"{survey_name} not found "
                    f"in split catalog."
                )

                continue

            subdf = surveys[
                survey_name
            ]

            sf = np.asarray(
                sf_set.sf_dict[
                    survey_name
                ],
                dtype=float,
            )

            # ----------------------------------------------------------
            # Coordinate transform
            # ----------------------------------------------------------

            if coord_system == "galactic":

                sf_plot = (
                    rotate_healpix_map_to_galactic(
                        sf,
                        coord_in="C",
                        coord_out="G",
                    )
                )

                coords = SkyCoord(

                    ra=(
                        subdf["RA"].to_numpy(float)
                        * u.degree
                    ),

                    dec=(
                        subdf["DEC"].to_numpy(float)
                        * u.degree
                    ),

                    frame="icrs",
                )

                lon = (
                    coords
                    .galactic
                    .l
                    .degree
                )

                lat = (
                    coords
                    .galactic
                    .b
                    .degree
                )

            else:

                sf_plot = sf

                lon = (
                    subdf["RA"]
                    .to_numpy(float)
                )

                lat = (
                    subdf["DEC"]
                    .to_numpy(float)
                )

            # ----------------------------------------------------------
            # Diagnostics
            # ----------------------------------------------------------

            sf_positive = sf[
                np.isfinite(sf)
                & (sf > 0.0)
            ]

            active_coverage = float(
                np.mean(
                    sf
                    > np.nanmax(sf)
                    * 1e-3
                )
                * 100.0
            )

            if len(sf_positive) > 0:

                entropy = float(
                    -np.sum(
                        sf_positive
                        * np.log2(
                            sf_positive
                        )
                    )
                )

            else:

                entropy = np.nan

            # ----------------------------------------------------------
            # Plot
            # ----------------------------------------------------------

            hp.mollview(
                sf_plot,
                fig=fig.number,
                sub=(
                    nrows,
                    ncols,
                    idx,
                ),

                title=(
                    f"{survey_name} | "
                    f"N={len(subdf)} | "
                    f"w={sf_set.survey_weights[survey_name]:.1%}"
                ),

                cbar=True,
                min=0.0,
                cmap=cmap,
            )

            hp.projscatter(
                lon,
                lat,
                lonlat=True,
                s=8,
                c="black",
                alpha=0.65,
                edgecolors="none",
            )

            # ----------------------------------------------------------
            # Save diagnostics
            # ----------------------------------------------------------

            rows.append(
                {

                    "survey":
                        survey_name,

                    "n_objects":
                        int(
                            len(subdf)
                        ),

                    "weight":
                        float(
                            sf_set
                            .survey_weights[
                                survey_name
                            ]
                        ),

                    "active_coverage_pct":
                        active_coverage,

                    "entropy_bits":
                        entropy,

                    "nside":
                        int(
                            sf_set.nside
                        ),

                    "coordinate_system":
                        coord_system,

                    "use_gal_mask":
                        bool(
                            config.use_gal_mask
                        ),

                    "run_tag":
                        config.run_tag,
                }
            )

        # ==============================================================
        # Save figure
        # ==============================================================

        filename = (
            output_prefix
            .with_name(
                output_prefix.name
                + f"_{coord_system}.png"
            )
        )

        plt.savefig(
            filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.show()

        plt.close(fig)

        print(
            f"Saved survey maps: "
            f"{filename}"
        )

    return pd.DataFrame(
        rows
    )


def plot_results_with_jackknife(
    context: RuntimeContext,
    theta: FloatArray,
    w_obs: FloatArray,
    abs_obs: FloatArray,
    all_w_h0: FloatArray,
    all_abs_h0: FloatArray,
    jackknife: JackknifeResult,
    output_2pacf: str | Path | None = None,
    output_abs: str | Path | None = None,
) -> None:
    """
    Plot isotropy diagnostics using:
        - H0 mock confidence bands
        - Jackknife observational uncertainties

    Notes
    -----
    H0 depends on configuration:

    use_sel_func = False
        -> perfect isotropic sky

    use_sel_func = True
        -> isotropy convolved with survey selection functions
    """

    config = context.config

    outputs = context.outputs

    # ------------------------------------------------------------------
    # Output filenames
    # ------------------------------------------------------------------

    if output_2pacf is None:

        output_2pacf = (
            outputs.figures_dir
            / "2pacf_jackknife.png"
        )

    if output_abs is None:

        output_abs = (
            outputs.figures_dir
            / "absolute_anisotropy_jackknife.png"
        )

    output_2pacf = Path(
        output_2pacf
    )

    output_abs = Path(
        output_abs
    )

    # ------------------------------------------------------------------
    # Numerical safety
    # ------------------------------------------------------------------

    theta = np.asarray(
        theta,
        dtype=float,
    )

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

    if theta.ndim != 1:

        raise ValueError(
            "theta must be 1-dimensional."
        )

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

    if all_abs_h0.shape[1] != len(abs_obs):

        raise ValueError(
            "all_abs_h0 and abs_obs "
            "have incompatible shapes."
        )

    if len(jackknife.w_err) != len(theta):

        raise ValueError(
            "jackknife.w_err and theta "
            "have incompatible shapes."
        )

    if len(jackknife.abs_err) != len(abs_obs):

        raise ValueError(
            "jackknife.abs_err and abs_obs "
            "have incompatible shapes."
        )

    # ------------------------------------------------------------------
    # Remove invalid H0 realizations
    # ------------------------------------------------------------------

    valid_w = np.all(
        np.isfinite(all_w_h0),
        axis=1,
    )

    valid_abs = np.all(
        np.isfinite(all_abs_h0),
        axis=1,
    )

    if not np.all(valid_w):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid_w)} invalid "
            f"H0 w(theta) realizations."
        )

        all_w_h0 = all_w_h0[
            valid_w
        ]

    if not np.all(valid_abs):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid_abs)} invalid "
            f"H0 absolute-sum realizations."
        )

        all_abs_h0 = all_abs_h0[
            valid_abs
        ]

    if len(all_w_h0) < 2:

        raise RuntimeError(
            "Too few valid H0 w(theta) "
            "realizations remain."
        )

    if len(all_abs_h0) < 2:

        raise RuntimeError(
            "Too few valid H0 absolute-sum "
            "realizations remain."
        )

    # ------------------------------------------------------------------
    # Plot style
    # ------------------------------------------------------------------

    sns.set_theme(
        style="white",
        context="talk",
        rc={
            "axes.edgecolor": "0.25",
            "axes.linewidth": 1.1,
        },
    )

    # ------------------------------------------------------------------
    # H0 ensemble statistics
    # ------------------------------------------------------------------

    h0_mean = np.nanmean(
        all_w_h0,
        axis=0,
    )

    h0_std = np.nanstd(
        all_w_h0,
        axis=0,
        ddof=1,
    )

    h0_abs_mean = np.nanmean(
        all_abs_h0,
        axis=0,
    )

    h0_abs_std = np.nanstd(
        all_abs_h0,
        axis=0,
        ddof=1,
    )

    h0_std = np.nan_to_num(
        h0_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    h0_abs_std = np.nan_to_num(
        h0_abs_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    band_3 = "#c9daeb"
    band_2 = "#97b8d3"
    band_1 = "#5b8ebb"

    h0_line = "#2f4458"

    jk_err = "firebrick"

    # ------------------------------------------------------------------
    # H0 label
    # ------------------------------------------------------------------

    h0_label = (
        "Pure isotropic H0"
        if not config.use_sel_func
        else "Isotropy + survey SF H0"
    )

    # ------------------------------------------------------------------
    # 2pACF visualization
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.fill_between(
        theta,
        h0_mean - 3.0 * h0_std,
        h0_mean + 3.0 * h0_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        theta,
        h0_mean - 2.0 * h0_std,
        h0_mean + 2.0 * h0_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )

    ax.fill_between(
        theta,
        h0_mean - h0_std,
        h0_mean + h0_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
        zorder=3,
    )

    obs_handle = ax.scatter(
        theta,
        w_obs,
        s=45,
        facecolor=jk_err,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
        label="Observed 2pACF",
    )

    jk_handle = ax.errorbar(
        theta,
        w_obs,
        yerr=jackknife.w_err,
        fmt="none",
        ecolor=jk_err,
        elinewidth=2.0,
        capsize=3.0,
        capthick=1.2,
        alpha=0.95,
        zorder=4,
        label="Jackknife",
    )

    h0_line_handle, = ax.plot(
        theta,
        h0_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=5,
    )

    ax.axhline(
        0.0,
        color="0.15",
        linewidth=1.1,
        alpha=0.9,
        zorder=0,
    )

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$w(\theta)$"
    )

    ax.set_title(
        "Angular Two-Point Correlation Function\n"
        f"(Jackknife uncertainties | {h0_label})"
    )

    h0_patch = Patch(
        facecolor=band_2,
        edgecolor=band_1,
        alpha=0.85,
        label=r"$H_0$ confidence bands",
    )

    ax.legend(
        [
            obs_handle,
            jk_handle,
            h0_line_handle,
            h0_patch,
        ],
        [
            "Observed 2pACF",
            "Jackknife",
            r"$H_0$ mean",
            r"$H_0$ confidence bands",
        ],
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    fig.savefig(
        output_2pacf,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"Saved: {output_2pacf}"
    )

    # ------------------------------------------------------------------
    # Absolute anisotropy visualization
    # ------------------------------------------------------------------

    coarse_centers = np.asarray(
        config.coarse_centers,
        dtype=float,
    )

    if len(coarse_centers) != len(abs_obs):

        raise ValueError(
            "coarse_centers and abs_obs "
            "have incompatible shapes."
        )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - 3.0 * h0_abs_std,
        h0_abs_mean + 3.0 * h0_abs_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - 2.0 * h0_abs_std,
        h0_abs_mean + 2.0 * h0_abs_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - h0_abs_std,
        h0_abs_mean + h0_abs_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
        zorder=3,
    )

    obs_handle = ax.scatter(
        coarse_centers,
        abs_obs,
        s=45,
        facecolor=jk_err,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
        label="Observed statistic",
    )

    jk_handle = ax.errorbar(
        coarse_centers,
        abs_obs,
        yerr=jackknife.abs_err,
        fmt="none",
        ecolor=jk_err,
        elinewidth=2.0,
        capsize=3.0,
        capthick=1.2,
        alpha=0.95,
        zorder=4,
        label="Jackknife",
    )

    h0_line_handle, = ax.plot(
        coarse_centers,
        h0_abs_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=5,
    )

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$|\langle w \rangle|$"
    )

    ax.set_title(
        "Absolute Sum Test\n"
        f"(Jackknife uncertainties | {h0_label})"
    )

    h0_patch = Patch(
        facecolor=band_2,
        edgecolor=band_1,
        alpha=0.85,
        label=r"$H_0$ confidence bands",
    )

    ax.legend(
        [
            obs_handle,
            jk_handle,
            h0_line_handle,
            h0_patch,
        ],
        [
            "Observed statistic",
            "Jackknife",
            r"$H_0$ mean",
            r"$H_0$ confidence bands",
        ],
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    fig.savefig(
        output_abs,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"Saved: {output_abs}"
    )


def plot_results_with_bootstrap(
    context: RuntimeContext,
    theta: FloatArray,
    w_obs: FloatArray,
    abs_obs: FloatArray,
    all_w_h0: FloatArray,
    all_abs_h0: FloatArray,
    bootstrap: BootstrapResult,
    output_2pacf: str | Path | None = None,
    output_abs: str | Path | None = None,
) -> None:
    """
    Plot isotropy diagnostics using:
        - H0 mock confidence bands
        - Bootstrap observational uncertainties

    Notes
    -----
    H0 depends on configuration:

    use_sel_func = False
        -> perfect isotropic sky

    use_sel_func = True
        -> isotropy convolved with survey selection functions
    """

    config = context.config

    outputs = context.outputs

    # ------------------------------------------------------------------
    # Output filenames
    # ------------------------------------------------------------------

    if output_2pacf is None:

        output_2pacf = (
            outputs.figures_dir
            / "2pacf_bootstrap.png"
        )

    if output_abs is None:

        output_abs = (
            outputs.figures_dir
            / "absolute_anisotropy_bootstrap.png"
        )

    output_2pacf = Path(
        output_2pacf
    )

    output_abs = Path(
        output_abs
    )

    # ------------------------------------------------------------------
    # Numerical safety
    # ------------------------------------------------------------------

    theta = np.asarray(
        theta,
        dtype=float,
    )

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

    if theta.ndim != 1:

        raise ValueError(
            "theta must be 1-dimensional."
        )

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

    if all_abs_h0.shape[1] != len(abs_obs):

        raise ValueError(
            "all_abs_h0 and abs_obs "
            "have incompatible shapes."
        )

    if len(bootstrap.w_err) != len(theta):

        raise ValueError(
            "bootstrap.w_err and theta "
            "have incompatible shapes."
        )

    if len(bootstrap.abs_err) != len(abs_obs):

        raise ValueError(
            "bootstrap.abs_err and abs_obs "
            "have incompatible shapes."
        )

    # ------------------------------------------------------------------
    # Remove invalid H0 realizations
    # ------------------------------------------------------------------

    valid_w = np.all(
        np.isfinite(all_w_h0),
        axis=1,
    )

    valid_abs = np.all(
        np.isfinite(all_abs_h0),
        axis=1,
    )

    if not np.all(valid_w):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid_w)} invalid "
            f"H0 w(theta) realizations."
        )

        all_w_h0 = all_w_h0[
            valid_w
        ]

    if not np.all(valid_abs):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid_abs)} invalid "
            f"H0 absolute-sum realizations."
        )

        all_abs_h0 = all_abs_h0[
            valid_abs
        ]

    if len(all_w_h0) < 2:

        raise RuntimeError(
            "Too few valid H0 w(theta) "
            "realizations remain."
        )

    if len(all_abs_h0) < 2:

        raise RuntimeError(
            "Too few valid H0 absolute-sum "
            "realizations remain."
        )

    # ------------------------------------------------------------------
    # Plot style
    # ------------------------------------------------------------------

    sns.set_theme(
        style="white",
        context="talk",
        rc={
            "axes.edgecolor": "0.25",
            "axes.linewidth": 1.1,
        },
    )

    # ------------------------------------------------------------------
    # H0 ensemble statistics
    # ------------------------------------------------------------------

    h0_mean = np.nanmean(
        all_w_h0,
        axis=0,
    )

    h0_std = np.nanstd(
        all_w_h0,
        axis=0,
        ddof=1,
    )

    h0_abs_mean = np.nanmean(
        all_abs_h0,
        axis=0,
    )

    h0_abs_std = np.nanstd(
        all_abs_h0,
        axis=0,
        ddof=1,
    )

    h0_std = np.nan_to_num(
        h0_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    h0_abs_std = np.nan_to_num(
        h0_abs_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    band_3 = "#c9daeb"
    band_2 = "#97b8d3"
    band_1 = "#5b8ebb"

    h0_line = "#2f4458"

    obs_color = "white"

    bootstrap_color = "black"

    # ------------------------------------------------------------------
    # H0 label
    # ------------------------------------------------------------------

    h0_label = (
        "Pure isotropic H0"
        if not config.use_sel_func
        else "Isotropy + survey SF H0"
    )

    # ------------------------------------------------------------------
    # 2pACF visualization
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.fill_between(
        theta,
        h0_mean - 3.0 * h0_std,
        h0_mean + 3.0 * h0_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        theta,
        h0_mean - 2.0 * h0_std,
        h0_mean + 2.0 * h0_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )

    ax.fill_between(
        theta,
        h0_mean - h0_std,
        h0_mean + h0_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
        zorder=3,
    )

    obs_handle = ax.scatter(
        theta,
        w_obs,
        s=45,
        facecolor="firebrick",
        edgecolor=obs_color,
        linewidth=1.0,
        zorder=6,
        label="Observed 2pACF",
    )

    bootstrap_handle = ax.errorbar(
        theta,
        w_obs,
        yerr=bootstrap.w_err,
        fmt="none",
        ecolor=bootstrap_color,
        elinewidth=2.0,
        alpha=0.95,
        capsize=3.0,
        capthick=1.2,
        zorder=5,
        label="Bootstrap",
    )

    h0_line_handle, = ax.plot(
        theta,
        h0_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=4,
    )

    ax.axhline(
        0.0,
        color="0.15",
        linewidth=1.1,
        alpha=0.9,
        zorder=0,
    )

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$w(\theta)$"
    )

    ax.set_title(
        "Angular Two-Point Correlation Function\n"
        f"(Bootstrap uncertainties | {h0_label})"
    )

    h0_patch = Patch(
        facecolor=band_2,
        edgecolor=band_1,
        alpha=0.85,
        label=r"$H_0$ confidence bands",
    )

    ax.legend(
        [
            obs_handle,
            bootstrap_handle,
            h0_line_handle,
            h0_patch,
        ],
        [
            "Observed 2pACF",
            "Bootstrap",
            r"$H_0$ mean",
            r"$H_0$ confidence bands",
        ],
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    fig.savefig(
        output_2pacf,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"Saved: {output_2pacf}"
    )

    # ------------------------------------------------------------------
    # Absolute anisotropy visualization
    # ------------------------------------------------------------------

    coarse_centers = np.asarray(
        config.coarse_centers,
        dtype=float,
    )

    if len(coarse_centers) != len(abs_obs):

        raise ValueError(
            "coarse_centers and abs_obs "
            "have incompatible shapes."
        )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - 3.0 * h0_abs_std,
        h0_abs_mean + 3.0 * h0_abs_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - 2.0 * h0_abs_std,
        h0_abs_mean + 2.0 * h0_abs_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - h0_abs_std,
        h0_abs_mean + h0_abs_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
        zorder=3,
    )

    obs_handle = ax.scatter(
        coarse_centers,
        abs_obs,
        s=48,
        facecolor="firebrick",
        edgecolor=obs_color,
        linewidth=1.0,
        zorder=6,
        label="Observed statistic",
    )

    bootstrap_handle = ax.errorbar(
        coarse_centers,
        abs_obs,
        yerr=bootstrap.abs_err,
        fmt="none",
        ecolor=bootstrap_color,
        elinewidth=2.0,
        alpha=0.95,
        capsize=3.0,
        capthick=1.2,
        zorder=5,
        label="Bootstrap",
    )

    h0_line_handle, = ax.plot(
        coarse_centers,
        h0_abs_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=4,
    )

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$|\langle w \rangle|$"
    )

    ax.set_title(
        "Absolute Sum Test\n"
        f"(Bootstrap uncertainties | {h0_label})"
    )

    h0_patch = Patch(
        facecolor=band_2,
        edgecolor=band_1,
        alpha=0.85,
        label=r"$H_0$ confidence bands",
    )

    ax.legend(
        [
            obs_handle,
            bootstrap_handle,
            h0_line_handle,
            h0_patch,
        ],
        [
            "Observed statistic",
            "Bootstrap",
            r"$H_0$ mean",
            r"$H_0$ confidence bands",
        ],
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    fig.savefig(
        output_abs,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"Saved: {output_abs}"
    )
    

def plot_results_jk_vs_bootstrap(
    context: RuntimeContext,
    theta: FloatArray,
    w_obs: FloatArray,
    abs_obs: FloatArray,
    all_w_h0: FloatArray,
    all_abs_h0: FloatArray,
    jackknife: JackknifeResult,
    bootstrap: BootstrapResult,
    output_2pacf: str | Path | None = None,
    output_abs: str | Path | None = None,
) -> None:
    """
    Compare jackknife and bootstrap uncertainty estimates.

    Notes
    -----
    H0 depends on configuration:

    use_sel_func = False
        -> perfect isotropic sky

    use_sel_func = True
        -> isotropy convolved with survey selection functions
    """

    config = context.config

    outputs = context.outputs

    # ------------------------------------------------------------------
    # Output filenames
    # ------------------------------------------------------------------

    if output_2pacf is None:

        output_2pacf = (
            outputs.figures_dir
            / "2pacf_jk_vs_bootstrap.png"
        )

    if output_abs is None:

        output_abs = (
            outputs.figures_dir
            / "absolute_anisotropy_jk_vs_bootstrap.png"
        )

    output_2pacf = Path(
        output_2pacf
    )

    output_abs = Path(
        output_abs
    )

    # ------------------------------------------------------------------
    # Numerical safety
    # ------------------------------------------------------------------

    theta = np.asarray(
        theta,
        dtype=float,
    )

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

    if theta.ndim != 1:

        raise ValueError(
            "theta must be 1-dimensional."
        )

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

    if all_abs_h0.shape[1] != len(abs_obs):

        raise ValueError(
            "all_abs_h0 and abs_obs "
            "have incompatible shapes."
        )

    if len(jackknife.w_err) != len(theta):

        raise ValueError(
            "jackknife.w_err and theta "
            "have incompatible shapes."
        )

    if len(bootstrap.w_err) != len(theta):

        raise ValueError(
            "bootstrap.w_err and theta "
            "have incompatible shapes."
        )

    if len(jackknife.abs_err) != len(abs_obs):

        raise ValueError(
            "jackknife.abs_err and abs_obs "
            "have incompatible shapes."
        )

    if len(bootstrap.abs_err) != len(abs_obs):

        raise ValueError(
            "bootstrap.abs_err and abs_obs "
            "have incompatible shapes."
        )

    # ------------------------------------------------------------------
    # Remove invalid H0 realizations
    # ------------------------------------------------------------------

    valid_w = np.all(
        np.isfinite(all_w_h0),
        axis=1,
    )

    valid_abs = np.all(
        np.isfinite(all_abs_h0),
        axis=1,
    )

    if not np.all(valid_w):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid_w)} invalid "
            f"H0 w(theta) realizations."
        )

        all_w_h0 = all_w_h0[
            valid_w
        ]

    if not np.all(valid_abs):

        print(
            f"\nWARNING: removing "
            f"{np.sum(~valid_abs)} invalid "
            f"H0 absolute-sum realizations."
        )

        all_abs_h0 = all_abs_h0[
            valid_abs
        ]

    if len(all_w_h0) < 2:

        raise RuntimeError(
            "Too few valid H0 w(theta) "
            "realizations remain."
        )

    if len(all_abs_h0) < 2:

        raise RuntimeError(
            "Too few valid H0 absolute-sum "
            "realizations remain."
        )

    # ------------------------------------------------------------------
    # Plot style
    # ------------------------------------------------------------------

    sns.set_theme(
        style="white",
        context="talk",
        rc={
            "axes.edgecolor": "0.25",
            "axes.linewidth": 1.1,
        },
    )

    # ------------------------------------------------------------------
    # H0 ensemble statistics
    # ------------------------------------------------------------------

    h0_mean = np.nanmean(
        all_w_h0,
        axis=0,
    )

    h0_std = np.nanstd(
        all_w_h0,
        axis=0,
        ddof=1,
    )

    h0_abs_mean = np.nanmean(
        all_abs_h0,
        axis=0,
    )

    h0_abs_std = np.nanstd(
        all_abs_h0,
        axis=0,
        ddof=1,
    )

    h0_std = np.nan_to_num(
        h0_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    h0_abs_std = np.nan_to_num(
        h0_abs_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    band_3 = "#c9daeb"

    band_2 = "#97b8d3"

    band_1 = "#5b8ebb"

    h0_line = "#2f4458"

    jk_err = "firebrick"

    bootstrap_color = "black"

    # ------------------------------------------------------------------
    # H0 label
    # ------------------------------------------------------------------

    h0_label = (
        "Pure isotropic H0"
        if not config.use_sel_func
        else "Isotropy + survey SF H0"
    )

    # ------------------------------------------------------------------
    # 2pACF visualization
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    # ------------------------------------------------------------------
    # H0 confidence bands
    # ------------------------------------------------------------------

    ax.fill_between(
        theta,
        h0_mean - 3.0 * h0_std,
        h0_mean + 3.0 * h0_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        theta,
        h0_mean - 2.0 * h0_std,
        h0_mean + 2.0 * h0_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )

    ax.fill_between(
        theta,
        h0_mean - h0_std,
        h0_mean + h0_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Observed measurements
    # ------------------------------------------------------------------

    obs_handle = ax.scatter(
        theta,
        w_obs,
        s=45,
        facecolor=jk_err,
        edgecolor="white",
        linewidth=1.0,
        zorder=7,
        label="Observed 2pACF",
    )

    # ------------------------------------------------------------------
    # Bootstrap uncertainties
    # ------------------------------------------------------------------

    bootstrap_handle = ax.errorbar(
        theta,
        w_obs,
        yerr=bootstrap.w_err,
        fmt="none",
        ecolor=bootstrap_color,
        elinewidth=2.0,
        capsize=3.0,
        capthick=1.2,
        alpha=0.90,
        zorder=4,
        label="Bootstrap",
    )

    # ------------------------------------------------------------------
    # Jackknife uncertainties
    # ------------------------------------------------------------------

    jk_handle = ax.errorbar(
        theta,
        w_obs,
        yerr=jackknife.w_err,
        fmt="none",
        ecolor=jk_err,
        elinewidth=1.4,
        capsize=2.0,
        capthick=1.0,
        alpha=0.75,
        zorder=5,
        label="Jackknife",
    )

    # ------------------------------------------------------------------
    # H0 mean
    # ------------------------------------------------------------------

    h0_line_handle, = ax.plot(
        theta,
        h0_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=6,
    )

    ax.axhline(
        0.0,
        color="0.15",
        linewidth=1.1,
        alpha=0.9,
        zorder=0,
    )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$w(\theta)$"
    )

    ax.set_title(
        "2pACF Uncertainty Comparison\n"
        f"(Jackknife vs Bootstrap | {h0_label})"
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    h0_patch = Patch(
        facecolor=band_2,
        edgecolor=band_1,
        alpha=0.85,
        label=r"$H_0$ confidence bands",
    )

    ax.legend(
        [
            obs_handle,
            jk_handle,
            bootstrap_handle,
            h0_line_handle,
            h0_patch,
        ],
        [
            "Observed 2pACF",
            "Jackknife",
            "Bootstrap",
            r"$H_0$ mean",
            r"$H_0$ confidence bands",
        ],
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    fig.savefig(
        output_2pacf,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"Saved: {output_2pacf}"
    )

    # ------------------------------------------------------------------
    # Absolute anisotropy visualization
    # ------------------------------------------------------------------

    coarse_centers = np.asarray(
        config.coarse_centers,
        dtype=float,
    )

    if len(coarse_centers) != len(abs_obs):

        raise ValueError(
            "coarse_centers and abs_obs "
            "have incompatible shapes."
        )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    # ------------------------------------------------------------------
    # H0 confidence bands
    # ------------------------------------------------------------------

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - 3.0 * h0_abs_std,
        h0_abs_mean + 3.0 * h0_abs_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - 2.0 * h0_abs_std,
        h0_abs_mean + 2.0 * h0_abs_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )

    ax.fill_between(
        coarse_centers,
        h0_abs_mean - h0_abs_std,
        h0_abs_mean + h0_abs_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Observed statistic
    # ------------------------------------------------------------------

    obs_handle = ax.scatter(
        coarse_centers,
        abs_obs,
        s=45,
        facecolor=jk_err,
        edgecolor="white",
        linewidth=1.0,
        zorder=7,
        label="Observed statistic",
    )

    # ------------------------------------------------------------------
    # Bootstrap uncertainties
    # ------------------------------------------------------------------

    bootstrap_handle = ax.errorbar(
        coarse_centers,
        abs_obs,
        yerr=bootstrap.abs_err,
        fmt="none",
        ecolor=bootstrap_color,
        elinewidth=2.0,
        capsize=3.0,
        capthick=1.2,
        alpha=0.90,
        zorder=4,
        label="Bootstrap",
    )

    # ------------------------------------------------------------------
    # Jackknife uncertainties
    # ------------------------------------------------------------------

    jk_handle = ax.errorbar(
        coarse_centers,
        abs_obs,
        yerr=jackknife.abs_err,
        fmt="none",
        ecolor=jk_err,
        elinewidth=1.4,
        capsize=2.0,
        capthick=1.0,
        alpha=0.75,
        zorder=5,
        label="Jackknife",
    )

    # ------------------------------------------------------------------
    # H0 mean
    # ------------------------------------------------------------------

    h0_line_handle, = ax.plot(
        coarse_centers,
        h0_abs_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=6,
    )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$|\langle w \rangle|$"
    )

    ax.set_title(
        "Absolute Sum Test Uncertainty Comparison\n"
        f"(Jackknife vs Bootstrap | {h0_label})"
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    h0_patch = Patch(
        facecolor=band_2,
        edgecolor=band_1,
        alpha=0.85,
        label=r"$H_0$ confidence bands",
    )

    ax.legend(
        [
            obs_handle,
            jk_handle,
            bootstrap_handle,
            h0_line_handle,
            h0_patch,
        ],
        [
            "Observed statistic",
            "Jackknife",
            "Bootstrap",
            r"$H_0$ mean",
            r"$H_0$ confidence bands",
        ],
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    fig.savefig(
        output_abs,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    print(
        f"Saved: {output_abs}"
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


def save_report(
    context: RuntimeContext,
    results: dict[str, object],
    paths: dict[str, str] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Save consolidated human-readable analysis report.
    """

    config = context.config

    outputs = context.outputs

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------

    if output_path is None:

        output_path = (
            outputs.report_dir
            / "summary.md"
        )

    output_path = Path(
        output_path
    )

    # ------------------------------------------------------------------
    # Required entries
    # ------------------------------------------------------------------

    required_keys = [
        "stats",
        "jackknife",
        "df_data",
        "all_w_h0",
        "all_abs_h0",
    ]

    missing = [
        key
        for key in required_keys
        if key not in results
    ]

    if len(missing) > 0:

        raise KeyError(
            "Missing required report entries: "
            f"{missing}"
        )

    # ------------------------------------------------------------------
    # Required objects
    # ------------------------------------------------------------------

    stats = results["stats"]

    jackknife = results["jackknife"]

    df_data = results["df_data"]

    all_w_h0 = np.asarray(
        results["all_w_h0"],
        dtype=float,
    )

    all_abs_h0 = np.asarray(
        results["all_abs_h0"],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Optional diagnostics
    # ------------------------------------------------------------------

    chi2_bin_diagnostics = results.get(
        "chi2_bin_diagnostics"
    )

    chi2_svd_modes = results.get(
        "svd_mode_contributions"
    )

    if chi2_svd_modes is None:

        chi2_svd_modes = results.get(
            "chi2_svd_modes"
        )

    sf_validation = results.get(
        "sf_validation"
    )

    corr_df = results.get(
        "corr_df"
    )

    if corr_df is None:

        corr_df = results.get(
            "intersurvey_corr"
        )

    overlap_df = results.get(
        "overlap_df"
    )

    if overlap_df is None:

        overlap_df = results.get(
            "intersurvey_overlap"
        )

    # ------------------------------------------------------------------
    # Markdown buffer
    # ------------------------------------------------------------------

    lines: list[str] = []

    lines.append(
        "# FRB Isotropy Analysis Report"
    )

    lines.append("")

    lines.append(
        "Consolidated output of the isotropy pipeline."
    )

    lines.append("")

    # ------------------------------------------------------------------
    # Run configuration
    # ------------------------------------------------------------------

    _append_key_values(
        lines,
        "Run Configuration",
        {
            "run_tag":
                config.run_tag,

            "catalog_path":
                str(
                    config.catalog_path
                ),

            "catalog_size":
                int(len(df_data)),

            "use_gal_mask":
                bool(
                    config.use_gal_mask
                ),

            "use_selection_function":
                bool(
                    config.use_sel_func
                ),

            "gal_cut_deg":
                config.gal_cut,

            "bin_size_deg":
                config.bin_size,

            "n_bins":
                stats.n_bins,

            "n_rand_factor":
                config.n_rand_factor,

            "n_mocks":
                stats.n_mocks,

            "n_ensemble":
                config.n_ensemble,

            "n_mocks_per_ensemble":
                config.n_mocks_per_ensemble,

            "nside_sf":
                config.nside_sf,

            "smooth_sigma_deg":
                config.smooth_sigma,

            "perturbation_scale":
                config.perturbation_scale,

            "nside_jackknife":
                config.nside_jackknife,

            "svd_eigenvalue_cut":
                stats.covariance.svd_eigenvalue_cut,
        },
    )

    # ------------------------------------------------------------------
    # Full covariance statistics
    # ------------------------------------------------------------------

    _append_key_values(
        lines,
        "Full Covariance Diagnostic",
        {
            "chi2":
                stats.chi2.chi2,

            "chi2_red":
                stats.chi2.chi2_red,

            "p_chi2_analytic":
                stats.chi2.p_chi2,

            "p_chi2_empirical":
                stats.chi2.p_empirical,

            "p_chi2_empirical_floor":
                stats.chi2.p_empirical_floor,

            "sigma_equiv_empirical":
                stats.chi2.sigma_equiv,

            "hartlap_factor":
                stats.covariance.hartlap_factor,

            "rms_normalized_deviation":
                stats.absolute.global_tension,
        },
    )

    # ------------------------------------------------------------------
    # SVD statistics
    # ------------------------------------------------------------------

    _append_key_values(
        lines,
        "Primary Isotropy Statistic (SVD-Regularized)",
        {
            "chi2_svd":
                stats.svd.chi2_svd,

            "chi2_svd_red":
                stats.svd.chi2_svd_red,

            "p_chi2_svd_analytic":
                stats.svd.p_chi2_svd,

            "p_chi2_svd_empirical":
                stats.svd.p_svd_empirical,

            "sigma_svd_empirical":
                stats.svd.sigma_svd_equiv,

            "svd_modes_kept":
                (
                    f"{stats.covariance.svd_modes_kept}/"
                    f"{stats.n_bins}"
                ),

            "svd_retained_condition":
                stats.covariance.svd_condition,
        },
    )

    # ------------------------------------------------------------------
    # Covariance diagnostics
    # ------------------------------------------------------------------

    _append_key_values(
        lines,
        "Covariance Diagnostics",
        {
            "mocks":
                stats.n_mocks,

            "bins":
                stats.n_bins,

            "covariance_rank":
                (
                    f"{stats.covariance.covariance_rank}/"
                    f"{stats.n_bins}"
                ),

            "covariance_condition":
                stats.covariance.covariance_condition,

            "effective_modes":
                stats.covariance.n_eff,

            "h0_w_shape":
                tuple(
                    all_w_h0.shape
                ),

            "h0_abs_shape":
                tuple(
                    all_abs_h0.shape
                ),
        },
    )

    # ------------------------------------------------------------------
    # Non-parametric tests
    # ------------------------------------------------------------------

    _append_key_values(
        lines,
        "Non-parametric Profile Tests",
        {
            "w_ks_stat":
                stats.nonparametric.ks_w_stat,

            "w_ks_pvalue":
                stats.nonparametric.ks_w_pvalue,

            "w_ks_empirical_p":
                stats.nonparametric.ks_w_empirical_p,

            "w_ad_stat":
                stats.nonparametric.ad_w_stat,

            "w_ad_pvalue":
                stats.nonparametric.ad_w_pvalue,

            "w_ad_empirical_p":
                stats.nonparametric.ad_w_empirical_p,

            "abs_ks_stat":
                stats.nonparametric.ks_abs_stat,

            "abs_ks_pvalue":
                stats.nonparametric.ks_abs_pvalue,

            "abs_ks_empirical_p":
                stats.nonparametric.ks_abs_empirical_p,

            "abs_ad_stat":
                stats.nonparametric.ad_abs_stat,

            "abs_ad_pvalue":
                stats.nonparametric.ad_abs_pvalue,

            "abs_ad_empirical_p":
                stats.nonparametric.ad_abs_empirical_p,
        },
    )

    # ------------------------------------------------------------------
    # Absolute anisotropy
    # ------------------------------------------------------------------

    _append_key_values(
        lines,
        "Absolute Anisotropy Amplitude",
        {
            "observed_rms_absolute_amplitude":
                stats.absolute.abs_observed_stat,

            "empirical_pvalue":
                stats.absolute.abs_empirical_p,
        },
    )

    # ------------------------------------------------------------------
    # Jackknife summary
    # ------------------------------------------------------------------

    _append_key_values(
        lines,
        "Jackknife Summary",
        {
            "n_regions":
                int(
                    len(
                        jackknife.regions
                    )
                ),

            "median_sigma_w":
                float(
                    np.nanmedian(
                        jackknife.w_err
                    )
                ),

            "min_sigma_w":
                float(
                    np.nanmin(
                        jackknife.w_err
                    )
                ),

            "max_sigma_w":
                float(
                    np.nanmax(
                        jackknife.w_err
                    )
                ),

            "median_sigma_abs":
                float(
                    np.nanmedian(
                        jackknife.abs_err
                    )
                ),

            "min_sigma_abs":
                float(
                    np.nanmin(
                        jackknife.abs_err
                    )
                ),

            "max_sigma_abs":
                float(
                    np.nanmax(
                        jackknife.abs_err
                    )
                ),
        },
    )

    # ------------------------------------------------------------------
    # Chi-square bin audit
    # ------------------------------------------------------------------

    if (
        isinstance(
            chi2_bin_diagnostics,
            pd.DataFrame,
        )
        and not chi2_bin_diagnostics.empty
    ):

        lines.append(
            "## Chi-square Bin Audit"
        )

        lines.append("")

        lines.append(
            "```text"
        )

        bin_columns = [
            "bin_index",
            "theta_deg",
            "w_obs",
            "h0_mean",
            "h0_std",
            "delta",
            "pull",
            "diagonal_chi2_contribution",
        ]

        valid_columns = [
            col
            for col in bin_columns
            if col in chi2_bin_diagnostics.columns
        ]

        lines.append(
            chi2_bin_diagnostics[
                valid_columns
            ]
            .head(12)
            .to_string(
                index=False
            )
        )

        lines.append(
            "```"
        )

        lines.append("")

    # ------------------------------------------------------------------
    # SVD audit
    # ------------------------------------------------------------------

    if (
        isinstance(
            chi2_svd_modes,
            pd.DataFrame,
        )
        and not chi2_svd_modes.empty
    ):

        lines.append(
            "## Chi-square SVD Mode Audit"
        )

        lines.append("")

        lines.append(
            "```text"
        )

        mode_columns = [
            "mode_index",
            "eigenvalue",
            "relative_eigenvalue",
            "kept_by_svd_cut",
            "delta_projection",
            "raw_chi2_contribution",
            "svd_chi2_contribution",
            "fractional_svd_chi2_contribution",
            "dominant_theta_deg",
            "dominant_loading",
        ]

        valid_columns = [
            col
            for col in mode_columns
            if col in chi2_svd_modes.columns
        ]

        lines.append(
            chi2_svd_modes[
                valid_columns
            ]
            .head(12)
            .to_string(
                index=False
            )
        )

        lines.append(
            "```"
        )

        lines.append("")

    # ------------------------------------------------------------------
    # Selection-function validation
    # ------------------------------------------------------------------

    if (
        isinstance(
            sf_validation,
            pd.DataFrame,
        )
        and not sf_validation.empty
    ):

        lines.append(
            "## Selection Function Validation"
        )

        lines.append("")

        lines.append(
            "```text"
        )

        sf_top = (
            sf_validation
            .sort_values(
                "weight",
                ascending=False,
            )
            .head(12)
        )

        lines.append(
            sf_top.to_string(
                index=False
            )
        )

        lines.append(
            "```"
        )

        lines.append("")

    # ------------------------------------------------------------------
    # Intersurvey correlations
    # ------------------------------------------------------------------

    if (
        isinstance(
            corr_df,
            pd.DataFrame,
        )
        and not corr_df.empty
    ):

        lines.append(
            "## Intersurvey Correlations"
        )

        lines.append("")

        lines.append(
            "```text"
        )

        top_corr = _top_matrix_pairs(
            corr_df,
            n=12,
            absolute=True,
        )

        lines.append(
            top_corr.to_string(
                index=False
            )
        )

        lines.append(
            "```"
        )

        lines.append("")

    # ------------------------------------------------------------------
    # Intersurvey overlap
    # ------------------------------------------------------------------

    if (
        isinstance(
            overlap_df,
            pd.DataFrame,
        )
        and not overlap_df.empty
    ):

        lines.append(
            "## Intersurvey Overlap"
        )

        lines.append("")

        lines.append(
            "```text"
        )

        top_overlap = _top_matrix_pairs(
            overlap_df,
            n=12,
            absolute=False,
        )

        lines.append(
            top_overlap.to_string(
                index=False
            )
        )

        lines.append(
            "```"
        )

        lines.append("")

    # ------------------------------------------------------------------
    # Saved files
    # ------------------------------------------------------------------

    if paths is not None and len(paths) > 0:

        lines.append(
            "## Saved Files"
        )

        lines.append("")

        lines.append(
            "```text"
        )

        for name, path in paths.items():

            lines.append(
                f"{name}: {path}"
            )

        lines.append(
            "```"
        )

        lines.append("")

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "\n".join(lines).rstrip()
            + "\n"
        )

    print(
        f"\nSaved report: "
        f"{output_path}"
    )

    return str(output_path)


def save_tables(
    context: RuntimeContext,
    results: dict[str, object],
    prefix: str | None = None,
) -> dict[str, str]:
    """
    Save analysis tables and diagnostics.
    """

    config = context.config

    outputs = context.outputs

    # ------------------------------------------------------------------
    # Prefix
    # ------------------------------------------------------------------

    if prefix is None:

        prefix = config.run_tag

    prefix = str(prefix)

    # ------------------------------------------------------------------
    # Ensure output directories exist
    # ------------------------------------------------------------------

    outputs.tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    paths = {

        "sf_validation":
            outputs.tables_dir
            / f"{prefix}_sf_validation.csv",

        "intersurvey_corr":
            outputs.tables_dir
            / f"{prefix}_intersurvey_corr.csv",

        "intersurvey_overlap":
            outputs.tables_dir
            / f"{prefix}_intersurvey_overlap.csv",

        "stats_summary":
            outputs.tables_dir
            / f"{prefix}_stats_summary.csv",

        "jackknife_w_errors":
            outputs.tables_dir
            / f"{prefix}_jackknife_w_errors.csv",

        "jackknife_abs_errors":
            outputs.tables_dir
            / f"{prefix}_jackknife_abs_errors.csv",

        "covariance_matrix":
            outputs.tables_dir
            / f"{prefix}_covariance_matrix.csv",

        "chi2_bin_diagnostics":
            outputs.tables_dir
            / f"{prefix}_chi2_bin_diagnostics.csv",

        "svd_mode_contributions":
            outputs.tables_dir
            / f"{prefix}_svd_mode_contributions.csv",

        "report":
            outputs.report_dir
            / f"{prefix}_summary.md",
    }

    # ------------------------------------------------------------------
    # Selection-function diagnostics
    # ------------------------------------------------------------------

    sf_validation = results.get(
        "sf_validation"
    )

    if (
        isinstance(
            sf_validation,
            pd.DataFrame,
        )
        and not sf_validation.empty
    ):

        sf_validation.to_csv(
            paths["sf_validation"],
            index=False,
        )

    # ------------------------------------------------------------------
    # Intersurvey correlation matrix
    # ------------------------------------------------------------------

    intersurvey_corr = (
        results.get(
            "corr_df"
        )
    )

    if intersurvey_corr is None:

        intersurvey_corr = (
            results.get(
                "intersurvey_corr"
            )
        )

    if (
        isinstance(
            intersurvey_corr,
            pd.DataFrame,
        )
        and not intersurvey_corr.empty
    ):

        intersurvey_corr.to_csv(
            paths["intersurvey_corr"]
        )

    # ------------------------------------------------------------------
    # Intersurvey overlap matrix
    # ------------------------------------------------------------------

    intersurvey_overlap = (
        results.get(
            "overlap_df"
        )
    )

    if intersurvey_overlap is None:

        intersurvey_overlap = (
            results.get(
                "intersurvey_overlap"
            )
        )

    if (
        isinstance(
            intersurvey_overlap,
            pd.DataFrame,
        )
        and not intersurvey_overlap.empty
    ):

        intersurvey_overlap.to_csv(
            paths["intersurvey_overlap"]
        )

    # ------------------------------------------------------------------
    # Statistics summary
    # ------------------------------------------------------------------

    if "stats" not in results:

        raise KeyError(
            "results must contain "
            "'stats'."
        )

    stats = results["stats"]

    stats_rows = [

        [
            "chi2",
            "chi2",
            stats.chi2.chi2,
        ],

        [
            "chi2",
            "chi2_red",
            stats.chi2.chi2_red,
        ],

        [
            "chi2",
            "p_empirical",
            stats.chi2.p_empirical,
        ],

        [
            "svd",
            "chi2_svd",
            stats.svd.chi2_svd,
        ],

        [
            "svd",
            "chi2_svd_red",
            stats.svd.chi2_svd_red,
        ],

        [
            "svd",
            "p_svd_empirical",
            stats.svd.p_svd_empirical,
        ],

        [
            "covariance",
            "effective_modes",
            stats.covariance.n_eff,
        ],

        [
            "covariance",
            "condition",
            stats.covariance.covariance_condition,
        ],

        [
            "anisotropy",
            "abs_stat",
            stats.absolute.abs_observed_stat,
        ],

        [
            "anisotropy",
            "abs_p",
            stats.absolute.abs_empirical_p,
        ],
    ]

    stats_df = pd.DataFrame(
        stats_rows,
        columns=[
            "category",
            "statistic",
            "value",
        ],
    )

    stats_df.to_csv(
        paths["stats_summary"],
        index=False,
    )

    # ------------------------------------------------------------------
    # Jackknife diagnostics
    # ------------------------------------------------------------------

    required_keys = [
        "jackknife",
        "theta",
        "w_obs",
        "abs_obs",
    ]

    missing = [
        key
        for key in required_keys
        if key not in results
    ]

    if len(missing) > 0:

        raise KeyError(
            "Missing required results entries: "
            f"{missing}"
        )

    jackknife = results["jackknife"]

    theta = np.asarray(
        results["theta"],
        dtype=float,
    )

    w_obs = np.asarray(
        results["w_obs"],
        dtype=float,
    )

    abs_obs = np.asarray(
        results["abs_obs"],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # w(theta) jackknife table
    # ------------------------------------------------------------------

    jackknife_w_df = pd.DataFrame(
        {
            "theta_deg":
                theta,

            "w_obs":
                w_obs,

            "w_jackknife_mean":
                np.asarray(
                    jackknife.w_mean,
                    dtype=float,
                ),

            "w_jackknife_err":
                np.asarray(
                    jackknife.w_err,
                    dtype=float,
                ),
        }
    )

    jackknife_w_df.to_csv(
        paths["jackknife_w_errors"],
        index=False,
    )

    # ------------------------------------------------------------------
    # Absolute statistic jackknife table
    # ------------------------------------------------------------------

    jackknife_abs_df = pd.DataFrame(
        {
            "theta_center_deg":
                np.asarray(
                    config.coarse_centers,
                    dtype=float,
                ),

            "abs_obs":
                abs_obs,

            "abs_jackknife_mean":
                np.asarray(
                    jackknife.abs_mean,
                    dtype=float,
                ),

            "abs_jackknife_err":
                np.asarray(
                    jackknife.abs_err,
                    dtype=float,
                ),
        }
    )

    jackknife_abs_df.to_csv(
        paths["jackknife_abs_errors"],
        index=False,
    )

    # ------------------------------------------------------------------
    # Chi-square diagnostics
    # ------------------------------------------------------------------

    if (
        "all_w_h0"
        not in results
    ):

        raise KeyError(
            "results must contain "
            "'all_w_h0'."
        )

    (
        chi2_bin_diagnostics,
        svd_mode_contributions,
    ) = compute_chi2_diagnostic_tables(

        context=context,

        theta=theta,

        w_obs=w_obs,

        all_w_h0=np.asarray(
            results["all_w_h0"],
            dtype=float,
        ),

        eigenvalue_cut=(
            config.svd_eigenvalue_cut
        ),
    )

    # ------------------------------------------------------------------
    # Store back into results
    # ------------------------------------------------------------------

    results[
        "chi2_bin_diagnostics"
    ] = chi2_bin_diagnostics

    results[
        "svd_mode_contributions"
    ] = svd_mode_contributions

    # ------------------------------------------------------------------
    # Save chi2 diagnostics
    # ------------------------------------------------------------------

    chi2_bin_diagnostics.to_csv(
        paths["chi2_bin_diagnostics"],
        index=False,
    )

    svd_mode_contributions.to_csv(
        paths["svd_mode_contributions"],
        index=False,
    )

    # ------------------------------------------------------------------
    # Covariance matrix
    # ------------------------------------------------------------------

    cov = results.get(
        "covariance_matrix"
    )

    if cov is not None:

        cov = np.asarray(
            cov,
            dtype=float,
        )

        pd.DataFrame(
            cov
        ).to_csv(
            paths["covariance_matrix"],
            index=False,
        )

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    save_report(

        context=context,

        results=results,

        paths={
            key: str(value)
            for key, value in paths.items()
        },

        output_path=str(
            paths["report"]
        ),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(
        "\n--- Saved analysis tables ---"
    )

    for name, path in paths.items():

        print(
            f"{name}: {path}"
        )

    # ------------------------------------------------------------------
    # Return stringified paths
    # ------------------------------------------------------------------

    return {
        key: str(value)
        for key, value in paths.items()
    }


def main(
    config: AnalysisConfig,
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

    context = create_runtime_context(
        config
    )

    log_runtime_configuration(
        context
    )

    outputs = context.outputs

    # ------------------------------------------------------------------
    # Runtime switches
    # ------------------------------------------------------------------

    RUN_TOP_SURVEY_MAPS = True

    print("\n==================================================")
    print("FRB ISOTROPY ANALYSIS PIPELINE")
    print("==================================================")

    # ------------------------------------------------------------------
    # Load observed catalog
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("LOAD OBSERVED CATALOG")
    print("==================================================")

    df_data = load_catalog(
        context
    )

    df_data = apply_mask(
        context,
        df_data,
    )

    print(
        f"Masked catalog size: "
        f"{len(df_data)}"
    )

    if len(df_data) < 10:

        raise RuntimeError(
            "Catalog too small after masking."
        )

    # ------------------------------------------------------------------
    # Selection functions
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("SELECTION FUNCTIONS")
    print("==================================================")

    selection_functions = (
        build_survey_selection_functions_improved(

            context=context,

            df_data=df_data,

            nside=config.nside_sf,

            smooth_sigma=config.smooth_sigma,
        )
    )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    if selection_functions is not None:

        sf_dict = (
            selection_functions.sf_dict
        )

        sf_cov_diag_dict = (
            selection_functions.sf_cov_diag_dict
        )

        survey_weights = (
            selection_functions.survey_weights
        )

    else:

        sf_dict = None

        sf_cov_diag_dict = None

        survey_weights = None

    # ------------------------------------------------------------------
    # Selection-function validation
    # ------------------------------------------------------------------

    sf_validation = None

    if config.use_sel_func:

        validator = (
            SelectionFunctionValidator(

                context=context,

                sf_set=selection_functions,
            )
        )

        sf_validation = (
            validator.check_coverage()
        )

        validator.plot_sf(

            save_prefix=(

                outputs.figures_dir
                / "sf_validation"
            ),
        )

    else:

        print(
            "\nSelection-function validation skipped."
        )

    # ------------------------------------------------------------------
    # Optional survey maps
    # ------------------------------------------------------------------

    survey_map_summary = None

    if (
        RUN_TOP_SURVEY_MAPS
        and config.use_sel_func
    ):

        survey_map_summary = (
            plot_top_survey_maps(

                context=context,

                df_data=df_data,

                sf_set=selection_functions,

                max_surveys=12,

                output_prefix=(

                    outputs.figures_dir
                    / "survey_maps"
                ),
            )
        )

    # ------------------------------------------------------------------
    # Intersurvey diagnostics
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("INTERSURVEY DIAGNOSTICS")
    print("==================================================")

    surveys = split_by_survey(
        context,
        df_data,
    )

    intersurvey = (
        analyze_intersurvey_correlations(

            context,

            surveys,
        )
    )

    corr_df = (
        intersurvey.correlation_matrix
    )

    overlap_df = (
        intersurvey.overlap_matrix
    )

    # ------------------------------------------------------------------
    # Observed random catalog
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("OBSERVED RANDOM CATALOG")
    print("==================================================")

    rng_obs = np.random.default_rng(
        config.random_seed
    )

    df_rand_obs = (
        generate_mixture_catalog_improved(

            context=context,

            n_observed=(
                len(df_data)
                * config.n_rand_factor
            ),

            sf_set=selection_functions,

            jitter_pixels=True,

            use_poisson=False,

            rng=rng_obs,
        )
    )

    print(
        f"Observed random catalog size = "
        f"{len(df_rand_obs)}"
    )

    # ------------------------------------------------------------------
    # Observed angular statistics
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("OBSERVED ANGULAR STATISTICS")
    print("==================================================")

    theta, w_obs = compute_2pacf(

        context=context,

        df_data=df_data,

        df_rand=df_rand_obs,
    )

    theta = np.asarray(
        theta,
        dtype=float,
    )

    w_obs = np.asarray(
        w_obs,
        dtype=float,
    )

    w_obs = np.nan_to_num(
        w_obs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    abs_obs = np.asarray(
        get_absolute_sum(

            context=context,

            theta=theta,

            w=w_obs,
        ),
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Jackknife
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("JACKKNIFE UNCERTAINTIES")
    print("==================================================")

    jackknife = run_jackknife_errors(

        context=context,

        df_data=df_data,

        sf_set=selection_functions,

        nside_jackknife=(
            config.nside_jackknife
        ),

        min_regions=(
            config.min_jackknife_regions
        ),
    )

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("BOOTSTRAP UNCERTAINTIES")
    print("==================================================")

    bootstrap = run_bootstrap_errors(

        context=context,

        df_data=df_data,

        sf_set=selection_functions,

        n_bootstrap=(
            config.n_bootstrap
        ),

        n_jobs=config.n_jobs,
    )

    # ------------------------------------------------------------------
    # H0 ensemble
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("H0 ENSEMBLE")
    print("==================================================")

    mock_ensemble = (
        run_ensemble_mocks(

            context=context,

            df_data=df_data,

            sf_set=(
                selection_functions
                if config.use_sel_func
                else None
            ),

            n_ensemble=(
                config.n_ensemble
            ),

            n_mocks_per=(
                config.n_mocks_per_ensemble
            ),

            perturbation_scale=(
                config.perturbation_scale
            ),

            n_rand_factor=(
                config.n_rand_factor
            ),

            n_jobs=config.n_jobs,
        )
    )

    all_w_h0 = np.asarray(
        mock_ensemble.w_theta,
        dtype=float,
    )

    all_abs_h0 = np.asarray(
        mock_ensemble.abs_statistics,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Remove invalid mocks
    # ------------------------------------------------------------------

    valid_h0 = (

        np.all(
            np.isfinite(all_w_h0),
            axis=1,
        )

        &

        np.all(
            np.isfinite(all_abs_h0),
            axis=1,
        )
    )

    if not np.all(valid_h0):

        all_w_h0 = all_w_h0[
            valid_h0
        ]

        all_abs_h0 = all_abs_h0[
            valid_h0
        ]

    # ------------------------------------------------------------------
    # Statistical inference
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("STATISTICAL INFERENCE")
    print("==================================================")

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

    print("\n==================================================")
    print("MAIN VISUALIZATIONS")
    print("==================================================")

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
    # Chi-square diagnostics
    # ------------------------------------------------------------------

    (
        chi2_bin_diagnostics,
        svd_mode_contributions,
    ) = compute_chi2_diagnostic_tables(

        context=context,

        theta=theta,

        w_obs=w_obs,

        all_w_h0=all_w_h0,

        eigenvalue_cut=(
            config.svd_eigenvalue_cut
        ),
    )

    # ------------------------------------------------------------------
    # Results container
    # ------------------------------------------------------------------

    results = {

        "context":
            context,

        "config":
            config,

        "df_data":
            df_data,

        "theta":
            theta,

        "w_obs":
            w_obs,

        "abs_obs":
            abs_obs,

        "selection_functions":
            selection_functions,

        "sf_validation":
            sf_validation,

        "survey_map_summary":
            survey_map_summary,

        "corr_df":
            corr_df,

        "overlap_df":
            overlap_df,

        "jackknife":
            jackknife,

        "bootstrap":
            bootstrap,

        "all_w_h0":
            all_w_h0,

        "all_abs_h0":
            all_abs_h0,

        "stats":
            stats,

        "chi2_bin_diagnostics":
            chi2_bin_diagnostics,

        "svd_mode_contributions":
            svd_mode_contributions,

        "covariance_matrix":
            getattr(
                context,
                "last_covariance_matrix",
                None,
            ),
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    if save_outputs:

        print(
            "\n--- Saving outputs ---"
        )

        results[
            "saved_tables"
        ] = save_tables(

            context=context,

            results=results,

            prefix=config.run_tag,
        )

    print("\n==================================================")
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("==================================================")

    return results