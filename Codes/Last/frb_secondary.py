"""FRB isotropy secondary analysis pipeline.

This module loads the FRB catalog "Random_SkyPosition.csv", that includes real
data as well as simulated FRB observations. Since CHIME is the dominant survey,
contributing with ~93% of the total observations, and essentially populating
the north pole, we have buffed up the south pole with isotropic randoms in order
to mitigate the extreme anisotropy of the raw catalog and enable more realistic
isotropic null tests.
"""


# ==============================================================================
# Imports and environment setup
# ==============================================================================

from __future__ import annotations

import os
from pathlib import Path

# Thread control for HPC parallel execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Standard library
from dataclasses import dataclass
import warnings
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.covariance import LedoitWolf
from scipy.stats import anderson_ksamp
from scipy.stats import chi2 as chi2_dist
from scipy.stats import ks_2samp
from scipy.stats import norm
from joblib import Parallel, delayed
import treecorr
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

# Warning suppression
warnings.filterwarnings(
    "ignore",
    message='.*"verbose" was deprecated.*',
    category=Warning,
)


# ==============================================================================
# Global analysis configuration
# ==============================================================================

# Parallel processing configuration
N_JOBS = 100


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration flags that define one analysis run."""

    use_gal_mask: bool = True
    use_sel_func: bool = True
    run_sensitivity: bool = False

    def __post_init__(self) -> None:
        if self.run_sensitivity and not self.use_sel_func:
            raise ValueError(
                "Sensitivity analysis requires use_sel_func=True."
            )


# Default run configuration
DEFAULT_CONFIG = AnalysisConfig()

# Active run configuration
USE_GAL_MASK = DEFAULT_CONFIG.use_gal_mask
USE_SEL_FUNC = DEFAULT_CONFIG.use_sel_func
RUN_SENSITIVITY = DEFAULT_CONFIG.run_sensitivity

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_ROOT = BASE_DIR / "outputs"

BASE_PATH = (
    "/home/brunowesley/projetos/"
    "FRB-isotropy-tests/FRB_catalogs/"
)

ALL_FRB_PATH = os.path.join(
    BASE_PATH,
    "Random_SkyPosition.csv",
)

# Observational sky mask
GAL_CUT = 20.0  # deg

# Angular configuration
MIN_SEP = 0.1
MAX_SEP = 180.0
BIN_SIZE = 10.0
N_BINS = int(
    (MAX_SEP - MIN_SEP) / BIN_SIZE
)

# Absolute anisotropy estimator
COARSE_BINS = np.arange(0, 181, 20)
COARSE_CENTERS = 0.5 * (
    COARSE_BINS[:-1]
    + COARSE_BINS[1:]
)

# Selection-function modeling
NSIDE_SF = 64             # HEALPix resolution of the SF maps
SMOOTH_SIGMA = 3.0        # Gaussian smoothing scale [deg]
PERTURBATION_SCALE = 1.0  # SF perturbation amplitude

# H0 mock generation
N_RAND_FACTOR = 20
N_ENSEMBLE = 20
N_MOCKS_PER_ENSEMBLE = 50
N_MOCKS = (
    N_ENSEMBLE
    * N_MOCKS_PER_ENSEMBLE
)

# Covariance regularization
SVD_EIGENVALUE_CUT = 1e-2  # Relative eigenvalue threshold

# Jackknife and bootstrap
NSIDE_JACKKNIFE = 4
MIN_JACKKNIFE_REGIONS = 20
N_BOOTSTRAP = 500

# Intersurvey overlap analysis
OVERLAP_RADIUS_DEG = 5.0
OVERLAP_NSIDE = 32

# Sensitivity analysis
NSIDE_SF_RANGE = [32, 64, 128]
SMOOTH_SIGMA_RANGE = [3.0, 5.0, 8.0, 10.0]
N_SENSITIVITY_EMPIRICAL_MOCKS = 1000

def build_run_tag(config: AnalysisConfig) -> str:
    """Build a stable output tag for one analysis configuration."""

    mask_tag = "mask" if config.use_gal_mask else "nomask"
    sf_tag = "sf" if config.use_sel_func else "nosf"
    sens_tag = "sens" if config.run_sensitivity else "nosens"

    return (
        f"{mask_tag}_"
        f"{sf_tag}_"
        f"{sens_tag}_"
        f"bin{int(BIN_SIZE)}"
    )


def apply_analysis_config(
    config: AnalysisConfig,
) -> AnalysisConfig:
    """Activate one run configuration for the module-level pipeline."""

    global USE_GAL_MASK
    global USE_SEL_FUNC
    global RUN_SENSITIVITY
    global MASK_TAG
    global SF_TAG
    global SENS_TAG
    global RUN_TAG
    global SUMMARY_CSV
    global DIAGNOSTICS_XLSX
    global REPORT_MD
    global RESULTS_PKL
    global RUN_OUTPUT_DIR
    global FIGURES_DIR
    global TABLES_DIR
    global REPORT_DIR

    USE_GAL_MASK = config.use_gal_mask
    USE_SEL_FUNC = config.use_sel_func
    RUN_SENSITIVITY = config.run_sensitivity

    MASK_TAG = "mask" if USE_GAL_MASK else "nomask"
    SF_TAG = "sf" if USE_SEL_FUNC else "nosf"
    SENS_TAG = "sens" if RUN_SENSITIVITY else "nosens"

    RUN_TAG = build_run_tag(config)

    RUN_OUTPUT_DIR = OUTPUTS_ROOT / RUN_TAG
    FIGURES_DIR = RUN_OUTPUT_DIR / "figures"
    TABLES_DIR = RUN_OUTPUT_DIR / "tables"
    REPORT_DIR = RUN_OUTPUT_DIR / "report"

    for directory in (
        OUTPUTS_ROOT,
        FIGURES_DIR,
        TABLES_DIR,
        REPORT_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    SUMMARY_CSV = TABLES_DIR / "summary.csv"
    DIAGNOSTICS_XLSX = TABLES_DIR / "diagnostics.xlsx"
    REPORT_MD = REPORT_DIR / "summary.md"
    RESULTS_PKL = RUN_OUTPUT_DIR / "results.pkl"

    return config


# Output tagging system
apply_analysis_config(DEFAULT_CONFIG)


# ==============================================================================

# Pipeline data structures
# ==============================================================================


@dataclass
class TestStatistics:

    chi2: float
    chi2_red: float

    p_chi2: float
    p_empirical: float
    p_empirical_floor: float

    sigma_equiv: float

    chi2_svd: float
    chi2_svd_red: float

    p_chi2_svd: float
    p_svd_empirical: float
    p_svd_empirical_floor: float

    sigma_svd_equiv: float

    global_tension: float

    ks_w_stat: float
    ks_w_pvalue: float
    ks_w_empirical_p: float

    ad_w_stat: float
    ad_w_pvalue: float
    ad_w_empirical_p: float

    ks_abs_stat: float
    ks_abs_pvalue: float
    ks_abs_empirical_p: float

    ad_abs_stat: float
    ad_abs_pvalue: float
    ad_abs_empirical_p: float

    abs_observed_stat: float
    abs_empirical_p: float

    hartlap_factor: float

    covariance_rank: int
    covariance_condition: float
    n_eff: float

    svd_modes_kept: int
    svd_eigenvalue_cut: float
    svd_condition: float

    n_mocks: int
    n_bins: int


@dataclass
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


@dataclass
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
# Catalog loading and sky geometry
# ==============================================================================


def load_catalog(path: str = ALL_FRB_PATH) -> pd.DataFrame:
    """
    Load FRB catalog and standardize column names.
    """

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    if "Reporting_Group_s" not in df.columns:
        df["Reporting_Group_s"] = "FULLSKY"

    # Detect if RA/DEC are provided in radians (common for this catalog)
    # and convert them to degrees to keep the pipeline consistent.
    try:
        ra_vals = df["RA"].astype(float)
        dec_vals = df["DEC"].astype(float)

        if (
            np.nanmax(np.abs(ra_vals)) <= 2 * np.pi
            and np.nanmax(np.abs(dec_vals)) <= (np.pi / 2)
        ):
            df["RA"] = np.degrees(ra_vals)
            df["DEC"] = np.degrees(dec_vals)
            print("\n--- Converted RA/DEC from radians to degrees ---")
    except Exception:
        # If conversion/check fails, leave values as-is and proceed.
        pass

    df = (df[["RA",
              "DEC",
              "Reporting_Group_s",
             ]
            ]
        .dropna()
        .reset_index(drop=True)
    )

    print("\n--- Catalog loaded ---")
    print(f"Objects: {len(df)}")

    return df


def apply_mask(
    df: pd.DataFrame,
    gal_cut: float = GAL_CUT,
    use_mask: bool | None = None,
) -> pd.DataFrame:
    """
    Apply Galactic latitude mask to observed catalog.

    If USE_GAL_MASK=False:
        return original catalog unchanged.
    """

    if use_mask is None:
        use_mask = USE_GAL_MASK

    if not use_mask:

        print("\n--- Galactic mask disabled ---")

        return df.reset_index(drop=True)

    coords = SkyCoord(
        ra=df["RA"].values * u.degree,
        dec=df["DEC"].values * u.degree,
        frame="icrs",
    )

    gal_b = coords.galactic.b.degree

    mask = np.abs(gal_b) > gal_cut

    df_masked = (
        df[mask]
        .reset_index(drop=True)
    )

    print("\n--- Galactic mask applied ---")
    print(f"gal_cut = ±{gal_cut:.1f} deg")
    print(f"Remaining objects: {len(df_masked)}")
    print(f"Removed objects: {len(df) - len(df_masked)}")

    return df_masked


def healpix_galactic_mask(
    nside: int,
    gal_cut: float = GAL_CUT,
    use_mask: bool | None = None,
) -> np.ndarray:
    """
    Build Galactic mask in HEALPix space.

    Returns
    -------
    np.ndarray
        Boolean mask:
            True  -> usable pixel
            False -> masked pixel
    """

    npix = hp.nside2npix(nside)

    if use_mask is None:
        use_mask = USE_GAL_MASK

    # No Galactic masking
    if not use_mask:
        return np.ones(npix, dtype=bool)

    # Standard Galactic mask
    theta, phi = hp.pix2ang(
        nside,
        np.arange(npix),
    )

    coords = SkyCoord(
        ra=np.degrees(phi) * u.degree,
        dec=(90.0 - np.degrees(theta)) * u.degree,
        frame="icrs",
    )

    gal_b = coords.galactic.b.degree

    return np.abs(gal_b) > gal_cut


def rotate_healpix_map_to_galactic(
    hmap: np.ndarray,
    coord_in: str = "C",
    coord_out: str = "G",
) -> np.ndarray:
    """
    Rotate a HEALPix map between coordinate systems.

    Parameters
    ----------
    coord_in : str
        Input coordinate system:
            "C" = Equatorial / ICRS
            "G" = Galactic

    coord_out : str
        Output coordinate system.

    Returns
    -------
    np.ndarray
        Rotated HEALPix map.
    """

    nside = hp.get_nside(hmap)

    rotator = hp.Rotator(
        coord=[coord_out, coord_in]
    )

    theta, phi = hp.pix2ang(
        nside,
        np.arange(hp.nside2npix(nside)),
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

    return hmap[pix_rot]


# ==============================================================================
# Survey modeling and selection functions
# ==============================================================================


def split_by_survey(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Partition catalog into observational surveys.

    Behavior depends on USE_SEL_FUNC.

    USE_SEL_FUNC = True
        -> split by survey

    USE_SEL_FUNC = False
        -> treat entire catalog as one population
    """

    # No selection function
    if not USE_SEL_FUNC:
        print("\n--- Survey partitioning disabled ---")
        print(
            "USE_SEL_FUNC=False "
            "→ treating full catalog as one population"
        )
        return {"FULLSKY": (df[["RA", "DEC"]].copy().reset_index(drop=True))}

    # With selection function
    df = df.copy()
    df["Reporting_Group_s"] = (
        df["Reporting_Group_s"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    surveys: dict[str, list[tuple[float, float]]] = {}

    for row in df.itertuples(index=False):

        groups = [
            g.strip()
            for g in str(row.Reporting_Group_s).split(",")
        ]

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

    survey_dict = {
        name: pd.DataFrame(
            values,
            columns=["RA", "DEC"],
        )
        for name, values in surveys.items()
    }

    # Diagnostics
    counts = {
        k: len(v)
        for k, v in survey_dict.items()
    }

    sorted_counts = sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n--- Survey partitioning ---")
    print(f"Unique surveys: {len(survey_dict)}")

    print("\nSurvey populations:")

    for name, count in sorted_counts:

        frac = 100.0 * count / len(df)

        print(
            f"{name:>15s} | "
            f"N={count:4d} | "
            f"{frac:6.2f}%"
        )

    return survey_dict


def build_selection_function_improved(
    subdf: pd.DataFrame,
    nside: int = NSIDE_SF,
    smooth_sigma: float = SMOOTH_SIGMA,
    gal_cut: float = GAL_CUT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build probabilistic sky selection function for one survey.

    Returns
    -------
    sf : np.ndarray
        Normalized sky probability distribution.

    sf_cov_diag : np.ndarray
        Approximate diagonal covariance.
    """

    # Safety checks
    if len(subdf) == 0:

        raise ValueError(
            "Cannot build selection function from empty survey."
        )

    npix = hp.nside2npix(nside)

    # Pixelized survey map
    theta = np.radians(
        90.0 - subdf["DEC"].values
    )

    phi = np.radians(
        subdf["RA"].values
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

    # Small numerical floor
    counts += 1e-12

    total_counts = counts.sum()

    if total_counts <= 0:

        raise ValueError(
            "Selection function has zero counts."
        )

    # Raw selection function
    sf = counts / total_counts

    # Poisson uncertainty
    sf_err_counts = np.sqrt(counts)

    sf_cov_diag = (
        sf_err_counts / total_counts
    ) ** 2

    if smooth_sigma > 0:

        sf = hp.smoothing(
            sf,
            sigma=np.radians(smooth_sigma),
            verbose=False,
        )

        sf = np.clip(
            sf,
            0.0,
            None,
        )

    # Galactic mask
    gal_mask = healpix_galactic_mask(
        nside,
        gal_cut=gal_cut,
        use_mask=USE_GAL_MASK,
    )

    sf[~gal_mask] = 0.0

    # Final normalization
    sf_sum = sf.sum()

    if sf_sum <= 0:

        raise ValueError(
            "Selection function vanished after masking."
        )

    sf /= sf_sum

    return sf, sf_cov_diag


def build_survey_selection_functions_improved(
    df: pd.DataFrame,
    nside: int = NSIDE_SF,
    smooth_sigma: float = SMOOTH_SIGMA,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
    int,
]:
    """
    Build survey-based sky probability maps.

    Behavior depends on USE_SEL_FUNC.

    USE_SEL_FUNC = True
        -> one selection function per survey

    USE_SEL_FUNC = False
        -> one global FULLSKY probability map
    """

    surveys = split_by_survey(df)

    total_memberships = sum(
        len(subdf)
        for subdf in surveys.values()
    )

    sf_dict: dict[str, np.ndarray] = {}

    sf_cov_diag_dict: dict[str, np.ndarray] = {}

    survey_weights: dict[str, float] = {}

    print("\n--- Building selection functions ---")
    print(f"Unique surveys: {len(surveys)}")

    for name, subdf in surveys.items():

        # Adaptive smoothing for poorly sampled surveys
        effective_sigma = smooth_sigma

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

        # Build survey SF
        sf, sf_cov_diag = (
            build_selection_function_improved(
                subdf,
                nside=nside,
                smooth_sigma=effective_sigma,
            )
        )

        sf_dict[name] = sf

        sf_cov_diag_dict[name] = sf_cov_diag

        survey_weights[name] = (
            len(subdf)
            / total_memberships
        )

        print(
            f"{name:>15s} | "
            f"N={len(subdf):4d} | "
            f"sigma={effective_sigma:4.1f} deg | "
            f"weight={survey_weights[name]:.4f}"
        )

    print("\nSelection functions successfully built.")

    return (
        sf_dict,
        sf_cov_diag_dict,
        survey_weights,
        nside,
    )


def generate_sf_variant(
    sf_dict: dict[str, np.ndarray],
    sf_cov_diag_dict: dict[str, np.ndarray],
    perturbation_scale: float = PERTURBATION_SCALE,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate perturbed realizations of survey selection functions.

    Notes
    -----
    This function is only physically relevant when:
        USE_SEL_FUNC = True

    When USE_SEL_FUNC = False:
        the input FULLSKY map is returned unchanged.
    """

    # Pure isotropic mode
    if not USE_SEL_FUNC:

        return {
            k: v.copy()
            for k, v in sf_dict.items()
        }

    # Selection-function perturbation mode
    rng = np.random.default_rng(seed)

    sf_dict_perturbed: dict[str, np.ndarray] = {}

    for survey_name, sf in sf_dict.items():

        sigma = np.sqrt(
            sf_cov_diag_dict[survey_name]
        )

        noise = rng.normal(
            0.0,
            perturbation_scale * sigma,
        )

        sf_pert = np.clip(
            sf + noise,
            0.0,
            None,
        )

        # Numerical safeguard
        if sf_pert.sum() <= 0:

            sf_pert = sf.copy()

        else:

            sf_pert /= sf_pert.sum()

        sf_dict_perturbed[survey_name] = sf_pert

    return sf_dict_perturbed


def analyze_intersurvey_correlations(
    df_data: pd.DataFrame,
    radius_deg: float = OVERLAP_RADIUS_DEG,
    nside: int = OVERLAP_NSIDE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    USE_SEL_FUNC = True.
    """

    # Disabled case
    if not USE_SEL_FUNC:

        print(
            "\n--- Intersurvey analysis skipped ---"
        )

        print(
            "USE_SEL_FUNC=False "
            "→ no survey partition exists."
        )

        empty = pd.DataFrame()

        return empty, empty

    # Survey partitioning
    surveys = split_by_survey(df_data)

    survey_names = list(
        surveys.keys()
    )

    n_surveys = len(
        survey_names
    )

    print("\n==================================================")
    print("INTERSURVEY CORRELATION ANALYSIS")
    print("==================================================")

    print(
        f"\nUnique surveys: {n_surveys}"
    )

    print(
        f"HEALPix nside: {nside}"
    )

    print(
        f"Overlap radius: {radius_deg:.1f} deg"
    )

    # Storage
    corr_matrix = np.zeros(
        (n_surveys, n_surveys),
        dtype=float,
    )

    overlap_fractions = np.zeros(
        (n_surveys, n_surveys),
        dtype=float,
    )

    npix = hp.nside2npix(nside)

    # SkyCoord cache
    coords_by_survey = {

        name: SkyCoord(
            ra=subdf["RA"].values * u.degree,
            dec=subdf["DEC"].values * u.degree,
            frame="icrs",
        )

        for name, subdf in surveys.items()
    }

    # HEALPix map cache
    histograms = {}

    for name, subdf in surveys.items():

        theta = np.radians(
            90.0 - subdf["DEC"].values
        )

        phi = np.radians(
            subdf["RA"].values
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

        # Normalize map
        if hist.sum() > 0:

            hist /= hist.sum()

        histograms[name] = hist

    # Pairwise analysis
    for i, survey1 in enumerate(survey_names):

        for j, survey2 in enumerate(survey_names):

            # Diagonal
            if i == j:

                corr_matrix[i, j] = 1.0

                overlap_fractions[i, j] = 1.0

                continue

            # Pearson sky correlation
            hist1 = histograms[survey1]
            hist2 = histograms[survey2]

            corr = np.corrcoef(
                hist1,
                hist2,
            )[0, 1]

            if not np.isfinite(corr):

                corr = 0.0

            corr_matrix[i, j] = corr

            # Nearest-neighbor overlap
            coords1 = coords_by_survey[survey1]

            coords2 = coords_by_survey[survey2]

            _, sep2d, _ = (
                coords1.match_to_catalog_sky(
                    coords2
                )
            )

            overlap = np.mean(
                sep2d < radius_deg * u.degree
            )

            overlap_fractions[i, j] = float(
                overlap
            )

    # DataFrames
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

    # Diagnostics
    offdiag = ~np.eye(
        n_surveys,
        dtype=bool,
    )

    mean_corr = np.mean(
        corr_matrix[offdiag]
    )

    max_corr = np.max(
        corr_matrix[offdiag]
    )

    mean_overlap = np.mean(
        overlap_fractions[offdiag]
    )

    max_overlap = np.max(
        overlap_fractions[offdiag]
    )

    print("\n--- Spatial correlation summary ---")

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
        f"{100*mean_overlap:.2f}%"
    )

    print(
        f"Max overlap fraction: "
        f"{100*max_overlap:.2f}%"
    )

    print(
        "\nHigh overlap/correlation pairs "
        "may indicate non-independent "
        "observational footprints."
    )

    return (corr_df, overlap_df)


class SelectionFunctionValidator:
    """
    Numerical and visual validation of survey selection functions.

    This validator is only physically meaningful when:

        USE_SEL_FUNC = True
    """

    def __init__(
        self,
        sf_dict: dict[str, np.ndarray],
        survey_weights: dict[str, float],
        nside: int,
    ) -> None:

        self.sf_dict = sf_dict
        self.survey_weights = survey_weights
        self.nside = nside

    # Numerical validation
    def check_coverage(
        self,
    ) -> pd.DataFrame:

        # Disabled case
        if not USE_SEL_FUNC:
            print(
                "\n--- Selection-function validation skipped ---"
            )
            print(
                "USE_SEL_FUNC=False "
                "→ no instrumental selection model."
            )
            return pd.DataFrame()

        print("\n==================================================")
        print("SELECTION FUNCTION VALIDATION")
        print("==================================================")

        rows = []

        for survey_name, sf in self.sf_dict.items():
            sf = np.asarray(
                sf,
                dtype=float,
            )

            # Peak position
            max_pix = int(
                np.argmax(sf)
            )

            theta_max, phi_max = hp.pix2ang(
                self.nside,
                max_pix,
            )

            dec_max = (
                90.0
                - np.degrees(theta_max)
            )

            ra_max = np.degrees(
                phi_max
            )

            # Coverage
            threshold = (
                np.max(sf) * 1e-3
            )

            active = sf > threshold

            coverage = (
                np.mean(active)
                * 100.0
            )

            # Entropy
            positive = sf > 0

            entropy = -np.sum(
                sf[positive]
                * np.log2(sf[positive])
            )

            # Effective sky area
            npix_active = int(
                np.sum(active)
            )

            sky_fraction = (
                npix_active
                / len(sf)
            )

            # Row
            rows.append(
                {

                    "run_tag":
                        RUN_TAG,

                    "survey":
                        survey_name,

                    "weight":
                        self.survey_weights[survey_name],

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

            # Diagnostics
            print(f"\n{survey_name}:")

            print(
                f"  Weight: "
                f"{self.survey_weights[survey_name]:.2%}"
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

            # Simple physical sanity checks
            if (
                "CHIME" in survey_name.upper()
                and dec_max < 20.0
            ):

                print(
                    "  WARNING: "
                    "CHIME peak unexpectedly far south."
                )

            if (
                "PARKES" in survey_name.upper()
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

    # Visual validation
    def plot_sf(
        self,
        save_prefix: str = "SF_validation",
        max_surveys: int = 12,
        cmap: str = "viridis",
        log_scale: bool = False,
    ) -> None:

        # Disabled case
        if not USE_SEL_FUNC:
            print(
                "\n--- SF plots skipped ---"
            )
            print(
                "USE_SEL_FUNC=False "
                "→ no selection functions to visualize."
            )

            return

        # Survey ordering
        ordered = sorted(
            self.sf_dict.keys(),
            key=lambda name: (
                self.survey_weights[name]
            ),
            reverse=True,
        )[:max_surveys]

        ncols = min(
            4,
            len(ordered),
        )

        nrows = int(
            np.ceil(
                len(ordered) / ncols
            )
        )

        # EQUATORIAL MAPS
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

            sf_plot = self.sf_dict[name]

            if log_scale:

                sf_plot = np.log10(
                    sf_plot + 1e-12
                )

            hp.mollview(
                sf_plot,
                title=(
                    f"{name}\n"
                    f"(w={self.survey_weights[name]:.1%})"
                ),
                sub=(
                    nrows,
                    ncols,
                    i,
                ),
                cbar=True,
                cmap=cmap,
            )

        equatorial_path = Path(
            f"{save_prefix}_equatorial.png"
        )

        plt.savefig(
            equatorial_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()

        # GALACTIC MAPS
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

            sf_gal = rotate_healpix_map_to_galactic(
                self.sf_dict[name],
                coord_in="C",
                coord_out="G",
            )

            if log_scale:

                sf_gal = np.log10(
                    sf_gal + 1e-12
                )

            hp.mollview(
                sf_gal,
                title=(
                    f"{name}\n"
                    f"(w={self.survey_weights[name]:.1%})"
                ),
                sub=(
                    nrows,
                    ncols,
                    i,
                ),
                cbar=True,
                cmap=cmap,
            )

        galactic_path = Path(
            f"{save_prefix}_galactic.png"
        )

        plt.savefig(
            galactic_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()

        print("\nSelection-function maps saved:")
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
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomize positions inside HEALPix pixels.

    This suppresses artificial angular quantization
    produced by finite HEALPix resolution.
    """

    pix_radius_deg = np.degrees(
        hp.max_pixrad(nside)
    )

    # Declination jitter
    dec_jitter = rng.uniform(
        -pix_radius_deg,
        pix_radius_deg,
        size=len(dec),
    )

    # Right ascension jitter
    cos_dec = np.clip(
        np.cos(np.radians(dec)),
        1e-3,
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

    # Final coordinates
    ra_out = (
        ra + ra_jitter
    ) % 360.0

    dec_out = np.clip(
        dec + dec_jitter,
        -89.999,
        89.999,
    )

    return ra_out, dec_out


def generate_mixture_catalog_improved(
    n_observed: int,
    sf_dict: dict[str, np.ndarray] | None,
    weights: dict[str, float] | None,
    nside: int,
    seed: int | None = None,
    jitter_pixels: bool = True,
    use_poisson: bool = False,
) -> pd.DataFrame:

    """
    Generate isotropic mock catalogs.

    Two physical modes are supported.

    ----------------------------------------------------------
    USE_SEL_FUNC = True
        Isotropic sky filtered through survey selection
        functions.

    USE_SEL_FUNC = False
        Perfect isotropic sky realization.
    ----------------------------------------------------------
    """

    rng = np.random.default_rng(seed)

    # Catalog size
    if use_poisson:
        n = rng.poisson(n_observed)
    else:
        n = int(n_observed)
    if n <= 0:

        return pd.DataFrame(
            columns=["RA", "DEC"]
        )

    # If no selection-function dictionary was provided, treat as
    # pure isotropic mode regardless of the module-level flag.
    effective_use_sel = bool(USE_SEL_FUNC and (sf_dict is not None))

    # Perfect isotropic sky
    if not effective_use_sel:

        ra_all = []
        dec_all = []

        while len(ra_all) < n:

            n_remaining = n - len(ra_all)

            # Uniform isotropic sky
            ra = 360.0 * rng.random(n_remaining)
            dec = np.degrees(
                np.arcsin(2.0 * rng.random(n_remaining) - 1.0)
            )

            # Optional Galactic mask
            if USE_GAL_MASK:

                coords = SkyCoord(
                    ra=ra * u.degree,
                    dec=dec * u.degree,
                    frame="icrs",
                )

                gal_b = coords.galactic.b.degree

                keep = (
                    np.abs(gal_b)
                    > GAL_CUT
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
                "RA": np.array(ra_all[:n]),
                "DEC": np.array(dec_all[:n]),
            }
        )

    # Selection-function-based sky
    surveys = np.array(
        list(sf_dict.keys())
    )

    survey_probs = np.array(
        [weights[s] for s in surveys],
        dtype=float,
    )

    survey_probs /= survey_probs.sum()

    chosen_surveys = rng.choice(
        surveys,
        size=n,
        p=survey_probs,
    )

    ra = np.empty(n)
    dec = np.empty(n)

    # Generate positions from survey SFs
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
            sf_dict[survey],
            dtype=float,
        )

        sf = np.clip(
            sf,
            0.0,
            None,
        )

        sf_sum = sf.sum()

        if sf_sum <= 0:

            raise ValueError(
                f"Selection function vanished for survey {survey}"
            )

        sf /= sf_sum

        pix = rng.choice(
            len(sf),
            size=n_survey,
            p=sf,
        )

        theta, phi = hp.pix2ang(
            nside,
            pix,
        )

        ra_survey = np.degrees(phi)

        dec_survey = (
            90.0
            - np.degrees(theta)
        )

        # Sub-pixel randomization
        if jitter_pixels:

            ra_survey, dec_survey = (
                _jitter_pixel_centers(
                    ra_survey,
                    dec_survey,
                    nside,
                    rng,
                )
            )

        ra[mask] = ra_survey
        dec[mask] = dec_survey

    return pd.DataFrame(
        {
            "RA": ra,
            "DEC": dec,
        }
    )


def _run_one_independent_mock(
    sf_variant: dict[str, np.ndarray],
    seed_data: int,
    seed_rand: int,
    n_data: int,
    survey_weights: dict[str, float],
    nside: int,
    n_rand_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one statistically independent mock realization
    from a FIXED selection-function variant.

    IMPORTANT:
    - SF is shared inside one ensemble
    - catalogs are independent
    """

    # ----------------------------------------------------------
    # Data catalog
    # ----------------------------------------------------------

    d_iso = generate_mixture_catalog_improved(
        n_observed=n_data,
        sf_dict=sf_variant,
        weights=survey_weights,
        nside=nside,
        seed=seed_data,
        use_poisson=False,
    )

    # ----------------------------------------------------------
    # Random catalog
    # ----------------------------------------------------------

    r_iso = generate_mixture_catalog_improved(
        n_observed=n_data * n_rand_factor,
        sf_dict=sf_variant,
        weights=survey_weights,
        nside=nside,
        seed=seed_rand,
        use_poisson=False,
    )

    # ----------------------------------------------------------
    # Proper 2pACF
    # ----------------------------------------------------------

    theta_h0, w_h0 = compute_2pacf(
        d_iso,
        r_iso,
    )

    abs_h0 = get_absolute_sum(
        theta_h0,
        w_h0,
    )

    return w_h0, abs_h0


def run_ensemble_mocks(
    df_data: pd.DataFrame,
    sf_dict_nominal: dict[str, np.ndarray],
    sf_cov_diag_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    n_ensemble: int = N_ENSEMBLE,
    n_mocks_per: int = N_MOCKS_PER_ENSEMBLE,
    perturbation_scale: float = PERTURBATION_SCALE,
    n_rand_factor: int = N_RAND_FACTOR,
    n_jobs: int = N_JOBS,
) -> tuple[np.ndarray, np.ndarray]:

    """
    Generate H0 isotropic mock ensemble.

    Behavior depends on USE_SEL_FUNC.

    USE_SEL_FUNC = True
        -> hierarchical SF ensemble

    USE_SEL_FUNC = False
        -> pure isotropic realizations
    """

    tasks = []

    # ----------------------------------------------------------
    # Case 1:
    # With selection functions
    # ----------------------------------------------------------

    if USE_SEL_FUNC:

        print(
            "\n--- Generating H0 ensemble mocks "
            "(with selection functions) ---"
        )

        for i_ens in range(n_ensemble):

            # One perturbed SF realization
            sf_variant = generate_sf_variant(
                sf_dict_nominal,
                sf_cov_diag_dict,
                perturbation_scale=perturbation_scale,
                seed=100000 + i_ens,
            )

            # Independent mocks sharing same SF
            for i_mock in range(n_mocks_per):

                seed_data = (
                    200000
                    + i_ens * 1000
                    + i_mock
                )

                seed_rand = (
                    300000
                    + i_ens * 1000
                    + i_mock
                )

                tasks.append(
                    (
                        sf_variant,
                        seed_data,
                        seed_rand,
                    )
                )

    # ----------------------------------------------------------
    # Case 2:
    # Pure isotropic sky
    # ----------------------------------------------------------

    else:

        print(
            "\n--- Generating pure isotropic H0 mocks ---"
        )

        n_total = (
            n_ensemble
            * n_mocks_per
        )

        for i_mock in range(n_total):

            seed_data = 200000 + i_mock
            seed_rand = 300000 + i_mock

            tasks.append(
                (
                    None,
                    seed_data,
                    seed_rand,
                )
            )

    # Total number of mocks
    n_total = len(tasks)

    print(f"Total H0 mocks: {n_total}")

    # Parallel execution
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
            delayed(_run_one_independent_mock)(
                sf_variant,
                seed_data,
                seed_rand,
                len(df_data),
                survey_weights,
                nside,
                n_rand_factor,
            )
            for (
                sf_variant,
                seed_data,
                seed_rand,
            ) in tasks
        )

    # Stack results
    all_w_h0, all_abs_h0 = zip(*results)
    all_w_h0 = np.asarray(all_w_h0)
    all_abs_h0 = np.asarray(all_abs_h0)

    print(f"\nGenerated mocks:")
    print(f"  w(theta): {all_w_h0.shape}")
    print(f"  |<w>|: {all_abs_h0.shape}")

    return all_w_h0, all_abs_h0


# ==============================================================================
# Physical estimators
# ==============================================================================


def compute_2pacf(
    df_data: pd.DataFrame,
    df_rand: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the angular two-point correlation function.

    Uses:
        - linear angular bins
        - Landy-Szalay estimator
        - TreeCorr pair counting
    """

    # Build TreeCorr catalogs
    cat_data = treecorr.Catalog(
        ra=df_data["RA"],
        dec=df_data["DEC"],
        ra_units="deg",
        dec_units="deg",
    )

    cat_rand = treecorr.Catalog(
        ra=df_rand["RA"],
        dec=df_rand["DEC"],
        ra_units="deg",
        dec_units="deg",
    )

    # Pair-counting configuration
    corr = treecorr.NNCorrelation(
        min_sep=MIN_SEP,
        max_sep=MAX_SEP,
        nbins=N_BINS,
        sep_units="deg",
        metric="Arc",
        bin_type="Linear",
    )

    rr = treecorr.NNCorrelation(
        **corr.config
    )

    dr = treecorr.NNCorrelation(
        **corr.config
    )

    # Pair counts
    corr.process(cat_data)

    rr.process(cat_rand)

    dr.process(
        cat_data,
        cat_rand,
    )

    # Landy-Szalay estimator
    corr.calculateXi(
        rr=rr,
        dr=dr,
    )

    theta = corr.meanr

    w_theta = np.asarray(
        corr.xi,
        dtype=float,
    )

    # Numerical cleanup
    bad = ~np.isfinite(w_theta)

    if np.any(bad):

        print(
            f"WARNING: {np.sum(bad)} invalid "
            f"2pACF bins detected."
        )

        w_theta[bad] = 0.0

    return theta, w_theta


def get_absolute_sum(
    theta: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute coarse-grained absolute anisotropy amplitudes.

    The statistic averages |w(theta)| inside broad angular bins.
    """

    vals = []

    for i in range(
        len(COARSE_BINS) - 1
    ):

        mask = (
            (theta >= COARSE_BINS[i])
            &
            (theta < COARSE_BINS[i + 1])
        )

        if np.any(mask):

            vals.append(
                np.abs(
                    np.mean(w[mask])
                )
            )

        else:

            vals.append(0.0)

    return np.array(vals)


def absolute_global_stat(
    abs_values: np.ndarray,
) -> float:
    """
    Compute RMS global anisotropy amplitude.
    """

    abs_values = np.asarray(
        abs_values,
        dtype=float,
    )

    abs_values = abs_values[
        np.isfinite(abs_values)
    ]

    if len(abs_values) == 0:

        return 0.0

    return float(
        np.sqrt(
            np.mean(abs_values**2)
        )
    )


# ==============================================================================
# Statistical inference and diagnostics
# ==============================================================================


def sigma_equivalent_from_p(
    p_value: float,
) -> float:
    """
    Convert a p-value into Gaussian-equivalent significance.

    Returns
    -------
    sigma : float
        One-sided Gaussian significance.
    """

    p_clipped = np.clip(
        p_value,
        1e-300,
        1.0 - 1e-16,
    )

    return float(
        norm.isf(p_clipped)
    )


def compute_svd_regularized_chi2(
    delta: np.ndarray,
    mock_delta: np.ndarray,
    cov: np.ndarray,
    hartlap_factor: float,
    eigenvalue_cut: float = SVD_EIGENVALUE_CUT,
) -> dict[str, float]:
    """
    Compute SVD-regularized chi-square statistic.

    Strategy
    --------
    - diagonalize covariance matrix
    - discard noisy eigenmodes
    - compute chi-square in stable subspace

    This avoids catastrophic amplification of
    poorly constrained covariance modes.
    """

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Remove pathological eigenvalues
    finite = np.isfinite(eigvals)
    positive = eigvals > 0.0
    valid = finite & positive

    if not np.any(valid):
        raise ValueError(
            "Covariance matrix has no valid positive eigenvalues."
        )

    eigvals_valid = eigvals[valid]
    eigvecs_valid = eigvecs[:, valid]

    # Relative eigenvalue cut
    max_eig = float(
        eigvals_valid[0]
    )

    keep = (
        eigvals_valid
        >= eigenvalue_cut * max_eig
    )

    # Safety:
    # always keep at least one mode
    if not np.any(keep):
        keep[0] = True
    kept_eigvals = eigvals_valid[keep]
    kept_eigvecs = eigvecs_valid[:, keep]
    modes_kept = int(
        len(kept_eigvals)
    )

    # Project onto stable eigenbasis
    delta_modes = (
        delta @ kept_eigvecs
    )

    mock_modes = (
        mock_delta @ kept_eigvecs
    )

    # Chi-square statistic
    chi2_svd = float(
        hartlap_factor
        * np.sum(
            delta_modes**2
            / kept_eigvals
        )
    )

    chi2_svd_mocks = (
        hartlap_factor
        * np.sum(
            mock_modes**2
            / kept_eigvals,
            axis=1,
        )
    )

    # Empirical p-value
    n_extreme = int(
        np.sum(
            chi2_svd_mocks >= chi2_svd
        )
    )

    n_mocks = len(
        chi2_svd_mocks
    )

    p_empirical_floor = (
        1.0 / (n_mocks + 1)
    )

    p_empirical = float(
        (n_extreme + 1)
        / (n_mocks + 1)
    )

    # Analytic p-value
    p_analytic = float(
        chi2_dist.sf(
            chi2_svd,
            modes_kept,
        )
    )

    # Diagnostics
    svd_condition = float(
        kept_eigvals[0]
        / kept_eigvals[-1]
    )

    eig_fraction = float(
        np.sum(kept_eigvals)
        / np.sum(eigvals_valid)
    )

    print("\n--- SVD covariance regularization ---")

    print(
        f"Modes kept: "
        f"{modes_kept}/{len(eigvals_valid)}"
    )

    print(
        f"Eigenvalue threshold: "
        f"{eigenvalue_cut:.2e}"
    )

    print(
        f"Explained variance kept: "
        f"{100.0 * eig_fraction:.2f}%"
    )

    print(
        f"SVD condition number: "
        f"{svd_condition:.2e}"
    )

    # Return
    return {

        "chi2": chi2_svd,

        "chi2_red": (
            chi2_svd / modes_kept
        ),

        "p_chi2": p_analytic,

        "p_empirical": p_empirical,

        "p_empirical_floor": (
            p_empirical_floor
        ),

        "sigma_equiv": (
            sigma_equivalent_from_p(
                p_empirical
            )
        ),

        "n_extreme_mocks": float(
            n_extreme
        ),

        "modes_kept": float(
            modes_kept
        ),

        "eigenvalue_cut": float(
            eigenvalue_cut
        ),

        "condition": svd_condition,

        "explained_variance_fraction": (
            eig_fraction
        ),

        "largest_eigenvalue": float(
            kept_eigvals[0]
        ),

        "smallest_eigenvalue": float(
            kept_eigvals[-1]
        ),
    }


def _anderson_ksamp_stat_p(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
) -> tuple[float, float]:
    """
    Anderson-Darling two-sample test.

    Wrapper around scipy.stats.anderson_ksamp
    with numerical protection.
    """

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        try:

            result = anderson_ksamp(
                [sample_a, sample_b]
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

    return stat, pvalue


def _profile_nonparametric_stats(
    observed: np.ndarray,
    mocks: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """
    Nonparametric comparison between observed angular
    profiles and isotropic benchmark profiles.

    This follows the methodology adopted in:

    Andrade et al. (2019)
    "Revisiting the statistical isotropy of GRB sky distribution"

    using:
        - Kolmogorov-Smirnov test
        - Anderson-Darling test
    """

    # Observed vs benchmark
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

    # Empirical calibration against H0 mocks
    ks_mock_stats = []
    ad_mock_stats = []

    for mock in mocks:

        # KS
        ks_mock = ks_2samp(
            mock,
            reference,
            alternative="two-sided",
            mode="auto",
        ).statistic

        ks_mock_stats.append(
            ks_mock
        )

        # AD
        ad_mock, _ = (
            _anderson_ksamp_stat_p(
                mock,
                reference,
            )
        )

        ad_mock_stats.append(
            ad_mock
        )

    ks_mock_stats = np.asarray(
        ks_mock_stats,
        dtype=float,
    )

    ad_mock_stats = np.asarray(
        ad_mock_stats,
        dtype=float,
    )

    # Remove invalid realizations
    ks_mock_stats = ks_mock_stats[
        np.isfinite(ks_mock_stats)
    ]

    ad_mock_stats = ad_mock_stats[
        np.isfinite(ad_mock_stats)
    ]

    # Empirical p-values
    ks_n_extreme = int(
        np.sum(
            ks_mock_stats >= ks_stat
        )
    )

    ad_n_extreme = int(
        np.sum(
            ad_mock_stats >= ad_stat
        )
    )

    ks_empirical_p = (
        (ks_n_extreme + 1)
        / (len(ks_mock_stats) + 1)
    )

    ad_empirical_p = (
        (ad_n_extreme + 1)
        / (len(ad_mock_stats) + 1)
    )

    # Diagnostics
    print("\n--- KS / AD profile tests ---")
    print(f"KS statistic   = {ks_stat:.5f}")
    print(f"KS p-value     = {ks_pvalue:.5f}")
    print(f"KS empirical p = {ks_empirical_p:.5f}")

    print(f"AD statistic   = {ad_stat:.5f}")
    print(f"AD p-value     = {ad_pvalue:.5f}")
    print(f"AD empirical p = {ad_empirical_p:.5f}")

    # Return
    return {

        # KS
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "ks_empirical_p": float(
            ks_empirical_p
        ),
        "ks_empirical_floor": float(
            1.0 / (len(ks_mock_stats) + 1)
        ),
        "ks_n_extreme_mocks": float(
            ks_n_extreme
        ),

        # AD
        "ad_stat": ad_stat,
        "ad_pvalue": ad_pvalue,
        "ad_empirical_p": float(
            ad_empirical_p
        ),
        "ad_empirical_floor": float(
            1.0 / (len(ad_mock_stats) + 1)
        ),
        "ad_n_extreme_mocks": float(
            ad_n_extreme
        ),
    }


def compute_nonparametric_tests(
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
) -> dict[str, float]:
    """
    Nonparametric isotropy diagnostics.

    This follows the methodology commonly adopted in the
    GRB isotropy literature:

        - Kolmogorov-Smirnov test
        - Anderson-Darling test

    comparing:
        observed profile
            vs
        isotropic benchmark profile

    Notes
    -----
    These tests are complementary diagnostics and do NOT
    replace the covariance-aware chi-square inference.

    The empirical p-values are calibrated using the H0
    mock ensemble itself.
    """

    # Benchmark reference profiles
    w_reference = np.mean(
        all_w_h0,
        axis=0,
    )

    abs_reference = np.mean(
        all_abs_h0,
        axis=0,
    )

    # Remove invalid values
    w_obs = np.asarray(
        w_obs,
        dtype=float,
    )

    abs_obs = np.asarray(
        abs_obs,
        dtype=float,
    )

    valid_w = (
        np.isfinite(w_obs)
        & np.isfinite(w_reference)
    )

    valid_abs = (
        np.isfinite(abs_obs)
        & np.isfinite(abs_reference)
    )

    if np.sum(valid_w) < 2:

        raise ValueError(
            "Not enough valid bins for w(theta) "
            "nonparametric tests."
        )

    if np.sum(valid_abs) < 2:

        raise ValueError(
            "Not enough valid bins for absolute-sum "
            "nonparametric tests."
        )

    # Profile tests
    w_stats = _profile_nonparametric_stats(
        observed=w_obs[valid_w],

        mocks=all_w_h0[:, valid_w],

        reference=w_reference[valid_w],
    )

    abs_stats = _profile_nonparametric_stats(
        observed=abs_obs[valid_abs],

        mocks=all_abs_h0[:, valid_abs],

        reference=abs_reference[valid_abs],
    )

    # Diagnostics
    print("\n==================================================")
    print("NONPARAMETRIC ISOTROPY DIAGNOSTICS")
    print("==================================================")

    print("\n[w(theta)]")

    print(
        f"KS p-value          = "
        f"{w_stats['ks_pvalue']:.5f}"
    )

    print(
        f"KS empirical p      = "
        f"{w_stats['ks_empirical_p']:.5f}"
    )

    print(
        f"AD p-value          = "
        f"{w_stats['ad_pvalue']:.5f}"
    )

    print(
        f"AD empirical p      = "
        f"{w_stats['ad_empirical_p']:.5f}"
    )

    print("\n[Absolute sum]")

    print(
        f"KS p-value          = "
        f"{abs_stats['ks_pvalue']:.5f}"
    )

    print(
        f"KS empirical p      = "
        f"{abs_stats['ks_empirical_p']:.5f}"
    )

    print(
        f"AD p-value          = "
        f"{abs_stats['ad_pvalue']:.5f}"
    )

    print(
        f"AD empirical p      = "
        f"{abs_stats['ad_empirical_p']:.5f}"
    )

    # Return
    return {

        # w(theta)
        "ks_w_stat":
            w_stats["ks_stat"],

        "ks_w_pvalue":
            w_stats["ks_pvalue"],

        "ks_w_empirical_p":
            w_stats["ks_empirical_p"],

        "ks_w_empirical_floor":
            w_stats["ks_empirical_floor"],

        "ks_w_n_extreme_mocks":
            w_stats["ks_n_extreme_mocks"],

        "ad_w_stat":
            w_stats["ad_stat"],

        "ad_w_pvalue":
            w_stats["ad_pvalue"],

        "ad_w_empirical_p":
            w_stats["ad_empirical_p"],

        "ad_w_empirical_floor":
            w_stats["ad_empirical_floor"],

        "ad_w_n_extreme_mocks":
            w_stats["ad_n_extreme_mocks"],

        # Absolute-sum statistic
        "ks_abs_stat":
            abs_stats["ks_stat"],

        "ks_abs_pvalue":
            abs_stats["ks_pvalue"],

        "ks_abs_empirical_p":
            abs_stats["ks_empirical_p"],

        "ks_abs_empirical_floor":
            abs_stats["ks_empirical_floor"],

        "ks_abs_n_extreme_mocks":
            abs_stats["ks_n_extreme_mocks"],

        "ad_abs_stat":
            abs_stats["ad_stat"],

        "ad_abs_pvalue":
            abs_stats["ad_pvalue"],

        "ad_abs_empirical_p":
            abs_stats["ad_empirical_p"],

        "ad_abs_empirical_floor":
            abs_stats["ad_empirical_floor"],

        "ad_abs_n_extreme_mocks":
            abs_stats["ad_n_extreme_mocks"],
    }


def compute_statistics(
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
    label: str = (
        f"FRB isotropy "
        f"({RUN_TAG})"
    ),
) -> TestStatistics:

    """
    Perform the full covariance-aware isotropy analysis
    against the H0 ensemble.
    """

    print("\n==================================================")
    print("GLOBAL ISOTROPY INFERENCE")
    print("==================================================")

    print(f"RUN_TAG = {RUN_TAG}")

    print(
        f"USE_GAL_MASK = {USE_GAL_MASK}"
    )

    print(
        f"USE_SEL_FUNC = {USE_SEL_FUNC}"
    )

    # H0 mean
    h0_mean = np.mean(
        all_w_h0,
        axis=0,
    )

    print(
        f"\nMax |H0 mean| = "
        f"{np.max(np.abs(h0_mean)):.4e}"
    )

    # Shrinkage covariance (Ledoit-Wolf)
    lw = LedoitWolf()
    lw.fit(all_w_h0)
    cov = lw.covariance_

    global LAST_COVARIANCE_MATRIX
    global LAST_EIGENVALUES

    LAST_COVARIANCE_MATRIX = cov.copy()

    # Correlation diagnostics
    plot_covariance_correlation_matrix(
        cov,
        title="H0 Correlation Matrix",
        output="h0_correlation_matrix",
    )

    # Mock similarity diagnostics
    plot_mock_similarity(
        all_w_h0,
        n_compare=min(
            100,
            len(all_w_h0),
        ),
        output="h0_mock_similarity",
    )

    # Eigenvalue spectrum
    eigvals = np.linalg.eigvalsh(cov)
    eigvals_pos = eigvals[
        eigvals > 1e-15
    ]

    LAST_EIGENVALUES = np.sort(
        eigvals_pos
    )[::-1]

    pd.DataFrame(
        {
            "eigenvalue": LAST_EIGENVALUES
        }
    ).to_csv(
        TABLES_DIR / "covariance_eigenvalues.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))

    plt.semilogy(
        LAST_EIGENVALUES,
        marker="o",
    )

    plt.xlabel("Mode")

    plt.ylabel("Eigenvalue")

    plt.title(
        "Covariance Eigenvalue Spectrum\n"
        f"({RUN_TAG})"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "covariance_eigenspectrum.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    # Effective number of modes
    n_eff = (
        eigvals_pos.sum() ** 2
    ) / np.sum(
        eigvals_pos ** 2
    )

    print(
        f"\nEffective number of modes = "
        f"{n_eff:.2f}"
    )

    # Covariance diagnostics
    n_mocks, n_bins = all_w_h0.shape

    covariance_rank = int(
        np.linalg.matrix_rank(cov)
    )

    covariance_condition = float(
        np.max(eigvals_pos)
        / np.min(eigvals_pos)
    )

    hartlap_factor = (
        (n_mocks - n_bins - 2)
        / (n_mocks - 1)
    )

    hartlap_factor = (
        hartlap_factor
        if hartlap_factor > 0
        else 1.0
    )

    # Inverse covariance
    inv_cov = (
        hartlap_factor
        * np.linalg.pinv(cov)
    )

    # Data residuals
    delta = (
        w_obs
        - h0_mean
    )

    mock_delta = (
        all_w_h0
        - h0_mean
    )

    # SVD-regularized chi-square
    svd_stats = compute_svd_regularized_chi2(
        delta,
        mock_delta,
        cov,
        hartlap_factor,
        eigenvalue_cut=SVD_EIGENVALUE_CUT,
    )

    dof = int(
        svd_stats["modes_kept"]
    )

    # Full chi-square
    chi2 = float(
        delta @ inv_cov @ delta
    )

    chi2_red = chi2 / dof

    p_chi2 = float(
        chi2_dist.sf(
            chi2,
            dof,
        )
    )

    # Empirical chi-square p-value
    chi2_mocks = np.einsum(
        "ij,jk,ik->i",
        mock_delta,
        inv_cov,
        mock_delta,
    )

    n_chi2_extreme = int(
        np.sum(
            chi2_mocks >= chi2
        )
    )

    p_empirical_floor = (
        1.0
        / (len(chi2_mocks) + 1)
    )

    p_empirical = float(
        (n_chi2_extreme + 1)
        / (len(chi2_mocks) + 1)
    )

    sigma_equiv = sigma_equivalent_from_p(
        p_empirical
    )

    # Global normalized tension
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

    # Absolute anisotropy amplitude
    abs_observed_stat = absolute_global_stat(
        abs_obs
    )

    abs_mock_stats = np.array(
        [
            absolute_global_stat(row)
            for row in all_abs_h0
        ]
    )

    abs_empirical_p = float(
        (
            np.sum(
                abs_mock_stats
                >= abs_observed_stat
            ) + 1
        )
        / (
            len(abs_mock_stats) + 1
        )
    )

    # Non-parametric tests
    nonparam_stats = compute_nonparametric_tests(
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
    )

    # Print summary
    print(
        f"\n--- Full covariance diagnostic ({label}) ---"
    )

    print(f"Chi2 = {chi2:.3f}")

    print(
        f"Chi2/dof = {chi2_red:.3f}"
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
            f"p <= {p_empirical_floor:.4e}"
        )

    print(
        f"Sigma-equivalent "
        f"(from empirical p) = "
        f"{sigma_equiv:.2f}"
    )

    # SVD statistics
    print(
        "\n--- Primary isotropy statistic "
        "(SVD-regularized) ---"
    )

    print(
        f"Eigenvalue cut = "
        f"{svd_stats['eigenvalue_cut']:.1e}"
    )

    print(
        f"SVD modes kept = "
        f"{int(svd_stats['modes_kept'])}/{n_bins}"
    )

    print(
        f"Chi2 SVD = "
        f"{svd_stats['chi2']:.3f}"
    )

    print(
        f"Chi2 SVD/dof = "
        f"{svd_stats['chi2_red']:.3f}"
    )

    print(
        f"p-value SVD "
        f"(Chi2 analytic) = "
        f"{svd_stats['p_chi2']:.4e}"
    )

    print(
        f"p-value SVD "
        f"(empirical) = "
        f"{svd_stats['p_empirical']:.4e}"
    )

    if svd_stats["n_extreme_mocks"] == 0:

        print(
            "p-value SVD "
            "(empirical) is at the Monte Carlo floor: "
            f"p <= "
            f"{svd_stats['p_empirical_floor']:.4e}"
        )

    print(
        f"Sigma-equivalent SVD "
        f"(from empirical p) = "
        f"{svd_stats['sigma_equiv']:.2f}"
    )

    # Covariance diagnostics
    print("\n--- Covariance diagnostics ---")

    print(f"Mocks = {n_mocks}")

    print(f"Bins = {n_bins}")

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
        f"{svd_stats['condition']:.4e}"
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

    # Non-parametric tests
    print(
        "\n--- Non-parametric profile tests ---"
    )

    print(
        "w(theta) KS: "
        f"D={nonparam_stats['ks_w_stat']:.3f}, "
        f"p={nonparam_stats['ks_w_pvalue']:.4e}, "
        f"p_emp={nonparam_stats['ks_w_empirical_p']:.4e}"
    )

    print(
        "w(theta) AD: "
        f"A2={nonparam_stats['ad_w_stat']:.3f}, "
        f"p={nonparam_stats['ad_w_pvalue']:.4e}, "
        f"p_emp={nonparam_stats['ad_w_empirical_p']:.4e}"
    )

    print(
        "|<w>| KS: "
        f"D={nonparam_stats['ks_abs_stat']:.3f}, "
        f"p={nonparam_stats['ks_abs_pvalue']:.4e}, "
        f"p_emp={nonparam_stats['ks_abs_empirical_p']:.4e}"
    )

    print(
        "|<w>| AD: "
        f"A2={nonparam_stats['ad_abs_stat']:.3f}, "
        f"p={nonparam_stats['ad_abs_pvalue']:.4e}, "
        f"p_emp={nonparam_stats['ad_abs_empirical_p']:.4e}"
    )

    # Absolute anisotropy
    print(
        "\n--- Absolute anisotropy amplitude ---"
    )

    print(
        f"Observed RMS absolute amplitude = "
        f"{abs_observed_stat:.4e}"
    )

    print(
        f"Empirical p-value = "
        f"{abs_empirical_p:.4e}"
    )

    # Physical interpretation
    print("\n==================================================")
    print("PHYSICAL INTERPRETATION")
    print("==================================================")

    if USE_SEL_FUNC:
        print(
            "H0 hypothesis:"
        )

        print(
            "  isotropy + survey selection effects"
        )

    else:

        print(
            "H0 hypothesis:"
        )

        print(
            "  perfect isotropy"
        )

    if p_empirical > 0.05:

        print("\nResult:")

        print(
            "  ✓ Data are statistically compatible with H0."
        )

    else:

        print("\nResult:")

        print(
            "  ⚠ Possible deviation from H0 detected."
        )

    # Return dataclass
    return TestStatistics(
        chi2=chi2,
        chi2_red=chi2_red,
        p_chi2=p_chi2,
        p_empirical=p_empirical,
        p_empirical_floor=p_empirical_floor,
        sigma_equiv=sigma_equiv,
        chi2_svd=svd_stats["chi2"],
        chi2_svd_red=svd_stats["chi2_red"],
        p_chi2_svd=svd_stats["p_chi2"],
        p_svd_empirical=svd_stats["p_empirical"],
        p_svd_empirical_floor=svd_stats["p_empirical_floor"],
        sigma_svd_equiv=svd_stats["sigma_equiv"],
        global_tension=global_tension,
        ks_w_stat=nonparam_stats["ks_w_stat"],
        ks_w_pvalue=nonparam_stats["ks_w_pvalue"],
        ks_w_empirical_p=nonparam_stats["ks_w_empirical_p"],
        ad_w_stat=nonparam_stats["ad_w_stat"],
        ad_w_pvalue=nonparam_stats["ad_w_pvalue"],
        ad_w_empirical_p=nonparam_stats["ad_w_empirical_p"],
        ks_abs_stat=nonparam_stats["ks_abs_stat"],
        ks_abs_pvalue=nonparam_stats["ks_abs_pvalue"],
        ks_abs_empirical_p=nonparam_stats["ks_abs_empirical_p"],
        ad_abs_stat=nonparam_stats["ad_abs_stat"],
        ad_abs_pvalue=nonparam_stats["ad_abs_pvalue"],
        ad_abs_empirical_p=nonparam_stats["ad_abs_empirical_p"],
        abs_observed_stat=abs_observed_stat,
        abs_empirical_p=abs_empirical_p,
        hartlap_factor=hartlap_factor,
        covariance_rank=covariance_rank,
        covariance_condition=covariance_condition,
        n_eff=n_eff,
        svd_modes_kept=int(
            svd_stats["modes_kept"]
        ),
        svd_eigenvalue_cut=svd_stats[
            "eigenvalue_cut"
        ],
        svd_condition=svd_stats[
            "condition"
        ],
        n_mocks=n_mocks,
        n_bins=n_bins,
    )


def compute_chi2_diagnostic_tables(
    theta: np.ndarray,
    w_obs: np.ndarray,
    all_w_h0: np.ndarray,
    eigenvalue_cut: float = SVD_EIGENVALUE_CUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build diagnostic tables for covariance-aware chi-square analysis.

    Returns
    -------
    bin_df
        Bin-level diagnostics.

    mode_df
        Covariance eigenmode diagnostics.

    Notes
    -----
    The diagonal bin contributions are only approximate because
    the full covariance contains off-diagonal terms.

    The exact decomposition of the regularized chi-square is
    given by the covariance eigenmodes.
    """

    # H0 statistics
    h0_mean = np.mean(
        all_w_h0,
        axis=0,
    )

    h0_std = np.std(
        all_w_h0,
        axis=0,
        ddof=1,
    )

    n_mocks, n_bins = (
        all_w_h0.shape
    )

    # Covariance estimation
    lw = LedoitWolf()
    lw.fit(all_w_h0)
    cov = lw.covariance_

    # Hartlap correction
    hartlap_factor = (
        (n_mocks - n_bins - 2)
        / (n_mocks - 1)
    )

    if hartlap_factor <= 0:

        print(
            "\nWARNING: Hartlap factor <= 0 "
            "→ forcing Hartlap = 1"
        )

        hartlap_factor = 1.0

    # Residual vector
    delta = (
        w_obs - h0_mean
    )

    pull = delta / (
        h0_std + 1e-12
    )

    diagonal_chi2 = (
        hartlap_factor
        * delta**2
        / (h0_std**2 + 1e-24)
    )

    # Bin-level diagnostics
    bin_df = pd.DataFrame(
        {

            "run_tag":
                RUN_TAG,

            "bin_index":
                np.arange(len(theta)),

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

    # Covariance eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(
        cov
    )

    order = np.argsort(
        eigvals
    )[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # SVD mode selection
    positive = eigvals > 0.0

    if not np.any(positive):

        raise ValueError(
            "Covariance matrix has no positive eigenvalues."
        )

    max_eig = float(
        eigvals[positive][0]
    )

    keep = (
        positive
        & (
            eigvals
            >= eigenvalue_cut * max_eig
        )
    )

    # Safety fallback
    if not np.any(keep):

        keep[
            np.flatnonzero(positive)[0]
        ] = True

    # Projection into covariance modes
    delta_modes = (
        delta @ eigvecs
    )

    raw_mode_chi2 = np.zeros(
        len(eigvals),
        dtype=float,
    )

    valid = eigvals > 0.0

    raw_mode_chi2[valid] = (
        hartlap_factor
        * delta_modes[valid]**2
        / eigvals[valid]
    )

    svd_mode_chi2 = np.where(
        keep,
        raw_mode_chi2,
        0.0,
    )

    # Dominant angular scale per mode
    max_loading_idx = np.argmax(
        np.abs(eigvecs),
        axis=0,
    )

    dominant_theta = theta[
        max_loading_idx
    ]

    dominant_loading = eigvecs[
        max_loading_idx,
        np.arange(len(eigvals)),
    ]

    # Mode-level diagnostics
    mode_df = pd.DataFrame(
        {

            "run_tag":
                RUN_TAG,

            "mode_index":
                np.arange(len(eigvals)),

            "eigenvalue":
                eigvals,

            "relative_eigenvalue":
                eigvals / max_eig,

            "kept_by_svd_cut":
                keep,

            "delta_projection":
                delta_modes,

            "abs_delta_projection":
                np.abs(delta_modes),

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

    # Cumulative contributions
    mode_df[
        "cumulative_raw_chi2_contribution"
    ] = (
        mode_df[
            "raw_chi2_contribution"
        ].cumsum()
    )

    total_raw_mode_chi2 = (
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

    total_svd_mode_chi2 = (
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

    # Diagnostics
    print("\n==================================================")
    print("CHI-SQUARE DIAGNOSTICS")
    print("==================================================")

    print(f"\nRun tag: {RUN_TAG}")

    print(
        f"\nMocks: {n_mocks}"
    )

    print(
        f"Bins: {n_bins}"
    )

    print(
        f"Hartlap factor: {hartlap_factor:.5f}"
    )

    print(
        f"SVD modes kept: {np.sum(keep)} / {len(keep)}"
    )

    print(
        f"Eigenvalue cut: {eigenvalue_cut:.3e}"
    )

    print(
        f"Covariance condition number: "
        f"{np.linalg.cond(cov):.3e}"
    )

    return (bin_df, mode_df)


def compare_covariance_statistics(
    all_w_h0_old: np.ndarray,
    all_w_h0_new: np.ndarray,
) -> pd.DataFrame:
    """
    Compare covariance properties before/after pipeline fixes.

    Useful for diagnosing:
        - mock correlations
        - covariance conditioning
        - effective dimensionality
        - rank deficiencies
    """

    stats = []

    for label, w_array in [
        ("Before (correlated)", all_w_h0_old),
        ("After (fixed)", all_w_h0_new),
    ]:

        # Covariance estimation
        lw = LedoitWolf()
        lw.fit(w_array)
        cov = lw.covariance_

        # Eigenstructure
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_pos = eigvals[
            eigvals > 1e-15
        ]

        if len(eigvals_pos) == 0:

            raise ValueError(
                "Covariance matrix has no positive eigenvalues."
            )

        # Effective dimensionality
        n_eff = (
            eigvals_pos.sum() ** 2
        ) / np.sum(
            eigvals_pos ** 2
        )

        # Condition number
        condition = (
            np.max(eigvals_pos)
            / np.min(eigvals_pos)
        )

        # Matrix rank
        rank = int(
            np.linalg.matrix_rank(cov)
        )

        # Mock similarity diagnostics
        corr = np.corrcoef(w_array)

        mask = ~np.eye(
            len(corr),
            dtype=bool,
        )

        offdiag = corr[mask]

        stats.append(
            {
                "Version": label,

                "N_mocks": len(w_array),

                "N_bins": w_array.shape[1],

                "Covariance_rank": rank,

                "Effective_modes": n_eff,

                "Condition_number": condition,

                "Max_eigenvalue": np.max(eigvals_pos),

                "Min_eigenvalue": np.min(eigvals_pos),

                "Eigenvalue_ratio": (
                    np.max(eigvals_pos)
                    / np.min(eigvals_pos)
                ),

                "Mean_mock_corr": np.mean(offdiag),

                "Median_mock_corr": np.median(offdiag),

                "Max_mock_corr": np.max(offdiag),
            }
        )

    df = pd.DataFrame(stats)

    print("\n==================================================")
    print("COVARIANCE COMPARISON DIAGNOSTICS")
    print("==================================================")

    print(df.round(4))

    return df


# ==============================================================================
# Observational uncertainty estimation
# ==============================================================================


def assign_jackknife_regions(
    df_data: pd.DataFrame,
    nside_jackknife: int = NSIDE_JACKKNIFE,
    min_regions: int = MIN_JACKKNIFE_REGIONS,
) -> tuple[pd.DataFrame, np.ndarray]:

    """
    Assign observed FRBs to coarse HEALPix jackknife regions.
    """

    df = df_data.copy()

    theta = np.radians(
        90.0 - df["DEC"].values
    )

    phi = np.radians(
        df["RA"].values
    )

    region_pix = hp.ang2pix(
        nside_jackknife,
        theta,
        phi,
    )

    unique_regions, region_ids = np.unique(
        region_pix,
        return_inverse=True,
    )

    if len(unique_regions) < min_regions:

        raise ValueError(
            f"Only {len(unique_regions)} jackknife regions "
            f"are populated.\n"
            f"Decrease MIN_JACKKNIFE_REGIONS or "
            f"increase NSIDE_JACKKNIFE."
        )

    df["jackknife_region"] = region_ids

    print("\n--- Jackknife region assignment ---")

    print(
        f"NSIDE_JACKKNIFE = "
        f"{nside_jackknife}"
    )

    print(
        f"Populated regions = "
        f"{len(unique_regions)}"
    )

    return df, np.arange(len(unique_regions))


def _run_one_jackknife_region(
    region: int,
    df_data_with_regions: pd.DataFrame,
    sf_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    seed_base: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:

    """
    Compute one leave-one-region-out jackknife realization.
    """

    # Remove one region
    df_jk = df_data_with_regions[
        df_data_with_regions["jackknife_region"]
        != region
    ].reset_index(drop=True)

    # Build corresponding random catalog
    df_rand_jk = generate_mixture_catalog_improved(
        n_observed=(
            len(df_jk)
            * N_RAND_FACTOR
        ),
        sf_dict=sf_dict,
        weights=survey_weights,
        nside=nside,
        seed=(
            seed_base
            + int(region)
        ),
        use_poisson=False,
    )

    # 2pACF
    theta_jk, w_jk = compute_2pacf(
        df_jk,
        df_rand_jk,
    )

    abs_jk = get_absolute_sum(
        theta_jk,
        w_jk,
    )

    return (
        theta_jk,
        w_jk,
        abs_jk,
        len(df_jk),
    )


def jackknife_covariance(
    samples: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    """
    Return jackknife mean, covariance,
    and 1-sigma uncertainties.
    """

    samples = np.asarray(
        samples,
        dtype=float,
    )

    n_regions = samples.shape[0]

    mean = np.mean(
        samples,
        axis=0,
    )

    centered = (
        samples
        - mean
    )

    cov = (
        (n_regions - 1)
        / n_regions
    ) * (
        centered.T @ centered
    )

    err = np.sqrt(
        np.clip(
            np.diag(cov),
            0.0,
            None,
        )
    )

    return mean, cov, err


def run_jackknife_errors(
    df_data: pd.DataFrame,
    sf_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    nside_jackknife: int = NSIDE_JACKKNIFE,
    min_regions: int = MIN_JACKKNIFE_REGIONS,
    n_jobs: int = N_JOBS,
    seed_base: int = 9000,
) -> JackknifeResult:

    """
    Estimate statistical uncertainties using
    leave-one-region-out jackknife resampling.
    """

    print("\n==================================================")
    print("JACKKNIFE UNCERTAINTY ESTIMATION")
    print("==================================================")

    print(f"RUN_TAG = {RUN_TAG}")

    # Region assignment
    df_regions, regions = assign_jackknife_regions(
        df_data,
        nside_jackknife=nside_jackknife,
        min_regions=min_regions,
    )

    counts = (
        df_regions["jackknife_region"]
        .value_counts()
        .sort_index()
    )

    print("\n--- Jackknife region statistics ---")

    print(
        f"Objects per region:\n"
        f"min    = {counts.min()}\n"
        f"median = {counts.median():.1f}\n"
        f"max    = {counts.max()}"
    )

    # Parallel leave-one-out realizations
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
            delayed(_run_one_jackknife_region)(
                int(region),
                df_regions,
                sf_dict,
                survey_weights,
                nside,
                seed_base,
            )
            for region in regions
        )

    # Collect realizations
    (
        theta_samples,
        w_samples,
        abs_samples,
        n_kept,
    ) = zip(*results)

    theta = np.mean(
        np.array(theta_samples),
        axis=0,
    )

    w_samples = np.array(w_samples)

    abs_samples = np.array(abs_samples)

    # Covariances
    w_mean, w_cov, w_err = (
        jackknife_covariance(
            w_samples
        )
    )

    abs_mean, abs_cov, abs_err = (
        jackknife_covariance(
            abs_samples
        )
    )

    # Diagnostics
    print("\n--- Jackknife summary ---")

    print(
        f"Leave-one-region samples = "
        f"{len(regions)}"
    )

    print(
        f"Objects kept per sample:\n"
        f"min = {min(n_kept)}\n"
        f"max = {max(n_kept)}"
    )

    print(
        f"Median sigma[w(theta)] = "
        f"{np.median(w_err):.4e}"
    )

    print(
        f"Median sigma[|<w>|] = "
        f"{np.median(abs_err):.4e}"
    )

    # Save covariance diagnostics
    plot_covariance_correlation_matrix(
        w_cov,
        title=(
            "Jackknife Correlation Matrix\n"
            f"({RUN_TAG})"
        ),
        output=(
            "jk_correlation_matrix"
        ),
    )

    # Return container
    return JackknifeResult(
        regions=regions,
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
    seed: int,
    df_data: pd.DataFrame,
    df_rand_fixed: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    """
    Generate one bootstrap realization.

    IMPORTANT
    ---------
    - Resamples ONLY observed FRBs
    - Random catalog remains FIXED
    - Measures observational sampling uncertainty
    - Avoids artificial Monte Carlo covariance inflation
    """

    rng = np.random.default_rng(seed)

    # Bootstrap resampling with replacement
    indices = rng.choice(
        len(df_data),
        size=len(df_data),
        replace=True,
    )

    df_boot = (
        df_data.iloc[indices]
        .reset_index(drop=True)
    )

    # 2pACF
    theta_boot, w_boot = compute_2pacf(
        df_boot,
        df_rand_fixed,
    )

    # Absolute anisotropy estimator
    abs_boot = get_absolute_sum(
        theta_boot,
        w_boot,
    )

    return w_boot, abs_boot


def run_bootstrap_errors(
    df_data: pd.DataFrame,
    sf_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    n_bootstrap: int = N_BOOTSTRAP,
    n_jobs: int = N_JOBS,
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

    print("\n==================================================")
    print("BOOTSTRAP UNCERTAINTY ESTIMATION")
    print("==================================================")

    print(f"RUN_TAG = {RUN_TAG}")

    print(
        f"\nBootstrap realizations = "
        f"{n_bootstrap}"
    )

    # Fixed random catalog
    print("\nGenerating fixed random catalog...")

    df_rand_fixed = generate_mixture_catalog_improved(
        n_observed=(
            len(df_data)
            * N_RAND_FACTOR
        ),
        sf_dict=sf_dict,
        weights=survey_weights,
        nside=nside,
        seed=123456,
        use_poisson=False,
    )

    print(
        f"Random catalog size = "
        f"{len(df_rand_fixed)}"
    )

    # Parallel bootstrap realizations
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
                seed=i,
                df_data=df_data,
                df_rand_fixed=df_rand_fixed,
            )
            for i in range(n_bootstrap)
        )

    # Stack realizations
    w_realizations = np.array(
        [r[0] for r in results]
    )

    abs_realizations = np.array(
        [r[1] for r in results]
    )

    # Mean profiles
    w_mean = np.mean(
        w_realizations,
        axis=0,
    )

    abs_mean = np.mean(
        abs_realizations,
        axis=0,
    )

    # Covariance matrices
    w_cov = np.cov(
        w_realizations,
        rowvar=False,
    )

    abs_cov = np.cov(
        abs_realizations,
        rowvar=False,
    )

    # 1-sigma errors
    w_err = np.std(
        w_realizations,
        axis=0,
        ddof=1,
    )

    abs_err = np.std(
        abs_realizations,
        axis=0,
        ddof=1,
    )

    # Diagnostics
    print("\n--- Bootstrap summary ---")

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

    # Correlation matrix diagnostics
    plot_covariance_correlation_matrix(
        w_cov,
        title=(
            "Bootstrap Correlation Matrix\n"
            f"({RUN_TAG})"
        ),
        output=(
            "bootstrap_correlation_matrix"
        ),
    )

    # Return container
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
# Selection-function sensitivity analysis
# ==============================================================================


def _run_one_sensitivity_mock(
    seed: int,
    n_data: int,
    sf_dict: dict[str, np.ndarray],
    sf_cov_diag_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
) -> tuple[np.ndarray, np.ndarray]:

    """
    Generate one isotropic mock realization including
    selection-function uncertainty.

    IMPORTANT
    ---------
    This function ONLY makes physical sense when:

        USE_SEL_FUNC = True

    because the sensitivity analysis probes uncertainty
    in the selection-function modeling.
    """

    if not USE_SEL_FUNC:

        raise RuntimeError(
            "_run_one_sensitivity_mock() requires "
            "USE_SEL_FUNC = True."
        )

    # Perturbed SF realization
    sf_variant = generate_sf_variant(
        sf_dict,
        sf_cov_diag_dict,
        perturbation_scale=PERTURBATION_SCALE,
        seed=seed,
    )

    # Data mock
    d_iso = generate_mixture_catalog_improved(
        n_observed=n_data,
        sf_dict=sf_variant,
        weights=survey_weights,
        nside=nside,
        seed=seed + 1000,
        use_poisson=False,
    )

    # Random mock
    r_iso = generate_mixture_catalog_improved(
        n_observed=(
            n_data
            * N_RAND_FACTOR
        ),
        sf_dict=sf_variant,
        weights=survey_weights,
        nside=nside,
        seed=seed + 5000,
        use_poisson=False,
    )

    # 2pACF
    theta_h0, w_h0 = compute_2pacf(
        d_iso,
        r_iso,
    )

    # Absolute anisotropy
    abs_h0 = get_absolute_sum(
        theta_h0,
        w_h0,
    )

    return w_h0, abs_h0


def _empirical_sensitivity_stats(
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
) -> dict[str, float]:

    """
    Compute isotropy diagnostics for one
    sensitivity-analysis configuration.

    IMPORTANT
    ---------
    This analysis ONLY makes sense when:

        USE_SEL_FUNC = True

    because sensitivity analysis quantifies uncertainty
    induced by the selection-function modeling.
    """

    if not USE_SEL_FUNC:

        raise RuntimeError(
            "_empirical_sensitivity_stats() requires "
            "USE_SEL_FUNC = True."
        )

    # H0 mean
    h0_mean = np.mean(
        all_w_h0,
        axis=0,
    )

    # Shrinkage covariance
    lw = LedoitWolf()
    lw.fit(all_w_h0)
    cov = lw.covariance_

    # Eigenvalue spectrum
    eigvals = np.linalg.eigvalsh(cov)
    eigvals_pos = eigvals[eigvals > 0]
    n_eff = (
        (eigvals_pos.sum() ** 2)
        / np.sum(eigvals_pos ** 2)
    )

    # Covariance diagnostics
    n_mocks, n_bins = all_w_h0.shape

    covariance_rank = int(
        np.linalg.matrix_rank(cov)
    )

    covariance_condition = float(
        np.linalg.cond(cov)
    )

    hartlap_factor = (
        (n_mocks - n_bins - 2)
        / (n_mocks - 1)
    )

    hartlap_factor = (
        hartlap_factor
        if hartlap_factor > 0
        else 1.0
    )

    # Inverse covariance
    inv_cov = (
        hartlap_factor
        * np.linalg.inv(cov)
    )

    # Observed residuals
    delta = (
        w_obs
        - h0_mean
    )

    mock_delta = (
        all_w_h0
        - h0_mean
    )

    # SVD-regularized chi2
    svd_stats = compute_svd_regularized_chi2(
        delta,
        mock_delta,
        cov,
        hartlap_factor,
        eigenvalue_cut=SVD_EIGENVALUE_CUT,
    )

    dof = int(
        svd_stats["modes_kept"]
    )

    # Full chi2
    chi2 = float(
        delta @ inv_cov @ delta
    )

    chi2_red = chi2 / dof

    p_chi2 = float(
        chi2_dist.sf(
            chi2,
            dof,
        )
    )

    # Empirical chi2 calibration
    chi2_mocks = np.einsum(
        "ij,jk,ik->i",
        mock_delta,
        inv_cov,
        mock_delta,
    )

    n_chi2_extreme = int(
        np.sum(
            chi2_mocks >= chi2
        )
    )

    p_empirical_floor = (
        1.0
        / (len(chi2_mocks) + 1)
    )

    p_empirical = float(
        (n_chi2_extreme + 1)
        / (len(chi2_mocks) + 1)
    )

    sigma_empirical = (
        sigma_equivalent_from_p(
            p_empirical
        )
    )

    # Absolute anisotropy
    abs_obs_stat = absolute_global_stat(
        abs_obs
    )

    abs_mock_stats = np.array(
        [
            absolute_global_stat(row)
            for row in all_abs_h0
        ]
    )

    p_abs_empirical = float(
        (
            np.sum(
                abs_mock_stats
                >= abs_obs_stat
            ) + 1
        )
        / (len(abs_mock_stats) + 1)
    )

    # Nonparametric tests
    nonparam_stats = compute_nonparametric_tests(
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
    )

    # Return diagnostics
    return {

        # Full chi2
        "chi2":
            chi2,

        "chi2_red":
            chi2_red,

        "p_chi2_analytic":
            p_chi2,

        "p_chi2_empirical":
            p_empirical,

        "p_chi2_empirical_floor":
            p_empirical_floor,

        "n_chi2_extreme_mocks":
            float(n_chi2_extreme),

        "sigma_empirical":
            sigma_empirical,

        # SVD
        "chi2_svd":
            svd_stats["chi2"],

        "chi2_svd_red":
            svd_stats["chi2_red"],

        "p_chi2_svd_analytic":
            svd_stats["p_chi2"],

        "p_chi2_svd_empirical":
            svd_stats["p_empirical"],

        "p_chi2_svd_empirical_floor":
            svd_stats["p_empirical_floor"],

        "n_chi2_svd_extreme_mocks":
            svd_stats["n_extreme_mocks"],

        "sigma_svd_empirical":
            svd_stats["sigma_equiv"],

        "svd_modes_kept":
            svd_stats["modes_kept"],

        "svd_eigenvalue_cut":
            svd_stats["eigenvalue_cut"],

        "svd_condition":
            svd_stats["condition"],

        # COVARIANCE
        "effective_modes":
            n_eff,

        "hartlap_factor":
            hartlap_factor,

        "n_mocks":
            float(n_mocks),

        "n_bins":
            float(n_bins),

        "covariance_rank":
            float(covariance_rank),

        "covariance_condition":
            covariance_condition,

        # Absolute anisotropy
        "abs_observed_stat":
            abs_obs_stat,

        "p_abs_empirical":
            p_abs_empirical,

        # NONPARAMETRIC
        **nonparam_stats,
    }


def run_empirical_sensitivity_analysis(
    df_data: pd.DataFrame,
    nside_range: list[int] = NSIDE_SF_RANGE,
    smooth_sigma_range: list[float] = SMOOTH_SIGMA_RANGE,
    n_mocks: int = N_SENSITIVITY_EMPIRICAL_MOCKS,
    n_jobs: int = N_JOBS,
    output_csv: str | None = None,
) -> pd.DataFrame:

    """
    Evaluate robustness against selection-function modeling choices.

    IMPORTANT
    ---------
    This analysis ONLY applies when:

        USE_SEL_FUNC = True

    because the sensitivity scan probes uncertainty
    in the survey selection-function modeling.

    The Galactic mask configuration is automatically
    inherited from USE_GAL_MASK.
    """

    # Safety checks
    if not USE_SEL_FUNC:
        raise RuntimeError(
            "Sensitivity analysis requires "
            "USE_SEL_FUNC = True."
        )

    if not RUN_SENSITIVITY:
        print(
            "\nSensitivity analysis skipped "
            "(RUN_SENSITIVITY = False)."
        )
        return pd.DataFrame()

    # Output name
    if output_csv is None:
        output_csv = (
            TABLES_DIR / "sensitivity.csv"
        )

    # Initialization
    rows = []

    print("\n--- Empirical sensitivity analysis ---")
    print(
        f"USE_SEL_FUNC = {USE_SEL_FUNC}"
    )
    print(
        f"USE_GAL_MASK = {USE_GAL_MASK}"
    )
    print(
        f"Mocks per configuration = {n_mocks}"
    )

    # Parameter grid
    for nside in nside_range:

        for smooth_sigma in smooth_sigma_range:

            print(
                f"\nTesting:"
                f"\n  NSIDE = {nside}"
                f"\n  smooth_sigma = {smooth_sigma:.1f} deg"
            )

            # Build SF model
            (
                sf_dict,
                sf_cov_diag_dict,
                survey_weights,
                sf_nside,
            ) = build_survey_selection_functions_improved(
                df_data,
                nside=nside,
                smooth_sigma=smooth_sigma,
            )

            # Observed random catalog
            df_rand_obs = (
                generate_mixture_catalog_improved(
                    n_observed=(
                        len(df_data)
                        * N_RAND_FACTOR
                    ),
                    sf_dict=sf_dict,
                    weights=survey_weights,
                    nside=sf_nside,
                    seed=(
                        2000
                        + nside
                        + int(10 * smooth_sigma)
                    ),
                    use_poisson=False,
                )
            )

            # Observed 2pACF
            theta_obs, w_obs = compute_2pacf(
                df_data,
                df_rand_obs,
            )

            abs_obs = get_absolute_sum(
                theta_obs,
                w_obs,
            )

            # Monte Carlo seeds
            seeds = [

                (
                    100000
                    + 1000 * nside
                    + 10 * int(smooth_sigma)
                    + i
                )

                for i in range(n_mocks)
            ]

            # Sensitivity mocks
            with tqdm_joblib(
                tqdm(
                    desc="Sensitivity mocks",
                    total=n_mocks,
                )
            ):

                mock_results = Parallel(
                    n_jobs=n_jobs,
                    backend="loky",
                    batch_size="auto",
                )(
                    delayed(
                        _run_one_sensitivity_mock
                    )(
                        seed,
                        len(df_data),
                        sf_dict,
                        sf_cov_diag_dict,
                        survey_weights,
                        sf_nside,
                    )
                    for seed in seeds
                )

            # Stack mocks
            all_w_h0, all_abs_h0 = zip(
                *mock_results
            )

            all_w_h0 = np.array(all_w_h0)
            all_abs_h0 = np.array(all_abs_h0)

            # Statistics
            stats = _empirical_sensitivity_stats(
                w_obs,
                abs_obs,
                all_w_h0,
                all_abs_h0,
            )

            # Save row
            row = {

                # Configuration
                "use_sel_func":
                    USE_SEL_FUNC,

                "use_gal_mask":
                    USE_GAL_MASK,

                "gal_cut_deg":
                    (
                        GAL_CUT
                        if USE_GAL_MASK
                        else np.nan
                    ),

                "nside":
                    nside,

                "smooth_sigma_deg":
                    smooth_sigma,

                # Statistics
                **stats,
            }

            rows.append(row)

            # Summary
            print(
                "\nConfiguration summary:"
            )

            print(
                f"  chi2/dof = "
                f"{row['chi2_red']:.3f}"
            )

            print(
                f"  p_emp = "
                f"{row['p_chi2_empirical']:.4e}"
            )

            print(
                f"  p_svd = "
                f"{row['p_chi2_svd_empirical']:.4e}"
            )

            print(
                f"  modes = "
                f"{int(row['svd_modes_kept'])}"
                f"/{int(row['n_bins'])}"
            )

            print(
                f"  sigma_emp = "
                f"{row['sigma_empirical']:.2f}"
            )

            print(
                f"  p_abs = "
                f"{row['p_abs_empirical']:.4e}"
            )

    # Results table
    results = pd.DataFrame(rows)

    results.to_csv(
        output_csv,
        index=False,
    )

    print("\n--- Empirical sensitivity summary ---")
    print(
        results.round(4)
    )
    print(
        f"\nSaved: {output_csv}"
    )

    return results


# ==============================================================================
# Visualization
# ==============================================================================


def plot_covariance_correlation_matrix(
    cov: np.ndarray,
    title: str = "Correlation Matrix",
    output: str = "correlation_matrix",
) -> None:
    """
    Visualize covariance-derived correlation matrix.
    """

    # Numerical stabilization
    diag = np.diag(cov)

    sigma = np.sqrt(
        np.clip(diag, 1e-30, None)
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

    # Diagnostics
    offdiag = corr[
        ~np.eye(len(corr), dtype=bool)
    ]

    print("\n--- Covariance correlation diagnostics ---")

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

    # Plot
    plt.figure(figsize=(10, 8))

    im = plt.imshow(
        corr,
        origin="lower",
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    plt.colorbar(
        im,
        label="Correlation",
    )

    plt.xlabel("Angular bin index")
    plt.ylabel("Angular bin index")

    plt.title(
        f"{title}\n"
        f"({RUN_TAG})"
    )

    plt.tight_layout()

    filename = FIGURES_DIR / f"{output}.png"

    plt.savefig(
        filename,
        dpi=300,
    )

    plt.show()

    print(
        f"\nSaved correlation matrix:"
    )

    print(f"  {filename}")


def plot_mock_similarity(
    all_w_h0: np.ndarray,
    n_compare: int = 100,
    output: str = "mock_similarity",
) -> None:

    """
    Diagnose statistical similarity between H0 mocks.

    IMPORTANT
    ---------
    We normalize each angular bin independently before
    computing mock-to-mock correlations.

    Otherwise:
        - the common mean structure dominates;
        - correlations become artificially large.
    """

    # Subsample
    subset = all_w_h0[
        :min(n_compare, len(all_w_h0))
    ]

    # Normalize angular bins
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
        subset - mock_mean
    ) / (
        mock_std + 1e-12
    )

    # Mock correlation matrix
    corr = np.corrcoef(
        normalized
    )

    corr = np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mask = ~np.eye(
        len(corr),
        dtype=bool,
    )

    offdiag = corr[mask]

    # Diagnostics
    print("\n==================================================")
    print("MOCK SIMILARITY DIAGNOSTICS")
    print("==================================================")

    print(
        f"Mocks analyzed: "
        f"{len(subset)}"
    )

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

    # Physical interpretation
    median_corr = np.median(
        np.abs(offdiag)
    )

    if median_corr < 0.05:

        print(
            "\nInterpretation:"
        )

        print(
            "  ✓ Mock ensemble appears statistically independent."
        )

    elif median_corr < 0.15:

        print(
            "\nInterpretation:"
        )

        print(
            "  ⚠ Mild residual correlations detected."
        )

    else:

        print(
            "\nInterpretation:"
        )

        print(
            "  ⚠ Strong mock correlations detected."
        )

        print(
            "  Covariance estimation may be biased."
        )

    # Plot
    plt.figure(figsize=(10, 8))

    im = plt.imshow(
        corr,
        origin="lower",
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    plt.colorbar(
        im,
        label="Mock-to-mock correlation",
    )

    plt.title(
        "Normalized Mock Similarity Matrix\n"
        f"({RUN_TAG})"
    )

    plt.xlabel("Mock index")
    plt.ylabel("Mock index")

    plt.tight_layout()

    filename = FIGURES_DIR / f"{output}.png"

    plt.savefig(
        filename,
        dpi=300,
    )

    plt.show()

    print(
        "\nSaved mock similarity matrix:"
    )

    print(f"  {filename}")


def plot_top_survey_maps(
    df_data: pd.DataFrame,
    sf_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    max_surveys: int = 12,
    output_prefix: str | None = None,
    cmap: str = "viridis",
) -> pd.DataFrame:

    """
    Generate sky maps for dominant surveys.

    Outputs:
        - Equatorial maps
        - Galactic maps

    Notes
    -----
    This visualization only makes sense when
    USE_SEL_FUNC = True.
    """

    # ----------------------------------------------------------
    # Skip if SF disabled
    # ----------------------------------------------------------

    if not USE_SEL_FUNC:

        print(
            "\n--- Survey map visualization skipped ---"
        )

        print(
            "USE_SEL_FUNC=False "
            "→ no survey selection functions available."
        )

        return pd.DataFrame()

    # ----------------------------------------------------------
    # Survey partitioning
    # ----------------------------------------------------------

    surveys = split_by_survey(df_data)

    ordered = sorted(
        sf_dict.keys(),
        key=lambda name: survey_weights[name],
        reverse=True,
    )[:max_surveys]

    rows = []

    # ----------------------------------------------------------
    # Coordinate systems
    # ----------------------------------------------------------

    for coord_system in [
        "equatorial",
        "galactic",
    ]:

        ncols = 4

        nrows = int(
            np.ceil(len(ordered) / ncols)
        )

        fig = plt.figure(
            figsize=(
                5.0 * ncols,
                3.8 * nrows,
            )
        )

        # ======================================================
        # Survey loop
        # ======================================================

        for idx, survey_name in enumerate(
            ordered,
            start=1,
        ):

            subdf = surveys[survey_name]

            sf = sf_dict[survey_name]

            # --------------------------------------------------
            # Galactic visualization
            # --------------------------------------------------

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
                        subdf["RA"].values
                        * u.degree
                    ),
                    dec=(
                        subdf["DEC"].values
                        * u.degree
                    ),
                    frame="icrs",
                )

                lon = (
                    coords.galactic.l.degree
                )

                lat = (
                    coords.galactic.b.degree
                )

            # --------------------------------------------------
            # Equatorial visualization
            # --------------------------------------------------

            else:

                sf_plot = sf

                lon = (
                    subdf["RA"].values
                )

                lat = (
                    subdf["DEC"].values
                )

            # --------------------------------------------------
            # Diagnostics
            # --------------------------------------------------

            active_coverage = float(
                np.mean(
                    sf > np.max(sf) * 1e-3
                )
                * 100.0
            )

            entropy = float(
                -np.sum(
                    sf[sf > 0]
                    * np.log2(sf[sf > 0])
                )
            )

            # --------------------------------------------------
            # Plot
            # --------------------------------------------------

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
                    f"w={survey_weights[survey_name]:.1%}"
                ),
                cbar=True,
                min=0,
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

            # --------------------------------------------------
            # Save diagnostics
            # --------------------------------------------------

            rows.append(
                {
                    "survey": survey_name,
                    "n_objects": len(subdf),
                    "weight": (
                        survey_weights[
                            survey_name
                        ]
                    ),
                    "active_coverage_pct": (
                        active_coverage
                    ),
                    "entropy_bits": entropy,
                    "nside": nside,
                    "coordinate_system": (
                        coord_system
                    ),
                    "use_gal_mask": (
                        USE_GAL_MASK
                    ),
                    "run_tag": RUN_TAG,
                }
            )

        # ======================================================
        # Save figure
        # ======================================================

        filename = (
            f"{output_prefix}_"
            f"{coord_system}.png"
        )

        plt.savefig(
            filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.show()

        print(
            f"Saved survey maps: "
            f"{filename}"
        )

    return pd.DataFrame(rows)


def plot_results_with_jackknife(
    theta: np.ndarray,
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
    jackknife: JackknifeResult,
    output_2pacf: str | None = None,
    output_abs: str | None = None,
) -> None:

    """
    Plot isotropy diagnostics using:
        - H0 mock confidence bands
        - Jackknife observational uncertainties

    Notes
    -----
    H0 depends on configuration:

    USE_SEL_FUNC = False
        -> perfect isotropic sky

    USE_SEL_FUNC = True
        -> isotropy convolved with survey selection functions
    """

    # ----------------------------------------------------------
    # Output filenames
    # ----------------------------------------------------------

    if output_2pacf is None:

        output_2pacf = (
            FIGURES_DIR / "2pacf_jackknife.png"
        )

    if output_abs is None:

        output_abs = (
            FIGURES_DIR / "absolute_anisotropy_jackknife.png"
        )

    # ----------------------------------------------------------
    # Plot style
    # ----------------------------------------------------------

    sns.set_theme(
        style="white",
        context="talk",
        rc={
            "axes.edgecolor": "0.25",
            "axes.linewidth": 1.1,
        },
    )

    # ----------------------------------------------------------
    # H0 ensemble statistics
    # ----------------------------------------------------------

    h0_mean = np.mean(
        all_w_h0,
        axis=0,
    )

    h0_std = np.std(
        all_w_h0,
        axis=0,
        ddof=1,
    )

    h0_abs_mean = np.mean(
        all_abs_h0,
        axis=0,
    )

    h0_abs_std = np.std(
        all_abs_h0,
        axis=0,
        ddof=1,
    )

    # ----------------------------------------------------------
    # Colors
    # ----------------------------------------------------------

    band_3 = "#c9daeb"
    band_2 = "#97b8d3"
    band_1 = "#5b8ebb"

    h0_line = "#2f4458"

    jk_err = "firebrick"

    # ----------------------------------------------------------
    # 2pACF visualization
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    # ----------------------------------------------------------
    # H0 confidence bands
    # ----------------------------------------------------------

    ax.fill_between(
        theta,
        h0_mean - 3 * h0_std,
        h0_mean + 3 * h0_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        theta,
        h0_mean - 2 * h0_std,
        h0_mean + 2 * h0_std,
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

    # ----------------------------------------------------------
    # Observed measurements
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Jackknife uncertainties
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # H0 mean
    # ----------------------------------------------------------

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
        0,
        color="0.15",
        linewidth=1.1,
        alpha=0.9,
        zorder=0,
    )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$w(\theta)$"
    )

    h0_label = (
        "Pure isotropic H0"
        if not USE_SEL_FUNC
        else "Isotropy + survey SF H0"
    )

    ax.set_title(
        "Angular Two-Point Correlation Function\n"
        f"(Jackknife uncertainties | {h0_label})"
    )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------

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

    plt.tight_layout()

    plt.savefig(
        output_2pacf,
        dpi=300,
    )

    plt.show()

    print(
        f"Saved: {output_2pacf}"
    )

    # ----------------------------------------------------------
    # Absolute anisotropy visualization
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    # ----------------------------------------------------------
    # H0 confidence bands
    # ----------------------------------------------------------

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - 3 * h0_abs_std,
        h0_abs_mean + 3 * h0_abs_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - 2 * h0_abs_std,
        h0_abs_mean + 2 * h0_abs_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - h0_abs_std,
        h0_abs_mean + h0_abs_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
        zorder=3,
    )

    # ----------------------------------------------------------
    # Observed statistic
    # ----------------------------------------------------------

    obs_handle = ax.scatter(
        COARSE_CENTERS,
        abs_obs,
        s=45,
        facecolor=jk_err,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
        label="Observed statistic",
    )

    # ----------------------------------------------------------
    # Jackknife uncertainties
    # ----------------------------------------------------------

    jk_handle = ax.errorbar(
        COARSE_CENTERS,
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

    # ----------------------------------------------------------
    # H0 mean
    # ----------------------------------------------------------

    h0_line_handle, = ax.plot(
        COARSE_CENTERS,
        h0_abs_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=5,
    )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------

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

    plt.tight_layout()

    plt.savefig(
        output_abs,
        dpi=300,
    )

    plt.show()

    print(
        f"Saved: {output_abs}"
    )


def plot_results_with_bootstrap(
    theta: np.ndarray,
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
    bootstrap: BootstrapResult,
    output_2pacf: str | None = None,
    output_abs: str | None = None,
) -> None:

    """
    Plot isotropy diagnostics using:
        - H0 mock confidence bands
        - Bootstrap observational uncertainties

    Notes
    -----
    H0 depends on configuration:

    USE_SEL_FUNC = False
        -> perfect isotropic sky

    USE_SEL_FUNC = True
        -> isotropy convolved with survey selection functions
    """

    # ----------------------------------------------------------
    # Output filenames
    # ----------------------------------------------------------

    if output_2pacf is None:

        output_2pacf = (
            FIGURES_DIR / "2pacf_bootstrap.png"
        )

    if output_abs is None:

        output_abs = (
            FIGURES_DIR / "absolute_anisotropy_bootstrap.png"
        )

    # ----------------------------------------------------------
    # Plot style
    # ----------------------------------------------------------

    sns.set_theme(
        style="white",
        context="talk",
        rc={
            "axes.edgecolor": "0.25",
            "axes.linewidth": 1.1,
        },
    )

    # ----------------------------------------------------------
    # H0 ensemble statistics
    # ----------------------------------------------------------

    h0_mean = np.mean(
        all_w_h0,
        axis=0,
    )

    h0_std = np.std(
        all_w_h0,
        axis=0,
        ddof=1,
    )

    h0_abs_mean = np.mean(
        all_abs_h0,
        axis=0,
    )

    h0_abs_std = np.std(
        all_abs_h0,
        axis=0,
        ddof=1,
    )

    # ----------------------------------------------------------
    # Colors
    # ----------------------------------------------------------

    band_3 = "#c9daeb"
    band_2 = "#97b8d3"
    band_1 = "#5b8ebb"

    h0_line = "#2f4458"

    obs_color = "white"

    bootstrap_color = "black"

    # ----------------------------------------------------------
    # 2pACF visualization
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    # ----------------------------------------------------------
    # H0 confidence bands
    # ----------------------------------------------------------

    ax.fill_between(
        theta,
        h0_mean - 3 * h0_std,
        h0_mean + 3 * h0_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        theta,
        h0_mean - 2 * h0_std,
        h0_mean + 2 * h0_std,
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

    # ----------------------------------------------------------
    # Observed measurements
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Bootstrap uncertainties
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # H0 mean
    # ----------------------------------------------------------

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
        0,
        color="0.15",
        linewidth=1.1,
        alpha=0.9,
    )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$w(\theta)$"
    )

    h0_label = (
        "Pure isotropic H0"
        if not USE_SEL_FUNC
        else "Isotropy + survey SF H0"
    )

    ax.set_title(
        "Angular Two-Point Correlation Function\n"
        f"(Bootstrap uncertainties | {h0_label})"
    )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------

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

    plt.tight_layout()

    plt.savefig(
        output_2pacf,
        dpi=300,
    )

    plt.show()

    print(
        f"Saved: {output_2pacf}"
    )

    # ----------------------------------------------------------
    # Absolute anisotropy visualization
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    # ----------------------------------------------------------
    # H0 confidence bands
    # ----------------------------------------------------------

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - 3 * h0_abs_std,
        h0_abs_mean + 3 * h0_abs_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - 2 * h0_abs_std,
        h0_abs_mean + 2 * h0_abs_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - h0_abs_std,
        h0_abs_mean + h0_abs_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
        zorder=3,
    )

    # ----------------------------------------------------------
    # Observed statistic
    # ----------------------------------------------------------

    obs_handle = ax.scatter(
        COARSE_CENTERS,
        abs_obs,
        s=48,
        facecolor="firebrick",
        edgecolor=obs_color,
        linewidth=1.0,
        zorder=6,
        label="Observed statistic",
    )

    # ----------------------------------------------------------
    # Bootstrap uncertainties
    # ----------------------------------------------------------

    bootstrap_handle = ax.errorbar(
        COARSE_CENTERS,
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

    # ----------------------------------------------------------
    # H0 mean
    # ----------------------------------------------------------

    h0_line_handle, = ax.plot(
        COARSE_CENTERS,
        h0_abs_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=4,
    )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------

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

    plt.tight_layout()

    plt.savefig(
        output_abs,
        dpi=300,
    )

    plt.show()

    print(
        f"Saved: {output_abs}"
    )


def plot_results_jk_vs_bootstrap(
    theta: np.ndarray,
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
    jackknife: JackknifeResult,
    bootstrap: BootstrapResult,
    output_2pacf: str | None = None,
    output_abs: str | None = None,
) -> None:

    """
    Compare jackknife and bootstrap uncertainty estimates.

    Notes
    -----
    H0 depends on configuration:

    USE_SEL_FUNC = False
        -> perfect isotropic sky

    USE_SEL_FUNC = True
        -> isotropy convolved with survey selection functions
    """

    # ----------------------------------------------------------
    # Output filenames
    # ----------------------------------------------------------

    if output_2pacf is None:

        output_2pacf = (
            FIGURES_DIR / "2pacf_jk_vs_bootstrap.png"
        )

    if output_abs is None:

        output_abs = (
            FIGURES_DIR / "absolute_anisotropy_jk_vs_bootstrap.png"
        )

    # ----------------------------------------------------------
    # Plot style
    # ----------------------------------------------------------

    sns.set_theme(
        style="white",
        context="talk",
    )

    # ----------------------------------------------------------
    # H0 ensemble statistics
    # ----------------------------------------------------------

    h0_mean = np.mean(
        all_w_h0,
        axis=0,
    )

    h0_std = np.std(
        all_w_h0,
        axis=0,
        ddof=1,
    )

    h0_abs_mean = np.mean(
        all_abs_h0,
        axis=0,
    )

    h0_abs_std = np.std(
        all_abs_h0,
        axis=0,
        ddof=1,
    )

    # ----------------------------------------------------------
    # Colors
    # ----------------------------------------------------------

    band_3 = "#c9daeb"
    band_2 = "#97b8d3"
    band_1 = "#5b8ebb"

    h0_line = "#2f4458"

    jk_err = "firebrick"

    bootstrap_color = "black"

    # ----------------------------------------------------------
    # 2pACF visualization
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    # ----------------------------------------------------------
    # H0 confidence bands
    # ----------------------------------------------------------

    ax.fill_between(
        theta,
        h0_mean - 3 * h0_std,
        h0_mean + 3 * h0_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
    )

    ax.fill_between(
        theta,
        h0_mean - 2 * h0_std,
        h0_mean + 2 * h0_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
    )

    ax.fill_between(
        theta,
        h0_mean - h0_std,
        h0_mean + h0_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
    )

    # ----------------------------------------------------------
    # Observed measurements
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Jackknife uncertainties
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Bootstrap uncertainties
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # H0 mean
    # ----------------------------------------------------------

    h0_line_handle, = ax.plot(
        theta,
        h0_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
    )

    ax.axhline(
        0,
        color="0.15",
        linewidth=1.1,
        alpha=0.9,
    )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------

    ax.set_xlabel(
        r"$\theta$ [deg]"
    )

    ax.set_ylabel(
        r"$w(\theta)$"
    )

    h0_label = (
        "Pure isotropic H0"
        if not USE_SEL_FUNC
        else "Isotropy + survey SF H0"
    )

    ax.set_title(
        "2pACF Uncertainty Comparison\n"
        f"(Jackknife vs Bootstrap | {h0_label})"
    )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------

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

    plt.tight_layout()

    plt.savefig(
        output_2pacf,
        dpi=300,
    )

    plt.show()

    print(
        f"Saved: {output_2pacf}"
    )

    # ----------------------------------------------------------
    # Absolute anisotropy visualization
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    # ----------------------------------------------------------
    # H0 confidence bands
    # ----------------------------------------------------------

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - 3 * h0_abs_std,
        h0_abs_mean + 3 * h0_abs_std,
        color=band_3,
        alpha=0.95,
        linewidth=0,
    )

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - 2 * h0_abs_std,
        h0_abs_mean + 2 * h0_abs_std,
        color=band_2,
        alpha=0.78,
        linewidth=0,
    )

    ax.fill_between(
        COARSE_CENTERS,
        h0_abs_mean - h0_abs_std,
        h0_abs_mean + h0_abs_std,
        color=band_1,
        alpha=0.58,
        linewidth=0,
    )

    # ----------------------------------------------------------
    # Observed statistic
    # ----------------------------------------------------------

    obs_handle = ax.scatter(
        COARSE_CENTERS,
        abs_obs,
        s=45,
        facecolor=jk_err,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
        label="Observed statistic",
    )

    # ----------------------------------------------------------
    # Jackknife uncertainties
    # ----------------------------------------------------------

    jk_handle = ax.errorbar(
        COARSE_CENTERS,
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

    # ----------------------------------------------------------
    # Bootstrap uncertainties
    # ----------------------------------------------------------

    bootstrap_handle = ax.errorbar(
        COARSE_CENTERS,
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

    # ----------------------------------------------------------
    # H0 mean
    # ----------------------------------------------------------

    h0_line_handle, = ax.plot(
        COARSE_CENTERS,
        h0_abs_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
    )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------

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

    plt.tight_layout()

    plt.savefig(
        output_abs,
        dpi=300,
    )

    plt.show()

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

    if isinstance(
        value,
        (float, np.floating),
    ):

        if np.isfinite(value):

            return (
                f"{float(value):.{precision}g}"
            )

        return str(value)

    if isinstance(
        value,
        (int, np.integer),
    ):

        return str(int(value))

    if isinstance(value, bool):

        return str(value)

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

    lines.append(f"## {title}")

    lines.append("")

    max_key = max(
        len(str(key))
        for key in values
    )

    lines.append("```text")

    for key, value in values.items():

        formatted = _format_report_value(
            value,
            precision=precision,
        )

        lines.append(
            f"{key:<{max_key}} : {formatted}"
        )

    lines.append("```")

    lines.append("")


def _top_matrix_pairs(
    matrix: pd.DataFrame,
    n: int = 10,
    absolute: bool = True,
) -> pd.DataFrame:
    """
    Extract strongest pairwise survey relations.
    """

    if matrix.empty:

        return pd.DataFrame(
            columns=[
                "survey_1",
                "survey_2",
                "value",
            ]
        )

    rows = []

    columns = list(matrix.columns)

    for i, row_name in enumerate(matrix.index):

        for j, col_name in enumerate(columns):

            # Upper triangle only
            if j <= i:
                continue

            value = float(
                matrix.iloc[i, j]
            )

            rows.append(
                {
                    "survey_1": row_name,
                    "survey_2": col_name,
                    "value": value,
                    "abs_value": abs(value),
                }
            )

    if len(rows) == 0:

        return pd.DataFrame(
            columns=[
                "survey_1",
                "survey_2",
                "value",
            ]
        )

    df_pairs = pd.DataFrame(rows)

    sort_col = (
        "abs_value"
        if absolute
        else "value"
    )

    return (
        df_pairs
        .sort_values(
            sort_col,
            ascending=False,
        )
        .head(n)
        [
            [
                "survey_1",
                "survey_2",
                "value",
            ]
        ]
        .reset_index(drop=True)
    )


def save_report(
    results: dict[str, object],
    paths: dict[str, str] | None = None,
    output_path: str | None = None,
) -> str:
    """
    Save consolidated human-readable analysis report.
    """

    if output_path is None:
        output_path = REPORT_DIR / "summary.md"

    stats = results["stats"]

    jackknife = results["jackknife"]

    df_data = results["df_data"]

    all_w_h0 = results["all_w_h0"]

    all_abs_h0 = results["all_abs_h0"]

    chi2_bin_diagnostics = results.get(
        "chi2_bin_diagnostics"
    )

    chi2_svd_modes = results.get(
        "chi2_svd_modes"
    )

    lines: list[str] = []

    lines.append(
        "# FRB Isotropy Analysis Report"
    )

    lines.append("")

    lines.append(
        "Consolidated output of the isotropy pipeline."
    )

    lines.append("")

    # ----------------------------------------------------------
    # Run configuration
    # ----------------------------------------------------------

    _append_key_values(
        lines,
        "Run Configuration",
        {
            "run_tag": RUN_TAG,
            "catalog_path": ALL_FRB_PATH,
            "catalog_size": len(df_data),

            "use_gal_mask": USE_GAL_MASK,
            "use_selection_function": USE_SEL_FUNC,
            "run_sensitivity": RUN_SENSITIVITY,

            "gal_cut_deg": GAL_CUT,

            "bin_size_deg": BIN_SIZE,
            "n_bins": stats.n_bins,

            "n_rand_factor": N_RAND_FACTOR,

            "n_mocks": stats.n_mocks,
            "n_ensemble": N_ENSEMBLE,
            "n_mocks_per_ensemble": (
                N_MOCKS_PER_ENSEMBLE
            ),

            "nside_sf": NSIDE_SF,
            "smooth_sigma_deg": SMOOTH_SIGMA,
            "perturbation_scale": (
                PERTURBATION_SCALE
            ),

            "nside_jackknife": (
                NSIDE_JACKKNIFE
            ),

            "svd_eigenvalue_cut": (
                stats.svd_eigenvalue_cut
            ),
        },
    )

    # ----------------------------------------------------------
    # Full covariance statistics
    # ----------------------------------------------------------

    _append_key_values(
        lines,
        "Full Covariance Diagnostic",
        {
            "chi2": stats.chi2,
            "chi2_red": stats.chi2_red,

            "p_chi2_analytic": stats.p_chi2,

            "p_chi2_empirical": (
                stats.p_empirical
            ),

            "p_chi2_empirical_floor": (
                stats.p_empirical_floor
            ),

            "sigma_equiv_empirical": (
                stats.sigma_equiv
            ),

            "hartlap_factor": (
                stats.hartlap_factor
            ),

            "rms_normalized_deviation": (
                stats.global_tension
            ),
        },
    )

    # ----------------------------------------------------------
    # SVD statistics
    # ----------------------------------------------------------

    _append_key_values(
        lines,
        "Primary Isotropy Statistic (SVD-Regularized)",
        {
            "chi2_svd": stats.chi2_svd,

            "chi2_svd_red": (
                stats.chi2_svd_red
            ),

            "p_chi2_svd_analytic": (
                stats.p_chi2_svd
            ),

            "p_chi2_svd_empirical": (
                stats.p_svd_empirical
            ),

            "sigma_svd_empirical": (
                stats.sigma_svd_equiv
            ),

            "svd_modes_kept": (
                f"{stats.svd_modes_kept}/"
                f"{stats.n_bins}"
            ),

            "svd_retained_condition": (
                stats.svd_condition
            ),
        },
    )

    # ----------------------------------------------------------
    # Covariance diagnostics
    # ----------------------------------------------------------

    _append_key_values(
        lines,
        "Covariance Diagnostics",
        {
            "mocks": stats.n_mocks,

            "bins": stats.n_bins,

            "covariance_rank": (
                f"{stats.covariance_rank}/"
                f"{stats.n_bins}"
            ),

            "covariance_condition": (
                stats.covariance_condition
            ),

            "effective_modes": (
                stats.n_eff
            ),

            "h0_w_shape": (
                tuple(all_w_h0.shape)
            ),

            "h0_abs_shape": (
                tuple(all_abs_h0.shape)
            ),
        },
    )

    # ----------------------------------------------------------
    # Non-parametric tests
    # ----------------------------------------------------------

    _append_key_values(
        lines,
        "Non-parametric Profile Tests",
        {
            "w_ks_stat": stats.ks_w_stat,
            "w_ks_pvalue": stats.ks_w_pvalue,
            "w_ks_empirical_p": (
                stats.ks_w_empirical_p
            ),

            "w_ad_stat": stats.ad_w_stat,
            "w_ad_pvalue": stats.ad_w_pvalue,
            "w_ad_empirical_p": (
                stats.ad_w_empirical_p
            ),

            "abs_ks_stat": stats.ks_abs_stat,
            "abs_ks_pvalue": (
                stats.ks_abs_pvalue
            ),
            "abs_ks_empirical_p": (
                stats.ks_abs_empirical_p
            ),

            "abs_ad_stat": stats.ad_abs_stat,
            "abs_ad_pvalue": (
                stats.ad_abs_pvalue
            ),
            "abs_ad_empirical_p": (
                stats.ad_abs_empirical_p
            ),
        },
    )

    # ----------------------------------------------------------
    # Absolute anisotropy
    # ----------------------------------------------------------

    _append_key_values(
        lines,
        "Absolute Anisotropy Amplitude",
        {
            "observed_rms_absolute_amplitude": (
                stats.abs_observed_stat
            ),

            "empirical_pvalue": (
                stats.abs_empirical_p
            ),
        },
    )

    # ----------------------------------------------------------
    # Jackknife
    # ----------------------------------------------------------

    _append_key_values(
        lines,
        "Jackknife Summary",
        {
            "n_regions": (
                len(jackknife.regions)
            ),

            "median_sigma_w": float(
                np.median(jackknife.w_err)
            ),

            "min_sigma_w": float(
                np.min(jackknife.w_err)
            ),

            "max_sigma_w": float(
                np.max(jackknife.w_err)
            ),

            "median_sigma_abs": float(
                np.median(jackknife.abs_err)
            ),

            "min_sigma_abs": float(
                np.min(jackknife.abs_err)
            ),

            "max_sigma_abs": float(
                np.max(jackknife.abs_err)
            ),
        },
    )

    # ----------------------------------------------------------
    # Chi-square bin audit
    # ----------------------------------------------------------

    if chi2_bin_diagnostics is not None:

        lines.append(
            "## Chi-square Bin Audit"
        )

        lines.append("")

        lines.append("```text")

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

        lines.append(
            chi2_bin_diagnostics[
                bin_columns
            ]
            .head(12)
            .to_string(index=False)
        )

        lines.append("```")

        lines.append("")

    # ----------------------------------------------------------
    # SVD audit
    # ----------------------------------------------------------

    if chi2_svd_modes is not None:

        lines.append(
            "## Chi-square SVD Mode Audit"
        )

        lines.append("")

        lines.append("```text")

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

        lines.append(
            chi2_svd_modes[
                mode_columns
            ]
            .head(12)
            .to_string(index=False)
        )

        lines.append("```")

        lines.append("")

    # ----------------------------------------------------------
    # Selection-function validation
    # ----------------------------------------------------------

    sf_validation = results.get(
        "sf_validation"
    )

    if sf_validation is not None:

        lines.append(
            "## Selection Function Validation"
        )

        lines.append("")

        lines.append("```text")

        sf_top = (
            sf_validation
            .sort_values(
                "weight",
                ascending=False,
            )
            .head(12)
        )

        lines.append(
            sf_top.to_string(index=False)
        )

        lines.append("```")

        lines.append("")

    # ----------------------------------------------------------
    # Intersurvey correlations
    # ----------------------------------------------------------

    corr_df = results.get(
        "intersurvey_corr"
    )

    if corr_df is not None:

        lines.append(
            "## Intersurvey Correlations"
        )

        lines.append("")

        lines.append("```text")

        lines.append(
            _top_matrix_pairs(
                corr_df,
                n=12,
                absolute=True,
            ).to_string(index=False)
        )

        lines.append("```")

        lines.append("")

    # ----------------------------------------------------------
    # Intersurvey overlap
    # ----------------------------------------------------------

    overlap_df = results.get(
        "intersurvey_overlap"
    )

    if overlap_df is not None:

        lines.append(
            "## Intersurvey Overlap"
        )

        lines.append("")

        lines.append("```text")

        lines.append(
            _top_matrix_pairs(
                overlap_df,
                n=12,
                absolute=False,
            ).to_string(index=False)
        )

        lines.append("```")

        lines.append("")

    # ----------------------------------------------------------
    # Sensitivity analysis
    # ----------------------------------------------------------

    sensitivity = results.get(
        "sensitivity"
    )

    if sensitivity is not None:

        lines.append(
            "## Sensitivity Analysis"
        )

        lines.append("")

        lines.append("```text")

        lines.append(
            sensitivity.to_string(
                index=False
            )
        )

        lines.append("```")

        lines.append("")

    # ----------------------------------------------------------
    # Saved files
    # ----------------------------------------------------------

    if paths:

        lines.append(
            "## Saved Files"
        )

        lines.append("")

        lines.append("```text")

        for name, path in paths.items():

            lines.append(
                f"{name}: {path}"
            )

        lines.append("```")

        lines.append("")

    # ----------------------------------------------------------
    # Write report
    # ----------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "\n".join(lines).rstrip()
            + "\n"
        )

    return output_path


def save_tables(
    results: dict[str, object],
    prefix: str | None = None,
) -> dict[str, str]:

    """
    Save analysis tables and diagnostics.
    """

    if prefix is None:
        prefix = RUN_TAG

    paths = {

        "sf_validation":
            TABLES_DIR / "sf_validation.csv",

        "intersurvey_corr":
            TABLES_DIR / "intersurvey_corr.csv",

        "intersurvey_overlap":
            TABLES_DIR / "intersurvey_overlap.csv",

        "stats_summary":
            TABLES_DIR / "stats_summary.csv",

        "jackknife_w_errors":
            TABLES_DIR / "jackknife_w_errors.csv",

        "jackknife_abs_errors":
            TABLES_DIR / "jackknife_abs_errors.csv",

        "covariance_matrix":
            TABLES_DIR / "covariance_matrix.csv",

        "chi2_bin_diagnostics":
            TABLES_DIR / "chi2_bin_diagnostics.csv",

        "svd_mode_contributions":
            TABLES_DIR / "svd_mode_contributions.csv",
    }

    # ----------------------------------------------------------
    # Selection-function diagnostics
    # ----------------------------------------------------------

    if results.get("sf_validation") is not None:

        results["sf_validation"].to_csv(
            paths["sf_validation"],
            index=False,
        )

    if results.get("intersurvey_corr") is not None:

        results["intersurvey_corr"].to_csv(
            paths["intersurvey_corr"]
        )

    if results.get("intersurvey_overlap") is not None:

        results["intersurvey_overlap"].to_csv(
            paths["intersurvey_overlap"]
        )

    # ----------------------------------------------------------
    # Statistics summary
    # ----------------------------------------------------------

    stats = results["stats"]

    stats_rows = [
        ["chi2", "chi2", stats.chi2],
        ["chi2", "chi2_red", stats.chi2_red],
        ["chi2", "p_empirical", stats.p_empirical],

        ["svd", "chi2_svd", stats.chi2_svd],
        ["svd", "chi2_svd_red", stats.chi2_svd_red],
        ["svd", "p_svd_empirical", stats.p_svd_empirical],

        ["covariance", "effective_modes", stats.n_eff],
        ["covariance", "condition", stats.covariance_condition],

        ["anisotropy", "abs_stat", stats.abs_observed_stat],
        ["anisotropy", "abs_p", stats.abs_empirical_p],
    ]

    pd.DataFrame(
        stats_rows,
        columns=[
            "category",
            "statistic",
            "value",
        ],
    ).to_csv(
        paths["stats_summary"],
        index=False,
    )

    # ----------------------------------------------------------
    # Jackknife diagnostics
    # ----------------------------------------------------------

    jackknife = results["jackknife"]

    pd.DataFrame(
        {
            "theta_deg": results["theta"],
            "w_obs": results["w_obs"],
            "w_jackknife_mean": (
                jackknife.w_mean
            ),
            "w_jackknife_err": (
                jackknife.w_err
            ),
        }
    ).to_csv(
        paths["jackknife_w_errors"],
        index=False,
    )

    pd.DataFrame(
        {
            "theta_center_deg":
                COARSE_CENTERS,

            "abs_obs":
                results["abs_obs"],

            "abs_jackknife_mean":
                jackknife.abs_mean,

            "abs_jackknife_err":
                jackknife.abs_err,
        }
    ).to_csv(
        paths["jackknife_abs_errors"],
        index=False,
    )

    # ----------------------------------------------------------
    # Chi-square diagnostics
    # ----------------------------------------------------------

    (
        chi2_bin_diagnostics,
        svd_mode_contributions,
    ) = compute_chi2_diagnostic_tables(
        results["theta"],
        results["w_obs"],
        results["all_w_h0"],
        eigenvalue_cut=(
            SVD_EIGENVALUE_CUT
        ),
    )

    results[
        "chi2_bin_diagnostics"
    ] = chi2_bin_diagnostics

    results[
        "chi2_svd_modes"
    ] = svd_mode_contributions

    chi2_bin_diagnostics.to_csv(
        paths["chi2_bin_diagnostics"],
        index=False,
    )

    svd_mode_contributions.to_csv(
        paths["svd_mode_contributions"],
        index=False,
    )

    # ----------------------------------------------------------
    # Covariance matrix
    # ----------------------------------------------------------

    cov = results.get(
        "covariance_matrix"
    )

    if cov is not None:

        pd.DataFrame(cov).to_csv(
            paths["covariance_matrix"],
            index=False,
        )

    # ----------------------------------------------------------
    # Sensitivity analysis
    # ----------------------------------------------------------

    sensitivity = results.get(
        "sensitivity"
    )

    if sensitivity is not None:

        paths["sensitivity"] = (
            TABLES_DIR / "sensitivity.csv"
        )

        sensitivity.to_csv(
            paths["sensitivity"],
            index=False,
        )

    # ----------------------------------------------------------
    # Markdown report
    # ----------------------------------------------------------

    paths["report"] = (
        REPORT_DIR / "summary.md"
    )

    save_report(
        results,
        paths=paths,
        output_path=paths["report"],
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n--- Saved analysis tables ---")

    for name, path in paths.items():

        print(f"{name}: {path}")

    return paths


# ==============================================================================
# Pipeline orchestration
# ==============================================================================


def main(
    config: AnalysisConfig | None = None,
    save_outputs: bool = True,
    n_jobs: int = N_JOBS,
) -> dict[str, object]:

    """
    Execute the complete FRB isotropy analysis pipeline.

    Parameters
    ----------
    config
        Run-specific analysis flags. If omitted, the currently active
        module configuration is reused.

    save_outputs
        Save tables and the consolidated report when True.

    Notes
    -----
    Supported physical modes are defined by ``config``:
        - use_sel_func=False -> pure isotropic H0
        - use_sel_func=True  -> isotropy convolved with survey selection functions
        - run_sensitivity=True requires use_sel_func=True
    """

    if config is None:
        config = AnalysisConfig(
            use_gal_mask=USE_GAL_MASK,
            use_sel_func=USE_SEL_FUNC,
            run_sensitivity=RUN_SENSITIVITY,
        )

    apply_analysis_config(config)

    print("\n==================================================")
    print("FRB ISOTROPY ANALYSIS PIPELINE")
    print("==================================================")

    print("\n--- Configuration ---")

    print(f"USE_SEL_FUNC      = {USE_SEL_FUNC}")

    print(f"USE_GAL_MASK      = {USE_GAL_MASK}")

    print(f"RUN_SENSITIVITY   = {RUN_SENSITIVITY}")

    print(f"RUN_TAG           = {RUN_TAG}")

    print(f"N_JOBS            = {n_jobs}")

    print(f"BIN_TYPE          = Linear")

    # ----------------------------------------------------------
    # Load and mask catalog
    # ----------------------------------------------------------

    print("\n--- Loading catalog ---")

    df_data = apply_mask(
        load_catalog(ALL_FRB_PATH)
    )

    print(
        f"Masked catalog size: "
        f"{len(df_data)}"
    )

    # ----------------------------------------------------------
    # Build selection functions
    # ----------------------------------------------------------

    print("\n--- Building selection functions ---")

    sf_dict, sf_cov_diag_dict, survey_weights, sf_nside = (
        build_survey_selection_functions_improved(
            df_data,
            nside=NSIDE_SF,
            smooth_sigma=SMOOTH_SIGMA,
        )
    )

    # ----------------------------------------------------------
    # Selection-function diagnostics
    # ----------------------------------------------------------

    sf_validation = None

    corr_df = None

    overlap_df = None

    survey_map_summary = None

    if USE_SEL_FUNC:

        validator = SelectionFunctionValidator(
            sf_dict,
            survey_weights,
            sf_nside,
        )

        sf_validation = (
            validator.check_coverage()
        )

        validator.plot_sf(
            save_prefix=FIGURES_DIR / "sf_validation"
        )

        corr_df, overlap_df = (
            analyze_intersurvey_correlations(
                df_data
            )
        )

        survey_map_summary = (
            plot_top_survey_maps(
                df_data,
                sf_dict,
                survey_weights,
                sf_nside,
                output_prefix=(
                    FIGURES_DIR / "survey_maps"
                ),
            )
        )

    # ----------------------------------------------------------
    # Observed random catalog
    # ----------------------------------------------------------

    print("\n--- Generating observed random catalog ---")

    df_rand_obs = (
        generate_mixture_catalog_improved(
            n_observed=(
                len(df_data)
                * N_RAND_FACTOR
            ),
            sf_dict=sf_dict,
            weights=survey_weights,
            nside=sf_nside,
            seed=2000,
            use_poisson=False,
        )
    )

    # ----------------------------------------------------------
    # Observed 2pACF
    # ----------------------------------------------------------

    print("\n--- Computing observed statistics ---")

    theta, w_obs = compute_2pacf(
        df_data,
        df_rand_obs,
    )

    abs_obs = get_absolute_sum(
        theta,
        w_obs,
    )

    # ----------------------------------------------------------
    # Jackknife uncertainties
    # ----------------------------------------------------------

    print("\n--- Running jackknife ---")

    jackknife = run_jackknife_errors(
        df_data,
        sf_dict,
        survey_weights,
        sf_nside,
        nside_jackknife=NSIDE_JACKKNIFE,
        min_regions=MIN_JACKKNIFE_REGIONS,
        n_jobs=n_jobs,
    )

    # ----------------------------------------------------------
    # Bootstrap uncertainties
    # ----------------------------------------------------------

    print("\n--- Running bootstrap ---")

    bootstrap = run_bootstrap_errors(
        df_data,
        sf_dict,
        survey_weights,
        sf_nside,
        n_bootstrap=N_BOOTSTRAP,
        n_jobs=n_jobs,
    )

    # ----------------------------------------------------------
    # H0 ensemble
    # ----------------------------------------------------------

    print("\n--- Running H0 ensemble ---")

    all_w_h0, all_abs_h0 = (
        run_ensemble_mocks(
            df_data,
            sf_dict,
            sf_cov_diag_dict,
            survey_weights,
            sf_nside,
            n_ensemble=N_ENSEMBLE,
            n_mocks_per=N_MOCKS_PER_ENSEMBLE,
            perturbation_scale=PERTURBATION_SCALE,
            n_jobs=n_jobs,
        )
    )

    # ----------------------------------------------------------
    # Main statistics
    # ----------------------------------------------------------

    print("\n--- Computing statistics ---")

    stats = compute_statistics(
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
        label="Jackknife errors",
    )

    # ----------------------------------------------------------
    # Visualizations
    # ----------------------------------------------------------

    print("\n--- Generating plots ---")

    plot_results_with_jackknife(
        theta,
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
        jackknife,
        output_2pacf=FIGURES_DIR / "2pacf_jackknife.png",
        output_abs=FIGURES_DIR / "absolute_anisotropy_jackknife.png",
    )

    plot_results_with_bootstrap(
        theta,
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
        bootstrap,
        output_2pacf=FIGURES_DIR / "2pacf_bootstrap.png",
        output_abs=FIGURES_DIR / "absolute_anisotropy_bootstrap.png",
    )

    plot_results_jk_vs_bootstrap(
        theta,
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
        jackknife,
        bootstrap,
        output_2pacf=FIGURES_DIR / "2pacf_jk_vs_bootstrap.png",
        output_abs=FIGURES_DIR / "absolute_anisotropy_jk_vs_bootstrap.png",
    )

    # ----------------------------------------------------------
    # Chi-square diagnostics
    # ----------------------------------------------------------

    print("\n--- Computing chi-square diagnostics ---")

    chi2_bin_diagnostics, chi2_svd_modes = (
        compute_chi2_diagnostic_tables(
            theta,
            w_obs,
            all_w_h0,
            eigenvalue_cut=SVD_EIGENVALUE_CUT,
        )
    )

    # ----------------------------------------------------------
    # Sensitivity analysis
    # ----------------------------------------------------------

    sensitivity_results = None

    if RUN_SENSITIVITY and USE_SEL_FUNC:

        print("\n--- Running sensitivity analysis ---")

        sensitivity_results = (
            run_empirical_sensitivity_analysis(
                df_data,
                n_jobs=n_jobs,
                output_csv=(
                    TABLES_DIR / "sensitivity.csv"
                ),
            )
        )

    elif RUN_SENSITIVITY and not USE_SEL_FUNC:

        print(
            "\nSensitivity analysis skipped: "
            "USE_SEL_FUNC=False"
        )

    # ----------------------------------------------------------
    # Results dictionary
    # ----------------------------------------------------------

    results = {

        # ------------------------------------------------------
        # Configuration
        # ------------------------------------------------------

        "config": config,

        # ------------------------------------------------------
        # Data
        # ------------------------------------------------------

        "df_data": df_data,

        "theta": theta,

        "w_obs": w_obs,

        "abs_obs": abs_obs,

        # ------------------------------------------------------
        # Selection functions
        # ------------------------------------------------------

        "sf_dict": sf_dict,

        "sf_cov_diag_dict": sf_cov_diag_dict,

        "survey_weights": survey_weights,

        "sf_validation": sf_validation,

        "survey_map_summary": (
            survey_map_summary
        ),

        # ------------------------------------------------------
        # Survey diagnostics
        # ------------------------------------------------------

        "intersurvey_corr": corr_df,

        "intersurvey_overlap": overlap_df,

        # ------------------------------------------------------
        # Uncertainty estimation
        # ------------------------------------------------------

        "jackknife": jackknife,

        "bootstrap": bootstrap,

        # ------------------------------------------------------
        # H0 ensemble
        # ------------------------------------------------------

        "all_w_h0": all_w_h0,

        "all_abs_h0": all_abs_h0,

        # ------------------------------------------------------
        # Statistics
        # ------------------------------------------------------

        "stats": stats,

        # ------------------------------------------------------
        # Chi-square diagnostics
        # ------------------------------------------------------

        "chi2_bin_diagnostics": (
            chi2_bin_diagnostics
        ),

        "chi2_svd_modes": (
            chi2_svd_modes
        ),

        # ------------------------------------------------------
        # Covariance
        # ------------------------------------------------------

        "covariance_matrix": (
            LAST_COVARIANCE_MATRIX
        ),

        # ------------------------------------------------------
        # Sensitivity analysis
        # ------------------------------------------------------

        "sensitivity": (
            sensitivity_results
        ),
    }

    # ----------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------

    if save_outputs:

        print("\n--- Saving outputs ---")

        results["saved_tables"] = (
            save_tables(
                results,
                prefix=RUN_TAG,
            )
        )

    print("\n==================================================")
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("==================================================")

    return results


# ==============================================================================
# Script entry point
# ==============================================================================

if __name__ == "__main__":

    results = main(
        config=DEFAULT_CONFIG,
        save_outputs=True,
        n_jobs=N_JOBS,
    )