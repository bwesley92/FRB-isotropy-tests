"""
Level 4 pipeline for FRB isotropy analysis.

This pipeline tests the statistical compatibility of the observed FRB sky
distribution with the Cosmological Principle under observationally realistic
conditions.

The analysis combines:
- angular two-point correlation statistics (2pACF);
- covariance-aware chi-square inference;
- SVD-regularized isotropy diagnostics;
- non-parametric profile tests (KS and Anderson-Darling);
- absolute anisotropy estimators;
- jackknife spatial uncertainty estimation;
- multi-survey selection-function modeling;
- isotropic mock ensemble generation;
- empirical systematic robustness tests.

The null hypothesis (H0) is not an idealized perfectly uniform sky.
Instead, isotropic mock catalogs are generated through probabilistic
selection functions inferred from the real observational surveys,
including:
- heterogeneous sky coverage;
- instrumental selection effects;
- survey-dependent angular footprints;
- masking effects;
- finite-sampling fluctuations.

This Level 4 implementation extends the previous Level 3 framework by adding:
- leave-one-region-out jackknife covariance estimation;
- empirical spatial stability diagnostics;
- SVD covariance auditing tools;
- systematic sensitivity analyses against selection-function modeling choices;
- automated scientific reporting and export tables.

The resulting pipeline provides a reproducible and fully auditable framework
for observational isotropy inference in FRB catalogs.
"""

# ==============================================================
# 0. IMPORTS AND ENVIRONMENT SETUP
# ==============================================================

from __future__ import annotations

from dataclasses import dataclass
import os
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
from joblib import Parallel, delayed
from scipy.stats import anderson_ksamp
from scipy.stats import chi2 as chi2_dist
from scipy.stats import ks_2samp
from scipy.stats import norm
import treecorr
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

warnings.filterwarnings(
    "ignore",
    message='.*"verbose" was deprecated.*',
    category=Warning,
)


# ==============================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================

# Paths and input catalog
BASE_PATH = "/home/brunowesley/projetos/FRB-isotropy-tests/FRB_catalogs/"
ALL_FRB_PATH = os.path.join(BASE_PATH, "SkyPosition.csv")

# Observational sky mask
GAL_CUT = 20.0  # Galactic latitude cut [deg]

# SF modeling
NSIDE_SF = 64             # SF HEALPix resolution
SMOOTH_SIGMA = 5.0        # Angular smoothing scale [deg]
PERTURBATION_SCALE = 1.0  # Random perturbation amplitude

# Angular 2pACF configuration
USE_LOG = False
MIN_SEP = 0.1   # [deg]
MAX_SEP = 180.0 # [deg]
if USE_LOG:
    BIN_TYPE = "Log"
    N_BINS = 20
else:
    BIN_TYPE = "Linear"
    BIN_SIZE = 10.0  # [deg]
    N_BINS = int((MAX_SEP - MIN_SEP) / BIN_SIZE)

# Absolute anisotropy estimator
COARSE_BINS = np.arange(0, 181, 20)
COARSE_CENTERS = 0.5 * (COARSE_BINS[:-1] + COARSE_BINS[1:])

# H0 mock generation
N_RAND_FACTOR = 20
N_ENSEMBLE = 20
N_MOCKS_PER_ENSEMBLE = 100
N_MOCKS = 2000  # N_ENSEMBLE * N_MOCKS_PER_ENSEMBLE

# Covariance regularization
SVD_EIGENVALUE_CUT = 1e-2

# JK resampling
NSIDE_JACKKNIFE = 8  # HEALPix resolution for JK regions
MIN_JACKKNIFE_REGIONS = 20

# Intersurvey overlap analysis
OVERLAP_RADIUS_DEG = 5.0  # Angular matching radius [deg]
OVERLAP_NSIDE = 32        # HEALPix resolution for overlap diagnostics

# Sensitivity / robustness analysis
NSIDE_SF_RANGE = [32, 64, 128]
SMOOTH_SIGMA_RANGE = [3.0, 5.0, 8.0, 10.0]
N_SENSITIVITY_EMPIRICAL_MOCKS = 100


# ==============================================================
# 2. PIPELINE DATA STRUCTURES
# ==============================================================

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


# ==============================================================
# 3. DATA LOADING
# ==============================================================

def load_catalog(path: str = ALL_FRB_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )
    return df[["RA", "DEC", "Reporting_Group_s"]].dropna().reset_index(drop=True)


# ==============================================================
# 4. SKY MASKING
# ==============================================================

# Apply the Galactic mask to the observed FRB catalog
def apply_mask(df: pd.DataFrame, gal_cut: float = GAL_CUT) -> pd.DataFrame:
    coords = SkyCoord(
        ra=df["RA"].values * u.degree,
        dec=df["DEC"].values * u.degree,
        frame="icrs",
    )
    b = coords.galactic.b.degree
    return df[np.abs(b) > gal_cut].reset_index(drop=True)


# ==============================================================
# 5. HEALPIX SKY MASKS
# ==============================================================

# Build the Galactic mask in HEALPix space for
# SF maps and mock catalog generation
def healpix_galactic_mask(nside: int, gal_cut: float = GAL_CUT) -> np.ndarray:
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    coords = SkyCoord(
        ra=np.degrees(phi) * u.degree,
        dec=(90.0 - np.degrees(theta)) * u.degree,
        frame="icrs",
    )
    return np.abs(coords.galactic.b.degree) > gal_cut


# ==============================================================
# 6. HEALPIX MAP TRANSFORMATIONS
# ==============================================================

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
        "C" = equatorial (ICRS)
        "G" = Galactic

    coord_out : str
        Output coordinate system.

    Returns
    -------
    np.ndarray
        Rotated HEALPix map.
    """

    nside = hp.get_nside(hmap)
    rotator = hp.Rotator(coord=[coord_out, coord_in])

    theta, phi = hp.pix2ang(
        nside,
        np.arange(hp.nside2npix(nside)),
    )

    theta_rot, phi_rot = rotator(theta, phi)

    pix_rot = hp.ang2pix(
        nside,
        theta_rot,
        phi_rot,
    )

    return hmap[pix_rot]


# ==============================================================
# 7. SURVEY PARTITIONING
# ==============================================================

# Split the full FRB catalog into survey-specific subcatalogs
def split_by_survey(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["Reporting_Group_s"] = df["Reporting_Group_s"].fillna("UNKNOWN")

    surveys: dict[str, list[tuple[float, float]]] = {}
    for row in df.itertuples(index=False):
        for group in str(row.Reporting_Group_s).split(","):
            group = group.strip()
            surveys.setdefault(group, []).append((row.RA, row.DEC))

    return {
        name: pd.DataFrame(values, columns=["RA", "DEC"])
        for name, values in surveys.items()
    }


# ==============================================================
# 8. SURVEY SELECTION FUNCTION MODELING
# ==============================================================

# Build a probabilistic model of the observable sky for a given survey
def build_selection_function_improved(
    subdf: pd.DataFrame,
    nside: int = NSIDE_SF,
    smooth_sigma: float = SMOOTH_SIGMA,
    gal_cut: float = GAL_CUT,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one survey selection function and diagonal Poisson uncertainty."""
    npix = hp.nside2npix(nside)

    theta = np.radians(90.0 - subdf["DEC"].values)
    phi = np.radians(subdf["RA"].values)
    pix = hp.ang2pix(nside, theta, phi)

    counts = np.bincount(pix, minlength=npix).astype(float)
    total_counts = counts.sum()
    if total_counts <= 0:
        raise ValueError("Cannot build a selection function from an empty survey.")

    sf_err_counts = np.sqrt(np.maximum(counts, 1.0))
    sf = counts / total_counts
    sf_cov_diag = (sf_err_counts / total_counts) ** 2

    if smooth_sigma > 0:
        sf = hp.smoothing(sf, sigma=np.radians(smooth_sigma))
        sf = np.clip(sf, 0.0, None)
        sf_cov_diag *= 2.0

    gal_mask = healpix_galactic_mask(nside, gal_cut=gal_cut)
    sf[~gal_mask] = 0.0

    sf_sum = sf.sum()
    if sf_sum <= 0:
        raise ValueError("Selection function vanished after masking.")
    sf /= sf_sum

    return sf, sf_cov_diag


# ==============================================================
# 9. MULTI-SURVEY SELECTION FUNCTION ASSEMBLY
# ==============================================================

# Build the multi-survey ensemble of observable-sky models.
def build_survey_selection_functions_improved(
    df: pd.DataFrame,
    nside: int = NSIDE_SF,
    smooth_sigma: float = SMOOTH_SIGMA,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], int]:
    surveys = split_by_survey(df)
    total_memberships = sum(len(subdf) for subdf in surveys.values())

    sf_dict: dict[str, np.ndarray] = {}
    sf_cov_diag_dict: dict[str, np.ndarray] = {}
    survey_weights: dict[str, float] = {}

    for name, subdf in surveys.items():
        sf, sf_cov_diag = build_selection_function_improved(
            subdf,
            nside=nside,
            smooth_sigma=smooth_sigma,
        )
        sf_dict[name] = sf
        sf_cov_diag_dict[name] = sf_cov_diag
        survey_weights[name] = len(subdf) / total_memberships

    return sf_dict, sf_cov_diag_dict, survey_weights, nside


# ==============================================================
# 10. SELECTION FUNCTION ENSEMBLE PERTURBATIONS
# ==============================================================

# Generate perturbed realizations of the survey SFs
def generate_sf_variant(
    sf_dict: dict[str, np.ndarray],
    sf_cov_diag_dict: dict[str, np.ndarray],
    perturbation_scale: float = PERTURBATION_SCALE,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Perturb each selection function inside its diagonal uncertainty model."""
    rng = np.random.default_rng(seed)
    sf_dict_perturbed: dict[str, np.ndarray] = {}

    for survey_name, sf in sf_dict.items():
        sigma = np.sqrt(sf_cov_diag_dict[survey_name])
        noise = rng.normal(0.0, perturbation_scale * sigma)
        sf_pert = np.clip(sf + noise, 0.0, None)

        if sf_pert.sum() <= 0:
            sf_pert = sf.copy()
        else:
            sf_pert /= sf_pert.sum()

        sf_dict_perturbed[survey_name] = sf_pert

    return sf_dict_perturbed


# ==============================================================
# 11. SUB-PIXEL ANGULAR RANDOMIZATION
# ==============================================================

def _jitter_pixel_centers(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pix_radius_deg = np.degrees(hp.max_pixrad(nside))
    dec_jitter = rng.uniform(-pix_radius_deg, pix_radius_deg, size=len(dec))
    cos_dec = np.clip(np.cos(np.radians(dec)), 0.1, None)
    ra_jitter = rng.uniform(-pix_radius_deg, pix_radius_deg, size=len(ra)) / cos_dec

    ra_out = (ra + ra_jitter) % 360.0
    dec_out = np.clip(dec + dec_jitter, -89.999, 89.999)
    return ra_out, dec_out


# ==============================================================
# 11. SUB-PIXEL ANGULAR RANDOMIZATION
# ==============================================================

# Randomize mock positions within the HEALPix pixel scale
# to remove artificial discretization from the HEALPix grid
def generate_mixture_catalog_improved(
    n: int,
    sf_dict: dict[str, np.ndarray],
    weights: dict[str, float],
    nside: int,
    seed: int | None = None,
    jitter_pixels: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    surveys = np.array(list(sf_dict.keys()))
    w = np.array([weights[s] for s in surveys], dtype=float)
    w /= w.sum()

    chosen_surveys = rng.choice(surveys, size=n, p=w)
    ra = np.empty(n, dtype=float)
    dec = np.empty(n, dtype=float)

    for survey in surveys:
        mask = chosen_surveys == survey
        n_survey = int(mask.sum())
        if n_survey == 0:
            continue

        probs = np.clip(sf_dict[str(survey)], 0.0, None)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            raise ValueError(f"Invalid zero-probability SF for {survey}.")
        probs = probs / probs_sum

        pix = rng.choice(len(probs), size=n_survey, p=probs)
        theta, phi = hp.pix2ang(nside, pix)
        ra_survey = np.degrees(phi)
        dec_survey = 90.0 - np.degrees(theta)

        if jitter_pixels:
            ra_survey, dec_survey = _jitter_pixel_centers(
                ra_survey,
                dec_survey,
                nside,
                rng,
            )

        ra[mask] = ra_survey
        dec[mask] = dec_survey

    return pd.DataFrame({"RA": ra, "DEC": dec})


# ==============================================================
# 12. TWO-POINT ANGULAR CORRELATION FUNCTION
# ==============================================================

# Compute the 2pACP of the FRB catalog
def compute_2pacf(
    df_data: pd.DataFrame,
    df_rand: pd.DataFrame,
    log_spacing: bool = USE_LOG,
) -> tuple[np.ndarray, np.ndarray]:
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

    corr = treecorr.NNCorrelation(
        min_sep=MIN_SEP,
        max_sep=MAX_SEP,
        nbins=N_BINS,
        sep_units="deg",
        metric="Arc",
        bin_type="Log" if log_spacing else "Linear",
    )
    rr = treecorr.NNCorrelation(**corr.config)
    dr = treecorr.NNCorrelation(**corr.config)

    corr.process(cat_data)
    rr.process(cat_rand)
    dr.process(cat_data, cat_rand)
    corr.calculateXi(rr=rr, dr=dr)

    theta = np.exp(corr.meanlogr) if log_spacing else corr.meanr
    return theta, corr.xi


# ==============================================================
# 13. ABSOLUTE ANISOTROPY ESTIMATOR
# ==============================================================

# Compute the coarse-grained absolute angular anisotropy amplitude
def get_absolute_sum(theta: np.ndarray, w: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(len(COARSE_BINS) - 1):
        mask = (theta >= COARSE_BINS[i]) & (theta < COARSE_BINS[i + 1])
        vals.append(np.abs(np.mean(w[mask])) if np.any(mask) else 0.0)
    return np.array(vals)


# ==============================================================
# 14. GLOBAL ABSOLUTE ANISOTROPY AMPLITUDE
# ==============================================================

# Compute a single scalar estimator for the global anisotropy level
def absolute_global_stat(abs_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(abs_values**2)))


# ==============================================================
# 15. GAUSSIAN SIGNIFICANCE CONVERSION
# ==============================================================

# Convert a p-value into an equivalent Gaussian sigma significance
def sigma_equivalent_from_p(p_value: float) -> float:
    p_clipped = np.clip(p_value, 1e-300, 1.0 - 1e-16)
    return float(norm.isf(p_clipped))


# ==============================================================
# 16. SVD-REGULARIZED COVARIANCE INFERENCE
# ==============================================================

# Compute a stable chi-square statistic after removing noisy covariance modes
def compute_svd_regularized_chi2(
    delta: np.ndarray,
    mock_delta: np.ndarray,
    cov: np.ndarray,
    hartlap_factor: float,
    eigenvalue_cut: float = SVD_EIGENVALUE_CUT,
) -> dict[str, float]:
    """Compute chi-square after discarding poorly constrained covariance modes."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 0.0
    if not np.any(positive):
        raise ValueError("Covariance matrix has no positive eigenvalues.")

    max_eig = float(eigvals[positive][0])
    keep = positive & (eigvals >= eigenvalue_cut * max_eig)
    if not np.any(keep):
        keep[np.flatnonzero(positive)[0]] = True

    kept_eigvals = eigvals[keep]
    kept_eigvecs = eigvecs[:, keep]

    delta_modes = delta @ kept_eigvecs
    mock_modes = mock_delta @ kept_eigvecs

    chi2_svd = float(hartlap_factor * np.sum(delta_modes**2 / kept_eigvals))
    chi2_svd_mocks = hartlap_factor * np.sum(mock_modes**2 / kept_eigvals, axis=1)

    n_extreme = int(np.sum(chi2_svd_mocks >= chi2_svd))
    p_empirical_floor = 1.0 / (len(chi2_svd_mocks) + 1)
    p_empirical = float((n_extreme + 1) / (len(chi2_svd_mocks) + 1))

    modes_kept = int(np.sum(keep))
    svd_condition = float(kept_eigvals[0] / kept_eigvals[-1])

    return {
        "chi2": chi2_svd,
        "chi2_red": chi2_svd / modes_kept,
        "p_chi2": float(chi2_dist.sf(chi2_svd, modes_kept)),
        "p_empirical": p_empirical,
        "p_empirical_floor": p_empirical_floor,
        "sigma_equiv": sigma_equivalent_from_p(p_empirical),
        "n_extreme_mocks": float(n_extreme),
        "modes_kept": float(modes_kept),
        "eigenvalue_cut": float(eigenvalue_cut),
        "condition": svd_condition,
    }


# ==============================================================
# 17. ANDERSON-DARLING TWO-SAMPLE TEST
# ==============================================================

# Compute the Anderson-Darling statistic and p-value for two samples
def _anderson_ksamp_stat_p(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = anderson_ksamp([sample_a, sample_b])
    return float(result.statistic), float(result.pvalue)


# ==============================================================
# 18. EMPIRICALLY CALIBRATED NONPARAMETRIC PROFILE TESTS
# ==============================================================

# Compare observed angular profiles against the isotropic mock ensemble
def _profile_nonparametric_stats(
    observed: np.ndarray,
    mocks: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """KS/AD profile tests calibrated against the H0 mock ensemble."""
    ks_result = ks_2samp(observed, reference, alternative="two-sided", mode="auto")
    ks_stat = float(ks_result.statistic)
    ks_pvalue = float(ks_result.pvalue)
    ad_stat, ad_pvalue = _anderson_ksamp_stat_p(observed, reference)

    ks_mock_stats = np.array(
        [
            ks_2samp(mock, reference, alternative="two-sided", mode="auto").statistic
            for mock in mocks
        ],
        dtype=float,
    )
    ad_mock_stats = np.array(
        [_anderson_ksamp_stat_p(mock, reference)[0] for mock in mocks],
        dtype=float,
    )

    n_mocks = len(mocks)
    ks_n_extreme = int(np.sum(ks_mock_stats >= ks_stat))
    ad_n_extreme = int(np.sum(ad_mock_stats >= ad_stat))

    return {
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "ks_empirical_p": float((ks_n_extreme + 1) / (n_mocks + 1)),
        "ks_empirical_floor": float(1.0 / (n_mocks + 1)),
        "ks_n_extreme_mocks": float(ks_n_extreme),
        "ad_stat": ad_stat,
        "ad_pvalue": ad_pvalue,
        "ad_empirical_p": float((ad_n_extreme + 1) / (n_mocks + 1)),
        "ad_empirical_floor": float(1.0 / (n_mocks + 1)),
        "ad_n_extreme_mocks": float(ad_n_extreme),
    }


# ==============================================================
# 19. NONPARAMETRIC ISOTROPY DIAGNOSTICS
# ==============================================================

# Compare observed angular statistics against the isotropic mock ensemble
def compute_nonparametric_tests(
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
) -> dict[str, float]:
    """
    KS/AD diagnostics.

    These tests compare the observed binned profile against the H0 benchmark
    mean and calibrate the statistic by applying the same comparison to each
    H0 mock. They are complementary diagnostics, not replacements for the
    covariance-aware chi-square tests.
    """
    w_reference = np.mean(all_w_h0, axis=0)
    abs_reference = np.mean(all_abs_h0, axis=0)

    w_stats = _profile_nonparametric_stats(w_obs, all_w_h0, w_reference)
    abs_stats = _profile_nonparametric_stats(abs_obs, all_abs_h0, abs_reference)

    return {
        "ks_w_stat": w_stats["ks_stat"],
        "ks_w_pvalue": w_stats["ks_pvalue"],
        "ks_w_empirical_p": w_stats["ks_empirical_p"],
        "ks_w_empirical_floor": w_stats["ks_empirical_floor"],
        "ks_w_n_extreme_mocks": w_stats["ks_n_extreme_mocks"],
        "ad_w_stat": w_stats["ad_stat"],
        "ad_w_pvalue": w_stats["ad_pvalue"],
        "ad_w_empirical_p": w_stats["ad_empirical_p"],
        "ad_w_empirical_floor": w_stats["ad_empirical_floor"],
        "ad_w_n_extreme_mocks": w_stats["ad_n_extreme_mocks"],
        "ks_abs_stat": abs_stats["ks_stat"],
        "ks_abs_pvalue": abs_stats["ks_pvalue"],
        "ks_abs_empirical_p": abs_stats["ks_empirical_p"],
        "ks_abs_empirical_floor": abs_stats["ks_empirical_floor"],
        "ks_abs_n_extreme_mocks": abs_stats["ks_n_extreme_mocks"],
        "ad_abs_stat": abs_stats["ad_stat"],
        "ad_abs_pvalue": abs_stats["ad_pvalue"],
        "ad_abs_empirical_p": abs_stats["ad_empirical_p"],
        "ad_abs_empirical_floor": abs_stats["ad_empirical_floor"],
        "ad_abs_n_extreme_mocks": abs_stats["ad_n_extreme_mocks"],
    }


# ==============================================================
# 20. CHI-SQUARE MODE AND BIN DIAGNOSTICS
# ==============================================================

# Decompose the isotropy tension into angular bins and covariance eigenmodes
def compute_chi2_diagnostic_tables(
    theta: np.ndarray,
    w_obs: np.ndarray,
    all_w_h0: np.ndarray,
    eigenvalue_cut: float = SVD_EIGENVALUE_CUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return bin-level and SVD-mode diagnostics for auditing chi-square tension.

    The bin table is a diagonal diagnostic: it shows per-bin pulls and
    delta^2 / sigma^2 contributions. The full chi-square is not separable by
    bin when off-diagonal covariance terms are present, so the SVD table gives
    the exact mode-by-mode decomposition used by the regularized chi-square.
    """
    h0_mean = np.mean(all_w_h0, axis=0)
    h0_std = np.std(all_w_h0, axis=0, ddof=1)
    lw = LedoitWolf()
    lw.fit(all_w_h0)
    cov = lw.covariance_
    n_mocks, n_bins = all_w_h0.shape
    hartlap_factor = (n_mocks - n_bins - 2) / (n_mocks - 1)
    hartlap_factor = hartlap_factor if hartlap_factor > 0 else 1.0

    delta = w_obs - h0_mean
    pull = delta / (h0_std + 1e-12)
    diagonal_chi2 = hartlap_factor * delta**2 / (h0_std**2 + 1e-24)

    bin_df = pd.DataFrame(
        {
            "bin_index": np.arange(len(theta)),
            "theta_deg": theta,
            "w_obs": w_obs,
            "h0_mean": h0_mean,
            "h0_std": h0_std,
            "delta": delta,
            "pull": pull,
            "abs_pull": np.abs(pull),
            "diagonal_chi2_contribution": diagonal_chi2,
        }
    ).sort_values("abs_pull", ascending=False).reset_index(drop=True)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 0.0
    max_eig = float(eigvals[positive][0]) if np.any(positive) else np.nan
    keep = positive & (eigvals >= eigenvalue_cut * max_eig)
    if np.any(positive) and not np.any(keep):
        keep[np.flatnonzero(positive)[0]] = True

    delta_modes = delta @ eigvecs
    raw_mode_chi2 = np.zeros_like(eigvals, dtype=float)
    valid = eigvals > 0.0
    raw_mode_chi2[valid] = hartlap_factor * delta_modes[valid] ** 2 / eigvals[valid]
    svd_mode_chi2 = np.where(keep, raw_mode_chi2, 0.0)

    max_loading_idx = np.argmax(np.abs(eigvecs), axis=0)
    mode_df = pd.DataFrame(
        {
            "mode_index": np.arange(len(eigvals)),
            "eigenvalue": eigvals,
            "relative_eigenvalue": eigvals / max_eig if np.isfinite(max_eig) else np.nan,
            "kept_by_svd_cut": keep,
            "delta_projection": delta_modes,
            "raw_chi2_contribution": raw_mode_chi2,
            "svd_chi2_contribution": svd_mode_chi2,
            "abs_delta_projection": np.abs(delta_modes),
            "dominant_theta_deg": theta[max_loading_idx],
            "dominant_loading": eigvecs[max_loading_idx, np.arange(len(eigvals))],
        }
    ).sort_values("raw_chi2_contribution", ascending=False).reset_index(drop=True)

    mode_df["cumulative_raw_chi2_contribution"] = mode_df["raw_chi2_contribution"].cumsum()
    total_raw_mode_chi2 = mode_df["raw_chi2_contribution"].sum()
    mode_df["fractional_raw_chi2_contribution"] = (
        mode_df["raw_chi2_contribution"] / total_raw_mode_chi2
        if total_raw_mode_chi2 > 0
        else 0.0
    )
    total_svd_mode_chi2 = mode_df["svd_chi2_contribution"].sum()
    mode_df["fractional_svd_chi2_contribution"] = (
        mode_df["svd_chi2_contribution"] / total_svd_mode_chi2
        if total_svd_mode_chi2 > 0
        else 0.0
    )

    return bin_df, mode_df


# ==============================================================
# 21. INTERSURVEY SPATIAL CORRELATION ANALYSIS
# ==============================================================

# Quantify angular correlations and sky overlap between observational surveys
def analyze_intersurvey_correlations(
    df_data: pd.DataFrame,
    radius_deg: float = OVERLAP_RADIUS_DEG,
    nside: int = OVERLAP_NSIDE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    surveys = split_by_survey(df_data)
    survey_names = list(surveys.keys())
    n_surveys = len(survey_names)

    corr_matrix = np.zeros((n_surveys, n_surveys))
    overlap_fractions = np.zeros((n_surveys, n_surveys))

    coords_by_survey = {
        name: SkyCoord(
            ra=subdf["RA"].values * u.degree,
            dec=subdf["DEC"].values * u.degree,
            frame="icrs",
        )
        for name, subdf in surveys.items()
    }

    for i, survey1 in enumerate(survey_names):
        data1 = surveys[survey1][["RA", "DEC"]].values
        theta1 = np.radians(90.0 - data1[:, 1])
        phi1 = np.radians(data1[:, 0])
        pix1 = hp.ang2pix(nside, theta1, phi1)
        hist1 = np.bincount(pix1, minlength=hp.nside2npix(nside))

        for j, survey2 in enumerate(survey_names):
            if i == j:
                corr_matrix[i, j] = 1.0
                overlap_fractions[i, j] = 1.0
                continue

            data2 = surveys[survey2][["RA", "DEC"]].values
            theta2 = np.radians(90.0 - data2[:, 1])
            phi2 = np.radians(data2[:, 0])
            pix2 = hp.ang2pix(nside, theta2, phi2)
            hist2 = np.bincount(pix2, minlength=hp.nside2npix(nside))

            corr = np.corrcoef(hist1, hist2)[0, 1]
            corr_matrix[i, j] = corr if np.isfinite(corr) else 0.0

            _, sep2d, _ = coords_by_survey[survey1].match_to_catalog_sky(
                coords_by_survey[survey2]
            )
            overlap_fractions[i, j] = float(np.mean(sep2d < radius_deg * u.degree))

    corr_df = pd.DataFrame(corr_matrix, index=survey_names, columns=survey_names)
    overlap_df = pd.DataFrame(
        overlap_fractions * 100.0,
        index=survey_names,
        columns=survey_names,
    )

    print("\n--- Spatial Correlation Matrix (Pearson) ---")
    print(corr_df.round(3))
    print(f"\n--- Overlap Fractions (% within {radius_deg:.1f} deg) ---")
    print(overlap_df.round(1))

    return corr_df, overlap_df


# ==============================================================
# 22. SELECTION FUNCTION VALIDATION AND VISUAL DIAGNOSTICS
# ==============================================================

# Perform numerical and visual validation of survey selection functions
class SelectionFunctionValidator:
    """Numerical and visual sanity checks for survey selection functions."""

    def __init__(
        self,
        sf_dict: dict[str, np.ndarray],
        survey_weights: dict[str, float],
        nside: int,
    ) -> None:
        self.sf_dict = sf_dict
        self.survey_weights = survey_weights
        self.nside = nside

    def check_coverage(self) -> pd.DataFrame:
        rows = []
        print("\n--- Selection Function Validation ---")

        for survey_name, sf in self.sf_dict.items():
            max_pix = int(np.argmax(sf))
            theta_max, phi_max = hp.pix2ang(self.nside, max_pix)
            dec_max = 90.0 - np.degrees(theta_max)
            ra_max = np.degrees(phi_max)

            active = sf > (np.max(sf) * 1e-3)
            coverage = np.mean(active) * 100.0
            entropy = -np.sum(sf[sf > 0] * np.log2(sf[sf > 0]))

            rows.append(
                {
                    "survey": survey_name,
                    "weight": self.survey_weights[survey_name],
                    "peak_ra": ra_max,
                    "peak_dec": dec_max,
                    "active_coverage_pct": coverage,
                    "entropy_bits": entropy,
                }
            )

            print(f"\n{survey_name}:")
            print(f"  Weight: {self.survey_weights[survey_name]:.2%}")
            print(f"  Peak density at: RA={ra_max:.1f} deg, DEC={dec_max:.1f} deg")
            print(f"  Active coverage: {coverage:.1f}%")
            print(f"  Entropy: {entropy:.2f} bits")

            if "CHIME" in survey_name.upper() and dec_max < 20.0:
                print("  WARNING: CHIME peak is unexpectedly far south.")
            if "PARKES" in survey_name.upper() and dec_max > 0.0:
                print("  WARNING: PARKES peak is unexpectedly far north.")

        return pd.DataFrame(rows)

    def plot_sf(
        self,
        save_prefix: str = "SF_validation_level4",
        max_surveys: int = 12,
        # cmap: str = "viridis",
    ) -> None:

        ordered = sorted(
            self.sf_dict.keys(),
            key=lambda name: self.survey_weights[name],
            reverse=True,
        )[:max_surveys]

        ncols = min(4, len(ordered))
        nrows = int(np.ceil(len(ordered) / ncols))

        # ==========================================================
        # EQUATORIAL MAPS
        # ==========================================================

        plt.figure(figsize=(5 * ncols, 3.8 * nrows))

        for i, name in enumerate(ordered, start=1):

            hp.mollview(
                self.sf_dict[name],
                title=f"{name} (w={self.survey_weights[name]:.1%})",
                sub=(nrows, ncols, i),
                cbar=True,
            )

        plt.savefig(
            f"{save_prefix}_equatorial.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()

        # ==========================================================
        # GALACTIC MAPS
        # ==========================================================

        plt.figure(figsize=(5 * ncols, 3.8 * nrows))

        for i, name in enumerate(ordered, start=1):

            sf_gal = rotate_healpix_map_to_galactic(
                self.sf_dict[name],
                coord_in="C",
                coord_out="G",
            )

            hp.mollview(
                sf_gal,
                title=f"{name} (w={self.survey_weights[name]:.1%})",
                sub=(nrows, ncols, i),
                cbar=True,
            )

        plt.savefig(
            f"{save_prefix}_galactic.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()


# ==============================================================
# 23. ISOTROPIC MOCK REALIZATION PIPELINE
# ==============================================================

# Generate one complete H0 isotropic realization and its angular statistics
def run_one_mock(
    seed: int,
    n_data: int,
    sf_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
) -> tuple[np.ndarray, np.ndarray]:
    d_iso = generate_mixture_catalog_improved(
        n_data,
        sf_dict,
        survey_weights,
        nside,
        seed=seed,
    )
    r_iso = generate_mixture_catalog_improved(
        n_data * N_RAND_FACTOR,
        sf_dict,
        survey_weights,
        nside,
        seed=seed + 5000,
    )
    theta_h0, w_h0 = compute_2pacf(d_iso, r_iso, log_spacing=USE_LOG)
    return w_h0, get_absolute_sum(theta_h0, w_h0)


# ==============================================================
# 24. H0 ENSEMBLE MONTE CARLO GENERATION
# ==============================================================

# Generate the full isotropic mock ensemble including SF uncertainty realizations
def run_ensemble_mocks(
    df_data: pd.DataFrame,
    sf_dict_nominal: dict[str, np.ndarray],
    sf_cov_diag_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    n_ensemble: int = N_ENSEMBLE,
    n_mocks_per: int = N_MOCKS_PER_ENSEMBLE,
    perturbation_scale: float = PERTURBATION_SCALE,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    tasks = []
    for i_ens in range(n_ensemble):
        sf_variant = generate_sf_variant(
            sf_dict_nominal,
            sf_cov_diag_dict,
            perturbation_scale=perturbation_scale,
            seed=i_ens,
        )
        for i_mock in range(n_mocks_per):
            tasks.append((i_ens * 1000 + i_mock, sf_variant))

    print(
        f"\n--- Generating H0 ensemble mocks "
        f"({n_ensemble} SF variants x {n_mocks_per} mocks) ---"
    )
    with tqdm_joblib(tqdm(desc="H0 ensemble mocks", total=len(tasks))):
        results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=5)(
            delayed(run_one_mock)(
                seed,
                len(df_data),
                sf_variant,
                survey_weights,
                nside,
            )
            for seed, sf_variant in tasks
        )

    all_w_h0, all_abs_h0 = zip(*results)
    return np.array(all_w_h0), np.array(all_abs_h0)


# ==============================================================
# 25. GLOBAL ISOTROPY INFERENCE AND STATISTICAL DIAGNOSTICS
# ==============================================================

# Perform the full covariance-aware isotropy analysis against the H0 ensemble
def compute_statistics(
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
    label: str = "Level 4: jackknife errors",
) -> TestStatistics:

    # ==========================================================
    # H0 mean
    # ==========================================================

    h0_mean = np.mean(all_w_h0, axis=0)

    print(f"Max |H0 mean| = {np.max(np.abs(h0_mean)):.4e}")

    # ==========================================================
    # Shrinkage covariance (Ledoit-Wolf)
    # ==========================================================

    lw = LedoitWolf()
    lw.fit(all_w_h0)

    cov = lw.covariance_

    # ==========================================================
    # Covariance eigenvalue spectrum
    # ==========================================================

    eigvals = np.linalg.eigvalsh(cov)

    eigvals_pos = eigvals[eigvals > 0]

    pd.DataFrame({
        "eigenvalue": np.sort(eigvals_pos)[::-1]
    }).to_csv(
        "covariance_eigenvalues.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))

    plt.semilogy(
        np.sort(eigvals_pos)[::-1],
        marker="o",
    )

    plt.xlabel("Mode")
    plt.ylabel("Eigenvalue")
    plt.title("Covariance Eigenvalue Spectrum")
    plt.grid(True)

    plt.savefig(
        "covariance_eigenspectrum.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    # ==========================================================
    # Effective number of modes
    # ==========================================================

    n_eff = (
        (eigvals_pos.sum() ** 2)
        / np.sum(eigvals_pos ** 2)
    )

    print(f"Effective number of modes = {n_eff:.2f}")

    # ==========================================================
    # Covariance diagnostics
    # ==========================================================

    n_mocks, n_bins = all_w_h0.shape

    covariance_rank = int(np.linalg.matrix_rank(cov))

    covariance_condition = float(np.linalg.cond(cov))

    hartlap_factor = (
        (n_mocks - n_bins - 2)
        / (n_mocks - 1)
    )

    hartlap_factor = (
        hartlap_factor
        if hartlap_factor > 0
        else 1.0
    )

    # ==========================================================
    # Inverse covariance
    # ==========================================================

    inv_cov = hartlap_factor * np.linalg.inv(cov)

    # ==========================================================
    # Data residuals
    # ==========================================================

    delta = w_obs - h0_mean

    mock_delta = all_w_h0 - h0_mean

    # ==========================================================
    # SVD-regularized chi-square
    # ==========================================================

    svd_stats = compute_svd_regularized_chi2(
        delta,
        mock_delta,
        cov,
        hartlap_factor,
        eigenvalue_cut=SVD_EIGENVALUE_CUT,
    )

    # ==========================================================
    # Degrees of freedom
    # ==========================================================

    dof = int(svd_stats["modes_kept"])

    # ==========================================================
    # Full chi-square
    # ==========================================================

    chi2 = float(delta @ inv_cov @ delta)

    chi2_red = chi2 / dof

    p_chi2 = float(
        chi2_dist.sf(chi2, dof)
    )

    # ==========================================================
    # Empirical chi-square p-value
    # ==========================================================

    chi2_mocks = np.einsum(
        "ij,jk,ik->i",
        mock_delta,
        inv_cov,
        mock_delta,
    )

    n_chi2_extreme = int(
        np.sum(chi2_mocks >= chi2)
    )

    p_empirical_floor = (
        1.0 / (len(chi2_mocks) + 1)
    )

    p_empirical = float(
        (n_chi2_extreme + 1)
        / (len(chi2_mocks) + 1)
    )

    sigma_equiv = sigma_equivalent_from_p(
        p_empirical
    )

    # ==========================================================
    # Global normalized tension
    # ==========================================================

    diag_sigma = (
        np.sqrt(np.diag(cov))
        + 1e-12
    )

    global_tension = float(
        np.sqrt(
            np.mean(
                (delta / diag_sigma) ** 2
            )
        )
    )

    # ==========================================================
    # Absolute anisotropy amplitude
    # ==========================================================

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
                abs_mock_stats >= abs_observed_stat
            ) + 1
        )
        / (len(abs_mock_stats) + 1)
    )

    # ==========================================================
    # Non-parametric tests
    # ==========================================================

    nonparam_stats = compute_nonparametric_tests(
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
    )

    # ==========================================================
    # Print summary
    # ==========================================================

    print(f"\n--- Full covariance diagnostic ({label}) ---")

    print(f"Chi2 = {chi2:.3f}")

    print(f"Chi2/dof = {chi2_red:.3f}")

    print(f"p-value (Chi2 analytic) = {p_chi2:.4e}")

    print(f"p-value (Chi2 empirical) = {p_empirical:.4e}")

    if n_chi2_extreme == 0:
        print(
            "p-value (Chi2 empirical) is at the Monte Carlo floor: "
            f"p <= {p_empirical_floor:.4e}"
        )

    print(
        f"Sigma-equivalent (from empirical p) = "
        f"{sigma_equiv:.2f}"
    )

    # ==========================================================
    # SVD statistics
    # ==========================================================

    print("\n--- Primary isotropy statistic (SVD-regularized) ---")

    print(
        f"Eigenvalue cut = "
        f"{svd_stats['eigenvalue_cut']:.1e}"
    )

    print(
        f"SVD modes kept = "
        f"{int(svd_stats['modes_kept'])}/{n_bins}"
    )

    print(f"Chi2 SVD = {svd_stats['chi2']:.3f}")

    print(
        f"Chi2 SVD/dof = "
        f"{svd_stats['chi2_red']:.3f}"
    )

    print(
        f"p-value SVD (Chi2 analytic) = "
        f"{svd_stats['p_chi2']:.4e}"
    )

    print(
        f"p-value SVD (empirical) = "
        f"{svd_stats['p_empirical']:.4e}"
    )

    if svd_stats["n_extreme_mocks"] == 0:
        print(
            "p-value SVD (empirical) is at the Monte Carlo floor: "
            f"p <= {svd_stats['p_empirical_floor']:.4e}"
        )

    print(
        f"Sigma-equivalent SVD (from empirical p) = "
        f"{svd_stats['sigma_equiv']:.2f}"
    )

    # ==========================================================
    # Covariance diagnostics
    # ==========================================================

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
        f"Effective modes = {n_eff:.2f}"
    )

    print(
        f"RMS normalized deviation = "
        f"{global_tension:.3f}"
    )

    print(
        f"Hartlap factor = "
        f"{hartlap_factor:.3f}"
    )

    # ==========================================================
    # Non-parametric tests
    # ==========================================================

    print("\n--- Non-parametric profile tests ---")

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

    # ==========================================================
    # Absolute anisotropy
    # ==========================================================

    print("\n--- Absolute anisotropy amplitude ---")

    print(
        f"Observed RMS absolute amplitude = "
        f"{abs_observed_stat:.4e}"
    )

    print(
        f"Empirical p-value = "
        f"{abs_empirical_p:.4e}"
    )

    # ==========================================================
    # Return dataclass
    # ==========================================================

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
        svd_modes_kept=int(svd_stats["modes_kept"]),
        svd_eigenvalue_cut=svd_stats["eigenvalue_cut"],
        svd_condition=svd_stats["condition"],
        n_mocks=n_mocks,
        n_bins=n_bins,
    )


# ==============================================================
# 26. JACKKNIFE SKY REGION ASSIGNMENT
# ==============================================================

# Assign observed FRBs to coarse HEALPix jackknife regions
def assign_jackknife_regions(
    df_data: pd.DataFrame,
    nside_jackknife: int = NSIDE_JACKKNIFE,
    min_regions: int = MIN_JACKKNIFE_REGIONS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Assign each observed FRB to a coarse HEALPix jackknife region."""
    df = df_data.copy()
    theta = np.radians(90.0 - df["DEC"].values)
    phi = np.radians(df["RA"].values)
    region_pix = hp.ang2pix(nside_jackknife, theta, phi)
    unique_regions, region_ids = np.unique(region_pix, return_inverse=True)

    if len(unique_regions) < min_regions:
        raise ValueError(
            f"Only {len(unique_regions)} jackknife regions are populated. "
            f"Decrease MIN_JACKKNIFE_REGIONS or increase NSIDE_JACKKNIFE."
        )

    df["jackknife_region"] = region_ids
    return df, np.arange(len(unique_regions))


# ==============================================================
# 27. SINGLE JACKKNIFE RESAMPLING REALIZATION
# ==============================================================

# Compute one leave-one-region-out jackknife realization
def _run_one_jackknife_region(
    region: int,
    df_data_with_regions: pd.DataFrame,
    sf_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    df_jk = df_data_with_regions[
        df_data_with_regions["jackknife_region"] != region
    ].reset_index(drop=True)

    df_rand_jk = generate_mixture_catalog_improved(
        n=len(df_jk) * N_RAND_FACTOR,
        sf_dict=sf_dict,
        weights=survey_weights,
        nside=nside,
        seed=seed_base + int(region),
    )
    theta_jk, w_jk = compute_2pacf(df_jk, df_rand_jk, log_spacing=USE_LOG)
    abs_jk = get_absolute_sum(theta_jk, w_jk)
    return theta_jk, w_jk, abs_jk, len(df_jk)


# ==============================================================
# 28. JACKKNIFE COVARIANCE ESTIMATION
# ==============================================================

# Compute jackknife mean, covariance matrix, and statistical uncertainties
def jackknife_covariance(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return jackknife mean, covariance, and 1-sigma errors."""
    samples = np.asarray(samples, dtype=float)
    n_regions = samples.shape[0]
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    cov = (n_regions - 1) / n_regions * centered.T @ centered
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return mean, cov, err


# ==============================================================
# 29. JACKKNIFE UNCERTAINTY PIPELINE
# ==============================================================

# Estimate statistical uncertainties using leave-one-region-out jackknife resampling
def run_jackknife_errors(
    df_data: pd.DataFrame,
    sf_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    nside_jackknife: int = NSIDE_JACKKNIFE,
    min_regions: int = MIN_JACKKNIFE_REGIONS,
    n_jobs: int = -1,
    seed_base: int = 9000,
) -> JackknifeResult:
    """Compute leave-one-region-out jackknife errors for w(theta) and |<w>|."""
    df_regions, regions = assign_jackknife_regions(
        df_data,
        nside_jackknife=nside_jackknife,
        min_regions=min_regions,
    )

    counts = df_regions["jackknife_region"].value_counts().sort_index()
    print("\n--- Jackknife regions ---")
    print(f"NSIDE_JACKKNIFE = {nside_jackknife}")
    print(f"Populated regions = {len(regions)}")
    print(f"Objects per region: min={counts.min()}, median={counts.median():.1f}, max={counts.max()}")

    with tqdm_joblib(tqdm(desc="Jackknife regions", total=len(regions))):
        results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=2)(
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

    theta_samples, w_samples, abs_samples, n_kept = zip(*results)
    theta = np.mean(np.array(theta_samples), axis=0)
    w_samples = np.array(w_samples)
    abs_samples = np.array(abs_samples)

    w_mean, w_cov, w_err = jackknife_covariance(w_samples)
    abs_mean, abs_cov, abs_err = jackknife_covariance(abs_samples)

    print("\n--- Jackknife summary ---")
    print(f"Leave-one-region samples = {len(regions)}")
    print(f"Objects kept per sample: min={min(n_kept)}, max={max(n_kept)}")
    print(f"Median sigma[w(theta)] = {np.median(w_err):.4e}")
    print(f"Median sigma[|<w>|] = {np.median(abs_err):.4e}")

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


# ==============================================================
# 30. SENSITIVITY-MARGINALIZED H0 MOCK REALIZATION
# ==============================================================

# Generate one isotropic mock realization including SF perturbation uncertainty
def _run_one_sensitivity_mock(
    seed: int,
    n_data: int,
    sf_dict: dict[str, np.ndarray],
    sf_cov_diag_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
) -> tuple[np.ndarray, np.ndarray]:
    sf_variant = generate_sf_variant(
        sf_dict,
        sf_cov_diag_dict,
        perturbation_scale=PERTURBATION_SCALE,
        seed=seed,
    )
    d_iso = generate_mixture_catalog_improved(
        n=n_data,
        sf_dict=sf_variant,
        weights=survey_weights,
        nside=nside,
        seed=seed + 1000,
    )
    r_iso = generate_mixture_catalog_improved(
        n=n_data * N_RAND_FACTOR,
        sf_dict=sf_variant,
        weights=survey_weights,
        nside=nside,
        seed=seed + 5000,
    )
    theta_h0, w_h0 = compute_2pacf(d_iso, r_iso, log_spacing=USE_LOG)
    return w_h0, get_absolute_sum(theta_h0, w_h0)


# ==============================================================
# 31. EMPIRICAL SENSITIVITY STATISTICS
# ==============================================================

# Compute isotropy statistics for one sensitivity-analysis configuration
def _empirical_sensitivity_stats(
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
) -> dict[str, float]:

    
    # H0 mean
    h0_mean = np.mean(all_w_h0, axis=0)

    # Shrinkage covariance (Ledoit-Wolf)
    lw = LedoitWolf()
    lw.fit(all_w_h0)
    cov = lw.covariance_

    # Covariance eigenvalues
    eigvals = np.linalg.eigvalsh(cov)
    eigvals_pos = eigvals[eigvals > 0]
    n_eff = (
        (eigvals_pos.sum() ** 2)
        / np.sum(eigvals_pos ** 2)
    )

    # Covariance diagnostics
    n_mocks, n_bins = all_w_h0.shape

    # Effective rank of the covariance matrix
    covariance_rank = int(
        np.linalg.matrix_rank(cov)
    )

    # Covariance matrix condition number
    covariance_condition = float(
        np.linalg.cond(cov)
    )

    # Hartlap correction factor for finite-mock covariance inversion
    hartlap_factor = (
        (n_mocks - n_bins - 2)
        / (n_mocks - 1)
    )

    # Prevent nonphysical Hartlap corrections
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

    # Residual profile between observed data and isotropic H0 expectation
    delta = w_obs - h0_mean

    # Residual profiles for all isotropic H0 mock realizations
    mock_delta = all_w_h0 - h0_mean

    # SVD-regularized chi-square
    svd_stats = compute_svd_regularized_chi2(
        delta,
        mock_delta,
        cov,
        hartlap_factor,
        eigenvalue_cut=SVD_EIGENVALUE_CUT,
    )

    # Effective degrees of freedom
    dof = int(svd_stats["modes_kept"])

    # Full chi-square
    chi2 = float(
        delta @ inv_cov @ delta
    )

    # Reduced chi-square
    chi2_red = chi2 / dof

    # Analytic chi-square p-value
    p_chi2 = float(
        chi2_dist.sf(chi2, dof)
    )

    # Chi-square values for all isotropic H0 mocks
    chi2_mocks = np.einsum(
        "ij,jk,ik->i",
        mock_delta,
        inv_cov,
        mock_delta,
    )

    # Number of mocks more extreme than the observed data
    n_chi2_extreme = int(
        np.sum(chi2_mocks >= chi2)
    )

    # Minimum empirical p-value allowed by the finite mock ensemble
    p_empirical_floor = (
        1.0 / (len(chi2_mocks) + 1)
    )

    # Empirical chi-square p-value calibrated from the H0 mocks
    p_empirical = float(
        (n_chi2_extreme + 1)
        / (len(chi2_mocks) + 1)
    )

    # Gaussian sigma-equivalent corresponding to the empirical p-value
    sigma_empirical = sigma_equivalent_from_p(
        p_empirical
    )

    # Absolute anisotropy statistic
    abs_obs_stat = absolute_global_stat(
        abs_obs
    )

    # Global absolute anisotropy amplitudes for all H0 mocks
    abs_mock_stats = np.array(
        [
            absolute_global_stat(row)
            for row in all_abs_h0
        ]
    )

    # Empirical p-value for the absolute anisotropy statistic
    p_abs_empirical = float(
        (
            np.sum(
                abs_mock_stats >= abs_obs_stat
            ) + 1
        )
        / (len(abs_mock_stats) + 1)
    )

    # Non-parametric tests
    nonparam_stats = compute_nonparametric_tests(
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
    )

    # Return diagnostics
    return {
        "chi2": chi2,
        "chi2_red": chi2_red,
        "p_chi2_analytic": p_chi2,
        "p_chi2_empirical": p_empirical,
        "p_chi2_empirical_floor": p_empirical_floor,
        "n_chi2_extreme_mocks": float(n_chi2_extreme),
        "sigma_empirical": sigma_empirical,
        "chi2_svd": svd_stats["chi2"],
        "chi2_svd_red": svd_stats["chi2_red"],
        "p_chi2_svd_analytic": svd_stats["p_chi2"],
        "p_chi2_svd_empirical": svd_stats["p_empirical"],
        "p_chi2_svd_empirical_floor": svd_stats["p_empirical_floor"],
        "n_chi2_svd_extreme_mocks": svd_stats["n_extreme_mocks"],
        "sigma_svd_empirical": svd_stats["sigma_equiv"],
        "svd_modes_kept": svd_stats["modes_kept"],
        "svd_eigenvalue_cut": svd_stats["eigenvalue_cut"],
        "svd_condition": svd_stats["condition"],
        "effective_modes": n_eff,
        "abs_observed_stat": abs_obs_stat,
        "p_abs_empirical": p_abs_empirical,
        **nonparam_stats,
        "hartlap_factor": hartlap_factor,
        "n_mocks": float(n_mocks),
        "n_bins": float(n_bins),
        "covariance_rank": float(covariance_rank),
        "covariance_condition": covariance_condition,
    }


# ==============================================================
# 32. SELECTION FUNCTION SENSITIVITY ANALYSIS
# ==============================================================

# Evaluate the robustness of isotropy results against SF modeling choices
def run_empirical_sensitivity_analysis(
    df_data: pd.DataFrame,
    nside_range: list[int] = NSIDE_SF_RANGE,
    smooth_sigma_range: list[float] = SMOOTH_SIGMA_RANGE,
    n_mocks: int = N_SENSITIVITY_EMPIRICAL_MOCKS,
    n_jobs: int = -1,
    output_csv: str = "Level4_sensitivity_empirical.csv",
) -> pd.DataFrame:
    """
    Sensitivity analysis calibrated with mocks.

    This is slower than the Level 3 diagnostic, but it reports empirical
    p-values for each (NSIDE, smooth_sigma) pair instead of relying only on
    an analytic chi-square approximation.
    """
    rows = []
    print("\n--- Level 4 empirical sensitivity analysis ---")
    print(f"Mocks per (NSIDE, smooth_sigma) pair = {n_mocks}")

    for nside in nside_range:
        for smooth_sigma in smooth_sigma_range:
            print(f"\nTesting NSIDE={nside}, smooth_sigma={smooth_sigma:.1f} deg...")
            sf_dict, sf_cov_diag_dict, survey_weights, sf_nside = (
                build_survey_selection_functions_improved(
                    df_data,
                    nside=nside,
                    smooth_sigma=smooth_sigma,
                )
            )

            df_rand_obs = generate_mixture_catalog_improved(
                n=len(df_data) * N_RAND_FACTOR,
                sf_dict=sf_dict,
                weights=survey_weights,
                nside=sf_nside,
                seed=2000 + nside + int(10 * smooth_sigma),
            )
            theta_obs, w_obs = compute_2pacf(df_data, df_rand_obs, log_spacing=USE_LOG)
            abs_obs = get_absolute_sum(theta_obs, w_obs)

            seeds = [
                100000 + 1000 * nside + 10 * int(smooth_sigma) + i
                for i in range(n_mocks)
            ]
            with tqdm_joblib(tqdm(desc="Sensitivity mocks", total=n_mocks)):
                mock_results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=5)(
                    delayed(_run_one_sensitivity_mock)(
                        seed,
                        len(df_data),
                        sf_dict,
                        sf_cov_diag_dict,
                        survey_weights,
                        sf_nside,
                    )
                    for seed in seeds
                )

            all_w_h0, all_abs_h0 = zip(*mock_results)
            stats = _empirical_sensitivity_stats(
                w_obs,
                abs_obs,
                np.array(all_w_h0),
                np.array(all_abs_h0),
            )
            row = {
                "nside": nside,
                "smooth_sigma_deg": smooth_sigma,
                **stats,
            }
            rows.append(row)

            print(
                "  "
                f"chi2/dof={row['chi2_red']:.3f}, "
                f"p_emp={row['p_chi2_empirical']:.4e}, "
                f"p_svd={row['p_chi2_svd_empirical']:.4e}, "
                f"modes={int(row['svd_modes_kept'])}/{int(row['n_bins'])}, "
                f"sigma_emp={row['sigma_empirical']:.2f}, "
                f"p_abs={row['p_abs_empirical']:.4e}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)
    print("\n--- Empirical sensitivity summary ---")
    print(results.round(4))
    print(f"Saved: {output_csv}")
    return results


# ==============================================================
# 33. REPORT VALUE FORMATTING
# ==============================================================

# Format numerical values for human-readable report output
def _format_report_value(value: object, precision: int = 6) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{float(value):.{precision}g}"
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


# ==============================================================
# 34. REPORT KEY-VALUE SECTION BUILDER
# ==============================================================

# Append a formatted key-value diagnostics block to the report text
def _append_key_values(
    lines: list[str],
    title: str,
    values: dict[str, object],
    precision: int = 6,
) -> None:
    lines.append(f"## {title}")
    lines.append("")
    max_key = max(len(key) for key in values)
    lines.append("```text")
    for key, value in values.items():
        lines.append(f"{key:<{max_key}} : {_format_report_value(value, precision)}")
    lines.append("```")
    lines.append("")


# ==============================================================
# 35. INTERSURVEY PAIRWISE RANKING
# ==============================================================

# Extract the strongest pairwise survey correlations or overlaps
def _top_matrix_pairs(
    matrix: pd.DataFrame,
    n: int = 10,
    absolute: bool = True,
) -> pd.DataFrame:
    rows = []
    columns = list(matrix.columns)
    for i, row_name in enumerate(matrix.index):
        for j, col_name in enumerate(columns):
            if j <= i:
                continue
            value = float(matrix.iloc[i, j])
            rows.append(
                {
                    "survey_1": row_name,
                    "survey_2": col_name,
                    "value": value,
                    "abs_value": abs(value),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["survey_1", "survey_2", "value"])

    df_pairs = pd.DataFrame(rows)
    sort_col = "abs_value" if absolute else "value"
    ascending = False
    return (
        df_pairs.sort_values(sort_col, ascending=ascending)
        .head(n)
        [["survey_1", "survey_2", "value"]]
        .reset_index(drop=True)
    )


# ==============================================================
# 36. LEVEL 4 REPORT GENERATION
# ==============================================================

# Generate a consolidated human-readable report of the full Level 4 analysis
def save_level4_report(
    results: dict[str, object],
    paths: dict[str, str] | None = None,
    output_path: str = "Level4_report.md",
) -> str:
    """Save a single human-readable report with the main Level 4 diagnostics."""
    stats = results["stats"]
    jackknife = results["jackknife"]
    df_data = results["df_data"]
    all_w_h0 = results["all_w_h0"]
    all_abs_h0 = results["all_abs_h0"]
    chi2_bin_diagnostics = results.get("chi2_bin_diagnostics")
    chi2_svd_modes = results.get("chi2_svd_modes")

    lines: list[str] = []
    lines.append("# Level 4 FRB Isotropy Report")
    lines.append("")
    lines.append("This report consolidates the main text outputs from the Level 4 run.")
    lines.append("")

    _append_key_values(
        lines,
        "Run Configuration",
        {
            "catalog_path": ALL_FRB_PATH,
            "masked_catalog_size": len(df_data),
            "bin_type": BIN_TYPE,
            "n_bins": stats.n_bins,
            "bin_size_deg": BIN_SIZE if not USE_LOG else "log",
            "n_rand_factor": N_RAND_FACTOR,
            "n_mocks": stats.n_mocks,
            "n_ensemble": N_ENSEMBLE,
            "n_mocks_per_ensemble": N_MOCKS_PER_ENSEMBLE,
            "nside_sf": NSIDE_SF,
            "smooth_sigma_deg": SMOOTH_SIGMA,
            "perturbation_scale": PERTURBATION_SCALE,
            "nside_jackknife": NSIDE_JACKKNIFE,
            "svd_eigenvalue_cut": stats.svd_eigenvalue_cut,
        },
    )

    _append_key_values(
        lines,
        "Full Covariance Diagnostic",
        {
            "chi2": stats.chi2,
            "chi2_red": stats.chi2_red,
            "p_chi2_analytic": stats.p_chi2,
            "p_chi2_empirical": stats.p_empirical,
            "p_chi2_empirical_floor": stats.p_empirical_floor,
            "p_chi2_at_mc_floor": np.isclose(
                stats.p_empirical,
                stats.p_empirical_floor,
            ),
            "sigma_equiv_empirical": stats.sigma_equiv,
            "hartlap_factor": stats.hartlap_factor,
            "rms_normalized_deviation": stats.global_tension,
        },
    )

    _append_key_values(
        lines,
        "Primary Isotropy Statistic (SVD-Regularized)",
        {
            "chi2_svd": stats.chi2_svd,
            "chi2_svd_red": stats.chi2_svd_red,
            "p_chi2_svd_analytic": stats.p_chi2_svd,
            "p_chi2_svd_empirical": stats.p_svd_empirical,
            "p_chi2_svd_empirical_floor": stats.p_svd_empirical_floor,
            "p_chi2_svd_at_mc_floor": np.isclose(
                stats.p_svd_empirical,
                stats.p_svd_empirical_floor,
            ),
            "sigma_svd_empirical": stats.sigma_svd_equiv,
            "svd_modes_kept": f"{stats.svd_modes_kept}/{stats.n_bins}",
            "svd_retained_condition": stats.svd_condition,
        },
    )

    _append_key_values(
        lines,
        "Covariance Diagnostics",
        {
            "mocks": stats.n_mocks,
            "bins": stats.n_bins,
            "covariance_rank": f"{stats.covariance_rank}/{stats.n_bins}",
            "covariance_condition": stats.covariance_condition,
            "effective_modes": stats.n_eff,
            "h0_w_shape": tuple(all_w_h0.shape),
            "h0_abs_shape": tuple(all_abs_h0.shape),
        },
    )

    _append_key_values(
        lines,
        "Non-parametric Profile Tests",
        {
            "w_ks_stat": stats.ks_w_stat,
            "w_ks_pvalue": stats.ks_w_pvalue,
            "w_ks_empirical_p": stats.ks_w_empirical_p,
            "w_ad_stat": stats.ad_w_stat,
            "w_ad_pvalue": stats.ad_w_pvalue,
            "w_ad_empirical_p": stats.ad_w_empirical_p,
            "abs_ks_stat": stats.ks_abs_stat,
            "abs_ks_pvalue": stats.ks_abs_pvalue,
            "abs_ks_empirical_p": stats.ks_abs_empirical_p,
            "abs_ad_stat": stats.ad_abs_stat,
            "abs_ad_pvalue": stats.ad_abs_pvalue,
            "abs_ad_empirical_p": stats.ad_abs_empirical_p,
        },
    )

    _append_key_values(
        lines,
        "Absolute Anisotropy Amplitude",
        {
            "observed_rms_absolute_amplitude": stats.abs_observed_stat,
            "empirical_pvalue": stats.abs_empirical_p,
        },
    )

    _append_key_values(
        lines,
        "Jackknife Summary",
        {
            "n_regions": len(jackknife.regions),
            "median_sigma_w": float(np.median(jackknife.w_err)),
            "min_sigma_w": float(np.min(jackknife.w_err)),
            "max_sigma_w": float(np.max(jackknife.w_err)),
            "median_sigma_abs": float(np.median(jackknife.abs_err)),
            "min_sigma_abs": float(np.min(jackknife.abs_err)),
            "max_sigma_abs": float(np.max(jackknife.abs_err)),
        },
    )

    if chi2_bin_diagnostics is not None:
        lines.append("## Chi-square Bin Audit")
        lines.append("")
        lines.append(
            "Largest diagonal pulls. These are not the full covariance-aware "
            "chi-square contributions, but they show which angular bins look "
            "most individually displaced."
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
            chi2_bin_diagnostics[bin_columns]
            .head(12)
            .to_string(index=False)
        )
        lines.append("```")
        lines.append("")

    if chi2_svd_modes is not None:
        lines.append("## Chi-square SVD Mode Audit")
        lines.append("")
        lines.append(
            "Largest covariance eigenmode contributions to the SVD chi-square. "
            "Modes marked `kept_by_svd_cut=False` are excluded from the "
            "regularized statistic."
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
            chi2_svd_modes[mode_columns]
            .head(12)
            .to_string(index=False)
        )
        lines.append("```")
        lines.append("")

    sf_validation = results.get("sf_validation")
    if sf_validation is not None:
        lines.append("## Selection Function Validation")
        lines.append("")
        lines.append("Top surveys by weight:")
        lines.append("")
        lines.append("```text")
        columns = [
            "survey",
            "weight",
            "peak_ra",
            "peak_dec",
            "active_coverage_pct",
            "entropy_bits",
        ]
        available_columns = [col for col in columns if col in sf_validation.columns]
        sf_top = sf_validation.sort_values("weight", ascending=False).head(12)
        lines.append(sf_top[available_columns].to_string(index=False))
        lines.append("```")
        lines.append("")

    corr_df = results.get("intersurvey_corr")
    if corr_df is not None:
        lines.append("## Intersurvey Correlations")
        lines.append("")
        lines.append("Largest absolute off-diagonal Pearson correlations:")
        lines.append("")
        lines.append("```text")
        lines.append(_top_matrix_pairs(corr_df, n=12, absolute=True).to_string(index=False))
        lines.append("```")
        lines.append("")

    overlap_df = results.get("intersurvey_overlap")
    if overlap_df is not None:
        lines.append("## Intersurvey Overlap")
        lines.append("")
        lines.append("Largest off-diagonal overlap fractions, in percent:")
        lines.append("")
        lines.append("```text")
        lines.append(_top_matrix_pairs(overlap_df, n=12, absolute=False).to_string(index=False))
        lines.append("```")
        lines.append("")

    sensitivity = results.get("sensitivity")
    if sensitivity is not None:
        lines.append("## Empirical Sensitivity Analysis")
        lines.append("")
        lines.append("```text")
        lines.append(sensitivity.to_string(index=False))
        lines.append("```")
        lines.append("")

    if paths:
        lines.append("## Saved Files")
        lines.append("")
        lines.append("```text")
        for name, path in paths.items():
            lines.append(f"{name}: {path}")
        lines.append("```")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as report:
        report.write("\n".join(lines).rstrip() + "\n")

    return output_path


# ==============================================================
# 37. LEVEL 4 TABLE EXPORT PIPELINE
# ==============================================================

def save_level4_tables(
    results: dict[str, object],
    prefix: str = "Level4",
) -> dict[str, str]:
    """
    Save publication/discussion tables from a completed Level 4 run.
    """

    paths = {
        "sf_validation": f"{prefix}_sf_validation.csv",
        "intersurvey_corr": f"{prefix}_intersurvey_corr.csv",
        "intersurvey_overlap": f"{prefix}_intersurvey_overlap.csv",
        "stats_summary": f"{prefix}_stats_summary.csv",
        "jackknife_w_errors": f"{prefix}_jackknife_w_errors.csv",
        "jackknife_abs_errors": f"{prefix}_jackknife_abs_errors.csv",
        "covariance_matrix": f"{prefix}_covariance_matrix.csv",
        "chi2_bin_diagnostics": f"{prefix}_chi2_bin_diagnostics.csv",
        "svd_mode_contributions": f"{prefix}_svd_mode_contributions.csv",
    }

    # ==========================================================
    # Selection-function diagnostics
    # ==========================================================

    results["sf_validation"].to_csv(
        paths["sf_validation"],
        index=False,
    )

    results["intersurvey_corr"].to_csv(
        paths["intersurvey_corr"]
    )

    results["intersurvey_overlap"].to_csv(
        paths["intersurvey_overlap"]
    )

    survey_map_summary = results.get("survey_map_summary")

    if survey_map_summary is not None:

        paths["survey_map_summary"] = (
            f"{prefix}_survey_map_summary.csv"
        )

        survey_map_summary.to_csv(
            paths["survey_map_summary"],
            index=False,
        )

    # ==========================================================
    # Statistics summary
    # ==========================================================

    stats = results["stats"]

    stats_rows = [

        # ======================================================
        # FULL covariance diagnostic
        # ======================================================

        ["chi2", "chi2",
         stats.chi2,
         "Full covariance chi-square"],

        ["chi2", "chi2_red",
         stats.chi2_red,
         "Reduced chi-square"],

        ["chi2", "p_chi2_analytic",
         stats.p_chi2,
         "Analytic chi-square p-value"],

        ["chi2", "p_chi2_empirical",
         stats.p_empirical,
         "Empirical chi-square p-value"],

        ["chi2", "sigma_equiv_empirical",
         stats.sigma_equiv,
         "Gaussian sigma equivalent from empirical p-value"],

        # ======================================================
        # SVD isotropy statistic
        # ======================================================

        ["svd", "chi2_svd",
         stats.chi2_svd,
         "SVD-regularized chi-square"],

        ["svd", "chi2_svd_red",
         stats.chi2_svd_red,
         "Reduced SVD chi-square"],

        ["svd", "p_chi2_svd_analytic",
         stats.p_chi2_svd,
         "Analytic SVD chi-square p-value"],

        ["svd", "p_chi2_svd_empirical",
         stats.p_svd_empirical,
         "Empirical SVD chi-square p-value"],

        ["svd", "sigma_svd_equiv",
         stats.sigma_svd_equiv,
         "Gaussian sigma equivalent from SVD empirical p-value"],

        ["svd", "svd_modes_kept",
         stats.svd_modes_kept,
         "Number of SVD modes retained"],

        # ======================================================
        # Covariance diagnostics
        # ======================================================

        ["covariance", "effective_modes",
         stats.n_eff,
         "Effective covariance dimensionality"],

        ["covariance", "covariance_rank",
         stats.covariance_rank,
         "Covariance matrix rank"],

        ["covariance", "covariance_condition",
         stats.covariance_condition,
         "Covariance condition number"],

        ["covariance", "hartlap_factor",
         stats.hartlap_factor,
         "Hartlap covariance correction factor"],

        ["covariance", "global_tension",
         stats.global_tension,
         "RMS normalized deviation"],

        # ======================================================
        # Absolute anisotropy
        # ======================================================

        ["anisotropy", "abs_observed_stat",
         stats.abs_observed_stat,
         "Observed RMS absolute anisotropy amplitude"],

        ["anisotropy", "abs_empirical_p",
         stats.abs_empirical_p,
         "Empirical p-value for absolute anisotropy"],

        # ======================================================
        # KS / AD — w(theta)
        # ======================================================

        ["nonparametric", "ks_w_stat",
         stats.ks_w_stat,
         "KS statistic for w(theta)"],

        ["nonparametric", "ks_w_pvalue",
         stats.ks_w_pvalue,
         "Analytic KS p-value for w(theta)"],

        ["nonparametric", "ks_w_empirical_p",
         stats.ks_w_empirical_p,
         "Empirical KS p-value for w(theta)"],

        ["nonparametric", "ad_w_stat",
         stats.ad_w_stat,
         "AD statistic for w(theta)"],

        ["nonparametric", "ad_w_pvalue",
         stats.ad_w_pvalue,
         "Analytic AD p-value for w(theta)"],

        ["nonparametric", "ad_w_empirical_p",
         stats.ad_w_empirical_p,
         "Empirical AD p-value for w(theta)"],

        # ======================================================
        # KS / AD — |<w>|
        # ======================================================

        ["nonparametric", "ks_abs_stat",
         stats.ks_abs_stat,
         "KS statistic for |<w>|"],

        ["nonparametric", "ks_abs_pvalue",
         stats.ks_abs_pvalue,
         "Analytic KS p-value for |<w>|"],

        ["nonparametric", "ks_abs_empirical_p",
         stats.ks_abs_empirical_p,
         "Empirical KS p-value for |<w>|"],

        ["nonparametric", "ad_abs_stat",
         stats.ad_abs_stat,
         "AD statistic for |<w>|"],

        ["nonparametric", "ad_abs_pvalue",
         stats.ad_abs_pvalue,
         "Analytic AD p-value for |<w>|"],

        ["nonparametric", "ad_abs_empirical_p",
         stats.ad_abs_empirical_p,
         "Empirical AD p-value for |<w>|"],
    ]

    stats_df = pd.DataFrame(
        stats_rows,
        columns=[
            "category",
            "statistic",
            "value",
            "description",
        ],
    )

    stats_df.to_csv(
        paths["stats_summary"],
        index=False,
    )

    # ==========================================================
    # Jackknife diagnostics
    # ==========================================================

    jackknife = results["jackknife"]

    pd.DataFrame(
        {
            "theta_deg": results["theta"],
            "w_obs": results["w_obs"],
            "w_jackknife_mean": jackknife.w_mean,
            "w_jackknife_err": jackknife.w_err,
        }
    ).to_csv(
        paths["jackknife_w_errors"],
        index=False,
    )

    pd.DataFrame(
        {
            "theta_center_deg": COARSE_CENTERS,
            "abs_obs": results["abs_obs"],
            "abs_jackknife_mean": jackknife.abs_mean,
            "abs_jackknife_err": jackknife.abs_err,
        }
    ).to_csv(
        paths["jackknife_abs_errors"],
        index=False,
    )

    # ==========================================================
    # Chi-square diagnostics
    # ==========================================================

    chi2_bin_diagnostics, svd_mode_contributions = (
        compute_chi2_diagnostic_tables(
            results["theta"],
            results["w_obs"],
            results["all_w_h0"],
            eigenvalue_cut=SVD_EIGENVALUE_CUT,
        )
    )

    results["chi2_bin_diagnostics"] = (
        chi2_bin_diagnostics
    )

    results["svd_mode_contributions"] = (
        svd_mode_contributions
    )

    chi2_bin_diagnostics.to_csv(
        paths["chi2_bin_diagnostics"],
        index=False,
    )

    svd_mode_contributions.to_csv(
        paths["svd_mode_contributions"],
        index=False,
    )

    # ==========================================================
    # Covariance matrix
    # ==========================================================

    cov = results.get("covariance_matrix")

    if cov is not None:

        pd.DataFrame(cov).to_csv(
            paths["covariance_matrix"],
            index=False,
        )

    # ==========================================================
    # Sensitivity analysis
    # ==========================================================

    sensitivity = results.get("sensitivity")

    if sensitivity is not None:

        paths["sensitivity"] = (
            f"{prefix}_sensitivity_empirical.csv"
        )

        sensitivity.to_csv(
            paths["sensitivity"],
            index=False,
        )

    # ==========================================================
    # Markdown report
    # ==========================================================

    paths["report"] = f"{prefix}_report.md"

    save_level4_report(
        results,
        paths=paths,
        output_path=paths["report"],
    )

    # ==========================================================
    # Summary
    # ==========================================================

    print("\n--- Saved Level 4 tables ---")

    for name, path in paths.items():
        print(f"{name}: {path}")

    return paths


# ==============================================================
# 38. MULTI-SURVEY SKY MAP VISUALIZATION
# ==============================================================

def plot_top_survey_maps(
    df_data: pd.DataFrame,
    sf_dict: dict[str, np.ndarray],
    survey_weights: dict[str, float],
    nside: int,
    max_surveys: int = 12,
    output_prefix: str = "Level4_top12_survey_maps",
    cmap: str = "viridis",
) -> pd.DataFrame:
    """
    Save top survey maps in BOTH:
    - Equatorial coordinates
    - Galactic coordinates

    Using a consistent colormap/style.
    """

    surveys = split_by_survey(df_data)

    ordered = sorted(
        sf_dict.keys(),
        key=lambda name: survey_weights[name],
        reverse=True,
    )[:max_surveys]

    rows = []

    # ==========================================================
    # LOOP OVER COORDINATE SYSTEMS
    # ==========================================================

    for coord_system in ["equatorial", "galactic"]:

        ncols = 4
        nrows = int(np.ceil(len(ordered) / ncols))

        fig = plt.figure(
            figsize=(5.0 * ncols, 3.8 * nrows)
        )

        for idx, survey_name in enumerate(ordered, start=1):

            subdf = surveys[survey_name]

            sf = sf_dict[survey_name]

            # ==================================================
            # Coordinate conversion
            # ==================================================

            if coord_system == "galactic":

                sf_plot = rotate_healpix_map_to_galactic(
                    sf,
                    coord_in="C",
                    coord_out="G",
                )

                coords = SkyCoord(
                    ra=subdf["RA"].values * u.degree,
                    dec=subdf["DEC"].values * u.degree,
                    frame="icrs",
                )

                lon = coords.galactic.l.degree
                lat = coords.galactic.b.degree

            else:

                sf_plot = sf

                lon = subdf["RA"].values
                lat = subdf["DEC"].values

            # ==================================================
            # Diagnostics
            # ==================================================

            active_coverage = float(
                np.mean(sf > np.max(sf) * 1e-3) * 100.0
            )

            entropy = float(
                -np.sum(sf[sf > 0] * np.log2(sf[sf > 0]))
            )

            # ==================================================
            # Plot map
            # ==================================================

            hp.mollview(
                sf_plot,
                fig=fig.number,
                sub=(nrows, ncols, idx),
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

            rows.append(
                {
                    "survey": survey_name,
                    "n_objects": len(subdf),
                    "weight": survey_weights[survey_name],
                    "active_coverage_pct": active_coverage,
                    "entropy_bits": entropy,
                    "nside": nside,
                    "coordinate_system": coord_system,
                }
            )

        plt.savefig(
            f"{output_prefix}_{coord_system}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show()

        print(
            f"Saved survey maps: "
            f"{output_prefix}_{coord_system}.png"
        )

    return pd.DataFrame(rows)


# ==============================================================
# 39. ISOTROPY DIAGNOSTIC VISUALIZATION
# ==============================================================

def plot_results_with_jackknife(
    theta: np.ndarray,
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
    jackknife: JackknifeResult,
    output_2pacf: str = "FRB_2PACF_level4_jackknife.png",
    output_abs: str = "FRB_ABS_level4_jackknife.png",
) -> None:
    """Plot Level 3 H0 bands plus Level 4 jackknife error bars."""
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.edgecolor": "0.25",
            "axes.linewidth": 1.1,
            "grid.color": "0.88",
            "grid.linewidth": 0.8,
        },
    )

    h0_mean = np.mean(all_w_h0, axis=0)
    h0_std = np.std(all_w_h0, axis=0)
    h0_abs_mean = np.mean(all_abs_h0, axis=0)
    h0_abs_std = np.std(all_abs_h0, axis=0)

    band_3 = "#d9e6f2"
    band_2 = "#a7c4dd"
    band_1 = "#5f8db3"
    h0_line = "#294c6b"
    obs_color = "firebrick"
    obs_err = "#7f2f2f"

    fig, ax = plt.subplots(figsize=(11, 8))
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
    ax.plot(
        theta,
        h0_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=4,
    )
    ax.errorbar(
        theta,
        w_obs,
        yerr=jackknife.w_err,
        fmt="o-",
        color=obs_color,
        ecolor=obs_err,
        elinewidth=1.1,
        capsize=2.5,
        capthick=1.1,
        lw=2.3,
        ms=5.2,
        mfc=obs_color,
        mec="white",
        mew=0.6,
        label="Observed with jackknife errors",
        zorder=5,
    )
    ax.axhline(0, color="0.15", linewidth=1.1, alpha=0.9, zorder=0)
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$w(\theta)$")
    ax.set_title("Angular Two-Point Correlation Function\n(Level 4: jackknife errors)")
    h0_patch = Patch(
        facecolor=band_2,
        edgecolor=band_1,
        alpha=0.85,
        label=r"$H_0$ (1$\sigma$, 2$\sigma$, 3$\sigma$)",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [h0_patch], frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_2pacf, dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 8))
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
    ax.plot(
        COARSE_CENTERS,
        h0_abs_mean,
        color=h0_line,
        lw=1.8,
        ls="--",
        label=r"$H_0$ mean",
        zorder=4,
    )
    ax.errorbar(
        COARSE_CENTERS,
        abs_obs,
        yerr=jackknife.abs_err,
        fmt="o-",
        color=obs_color,
        ecolor=obs_err,
        elinewidth=1.1,
        capsize=2.5,
        capthick=1.1,
        lw=2.3,
        ms=6.0,
        mfc=obs_color,
        mec="white",
        mew=0.7,
        label="Observed with jackknife errors",
        zorder=5,
    )
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$|\langle w \rangle|$")
    ax.set_title("Absolute Sum Test\n(Level 4: jackknife errors)")
    h0_patch = Patch(
        facecolor=band_2,
        edgecolor=band_1,
        alpha=0.85,
        label=r"$H_0$ (1$\sigma$, 2$\sigma$, 3$\sigma$)",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [h0_patch], frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_abs, dpi=300)
    plt.show()


# ==============================================================
# 40. LEVEL 4 MASTER ANALYSIS PIPELINE
# ==============================================================

# Execute the complete Level 4 FRB isotropy analysis workflow
def main_level4(
    run_sensitivity: bool = False,
    save_tables: bool = True,
    n_jobs: int = -1,
) -> dict[str, object]:
    print(f"\n--- Processing data ({BIN_TYPE}, Level 4) ---")
    df_data = apply_mask(load_catalog(ALL_FRB_PATH))
    print(f"Masked catalog size: {len(df_data)}")

    sf_dict, sf_cov_diag_dict, survey_weights, sf_nside = (
        build_survey_selection_functions_improved(
            df_data,
            nside=NSIDE_SF,
            smooth_sigma=SMOOTH_SIGMA,
        )
    )

    validator = SelectionFunctionValidator(sf_dict, survey_weights, sf_nside)
    sf_validation = validator.check_coverage()
    validator.plot_sf(save_path="SF_validation_level4.png")

    corr_df, overlap_df = analyze_intersurvey_correlations(df_data)

    df_rand_obs = generate_mixture_catalog_improved(
        n=len(df_data) * N_RAND_FACTOR,
        sf_dict=sf_dict,
        weights=survey_weights,
        nside=sf_nside,
        seed=2000,
    )
    theta, w_obs = compute_2pacf(df_data, df_rand_obs, log_spacing=USE_LOG)
    abs_obs = get_absolute_sum(theta, w_obs)

    jackknife = run_jackknife_errors(
        df_data,
        sf_dict,
        survey_weights,
        sf_nside,
        nside_jackknife=NSIDE_JACKKNIFE,
        min_regions=MIN_JACKKNIFE_REGIONS,
        n_jobs=n_jobs,
    )

    all_w_h0, all_abs_h0 = run_ensemble_mocks(
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

    stats = compute_statistics(
        w_obs,
        abs_obs,
        all_w_h0,
        all_abs_h0,
        label="Level 4: jackknife errors",
    )
    plot_results_with_jackknife(theta, w_obs, abs_obs, all_w_h0, all_abs_h0, jackknife)

    chi2_bin_diagnostics, chi2_svd_modes = compute_chi2_diagnostic_tables(
        theta,
        w_obs,
        all_w_h0,
        eigenvalue_cut=SVD_EIGENVALUE_CUT,
    )

    sensitivity_results = None
    if run_sensitivity:
        sensitivity_results = run_empirical_sensitivity_analysis(
            df_data,
            n_jobs=n_jobs,
        )

    results = {
        "df_data": df_data,
        "theta": theta,
        "w_obs": w_obs,
        "abs_obs": abs_obs,
        "sf_dict": sf_dict,
        "sf_cov_diag_dict": sf_cov_diag_dict,
        "survey_weights": survey_weights,
        "sf_validation": sf_validation,
        "intersurvey_corr": corr_df,
        "intersurvey_overlap": overlap_df,
        "jackknife": jackknife,
        "all_w_h0": all_w_h0,
        "all_abs_h0": all_abs_h0,
        "stats": stats,
        "chi2_bin_diagnostics": chi2_bin_diagnostics,
        "chi2_svd_modes": chi2_svd_modes,
        "sensitivity": sensitivity_results,
    }

    if save_tables:
        results["saved_tables"] = save_level4_tables(results)

    return results


if __name__ == "__main__":
    main_level4()
