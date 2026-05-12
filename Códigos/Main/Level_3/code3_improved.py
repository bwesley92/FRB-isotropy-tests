"""
Improved Level 3 pipeline for FRB isotropy tests.

Main upgrades relative to the original notebook:
- survey selection functions include a diagonal Poisson uncertainty model;
- H0 mocks marginalize over plausible selection-function variants;
- NSIDE/smoothing sensitivity analysis is automated;
- intersurvey spatial correlations and overlaps are diagnosed;
- selection functions are validated numerically and visually.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.coordinates import SkyCoord
import astropy.units as u
from joblib import Parallel, delayed
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm
import treecorr
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib


# ==============================================================
# 1. General configuration
# ==============================================================

BASE_PATH = "/home/brunowesley/projetos/FRB-isotropy-tests/FRB_catalogs/"
ALL_FRB_PATH = os.path.join(BASE_PATH, "SkyPosition.csv")

N_RAND_FACTOR = 20
N_MOCKS = 200
N_ENSEMBLE = 10
N_MOCKS_PER_ENSEMBLE = max(1, N_MOCKS // N_ENSEMBLE)

GAL_CUT = 20.0

NSIDE_SF = 64
SMOOTH_SIGMA = 5.0
PERTURBATION_SCALE = 1.0

NSIDE_SF_RANGE = [32, 64, 128]
SMOOTH_SIGMA_RANGE = [3.0, 5.0, 8.0, 10.0]
N_SENSITIVITY_MOCKS = 50

OVERLAP_RADIUS_DEG = 5.0
OVERLAP_NSIDE = 32

USE_LOG = False
MAX_SEP = 180.0
MIN_SEP = 0.5

if USE_LOG:
    BIN_TYPE = "Log"
    N_BINS = 20
else:
    BIN_TYPE = "Linear"
    BIN_SIZE = 5.0
    N_BINS = int((MAX_SEP - MIN_SEP) / BIN_SIZE)

COARSE_BINS = np.arange(0, 181, 20)
COARSE_CENTERS = 0.5 * (COARSE_BINS[:-1] + COARSE_BINS[1:])


@dataclass
class TestStatistics:
    chi2: float
    chi2_red: float
    p_chi2: float
    p_empirical: float
    sigma_equiv: float
    global_tension: float
    abs_observed_stat: float
    abs_empirical_p: float
    hartlap_factor: float


# ==============================================================
# 2. Catalog loading and mask
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


def apply_mask(df: pd.DataFrame, gal_cut: float = GAL_CUT) -> pd.DataFrame:
    coords = SkyCoord(
        ra=df["RA"].values * u.degree,
        dec=df["DEC"].values * u.degree,
        frame="icrs",
    )
    b = coords.galactic.b.degree
    return df[np.abs(b) > gal_cut].reset_index(drop=True)


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
# 3. Survey split and improved selection functions
# ==============================================================

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


def build_selection_function_improved(
    subdf: pd.DataFrame,
    nside: int = NSIDE_SF,
    smooth_sigma: float = SMOOTH_SIGMA,
    gal_cut: float = GAL_CUT,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one survey SF plus a diagonal Poisson covariance estimate."""
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
        # Smoothing induces pixel correlations; keep a conservative diagonal model.
        sf_cov_diag *= 2.0

    gal_mask = healpix_galactic_mask(nside, gal_cut=gal_cut)
    sf[~gal_mask] = 0.0

    sf_sum = sf.sum()
    if sf_sum <= 0:
        raise ValueError("Selection function vanished after masking.")
    sf /= sf_sum

    return sf, sf_cov_diag


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


def generate_sf_variant(
    sf_dict: dict[str, np.ndarray],
    sf_cov_diag_dict: dict[str, np.ndarray],
    perturbation_scale: float = PERTURBATION_SCALE,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Perturb each SF inside its diagonal Poisson uncertainty model."""
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
# 4. Random catalogs and 2pACF
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


def get_absolute_sum(theta: np.ndarray, w: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(len(COARSE_BINS) - 1):
        mask = (theta >= COARSE_BINS[i]) & (theta < COARSE_BINS[i + 1])
        vals.append(np.abs(np.mean(w[mask])) if np.any(mask) else 0.0)
    return np.array(vals)


def absolute_global_stat(abs_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(abs_values**2)))


# ==============================================================
# 5. Diagnostics and validation
# ==============================================================

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

            idx, sep2d, _ = coords_by_survey[survey1].match_to_catalog_sky(
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

    def plot_sf(self, save_path: str = "SF_validation.png", max_surveys: int = 12) -> None:
        ordered = sorted(
            self.sf_dict.keys(),
            key=lambda name: self.survey_weights[name],
            reverse=True,
        )[:max_surveys]

        ncols = min(4, len(ordered))
        nrows = int(np.ceil(len(ordered) / ncols))
        plt.figure(figsize=(5 * ncols, 3.8 * nrows))

        for i, name in enumerate(ordered, start=1):
            hp.mollview(
                self.sf_dict[name],
                title=f"{name} (w={self.survey_weights[name]:.1%})",
                sub=(nrows, ncols, i),
                cbar=True,
            )

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ==============================================================
# 6. Mocks, statistics, and sensitivity
# ==============================================================

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


def compute_statistics(
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
    label: str = "Level 3 improved",
) -> TestStatistics:
    h0_mean = np.mean(all_w_h0, axis=0)
    cov = np.cov(all_w_h0, rowvar=False)

    n_mocks, n_bins = all_w_h0.shape
    hartlap_factor = (n_mocks - n_bins - 2) / (n_mocks - 1)
    hartlap_factor = hartlap_factor if hartlap_factor > 0 else 1.0

    inv_cov = hartlap_factor * np.linalg.pinv(cov)
    delta = w_obs - h0_mean

    chi2 = float(delta @ inv_cov @ delta)
    dof = len(delta)
    chi2_red = chi2 / dof
    p_chi2 = float(chi2_dist.sf(chi2, dof))

    mock_delta = all_w_h0 - h0_mean
    chi2_mocks = np.einsum("ij,jk,ik->i", mock_delta, inv_cov, mock_delta)
    p_empirical = float((np.sum(chi2_mocks >= chi2) + 1) / (len(chi2_mocks) + 1))
    sigma_equiv = float(norm.isf(max(p_empirical, 1e-300)))

    diag_sigma = np.sqrt(np.diag(cov)) + 1e-12
    global_tension = float(np.sqrt(np.mean((delta / diag_sigma) ** 2)))

    abs_observed_stat = absolute_global_stat(abs_obs)
    abs_mock_stats = np.array([absolute_global_stat(row) for row in all_abs_h0])
    abs_empirical_p = float(
        (np.sum(abs_mock_stats >= abs_observed_stat) + 1) / (len(abs_mock_stats) + 1)
    )

    print(f"\n--- Statistical tests ({label}) ---")
    print(f"Chi2 = {chi2:.3f}")
    print(f"Chi2/dof = {chi2_red:.3f}")
    print(f"p-value (Chi2 analytic) = {p_chi2:.4e}")
    print(f"p-value (Chi2 empirical) = {p_empirical:.4e}")
    print(f"Sigma-equivalent (from empirical p) = {sigma_equiv:.2f}")
    print(f"Hartlap factor = {hartlap_factor:.3f}")
    print(f"RMS normalized deviation = {global_tension:.3f}")
    print("\n--- Absolute anisotropy amplitude ---")
    print(f"Observed RMS absolute amplitude = {abs_observed_stat:.4e}")
    print(f"Empirical p-value = {abs_empirical_p:.4e}")

    return TestStatistics(
        chi2=chi2,
        chi2_red=chi2_red,
        p_chi2=p_chi2,
        p_empirical=p_empirical,
        sigma_equiv=sigma_equiv,
        global_tension=global_tension,
        abs_observed_stat=abs_observed_stat,
        abs_empirical_p=abs_empirical_p,
        hartlap_factor=hartlap_factor,
    )


def run_sensitivity_analysis(
    df_data: pd.DataFrame,
    nside_range: list[int] = NSIDE_SF_RANGE,
    smooth_sigma_range: list[float] = SMOOTH_SIGMA_RANGE,
    n_mocks: int = N_SENSITIVITY_MOCKS,
) -> pd.DataFrame:
    rows = []
    print("\n--- Sensitivity analysis over NSIDE and smoothing ---")

    for nside in nside_range:
        for smooth_sigma in smooth_sigma_range:
            print(f"Testing NSIDE={nside}, smooth_sigma={smooth_sigma:.1f} deg...")
            sf_dict, sf_cov_diag_dict, survey_weights, sf_nside = (
                build_survey_selection_functions_improved(
                    df_data,
                    nside=nside,
                    smooth_sigma=smooth_sigma,
                )
            )

            df_rand_obs = generate_mixture_catalog_improved(
                len(df_data) * N_RAND_FACTOR,
                sf_dict,
                survey_weights,
                sf_nside,
                seed=2000 + nside + int(10 * smooth_sigma),
            )
            _, w_obs = compute_2pacf(df_data, df_rand_obs, log_spacing=USE_LOG)

            all_w = []
            for i in range(n_mocks):
                sf_variant = generate_sf_variant(
                    sf_dict,
                    sf_cov_diag_dict,
                    perturbation_scale=PERTURBATION_SCALE,
                    seed=10000 + i,
                )
                w_h0, _ = run_one_mock(
                    seed=20000 + i,
                    n_data=len(df_data),
                    sf_dict=sf_variant,
                    survey_weights=survey_weights,
                    nside=sf_nside,
                )
                all_w.append(w_h0)

            all_w_h0 = np.array(all_w)
            h0_mean = np.mean(all_w_h0, axis=0)
            cov = np.cov(all_w_h0, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            delta = w_obs - h0_mean
            chi2 = float(delta @ inv_cov @ delta)
            dof = len(delta)
            chi2_red = chi2 / dof
            p_chi2 = float(chi2_dist.sf(chi2, dof))

            rows.append(
                {
                    "nside": nside,
                    "smooth_sigma": smooth_sigma,
                    "chi2": chi2,
                    "chi2_red": chi2_red,
                    "p_chi2": p_chi2,
                }
            )
            print(f"  chi2/dof={chi2_red:.3f}, p={p_chi2:.4e}")

    results = pd.DataFrame(rows)
    print("\n--- Sensitivity summary ---")
    print(results.round(4))
    print(
        "chi2/dof range: "
        f"{results['chi2_red'].min():.3f} - {results['chi2_red'].max():.3f}"
    )
    print(f"Variation: {results['chi2_red'].max() - results['chi2_red'].min():.3f}")
    return results


# ==============================================================
# 7. Plots and main pipeline
# ==============================================================

def plot_results(
    theta: np.ndarray,
    w_obs: np.ndarray,
    abs_obs: np.ndarray,
    all_w_h0: np.ndarray,
    all_abs_h0: np.ndarray,
    output_2pacf: str = "FRB_2PACF_level3_improved.png",
    output_abs: str = "FRB_ABS_level3_improved.png",
) -> None:
    sns.set_theme(style="whitegrid")

    h0_mean = np.mean(all_w_h0, axis=0)
    h0_std = np.std(all_w_h0, axis=0)
    h0_abs_mean = np.mean(all_abs_h0, axis=0)
    h0_abs_std = np.std(all_abs_h0, axis=0)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.fill_between(theta, h0_mean - 3 * h0_std, h0_mean + 3 * h0_std,
                    color="silver", alpha=0.2)
    ax.fill_between(theta, h0_mean - 2 * h0_std, h0_mean + 2 * h0_std,
                    color="darkgray", alpha=0.4)
    ax.fill_between(theta, h0_mean - h0_std, h0_mean + h0_std,
                    color="gray", alpha=0.6)
    ax.plot(theta, w_obs, "o-", color="firebrick", lw=2.5,
            label="Observed (masked + marginalized multi-survey SF)")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$w(\theta)$")
    ax.set_title("Angular Two-Point Correlation Function\n(Level 3 improved)")
    h0_patch = Patch(facecolor="gray", alpha=0.6,
                     label=r"$H_0$ ensemble (1 sigma, 2 sigma, 3 sigma)")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [h0_patch])
    plt.tight_layout()
    plt.savefig(output_2pacf, dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.fill_between(COARSE_CENTERS,
                    h0_abs_mean - 3 * h0_abs_std,
                    h0_abs_mean + 3 * h0_abs_std,
                    color="silver", alpha=0.2)
    ax.fill_between(COARSE_CENTERS,
                    h0_abs_mean - 2 * h0_abs_std,
                    h0_abs_mean + 2 * h0_abs_std,
                    color="darkgray", alpha=0.4)
    ax.fill_between(COARSE_CENTERS,
                    h0_abs_mean - h0_abs_std,
                    h0_abs_mean + h0_abs_std,
                    color="gray", alpha=0.6)
    ax.plot(COARSE_CENTERS, abs_obs, "o-", color="firebrick", lw=2.5,
            label="Observed")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$|\langle w \rangle|$")
    ax.set_title("Absolute Sum Test\n(Level 3 improved)")
    h0_patch = Patch(facecolor="gray", alpha=0.6,
                     label=r"$H_0$ ensemble")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [h0_patch])
    plt.tight_layout()
    plt.savefig(output_abs, dpi=300)
    plt.show()


def main(
    run_sensitivity: bool = True,
    n_jobs: int = -1,
) -> dict[str, object]:
    print(f"\n--- Processing data ({BIN_TYPE}) ---")
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
    validator.plot_sf(save_path="SF_validation.png")

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

    stats = compute_statistics(w_obs, abs_obs, all_w_h0, all_abs_h0)
    plot_results(theta, w_obs, abs_obs, all_w_h0, all_abs_h0)

    sensitivity_results = None
    if run_sensitivity:
        sensitivity_results = run_sensitivity_analysis(df_data)

    return {
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
        "all_w_h0": all_w_h0,
        "all_abs_h0": all_abs_h0,
        "stats": stats,
        "sensitivity": sensitivity_results,
    }


if __name__ == "__main__":
    main()
