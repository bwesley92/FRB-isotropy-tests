from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..catalog.catalog import healpix_galactic_mask
from ..core.config import RuntimeContext
from ..core.models import MockEnsemble, SelectionFunctionSet
from ..core.types import FloatArray
from ..simulations.mocks import run_ensemble_mocks
from .instrumental import (
    ChimeInstrumentalComponents,
    build_chime_instrumental_sf,
)


# ==============================================================================
# Data containers
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class SelectionMapComparison:
    """
    Structured comparison between two normalized
    HEALPix selection-function maps.
    """

    empirical: FloatArray
    instrumental: FloatArray

    delta: FloatArray

    ratio: FloatArray

    ratio_mask: np.ndarray

    metrics: pd.DataFrame

    nside: int


@dataclass(
    frozen=True,
    slots=True,
)
class ChimeMockComparison:
    """
    Pair of mock ensembles generated with empirical
    and instrumental CHIME selection functions.
    """

    empirical: MockEnsemble
    instrumental: MockEnsemble
    metadata: dict[str, int | float | str | bool]

# ==============================================================================
# Basic map utilities
# ==============================================================================

def infer_healpix_nside(
    hmap: FloatArray,
) -> int:
    """
    Infer and validate HEALPix nside from a 1D map.
    """

    arr = np.asarray(
        hmap,
        dtype=float,
    )

    if arr.ndim != 1:

        raise ValueError(
            "HEALPix maps must be 1D arrays."
        )

    npix = len(arr)

    if not hp.isnpixok(npix):

        raise ValueError(
            f"Invalid HEALPix map size: {npix}"
        )

    return int(
        hp.npix2nside(npix)
    )


def normalize_selection_map(
    sf: FloatArray,
    *,
    clip_negative: bool = True,
) -> FloatArray:
    """
    Return a finite, non-negative, sum-normalized map.
    """

    arr = np.asarray(
        sf,
        dtype=float,
    ).copy()

    if arr.ndim != 1:

        raise ValueError(
            "Selection function must be a 1D HEALPix map."
        )

    if not np.all(
        np.isfinite(arr)
    ):

        raise ValueError(
            "Selection function contains non-finite values."
        )

    if clip_negative:

        arr = np.clip(
            arr,
            0.0,
            None,
        )

    elif np.any(arr < 0.0):

        raise ValueError(
            "Selection function contains negative values."
        )

    norm = float(
        arr.sum()
    )

    if norm <= 0.0:

        raise ValueError(
            "Selection function has zero support."
        )

    arr /= norm

    return np.asarray(
        arr,
        dtype=float,
    )


def apply_common_chime_support(
    context: RuntimeContext,
    sf: FloatArray,
    *,
    nside: int | None = None,
    gal_cut: float | None = None,
    use_gal_mask: bool | None = None,
) -> FloatArray:
    """
    Apply the same Galactic support used by empirical
    survey selection functions and renormalize.
    """

    if nside is None:

        nside = infer_healpix_nside(
            sf
        )

    expected_npix = hp.nside2npix(
        nside
    )

    if len(sf) != expected_npix:

        raise ValueError(
            "Selection-function map size does not "
            f"match nside={nside}: {len(sf)} != {expected_npix}"
        )

    arr = normalize_selection_map(
        sf
    )

    mask = healpix_galactic_mask(
        context=context,
        nside=nside,
        gal_cut=gal_cut,
        use_mask=use_gal_mask,
    )

    arr[~mask] = 0.0

    return normalize_selection_map(
        arr
    )


def build_masked_chime_instrumental_sf(
    context: RuntimeContext,
    components: ChimeInstrumentalComponents,
    *,
    nside: int | None = None,
    gal_cut: float | None = None,
    use_gal_mask: bool | None = None,
) -> FloatArray:
    """
    Build the CHIME instrumental SF and put it on the
    same support convention used by empirical SFs.
    """

    sf = build_chime_instrumental_sf(
        components
    )

    if nside is None:

        nside = infer_healpix_nside(
            sf
        )

    return apply_common_chime_support(
        context=context,
        sf=sf,
        nside=nside,
        gal_cut=gal_cut,
        use_gal_mask=use_gal_mask,
    )


# ==============================================================================
# Metrics
# ==============================================================================

def selection_entropy_bits(
    sf: FloatArray,
) -> float:
    """
    Shannon entropy of a normalized selection map.
    """

    p = normalize_selection_map(
        sf
    )

    positive = p > 0.0

    return float(
        -np.sum(
            p[positive]
            * np.log2(
                p[positive]
            )
        )
    )


def selection_effective_pixels(
    sf: FloatArray,
) -> float:
    """
    Effective number of pixels, exp(H), using
    Shannon entropy in natural units.
    """

    p = normalize_selection_map(
        sf
    )

    positive = p > 0.0

    entropy_nats = -np.sum(
        p[positive]
        * np.log(
            p[positive]
        )
    )

    return float(
        np.exp(entropy_nats)
    )


def kl_divergence(
    p: FloatArray,
    q: FloatArray,
    *,
    eps: float = 1e-30,
) -> float:
    """
    Kullback-Leibler divergence D_KL(p || q).
    """

    p_norm = normalize_selection_map(
        p
    )

    q_norm = normalize_selection_map(
        q
    )

    p_safe = np.clip(
        p_norm,
        eps,
        None,
    )

    q_safe = np.clip(
        q_norm,
        eps,
        None,
    )

    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()

    return float(
        np.sum(
            p_safe
            * np.log(
                p_safe / q_safe
            )
        )
    )

def selection_map_correlation(
    empirical: FloatArray,
    instrumental: FloatArray,
    *,
    threshold_fraction: float = 1e-4,
) -> float:
    """
    Pearson correlation between empirical and
    instrumental selection maps over the valid
    instrumental support.
    """

    empirical = normalize_selection_map(
        empirical
    )

    instrumental = normalize_selection_map(
        instrumental
    )

    valid = (
        instrumental
        > threshold_fraction
        * instrumental.max()
    )

    if np.sum(valid) < 2:

        return np.nan

    if (
        np.std(empirical[valid]) == 0.0
        or
        np.std(instrumental[valid]) == 0.0
    ):

        return np.nan

    return float(
        np.corrcoef(
            empirical[valid],
            instrumental[valid],
        )[0, 1]
    )

def ratio_statistics(
    empirical: FloatArray,
    instrumental: FloatArray,
    *,
    threshold_fraction: float = 1e-4,
) -> dict[str, float]:
    """
    Summary statistics of the empirical /
    instrumental ratio over the valid
    instrumental support.
    """

    empirical = normalize_selection_map(
        empirical
    )

    instrumental = normalize_selection_map(
        instrumental
    )

    valid = (
        instrumental
        > threshold_fraction
        * instrumental.max()
    )

    if np.sum(valid) == 0:

        return {
            "mean_ratio": np.nan,
            "median_ratio": np.nan,
            "std_ratio": np.nan,
        }

    ratio = (
        empirical[valid]
        /
        instrumental[valid]
    )

    return {
        "mean_ratio":
            float(np.mean(ratio)),
        "median_ratio":
            float(np.median(ratio)),
        "std_ratio":
            float(np.std(ratio)),
    }

def jensen_shannon_divergence(
    p: FloatArray,
    q: FloatArray,
    *,
    eps: float = 1e-30,
) -> float:
    """
    Symmetric Jensen-Shannon divergence in nats.
    """

    p_norm = normalize_selection_map(
        p
    )

    q_norm = normalize_selection_map(
        q
    )

    mixture = 0.5 * (
        p_norm
        + q_norm
    )

    return float(
        0.5
        * kl_divergence(
            p_norm,
            mixture,
            eps=eps,
        )
        + 0.5
        * kl_divergence(
            q_norm,
            mixture,
            eps=eps,
        )
    )


def summarize_selection_map(
    sf: FloatArray,
    *,
    label: str,
) -> dict[str, float | str]:
    """
    Summary diagnostics for one normalized selection map.
    """

    arr = normalize_selection_map(
        sf
    )

    active = arr > 0.0

    effective_pixels = selection_effective_pixels(
        arr
    )

    return {
        "label": label,
        "sum": float(arr.sum()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "nonzero_pixels": float(np.sum(active)),
        "nonzero_fraction": float(np.mean(active)),
        "entropy_bits": selection_entropy_bits(arr),
        "effective_pixels": effective_pixels,
        "effective_sky_fraction": float(
            effective_pixels / len(arr)
        ),
    }


def compare_chime_selection_maps(
    sf_empirical: FloatArray,
    sf_instrumental: FloatArray,
    *,
    eps: float = 1e-30,
) -> SelectionMapComparison:
    """
    Compare empirical and instrumental CHIME SF maps.
    """

    nside_emp = infer_healpix_nside(
        sf_empirical
    )

    nside_inst = infer_healpix_nside(
        sf_instrumental
    )

    if nside_emp != nside_inst:

        raise ValueError(
            "Empirical and instrumental SFs must have "
            f"the same nside: {nside_emp} != {nside_inst}"
        )

    empirical = normalize_selection_map(
        sf_empirical
    )

    instrumental = normalize_selection_map(
        sf_instrumental
    )

    delta = empirical - instrumental

    valid = (
        instrumental
        > 1e-4 * np.max(instrumental)
    )

    ratio = np.full_like(
        instrumental,
        np.nan,
    )

    ratio[valid] = (
        empirical[valid]
        /
        instrumental[valid]
    )

    correlation = (
        selection_map_correlation(
            empirical,
            instrumental,
        )
    )

    ratio_stats = (
        ratio_statistics(
            empirical,
            instrumental,
        )
    )

    rows: list[dict[str, float | str]] = [
        summarize_selection_map(
            empirical,
            label="empirical",
        ),
        summarize_selection_map(
            instrumental,
            label="instrumental",
        ),
    ]

    rows.append(
        {
            "label": "comparison",
            "sum": float(delta.sum()),
            "min": float(delta.min()),
            "max": float(delta.max()),
            "nonzero_pixels": float(
                np.sum(delta != 0.0)
            ),
            "nonzero_fraction": float(
                np.mean(delta != 0.0)
            ),
            "entropy_bits": np.nan,
            "effective_pixels": np.nan,
            "effective_sky_fraction": np.nan,
            "l1_distance": float(
                np.sum(
                    np.abs(delta)
                )
            ),
            "max_abs_delta": float(
                np.max(
                    np.abs(delta)
                )
            ),
            "kl_empirical_to_instrumental": kl_divergence(
                empirical,
                instrumental,
                eps=eps,
            ),
            "kl_instrumental_to_empirical": kl_divergence(
                instrumental,
                empirical,
                eps=eps,
            ),
            "jensen_shannon": jensen_shannon_divergence(
                empirical,
                instrumental,
                eps=eps,
            ),
            "correlation": correlation,

            "mean_ratio":
                ratio_stats["mean_ratio"],

            "median_ratio":
                ratio_stats["median_ratio"],

            "std_ratio":
                ratio_stats["std_ratio"],

        }
    )

    return SelectionMapComparison(
        empirical=empirical,
        instrumental=instrumental,
        delta=np.asarray(delta, dtype=float),
        ratio=np.asarray(ratio, dtype=float),
        ratio_mask=np.asarray(
            valid,
            dtype=bool,
        ),
        metrics=pd.DataFrame(rows),
        nside=nside_emp,
    )


# ==============================================================================
# Visualization
# ==============================================================================

def plot_chime_selection_comparison(
    comparison: SelectionMapComparison,
    *,
    save_path: str | Path | None = None,
    cmap: str = "viridis",
    difference_cmap: str = "coolwarm",
    ratio_cmap: str = "magma",
) -> None:
    """
    Plot empirical, instrumental, delta, and ratio maps.
    """

    hp.mollview(
        comparison.empirical,
        title="CHIME empirical SF",
        cmap=cmap,
        sub=(2, 2, 1),
    )

    hp.mollview(
        comparison.instrumental,
        title="CHIME instrumental SF",
        cmap=cmap,
        sub=(2, 2, 2),
    )

    vmax = float(
        np.max(
            np.abs(
                comparison.delta
            )
        )
    )

    hp.mollview(
        comparison.delta,
        title="Empirical - Instrumental",
        cmap=difference_cmap,
        min=-vmax if vmax > 0.0 else None,
        max=vmax if vmax > 0.0 else None,
        sub=(2, 2, 3),
    )

    ratio = np.asarray(
        comparison.ratio,
        dtype=float,
    )

    finite = np.isfinite(
        ratio
    )

    ratio_max = None

    if np.any(finite):

        ratio_max = float(
            np.percentile(
                ratio[finite],
                99.0,
            )
        )

    hp.mollview(
        ratio,
        title="Empirical / Instrumental",
        cmap=ratio_cmap,
        min=0.0,
        max=ratio_max,
        sub=(2, 2, 4),
    )

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

def plot_ratio_histogram(
    comparison: SelectionMapComparison,
    *,
    bins: int = 50,
    log10_ratio: bool = True,
) -> None:
    """
    Histogram of empirical/instrumental ratio
    over the valid support region.
    """

    ratio = np.asarray(
        comparison.ratio,
        dtype=float,
    )

    valid = np.isfinite(
        ratio
    )

    values = ratio[valid]

    values = values[
        values > 0.0
    ]

    if len(values) == 0:

        raise ValueError(
            "No positive ratio values available."
        )

    if log10_ratio:

        values = np.log10(
            values
        )

        xlabel = (
            r"$\log_{10}"
            r"(\phi_{\rm emp}/"
            r"\phi_{\rm inst})$"
        )

        reference_value = 0.0

    else:

        xlabel = (
            r"$\phi_{\rm emp}/"
            r"\phi_{\rm inst}$"
        )

        reference_value = 1.0

    plt.figure(
        figsize=(6, 4)
    )

    plt.hist(
        values,
        bins=bins,
    )

    plt.axvline(
        reference_value,
        linestyle="--",
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        "Pixels"
    )

    plt.title(
        "CHIME SF Ratio Distribution"
    )

    plt.tight_layout()

    plt.show()


# ==============================================================================
# SelectionFunctionSet adapters
# ==============================================================================

def extract_chime_empirical_sf(
    sf_set: SelectionFunctionSet,
    *,
    survey_name: str = "CHIME",
) -> FloatArray:
    """
    Extract the empirical CHIME map from a survey SF set.
    """

    if survey_name not in sf_set.sf_dict:

        raise KeyError(
            f"Survey not found in selection-function set: {survey_name}"
        )

    return normalize_selection_map(
        sf_set.sf_dict[survey_name]
    )


def build_chime_instrumental_selection_set(
    empirical_sf_set: SelectionFunctionSet,
    sf_instrumental: FloatArray,
    *,
    survey_name: str = "CHIME",
    covariance_mode: str = "zero",
) -> SelectionFunctionSet:
    """
    Return a copy of an empirical survey SF set with
    only the CHIME map replaced by the instrumental map.

    covariance_mode:
        "zero"
            No instrumental SF perturbation in mock ensembles.

        "empirical"
            Reuse the empirical CHIME covariance diagonal.
    """

    if survey_name not in empirical_sf_set.sf_dict:

        raise KeyError(
            f"Survey not found in selection-function set: {survey_name}"
        )

    expected_npix = hp.nside2npix(
        empirical_sf_set.nside
    )

    if len(sf_instrumental) != expected_npix:

        raise ValueError(
            "Instrumental CHIME SF has incompatible size: "
            f"{len(sf_instrumental)} != {expected_npix}"
        )

    sf_dict = {
        name: np.asarray(sf, dtype=float).copy()
        for name, sf in empirical_sf_set.sf_dict.items()
    }

    cov_dict = {
        name: np.asarray(cov, dtype=float).copy()
        for name, cov in empirical_sf_set.sf_cov_diag_dict.items()
    }

    sf_dict[survey_name] = normalize_selection_map(
        sf_instrumental
    )

    if covariance_mode == "zero":

        cov_dict[survey_name] = np.zeros_like(
            sf_dict[survey_name],
            dtype=float,
        )

    elif covariance_mode == "empirical":

        cov_dict[survey_name] = np.asarray(
            empirical_sf_set.sf_cov_diag_dict[survey_name],
            dtype=float,
        ).copy()

    else:

        raise ValueError(
            "covariance_mode must be 'zero' or 'empirical'."
        )

    return SelectionFunctionSet(
        sf_dict=sf_dict,
        sf_cov_diag_dict=cov_dict,
        survey_weights=dict(
            empirical_sf_set.survey_weights
        ),
        nside=empirical_sf_set.nside,
    )


def build_chime_only_selection_set(
    sf_chime: FloatArray,
    *,
    nside: int | None = None,
    survey_name: str = "CHIME",
    covariance: FloatArray | None = None,
) -> SelectionFunctionSet:
    """
    Build a CHIME-only SelectionFunctionSet.
    """

    sf = normalize_selection_map(
        sf_chime
    )

    inferred_nside = infer_healpix_nside(
        sf
    )

    if nside is None:

        nside = inferred_nside

    if nside != inferred_nside:

        raise ValueError(
            f"nside mismatch: {nside} != {inferred_nside}"
        )

    if covariance is None:

        covariance = np.zeros_like(
            sf,
            dtype=float,
        )

    cov = np.asarray(
        covariance,
        dtype=float,
    )

    if cov.shape != sf.shape:

        raise ValueError(
            "CHIME-only covariance shape mismatch."
        )

    return SelectionFunctionSet(
        sf_dict={survey_name: sf},
        sf_cov_diag_dict={survey_name: cov},
        survey_weights={survey_name: 1.0},
        nside=nside,
    )


# ==============================================================================
# Mock comparison
# ==============================================================================

def run_chime_mock_comparison(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    empirical_sf_set: SelectionFunctionSet,
    instrumental_sf_set: SelectionFunctionSet,
    *,
    n_ensemble: int | None = None,
    n_mocks_per: int | None = None,
    perturbation_scale: float = 0.0,
    n_rand_factor: int | None = None,
    n_jobs: int | None = None,
) -> ChimeMockComparison:
    """
    Run matched H0 mock ensembles for empirical and
    instrumental CHIME selection-function models.

    By default perturbation_scale is zero so the
    empirical-vs-instrumental difference is not mixed
    with SF perturbation noise.
    """

    empirical_mocks = run_ensemble_mocks(
        context=context,
        df_data=df_data,
        sf_set=empirical_sf_set,
        n_ensemble=n_ensemble,
        n_mocks_per=n_mocks_per,
        perturbation_scale=perturbation_scale,
        n_rand_factor=n_rand_factor,
        n_jobs=n_jobs,
    )

    instrumental_mocks = run_ensemble_mocks(
        context=context,
        df_data=df_data,
        sf_set=instrumental_sf_set,
        n_ensemble=n_ensemble,
        n_mocks_per=n_mocks_per,
        perturbation_scale=perturbation_scale,
        n_rand_factor=n_rand_factor,
        n_jobs=n_jobs,
    )

    return ChimeMockComparison(
        empirical=empirical_mocks,
        instrumental=instrumental_mocks,
        metadata={
            "n_empirical_mocks": int(
                empirical_mocks.metadata["n_mocks"]
            ),
            "n_instrumental_mocks": int(
                instrumental_mocks.metadata["n_mocks"]
            ),
            "perturbation_scale": float(
                perturbation_scale
            ),
            "comparison": "CHIME empirical vs instrumental SF",
        },
    )

def plot_empirical_vs_instrumental_scatter(
    comparison: SelectionMapComparison,
    *,
    threshold_fraction: float = 1e-4,
    max_points: int | None = 20000,
) -> None:
    """
    Scatter plot of empirical versus instrumental
    selection-function values over the valid
    instrumental support.

    The dashed line indicates perfect agreement:
        empirical = instrumental
    """

    empirical = np.asarray(
        comparison.empirical,
        dtype=float,
    )

    instrumental = np.asarray(
        comparison.instrumental,
        dtype=float,
    )

    valid = (
        instrumental
        > threshold_fraction
        * instrumental.max()
    )

    x = instrumental[valid]
    y = empirical[valid]

    if len(x) == 0:

        raise ValueError(
            "No valid pixels available for scatter plot."
        )

    if (
        max_points is not None
        and len(x) > max_points
    ):

        rng = np.random.default_rng(
            42
        )

        idx = rng.choice(
            len(x),
            size=max_points,
            replace=False,
        )

        x = x[idx]
        y = y[idx]

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        x,
        y,
        s=5,
        alpha=0.4,
    )

    xy_max = float(
        max(
            np.max(x),
            np.max(y),
        )
    )

    plt.plot(
        [0.0, xy_max],
        [0.0, xy_max],
        linestyle="--",
        linewidth=2,
        label=r"$y=x$",
    )

    corr = selection_map_correlation(
        comparison.empirical,
        comparison.instrumental,
        threshold_fraction=threshold_fraction,
    )

    plt.xlabel(
        r"$\phi_{\rm inst}$"
    )

    plt.ylabel(
        r"$\phi_{\rm emp}$"
    )

    plt.title(
        f"CHIME SF Scatter "
        f"(r={corr:.3f})"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


def summarize_wtheta_difference(
    mock_comparison: ChimeMockComparison,
) -> pd.DataFrame:
    """
    Compare mean w(theta) profiles obtained from
    empirical and instrumental selection functions.
    """

    emp_w = np.asarray(
        mock_comparison.empirical.w_theta,
        dtype=float,
    )

    inst_w = np.asarray(
        mock_comparison.instrumental.w_theta,
        dtype=float,
    )

    emp_mean = np.mean(
        emp_w,
        axis=0,
    )

    inst_mean = np.mean(
        inst_w,
        axis=0,
    )

    emp_std = np.std(
        emp_w,
        axis=0,
        ddof=1,
    )

    inst_std = np.std(
        inst_w,
        axis=0,
        ddof=1,
    )

    delta = (
        emp_mean
        - inst_mean
    )

    sigma = np.sqrt(
        emp_std**2
        +
        inst_std**2
    )

    zscore = np.full_like(
        delta,
        np.nan,
    )

    mask = sigma > 0.0

    zscore[mask] = (
        delta[mask]
        /
        sigma[mask]
    )

    return pd.DataFrame(
        {
            "bin": np.arange(
                len(emp_mean)
            ),
            "empirical_mean": emp_mean,
            "instrumental_mean": inst_mean,
            "empirical_std": emp_std,
            "instrumental_std": inst_std,
            "delta": delta,
            "sigma_combined": sigma,
            "zscore": zscore,
        }
    )


def plot_wtheta_comparison(
    mock_comparison: ChimeMockComparison,
) -> None:
    """
    Compare the ensemble-mean w(theta)
    profiles from empirical and instrumental
    selection functions.
    """

    emp_w = np.asarray(
        mock_comparison.empirical.w_theta,
        dtype=float,
    )

    inst_w = np.asarray(
        mock_comparison.instrumental.w_theta,
        dtype=float,
    )

    emp_mean = np.mean(
        emp_w,
        axis=0,
    )

    inst_mean = np.mean(
        inst_w,
        axis=0,
    )

    emp_std = np.std(
        emp_w,
        axis=0,
        ddof=1,
    )

    inst_std = np.std(
        inst_w,
        axis=0,
        ddof=1,
    )

    bins = np.arange(
        len(emp_mean)
    )

    plt.figure(
        figsize=(8, 4)
    )

    plt.errorbar(
        bins,
        emp_mean,
        yerr=emp_std,
        marker="o",
        capsize=3,
        label="Empirical SF",
    )

    plt.errorbar(
        bins,
        inst_mean,
        yerr=inst_std,
        marker="s",
        capsize=3,
        label="Instrumental SF",
    )

    plt.xlabel(
        "2PACF bin"
    )

    plt.ylabel(
        r"$w(\theta)$"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()