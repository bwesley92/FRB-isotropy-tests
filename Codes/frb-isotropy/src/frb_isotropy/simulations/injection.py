from __future__ import annotations

import contextlib
import io

import healpy as hp
import numpy as np
import pandas as pd

from numpy.random import Generator

from ..core.config import RuntimeContext
from ..core.models import InjectionResult, InjectionSuite, SelectionFunctionSet
from ..core.types import FloatArray
from ..catalog.catalog import icrs_to_galactic_b
from ..simulations.mocks import generate_random_catalog, _jitter_pixel_centers
from ..statistics.estimators import compute_2pacf


# ==============================================================================
# Multipole injection
# ==============================================================================

def _legendre(ell: int, x: FloatArray) -> FloatArray:
    """Legendre polynomial P_ell(x) for the low multipoles used here."""

    x = np.asarray(x, dtype=float)

    if ell == 0:
        return np.ones_like(x)

    if ell == 1:
        return x

    if ell == 2:
        return 0.5 * (3.0 * x**2 - 1.0)

    if ell == 3:
        return 0.5 * (5.0 * x**3 - 3.0 * x)

    raise ValueError(f"Multipole ell={ell} not implemented.")


def _unit_vector_from_radec(
    ra_deg: FloatArray,
    dec_deg: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert ICRS RA/DEC to Cartesian unit vectors."""

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    cos_dec = np.cos(dec)

    return (
        cos_dec * np.cos(ra),
        cos_dec * np.sin(ra),
        np.sin(dec),
    )


def _multipole_modulation(
    ra_deg: FloatArray,
    dec_deg: FloatArray,
    *,
    epsilon: float,
    multipole: int,
    axis_ra_deg: float,
    axis_dec_deg: float,
) -> FloatArray:
    """Return 1 + epsilon P_ell(cos alpha) around a chosen sky axis."""

    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")

    x, y, z = _unit_vector_from_radec(ra_deg, dec_deg)
    ax, ay, az = _unit_vector_from_radec(axis_ra_deg, axis_dec_deg)

    cos_alpha = np.clip(
        x * ax + y * ay + z * az,
        -1.0,
        1.0,
    )

    weights = 1.0 + epsilon * _legendre(multipole, cos_alpha)

    if np.any(weights < 0.0):
        raise ValueError(
            f"Negative injection weights for epsilon={epsilon}. "
            "Reduce epsilon for this multipole."
        )

    return np.asarray(weights, dtype=float)


def inject_multipole_anisotropy(
    df_data: pd.DataFrame,
    epsilon: float,
    multipole: int = 1,
    rng: Generator | None = None,
    *,
    axis_ra_deg: float = 0.0,
    axis_dec_deg: float = 90.0,
) -> pd.DataFrame:
    """Legacy helper: resample an existing catalog with a known multipole."""

    if rng is None:
        rng = np.random.default_rng()

    weights = _multipole_modulation(
        df_data["RA"].to_numpy(float),
        df_data["DEC"].to_numpy(float),
        epsilon=epsilon,
        multipole=multipole,
        axis_ra_deg=axis_ra_deg,
        axis_dec_deg=axis_dec_deg,
    )

    weights = weights / weights.sum()
    indices = rng.choice(len(df_data), size=len(df_data), replace=True, p=weights)

    return df_data.iloc[indices].reset_index(drop=True)


def generate_injected_catalog_from_fixed_sf(
    context: RuntimeContext,
    sf_set: SelectionFunctionSet,
    *,
    n_observed: int,
    epsilon: float,
    multipole: int,
    rng: Generator,
    axis_ra_deg: float = 0.0,
    axis_dec_deg: float = 90.0,
    jitter_pixels: bool = True,
) -> pd.DataFrame:
    """Generate an observed synthetic catalog from fixed empirical SF x physics."""

    if n_observed <= 0:
        raise ValueError("n_observed must be positive.")

    surveys = np.asarray(sf_set.surveys)
    survey_probs = np.asarray([sf_set.survey_weights[s] for s in surveys], dtype=float)
    survey_probs /= survey_probs.sum()

    chosen_surveys = rng.choice(surveys, size=n_observed, p=survey_probs)

    ra = np.empty(n_observed, dtype=float)
    dec = np.empty(n_observed, dtype=float)

    for survey in surveys:
        mask = chosen_surveys == survey
        n_survey = int(np.sum(mask))

        if n_survey == 0:
            continue

        sf = np.asarray(sf_set.sf_dict[survey], dtype=float).copy()
        sf = np.clip(sf, 0.0, None)

        if sf.sum() <= 0.0:
            raise ValueError(f"Selection function vanished for survey: {survey}")

        theta_pix, phi_pix = hp.pix2ang(sf_set.nside, np.arange(len(sf)))
        ra_pix = np.degrees(phi_pix)
        dec_pix = 90.0 - np.degrees(theta_pix)

        modulation = _multipole_modulation(
            ra_pix,
            dec_pix,
            epsilon=epsilon,
            multipole=multipole,
            axis_ra_deg=axis_ra_deg,
            axis_dec_deg=axis_dec_deg,
        )

        prob = sf * modulation
        prob_sum = prob.sum()

        if prob_sum <= 0.0:
            raise ValueError(f"Injected probability vanished for survey: {survey}")

        prob /= prob_sum

        pix = rng.choice(len(prob), size=n_survey, p=prob)
        theta, phi = hp.pix2ang(sf_set.nside, pix)

        ra_survey = np.degrees(phi)
        dec_survey = 90.0 - np.degrees(theta)

        if jitter_pixels:
            ra_survey, dec_survey = _jitter_pixel_centers(
                context=context,
                ra=ra_survey,
                dec=dec_survey,
                nside=sf_set.nside,
                rng=rng,
            )

        ra[mask] = ra_survey
        dec[mask] = dec_survey

    out = pd.DataFrame(
        {
            "RA": ra,
            "DEC": dec,
            "Reporting_Group_s": chosen_surveys,
        }
    )

    if context.config.use_gal_mask:
        keep = np.abs(icrs_to_galactic_b(out["RA"].to_numpy(float), out["DEC"].to_numpy(float))) > context.config.gal_cut
        out = out.loc[keep].reset_index(drop=True)

    return out


# ==============================================================================
# Recovery measurement
# ==============================================================================

def measure_recovered_amplitude(
    theta: FloatArray,
    w_obs: FloatArray,
    w_h0_mean: FloatArray,
    multipole: int,
) -> float:
    """Project the 2pACF residual onto a multipole angular template."""

    delta = np.asarray(w_obs, dtype=float) - np.asarray(w_h0_mean, dtype=float)

    cos_theta = np.cos(np.radians(theta))
    template = _legendre(multipole, cos_theta)

    norm = float(np.dot(template, template))

    if norm < 1e-15:
        return 0.0

    return float(np.dot(delta, template) / norm)


def _safe_ratio(num: float, den: float, *, atol: float = 1e-12) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= atol:
        return float("nan")
    return float(num / den)


def _build_sf_quietly(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set_original: SelectionFunctionSet,
    *,
    verbose: bool,
) -> SelectionFunctionSet:
    from ..selection.selection import build_survey_selection_functions

    if verbose:
        return build_survey_selection_functions(
            context=context,
            df_data=df_data,
            nside=sf_set_original.nside,
            smooth_sigma=context.config.smooth_sigma,
        )

    with contextlib.redirect_stdout(io.StringIO()):
        return build_survey_selection_functions(
            context=context,
            df_data=df_data,
            nside=sf_set_original.nside,
            smooth_sigma=context.config.smooth_sigma,
        )


def run_injection_recovery_suite(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet,
    epsilons: FloatArray,
    multipole: int = 1,
    n_realizations: int = 10,
    n_rand_factor: int | None = None,
    *,
    axis_ra_deg: float = 0.0,
    axis_dec_deg: float = 90.0,
    rebuild_empirical_sf: bool = True,
    fixed_instrumental_sf_set: SelectionFunctionSet | None = None,
    verbose_sf: bool = False,
) -> InjectionSuite:
    """Run an empirical-SF absorption test for a known physical anisotropy.

    For each injected amplitude, synthetic observed catalogs are drawn from
    ``fixed empirical SF x physical multipole``. The same catalog is analyzed
    three ways: with no SF, with the fixed empirical SF, and with an empirical
    SF rebuilt from the injected catalog. If ``fixed_instrumental_sf_set`` is
    supplied, the same injected catalogs are also analyzed with that fixed
    instrumental SF.

    The reported ``signal_*`` values are baseline-subtracted projection
    coefficients of the 2pACF. The key diagnostic is
    ``retention_rebuilt_vs_fixed``; ``1 - retention`` is the fraction of the
    physical signal absorbed by rebuilding the empirical SF from the data.
    """

    from tqdm.auto import tqdm

    if sf_set is None:
        raise ValueError("sf_set must be provided for the absorption test.")

    if n_realizations <= 0:
        raise ValueError("n_realizations must be positive.")

    config = context.config

    if n_rand_factor is None:
        n_rand_factor = config.n_rand_factor

    epsilons = np.asarray(epsilons, dtype=float)

    raw_by_epsilon: list[dict[str, np.ndarray]] = []
    curve_by_epsilon: list[dict[str, np.ndarray]] = []
    theta_ref: FloatArray | None = None

    for i_eps, epsilon in enumerate(tqdm(epsilons, desc="Injection-recovery", unit="epsilon")):
        raw_no_sf: list[float] = []
        raw_fixed_sf: list[float] = []
        raw_rebuilt_sf: list[float] = []
        raw_fixed_instrumental_sf: list[float] = []

        curves_no_sf: list[FloatArray] = []
        curves_fixed_sf: list[FloatArray] = []
        curves_rebuilt_sf: list[FloatArray] = []
        curves_fixed_instrumental_sf: list[FloatArray] = []

        for i_real in tqdm(
            range(n_realizations),
            desc=f"  epsilon={epsilon:.2f}",
            leave=False,
        ):
            seed = config.random_seed + 100000 * i_eps + i_real
            rng = np.random.default_rng(seed)

            df_injected = generate_injected_catalog_from_fixed_sf(
                context=context,
                sf_set=sf_set,
                n_observed=len(df_data),
                epsilon=float(epsilon),
                multipole=multipole,
                axis_ra_deg=axis_ra_deg,
                axis_dec_deg=axis_dec_deg,
                rng=rng,
            )

            n_rand = len(df_injected) * n_rand_factor

            df_rand_no_sf = generate_random_catalog(
                context=context,
                n_observed=n_rand,
                sf_set=None,
                jitter_pixels=True,
                use_poisson=False,
                rng=rng,
            )
            theta, w_no_sf = compute_2pacf(context, df_injected, df_rand_no_sf)

            df_rand_fixed_sf = generate_random_catalog(
                context=context,
                n_observed=n_rand,
                sf_set=sf_set,
                jitter_pixels=True,
                use_poisson=False,
                rng=rng,
            )
            _, w_fixed_sf = compute_2pacf(context, df_injected, df_rand_fixed_sf)

            if rebuild_empirical_sf:
                sf_set_rebuilt = _build_sf_quietly(
                    context=context,
                    df_data=df_injected,
                    sf_set_original=sf_set,
                    verbose=verbose_sf,
                )
            else:
                sf_set_rebuilt = sf_set

            df_rand_rebuilt_sf = generate_random_catalog(
                context=context,
                n_observed=n_rand,
                sf_set=sf_set_rebuilt,
                jitter_pixels=True,
                use_poisson=False,
                rng=rng,
            )
            _, w_rebuilt_sf = compute_2pacf(context, df_injected, df_rand_rebuilt_sf)

            if fixed_instrumental_sf_set is not None:
                df_rand_fixed_instrumental_sf = generate_random_catalog(
                    context=context,
                    n_observed=n_rand,
                    sf_set=fixed_instrumental_sf_set,
                    jitter_pixels=True,
                    use_poisson=False,
                    rng=rng,
                )
                _, w_fixed_instrumental_sf = compute_2pacf(
                    context,
                    df_injected,
                    df_rand_fixed_instrumental_sf,
                )
            else:
                w_fixed_instrumental_sf = None

            w_h0_mean = np.zeros_like(w_no_sf)

            raw_no_sf.append(measure_recovered_amplitude(theta, w_no_sf, w_h0_mean, multipole))
            raw_fixed_sf.append(measure_recovered_amplitude(theta, w_fixed_sf, w_h0_mean, multipole))
            raw_rebuilt_sf.append(measure_recovered_amplitude(theta, w_rebuilt_sf, w_h0_mean, multipole))

            curves_no_sf.append(np.asarray(w_no_sf, dtype=float))
            curves_fixed_sf.append(np.asarray(w_fixed_sf, dtype=float))
            curves_rebuilt_sf.append(np.asarray(w_rebuilt_sf, dtype=float))

            if w_fixed_instrumental_sf is not None:
                raw_fixed_instrumental_sf.append(
                    measure_recovered_amplitude(
                        theta,
                        w_fixed_instrumental_sf,
                        w_h0_mean,
                        multipole,
                    )
                )
                curves_fixed_instrumental_sf.append(
                    np.asarray(w_fixed_instrumental_sf, dtype=float)
                )

            theta_ref = np.asarray(theta, dtype=float)

        raw_by_epsilon.append(
            {
                "no_sf": np.asarray(raw_no_sf, dtype=float),
                "fixed_sf": np.asarray(raw_fixed_sf, dtype=float),
                "rebuilt_sf": np.asarray(raw_rebuilt_sf, dtype=float),
                "fixed_instrumental_sf": np.asarray(raw_fixed_instrumental_sf, dtype=float),
            }
        )

        curves_for_epsilon = {
            "no_sf": np.mean(np.asarray(curves_no_sf, dtype=float), axis=0),
            "fixed_sf": np.mean(np.asarray(curves_fixed_sf, dtype=float), axis=0),
            "rebuilt_sf": np.mean(np.asarray(curves_rebuilt_sf, dtype=float), axis=0),
        }

        if curves_fixed_instrumental_sf:
            curves_for_epsilon["fixed_instrumental_sf"] = np.mean(
                np.asarray(curves_fixed_instrumental_sf, dtype=float),
                axis=0,
            )
        else:
            curves_for_epsilon["fixed_instrumental_sf"] = None

        curve_by_epsilon.append(curves_for_epsilon)

    baseline_idx = int(np.argmin(np.abs(epsilons)))
    baseline = {
        mode: float(np.mean(raw_by_epsilon[baseline_idx][mode]))
        for mode in ("no_sf", "fixed_sf", "rebuilt_sf", "fixed_instrumental_sf")
        if raw_by_epsilon[baseline_idx][mode].size > 0
    }

    results: list[InjectionResult] = []
    retention_rebuilt_vs_fixed: list[float] = []

    for epsilon, raw, curves in zip(epsilons, raw_by_epsilon, curve_by_epsilon):
        signals = {
            mode: raw[mode] - baseline[mode]
            for mode in ("no_sf", "fixed_sf", "rebuilt_sf")
        }

        if raw["fixed_instrumental_sf"].size > 0:
            signals["fixed_instrumental_sf"] = (
                raw["fixed_instrumental_sf"]
                - baseline["fixed_instrumental_sf"]
            )

        mean_no_sf = float(np.mean(signals["no_sf"]))
        mean_fixed_sf = float(np.mean(signals["fixed_sf"]))
        mean_rebuilt_sf = float(np.mean(signals["rebuilt_sf"]))

        fixed_vs_no = _safe_ratio(mean_fixed_sf, mean_no_sf)
        rebuilt_vs_fixed = _safe_ratio(mean_rebuilt_sf, mean_fixed_sf)

        if "fixed_instrumental_sf" in signals:
            mean_fixed_instrumental_sf = float(np.mean(signals["fixed_instrumental_sf"]))
            instrumental_vs_no = _safe_ratio(mean_fixed_instrumental_sf, mean_no_sf)
            instrumental_vs_fixed = _safe_ratio(mean_fixed_instrumental_sf, mean_fixed_sf)
            instrumental_absorption = (
                float(1.0 - instrumental_vs_no)
                if np.isfinite(instrumental_vs_no)
                else float("nan")
            )
            fixed_instrumental_std = (
                float(np.std(signals["fixed_instrumental_sf"], ddof=1))
                if n_realizations > 1
                else 0.0
            )
            raw_fixed_instrumental_mean = float(np.mean(raw["fixed_instrumental_sf"]))
            baseline_fixed_instrumental = baseline["fixed_instrumental_sf"]
            curve_fixed_instrumental = curves["fixed_instrumental_sf"]
        else:
            mean_fixed_instrumental_sf = float("nan")
            instrumental_vs_no = float("nan")
            instrumental_vs_fixed = float("nan")
            instrumental_absorption = float("nan")
            fixed_instrumental_std = float("nan")
            raw_fixed_instrumental_mean = float("nan")
            baseline_fixed_instrumental = float("nan")
            curve_fixed_instrumental = None

        absorption = float(1.0 - rebuilt_vs_fixed) if np.isfinite(rebuilt_vs_fixed) else float("nan")

        retention_rebuilt_vs_fixed.append(rebuilt_vs_fixed)

        results.append(
            InjectionResult(
                epsilon_injected=float(epsilon),
                multipole=multipole,
                signal_no_sf=mean_no_sf,
                signal_fixed_sf=mean_fixed_sf,
                signal_rebuilt_sf=mean_rebuilt_sf,
                signal_no_sf_std=float(np.std(signals["no_sf"], ddof=1)) if n_realizations > 1 else 0.0,
                signal_fixed_sf_std=float(np.std(signals["fixed_sf"], ddof=1)) if n_realizations > 1 else 0.0,
                signal_rebuilt_sf_std=float(np.std(signals["rebuilt_sf"], ddof=1)) if n_realizations > 1 else 0.0,
                retention_fixed_vs_no_sf=fixed_vs_no,
                retention_rebuilt_vs_fixed=rebuilt_vs_fixed,
                absorption_fraction=absorption,
                raw_no_sf=float(np.mean(raw["no_sf"])),
                raw_fixed_sf=float(np.mean(raw["fixed_sf"])),
                raw_rebuilt_sf=float(np.mean(raw["rebuilt_sf"])),
                baseline_no_sf=baseline["no_sf"],
                baseline_fixed_sf=baseline["fixed_sf"],
                baseline_rebuilt_sf=baseline["rebuilt_sf"],
                w_obs_no_sf=curves["no_sf"],
                w_obs_fixed_sf=curves["fixed_sf"],
                w_obs_rebuilt_sf=curves["rebuilt_sf"],
                theta=np.asarray(theta_ref, dtype=float),
                signal_fixed_instrumental_sf=mean_fixed_instrumental_sf,
                signal_fixed_instrumental_sf_std=fixed_instrumental_std,
                retention_instrumental_vs_no_sf=instrumental_vs_no,
                retention_instrumental_vs_fixed=instrumental_vs_fixed,
                absorption_instrumental_vs_no_sf=instrumental_absorption,
                raw_fixed_instrumental_sf=raw_fixed_instrumental_mean,
                baseline_fixed_instrumental_sf=baseline_fixed_instrumental,
                w_obs_fixed_instrumental_sf=curve_fixed_instrumental,
            )
        )

        message = (
            f"epsilon={epsilon:.2f} | "
            f"signal no SF={mean_no_sf:.4g} | "
            f"fixed SF={mean_fixed_sf:.4g} | "
            f"rebuilt SF={mean_rebuilt_sf:.4g} | "
        )
        if np.isfinite(mean_fixed_instrumental_sf):
            message += f"instrumental SF={mean_fixed_instrumental_sf:.4g} | "
        message += (
            f"retention rebuilt/fixed={rebuilt_vs_fixed:.2%} | "
            f"absorbed={absorption:.2%}"
        )
        if np.isfinite(instrumental_vs_no):
            message += f" | retention instrumental/no SF={instrumental_vs_no:.2%}"

        tqdm.write(message)

    return InjectionSuite(
        multipole=multipole,
        epsilons=epsilons,
        suppression_factors=np.asarray(retention_rebuilt_vs_fixed, dtype=float),
        results=results,
    )


def _rebuild_sf_from_injected(
    context: RuntimeContext,
    df_injected: pd.DataFrame,
    sf_set_original: SelectionFunctionSet,
) -> SelectionFunctionSet:
    """Backward-compatible wrapper for rebuilding an empirical SF."""

    return _build_sf_quietly(
        context=context,
        df_data=df_injected,
        sf_set_original=sf_set_original,
        verbose=True,
    )
