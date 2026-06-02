from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import seaborn as sns
import pandas as pd
import healpy as hp

from ..core.config import RuntimeContext
from ..core.types import (
    FloatArray,
)

from ..selection.selection import (
    split_by_survey,
)

from ..catalog.catalog import (
    rotate_healpix_map_to_galactic,
    icrs_to_galactic,
)

from ..core.models import (
    SelectionFunctionSet,
    JackknifeResult,
    BootstrapResult,
)


def plot_covariance_diagnostics(
    context: RuntimeContext,
    cov: FloatArray,
    all_w_h0: FloatArray,
    output_prefix: str = "h0",
) -> None:
    """
    Produce covariance diagnostic plots.

    Generates
    ---------
    - correlation matrix
    - mock similarity
    - covariance eigenspectrum
    """

    config = context.config

    outputs = context.outputs

    cov = np.asarray(
        cov,
        dtype=float,
    )

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    from ..analysis.diagnostics import (
        plot_covariance_correlation_matrix,
    )

    # ----------------------------------------------------------
    # Correlation matrix
    # ----------------------------------------------------------

    plot_covariance_correlation_matrix(

        context=context,

        cov=cov,

        title="H0 Correlation Matrix",

        output=(
            outputs.figures_dir
            / f"{output_prefix}_correlation_matrix.png"
        ),
    )

    # ----------------------------------------------------------
    # Mock similarity
    # ----------------------------------------------------------

    plot_mock_similarity(

        context=context,

        all_w_h0=all_w_h0,

        n_compare=min(
            100,
            len(all_w_h0),
        ),

        output=(
            outputs.figures_dir
            / f"{output_prefix}_mock_similarity.png"
        ),
    )

    # ----------------------------------------------------------
    # Eigenspectrum
    # ----------------------------------------------------------

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
        / f"{output_prefix}_eigenspectrum.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()


def plot_mock_similarity_matrix(
    context: RuntimeContext,
    corr: FloatArray,
    offdiag: FloatArray,
    n_mocks: int,
    output: str | Path = "mock_similarity.png",
) -> None:
    """
    Print diagnostics and visualize
    the mock-to-mock similarity matrix.
    """

    outputs = context.outputs

    output = Path(
        output
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("MOCK SIMILARITY DIAGNOSTICS")
    print("==================================================")

    print(
        f"Mocks analyzed: "
        f"{n_mocks}"
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


def plot_mock_similarity(
    context: RuntimeContext,
    all_w_h0: FloatArray,
    n_compare: int = 100,
    output: str | Path = "mock_similarity.png",
) -> None:
    from ..analysis.diagnostics import (
        prepare_mock_similarity_data,
        compute_mock_similarity_matrix,
    )

    normalized = (
        prepare_mock_similarity_data(
            all_w_h0=all_w_h0,
            n_compare=n_compare,
        )
    )

    corr, offdiag = (
        compute_mock_similarity_matrix(
            normalized
        )
    )

    plot_mock_similarity_matrix(
        context=context,
        corr=corr,
        offdiag=offdiag,
        n_mocks=len(normalized),
        output=output,
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

                lon, lat = (
                    icrs_to_galactic(

                        subdf["RA"].to_numpy(
                            dtype=float,
                        ),

                        subdf["DEC"].to_numpy(
                            dtype=float,
                        ),
                    )
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


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_BAND_3  = "#c9daeb"
_BAND_2  = "#97b8d3"
_BAND_1  = "#5b8ebb"
_H0_LINE = "#2f4458"
_OBS_RED = "firebrick"

# (color, alpha, n_sigma) — drawn back-to-front; zorder starts at 1
_BANDS = [
    (_BAND_3, 0.95, 3.0),
    (_BAND_2, 0.78, 2.0),
    (_BAND_1, 0.58, 1.0),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_plot_style() -> None:
    sns.set_theme(
        style="white",
        context="talk",
        rc={"axes.edgecolor": "0.25", "axes.linewidth": 1.1},
    )


def _clean_h0(arr: FloatArray, label: str) -> FloatArray:
    """Drop rows with non-finite values; raise if fewer than 2 remain."""
    valid = np.all(np.isfinite(arr), axis=1)
    if not valid.all():
        print(f"\nWARNING: removing {(~valid).sum()} invalid H0 {label} realizations.")
        arr = arr[valid]
    if len(arr) < 2:
        raise RuntimeError(f"Too few valid H0 {label} realizations remain.")
    return arr


def _h0_stats(arr: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(mean, std) along axis=0; non-finite std entries are replaced with 0."""
    mean = np.nanmean(arr, axis=0)
    std  = np.nan_to_num(np.nanstd(arr, axis=0, ddof=1), nan=0.0, posinf=0.0, neginf=0.0)
    return mean, std


def _h0_label(context: RuntimeContext) -> str:
    return (
        "Isotropy + survey SF H0"
        if context.config.use_sel_func
        else "Pure isotropic H0"
    )


def _draw_h0_bands(
    ax: plt.Axes,
    x: FloatArray,
    mean: FloatArray,
    std: FloatArray,
) -> None:
    for zorder, (color, alpha, sigma) in enumerate(_BANDS, start=1):
        ax.fill_between(
            x, mean - sigma * std, mean + sigma * std,
            color=color, alpha=alpha, linewidth=0, zorder=zorder,
        )


def _save_and_show(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def _validate_and_prepare(
    theta:     FloatArray,
    w_obs:     FloatArray,
    abs_obs:   FloatArray,
    all_w_h0:  FloatArray,
    all_abs_h0: FloatArray,
    result:    JackknifeResult | BootstrapResult,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Cast to float, validate shapes, and clean H0 arrays.

    Returns
    -------
    theta, w_obs, abs_obs, all_w_h0 (cleaned), all_abs_h0 (cleaned)
    """
    theta, w_obs, abs_obs = (np.asarray(a, dtype=float) for a in (theta, w_obs, abs_obs))
    all_w_h0   = np.asarray(all_w_h0,   dtype=float)
    all_abs_h0 = np.asarray(all_abs_h0, dtype=float)

    for name, arr, ndim in [
        ("theta",      theta,      1),
        ("w_obs",      w_obs,      1),
        ("abs_obs",    abs_obs,    1),
        ("all_w_h0",   all_w_h0,   2),
        ("all_abs_h0", all_abs_h0, 2),
    ]:
        if arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}-dimensional.")

    for condition, msg in [
        (len(theta) != len(w_obs),            "theta and w_obs must have identical lengths."),
        (all_w_h0.shape[1]   != len(theta),   "all_w_h0 and theta have incompatible shapes."),
        (all_abs_h0.shape[1] != len(abs_obs), "all_abs_h0 and abs_obs have incompatible shapes."),
        (len(result.w_err)   != len(theta),   "result.w_err and theta have incompatible shapes."),
        (len(result.abs_err) != len(abs_obs), "result.abs_err and abs_obs have incompatible shapes."),
    ]:
        if condition:
            raise ValueError(msg)

    all_w_h0   = _clean_h0(all_w_h0,   "w(theta)")
    all_abs_h0 = _clean_h0(all_abs_h0, "absolute-sum")

    return theta, w_obs, abs_obs, all_w_h0, all_abs_h0


# ---------------------------------------------------------------------------
# Shared preparation container (private — not a domain container)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _PlotData:
    output_2pacf:   Path
    output_abs:     Path
    theta:          FloatArray
    w_obs:          FloatArray
    abs_obs:        FloatArray
    coarse_centers: FloatArray
    h0_mean:        FloatArray
    h0_std:         FloatArray
    h0_abs_mean:    FloatArray
    h0_abs_std:     FloatArray
    h0_label:       str


def _prepare(
    context:      RuntimeContext,
    theta:        FloatArray,
    w_obs:        FloatArray,
    abs_obs:      FloatArray,
    all_w_h0:     FloatArray,
    all_abs_h0:   FloatArray,
    result:       JackknifeResult | BootstrapResult,
    stem:         str,
    output_2pacf: str | Path | None,
    output_abs:   str | Path | None,
) -> _PlotData:
    figs = context.outputs.figures_dir
    output_2pacf = Path(output_2pacf or figs / f"2pacf_{stem}.png")
    output_abs   = Path(output_abs   or figs / f"absolute_anisotropy_{stem}.png")

    theta, w_obs, abs_obs, all_w_h0, all_abs_h0 = _validate_and_prepare(
        theta, w_obs, abs_obs, all_w_h0, all_abs_h0, result,
    )

    _set_plot_style()

    h0_mean,     h0_std     = _h0_stats(all_w_h0)
    h0_abs_mean, h0_abs_std = _h0_stats(all_abs_h0)

    return _PlotData(
        output_2pacf=output_2pacf,
        output_abs=output_abs,
        theta=theta,
        w_obs=w_obs,
        abs_obs=abs_obs,
        coarse_centers=np.asarray(context.config.coarse_centers, dtype=float),
        h0_mean=h0_mean,
        h0_std=h0_std,
        h0_abs_mean=h0_abs_mean,
        h0_abs_std=h0_abs_std,
        h0_label=_h0_label(context),
    )


# ---------------------------------------------------------------------------
# Generic plot renderers
# ---------------------------------------------------------------------------

def _plot_2pacf(
    *,
    theta:     FloatArray,
    w_obs:     FloatArray,
    w_err:     FloatArray,
    h0_mean:   FloatArray,
    h0_std:    FloatArray,
    h0_label:  str,
    err_label: str,
    err_color: str,
    output:    Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    _draw_h0_bands(ax, theta, h0_mean, h0_std)

    obs_h = ax.scatter(theta, w_obs, s=45, facecolor=_OBS_RED,
                       edgecolor="white", linewidth=1.0, zorder=6)
    err_h = ax.errorbar(theta, w_obs, yerr=w_err, fmt="none",
                        ecolor=err_color, elinewidth=2.0,
                        capsize=3.0, capthick=1.2, alpha=0.95, zorder=4)
    h0_h, = ax.plot(theta, h0_mean, color=_H0_LINE, lw=1.8, ls="--", zorder=5)
    ax.axhline(0.0, color="0.15", linewidth=1.1, alpha=0.9, zorder=0)

    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$w(\theta)$")
    ax.set_title(f"Angular Two-Point Correlation Function\n({err_label} uncertainties | {h0_label})")

    h0_patch = Patch(facecolor=_BAND_2, edgecolor=_BAND_1, alpha=0.85)
    ax.legend(
        [obs_h, err_h, h0_h, h0_patch],
        ["Observed 2pACF", err_label, r"$H_0$ mean", r"$H_0$ confidence bands"],
        frameon=True, framealpha=0.95,
    )
    _save_and_show(fig, output)


def _plot_absolute_sum(
    *,
    coarse_centers: FloatArray,
    abs_obs:        FloatArray,
    abs_err:        FloatArray,
    h0_abs_mean:    FloatArray,
    h0_abs_std:     FloatArray,
    h0_label:       str,
    err_label:      str,
    err_color:      str,
    output:         Path,
) -> None:
    if len(coarse_centers) != len(abs_obs):
        raise ValueError("coarse_centers and abs_obs have incompatible shapes.")

    fig, ax = plt.subplots(figsize=(11, 8))
    _draw_h0_bands(ax, coarse_centers, h0_abs_mean, h0_abs_std)

    obs_h = ax.scatter(coarse_centers, abs_obs, s=45, facecolor=_OBS_RED,
                       edgecolor="white", linewidth=1.0, zorder=6)
    err_h = ax.errorbar(coarse_centers, abs_obs, yerr=abs_err, fmt="none",
                        ecolor=err_color, elinewidth=2.0,
                        capsize=3.0, capthick=1.2, alpha=0.95, zorder=4)
    h0_h, = ax.plot(coarse_centers, h0_abs_mean, color=_H0_LINE, lw=1.8, ls="--", zorder=5)

    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$|\langle w \rangle|$")
    ax.set_title(f"Absolute Sum Test\n({err_label} uncertainties | {h0_label})")

    h0_patch = Patch(facecolor=_BAND_2, edgecolor=_BAND_1, alpha=0.85)
    ax.legend(
        [obs_h, err_h, h0_h, h0_patch],
        ["Observed statistic", err_label, r"$H_0$ mean", r"$H_0$ confidence bands"],
        frameon=True, framealpha=0.95,
    )
    _save_and_show(fig, output)


def _draw_comparison(
    ax:     plt.Axes,
    x:      FloatArray,
    y:      FloatArray,
    jk_err: FloatArray,
    bs_err: FloatArray,
    mean:   FloatArray,
    std:    FloatArray,
) -> tuple[object, object, object, object]:
    """Render bands + obs + both error bars + H0 mean. Returns legend handles."""
    _draw_h0_bands(ax, x, mean, std)

    obs_h = ax.scatter(x, y, s=45, facecolor=_OBS_RED,
                       edgecolor="white", linewidth=1.0, zorder=7)
    bs_h  = ax.errorbar(x, y, yerr=bs_err, fmt="none", ecolor="black",
                        elinewidth=2.0, capsize=3.0, capthick=1.2, alpha=0.90, zorder=4)
    jk_h  = ax.errorbar(x, y, yerr=jk_err, fmt="none", ecolor=_OBS_RED,
                        elinewidth=1.4, capsize=2.0, capthick=1.0, alpha=0.75, zorder=5)
    h0_h, = ax.plot(x, mean, color=_H0_LINE, lw=1.8, ls="--", zorder=6)

    return obs_h, jk_h, bs_h, h0_h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_results_with_jackknife(
    context:      RuntimeContext,
    theta:        FloatArray,
    w_obs:        FloatArray,
    abs_obs:      FloatArray,
    all_w_h0:     FloatArray,
    all_abs_h0:   FloatArray,
    jackknife:    JackknifeResult,
    output_2pacf: str | Path | None = None,
    output_abs:   str | Path | None = None,
) -> None:
    """Plot isotropy diagnostics with jackknife uncertainties and H0 bands."""
    d = _prepare(context, theta, w_obs, abs_obs, all_w_h0, all_abs_h0,
                 jackknife, "jackknife", output_2pacf, output_abs)

    _plot_2pacf(
        theta=d.theta, w_obs=d.w_obs, w_err=jackknife.w_err,
        h0_mean=d.h0_mean, h0_std=d.h0_std, h0_label=d.h0_label,
        err_label="Jackknife", err_color=_OBS_RED, output=d.output_2pacf,
    )
    _plot_absolute_sum(
        coarse_centers=d.coarse_centers,
        abs_obs=d.abs_obs, abs_err=jackknife.abs_err,
        h0_abs_mean=d.h0_abs_mean, h0_abs_std=d.h0_abs_std, h0_label=d.h0_label,
        err_label="Jackknife", err_color=_OBS_RED, output=d.output_abs,
    )


def plot_results_with_bootstrap(
    context:      RuntimeContext,
    theta:        FloatArray,
    w_obs:        FloatArray,
    abs_obs:      FloatArray,
    all_w_h0:     FloatArray,
    all_abs_h0:   FloatArray,
    bootstrap:    BootstrapResult,
    output_2pacf: str | Path | None = None,
    output_abs:   str | Path | None = None,
) -> None:
    """Plot isotropy diagnostics with bootstrap uncertainties and H0 bands."""
    d = _prepare(context, theta, w_obs, abs_obs, all_w_h0, all_abs_h0,
                 bootstrap, "bootstrap", output_2pacf, output_abs)

    _plot_2pacf(
        theta=d.theta, w_obs=d.w_obs, w_err=bootstrap.w_err,
        h0_mean=d.h0_mean, h0_std=d.h0_std, h0_label=d.h0_label,
        err_label="Bootstrap", err_color="black", output=d.output_2pacf,
    )
    _plot_absolute_sum(
        coarse_centers=d.coarse_centers,
        abs_obs=d.abs_obs, abs_err=bootstrap.abs_err,
        h0_abs_mean=d.h0_abs_mean, h0_abs_std=d.h0_abs_std, h0_label=d.h0_label,
        err_label="Bootstrap", err_color="black", output=d.output_abs,
    )


def plot_results_jk_vs_bootstrap(
    context:      RuntimeContext,
    theta:        FloatArray,
    w_obs:        FloatArray,
    abs_obs:      FloatArray,
    all_w_h0:     FloatArray,
    all_abs_h0:   FloatArray,
    jackknife:    JackknifeResult,
    bootstrap:    BootstrapResult,
    output_2pacf: str | Path | None = None,
    output_abs:   str | Path | None = None,
) -> None:
    """Compare jackknife and bootstrap uncertainty estimates side-by-side."""
    d = _prepare(context, theta, w_obs, abs_obs, all_w_h0, all_abs_h0,
                 jackknife, "jk_vs_bootstrap", output_2pacf, output_abs)

    if len(bootstrap.w_err)   != len(d.theta):
        raise ValueError("bootstrap.w_err and theta have incompatible shapes.")
    if len(bootstrap.abs_err) != len(d.abs_obs):
        raise ValueError("bootstrap.abs_err and abs_obs have incompatible shapes.")

    h0_patch = Patch(facecolor=_BAND_2, edgecolor=_BAND_1, alpha=0.85)

    # 2pACF
    fig, ax = plt.subplots(figsize=(11, 8))
    obs_h, jk_h, bs_h, h0_h = _draw_comparison(
        ax, d.theta, d.w_obs, jackknife.w_err, bootstrap.w_err, d.h0_mean, d.h0_std,
    )
    ax.axhline(0.0, color="0.15", linewidth=1.1, alpha=0.9, zorder=0)
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$w(\theta)$")
    ax.set_title(f"2pACF Uncertainty Comparison\n(Jackknife vs Bootstrap | {d.h0_label})")
    ax.legend(
        [obs_h, jk_h, bs_h, h0_h, h0_patch],
        ["Observed 2pACF", "Jackknife", "Bootstrap", r"$H_0$ mean", r"$H_0$ confidence bands"],
        frameon=True, framealpha=0.95,
    )
    _save_and_show(fig, d.output_2pacf)

    # Absolute sum
    if len(d.coarse_centers) != len(d.abs_obs):
        raise ValueError("coarse_centers and abs_obs have incompatible shapes.")
    fig, ax = plt.subplots(figsize=(11, 8))
    obs_h, jk_h, bs_h, h0_h = _draw_comparison(
        ax, d.coarse_centers, d.abs_obs,
        jackknife.abs_err, bootstrap.abs_err,
        d.h0_abs_mean, d.h0_abs_std,
    )
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$|\langle w \rangle|$")
    ax.set_title(f"Absolute Sum Test Uncertainty Comparison\n(Jackknife vs Bootstrap | {d.h0_label})")
    ax.legend(
        [obs_h, jk_h, bs_h, h0_h, h0_patch],
        ["Observed statistic", "Jackknife", "Bootstrap", r"$H_0$ mean", r"$H_0$ confidence bands"],
        frameon=True, framealpha=0.95,
    )
    _save_and_show(fig, d.output_abs)