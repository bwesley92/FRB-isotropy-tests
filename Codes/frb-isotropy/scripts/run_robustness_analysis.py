from __future__ import annotations

import argparse
import contextlib
import dataclasses
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FIDUCIAL_GAL_CUT = 20.0
FIDUCIAL_SMOOTH_SIGMA = 3.0
FIDUCIAL_NSIDE_SF = 32
FIDUCIAL_BIN_SIZE = 10.0
DEFAULT_STABILITY_BAND = 0.5
DEFAULT_REJECTION_SIGMA = 3.0

OAT_GRID: dict[str, tuple[float | int, ...]] = {
    "gal_cut": (15.0, 20.0, 25.0),
    "smooth_sigma": (1.0, 3.0, 5.0),
    "nside_sf": (16, 32, 64),
    "bin_size": (5.0, 10.0, 15.0),
}

SF_GRID_SMOOTH_SIGMA = (1.0, 3.0, 5.0)
SF_GRID_NSIDE = (16, 32, 64)


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "frb_isotropy").exists():
            return path
    raise FileNotFoundError("Could not find frb-isotropy project root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OAT and smooth_sigma x nside_sf robustness sweeps.",
    )
    parser.add_argument(
        "--include-mask-only",
        action="store_true",
        help="Also run the gal_cut OAT subset with mask enabled and SF disabled.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Destination CSV. Defaults to outputs/robustness_summary.csv.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Worker count passed to each pipeline run.",
    )
    parser.add_argument(
        "--n-ensemble",
        type=int,
        default=20,
        help="Number of H0 ensemble batches per run.",
    )
    parser.add_argument(
        "--n-mocks-per-ensemble",
        type=int,
        default=50,
        help="Mocks per H0 ensemble batch.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Bootstrap realizations per run.",
    )
    parser.add_argument(
        "--n-rand-factor",
        type=int,
        default=20,
        help="Random-catalog size multiplier.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs and write no outputs.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Read the summary CSV and regenerate paper tables/figures only.",
    )
    parser.add_argument(
        "--no-paper-products",
        action="store_true",
        help="Skip the supplementary tables and paper figures.",
    )
    parser.add_argument(
        "--paper-output-dir",
        type=Path,
        default=None,
        help="Destination directory for paper products. Defaults to outputs/robustness.",
    )
    parser.add_argument(
        "--stability-band",
        type=float,
        default=DEFAULT_STABILITY_BAND,
        help="Allowed absolute sigma variation around the fiducial value.",
    )
    parser.add_argument(
        "--rejection-sigma",
        type=float,
        default=DEFAULT_REJECTION_SIGMA,
        help="Qualitative rejection threshold used in the robustness table.",
    )
    return parser.parse_args()


def build_fiducial_config(project_root: Path, args: argparse.Namespace):
    from frb_isotropy.core.config import AnalysisConfig

    return AnalysisConfig(
        name="RobustnessFiducial",
        project_root=project_root,
        outputs_root=project_root / "outputs" / "robustness" / "runs",
        catalog_path=project_root / "data" / "SkyPosition.csv",
        use_gal_mask=True,
        use_sel_func=True,
        n_jobs=args.n_jobs,
        gal_cut=FIDUCIAL_GAL_CUT,
        min_sep=0.0,
        max_sep=180.0,
        bin_size=FIDUCIAL_BIN_SIZE,
        coarse_bins=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0),
        nside_sf=FIDUCIAL_NSIDE_SF,
        smooth_sigma=FIDUCIAL_SMOOTH_SIGMA,
        perturbation_scale=1.0,
        n_rand_factor=args.n_rand_factor,
        n_ensemble=args.n_ensemble,
        n_mocks_per_ensemble=args.n_mocks_per_ensemble,
        random_seed=12345,
        mock_seed_base=100000,
        random_catalog_seed_base=200000,
        sf_perturbation_seed_base=300000,
        jackknife_seed_base=400000,
        bootstrap_seed_base=500000,
        selection_function_floor=1e-12,
        numerical_eigenvalue_floor=1e-15,
        stable_eigenvalue_floor=1e-12,
        numerical_zero_tolerance=1e-30,
        svd_eigenvalue_cut=1e-2,
        nside_jackknife=4,
        min_jackknife_regions=20,
        n_bootstrap=args.n_bootstrap,
        overlap_radius_deg=5.0,
        overlap_nside=32,
        run_top_survey_maps=False,
    )


def tag_value(value: float | int | str) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def physical_key(config: Any) -> tuple:
    """
    Fields that actually change the pipeline result.

    Two runs sharing this key are statistically identical (same
    fixed seeds are used across the whole sweep), so the second one
    can safely reuse the first one's result instead of recomputing.
    """
    return (
        bool(config.use_gal_mask),
        bool(config.use_sel_func),
        float(config.gal_cut),
        float(config.smooth_sigma),
        int(config.nside_sf),
        float(config.bin_size),
    )


def build_runs(fiducial: Any, *, include_mask_only: bool) -> list[tuple[str, str, Any]]:
    runs: list[tuple[str, str, Any]] = []

    runs.append(
        (
            "fiducial",
            "fiducial",
            dataclasses.replace(fiducial, name="robustness_fiducial"),
        )
    )

    for parameter, values in OAT_GRID.items():
        for value in values:
            label = f"{parameter}_{tag_value(value)}"
            runs.append(
                (
                    "oat_mask_sf",
                    parameter,
                    dataclasses.replace(
                        fiducial,
                        name=f"robustness_oat_{label}",
                        **{parameter: value},
                    ),
                )
            )

    for smooth_sigma in SF_GRID_SMOOTH_SIGMA:
        for nside_sf in SF_GRID_NSIDE:
            label = f"smooth{tag_value(smooth_sigma)}_nside{nside_sf}"
            runs.append(
                (
                    "smooth_nside_grid",
                    "smooth_sigma_x_nside_sf",
                    dataclasses.replace(
                        fiducial,
                        name=f"robustness_grid_{label}",
                        smooth_sigma=smooth_sigma,
                        nside_sf=nside_sf,
                    ),
                )
            )

    if include_mask_only:
        mask_only = dataclasses.replace(
            fiducial,
            use_sel_func=False,
            name="robustness_mask_only_fiducial",
        )
        runs.append(("oat_mask_only", "fiducial", mask_only))
        for gal_cut in OAT_GRID["gal_cut"]:
            runs.append(
                (
                    "oat_mask_only",
                    "gal_cut",
                    dataclasses.replace(
                        mask_only,
                        name=f"robustness_mask_only_gal{tag_value(gal_cut)}",
                        gal_cut=gal_cut,
                    ),
                )
            )

    return runs


def get_nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def metric_value(obj: Any, path: str) -> float | int | None:
    value = get_nested_attr(obj, path)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def base_row(sweep: str, parameter: str, config: Any) -> dict[str, Any]:
    if parameter == "smooth_sigma_x_nside_sf":
        parameter_value = f"smooth_sigma={config.smooth_sigma:g};nside_sf={config.nside_sf}"
    elif parameter == "fiducial":
        parameter_value = "fiducial"
    else:
        parameter_value = get_nested_attr(config, parameter)

    return {
        "status": "success",
        "sweep": sweep,
        "parameter": parameter,
        "value": parameter_value,
        "run_tag": config.run_tag,
        "name": config.name,
        "use_gal_mask": config.use_gal_mask,
        "use_sel_func": config.use_sel_func,
        "gal_cut": config.gal_cut,
        "smooth_sigma": config.smooth_sigma,
        "nside_sf": config.nside_sf,
        "bin_size": config.bin_size,
        "random_seed": config.random_seed,
        "mock_seed_base": config.mock_seed_base,
        "random_catalog_seed_base": config.random_catalog_seed_base,
        "sf_perturbation_seed_base": config.sf_perturbation_seed_base,
        "jackknife_seed_base": config.jackknife_seed_base,
        "bootstrap_seed_base": config.bootstrap_seed_base,
        "error_type": None,
        "error_message": None,
    }


def summarize_success(sweep: str, parameter: str, config: Any, results: dict[str, Any]) -> dict[str, Any]:
    stats = results["stats"]
    row = base_row(sweep, parameter, config)

    metric_paths = {
        "svd_sigma_svd_equiv": "svd.sigma_svd_equiv",
        "svd_p_svd_empirical": "svd.p_svd_empirical",
        "svd_chi2_svd_red": "svd.chi2_svd_red",
        "chi2_sigma_equiv": "chi2.sigma_equiv",
        "chi2_p_empirical": "chi2.p_empirical",
        "ks_w_pvalue": "nonparametric.ks_w_pvalue",
        "ad_w_pvalue": "nonparametric.ad_w_pvalue",
        "ks_w_empirical_p": "nonparametric.ks_w_empirical_p",
        "ad_w_empirical_p": "nonparametric.ad_w_empirical_p",
        "abs_empirical_p": "absolute.abs_empirical_p",
        "absolute_global_tension": "absolute.global_tension",
        "hartlap_factor": "covariance_result.diagnostics.hartlap_factor",
        "covariance_rank": "covariance_result.diagnostics.covariance_rank",
        "covariance_condition": "covariance_result.diagnostics.covariance_condition",
        "n_eff": "covariance_result.diagnostics.n_eff",
        "svd_modes_kept": "covariance_result.diagnostics.svd_modes_kept",
        "svd_condition": "covariance_result.diagnostics.svd_condition",
        "n_mocks": "n_mocks",
        "n_bins": "n_bins",
    }

    row.update(
        {
            column: metric_value(stats, path)
            for column, path in metric_paths.items()
        }
    )
    return row


def summarize_failure(
    sweep: str,
    parameter: str,
    config: Any,
    exc: Exception,
) -> dict[str, Any]:
    row = base_row(sweep, parameter, config)
    row.update(
        {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )
    return row


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved robustness summary: {path}")




def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return df.copy()
    return df.loc[df["status"].eq("success")].copy()


def fiducial_sigma(df: pd.DataFrame) -> float:
    ok = successful_rows(df)
    fid = ok.loc[ok["sweep"].eq("fiducial")]
    if fid.empty:
        fid = ok.loc[
            ok["use_gal_mask"].eq(True)
            & ok["use_sel_func"].eq(True)
            & np.isclose(ok["gal_cut"].astype(float), FIDUCIAL_GAL_CUT)
            & np.isclose(ok["smooth_sigma"].astype(float), FIDUCIAL_SMOOTH_SIGMA)
            & ok["nside_sf"].astype(int).eq(FIDUCIAL_NSIDE_SF)
            & np.isclose(ok["bin_size"].astype(float), FIDUCIAL_BIN_SIZE)
        ]
    if fid.empty:
        raise ValueError("No successful fiducial robustness row found.")
    return float(fid.iloc[0]["svd_sigma_svd_equiv"])


def paper_columns() -> list[str]:
    return [
        "status",
        "sweep",
        "parameter",
        "value",
        "run_tag",
        "cache_source",
        "use_gal_mask",
        "use_sel_func",
        "gal_cut",
        "smooth_sigma",
        "nside_sf",
        "bin_size",
        "svd_sigma_svd_equiv",
        "svd_p_svd_empirical",
        "svd_chi2_svd_red",
        "chi2_sigma_equiv",
        "chi2_p_empirical",
        "ks_w_pvalue",
        "ad_w_pvalue",
        "ks_w_empirical_p",
        "ad_w_empirical_p",
        "abs_empirical_p",
        "absolute_global_tension",
        "hartlap_factor",
        "covariance_condition",
        "n_eff",
        "svd_modes_kept",
        "n_mocks",
        "n_bins",
        "error_type",
        "error_message",
    ]


def write_supplementary_table(df: pd.DataFrame, tables_dir: Path) -> Path:
    available = [column for column in paper_columns() if column in df.columns]
    table = df.loc[:, available].copy()
    table_path = tables_dir / "robustness_supplementary_table.csv"
    latex_path = tables_dir / "robustness_supplementary_table.tex"
    table.to_csv(table_path, index=False)
    table.to_latex(latex_path, index=False, float_format="%.4g")
    return table_path


def write_robustness_criterion(
    df: pd.DataFrame,
    tables_dir: Path,
    *,
    stability_band: float,
    rejection_sigma: float,
) -> Path:
    ok = successful_rows(df)
    paper = ok.loc[ok["use_gal_mask"].eq(True) & ok["use_sel_func"].eq(True)].copy()
    fid_sigma = fiducial_sigma(paper)
    paper["delta_sigma_vs_fiducial"] = paper["svd_sigma_svd_equiv"].astype(float) - fid_sigma
    paper["abs_delta_sigma_vs_fiducial"] = paper["delta_sigma_vs_fiducial"].abs()
    paper["rejects_isotropy"] = paper["svd_sigma_svd_equiv"].astype(float) >= rejection_sigma

    fid_rejects = fid_sigma >= rejection_sigma
    qualitative_stable = bool(paper["rejects_isotropy"].eq(fid_rejects).all())
    max_abs_delta = float(paper["abs_delta_sigma_vs_fiducial"].max())
    band_stable = max_abs_delta <= stability_band

    if "cache_source" in paper.columns:
        n_unique_physical_configs = int(paper["cache_source"].isna().sum())
    else:
        n_unique_physical_configs = int(len(paper))

    summary = pd.DataFrame(
        [
            {
                "fiducial_sigma_svd_equiv": fid_sigma,
                "rejection_sigma": rejection_sigma,
                "stability_band": stability_band,
                "max_abs_delta_sigma": max_abs_delta,
                "fiducial_rejects_isotropy": fid_rejects,
                "qualitative_status_stable": qualitative_stable,
                "within_stability_band": band_stable,
                "robust_by_predefined_criterion": qualitative_stable and band_stable,
                "n_successful_paper_runs": int(len(paper)),
                "n_unique_physical_configs": n_unique_physical_configs,
                "n_failed_runs": int(df["status"].eq("failed").sum()) if "status" in df.columns else 0,
            }
        ]
    )

    criterion_path = tables_dir / "robustness_criterion_summary.csv"
    deltas_path = tables_dir / "robustness_sigma_deltas.csv"
    summary.to_csv(criterion_path, index=False)
    paper.to_csv(deltas_path, index=False)
    return criterion_path


def plot_sensitivity(df: pd.DataFrame, figures_dir: Path, *, stability_band: float) -> Path:
    import matplotlib.pyplot as plt

    ok = successful_rows(df)
    oat = ok.loc[ok["sweep"].eq("oat_mask_sf")].copy()
    if oat.empty:
        raise ValueError("No successful OAT mask+SF rows available for sensitivity plot.")

    fid_sigma = fiducial_sigma(ok)
    parameters = ["gal_cut", "smooth_sigma", "nside_sf", "bin_size"]
    labels = {
        "gal_cut": "Galactic cut (deg)",
        "smooth_sigma": "SF smoothing (deg)",
        "nside_sf": "SF Nside",
        "bin_size": "2pACF bin width (deg)",
    }
    fid_values = {
        "gal_cut": FIDUCIAL_GAL_CUT,
        "smooth_sigma": FIDUCIAL_SMOOTH_SIGMA,
        "nside_sf": FIDUCIAL_NSIDE_SF,
        "bin_size": FIDUCIAL_BIN_SIZE,
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharey=True)
    axes = axes.ravel()

    for ax, parameter in zip(axes, parameters):
        part = oat.loc[oat["parameter"].eq(parameter)].copy()
        part[parameter] = part[parameter].astype(float)
        part = part.sort_values(parameter)
        x = part[parameter].to_numpy(dtype=float)
        y = part["svd_sigma_svd_equiv"].to_numpy(dtype=float)

        for lower, upper, shade in [
            (-3.0, -2.0, "0.94"),
            (-2.0, -1.0, "0.88"),
            (-1.0, 1.0, "0.80"),
            (1.0, 2.0, "0.88"),
            (2.0, 3.0, "0.94"),
        ]:
            ax.axhspan(lower, upper, color=shade, zorder=0)

        ax.axhline(0.0, color="0.18", lw=1.35, ls="--", zorder=1)
        ax.plot(x, y, color="#1f77b4", lw=1.9, zorder=3)

        fid_mask = np.isclose(x, float(fid_values[parameter]))
        ax.scatter(
            x[~fid_mask],
            y[~fid_mask],
            s=40,
            facecolors="white",
            edgecolors="#1f77b4",
            linewidths=1.5,
            zorder=4,
        )
        ax.scatter(
            x[fid_mask],
            y[fid_mask],
            s=46,
            facecolors="#d62728",
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )
        ax.set_xlabel(labels[parameter])
        if parameter == "smooth_sigma":
            ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])

        y_limit = 6.0
        ax.set_ylim(-y_limit, y_limit)
        ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
        ax.grid(False)
        ax.set_axisbelow(True)

        # ------------------------------------------------------------
        # Flag points that fall outside the fixed y-range instead of
        # letting them silently disappear off the top/bottom edge.
        # ------------------------------------------------------------

        out_of_range = np.abs(y) > y_limit

        for xi, yi in zip(x[out_of_range], y[out_of_range]):
            edge = y_limit - 0.4 if yi > 0 else -y_limit + 0.4
            marker = "^" if yi > 0 else "v"
            ax.scatter([xi], [edge], marker=marker, s=70,
                       facecolors="#d62728", edgecolors="white",
                       linewidths=0.9, zorder=6)
            ax.annotate(f"{yi:.1f}", (xi, edge),
                        textcoords="offset points", xytext=(0, 6 if yi > 0 else -12),
                        ha="center", fontsize=7.5, color="#d62728")

        if np.any(out_of_range):
            print(
                f"WARNING: {parameter} has "
                f"{int(np.sum(out_of_range))} point(s) outside the "
                f"+/-{y_limit:g} sigma plot range "
                f"(values: {y[out_of_range].tolist()}); "
                f"marked with a triangle at the axis edge."
            )

    axes[0].set_ylabel("Signed SVD tension")
    axes[2].set_ylabel("Signed SVD tension")
    fig.suptitle("Robustness of the signed SVD isotropy tension")
    fig.tight_layout()

    png_path = figures_dir / "robustness_sensitivity_oat.png"
    pdf_path = figures_dir / "robustness_sensitivity_oat.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_smooth_nside_heatmap(df: pd.DataFrame, figures_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    ok = successful_rows(df)
    grid = ok.loc[ok["sweep"].eq("smooth_nside_grid")].copy()
    if grid.empty:
        raise ValueError("No successful smooth_sigma x nside_sf rows available for heatmap.")

    grid["smooth_sigma"] = grid["smooth_sigma"].astype(float)
    grid["nside_sf"] = grid["nside_sf"].astype(int)
    pivot = grid.pivot_table(
        index="smooth_sigma",
        columns="nside_sf",
        values="svd_sigma_svd_equiv",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(6.3, 4.9))
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        origin="lower",
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )
    fig.suptitle(
        "Signed SVD tension across SF resolution and smoothing",
        fontsize=10,
        y=0.965,
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns], fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index], fontsize=9)
    ax.set_xlabel("SF Nside", fontsize=10, labelpad=6)
    ax.set_ylabel("SF smoothing sigma (deg)", fontsize=10, labelpad=6)

    for iy in range(pivot.shape[0]):
        for ix in range(pivot.shape[1]):
            value = pivot.to_numpy(dtype=float)[iy, ix]
            if np.isfinite(value):
                ax.text(
                    ix,
                    iy,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9.5,
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.045)
    cbar.set_label("Signed SVD tension", fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925), pad=0.8)

    png_path = figures_dir / "robustness_smooth_nside_heatmap.png"
    pdf_path = figures_dir / "robustness_smooth_nside_heatmap.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def make_paper_products(
    summary_path: Path,
    output_dir: Path,
    *,
    stability_band: float,
    rejection_sigma: float,
) -> None:
    df = pd.read_csv(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_path = write_supplementary_table(df, tables_dir)
    criterion_path = write_robustness_criterion(
        df,
        tables_dir,
        stability_band=stability_band,
        rejection_sigma=rejection_sigma,
    )
    sensitivity_path = plot_sensitivity(df, figures_dir, stability_band=stability_band)
    heatmap_path = plot_smooth_nside_heatmap(df, figures_dir)

    manifest = pd.DataFrame(
        [
            {"product": "supplementary_table", "path": str(table_path)},
            {"product": "criterion_summary", "path": str(criterion_path)},
            {"product": "sensitivity_plot", "path": str(sensitivity_path)},
            {"product": "smooth_nside_heatmap", "path": str(heatmap_path)},
        ]
    )
    manifest_path = output_dir / "robustness_paper_products_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved robustness paper products: {output_dir}")


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    paper_output_dir = args.paper_output_dir or (project_root / "outputs" / "robustness")
    summary_path = args.summary_path or (paper_output_dir / "robustness_summary.csv")
    logs_dir = paper_output_dir / "logs"

    if args.plots_only:
        make_paper_products(
            summary_path,
            paper_output_dir,
            stability_band=args.stability_band,
            rejection_sigma=args.rejection_sigma,
        )
        return

    from frb_isotropy.pipeline import main as run_pipeline

    fiducial = build_fiducial_config(project_root, args)
    runs = build_runs(fiducial, include_mask_only=args.include_mask_only)

    print(f"Planned robustness runs: {len(runs)}")
    for index, (sweep, parameter, config) in enumerate(runs, start=1):
        print(f"{index:03d} {sweep:18s} {parameter:24s} {config.run_tag}")

    if args.dry_run:
        return

    logs_dir.mkdir(parents=True, exist_ok=True)

    sweep_totals = Counter(sweep for sweep, _, _ in runs)
    sweep_progress: Counter = Counter()

    # Physical configuration -> (source run_tag, pipeline results).
    # Several runs (e.g. the OAT value equal to the fiducial one)
    # are statistically identical given the fixed seeds, so reuse
    # the first computed result instead of paying for the full
    # pipeline again.
    cache: dict[tuple, tuple[str, dict[str, Any]]] = {}

    rows: list[dict[str, Any]] = []
    for index, (sweep, parameter, config) in enumerate(runs, start=1):
        sweep_progress[sweep] += 1
        tag = (
            f"[{sweep} {sweep_progress[sweep]}/{sweep_totals[sweep]}] "
            f"(overall {index}/{len(runs)})"
        )

        key = physical_key(config)
        cached = cache.get(key)
        start_time = time.perf_counter()

        if cached is not None:
            source_tag, results = cached
            print(f"{tag} {config.run_tag} -- reusing result from "
                  f"{source_tag} (identical configuration, not recomputed)")
        else:
            print(f"{tag} Running {config.run_tag}")
            log_path = logs_dir / f"{config.run_tag}.log"
            try:
                with log_path.open("w", encoding="utf-8") as log_file:
                    with contextlib.redirect_stdout(log_file), \
                            contextlib.redirect_stderr(log_file):
                        results = run_pipeline(config)
            except Exception as exc:
                elapsed_min = (time.perf_counter() - start_time) / 60.0
                print(f"{tag} FAILED {config.run_tag} after "
                      f"{elapsed_min:.1f} min: {type(exc).__name__}: {exc} "
                      f"(see {log_path})")
                rows.append(summarize_failure(sweep, parameter, config, exc))
                write_summary(rows, summary_path)
                continue
            cache[key] = (config.run_tag, results)

        elapsed_min = (time.perf_counter() - start_time) / 60.0
        row = summarize_success(sweep, parameter, config, results)
        row["cache_source"] = None if cached is None else source_tag
        sigma = row.get("svd_sigma_svd_equiv")
        sigma_text = f"{sigma:.2f}" if isinstance(sigma, (int, float)) else "n/a"
        print(f"{tag} done in {elapsed_min:.1f} min -- sigma_svd={sigma_text}")
        rows.append(row)
        write_summary(rows, summary_path)

    if not args.no_paper_products:
        make_paper_products(
            summary_path,
            paper_output_dir,
            stability_band=args.stability_band,
            rejection_sigma=args.rejection_sigma,
        )


if __name__ == "__main__":
    main()
