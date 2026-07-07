from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "frb_isotropy").exists():
            return path
    raise FileNotFoundError("Could not find frb-isotropy project root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CHIME instrumental-SF injection/recovery absorption tests.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "chime_only", "both"),
        default="both",
    )
    parser.add_argument("--n-realizations", type=int, default=20)
    parser.add_argument("--n-rand-factor", type=int, default=20)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--smooth-sigma", type=float, default=3.0)
    parser.add_argument("--eps-min", type=float, default=0.0)
    parser.add_argument("--eps-max", type=float, default=1.0)
    parser.add_argument("--eps-count", type=int, default=11)
    parser.add_argument("--axis-ra-deg", type=float, default=0.0)
    parser.add_argument("--axis-dec-deg", type=float, default=90.0)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def build_config(project_root: Path, *, mode: str, args: argparse.Namespace):
    from frb_isotropy.core.config import AnalysisConfig

    return AnalysisConfig(
        name=f"CHIMEInstrumentalSFAbsorption_{mode}",
        project_root=project_root,
        outputs_root=project_root / "outputs",
        catalog_path=project_root / "data" / "SkyPosition.csv",
        use_gal_mask=True,
        use_sel_func=True,
        n_jobs=args.n_jobs,
        gal_cut=20.0,
        min_sep=0.0,
        max_sep=180.0,
        bin_size=10.0,
        coarse_bins=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0),
        nside_sf=args.nside,
        smooth_sigma=args.smooth_sigma,
        perturbation_scale=1.0,
        n_rand_factor=args.n_rand_factor,
        n_ensemble=20,
        n_mocks_per_ensemble=50,
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
        n_bootstrap=500,
        overlap_radius_deg=5.0,
        overlap_nside=32,
        run_top_survey_maps=False,
    )


def build_instrumental_chime_sf_set(context, selection_functions):
    from frb_isotropy.selection import (
        build_chime_instrumental_selection_set,
        build_masked_chime_instrumental_sf,
    )
    from frb_isotropy.selection.instrumental import (
        ChimeInstrumentalComponents,
        analytic_chime_declination_exposure,
    )

    exposure = analytic_chime_declination_exposure(selection_functions.nside)
    components = ChimeInstrumentalComponents(exposure=exposure)

    sf_inst = build_masked_chime_instrumental_sf(
        context=context,
        components=components,
        nside=selection_functions.nside,
        gal_cut=context.config.gal_cut,
        use_gal_mask=context.config.use_gal_mask,
    )

    sf_set_inst = build_chime_instrumental_selection_set(
        empirical_sf_set=selection_functions,
        sf_instrumental=sf_inst,
        covariance_mode="zero",
    )

    return sf_inst, sf_set_inst


def result_table(suite, *, mode: str) -> pd.DataFrame:
    rows = []
    for r in suite.results:
        rows.append(
            {
                "mode": mode,
                "epsilon_injected": r.epsilon_injected,
                "signal_no_sf": r.signal_no_sf,
                "signal_no_sf_std": r.signal_no_sf_std,
                "signal_fixed_empirical_sf": r.signal_fixed_sf,
                "signal_fixed_empirical_sf_std": r.signal_fixed_sf_std,
                "signal_fixed_instrumental_sf": r.signal_fixed_instrumental_sf,
                "signal_fixed_instrumental_sf_std": r.signal_fixed_instrumental_sf_std,
                "signal_rebuilt_empirical_sf": r.signal_rebuilt_sf,
                "signal_rebuilt_empirical_sf_std": r.signal_rebuilt_sf_std,
                "retention_fixed_empirical_vs_no_sf": r.retention_fixed_vs_no_sf,
                "retention_fixed_instrumental_vs_no_sf": r.retention_instrumental_vs_no_sf,
                "retention_fixed_instrumental_vs_fixed_empirical": r.retention_instrumental_vs_fixed,
                "retention_rebuilt_empirical_vs_fixed_empirical": r.retention_rebuilt_vs_fixed,
                "absorption_rebuilt_empirical_vs_fixed_empirical": r.absorption_fraction,
                "absorption_fixed_instrumental_vs_no_sf": r.absorption_instrumental_vs_no_sf,
                "raw_no_sf": r.raw_no_sf,
                "raw_fixed_empirical_sf": r.raw_fixed_sf,
                "raw_fixed_instrumental_sf": r.raw_fixed_instrumental_sf,
                "raw_rebuilt_empirical_sf": r.raw_rebuilt_sf,
                "baseline_no_sf": r.baseline_no_sf,
                "baseline_fixed_empirical_sf": r.baseline_fixed_sf,
                "baseline_fixed_instrumental_sf": r.baseline_fixed_instrumental_sf,
                "baseline_rebuilt_empirical_sf": r.baseline_rebuilt_sf,
            }
        )
    return pd.DataFrame(rows)


def run_mode(project_root: Path, mode: str, args: argparse.Namespace) -> pd.DataFrame:
    from frb_isotropy.catalog.catalog import apply_mask, load_catalog
    from frb_isotropy.core.config import create_runtime_context
    from frb_isotropy.selection import (
        build_chime_only_selection_set,
        build_survey_selection_functions,
        extract_chime_empirical_sf,
    )
    from frb_isotropy.simulations.injection import run_injection_recovery_suite
    from frb_isotropy.visualization.plotting import plot_injection_recovery

    config = build_config(project_root, mode=mode, args=args)
    context = create_runtime_context(config)

    df_data = apply_mask(context, load_catalog(context))

    selection_functions = build_survey_selection_functions(
        context=context,
        df_data=df_data,
        nside=config.nside_sf,
        smooth_sigma=config.smooth_sigma,
    )

    sf_inst, sf_set_inst_full = build_instrumental_chime_sf_set(
        context,
        selection_functions,
    )

    if mode == "full":
        df_run = df_data
        sf_emp_run = selection_functions
        sf_inst_run = sf_set_inst_full
    elif mode == "chime_only":
        chime_mask = df_data["Reporting_Group_s"].astype(str).str.contains("CHIME", case=False, na=False)
        df_run = df_data.loc[chime_mask].reset_index(drop=True)
        sf_emp_run = build_chime_only_selection_set(
            extract_chime_empirical_sf(selection_functions),
            covariance=selection_functions.sf_cov_diag_dict["CHIME"],
        )
        sf_inst_run = build_chime_only_selection_set(sf_inst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    epsilons = np.linspace(args.eps_min, args.eps_max, args.eps_count)

    suite = run_injection_recovery_suite(
        context=context,
        df_data=df_run,
        sf_set=sf_emp_run,
        fixed_instrumental_sf_set=sf_inst_run,
        epsilons=epsilons,
        multipole=1,
        n_realizations=args.n_realizations,
        n_rand_factor=args.n_rand_factor,
        axis_ra_deg=args.axis_ra_deg,
        axis_dec_deg=args.axis_dec_deg,
        verbose_sf=False,
    )

    context.outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    context.outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    table = result_table(suite, mode=mode)
    table_path = context.outputs.tables_dir / "chime_instrumental_injection_recovery_summary.csv"
    figure_path = context.outputs.figures_dir / "chime_instrumental_injection_recovery.png"

    table.to_csv(table_path, index=False)
    plot_injection_recovery(suite, output_path=figure_path)

    print(f"Saved {mode} table: {table_path}")
    print(f"Saved {mode} figure: {figure_path}")

    return table


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    modes = ["full", "chime_only"] if args.mode == "both" else [args.mode]
    tables = [run_mode(project_root, mode, args) for mode in modes]

    if len(tables) > 1:
        combined = pd.concat(tables, ignore_index=True)
        combined_path = (
            project_root
            / "outputs"
            / "chime_instrumental_absorption_combined_summary.csv"
        )
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined table: {combined_path}")


if __name__ == "__main__":
    main()
