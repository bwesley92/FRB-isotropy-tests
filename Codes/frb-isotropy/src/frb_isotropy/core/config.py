from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from .types import (
    FloatArray,
    IntArray,
    BoolArray,
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

    random_seed: int

    mock_seed_base: int

    random_catalog_seed_base: int

    sf_perturbation_seed_base: int

    jackknife_seed_base: int

    bootstrap_seed_base: int

    # ------------------------------------------------------------------
    # Numerical stability
    # ------------------------------------------------------------------

    selection_function_floor: float

    numerical_eigenvalue_floor: float

    stable_eigenvalue_floor: float

    numerical_zero_tolerance: float

    # ------------------------------------------------------------------
    # Covariance regularization
    # ------------------------------------------------------------------

    svd_eigenvalue_cut: float

    # ------------------------------------------------------------------
    # Jackknife
    # ------------------------------------------------------------------

    nside_jackknife: int

    min_jackknife_regions: int

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    n_bootstrap: int

    # ------------------------------------------------------------------
    # Overlap analysis
    # ------------------------------------------------------------------

    overlap_radius_deg: float

    overlap_nside: int

    # ------------------------------------------------------------------
    # Optional pipeline products
    # ------------------------------------------------------------------

    run_top_survey_maps: bool

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
            f"max_sep ({config.max_sep}) "
            f"must exceed min_sep "
            f"({config.min_sep})."
        )

    if config.bin_size <= 0:

        raise ValueError(
            f"bin_size must be > 0 "
            f"(received {config.bin_size})."
        )

    if config.gal_cut < 0:

        raise ValueError(
            f"gal_cut must be >= 0 "
            f"(received {config.gal_cut})."
        )

    _ = config.n_bins

    # ------------------------------------------------------------------
    # Coarse-bin consistency
    # ------------------------------------------------------------------

    if len(config.coarse_bins) < 2:

        raise ValueError(
            "coarse_bins must contain at least "
            "two edges."
        )

    if np.any(
        np.diff(config.coarse_bins) <= 0
    ):

        raise ValueError(
            "coarse_bins must be strictly "
            "increasing."
        )

    if config.coarse_bins[0] < config.min_sep:

        raise ValueError(
            "First coarse-bin edge must be "
            "greater than or equal to min_sep."
        )

    if config.coarse_bins[-1] > config.max_sep:

        raise ValueError(
            "Last coarse-bin edge must be "
            "less than or equal to max_sep."
        )

    # ------------------------------------------------------------------
    # Numerical stability
    # ------------------------------------------------------------------

    positive_numerical_params = {

        "selection_function_floor":
            config.selection_function_floor,

        "numerical_eigenvalue_floor":
            config.numerical_eigenvalue_floor,

        "stable_eigenvalue_floor":
            config.stable_eigenvalue_floor,

        "numerical_zero_tolerance":
            config.numerical_zero_tolerance,
    }

    for name, value in (
        positive_numerical_params.items()
    ):

        if value <= 0:

            raise ValueError(
                f"Invalid AnalysisConfig: "
                f"'{name}' must be > 0 "
                f"(received {value})."
            )

    if (
        config.stable_eigenvalue_floor
        <
        config.numerical_eigenvalue_floor
    ):

        raise ValueError(
            "stable_eigenvalue_floor must be "
            "greater than or equal to "
            "numerical_eigenvalue_floor."
        )

    # ------------------------------------------------------------------
    # RNG validation
    # ------------------------------------------------------------------

    seed_params = {

        "random_seed":
            config.random_seed,

        "mock_seed_base":
            config.mock_seed_base,

        "random_catalog_seed_base":
            config.random_catalog_seed_base,

        "sf_perturbation_seed_base":
            config.sf_perturbation_seed_base,

        "jackknife_seed_base":
            config.jackknife_seed_base,

        "bootstrap_seed_base":
            config.bootstrap_seed_base,
    }

    for name, value in (
        seed_params.items()
    ):

        if value < 0:

            raise ValueError(
                f"Invalid AnalysisConfig: "
                f"'{name}' must be >= 0 "
                f"(received {value})."
            )

    # ------------------------------------------------------------------
    # HEALPix
    # ------------------------------------------------------------------

    healpix_params = {

        "nside_sf":
            config.nside_sf,

        "nside_jackknife":
            config.nside_jackknife,

        "overlap_nside":
            config.overlap_nside,
    }

    for name, value in (
        healpix_params.items()
    ):

        if not hp.isnsideok(value):

            raise ValueError(
                f"Invalid HEALPix nside: "
                f"'{name}' = {value}."
            )

    # ------------------------------------------------------------------
    # Positive-definite parameters
    # ------------------------------------------------------------------

    positive_params = {

        "n_jobs":
            config.n_jobs,

        "n_rand_factor":
            config.n_rand_factor,

        "n_ensemble":
            config.n_ensemble,

        "n_mocks_per_ensemble":
            config.n_mocks_per_ensemble,

        "n_bootstrap":
            config.n_bootstrap,

        "overlap_radius_deg":
            config.overlap_radius_deg,

        "svd_eigenvalue_cut":
            config.svd_eigenvalue_cut,
    }

    for name, value in (
        positive_params.items()
    ):

        if value <= 0:

            raise ValueError(
                f"Invalid AnalysisConfig: "
                f"'{name}' must be > 0 "
                f"(received {value})."
            )

    # ------------------------------------------------------------------
    # Selection-function configuration
    # ------------------------------------------------------------------

    if config.use_sel_func:

        if config.smooth_sigma <= 0:

            raise ValueError(
                "smooth_sigma must be > 0 "
                "when use_sel_func=True."
            )

        if config.perturbation_scale < 0:

            raise ValueError(
                "perturbation_scale must be "
                ">= 0 when use_sel_func=True."
            )

    # ------------------------------------------------------------------
    # Jackknife
    # ------------------------------------------------------------------

    if config.min_jackknife_regions < 2:

        raise ValueError(
            "min_jackknife_regions must be "
            "at least 2."
        )


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