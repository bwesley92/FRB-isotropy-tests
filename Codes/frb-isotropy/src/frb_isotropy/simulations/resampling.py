import numpy as np
import pandas as pd
import healpy as hp

from joblib import (
    Parallel,
    delayed,
)

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from ..core.types import FloatArray

from ..core.config import RuntimeContext
from ..core.models import (
    SelectionFunctionSet,
    JackknifeResult,
    BootstrapResult,
)

from .mocks import (
    generate_random_catalog,
)

from ..statistics.estimators import (
    compute_2pacf,
    get_absolute_sum,
)

from ..analysis.diagnostics import (
    plot_covariance_correlation_matrix,
)

from ..statistics.covariance import (
    build_shrinkage_covariance,
)


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


def remove_jackknife_region(
    df_data_with_regions: pd.DataFrame,
    region: int,
) -> pd.DataFrame:
    """
    Build one leave-one-region-out catalog.
    """

    if (
        "jackknife_region"
        not in df_data_with_regions.columns
    ):

        raise ValueError(
            "df_data_with_regions must contain "
            "'jackknife_region'."
        )

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

    return df_jk


def compute_jackknife_realization(
    context: RuntimeContext,
    df_jk: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    rng: np.random.Generator,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """
    Compute the statistics for one
    jackknife realization.
    """

    config = context.config

    df_rand_jk = (
        generate_random_catalog(
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

    theta_jk, w_jk = (
        compute_2pacf(
            context,
            df_jk,
            df_rand_jk,
        )
    )

    w_jk = np.nan_to_num(
        w_jk,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    abs_jk = get_absolute_sum(
        context,
        theta_jk,
        w_jk,
    )

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

    df_jk = remove_jackknife_region(
        df_data_with_regions=
            df_data_with_regions,
        region=region,
    )

    rng = np.random.default_rng(
        seed_base
        + int(region)
    )

    (
        theta_jk,
        w_jk,
        abs_jk,
    ) = compute_jackknife_realization(
        context=context,
        df_jk=df_jk,
        sf_set=sf_set,
        rng=rng,
    )

    return (
        theta_jk,
        w_jk,
        abs_jk,
        int(len(df_jk)),
    )


def validate_jackknife_samples(
    samples: FloatArray,
) -> tuple[
    int,
    int,
]:
    """
    Validate jackknife realization matrix.
    """

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

    return (
        int(n_regions),
        int(n_bins),
    )


def jackknife_covariance(
    samples: FloatArray,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
]:

    samples = np.asarray(
        samples,
        dtype=float,
    )

    n_regions, _ = (
        validate_jackknife_samples(
            samples
        )
    )

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
        centered.T
        @ centered
    )

    cov = np.asarray(
        cov,
        dtype=float,
    )

    cov = 0.5 * (
        cov
        + cov.T
    )

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

    return (
        np.asarray(
            mean,
            dtype=float,
        ),
        cov,
        err,
    )


def prepare_jackknife_analysis(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    nside_jackknife: int | None,
    min_regions: int | None,
    n_jobs: int | None,
    seed_base: int,
) -> tuple[
    int,
    pd.DataFrame,
    np.ndarray,
]:
    """
    Prepare jackknife analysis:
    runtime defaults, validation,
    diagnostics and region assignment.
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

    return (
        n_jobs,
        df_regions,
        regions,
    )


def run_jackknife_realizations(
    context: RuntimeContext,
    df_regions: pd.DataFrame,
    regions: np.ndarray,
    sf_set: SelectionFunctionSet | None,
    n_jobs: int,
    seed_base: int,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    np.ndarray,
]:
    """
    Execute all leave-one-region-out
    jackknife realizations and
    return the cleaned samples.
    """

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

    return (
        theta,
        w_samples,
        abs_samples,
        n_kept,
    )

def finalize_jackknife_statistics(
    context: RuntimeContext,
    regions: np.ndarray,
    theta: FloatArray,
    w_samples: FloatArray,
    abs_samples: FloatArray,
    n_kept: np.ndarray,
) -> JackknifeResult:
    """
    Compute jackknife covariances,
    diagnostics and final outputs.
    """

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

        title="Jackknife Correlation Matrix",

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


def run_jackknife_errors(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    nside_jackknife: int | None = None,
    min_regions: int | None = None,
    n_jobs: int | None = None,
    seed_base: int | None = None,
) -> JackknifeResult:
    """
    Estimate statistical uncertainties using
    leave-one-region-out jackknife resampling.
    """

    if seed_base is None:

        seed_base = (
            context.config.jackknife_seed_base
        )

    # ------------------------------------------------------------------
    # Preparation and region assignment
    # ------------------------------------------------------------------

    (
        n_jobs,
        df_regions,
        regions,
    ) = prepare_jackknife_analysis(
        context=context,
        df_data=df_data,
        sf_set=sf_set,
        n_jobs=n_jobs,
        nside_jackknife=nside_jackknife,
        min_regions=min_regions,
        seed_base=seed_base,
    )

    # ------------------------------------------------------------------
    # Leave-one-region realizations
    # ------------------------------------------------------------------

    (
        theta,
        w_samples,
        abs_samples,
        n_kept,
    ) = run_jackknife_realizations(
        context=context,
        df_regions=df_regions,
        regions=regions,
        sf_set=sf_set,
        n_jobs=n_jobs,
        seed_base=seed_base,
    )

    # ------------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------------

    return finalize_jackknife_statistics(
        context=context,
        regions=regions,
        theta=theta,
        w_samples=w_samples,
        abs_samples=abs_samples,
        n_kept=n_kept,
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


def prepare_bootstrap_analysis(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    n_bootstrap: int | None,
    n_jobs: int | None,
) -> tuple[
    int,
    int,
    pd.DataFrame,
]:
    """
    Prepare bootstrap analysis:
    runtime defaults, validation,
    diagnostics and fixed random catalog.
    """

    config = context.config

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
        generate_random_catalog(
            context=context,

            n_observed=(
                len(df_data)
                * config.n_rand_factor
            ),

            sf_set=sf_set,

            jitter_pixels=True,

            use_poisson=False,

            rng=np.random.default_rng(
                config.bootstrap_seed_base
            ),
        )
    )

    print(
        f"Random catalog size = "
        f"{len(df_rand_fixed)}"
    )

    return (
        n_jobs,
        n_bootstrap,
        df_rand_fixed,
    )


def run_bootstrap_realizations(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    df_rand_fixed: pd.DataFrame,
    n_bootstrap: int,
    n_jobs: int,
) -> tuple[
    FloatArray,
    FloatArray,
]:
    """
    Execute all bootstrap realizations
    and return cleaned samples.
    """

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

    return (
        w_realizations,
        abs_realizations,
    )


def finalize_bootstrap_statistics(
    context: RuntimeContext,
    w_realizations: FloatArray,
    abs_realizations: FloatArray,
) -> BootstrapResult:
    """
    Compute bootstrap covariances,
    diagnostics and final outputs.
    """

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

    w_cov = build_shrinkage_covariance(
        w_realizations
    )

    abs_cov = build_shrinkage_covariance(
        abs_realizations
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

        context=context,

        cov=w_cov,

        title="Bootstrap Correlation Matrix",

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

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    (
        n_jobs,
        n_bootstrap,
        df_rand_fixed,
    ) = prepare_bootstrap_analysis(
        context=context,
        df_data=df_data,
        sf_set=sf_set,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs,
    )

    # ------------------------------------------------------------------
    # Bootstrap realizations
    # ------------------------------------------------------------------

    (
        w_realizations,
        abs_realizations,
    ) = run_bootstrap_realizations(
        context=context,
        df_data=df_data,
        df_rand_fixed=df_rand_fixed,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs,
    )

    # ------------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------------

    return finalize_bootstrap_statistics(
        context=context,
        w_realizations=w_realizations,
        abs_realizations=abs_realizations,
    )