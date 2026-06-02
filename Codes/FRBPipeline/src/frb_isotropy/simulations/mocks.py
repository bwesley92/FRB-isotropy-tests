from __future__ import annotations

import healpy as hp
import numpy as np
import pandas as pd


from joblib import (
    Parallel,
    delayed,
)

from ..catalog.catalog import (
    icrs_to_galactic_b,
)

from numpy.random import Generator

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from ..core.config import RuntimeContext

from ..core.models import (
    MockEnsemble,
    SelectionFunctionSet,
)

from ..selection.selection import (
    generate_sf_variant,
)

from ..statistics.estimators import (
    compute_2pacf,
    get_absolute_sum,
)

from ..core.types import FloatArray


# ==============================================================================
# Mock catalog generation
# ==============================================================================

def _jitter_pixel_centers(
    context: RuntimeContext,
    ra: FloatArray,
    dec: FloatArray,
    nside: int,
    rng: Generator,
) -> tuple[FloatArray, FloatArray]:
    """
    Randomize positions inside HEALPix pixels.

    This suppresses artificial angular quantization
    produced by finite HEALPix resolution.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    if not isinstance(
        rng,
        Generator,
    ):

        raise TypeError(
            "rng must be a numpy.random.Generator."
        )

    if len(ra) != len(dec):

        raise ValueError(
            "ra and dec must have "
            "the same length."
        )

    if not np.all(
        np.isfinite(ra)
    ):

        raise ValueError(
            "ra contains non-finite values."
        )

    if not np.all(
        np.isfinite(dec)
    ):

        raise ValueError(
            "dec contains non-finite values."
        )

    # ------------------------------------------------------------------
    # HEALPix angular scale
    # ------------------------------------------------------------------

    pix_radius_deg = np.degrees(
        hp.max_pixrad(nside)
    )

    # ------------------------------------------------------------------
    # Declination jitter
    # ------------------------------------------------------------------

    dec_jitter = rng.uniform(
        -pix_radius_deg,
        pix_radius_deg,
        size=len(dec),
    )

    # ------------------------------------------------------------------
    # Right ascension jitter
    # ------------------------------------------------------------------

    cos_dec = np.clip(
        np.cos(
            np.radians(dec)
        ),
        config.numerical_zero_tolerance,
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

    # ------------------------------------------------------------------
    # Final coordinates
    # ------------------------------------------------------------------

    ra_out = (
        ra + ra_jitter
    ) % 360.0

    dec_out = np.clip(
        dec + dec_jitter,
        -89.999,
        89.999,
    )

    return (
        np.asarray(
            ra_out,
            dtype=float,
        ),
        np.asarray(
            dec_out,
            dtype=float,
        ),
    )


# ==============================================================================
# Isotropic mock catalog generation
# ==============================================================================

def generate_random_catalog(
    context: RuntimeContext,
    n_observed: int,
    sf_set: SelectionFunctionSet | None = None,
    jitter_pixels: bool = True,
    use_poisson: bool = False,
    rng: Generator | None = None,
) -> pd.DataFrame:
    
    """
    Generate isotropic mock catalogs.

    Modes
    -----
    sf_set is None
        Pure isotropic sky realization.

    sf_set exists
        Isotropic realization filtered through
        survey selection functions.
    """

    config = context.config

    # ------------------------------------------------------------------
    # RNG validation
    # ------------------------------------------------------------------

    if rng is None:

        raise ValueError(
            "rng must be explicitly provided."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if n_observed <= 0:

        return pd.DataFrame(
            columns=[
                "RA",
                "DEC",
            ]
        )

    # ------------------------------------------------------------------
    # Catalog size
    # ------------------------------------------------------------------

    if use_poisson:

        n = int(
            rng.poisson(
                n_observed
            )
        )

    else:

        n = int(
            n_observed
        )

    if n <= 0:

        return pd.DataFrame(
            columns=[
                "RA",
                "DEC",
            ]
        )

    # ==========================================================================
    # PURE ISOTROPIC SKY
    # ==========================================================================

    if sf_set is None:

        ra_all: list[float] = []

        dec_all: list[float] = []

        while len(ra_all) < n:

            n_remaining = (
                n
                - len(ra_all)
            )

            # --------------------------------------------------------------
            # Uniform isotropic sphere
            # --------------------------------------------------------------

            ra = (
                360.0
                * rng.random(
                    n_remaining
                )
            )

            dec = np.degrees(
                np.arcsin(
                    2.0
                    * rng.random(
                        n_remaining
                    )
                    - 1.0
                )
            )

            # --------------------------------------------------------------
            # Galactic masking
            # --------------------------------------------------------------

            if config.use_gal_mask:

                gal_b = icrs_to_galactic_b(
                    ra,
                    dec,
                )

                keep = (
                    np.abs(gal_b)
                    > config.gal_cut
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
                "RA": np.asarray(
                    ra_all[:n],
                    dtype=float,
                ),

                "DEC": np.asarray(
                    dec_all[:n],
                    dtype=float,
                ),
            }
        )

    # ==========================================================================
    # SELECTION-FUNCTION SKY
    # ==========================================================================

    surveys = np.asarray(
        list(
            sf_set.sf_dict.keys()
        )
    )

    survey_probs = np.asarray(
        [
            sf_set.survey_weights[s]
            for s in surveys
        ],
        dtype=float,
    )

    prob_sum = survey_probs.sum()

    if prob_sum <= 0:

        raise ValueError(
            "Survey weights sum to zero."
        )

    survey_probs /= prob_sum

    chosen_surveys = rng.choice(
        surveys,
        size=n,
        p=survey_probs,
    )

    ra = np.empty(
        n,
        dtype=float,
    )

    dec = np.empty(
        n,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Survey realizations
    # ------------------------------------------------------------------

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
            sf_set.sf_dict[
                survey
            ],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if not np.all(
            np.isfinite(sf)
        ):

            raise ValueError(
                f"Non-finite values detected "
                f"in survey SF: {survey}"
            )

        sf = np.clip(
            sf,
            0.0,
            None,
        )

        sf_sum = sf.sum()

        if sf_sum <= 0:

            raise ValueError(
                f"Selection function vanished "
                f"for survey: {survey}"
            )

        sf /= sf_sum

        # --------------------------------------------------------------
        # HEALPix sampling
        # --------------------------------------------------------------

        pix = rng.choice(
            len(sf),
            size=n_survey,
            p=sf,
        )

        theta, phi = hp.pix2ang(
            sf_set.nside,
            pix,
        )

        ra_survey = np.degrees(
            phi
        )

        dec_survey = (
            90.0
            - np.degrees(theta)
        )

        # --------------------------------------------------------------
        # Sub-pixel randomization
        # --------------------------------------------------------------

        if jitter_pixels:

            ra_survey, dec_survey = (

                _jitter_pixel_centers(

                    context=context,

                    ra=np.asarray(
                        ra_survey,
                        dtype=float,
                    ),

                    dec=np.asarray(
                        dec_survey,
                        dtype=float,
                    ),

                    nside=sf_set.nside,

                    rng=rng,
                )
            )

        ra[mask] = ra_survey

        dec[mask] = dec_survey

    return pd.DataFrame(
        {
            "RA": np.asarray(
                ra,
                dtype=float,
            ),

            "DEC": np.asarray(
                dec,
                dtype=float,
            ),
        }
    )


# Backwards-compatible alias for older notebooks/scripts.
generate_mixture_catalog_improved = generate_random_catalog


# ==============================================================================
# Independent mock realization
# ==============================================================================

def _run_one_independent_mock(
    context: RuntimeContext,
    sf_set: SelectionFunctionSet | None,
    seed_data: int,
    seed_rand: int,
    n_data: int,
    n_rand_factor: int,
) -> tuple[
    FloatArray,
    FloatArray,
]:
    """
    Generate one statistically independent
    isotropic mock realization.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if n_data <= 0:

        raise ValueError(
            "n_data must be positive."
        )

    if n_rand_factor <= 0:

        raise ValueError(
            "n_rand_factor must be positive."
        )

    # ------------------------------------------------------------------
    # Independent RNG streams
    # ------------------------------------------------------------------

    seed_sequence = np.random.SeedSequence(
        [
            seed_data,
            seed_rand,
        ]
    )

    child_sequences = seed_sequence.spawn(2)

    rng_data = np.random.default_rng(
        child_sequences[0]
    )

    rng_rand = np.random.default_rng(
        child_sequences[1]
    )

    # ------------------------------------------------------------------
    # Data catalog
    # ------------------------------------------------------------------

    d_iso = (
        generate_random_catalog(
            context=context,

            n_observed=n_data,

            sf_set=sf_set,

            jitter_pixels=True,

            use_poisson=False,

            rng=rng_data,
        )
    )

    # ------------------------------------------------------------------
    # Random catalog
    # ------------------------------------------------------------------

    r_iso = (
        generate_random_catalog(
            context=context,

            n_observed=(
                n_data
                * n_rand_factor
            ),

            sf_set=sf_set,

            jitter_pixels=True,

            use_poisson=False,

            rng=rng_rand,
        )
    )

    # ------------------------------------------------------------------
    # 2pACF
    # ------------------------------------------------------------------

    theta_h0, w_h0 = (
        compute_2pacf(
            context,
            d_iso,
            r_iso,
        )
    )

    # ------------------------------------------------------------------
    # Tomographic absolute anisotropy estimator
    # ------------------------------------------------------------------

    abs_h0 = get_absolute_sum(
        context,
        theta_h0,
        w_h0,
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    return (

        np.asarray(
            w_h0,
            dtype=float,
        ),

        np.asarray(
            abs_h0,
            dtype=float,
        ),
    )


# ==============================================================================
# Ensemble mock generation
# ==============================================================================

def run_ensemble_mocks(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    sf_set: SelectionFunctionSet | None,
    n_ensemble: int | None = None,
    n_mocks_per: int | None = None,
    perturbation_scale: float | None = None,
    n_rand_factor: int | None = None,
    n_jobs: int | None = None,
) -> MockEnsemble:
    """
    Generate isotropic H0 mock ensembles.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if n_jobs is None:

        n_jobs = config.n_jobs

    if n_ensemble is None:

        n_ensemble = (
            config.n_ensemble
        )

    if n_mocks_per is None:

        n_mocks_per = (
            config.n_mocks_per_ensemble
        )

    if perturbation_scale is None:

        perturbation_scale = (
            config.perturbation_scale
        )

    if n_rand_factor is None:

        n_rand_factor = (
            config.n_rand_factor
        )

    n_jobs = max(
        1,
        int(n_jobs),
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if n_ensemble <= 0:

        raise ValueError(
            "n_ensemble must be positive."
        )

    if n_mocks_per <= 0:

        raise ValueError(
            "n_mocks_per must be positive."
        )

    if n_rand_factor <= 0:

        raise ValueError(
            "n_rand_factor must be positive."
        )

    if perturbation_scale < 0:

        raise ValueError(
            "perturbation_scale must be "
            "non-negative."
        )

    if len(df_data) <= 0:

        raise ValueError(
            "Observed catalog is empty."
        )

    # ------------------------------------------------------------------
    # Master seed sequence
    # ------------------------------------------------------------------

    master_seed_sequence = (
        np.random.SeedSequence(
            config.random_seed
        )
    )

    total_mocks = (
        n_ensemble
        * n_mocks_per
    )

    # Independent random trees:
    #
    # master_seed
    # ├── sf_root
    # │   └── sf_seed_sequences
    # └── mock_root
    #     └── mock_seed_sequences
    #
    # This guarantees that selection-function perturbations and
    # isotropic mock realizations are generated from independent
    # SeedSequence branches.

    sf_root, mock_root = (
        master_seed_sequence.spawn(2)
    )

    sf_seed_sequences = (
        sf_root.spawn(
            n_ensemble
        )
    )

    mock_seed_sequences = (
        mock_root.spawn(
            total_mocks
        )
    )

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    tasks: list[
        tuple[
            SelectionFunctionSet | None,
            int,
            int,
        ]
    ] = []

    # ==========================================================================
    # SELECTION-FUNCTION MODE
    # ==========================================================================

    if config.use_sel_func:

        if sf_set is None:

            raise ValueError(
                "sf_set cannot be None "
                "when use_sel_func=True."
            )

        print(
            "\n--- Generating H0 ensemble mocks "
            "(with selection functions) ---"
        )

        task_index = 0

        for i_ens in range(
            n_ensemble
        ):

            # --------------------------------------------------------------
            # Independent ensemble RNG
            # --------------------------------------------------------------

            ensemble_rng = (
                np.random.default_rng(
                    sf_seed_sequences[
                        i_ens
                    ]
                )
            )

            # --------------------------------------------------------------
            # Shared SF realization
            # --------------------------------------------------------------

            sf_variant_dict = (
                generate_sf_variant(

                    context=context,

                    sf_set=sf_set,

                    perturbation_scale=(
                        perturbation_scale
                    ),

                    rng=ensemble_rng,
                )
            )

            # --------------------------------------------------------------
            # Rebuild immutable SF container
            # --------------------------------------------------------------

            sf_variant_set = (
                SelectionFunctionSet(

                    sf_dict=(
                        sf_variant_dict
                    ),

                    sf_cov_diag_dict=(
                        sf_set.sf_cov_diag_dict
                    ),

                    survey_weights=(
                        sf_set.survey_weights
                    ),

                    nside=(
                        sf_set.nside
                    ),
                )
            )

            # --------------------------------------------------------------
            # Mock realizations
            # --------------------------------------------------------------

            for i_mock in range(
                n_mocks_per
            ):

                mock_ss = (
                    mock_seed_sequences[
                        task_index
                    ]
                )

                child_ss = (
                    mock_ss.spawn(2)
                )

                seed_data = int(
                    child_ss[0]
                    .generate_state(1)[0]
                )

                seed_rand = int(
                    child_ss[1]
                    .generate_state(1)[0]
                )

                tasks.append(
                    (
                        sf_variant_set,
                        seed_data,
                        seed_rand,
                    )
                )

                task_index += 1

    # ==========================================================================
    # PURE ISOTROPIC SKY
    # ==========================================================================

    else:

        print(
            "\n--- Generating pure isotropic "
            "H0 mocks ---"
        )

        for i_mock in range(
            total_mocks
        ):

            mock_ss = (
                mock_seed_sequences[
                    i_mock
                ]
            )

            child_ss = (
                mock_ss.spawn(2)
            )

            seed_data = int(
                child_ss[0]
                .generate_state(1)[0]
            )

            seed_rand = int(
                child_ss[1]
                .generate_state(1)[0]
            )

            tasks.append(
                (
                    None,
                    seed_data,
                    seed_rand,
                )
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    n_total = len(
        tasks
    )

    if n_total == 0:

        raise RuntimeError(
            "No mock tasks were generated."
        )

    print(
        f"Total H0 mocks: {n_total}"
    )

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

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
            delayed(
                _run_one_independent_mock
            )(
                context=context,

                sf_set=task_sf_set,

                seed_data=seed_data,

                seed_rand=seed_rand,

                n_data=len(df_data),

                n_rand_factor=(
                    n_rand_factor
                ),
            )

            for (
                task_sf_set,
                seed_data,
                seed_rand,
            ) in tasks
        )

    if len(results) == 0:

        raise RuntimeError(
            "No mock realizations generated."
        )

    # ------------------------------------------------------------------
    # Stack outputs
    # ------------------------------------------------------------------

    all_w_h0, all_abs_h0 = zip(
        *results
    )

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    all_abs_h0 = np.asarray(
        all_abs_h0,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if all_w_h0.ndim != 2:

        raise RuntimeError(
            "all_w_h0 must be 2-dimensional."
        )

    if all_abs_h0.ndim != 2:

        raise RuntimeError(
            "all_abs_h0 must be 2-dimensional."
        )

    if not np.all(
        np.isfinite(all_w_h0)
    ):

        raise RuntimeError(
            "Non-finite values detected "
            "in all_w_h0."
        )

    if not np.all(
        np.isfinite(all_abs_h0)
    ):

        raise RuntimeError(
            "Non-finite values detected "
            "in all_abs_h0."
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print(
        "\nGenerated mocks:"
    )

    print(
        f"  w(theta): "
        f"{all_w_h0.shape}"
    )

    print(
        f"  |<w>| tomography: "
        f"{all_abs_h0.shape}"
    )

    # ------------------------------------------------------------------
    # Final structured output
    # ------------------------------------------------------------------

    return MockEnsemble(

        w_theta=all_w_h0,

        abs_statistics=all_abs_h0,

        metadata={

            "n_mocks": int(
                len(all_w_h0)
            ),

            "n_ensemble": int(
                n_ensemble
            ),

            "n_mocks_per_ensemble": int(
                n_mocks_per
            ),

            "selection_function_mode": bool(
                config.use_sel_func
            ),

            "random_seed": int(
                config.random_seed
            ),
        },
    )