from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.random import Generator

from ..core.config import RuntimeContext

from ..core.types import FloatArray

from ..core.models import (
    SelectionFunctionSet,
)

from ..catalog.catalog import (
    healpix_galactic_mask,
    rotate_healpix_map_to_galactic,
)
            

# ==============================================================================
# Survey modeling and selection functions
# ==============================================================================

def split_by_survey(
    context: RuntimeContext,
    df: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Partition catalog into observational surveys.

    Behavior depends on the active configuration.

    use_sel_func = True
        -> split by survey

    use_sel_func = False
        -> treat entire catalog as one population
    """

    config = context.config

    # ------------------------------------------------------------------
    # No selection function
    # ------------------------------------------------------------------

    if config.use_sel_func is False:

        if verbose:

            print(
                "\n--- Survey partitioning disabled ---"
            )

            print(
                "use_sel_func=False "
                "→ pure isotropic mode"
            )

        return {}

    # ------------------------------------------------------------------
    # Catalog normalization
    # ------------------------------------------------------------------

    df = df.copy()

    df["Reporting_Group_s"] = (

        df["Reporting_Group_s"]

        .fillna("UNKNOWN")

        .astype(str)
    )

    surveys: dict[
        str,
        list[
            tuple[float, float]
        ],
    ] = {}

    # ------------------------------------------------------------------
    # Survey extraction
    # ------------------------------------------------------------------

    for row in df.itertuples(index=False):

        groups = list({

            g.strip()

            for g in str(
                row.Reporting_Group_s
            ).split(",")

            if g.strip()
        })

        for group in groups:

            if group == "":

                group = "UNKNOWN"

            surveys.setdefault(
                group,
                [],
            ).append(
                (
                    row.RA,
                    row.DEC,
                )
            )

    # ------------------------------------------------------------------
    # Survey DataFrames
    # ------------------------------------------------------------------

    survey_dict = {

        name: pd.DataFrame(
            values,
            columns=[
                "RA",
                "DEC",
            ],
        )

        for name, values
        in surveys.items()
    }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    if verbose:

        counts = {

            name: len(subdf)

            for name, subdf
            in survey_dict.items()
        }

        sorted_counts = sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        print(
            "\n--- Survey partitioning ---"
        )

        print(
            f"Unique surveys: "
            f"{len(survey_dict)}"
        )

        print(
            "\nSurvey populations:"
        )

        for name, count in sorted_counts:

            frac = (
                100.0
                * count
                / len(df)
            )

            print(
                f"{name:>15s} | "
                f"N={count:4d} | "
                f"{frac:6.2f}%"
            )

    return survey_dict


def build_selection_function(
    context: RuntimeContext,
    subdf: pd.DataFrame,
    nside: int | None = None,
    smooth_sigma: float | None = None,
    gal_cut: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """
    Build probabilistic sky selection function for one survey.

    Returns
    -------
    sf : FloatArray
        Normalized sky probability distribution.

    sf_cov_diag : FloatArray
        Approximate diagonal covariance.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    if len(subdf) == 0:

        raise ValueError(
            "Cannot build selection function "
            "from empty survey."
        )

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if nside is None:

        nside = (
            config.nside_sf
        )

    if smooth_sigma is None:

        smooth_sigma = (
            config.smooth_sigma
        )

    if gal_cut is None:

        gal_cut = (
            config.gal_cut
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    if smooth_sigma < 0:

        raise ValueError(
            "smooth_sigma must be "
            "non-negative."
        )

    if gal_cut < 0:

        raise ValueError(
            "gal_cut must be "
            "non-negative."
        )

    npix = hp.nside2npix(
        nside
    )

    # ------------------------------------------------------------------
    # Pixelized survey map
    # ------------------------------------------------------------------

    theta = np.radians(
        90.0
        - subdf["DEC"].to_numpy(float)
    )

    phi = np.radians(
        subdf["RA"].to_numpy(float)
    )

    pix = hp.ang2pix(
        nside,
        theta,
        phi,
    )

    counts = np.bincount(
        pix,
        minlength=npix,
    ).astype(float)

    # ------------------------------------------------------------------
    # Small numerical floor
    # ------------------------------------------------------------------

    counts += (
        config.selection_function_floor
    )

    total_counts = counts.sum()

    if total_counts <= 0:

        raise ValueError(
            "Selection function has "
            "zero counts."
        )

    # ------------------------------------------------------------------
    # Raw selection function
    # ------------------------------------------------------------------

    sf = counts / total_counts

    # ------------------------------------------------------------------
    # Poisson uncertainty
    # ------------------------------------------------------------------

    sf_err_counts = np.sqrt(
        counts
    )

    sf_cov_diag = (
        sf_err_counts
        / total_counts
    ) ** 2

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    if smooth_sigma > 0:

        sf = hp.smoothing(
            sf,
            sigma=np.radians(
                smooth_sigma
            ),
        )

        sf = np.clip(
            sf,
            0.0,
            None,
        )

    # ------------------------------------------------------------------
    # Galactic mask
    # ------------------------------------------------------------------

    gal_mask = healpix_galactic_mask(
        context,
        nside,
        gal_cut=gal_cut,
        use_mask=config.use_gal_mask,
    )

    sf[~gal_mask] = 0.0

    # ------------------------------------------------------------------
    # Final normalization
    # ------------------------------------------------------------------

    sf_sum = sf.sum()

    if sf_sum <= 0:

        raise ValueError(
            "Selection function vanished "
            "after masking."
        )

    sf /= sf_sum

    return (
        np.asarray(
            sf,
            dtype=float,
        ),
        np.asarray(
            sf_cov_diag,
            dtype=float,
        ),
    )


# ==============================================================================
# Multi-survey selection functions
# ==============================================================================

def build_survey_selection_functions(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    nside: int | None = None,
    smooth_sigma: float | None = None,
    gal_cut: float | None = None,
    use_adaptive_smoothing: bool = False,
) -> SelectionFunctionSet | None:
    
    """
    Build survey-based sky probability maps.

    Behavior depends on the active configuration.

    use_sel_func = True
        -> one selection function per survey

    use_sel_func = False
        -> no selection-function modeling
    """

    config = context.config

    # ----------------------------------------------------------
    # Pure isotropic mode
    # ----------------------------------------------------------

    if config.use_sel_func is False:

        print(
            "\n--- Selection-function modeling disabled ---"
        )

        print(
            "use_sel_func=False "
            "→ no survey selection functions "
            "will be built"
        )

        return None

    # ----------------------------------------------------------
    # Runtime defaults
    # ----------------------------------------------------------

    if nside is None:

        nside = (
            config.nside_sf
        )

    if smooth_sigma is None:

        smooth_sigma = (
            config.smooth_sigma
        )

    if gal_cut is None:

        gal_cut = (
            config.gal_cut
        )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    if smooth_sigma < 0:

        raise ValueError(
            "smooth_sigma must be "
            "non-negative."
        )

    if gal_cut < 0:

        raise ValueError(
            "gal_cut must be "
            "non-negative."
        )

    # ----------------------------------------------------------
    # Survey partitioning
    # ----------------------------------------------------------

    surveys = split_by_survey(
        context,
        df_data,
    )

    total_survey_memberships = sum(

        len(subdf)

        for subdf
        in surveys.values()
    )

    if total_survey_memberships <= 0:

        raise ValueError(
            "No survey memberships available "
            "for selection-function modeling."
        )

    sf_dict: dict[
        str,
        FloatArray,
    ] = {}

    sf_cov_diag_dict: dict[
        str,
        FloatArray,
    ] = {}

    survey_weights: dict[
        str,
        float,
    ] = {}

    print(
        "\n--- Building selection functions ---"
    )

    print(
        f"Unique surveys: "
        f"{len(surveys)}"
    )

    print(
        f"Adaptive smoothing: "
        f"{use_adaptive_smoothing}"
    )

    # ----------------------------------------------------------
    # Survey loop
    # ----------------------------------------------------------

    for name, subdf in surveys.items():

        # ------------------------------------------------------
        # Optional adaptive smoothing
        # ------------------------------------------------------

        effective_sigma = smooth_sigma

        if use_adaptive_smoothing:

            if len(subdf) < 10:

                effective_sigma = max(
                    smooth_sigma,
                    8.0,
                )

            elif len(subdf) < 30:

                effective_sigma = max(
                    smooth_sigma,
                    5.0,
                )

        # ------------------------------------------------------
        # Build survey selection function
        # ------------------------------------------------------

        sf, sf_cov_diag = (

            build_selection_function(
                context,
                subdf,
                nside=nside,
                smooth_sigma=effective_sigma,
                gal_cut=gal_cut,
            )
        )

        sf_dict[name] = sf

        sf_cov_diag_dict[name] = (
            sf_cov_diag
        )

        survey_weights[name] = (

            len(subdf)
            / total_survey_memberships
        )

        print(
            f"{name:>15s} | "
            f"N={len(subdf):4d} | "
            f"sigma={effective_sigma:4.1f} deg | "
            f"weight={survey_weights[name]:.4f}"
        )

    print(
        "\nSelection functions "
        "successfully built."
    )

    # ----------------------------------------------------------
    # Build immutable validated container
    # ----------------------------------------------------------

    sf_set = SelectionFunctionSet(
        sf_dict=sf_dict,
        sf_cov_diag_dict=sf_cov_diag_dict,
        survey_weights=survey_weights,
        nside=nside,
    )

    sf_set.validate()

    return sf_set


# Backwards-compatible aliases for older notebooks/scripts.
build_selection_function_improved = build_selection_function
build_survey_selection_functions_improved = build_survey_selection_functions


# ==============================================================================
# Selection-function perturbations
# ==============================================================================

def generate_sf_variant(
    context: RuntimeContext,
    sf_set: SelectionFunctionSet,
    perturbation_scale: float | None = None,
    rng: Generator | None = None,
) -> dict[str, FloatArray]:
    
    """
    Generate perturbed realizations of survey selection functions.

    Notes
    -----
    This function is only physically relevant when
    the active configuration enables selection-function modeling.

    When selection functions are disabled,
    no perturbation is generated.

    IMPORTANT
    ---------
    A Generator instance must be passed explicitly
    to guarantee statistically independent realizations
    across mocks and parallel workers.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if perturbation_scale is None:

        perturbation_scale = (
            config.perturbation_scale
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if perturbation_scale < 0:

        raise ValueError(
            "perturbation_scale must be "
            "non-negative."
        )

    # ------------------------------------------------------------------
    # Pure isotropic mode
    # ------------------------------------------------------------------

    if config.use_sel_func is False:

        return {}

    # ------------------------------------------------------------------
    # RNG validation
    # ------------------------------------------------------------------

    if rng is None:

        raise ValueError(
            "A numpy.random.Generator instance "
            "must be provided explicitly."
        )

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------

    sf_dict_perturbed: dict[
        str,
        FloatArray,
    ] = {}

    # ------------------------------------------------------------------
    # Survey perturbation loop
    # ------------------------------------------------------------------

    for survey_name in sf_set.surveys:

        sf = np.asarray(
            sf_set.sf_dict[
                survey_name
            ],
            dtype=float,
        )

        sf_cov_diag = np.asarray(
            sf_set.sf_cov_diag_dict[
                survey_name
            ],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Shape validation
        # --------------------------------------------------------------

        if sf.shape != sf_cov_diag.shape:

            raise ValueError(
                f"Shape mismatch for survey "
                f"{survey_name}: "
                f"{sf.shape} != "
                f"{sf_cov_diag.shape}"
            )

        # --------------------------------------------------------------
        # Covariance validation
        # --------------------------------------------------------------

        if np.any(sf_cov_diag < 0):

            raise ValueError(
                f"Negative covariance entries "
                f"detected for survey: "
                f"{survey_name}"
            )

        # --------------------------------------------------------------
        # Gaussian perturbation
        # --------------------------------------------------------------

        sigma = np.sqrt(
            sf_cov_diag
        )

        noise = rng.normal(
            loc=0.0,
            scale=(
                perturbation_scale
                * sigma
            ),
            size=sf.shape,
        )

        sf_pert = np.clip(
            sf + noise,
            0.0,
            None,
        )

        # --------------------------------------------------------------
        # Numerical safeguard
        # --------------------------------------------------------------

        sf_sum = sf_pert.sum()

        if sf_sum <= 0:

            sf_pert = sf.copy()

        else:

            sf_pert /= sf_sum

        sf_dict_perturbed[
            survey_name
        ] = np.asarray(
            sf_pert,
            dtype=float,
        )

    return sf_dict_perturbed


# ==============================================================================
# Selection-function validation
# ==============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class SelectionFunctionValidator:
    """
    Numerical and visual validation of survey
    selection functions.

    This validator is only physically meaningful when:

        use_sel_func = True
    """

    context: RuntimeContext

    sf_set: SelectionFunctionSet

    # ==========================================================
    # Numerical validation
    # ==========================================================

    def check_coverage(
        self,
    ) -> pd.DataFrame:

        config = self.context.config

        # ------------------------------------------------------
        # Disabled case
        # ------------------------------------------------------

        if config.use_sel_func is False:

            print(
                "\n--- Selection-function validation skipped ---"
            )

            print(
                "use_sel_func=False "
                "→ no instrumental selection model."
            )

            return pd.DataFrame()

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        self.sf_set.validate()

        if not hp.isnsideok(
            self.sf_set.nside
        ):

            raise ValueError(
                f"Invalid HEALPix nside: "
                f"{self.sf_set.nside}"
            )

        print("\n==================================================")
        print("SELECTION FUNCTION VALIDATION")
        print("==================================================")

        rows: list[
            dict[
                str,
                float | str,
            ]
        ] = []

        # ------------------------------------------------------
        # Survey ordering
        # ------------------------------------------------------

        ordered = sorted(

            self.sf_set.surveys,

            key=lambda name: (
                self.sf_set.survey_weights[name]
            ),

            reverse=True,
        )


        # ------------------------------------------------------
        # Survey loop
        # ------------------------------------------------------

        for survey_name in ordered:

            sf = np.asarray(
                self.sf_set.sf_dict[
                    survey_name
                ],
                dtype=float,
            )

            # --------------------------------------------------
            # Numerical validation
            # --------------------------------------------------

            if not np.all(
                np.isfinite(sf)
            ):

                raise ValueError(
                    f"Selection function "
                    f"{survey_name} contains "
                    f"non-finite values."
                )

            # --------------------------------------------------
            # Peak position
            # --------------------------------------------------

            max_pix = int(
                np.argmax(sf)
            )

            theta_max, phi_max = hp.pix2ang(
                self.sf_set.nside,
                max_pix,
            )

            dec_max = (
                90.0
                - np.degrees(theta_max)
            )

            ra_max = np.degrees(
                phi_max
            )

            # --------------------------------------------------
            # Coverage
            # --------------------------------------------------

            threshold = (
                np.percentile(sf, 99)
                * 1e-2
            )

            active = sf > threshold

            coverage = float(
                np.mean(active)
                * 100.0
            )

            # --------------------------------------------------
            # Entropy
            # --------------------------------------------------

            positive = sf > 0.0

            entropy = float(
                -np.sum(
                    sf[positive]
                    * np.log2(
                        sf[positive]
                    )
                )
            )

            # --------------------------------------------------
            # Effective sky area
            # --------------------------------------------------

            npix_active = int(
                np.sum(active)
            )

            sky_fraction = float(
                npix_active
                / len(sf)
            )

            # --------------------------------------------------
            # Store row
            # --------------------------------------------------

            rows.append(
                {

                    "run_tag":
                        config.run_tag,

                    "survey":
                        survey_name,

                    "weight":
                        self.sf_set.survey_weights[
                            survey_name
                        ],

                    "peak_ra":
                        ra_max,

                    "peak_dec":
                        dec_max,

                    "active_coverage_pct":
                        coverage,

                    "sky_fraction":
                        sky_fraction,

                    "entropy_bits":
                        entropy,
                }
            )

            # --------------------------------------------------
            # Diagnostics
            # --------------------------------------------------

            print(
                f"\n{survey_name}:"
            )

            print(
                f"  Weight: "
                f"{self.sf_set.survey_weights[survey_name]:.2%}"
            )

            print(
                f"  Peak density: "
                f"RA={ra_max:.1f} deg, "
                f"DEC={dec_max:.1f} deg"
            )

            print(
                f"  Active coverage: "
                f"{coverage:.1f}%"
            )

            print(
                f"  Entropy: "
                f"{entropy:.2f} bits"
            )

            # --------------------------------------------------
            # Physical sanity checks
            # --------------------------------------------------

            survey_upper = (
                survey_name.upper()
            )

            if (
                "CHIME" in survey_upper
                and dec_max < 20.0
            ):

                print(
                    "  WARNING: "
                    "CHIME peak unexpectedly far south."
                )

            if (
                "PARKES" in survey_upper
                and dec_max > 5.0
            ):

                print(
                    "  WARNING: "
                    "PARKES peak unexpectedly far north."
                )

        validation_df = pd.DataFrame(
            rows
        )

        return validation_df

    # ==========================================================
    # Visual validation
    # ==========================================================

    def plot_sf(
        self,
        save_prefix: str | Path = "SF_validation",
        max_surveys: int = 12,
        cmap: str = "viridis",
        log_scale: bool = False,
    ) -> None:

        config = self.context.config

        floor = (
            config.selection_function_floor
        )

        save_prefix = Path(
            save_prefix
        )

        # ------------------------------------------------------
        # Disabled case
        # ------------------------------------------------------

        if config.use_sel_func is False:

            print(
                "\n--- SF plots skipped ---"
            )

            print(
                "use_sel_func=False "
                "→ no selection functions to visualize."
            )

            return

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        self.sf_set.validate()

        if max_surveys < 1:

            raise ValueError(
                "max_surveys must be positive."
            )

        # ------------------------------------------------------
        # Survey ordering
        # ------------------------------------------------------

        ordered = sorted(

            self.sf_set.surveys,

            key=lambda name: (
                self.sf_set.survey_weights[name]
            ),

            reverse=True,
        )[:max_surveys]

        # ------------------------------------------------------
        # Safety check
        # ------------------------------------------------------

        if len(ordered) == 0:

            print(
                "\nNo selection functions "
                "available for plotting."
            )

            return

        ncols = min(
            4,
            len(ordered),
        )

        nrows = int(
            np.ceil(
                len(ordered) / ncols
            )
        )

        scale_label = (
            "log10"
            if log_scale
            else "linear"
        )

        # ======================================================
        # EQUATORIAL MAPS
        # ======================================================

        plt.figure(
            figsize=(
                5 * ncols,
                3.8 * nrows,
            )
        )

        for i, name in enumerate(
            ordered,
            start=1,
        ):

            sf_plot = np.asarray(
                self.sf_set.sf_dict[name],
                dtype=float,
            )

            if log_scale:

                sf_plot = np.log10(
                    np.clip(
                        sf_plot,
                        floor,
                        None,
                    )
                )

            hp.mollview(
                sf_plot,
                title=(
                    f"{name} | "
                    # f"N={self.sf_set.survey_counts[name]} | "
                    f"w={self.sf_set.survey_weights[name]:.1%}"
                ),
                sub=(
                    nrows,
                    ncols,
                    i,
                ),
                cbar=True,
                cmap=cmap,
            )

        equatorial_path = (
            save_prefix
            .with_name(
                save_prefix.name
                + "_equatorial.png"
            )
        )

        plt.savefig(
            equatorial_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()

        plt.close()

        # ======================================================
        # GALACTIC MAPS
        # ======================================================

        plt.figure(
            figsize=(
                5 * ncols,
                3.8 * nrows,
            )
        )

        for i, name in enumerate(
            ordered,
            start=1,
        ):

            sf_gal = (
                rotate_healpix_map_to_galactic(
                    self.sf_set.sf_dict[name],
                    coord_in="C",
                    coord_out="G",
                )
            )

            if log_scale:

                sf_gal = np.log10(
                    np.clip(
                        sf_gal,
                        floor,
                        None,
                    )
                )

            hp.mollview(
                sf_gal,
                title=(
                    f"{name} | "
                    # f"N={self.sf_set.survey_counts[name]} | "
                    f"w={self.sf_set.survey_weights[name]:.1%}"
                ),
                sub=(
                    nrows,
                    ncols,
                    i,
                ),
                cbar=True,
                cmap=cmap,
            )

        galactic_path = (
            save_prefix
            .with_name(
                save_prefix.name
                + "_galactic.png"
            )
        )

        plt.savefig(
            galactic_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.show()

        plt.close()

        print(
            "\nSelection-function maps saved:"
        )

        print(
            f"  {equatorial_path}"
        )

        print(
            f"  {galactic_path}"
        )