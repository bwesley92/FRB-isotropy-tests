from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import treecorr

from ..core.config import RuntimeContext

from ..core.types import FloatArray


# ==============================================================================
# Physical estimators
# ==============================================================================

def compute_2pacf(
    context: RuntimeContext,
    df_data: pd.DataFrame,
    df_rand: pd.DataFrame,
) -> tuple[
    FloatArray,
    FloatArray,
]:
    """
    Compute the angular two-point correlation function.

    Uses
    ----
    - Linear angular bins
    - Landy-Szalay estimator
    - TreeCorr pair counting

    Notes
    -----
    Fixed nominal angular bin centers are used
    to guarantee consistency across mock realizations.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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

    if not required_cols.issubset(
        df_rand.columns
    ):

        raise ValueError(
            "df_rand must contain "
            "RA and DEC columns."
        )

    if len(df_data) == 0:

        raise ValueError(
            "df_data is empty."
        )

    if len(df_rand) == 0:

        raise ValueError(
            "df_rand is empty."
        )

    # ------------------------------------------------------------------
    # Coordinate arrays
    # ------------------------------------------------------------------

    ra_data = np.asarray(
        df_data["RA"],
        dtype=float,
    )

    dec_data = np.asarray(
        df_data["DEC"],
        dtype=float,
    )

    ra_rand = np.asarray(
        df_rand["RA"],
        dtype=float,
    )

    dec_rand = np.asarray(
        df_rand["DEC"],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Numerical validation
    # ------------------------------------------------------------------

    arrays_to_check = {

        "ra_data": ra_data,
        "dec_data": dec_data,
        "ra_rand": ra_rand,
        "dec_rand": dec_rand,
    }

    for name, arr in arrays_to_check.items():

        if not np.all(
            np.isfinite(arr)
        ):

            raise ValueError(
                f"{name} contains "
                f"non-finite values."
            )

    # ------------------------------------------------------------------
    # Physical coordinate validation
    # ------------------------------------------------------------------

    if np.any(
        (ra_data < 0.0)
        | (ra_data >= 360.0)
    ):

        raise ValueError(
            "ra_data outside [0, 360)."
        )

    if np.any(
        (ra_rand < 0.0)
        | (ra_rand >= 360.0)
    ):

        raise ValueError(
            "ra_rand outside [0, 360)."
        )

    if np.any(
        (dec_data < -90.0)
        | (dec_data > 90.0)
    ):

        raise ValueError(
            "dec_data outside [-90, 90]."
        )

    if np.any(
        (dec_rand < -90.0)
        | (dec_rand > 90.0)
    ):

        raise ValueError(
            "dec_rand outside [-90, 90]."
        )

    # ------------------------------------------------------------------
    # TreeCorr catalogs
    # ------------------------------------------------------------------

    cat_data = treecorr.Catalog(

        ra=ra_data,

        dec=dec_data,

        ra_units="deg",

        dec_units="deg",
    )

    cat_rand = treecorr.Catalog(

        ra=ra_rand,

        dec=dec_rand,

        ra_units="deg",

        dec_units="deg",
    )

    # ------------------------------------------------------------------
    # Pair-counting configuration
    # ------------------------------------------------------------------

    # TreeCorr threading
    #
    # Intentionally fixed to a single thread.
    #
    # Parallelism is managed externally by joblib
    # in run_ensemble_mocks().
    #
    # Enabling TreeCorr threading here would create
    # nested parallelism and may oversubscribe CPUs.

    corr_config = dict(

        min_sep=(
            config.min_sep
        ),

        max_sep=(
            config.max_sep
        ),

        nbins=(
            config.n_bins
        ),

        sep_units="deg",

        metric="Arc",

        bin_type="Linear",

        num_threads=1,
    )

    dd = treecorr.NNCorrelation(
        **corr_config
    )

    rr = treecorr.NNCorrelation(
        **corr_config
    )

    dr = treecorr.NNCorrelation(
        **corr_config
    )

    # ------------------------------------------------------------------
    # Pair counts
    # ------------------------------------------------------------------

    dd.process(
        cat_data
    )

    rr.process(
        cat_rand
    )

    dr.process(
        cat_data,
        cat_rand,
    )

    # ------------------------------------------------------------------
    # Landy-Szalay estimator
    # ------------------------------------------------------------------

    dd.calculateXi(
        rr=rr,
        dr=dr,
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    theta = np.asarray(
        dd.rnom,
        dtype=float,
    )

    w_theta = np.asarray(
        dd.xi,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Numerical cleanup
    # ------------------------------------------------------------------

    bad = ~np.isfinite(
        w_theta
    )

    if np.any(bad):

        warnings.warn(
            f"{np.sum(bad)} invalid "
            "2pACF bins detected.",
            RuntimeWarning,
        )

        w_theta[bad] = np.nan

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    if theta.shape != w_theta.shape:

        raise RuntimeError(
            "theta and w_theta have "
            "incompatible shapes."
        )

    if len(theta) != config.n_bins:

        raise RuntimeError(
            "Unexpected number of "
            "2pACF bins returned."
        )

    return (
        theta,
        w_theta,
    )


# ==============================================================================
# Absolute anisotropy estimator
# ==============================================================================

def get_absolute_sum(
    context: RuntimeContext,
    theta: FloatArray,
    w: FloatArray,
) -> FloatArray:
    """
    Compute the tomographic absolute-sum anisotropy estimator.

    This follows Eq. (4) of:

        Andrade et al. (2019)
        "Revisiting the statistical isotropy
        of GRB sky distribution"

    For each coarse angular interval Δθ:

        Absolute sum = Σ |w(θ)|

    where the summation is performed over all
    fine 2pACF bins inside that interval.

    Notes
    -----
    - Positive and negative oscillations
      do not cancel.

    - The estimator is sensitive to residual
      angular structure.

    - The estimator intentionally depends
      on the adopted angular binning.
    """

    config = context.config

    coarse_bins = (
        config.coarse_bins_array
    )

    theta = np.asarray(
        theta,
        dtype=float,
    )

    w = np.asarray(
        w,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if theta.shape != w.shape:

        raise ValueError(
            "theta and w must have "
            "identical shapes."
        )

    if len(coarse_bins) < 2:

        raise ValueError(
            "At least two coarse angular "
            "bin edges are required."
        )

    if not np.all(
        np.isfinite(theta)
    ):

        raise ValueError(
            "theta contains non-finite values."
        )

    if not np.any(
        np.isfinite(w)
    ):

        raise ValueError(
            "w contains no finite values."
        )

    if np.any(
        np.diff(coarse_bins) <= 0
    ):

        raise ValueError(
            "coarse_bins must be strictly "
            "increasing."
        )

    if coarse_bins[0] < 0:

        raise ValueError(
            "coarse_bins must be non-negative."
        )

    if coarse_bins[-1] <= coarse_bins[0]:

        raise ValueError(
            "Invalid coarse_bins range."
        )

    vals: list[float] = []

    # ------------------------------------------------------------------
    # Coarse angular tomography
    # ------------------------------------------------------------------

    for i in range(
        len(coarse_bins) - 1
    ):

        theta_min = (
            coarse_bins[i]
        )

        theta_max = (
            coarse_bins[i + 1]
        )

        mask = (

            (theta >= theta_min)

            &

            (theta < theta_max)
        )

        # --------------------------------------------------------------
        # Ignore invalid bins
        # --------------------------------------------------------------

        mask &= np.isfinite(
            w
        )

        if np.any(mask):

            val = float(
                np.sum(
                    np.abs(
                        w[mask]
                    )
                )
            )

        else:

            # No finite fine bins fall inside this coarse
            # interval: the statistic is undefined here, not
            # zero. Downstream consumers (compute_absolute_statistics,
            # pipeline._filter_valid_h0) already mask/drop NaNs.
            val = float("nan")

        vals.append(
            val
        )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    return np.asarray(
        vals,
        dtype=float,
    )