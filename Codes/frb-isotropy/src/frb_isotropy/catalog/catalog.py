from __future__ import annotations

import healpy as hp
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

from ..core.config import RuntimeContext
from ..core.types import FloatArray, BoolArray


_REQUIRED_CATALOG_COLUMNS = (
    "RA",
    "DEC",
    "Reporting_Group_s",
)


# ==============================================================================
# Coordinate utilities
# ==============================================================================

def icrs_to_galactic(
    ra_deg: FloatArray,
    dec_deg: FloatArray,
) -> tuple[
    FloatArray,
    FloatArray,
]:
    """
    Convert ICRS coordinates to Galactic coordinates.

    Parameters
    ----------
    ra_deg
        Right ascension in degrees.

    dec_deg
        Declination in degrees.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Galactic longitude (l) and latitude (b)
        in degrees.
    """

    coords = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        frame="icrs",
    )

    return (
        np.asarray(coords.galactic.l.degree, dtype=float),
        np.asarray(coords.galactic.b.degree, dtype=float),
    )


def icrs_to_galactic_b(
    ra_deg: FloatArray,
    dec_deg: FloatArray,
) -> FloatArray:
    """
    Convert ICRS coordinates to Galactic latitude.

    Parameters
    ----------
    ra_deg
        Right ascension in degrees.

    dec_deg
        Declination in degrees.

    Returns
    -------
    FloatArray
        Galactic latitude b in degrees.
    """

    _, gal_b = icrs_to_galactic(
        ra_deg,
        dec_deg,
    )

    return gal_b


# ==============================================================================
# Catalog loading
# ==============================================================================

def load_catalog(
    context: RuntimeContext,
) -> pd.DataFrame:
    """
    Load FRB catalog and standardize column names.
    """

    path = context.config.catalog_path

    df = pd.read_csv(path).copy()

    # ------------------------------------------------------------------
    # Standardize column names
    # ------------------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    missing = [
        col
        for col in _REQUIRED_CATALOG_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing catalog columns:\n{missing}"
        )

    # ------------------------------------------------------------------
    # Keep required columns
    # ------------------------------------------------------------------

    df = (
        df[
            ["RA", "DEC", "Reporting_Group_s"]
        ]
        .dropna()
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Coordinate validation
    # ------------------------------------------------------------------

    df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
    df["DEC"] = pd.to_numeric(df["DEC"], errors="coerce")

    df = (
        df.dropna(subset=["RA", "DEC"])
        .reset_index(drop=True)
    )

    valid = (
        (df["RA"] >= 0.0)
        & (df["RA"] < 360.0)
        & (df["DEC"] >= -90.0)
        & (df["DEC"] <= 90.0)
    )

    df = (
        df.loc[valid]
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Catalog is empty after validation."
        )

    print("\n--- Catalog loaded ---")
    print(f"Objects: {len(df)}")

    return df


def apply_mask(
    context: RuntimeContext,
    df: pd.DataFrame,
    gal_cut: float | None = None,
    use_mask: bool | None = None,
) -> pd.DataFrame:

    """
    Apply Galactic latitude mask to observed catalog.

    If the active configuration disables the mask,
    return the original catalog unchanged.
    """

    config = context.config

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if use_mask is None:
        use_mask = config.use_gal_mask

    if gal_cut is None:
        gal_cut = config.gal_cut

    # ------------------------------------------------------------------
    # Disabled mode
    # ------------------------------------------------------------------

    if not use_mask:

        print("\n--- Galactic mask disabled ---")

        return df.reset_index(drop=True).copy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if gal_cut < 0:

        raise ValueError(
            f"gal_cut must be non-negative "
            f"(received {gal_cut})."
        )

    # ------------------------------------------------------------------
    # Galactic coordinates
    # ------------------------------------------------------------------

    gal_b = icrs_to_galactic_b(
        df["RA"].to_numpy(dtype=float),
        df["DEC"].to_numpy(dtype=float),
    )

    # ------------------------------------------------------------------
    # Galactic mask
    # ------------------------------------------------------------------

    mask = np.abs(gal_b) > gal_cut

    df_masked = (
        df.loc[mask]
        .reset_index(drop=True)
        .copy()
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print("\n--- Galactic mask applied ---")
    print(f"gal_cut = ±{gal_cut:.1f} deg")
    print(f"Remaining objects: {len(df_masked)}")
    print(f"Removed objects: {len(df) - len(df_masked)}")

    return df_masked


# ==============================================================================
# Sky geometry
# ==============================================================================

def healpix_galactic_mask(
    context: RuntimeContext,
    nside: int,
    gal_cut: float | None = None,
    use_mask: bool | None = None,
) -> BoolArray:
    """
    Build Galactic mask in HEALPix space.

    Returns
    -------
    BoolArray
        Boolean mask:
            True  -> usable pixel
            False -> masked pixel
    """

    config = context.config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not hp.isnsideok(nside):

        raise ValueError(
            f"Invalid HEALPix nside: {nside}"
        )

    # ------------------------------------------------------------------
    # Runtime defaults
    # ------------------------------------------------------------------

    if use_mask is None:
        use_mask = config.use_gal_mask

    if gal_cut is None:
        gal_cut = config.gal_cut

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if gal_cut < 0:

        raise ValueError(
            "gal_cut must be non-negative."
        )

    npix = hp.nside2npix(nside)

    # ------------------------------------------------------------------
    # No Galactic masking
    # ------------------------------------------------------------------

    if not use_mask:

        return np.ones(npix, dtype=bool)

    # ------------------------------------------------------------------
    # Standard Galactic mask
    # ------------------------------------------------------------------

    theta, phi = hp.pix2ang(
        nside,
        np.arange(npix),
    )

    ra_deg = np.degrees(phi)
    dec_deg = 90.0 - np.degrees(theta)

    gal_b = icrs_to_galactic_b(
        ra_deg,
        dec_deg,
    )

    return np.abs(gal_b) > gal_cut


def rotate_healpix_map_to_galactic(
    hmap: FloatArray,
    coord_in: str = "C",
    coord_out: str = "G",
) -> FloatArray:
    """
    Rotate a HEALPix map between Equatorial (ICRS)
    and Galactic coordinate systems.

    Parameters
    ----------
    hmap : FloatArray
        Input HEALPix map.

    coord_in : str
        Input coordinate system:
            "C" = Equatorial / ICRS
            "G" = Galactic

    coord_out : str
        Output coordinate system ("C" or "G").

    Returns
    -------
    FloatArray
        Rotated HEALPix map.
    """

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    hmap = np.asarray(hmap, dtype=float)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if hmap.ndim != 1:

        raise ValueError(
            "Input HEALPix map must be "
            "1-dimensional."
        )

    valid_coords = {"C", "G"}

    if coord_in not in valid_coords:

        raise ValueError(
            f"Invalid coord_in: {coord_in}"
        )

    if coord_out not in valid_coords:

        raise ValueError(
            f"Invalid coord_out: {coord_out}"
        )

    if not np.all(np.isfinite(hmap)):

        raise ValueError(
            "Input HEALPix map contains "
            "non-finite values."
        )

    nside = hp.get_nside(hmap)

    if len(hmap) != hp.nside2npix(nside):

        raise ValueError(
            "Input map has inconsistent "
            "HEALPix size."
        )

    # ------------------------------------------------------------------
    # Coordinate rotation
    # ------------------------------------------------------------------

    rotator = hp.Rotator(
        coord=[coord_out, coord_in]
    )

    theta, phi = hp.pix2ang(
        nside,
        np.arange(len(hmap)),
    )

    theta_rot, phi_rot = rotator(
        theta,
        phi,
    )

    pix_rot = hp.ang2pix(
        nside,
        theta_rot,
        phi_rot,
    )

    return np.asarray(
        hmap[pix_rot],
        dtype=float,
    )