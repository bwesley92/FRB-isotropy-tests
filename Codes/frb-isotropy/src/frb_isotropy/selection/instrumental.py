from dataclasses import dataclass

import healpy as hp
import numpy as np

from ..core.types import FloatArray


@dataclass(
    frozen=True,
    slots=True,
)
class ChimeInstrumentalComponents:
    exposure: FloatArray
    beam: FloatArray | None = None
    daily_sensitivity: FloatArray | None = None
    spectral_response: FloatArray | None = None


def validate_chime_components(
    components: ChimeInstrumentalComponents,
) -> None:
    """
    Validate CHIME instrumental inputs.

    All available maps must:

    - have identical shape
    - contain finite values
    - contain at least one pixel
    """

    exposure = np.asarray(
        components.exposure,
        dtype=float,
    )

    if exposure.ndim != 1:
        raise ValueError(
            "Exposure map must be 1D."
        )

    if len(exposure) == 0:
        raise ValueError(
            "Exposure map is empty."
        )

    if not np.all(np.isfinite(exposure)):
        raise ValueError(
            "Exposure map contains non-finite values."
        )

    npix = len(exposure)

    optional_maps = {
        "beam": components.beam,
        "daily_sensitivity": (
            components.daily_sensitivity
        ),
        "spectral_response": (
            components.spectral_response
        ),
    }

    for name, arr in optional_maps.items():

        if arr is None:
            continue

        arr = np.asarray(
            arr,
            dtype=float,
        )

        if arr.ndim != 1:
            raise ValueError(
                f"{name} map must be 1D."
            )

        if len(arr) != npix:
            raise ValueError(
                f"{name} shape mismatch: "
                f"{len(arr)} != {npix}"
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{name} contains non-finite values."
            )
    

def build_chime_instrumental_sf(
    components: ChimeInstrumentalComponents,
) -> FloatArray:
    """
    Build CHIME instrumental selection function.

    Physics:

        SF =
            exposure
            × beam
            × daily_sensitivity
            × spectral_response

    Missing terms are ignored.
    """

    validate_chime_components(
        components
    )

    sf = np.asarray(
        components.exposure,
        dtype=float,
    ).copy()

    if components.beam is not None:

        sf *= np.asarray(
            components.beam,
            dtype=float,
        )

    if (
        components.daily_sensitivity
        is not None
    ):

        sf *= np.asarray(
            components.daily_sensitivity,
            dtype=float,
        )

    if (
        components.spectral_response
        is not None
    ):

        sf *= np.asarray(
            components.spectral_response,
            dtype=float,
        )

    sf = np.clip(
        sf,
        0.0,
        None,
    )

    norm = sf.sum()

    if norm <= 0:

        raise ValueError(
            "Selection function has zero support."
        )

    sf /= norm

    return np.asarray(
        sf,
        dtype=float,
    )


def summarize_chime_components(
    components: ChimeInstrumentalComponents,
) -> None:
    """
    Print summary of available
    CHIME instrumental inputs.
    """

    print("\nCHIME Instrumental Components")

    print("--------------------------------")

    print(f"Exposure            : YES")

    print(
        f"Beam                : "
        f"{'YES' if components.beam is not None else 'NO'}"
    )

    print(
        f"Daily sensitivity   : "
        f"{'YES' if components.daily_sensitivity is not None else 'NO'}"
    )

    print(
        f"Spectral response   : "
        f"{'YES' if components.spectral_response is not None else 'NO'}"
    )

    print(
        f"Npix                : "
        f"{len(components.exposure)}"
    )
    

def build_chime_exposure_only_sf(
    exposure_map: FloatArray,
) -> FloatArray:
    """
    Build CHIME selection function
    using exposure only.

    Corresponds to:

        SF ∝ Exposure
    """

    components = (
        ChimeInstrumentalComponents(
            exposure=np.asarray(
                exposure_map,
                dtype=float,
            ),
        )
    )

    return build_chime_instrumental_sf(
        components
    )


def analytic_chime_declination_exposure(
    nside: int,
    *,
    latitude_deg: float = 49.3207,
    dec_min_deg: float = -10.0,
    dec_max_deg: float = 90.0,
    pole_cap_deg: float = 89.0,
    normalize_peak: bool = True,
) -> FloatArray:
    """
    Approximate CHIME exposure for a transit telescope.

    The current approximation keeps only the dominant declination-window
    geometry: exposure is proportional to transit duration, modelled as
    sec(dec), with a cap near the north celestial pole.
    """

    _ = latitude_deg

    npix = hp.nside2npix(nside)
    theta, _phi = hp.pix2ang(nside, np.arange(npix))
    dec = 90.0 - np.degrees(theta)

    exposure = np.zeros(npix, dtype=float)
    visible = (dec >= dec_min_deg) & (dec <= dec_max_deg)

    dec_safe = np.clip(dec, -pole_cap_deg, pole_cap_deg)
    exposure[visible] = 1.0 / np.cos(np.radians(dec_safe[visible]))

    if normalize_peak and np.max(exposure) > 0.0:
        exposure /= np.max(exposure)

    return np.asarray(exposure, dtype=float)

