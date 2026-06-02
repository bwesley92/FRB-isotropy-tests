
from .config import RuntimeContext


# ==============================================================================
# Logging
# ==============================================================================

def log_runtime_configuration(
    context: RuntimeContext,
) -> None:
    """
    Print complete runtime configuration.
    """

    config = context.config

    print("\n==================================================")
    print("FRB ISOTROPY PIPELINE")
    print("==================================================")

    # ------------------------------------------------------
    # Core paths
    # ------------------------------------------------------

    print(
        f"RUN TAG                  : "
        f"{config.run_tag}"
    )

    print(
        f"PROJECT ROOT             : "
        f"{config.project_root}"
    )

    print(
        f"CATALOG PATH             : "
        f"{config.catalog_path}"
    )

    print(
        f"OUTPUT ROOT              : "
        f"{config.outputs_root}"
    )

    # ------------------------------------------------------
    # Physical model
    # ------------------------------------------------------

    print(
        f"USE GAL MASK             : "
        f"{config.use_gal_mask}"
    )

    print(
        f"USE SEL FUNC             : "
        f"{config.use_sel_func}"
    )

    # ------------------------------------------------------
    # Parallelization
    # ------------------------------------------------------

    print(
        f"N JOBS                   : "
        f"{config.n_jobs}"
    )

    # ------------------------------------------------------
    # Galactic masking
    # ------------------------------------------------------

    print(
        f"GAL CUT                  : "
        f"{config.gal_cut}"
    )

    # ------------------------------------------------------
    # Angular statistics
    # ------------------------------------------------------

    print(
        f"MIN SEP                  : "
        f"{config.min_sep}"
    )

    print(
        f"MAX SEP                  : "
        f"{config.max_sep}"
    )

    print(
        f"BIN SIZE                 : "
        f"{config.bin_size}"
    )

    print(
        f"N BINS                   : "
        f"{config.n_bins}"
    )

    print(
        f"COARSE BINS              : "
        f"{config.coarse_bins}"
    )

    # ------------------------------------------------------
    # Selection functions
    # ------------------------------------------------------

    print(
        f"NSIDE SF                 : "
        f"{config.nside_sf}"
    )

    print(
        f"SMOOTH SIGMA             : "
        f"{config.smooth_sigma}"
    )

    # ------------------------------------------------------
    # H0 perturbations
    # ------------------------------------------------------

    print(
        f"PERTURBATION SCALE       : "
        f"{config.perturbation_scale}"
    )

    # ------------------------------------------------------
    # Random catalogs
    # ------------------------------------------------------

    print(
        f"N RAND FACTOR            : "
        f"{config.n_rand_factor}"
    )

    # ------------------------------------------------------
    # H0 ensemble
    # ------------------------------------------------------

    print(
        f"N ENSEMBLE               : "
        f"{config.n_ensemble}"
    )

    print(
        f"N MOCKS PER ENSEMBLE     : "
        f"{config.n_mocks_per_ensemble}"
    )

    print(
        f"TOTAL H0 MOCKS           : "
        f"{config.n_mocks}"
    )

    # ------------------------------------------------------
    # Covariance regularization
    # ------------------------------------------------------

    print(
        f"SVD EIGENVALUE CUT       : "
        f"{config.svd_eigenvalue_cut}"
    )

    # ------------------------------------------------------
    # Jackknife
    # ------------------------------------------------------

    print(
        f"NSIDE JACKKNIFE          : "
        f"{config.nside_jackknife}"
    )

    print(
        f"MIN JK REGIONS           : "
        f"{config.min_jackknife_regions}"
    )

    # ------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------

    print(
        f"N BOOTSTRAP              : "
        f"{config.n_bootstrap}"
    )

    # ------------------------------------------------------
    # Overlap analysis
    # ------------------------------------------------------

    print(
        f"OVERLAP RADIUS           : "
        f"{config.overlap_radius_deg}"
    )

    print(
        f"OVERLAP NSIDE            : "
        f"{config.overlap_nside}"
    )

    print("==================================================")