import numpy as np

from sklearn.covariance import LedoitWolf

from ..core.types import FloatArray
from ..core.models import CovarianceDiagnostics


def build_shrinkage_covariance(
    all_w_h0: FloatArray,
) -> FloatArray:
    """
    Estimate the covariance matrix using
    Ledoit-Wolf shrinkage.

    Parameters
    ----------
    all_w_h0 : FloatArray
        Mock ensemble matrix with shape

            (n_mocks, n_bins)

    Returns
    -------
    FloatArray
        Shrinkage covariance matrix.
    """

    all_w_h0 = np.asarray(
        all_w_h0,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if all_w_h0.ndim != 2:

        raise ValueError(
            "all_w_h0 must be 2-dimensional."
        )

    if all_w_h0.shape[0] < 2:

        raise ValueError(
            "At least two mock realizations "
            "are required."
        )

    if not np.all(
        np.isfinite(all_w_h0)
    ):

        raise ValueError(
            "all_w_h0 contains non-finite values."
        )

    # ------------------------------------------------------------------
    # Ledoit-Wolf shrinkage covariance
    # ------------------------------------------------------------------

    lw = LedoitWolf()

    lw.fit(
        all_w_h0
    )

    cov = np.asarray(
        lw.covariance_,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if cov.ndim != 2:

        raise RuntimeError(
            "Covariance matrix must be "
            "2-dimensional."
        )

    if cov.shape[0] != cov.shape[1]:

        raise RuntimeError(
            "Covariance matrix must be square."
        )

    if not np.all(
        np.isfinite(cov)
    ):

        raise RuntimeError(
            "Covariance matrix contains "
            "non-finite values."
        )

    return cov


# ==============================================================================
# Eigenvalue diagnostics
# ==============================================================================

def _eigenvalue_summary(
    cov: FloatArray,
) -> dict[
    str,
    float | int | FloatArray
]:
    """
    Compute eigenvalue-based covariance diagnostics.
    """

    eigvals = np.linalg.eigvalsh(
        cov
    )

    eigvals = np.asarray(
        eigvals,
        dtype=float,
    )

    eigvals_pos = eigvals[
        eigvals > 1e-15
    ]

    if len(eigvals_pos) == 0:

        raise ValueError(
            "No positive covariance "
            "eigenvalues found."
        )

    eigvals_safe = eigvals_pos[
        eigvals_pos > 1e-12
    ]

    eig_sum = float(
        np.sum(
            eigvals_pos
        )
    )

    eig_sq_sum = float(
        np.sum(
            eigvals_pos**2
        )
    )

    if eig_sq_sum <= 0:

        n_eff = np.nan

    else:

        n_eff = float(
            eig_sum**2
            / eig_sq_sum
        )

    rank = int(
        np.linalg.matrix_rank(
            cov
        )
    )

    if len(eigvals_safe) > 0:

        smallest_safe = max(
            float(
                np.min(
                    eigvals_safe
                )
            ),
            np.finfo(float).tiny,
        )

        largest_safe = float(
            np.max(
                eigvals_safe
            )
        )

        condition = float(
            largest_safe
            / smallest_safe
        )

    else:

        smallest_safe = np.nan

        condition = np.nan

    return {

        "eigvals":
            eigvals,

        "eigvals_pos":
            eigvals_pos,

        "eigvals_safe":
            eigvals_safe,

        "rank":
            rank,

        "n_eff":
            n_eff,

        "condition":
            condition,

        "smallest_safe":
            smallest_safe,
    }


def compute_covariance_diagnostics(
    cov: FloatArray,
    n_mocks: int,
    n_bins: int | None = None,
    svd_modes_kept: int | None = None,
    svd_eigenvalue_cut: float | None = None,
    svd_condition: float | None = None,
) -> CovarianceDiagnostics:
    """
    Compute covariance quality diagnostics.

    The optional SVD fields keep this function compatible with callers that
    attach regularization metadata after computing the SVD statistic.
    """

    cov = np.asarray(
        cov,
        dtype=float,
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    if cov.ndim != 2:

        raise ValueError(
            "cov must be 2-dimensional."
        )

    if cov.shape[0] != cov.shape[1]:

        raise ValueError(
            "cov must be square."
        )

    if not np.all(
        np.isfinite(cov)
    ):

        raise ValueError(
            "cov contains non-finite values."
        )

    inferred_n_bins = int(
        cov.shape[0]
    )

    if n_bins is not None and int(n_bins) != inferred_n_bins:

        raise ValueError(
            "n_bins does not match covariance shape."
        )

    if n_mocks <= inferred_n_bins + 2:

        raise ValueError(
            "Hartlap correction invalid: "
            "N_mocks must satisfy "
            "N_mocks > N_bins + 2."
        )

    # ----------------------------------------------------------
    # Eigenvalue diagnostics
    # ----------------------------------------------------------

    summary = (
        _eigenvalue_summary(
            cov
        )
    )

    eigvals_safe = summary[
        "eigvals_safe"
    ]

    if len(eigvals_safe) == 0:

        raise ValueError(
            "No stable covariance "
            "eigenvalues found."
        )

    covariance_rank = int(
        summary["rank"]
    )

    n_eff = float(
        summary["n_eff"]
    )

    covariance_condition = float(
        summary["condition"]
    )

    hartlap_factor = float(
        (
            n_mocks
            - inferred_n_bins
            - 2
        )
        / (
            n_mocks
            - 1
        )
    )

    return CovarianceDiagnostics(
        hartlap_factor=hartlap_factor,
        covariance_rank=covariance_rank,
        covariance_condition=covariance_condition,
        n_eff=n_eff,
        svd_modes_kept=(
            int(svd_modes_kept)
            if svd_modes_kept is not None
            else covariance_rank
        ),
        svd_eigenvalue_cut=(
            float(svd_eigenvalue_cut)
            if svd_eigenvalue_cut is not None
            else np.nan
        ),
        svd_condition=(
            float(svd_condition)
            if svd_condition is not None
            else covariance_condition
        ),
    )


def compute_covariance_comparison_metrics(
    w_array: FloatArray,
    label: str,
) -> dict[
    str,
    float | int | str
]:
    """
    Compute covariance-comparison diagnostics
    for a mock ensemble.
    """

    cov = build_shrinkage_covariance(
        w_array
    )

    if not np.allclose(
        cov,
        cov.T,
        atol=1e-12,
        rtol=1e-10,
    ):

        raise ValueError(
            f"{label}: covariance matrix "
            "is not symmetric."
        )

    # --------------------------------------------------
    # Eigenvalue diagnostics
    # --------------------------------------------------

    summary = (
        _eigenvalue_summary(
            cov
        )
    )

    eigvals_pos = summary[
        "eigvals_pos"
    ]

    rank = int(
        summary["rank"]
    )

    n_eff = float(
        summary["n_eff"]
    )

    condition = float(
        summary["condition"]
    )

    min_safe_eig = float(
        summary["smallest_safe"]
    )

    # --------------------------------------------------
    # Mock correlations
    # --------------------------------------------------

    corr = np.corrcoef(
        w_array
    )

    corr = np.asarray(
        corr,
        dtype=float,
    )

    corr[
        ~np.isfinite(corr)
    ] = np.nan

    mask = ~np.eye(
        len(corr),
        dtype=bool,
    )

    offdiag = corr[
        mask
    ]

    offdiag = offdiag[
        np.isfinite(
            offdiag
        )
    ]

    if len(offdiag) == 0:

        raise ValueError(
            f"{label}: unable to compute "
            "off-diagonal mock correlations."
        )

    return {

        "version":
            label,

        "n_mocks":
            int(
                w_array.shape[0]
            ),

        "n_bins":
            int(
                w_array.shape[1]
            ),

        "covariance_rank":
            rank,

        "effective_modes":
            float(n_eff),

        "condition_number":
            float(condition),

        "max_eigenvalue":
            float(
                np.max(
                    eigvals_pos
                )
            ),

        "min_positive_eigenvalue":
            float(
                np.min(
                    eigvals_pos
                )
            ),

        "min_safe_eigenvalue":
            float(
                min_safe_eig
            ),

        "mean_mock_corr":
            float(
                np.mean(
                    offdiag
                )
            ),

        "median_mock_corr":
            float(
                np.median(
                    offdiag
                )
            ),

        "std_mock_corr":
            float(
                np.std(
                    offdiag
                )
            ),

        "max_mock_corr":
            float(
                np.max(
                    offdiag
                )
            ),

        "min_mock_corr":
            float(
                np.min(
                    offdiag
                )
            ),
    }