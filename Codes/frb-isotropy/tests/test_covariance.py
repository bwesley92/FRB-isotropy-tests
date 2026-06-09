import numpy as np
import pytest


from frb_isotropy.statistics.covariance import (
    build_shrinkage_covariance,
)


def test_covariance_shape():
    """
    Covariance matrix must have shape
    (n_bins, n_bins).
    """

    rng = np.random.default_rng(
        12345
    )

    w = rng.normal(
        size=(100, 10)
    )

    cov = build_shrinkage_covariance(
        w
    )

    assert cov.shape == (
        10,
        10,
    )


def test_covariance_symmetry():
    """
    Covariance matrix must be symmetric.
    """

    rng = np.random.default_rng(
        12345
    )

    w = rng.normal(
        size=(100, 10)
    )

    cov = build_shrinkage_covariance(
        w
    )

    assert np.allclose(
        cov,
        cov.T,
        atol=1e-12,
        rtol=1e-10,
    )


def test_covariance_positive_eigenvalues():
    """
    Ledoit-Wolf covariance should be
    positive definite.
    """

    rng = np.random.default_rng(
        12345
    )

    w = rng.normal(
        size=(100, 10)
    )

    cov = build_shrinkage_covariance(
        w
    )

    eigvals = np.linalg.eigvalsh(
        cov
    )

    assert np.all(
        eigvals > 0.0
    )


def test_invalid_dimension():

    x = np.ones(
        10,
        dtype=float,
    )

    with pytest.raises(
        ValueError
    ):

        build_shrinkage_covariance(
            x
        )


def test_too_few_mocks():

    x = np.ones(
        (1, 10),
        dtype=float,
    )

    with pytest.raises(
        ValueError
    ):

        build_shrinkage_covariance(
            x
        )