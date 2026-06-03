import numpy as np
import pytest

from dataclasses import replace

from frb_isotropy.core.config import (
    create_runtime_context,
)

from frb_isotropy.statistics.estimators import (
    get_absolute_sum,
)


def test_get_absolute_sum_known_solution(
    valid_config,
):

    config = replace(

        valid_config,

        coarse_bins=(
            0.0,
            40.0,
            80.0,
        ),
    )

    context = (
        create_runtime_context(
            config
        )
    )

    theta = np.array(
        [
            10.0,
            30.0,
            50.0,
            70.0,
        ]
    )

    w = np.array(
        [
            1.0,
            -2.0,
            3.0,
            -4.0,
        ]
    )

    result = get_absolute_sum(
        context,
        theta,
        w,
    )

    expected = np.array(
        [
            3.0,  # |1| + |-2|
            7.0,  # |3| + |-4|
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_absolute_sum_no_cancellation(
    valid_config,
):

    config = replace(

        valid_config,

        coarse_bins=(
            0.0,
            40.0,
        ),
    )

    context = (
        create_runtime_context(
            config
        )
    )

    theta = np.array(
        [
            10.0,
            20.0,
        ]
    )

    w = np.array(
        [
            1.0,
            -1.0,
        ]
    )

    result = get_absolute_sum(
        context,
        theta,
        w,
    )

    assert np.isclose(
        result[0],
        2.0,
    )


def test_absolute_sum_shape_validation(
    valid_config,
):

    context = (
        create_runtime_context(
            valid_config
        )
    )

    theta = np.array(
        [
            10.0,
            20.0,
            30.0,
        ]
    )

    w = np.array(
        [
            1.0,
            2.0,
        ]
    )

    with pytest.raises(
        ValueError
    ):

        get_absolute_sum(
            context,
            theta,
            w,
        )