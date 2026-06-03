import dataclasses
import pytest

from frb_isotropy.core.config import (
    validate_analysis_config,
)

from tests.conftest import (
    AnalysisConfig,
    make_valid_config,
)


def test_valid_config():
    config = (
        make_valid_config()
    )

    validate_analysis_config(
        config
    )


def test_negative_gal_cut():
    config = make_valid_config()
    bad_config = dataclasses.replace(config, gal_cut=-1.0)
    with pytest.raises(ValueError):
        validate_analysis_config(bad_config)