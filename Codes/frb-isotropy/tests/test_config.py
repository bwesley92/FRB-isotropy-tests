import dataclasses
import pytest

from frb_isotropy.core.config import (
    validate_analysis_config,
)


def test_valid_config(valid_config):
    validate_analysis_config(
        valid_config
    )


def test_negative_gal_cut(valid_config):
    bad_config = dataclasses.replace(valid_config, gal_cut=-1.0)
    with pytest.raises(ValueError):
        validate_analysis_config(bad_config)