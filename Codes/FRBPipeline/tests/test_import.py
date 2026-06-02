import importlib


def test_import_root_package() -> None:
    frb_isotropy = importlib.import_module("frb_isotropy")
    assert hasattr(frb_isotropy, "main")
    assert hasattr(frb_isotropy, "create_runtime_context")


def test_import_subpackages() -> None:
    from frb_isotropy.catalog import load_catalog
    from frb_isotropy.selection import split_by_survey
    from frb_isotropy.statistics import compute_2pacf

    assert callable(load_catalog)
    assert callable(split_by_survey)
    assert callable(compute_2pacf)
