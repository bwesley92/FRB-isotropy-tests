import pytest
from pathlib import Path
from frb_isotropy.core.config import AnalysisConfig, validate_analysis_config

@pytest.fixture
def valid_config() -> AnalysisConfig:

    return AnalysisConfig(
        name="test",

        project_root=Path.cwd(),
        outputs_root=Path.cwd() / "outputs",
        catalog_path=__file__,
        
        use_gal_mask=False,
        use_sel_func=False,
        
        n_jobs=1,
        
        gal_cut=20.0,
        
        min_sep=0.0,
        max_sep=180.0,
        bin_size=10.0,
        coarse_bins=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0),
        
        nside_sf=32,
        smooth_sigma=3.0,
        perturbation_scale=1.0,
        
        n_rand_factor=10,
        n_ensemble=10,
        n_mocks_per_ensemble=20,
        
        random_seed=123,
        mock_seed_base=1000,
        random_catalog_seed_base=2000,
        sf_perturbation_seed_base=3000,
        jackknife_seed_base=4000,
        bootstrap_seed_base=5000,
        
        selection_function_floor=1e-12,
        numerical_eigenvalue_floor=1e-15,
        stable_eigenvalue_floor=1e-12,
        numerical_zero_tolerance=1e-30,
        svd_eigenvalue_cut=1e-2,
        
        nside_jackknife=4,
        min_jackknife_regions=20,
        n_bootstrap=100,
        
        overlap_radius_deg=5.0,
        overlap_nside=32,
        
        run_top_survey_maps=False,
    )