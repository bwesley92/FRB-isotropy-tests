# FRB Isotropy Test — Pipeline Overview

This folder contains an analysis pipeline to test the angular isotropy of
Fast Radio Burst (FRB) sky positions. The canonical implementation lives in
`frb_isotropy.py` and an interactive workflow is provided in
`frb_isotropy_analysis.ipynb`.

This short README documents: quick start, core pipeline steps, outputs,
dependencies and useful notes.

## Quick start

1. Activate your virtual environment (recommended):

```bash
source /home/you/projetos/venv/bin/activate
```

2. Run a minimal import test to confirm the module loads and where outputs
   will be written:

```bash
cd FRB-isotropy-tests/Codes/Main
/path/to/venv/bin/python - <<'PY'
import frb_isotropy as iso
print('module_file:', iso.__file__)
print('OUTPUTS_ROOT:', iso.OUTPUTS_ROOT)
print('RUN_OUTPUT_DIR:', iso.RUN_OUTPUT_DIR)
PY
```

3. Open `frb_isotropy_analysis.ipynb` in Jupyter / VS Code and run cells to
   execute the full pipeline interactively.

## Core pipeline steps

The pipeline is implemented as a set of reusable functions in
`frb_isotropy.py`. High-level stages are:

- Load & standardize catalog (`load_catalog()`)
- Galactic masking (`apply_mask()`, `healpix_galactic_mask()`)
- Partition catalog by reporting group / survey (`split_by_survey()`)
- Build empirical HEALPix selection functions (`build_selection_function_improved()`)
- Generate isotropic mock catalogs and ensembles (`generate_mixture_catalog_improved()`, `run_ensemble_mocks()`)
- Measure angular two-point correlation (`compute_2pacf()`)
- Compute coarse absolute-anisotropy amplitudes (`get_absolute_sum()`)
- Estimate observational uncertainties (jackknife: `run_jackknife_errors()`, bootstrap helpers)
- Estimate covariance, regularize via SVD and compute statistics (`compute_statistics()`)
- Export diagnostic figures, tables and a short report under `outputs/<run_tag>/`.

For detailed usage and interactive exploration, prefer the notebook which
imports the module and demonstrates the workflow step-by-step.

## Inputs

- Catalog: `FRB_catalogs/SkyPosition.csv` (required columns: `RA`, `DEC`, `Reporting_Group_s`)

The `load_catalog()` helper normalizes column names and keeps only the
relevant fields.

## Outputs

When you run the analysis it writes results into a run-specific folder in
`outputs/`. The run tag encodes the configuration (masking, selection-function
usage, sensitivity flag and bin size). Example structure:

```
outputs/mask_sf_nosens_bin10/
  figures/
  tables/
  report/summary.md
```

Saved artifacts include plots (`figures/*.png`), diagnostic tables
(`tables/*.csv`) and a small markdown report (`report/summary.md`).

## Dependencies

Install required packages in your environment (venv recommended). A minimal
set:

```bash
/path/to/venv/bin/pip install healpy astropy treecorr tqdm seaborn scikit-learn scipy pandas matplotlib joblib tqdm_joblib
```

## Notes

- The code determines `OUTPUTS_ROOT` from the module location
  (`Path(__file__).resolve().parent / 'outputs'`). Moving the module and the
  notebook together inside the repo (for example renaming folders) is safe so
  long as their relative placement is preserved.
- If you change configuration flags (for example `BIN_SIZE` or `USE_SEL_FUNC`),
  the `RUN_TAG` will change and outputs will appear in a different subfolder.

---

If you want I can:

- add a `requirements.txt` or `environment.yml` for reproducibility;
- expose a `main()` CLI wrapper to run the full pipeline from the command line;
- or generate a short developer guide mapping functions to notebook cells.

Generated from the current `frb_isotropy.py` and `frb_isotropy_analysis.ipynb`.
