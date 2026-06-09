# FRB Isotropy Analysis

Python package for fast radio burst isotropy analysis.

## Project layout

- `src/frb_isotropy/` - package source code
- `tests/` - automated tests
- `data/` - source catalog and auxiliary files
- `outputs/` - generated analysis outputs

## Install

```bash
python -m pip install -e .
```

## Usage

```python
from frb_isotropy import main, AnalysisConfig, create_runtime_context
```

## Testing

```bash
python -m pytest
```
