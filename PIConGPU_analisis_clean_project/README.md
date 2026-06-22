# PIConGPU Nanoplasma Analysis

Clean Python package and reference notebooks for analyzing PIConGPU helium nanoplasma openPMD `.bp5` outputs.

This repo stores **code and notebooks only**. Do not commit generated data products such as `.bp5`, `.h5`, `.png`, movies, or full `simOutput/` folders.

## Recommended JUWELS/Jupyter workflow

Open the notebooks from inside the repository, for example:

```bash
cd /e/scratch/jureap18/medina2/PIConGPU_analisis/notebooks
```

The notebooks use a local `sys.path` bootstrap to import:

```python
from nanoplasma_analysis import NanoPlasmaRun
```

No `pip install -e .` is required on the cluster if the notebook is opened from inside the repo tree.

## Main notebooks

- `notebooks/00_basic_raw_openpmd_usage.ipynb`  
  Quick checks directly from raw PIConGPU `simOutput/openPMD/*.bp5` files.

- `notebooks/01_export_reduced_h5.ipynb`  
  Converts raw `.bp5` output into a compact reduced `.h5` archive with timeseries, radial data, spectra, VMI-like histograms, and slices.

- `notebooks/02_paper_figures_from_h5.ipynb`  
  Rebuilds the paper-style figures from the reduced `.h5` file. This is the replacement for the old `H5_plots_004.ipynb` workflow.

- `notebooks/ana_004.ipynb`  
  Current working raw-analysis notebook kept for continuity.

## Package layout

- `src/nanoplasma_analysis/core.py` — raw openPMD diagnostics.
- `src/nanoplasma_analysis/export.py` — reduced-HDF5 exporter.
- `src/nanoplasma_analysis/h5plots.py` — helper functions for reduced-HDF5 plotting.

Older versions are kept under `archive/` for traceability.

## Typical sequence

1. Run PIConGPU simulation.
2. Use `01_export_reduced_h5.ipynb` to create the reduced H5 file on scratch.
3. Use `02_paper_figures_from_h5.ipynb` to generate the paper figures.

Example raw simulation path:

```python
SIM_OUTPUT = Path("/e/scratch/jureap18/medina2/004_V1_LowDensity/simOutput")
```

Example reduced H5 path:

```python
OUT_H5 = SIM_OUTPUT.parent / f"{SIM_OUTPUT.parent.name}_reduced.h5"
```

## Optional editable install

On a local machine you may still install the package in editable mode:

```bash
pip install -e .
```

On the cluster this is optional; the notebooks already import the package from `src/` directly.

## Development checks

```bash
python -m compileall src tests
PYTHONPATH=src python - <<'PY'
import nanoplasma_analysis
print(nanoplasma_analysis.NanoPlasmaRun)
PY
```

## Git hygiene

The `.gitignore` excludes heavy/generated files:

- `*.bp5`
- `*.h5`
- `simOutput/`
- `openPMD/`
- figures/movies/cache/checkpoints

Before committing:

```bash
git status
```

Only commit code, notebooks, and documentation.
