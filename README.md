# PIConGPU Nanoplasma Analysis

Clean Python package and reference notebooks for analyzing PIConGPU helium nanoplasma openPMD `.bp5` outputs.

## Current reference versions

This repository merges the latest working notebook/script lineage into a package:

- `src/nanoplasma_analysis/core.py` — raw openPMD diagnostics, based on the newer `nanoplasma_ana.py`.
- `src/nanoplasma_analysis/export.py` — reduced-HDF5 exporter, based on `ana_extract_v4.py`.
- `src/nanoplasma_analysis/h5plots.py` — small reusable helpers for reduced-HDF5 archives.
- `notebooks/00_basic_raw_openpmd_usage.ipynb` — minimal raw `.bp5` usage.
- `notebooks/01_export_reduced_h5.ipynb` — reduced-HDF5 export workflow.
- `notebooks/02_paper_figures_from_h5.ipynb` — paper/reduced-HDF5 plotting reference notebook.

Older versions are kept under `archive/` for traceability. Heavy PIConGPU outputs, generated figures, movies, and `.h5` products are ignored by Git.

## Jupyter/JUWELS usage without installing

On the cluster, do **not** install the package into a shared profile. Open notebooks from inside the repo and use the bootstrap cell that adds `src/` to `sys.path` locally:

```python
from pathlib import Path
import sys

REPO = Path.cwd().resolve().parents[0]  # when running from notebooks/
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanoplasma_analysis import NanoPlasmaRun
```

Make sure `adios2` is available in the active PIConGPU environment. The package can be imported without `adios2`, but reading `.bp5` files requires it.

## Minimal example

```python
from nanoplasma_analysis import NanoPlasmaRun

run = NanoPlasmaRun(
    path="/path/to/simOutput",
    laser_peak_at_target=89603,
)

run.plot_particle_number(species="He_e")
run.plot_charge_state_evolution(Zmax=2)
run.plot_field_e_i_maps(file_index=-1)
```

## Export reduced HDF5

```python
from nanoplasma_analysis import NanoPlasmaRun

run = NanoPlasmaRun(path="/path/to/simOutput", laser_peak_at_target=89603)
run.export_reduced_h5(
    out_h5="Run004_reduced.h5",
    ion_species="He_i",
    e_species="He_e",
    Zmax=2,
    energy_binning="piecewise",
    store_pxy=True,
    store_slices=True,
)
```

## Development checks

```bash
python -m compileall src
python -m pytest
```
