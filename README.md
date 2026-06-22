# nanoplasma-analysis

Lightweight Python helpers for analyzing PIConGPU openPMD `.bp5` outputs from helium nanoplasma simulations.

The package wraps the notebook workflow into reusable functions based on `adios2.Stream` and explicit `/data/{step}/...` openPMD paths. It includes routines for particle counts, charge-state evolution, field/density maps, kinetic energy spectra, VMI-style momentum projections, angular distributions, asymmetry metrics, ion radial expansion, and mean kinetic temperature.

## Install

From this repository:

```bash
python -m pip install -e .
```

For a minimal environment:

```bash
python -m pip install -r requirements.txt
```

`adios2` is required to read PIConGPU `.bp5` output files. On HPC systems it may already be provided by the PIConGPU/openPMD software stack.

## Basic use

```python
from nanoplasma_analysis import NanoPlasmaRun

run = NanoPlasmaRun(
    path="/path/to/simOutput",
    laser_peak_at_target=89603,
)

run.plot_particle_number(species="He_e")
run.plot_charge_state_evolution(ion_species="He_i", Zmax=2)
run.plot_field_e_i_maps(file_index=-1)
run.plot_kinetic_energy_spectra(
    species="He_e",
    file_indices="all",
    bins=(0, 20, 100),
    normalize=True,
)
```

## Repository layout

```text
src/nanoplasma_analysis/core.py   # main analysis class and plotting functions
src/nanoplasma_analysis/__init__.py
examples/basic_usage.py
notebooks/ana_004.ipynb           # original working notebook
requirements.txt
pyproject.toml
```

## Notes

The current implementation preserves the data access style used in the original notebook: direct `adios2.Stream` reads and explicit openPMD paths. This makes it easier to compare script output with the notebook analysis.
