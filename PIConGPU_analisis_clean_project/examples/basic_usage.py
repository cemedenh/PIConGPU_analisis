"""Minimal usage example for nanoplasma-analysis.

Edit SIM_OUTPUT and LASER_PEAK_STEP before running.
"""

from nanoplasma_analysis import NanoPlasmaRun

SIM_OUTPUT = "/path/to/simOutput"
LASER_PEAK_STEP = 89603

run = NanoPlasmaRun(path=SIM_OUTPUT, laser_peak_at_target=LASER_PEAK_STEP)

run.plot_particle_number(species="He_e")
run.plot_charge_state_evolution(ion_species="He_i", Zmax=2)
run.plot_field_e_i_maps(file_index=-1)
run.plot_kinetic_energy_spectra(species="He_e", file_indices="all", bins=(0, 20, 100))
