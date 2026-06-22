"""
nanoplasma_ana.py

Extension module for PIConGPU nanoplasma analysis.

- Re-exports NanoPlasmaRun from nanoplasma_analysis_final.py
- Adds: export_reduced_h5(...) to create a compact HDF5 "science archive"
  containing time series, spectra, angular and radial histograms, and optional
  central slices, so you can delete large raw openPMD .bp5 outputs later.

Usage (in notebook)
-------------------
from nanoplasma_ana import NanoPlasmaRun

run = NanoPlasmaRun(path=".../simOutput", laser_peak_at_target=89603)

# export compact archive:
run.export_reduced_h5(
    out_h5="run004_reduced.h5",
    ion_species="He_i",
    e_species="He_e",
    Zmax=2,
    r_max_nm=2000.0,
    n_r=600,
    mu_bins=360,
    E_max_eV=2000.0,
    n_E=800,
    store_slices=True,
    slice_every=200,   # store slices every N outputs
)

Then verify plots from the HDF5 only, and delete raw openPMD folders.
"""

from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.constants as sc

import h5py
import adios2

# Import the existing implementation (already used in your notebooks)
from nanoplasma_analysis_final import NanoPlasmaRun as _NanoPlasmaRunBase, extract_step_from_filename


def _safe_decode(x):
    try:
        return x.decode("utf-8")
    except Exception:
        return str(x)


def _log_edges(E_min: float, E_max: float, n: int) -> np.ndarray:
    # Log-spaced edges including 0 is tricky; we start from a small epsilon.
    eps = max(E_min, 1e-6)
    return np.geomspace(eps, E_max, n + 1)


def _write_text_group(g: h5py.Group, name: str, text: str):
    dt = h5py.string_dtype(encoding="utf-8")
    g.create_dataset(name, data=np.array([text], dtype=dt))


class NanoPlasmaRun(_NanoPlasmaRunBase):
    """
    Subclass that adds export functionality on top of nanoplasma_analysis_final.NanoPlasmaRun.
    """

    def export_reduced_h5(
        self,
        out_h5: str,
        *,
        ion_species: str = "He_i",
        e_species: str = "He_e",
        Zmax: int = 2,
        # Binning
        r_max_nm: float = 2000.0,
        n_r: int = 600,
        mu_bins: int = 360,
        # Energy histogram
        E_min_eV: float = 1e-3,
        E_max_eV: float = 2000.0,
        n_E: int = 800,
        log_energy: bool = False,
        # 2D momentum histograms (VMI-like): store H(px,py) per output (weighted counts)
        store_pxy: bool = True,
        p_bins: int = 256,
        # If None, p_max is derived from E_max_eV and species mass: p_max = sqrt(2 m E_max)
        p_max_e_SI: float | None = None,
        p_max_i_SI: float | None = None,
        # Slices (2D planes) for paper-ready maps
        slice_y_nm: float = 300.0,
        electron_density_field: str = "He_e_all_density",
        ion_charge_density_field: str = "He_i_all_density",
        # Slices
        store_slices: bool = True,
        slice_every: int = 200,               # store slices every N outputs
        slice_fields: Sequence[str] = ("E",), # store |E| by default
        slice_densities: Sequence[str] = ("He_e", "He_i", "rho_net"),
        # Center
        center_mode: str = "first_com",       # "first_com" recommended
        # Metadata extras
        extra_meta: dict | None = None,
        # Optional: dump run param files (text)
        param_file_paths: Sequence[str] | None = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> str:
        """
        Create a compact HDF5 archive with derived diagnostics for all outputs.

        Returns the output path.
        """
        self._load_meta()
        files = list(self.files)
        if len(files) == 0:
            raise RuntimeError("No openPMD .bp5 files found under the provided path.")

        out_h5 = os.path.abspath(out_h5)
        if os.path.exists(out_h5):
            if overwrite:
                os.remove(out_h5)
            else:
                raise FileExistsError(out_h5)

        # Axes / bins
        r_edges_m = np.linspace(0.0, r_max_nm * 1e-9, n_r + 1)
        r_mid_m = 0.5 * (r_edges_m[1:] + r_edges_m[:-1])

        mu_edges = np.linspace(-1.0, 1.0, mu_bins + 1)
        mu_mid = 0.5 * (mu_edges[1:] + mu_edges[:-1])

        if log_energy:
            E_edges = _log_edges(E_min_eV, E_max_eV, n_E)
        else:
            E_edges = np.linspace(E_min_eV, E_max_eV, n_E + 1)
        E_mid = 0.5 * (E_edges[1:] + E_edges[:-1])

        # Masses
        m_e = self._mass_kg(e_species)
        m_i = self._mass_kg(ion_species)
        # Momentum axes for optional 2D histograms (px, py) in SI units
        if store_pxy:
            if p_max_e_SI is None:
                p_max_e = float(np.sqrt(2.0 * m_e * (E_max_eV * sc.e)))
            else:
                p_max_e = float(p_max_e_SI)
            if p_max_i_SI is None:
                p_max_i = float(np.sqrt(2.0 * m_i * (E_max_eV * sc.e)))
            else:
                p_max_i = float(p_max_i_SI)
            p_edges_e = np.linspace(-p_max_e, p_max_e, int(p_bins) + 1)
            p_mid_e = 0.5 * (p_edges_e[1:] + p_edges_e[:-1])
            p_edges_i = np.linspace(-p_max_i, p_max_i, int(p_bins) + 1)
            p_mid_i = 0.5 * (p_edges_i[1:] + p_edges_i[:-1])
        else:
            p_edges_e = p_mid_e = p_edges_i = p_mid_i = None

        # Determine reference center r0 (meters)
        r0 = None
        if center_mode == "given":
            raise ValueError("center_mode='given' not supported in exporter yet; use 'first_com' or 'each_com'.")
        if center_mode in ("first_com", "each_com"):
            # first_com: use COM of the first non-empty ion frame
            if center_mode == "first_com":
                for fn in files:
                    step = extract_step_from_filename(fn)
                    x, y, z, w = self._read_positions_and_weight(fn, step, ion_species)
                    if x is None or len(x) == 0:
                        continue
                    W = np.sum(w)
                    if W == 0:
                        continue
                    r0 = (np.sum(x * w) / W, np.sum(y * w) / W, np.sum(z * w) / W)
                    break
                if r0 is None:
                    # fallback to origin
                    r0 = (0.0, 0.0, 0.0)

        # Pre-allocate arrays
        Nt = len(files)
        t_fs = np.zeros(Nt, dtype=np.float64)

        # Time series
        ts = {}
        def _ts(name, shape, dtype=np.float64):
            ts[name] = np.full(shape, np.nan, dtype=dtype)

        _ts("fields_U_E_J", (Nt,))
        _ts("fields_U_B_J", (Nt,))

        for sp in (e_species, ion_species):
            _ts(f"{sp}_N_macro", (Nt,), dtype=np.int64)
            _ts(f"{sp}_N_real", (Nt,))  # sum weights
            _ts(f"{sp}_Ekin_total_eV", (Nt,))
            _ts(f"{sp}_Ekin_mean_eV", (Nt,))
            _ts(f"{sp}_P_total_SI", (Nt, 3))

        _ts(f"{ion_species}_Z_mean", (Nt,))
        _ts(f"{ion_species}_charge_frac", (Nt, Zmax + 1))
        _ts(f"{e_species}_T_mean_eV", (Nt,))
        # Tail temperature can be derived later; keep placeholder for convenience
        _ts(f"{e_species}_T_tail_eV", (Nt,))

        # Spectra and angular
        dNdE_e = np.zeros((Nt, n_E), dtype=np.float64)
        dNdE_i = np.zeros((Nt, n_E), dtype=np.float64)
        dNdmu_e = np.zeros((Nt, mu_bins), dtype=np.float64)
        dEdmu_e = np.zeros((Nt, mu_bins), dtype=np.float64)
        # Optional 2D momentum histograms H(px,py) per output (weighted counts).
        # These let you re-make VMI-like plots without storing per-particle momenta.
        if store_pxy:
            H_pxy_e = np.zeros((Nt, int(p_bins), int(p_bins)), dtype=np.float32)
            H_pxy_i = np.zeros((Nt, int(p_bins), int(p_bins)), dtype=np.float32)
        else:
            H_pxy_e = H_pxy_i = None

        # Radial profiles
        n_i_rt = np.zeros((Nt, n_r), dtype=np.float64)
        n_e_rt = np.zeros((Nt, n_r), dtype=np.float64)
        rho_net_rt = np.zeros((Nt, n_r), dtype=np.float64)  # C per shell volume (approx) OR charge per shell; we store charge-per-shell
        # Charge state vs radius (counts weighted by macroparticle weighting)
        charge_rZ = np.zeros((Nt, n_r, Zmax + 1), dtype=np.float64)

        # Helper: shell volumes for converting counts to density if desired
        # volume of spherical shell between edges: 4/3 pi (r2^3 - r1^3)
        shell_vol = (4.0 / 3.0) * np.pi * (r_edges_m[1:] ** 3 - r_edges_m[:-1] ** 3)
        shell_vol[shell_vol == 0] = np.nan

        # Which outputs get slices stored
        slice_indices = set()
        if store_slices and slice_every > 0:
            slice_indices = set(range(0, Nt, slice_every))
            # always include first and last
            slice_indices.add(0)
            slice_indices.add(Nt - 1)


        # Stored 2D slices indexed by step name (written at end into /slices)
        slices_by_step: dict[str, dict[str, np.ndarray]] = {}

        if verbose:
            print(f"[export_reduced_h5] outputs: {Nt}")
            print(f"[export_reduced_h5] r bins: {n_r} up to {r_max_nm} nm")
            print(f"[export_reduced_h5] mu bins: {mu_bins}, energy bins: {n_E} up to {E_max_eV} eV")
            print(f"[export_reduced_h5] slices: {'on' if store_slices else 'off'} ({len(slice_indices)} stored)")

        # Main loop
        for i, fn in enumerate(files):
            step = extract_step_from_filename(fn)
            t_fs[i] = self.time_fs_from_step(step)
            print(f"[export] {i+1}/{len(files)}  step={step}")
            # Read particle data
            # electrons
            px, py, pz, w_e = self._read_momentum_and_weight(fn, step, e_species)
            x_e, y_e, z_e, wpos_e = self._read_positions_and_weight(fn, step, e_species)

            # ions
            pxi, pyi, pzi, w_i = self._read_momentum_and_weight(fn, step, ion_species)
            x_i, y_i, z_i, wpos_i = self._read_positions_and_weight(fn, step, ion_species)

            # Optional 2D momentum histograms (px,py) per timestep
            if store_pxy:
                if px is not None and w_e is not None:
                    H2, _, _ = np.histogram2d(px, py, bins=[p_edges_e, p_edges_e], weights=w_e)
                    H_pxy_e[i, :, :] = H2.astype(np.float32)
                if pxi is not None and w_i is not None:
                    H2i, _, _ = np.histogram2d(pxi, pyi, bins=[p_edges_i, p_edges_i], weights=w_i)
                    H_pxy_i[i, :, :] = H2i.astype(np.float32)


            # Bound electrons for charge-state
            be = None
            w_be = None
            if x_i is not None:
                with adios2.Stream(fn, "r") as f:
                    for _ in f.steps():
                        try:
                            be = f.read(f"/data/{step}/particles/{ion_species}/boundElectrons")
                            w_be = f.read(f"/data/{step}/particles/{ion_species}/weighting")
                        except Exception:
                            be = None
                            w_be = None

            # Fields energies (global)
            with adios2.Stream(fn, "r") as f:
                for _ in f.steps():
                    # Some runs may not have these; keep NaN if missing
                    try:
                        # Approximate field energy from E meshes/fields (if available).
                        dV = float(self.Dx_SI * self.Dy_SI * self.Dz_SI)
                        # Prefer /fields, fallback to /meshes
                        Ex = Ey = Ez = None
                        uE = 1.0
                        try:
                            if f"/data/{step}/fields/E/x" in f.available_variables():
                                Ex = f.read(f"/data/{step}/fields/E/x")
                                Ey = f.read(f"/data/{step}/fields/E/y")
                                Ez = f.read(f"/data/{step}/fields/E/z")
                                try:
                                    uE = float(f.read_attribute(f"/data/{step}/fields/E/x/unitSI"))
                                except Exception:
                                    uE = 1.0
                            elif f"/data/{step}/meshes/E/x" in f.available_variables():
                                Ex = f.read(f"/data/{step}/meshes/E/x")
                                Ey = f.read(f"/data/{step}/meshes/E/y")
                                Ez = f.read(f"/data/{step}/meshes/E/z")
                                try:
                                    uE = float(f.read_attribute(f"/data/{step}/meshes/E/x/unitSI"))
                                except Exception:
                                    uE = 1.0
                        except Exception:
                            Ex = Ey = Ez = None
                        if Ex is not None:
                            Ex = Ex * uE; Ey = Ey * uE; Ez = Ez * uE
                            U_E = 0.5 * sc.epsilon_0 * float(np.sum(Ex*Ex + Ey*Ey + Ez*Ez)) * dV
                            ts["fields_U_E_J"][i] = U_E
                    except Exception:
                        pass
                    try:
                        # Magnetic energy from B meshes/fields (if available).
                        dV = float(self.Dx_SI * self.Dy_SI * self.Dz_SI)
                        Bx = By = Bz = None
                        uB = 1.0
                        try:
                            if f"/data/{step}/fields/B/x" in f.available_variables():
                                Bx = f.read(f"/data/{step}/fields/B/x")
                                By = f.read(f"/data/{step}/fields/B/y")
                                Bz = f.read(f"/data/{step}/fields/B/z")
                                try:
                                    uB = float(f.read_attribute(f"/data/{step}/fields/B/x/unitSI"))
                                except Exception:
                                    uB = 1.0
                            elif f"/data/{step}/meshes/B/x" in f.available_variables():
                                Bx = f.read(f"/data/{step}/meshes/B/x")
                                By = f.read(f"/data/{step}/meshes/B/y")
                                Bz = f.read(f"/data/{step}/meshes/B/z")
                                try:
                                    uB = float(f.read_attribute(f"/data/{step}/meshes/B/x/unitSI"))
                                except Exception:
                                    uB = 1.0
                        except Exception:
                            Bx = By = Bz = None
                        if Bx is not None:
                            Bx = Bx * uB; By = By * uB; Bz = Bz * uB
                            U_B = 0.5 / sc.mu_0 * float(np.sum(Bx*Bx + By*By + Bz*Bz)) * dV
                            ts["fields_U_B_J"][i] = U_B
                    except Exception:
                        pass

            # Time series: N, momentum, energies
            def fill_species(sp, px, py, pz, w, m_kg):
                if px is None:
                    ts[f"{sp}_N_macro"][i] = 0
                    ts[f"{sp}_N_real"][i] = 0.0
                    ts[f"{sp}_Ekin_total_eV"][i] = 0.0
                    ts[f"{sp}_Ekin_mean_eV"][i] = np.nan
                    ts[f"{sp}_P_total_SI"][i, :] = 0.0
                    return

                ts[f"{sp}_N_macro"][i] = int(len(px))
                W = float(np.sum(w))
                ts[f"{sp}_N_real"][i] = W

                p2 = px*px + py*py + pz*pz
                E_eV = p2 / (2.0 * m_kg) / sc.e
                E_tot = float(np.sum(E_eV * w))
                ts[f"{sp}_Ekin_total_eV"][i] = E_tot
                ts[f"{sp}_Ekin_mean_eV"][i] = E_tot / W if W > 0 else np.nan

                P = np.array([np.sum(px*w), np.sum(py*w), np.sum(pz*w)], dtype=np.float64)
                ts[f"{sp}_P_total_SI"][i, :] = P

                return E_eV

            E_eV_e = fill_species(e_species, px, py, pz, w_e, m_e)
            E_eV_i = fill_species(ion_species, pxi, pyi, pzi, w_i, m_i)

            # Temperature proxy from mean kinetic energy: (2/3)<E> in eV
            if ts[f"{e_species}_Ekin_mean_eV"][i] == ts[f"{e_species}_Ekin_mean_eV"][i]:
                ts[f"{e_species}_T_mean_eV"][i] = (2.0 / 3.0) * ts[f"{e_species}_Ekin_mean_eV"][i]

            # Spectra
            if E_eV_e is not None:
                h, _ = np.histogram(E_eV_e, bins=E_edges, weights=w_e)
                dNdE_e[i, :] = h
            if E_eV_i is not None:
                h, _ = np.histogram(E_eV_i, bins=E_edges, weights=w_i)
                dNdE_i[i, :] = h

            # Angular distributions for electrons wrt x-axis by default (same as existing code default)
            if px is not None:
                p_abs = np.sqrt(px*px + py*py + pz*pz)
                good = p_abs > 0
                mu = np.zeros_like(p_abs)
                mu[good] = px[good] / p_abs[good]
                hN, _ = np.histogram(mu, bins=mu_edges, weights=w_e)
                dNdmu_e[i, :] = hN
                # energy-weighted
                if E_eV_e is not None:
                    hE, _ = np.histogram(mu, bins=mu_edges, weights=w_e * E_eV_e)
                    dEdmu_e[i, :] = hE

            # Radial profiles (use r0 fixed unless each_com is requested)
            if center_mode == "each_com" and x_i is not None and len(x_i) > 0:
                W = np.sum(wpos_i)
                if W > 0:
                    r0_use = (np.sum(x_i*wpos_i)/W, np.sum(y_i*wpos_i)/W, np.sum(z_i*wpos_i)/W)
                else:
                    r0_use = r0
            else:
                r0_use = r0

            def radial_hist(x, y, z, w, charge_per_particle_C: float | None = None):
                if x is None:
                    return np.zeros(n_r, dtype=np.float64)
                rx = x - r0_use[0]
                ry = y - r0_use[1]
                rz = z - r0_use[2]
                r = np.sqrt(rx*rx + ry*ry + rz*rz)
                if charge_per_particle_C is None:
                    h, _ = np.histogram(r, bins=r_edges_m, weights=w)
                else:
                    h, _ = np.histogram(r, bins=r_edges_m, weights=w * charge_per_particle_C)
                return h

            n_i_rt[i, :] = radial_hist(x_i, y_i, z_i, wpos_i)
            n_e_rt[i, :] = radial_hist(x_e, y_e, z_e, wpos_e)

            # Net charge per shell: ions minus electrons (C), using mean ion charge from bound electrons if present.
            # If charge states are missing, we approximate ions as +1e (NOT ideal, but avoids NaNs).
            q_e_shell = radial_hist(x_e, y_e, z_e, wpos_e, charge_per_particle_C=-sc.e)

            if be is not None and w_be is not None:
                Z = np.clip(Zmax - be.astype(np.int32), 0, Zmax)
                # charge per macroparticle varies with Z: +Z e
                # We'll compute by binning each Z separately for the rZ cube, and sum for total ion charge.
                ion_charge_shell = np.zeros(n_r, dtype=np.float64)
                rx = x_i - r0_use[0]
                ry = y_i - r0_use[1]
                rz = z_i - r0_use[2]
                r = np.sqrt(rx*rx + ry*ry + rz*rz)
                for zval in range(0, Zmax + 1):
                    mask = (Z == zval)
                    if not np.any(mask):
                        continue
                    h, _ = np.histogram(r[mask], bins=r_edges_m, weights=w_be[mask])
                    charge_rZ[i, :, zval] = h
                    ion_charge_shell += h * (zval * sc.e)

                # charge fractions vs time (global)
                binsZ = np.arange(-0.5, Zmax + 1.5, 1.0)
                hZ, _ = np.histogram(Z, bins=binsZ, weights=w_be)
                denom = np.sum(hZ)
                if denom > 0:
                    ts[f"{ion_species}_charge_frac"][i, :] = hZ[: Zmax + 1] / denom
                    ts[f"{ion_species}_Z_mean"][i] = np.sum(np.arange(0, Zmax + 1) * (hZ[: Zmax + 1] / denom))
            else:
                # fallback
                ion_charge_shell = radial_hist(x_i, y_i, z_i, wpos_i, charge_per_particle_C=+sc.e)

            rho_net_rt[i, :] = ion_charge_shell + q_e_shell

            # Optional 2D slices for paper-ready maps (written into /slices)
            if store_slices and (i in slice_indices):
                step_name = f"step_{int(step):08d}"
                y_idx = int((slice_y_nm * 1e-9) / self.Dy_SI) if self.Dy_SI > 0 else 0
                y_idx = max(0, min(self.Ny - 1, y_idx))
                z_idx = int(self.Nz // 2)

                def _read_field(path_fields: str, path_meshes: str, *, start=None, count=None):
                    """Try /fields first, then /meshes. Returns (array, unitSI)."""
                    try:
                        if path_fields in f.available_variables():
                            arr = f.read(path_fields, start=start, count=count)
                            try:
                                unit = f.read_attribute(path_fields + "/unitSI")
                            except Exception:
                                unit = 1.0
                            return arr, float(unit)
                    except Exception:
                        pass
                    try:
                        if path_meshes in f.available_variables():
                            arr = f.read(path_meshes, start=start, count=count)
                            # meshes often already in SI; unit attr may not exist
                            try:
                                unit = f.read_attribute(path_meshes + "/unitSI")
                            except Exception:
                                unit = 1.0
                            return arr, float(unit)
                    except Exception:
                        pass
                    return None, 1.0

                with adios2.Stream(fn, "r") as f:
                    for _ in f.steps():
                        # Ex slice in the x-y plane at z = mid
                        Ex_xy, uE = _read_field(
                            f"/data/{step}/fields/E/x",
                            f"/data/{step}/meshes/E/x",
                            start=[z_idx, 0, 0],
                            count=[1, self.Ny, self.Nx],
                        )
                        if Ex_xy is not None:
                            Ex_xy = (Ex_xy[0, :, :] * uE).astype(np.float32)

                        # Electron charge density slice in the x-z plane at fixed y
                        rho_e_xz, urhoe = _read_field(
                            f"/data/{step}/fields/{electron_density_field}",
                            f"/data/{step}/meshes/{electron_density_field}",
                            start=[0, y_idx, 0],
                            count=[self.Nz, 1, self.Nx],
                        )
                        if rho_e_xz is not None:
                            rho_e_xz = (rho_e_xz[:, 0, :] * urhoe).astype(np.float32)

                        # Ion charge density slice in the x-z plane at fixed y
                        rho_i_xz, urhoi = _read_field(
                            f"/data/{step}/fields/{ion_charge_density_field}",
                            f"/data/{step}/meshes/{ion_charge_density_field}",
                            start=[0, y_idx, 0],
                            count=[self.Nz, 1, self.Nx],
                        )
                        if rho_i_xz is not None:
                            rho_i_xz = (rho_i_xz[:, 0, :] * urhoi).astype(np.float32)

                slices_by_step[step_name] = {
                    "Ex_xy_SI": Ex_xy if Ex_xy is not None else np.zeros((self.Ny, self.Nx), dtype=np.float32),
                    "rho_e_xz_SI": rho_e_xz if rho_e_xz is not None else np.zeros((self.Nz, self.Nx), dtype=np.float32),
                    "rho_i_xz_SI": rho_i_xz if rho_i_xz is not None else np.zeros((self.Nz, self.Nx), dtype=np.float32),
                    "y_idx": np.array([y_idx], dtype=np.int32),
                    "z_idx": np.array([z_idx], dtype=np.int32),
                    "slice_y_nm": np.array([slice_y_nm], dtype=np.float32),
                }

        # Sort by time (files usually are, but be safe)
        order = np.argsort(t_fs)
        t_fs = t_fs[order]
        for k in list(ts.keys()):
            ts[k] = ts[k][order]
        dNdE_e = dNdE_e[order]
        dNdE_i = dNdE_i[order]
        dNdmu_e = dNdmu_e[order]
        dEdmu_e = dEdmu_e[order]
        if store_pxy:
            H_pxy_e = H_pxy_e[order]
            H_pxy_i = H_pxy_i[order]
        n_i_rt = n_i_rt[order]
        n_e_rt = n_e_rt[order]
        rho_net_rt = rho_net_rt[order]
        charge_rZ = charge_rZ[order]

        # Write HDF5
        with h5py.File(out_h5, "w") as h5:
            # Meta
            gmeta = h5.create_group("meta")
            gmeta.attrs["source"] = "nanoplasma_ana.export_reduced_h5"
            gmeta.attrs["openpmd_path"] = os.path.abspath(self.path)
            gmeta.attrs["laser_peak_at_target_step"] = int(self.laser_peak_at_target)
            gmeta.attrs["center_mode"] = center_mode
            gmeta.attrs["center_r0_m"] = np.array(r0, dtype=np.float64)

            if extra_meta:
                gmeta.attrs["extra_meta_json"] = json.dumps(extra_meta)

            # Store param files as text
            if param_file_paths:
                gpar = gmeta.create_group("params")
                for pth in param_file_paths:
                    try:
                        with open(pth, "r", encoding="utf-8", errors="ignore") as f:
                            _write_text_group(gpar, os.path.basename(pth), f.read())
                    except Exception as e:
                        gpar.attrs[f"missing_{os.path.basename(pth)}"] = _safe_decode(str(e))

            # Axes
            gaxes = h5.create_group("axes")
            gaxes.create_dataset("time_fs", data=t_fs)
            gaxes.create_dataset("r_edges_m", data=r_edges_m)
            gaxes.create_dataset("r_mid_m", data=r_mid_m)
            gaxes.create_dataset("mu_edges", data=mu_edges)
            gaxes.create_dataset("mu_mid", data=mu_mid)
            gaxes.create_dataset("E_edges_eV", data=E_edges)
            gaxes.create_dataset("E_mid_eV", data=E_mid)

            # Momentum axes for optional 2D histograms
            if store_pxy:
                gaxes.create_dataset(f"{e_species}_p_edges_SI", data=p_edges_e)
                gaxes.create_dataset(f"{e_species}_p_mid_SI", data=p_mid_e)
                gaxes.create_dataset(f"{ion_species}_p_edges_SI", data=p_edges_i)
                gaxes.create_dataset(f"{ion_species}_p_mid_SI", data=p_mid_i)

            # Timeseries
            gts = h5.create_group("timeseries")
            for name, arr in ts.items():
                gts.create_dataset(name, data=arr)

            # Spectra
            gs = h5.create_group("spectra")
            gs.create_dataset(f"{e_species}_dNdE", data=dNdE_e, compression="gzip", compression_opts=4, chunks=(1, n_E))
            gs.create_dataset(f"{ion_species}_dNdE", data=dNdE_i, compression="gzip", compression_opts=4, chunks=(1, n_E))
            gs.create_dataset(f"{e_species}_dNdmu", data=dNdmu_e, compression="gzip", compression_opts=4, chunks=(1, mu_bins))
            gs.create_dataset(f"{e_species}_dEdmu", data=dEdmu_e, compression="gzip", compression_opts=4, chunks=(1, mu_bins))

            # 2D momentum histograms (px,py)
            if store_pxy:
                gs.create_dataset(f"{e_species}_H_pxy", data=H_pxy_e, compression="gzip", compression_opts=4, chunks=(1, int(p_bins), int(p_bins)))
                gs.create_dataset(f"{ion_species}_H_pxy", data=H_pxy_i, compression="gzip", compression_opts=4, chunks=(1, int(p_bins), int(p_bins)))

            # Radial
            gr = h5.create_group("radial")
            gr.create_dataset(f"{ion_species}_n_r", data=n_i_rt, compression="gzip", compression_opts=4, chunks=(1, n_r))
            gr.create_dataset(f"{e_species}_n_r", data=n_e_rt, compression="gzip", compression_opts=4, chunks=(1, n_r))
            gr.create_dataset("rho_net_shell_C", data=rho_net_rt, compression="gzip", compression_opts=4, chunks=(1, n_r))
            gr.create_dataset("shell_volume_m3", data=shell_vol)
            gr.create_dataset(f"{ion_species}_charge_rZ_counts", data=charge_rZ, compression="gzip", compression_opts=4, chunks=(1, n_r, Zmax+1))
            gr.create_dataset("Z_values", data=np.arange(0, Zmax + 1, dtype=np.int32))

            # Slices
            if store_slices:
                gslices = h5.create_group("slices")
                # write in a stable order
                for step_name in sorted(slices_by_step.keys()):
                    gstep = gslices.create_group(step_name)
                    dct = slices_by_step[step_name]
                    for dname, darr in dct.items():
                        # tiny scalar datasets (idx, etc.) store without compression
                        if isinstance(darr, np.ndarray) and darr.ndim <= 1 and darr.size <= 8:
                            gstep.create_dataset(dname, data=darr)
                        else:
                            gstep.create_dataset(dname, data=darr, compression="gzip", compression_opts=4)

        if verbose:
            size_mb = os.path.getsize(out_h5) / (1024**2)
            print(f"[export_reduced_h5] wrote: {out_h5} ({size_mb:.1f} MB)")

        return out_h5
            
    def _get_cell_size_from_openpmd(self, f, step):
        """
        Return (dx,dy,dz) in meters from any available mesh's gridSpacing.
        Works with PIConGPU openPMD layouts.
        """
        # Try a few common meshes that are usually present
        candidates = [
            f"/data/{step}/meshes/E/x",
            f"/data/{step}/meshes/B/x",
            f"/data/{step}/meshes/rho",
        ]
        for path in candidates:
            try:
                var = f.inquire_variable(path)
                if var is None:
                    continue
                # ADIOS2 gives attributes via available_attributes()
                # We'll read from attributes using their full name:
                attrs = f.available_attributes()
                # openPMD-style attribute names often look like: "<varpath>/gridSpacing"
                gs_key = path + "/gridSpacing"
                gu_key = path + "/gridUnitSI"
                if gs_key in attrs:
                    grid_spacing = np.array(f.read_attribute(gs_key), dtype=np.float64)
                else:
                    continue
                grid_unit_si = 1.0
                if gu_key in attrs:
                    grid_unit_si = float(np.array(f.read_attribute(gu_key)).reshape(-1)[0])
                # gridSpacing * gridUnitSI gives SI meters
                grid_spacing_si = grid_spacing * grid_unit_si
                # ensure length 3
                if grid_spacing_si.size == 3:
                    return float(grid_spacing_si[0]), float(grid_spacing_si[1]), float(grid_spacing_si[2])
            except Exception:
                pass
    
        # If we get here, we couldn't find it
        raise RuntimeError("Could not determine cell size from openPMD meshes (gridSpacing/gridUnitSI).")
    
    
    def _read_positions_and_weight(self, fn, step, species):
        self._load_meta()
        dx, dy, dz = float(self.Dx_SI), float(self.Dy_SI), float(self.Dz_SI)
    
        with adios2.Stream(fn, "r") as f:
            for _ in f.steps():
                av = f.available_variables()
                base = f"/data/{step}/particles/{species}"
    
                # if species not present, return "no particles"
                key_x = f"{base}/position/x"
                key_w = f"{base}/weighting"
                if key_x not in av or key_w not in av:
                    return None, None, None, None
    
                x = f.read(key_x).astype(np.float64)
                y = f.read(f"{base}/position/y").astype(np.float64) if f"{base}/position/y" in av else np.zeros_like(x)
                z = f.read(f"{base}/position/z").astype(np.float64) if f"{base}/position/z" in av else np.zeros_like(x)
                w = f.read(key_w).astype(np.float64)
    
                if len(x) == 0:
                    return None, None, None, None
    
                # offsets are optional
                ox = f"{base}/positionOffset/x"
                oy = f"{base}/positionOffset/y"
                oz = f"{base}/positionOffset/z"
    
                if ox in av and oy in av and oz in av:
                    xo = f.read(ox).astype(np.int64)
                    yo = f.read(oy).astype(np.int64)
                    zo = f.read(oz).astype(np.int64)
    
                    if np.nanmax(x) <= 1.5 and np.nanmax(y) <= 1.5 and np.nanmax(z) <= 1.5:
                        x = (xo + x) * dx
                        y = (yo + y) * dy
                        z = (zo + z) * dz
                    else:
                        x = x * dx; y = y * dy; z = z * dz
                else:
                    # assume cell units if values look like indices
                    if np.nanmax(x) > 2.0 and np.nanmax(x) < 1e7:
                        x = x * dx; y = y * dy; z = z * dz
    
                return x, y, z, w