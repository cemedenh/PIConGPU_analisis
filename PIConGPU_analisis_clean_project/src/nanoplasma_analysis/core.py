
"""
nanoplasma_analysis.py

Lightweight analysis helpers for PIConGPU openPMD .bp5 outputs read via adios2,
matching the workflow used in Cristian's notebooks (analysis-001/002).

Design goals:
- keep the same data access patterns (adios2.Stream, /data/{step}/... paths)
- minimal magic, explicit parameters
- each plot is a function you can call from notebooks/scripts

Typical use:

from nanoplasma_analysis import NanoPlasmaRun

run = NanoPlasmaRun(path="/p/scratch/.../simOutput", laser_peak_at_target=89603)
run.plot_particle_number(species="He_e")
run.plot_charge_state_evolution(Zmax=2)
run.plot_field_e_i_maps(file_index=-1)
run.plot_laser_envelope_and_electron_yield(species="He_e", tau_fwhm_fs=40, I0_Wcm2=4e14, lambda_um=0.8)
run.plot_kinetic_energy_spectra(species="He_e", file_indices=[4,6,8], bins=(0,20,100), normalize=True)
run.plot_tail_decay_vs_time(species="He_e", Emin_eV=1, Emax_eV=25, Ebins_max_eV=200, Nbins=400)
run.plot_vmi(species="He_e", file_index=-1, plane=("x","y"), Nbins=300)
run.plot_angular_distribution(species="He_e", file_index=-1, axis="x")
run.plot_angular_distribution_vs_time(species="He_e", axis="x")
run.plot_asymmetry_vs_time(species="He_e", axis="x")  # A = (N+ - N-)/(N+ + N-)
run.plot_relative_electron_ion_asymmetry_vs_time(axis="x")  # e positions relative to ion COM

"""

from __future__ import annotations
import glob
import os
import tempfile

import numpy as np
try:
    import adios2
except ImportError:  # allow importing/testing package without ADIOS2 installed
    class _MissingADIOS2:
        def __getattr__(self, name):
            raise ImportError("adios2 is required for reading PIConGPU/openPMD .bp5 files. Install adios2 in the analysis environment.")
    adios2 = _MissingADIOS2()
import matplotlib.pyplot as plt
import scipy.constants as sc

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


def extract_step_from_filename(filename: str) -> int:
    # matches notebook: int((filename.rpartition('_')[2]).rpartition('.')[0])
    return int((filename.rpartition('_')[2]).rpartition('.')[0])


class NanoPlasmaRun:
    def __init__(self, path: str, laser_peak_at_target: int, files_glob: str = "/openPMD/*.bp5"):
        self.path = path.rstrip("/")
        self.laser_peak_at_target = int(laser_peak_at_target)

        self.files = glob.glob(self.path + files_glob)
        self.files.sort(key=extract_step_from_filename)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found with: {self.path + files_glob}")

        # meta populated lazily
        self._meta_loaded = False

    # -------------------------
    # Meta + helpers
    # -------------------------
    def _load_meta(self):
        if self._meta_loaded:
            return

        with adios2.Stream(self.files[0], "r") as f:
            for _ in f.steps():
                t0 = extract_step_from_filename(self.files[0])

                # grid shape from a known field
                Nz, Ny, Nx = np.array(
                    f.available_variables().get(f"/data/{t0}/fields/B/x")["Shape"].split(", "),
                    dtype=int,
                )

                unit_length_SI = f.read_attribute(f"/data/{t0}/unit_length")
                Dy_SI = f.read_attribute(f"/data/{t0}/cell_height") * unit_length_SI
                Dx_SI = f.read_attribute(f"/data/{t0}/cell_width") * unit_length_SI
                Dz_SI = f.read_attribute(f"/data/{t0}/cell_depth") * unit_length_SI

                unit_time_SI = f.read_attribute(f"/data/{t0}/unit_time")
                Dt_SI = f.read_attribute(f"/data/{t0}/dt") * unit_time_SI
                dt_fs = Dt_SI * 1e15

        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.Dx_SI, self.Dy_SI, self.Dz_SI = float(Dx_SI), float(Dy_SI), float(Dz_SI)
        self.Dt_SI = float(Dt_SI)
        self.dt_fs = float(dt_fs)
        self._meta_loaded = True

    def time_fs_from_step(self, step: int) -> float:
        self._load_meta()
        return (int(step) - self.laser_peak_at_target) * self.dt_fs

    def _num_particles(self, filename: str, step: int, species: str) -> int:
        """Return total particle count for a species at a given step. Returns 0 if missing."""
        try:
            with adios2.Stream(filename, "r") as f:
                for _ in f.steps():
                    npp = f.read(f"/data/{step}/particles/{species}/particlePatches/numParticles")
                    return int(np.sum(npp))
        except Exception:
            return 0
        return 0

    def _mass_kg(self, species: str, *, mass_kg: float | None = None) -> float:
        """
        Return particle mass in kg for the given species.

        - If mass_kg is provided, it is used directly (safest for new species).
        - Electrons: He_e (and common aliases) -> m_e
        - Helium ions: He_i* -> 4 u

        Extend mapping here if you add more ion species.
        """
        if mass_kg is not None:
            return float(mass_kg)

        s = (species or "").strip()
        if s in ("He_e", "e", "electron") or s.endswith("_e"):
            return sc.m_e
        if s.startswith("He_i") or s.endswith("_i"):
            return 4.0 * sc.atomic_mass
        raise ValueError(f"Unknown mass for species='{species}'. Pass mass_kg explicitly.")


    def _read_momentum_and_weight(self, filename: str, step: int, species: str):
        """
        Returns (px, py, pz, w) with momentum in SI units (kg*m/s) after applying unitSI.
        If the species is empty at this step or variables are missing, returns (None, None, None, None).
        """
        # robust empty-check (important for neutral-start runs)
        if self._num_particles(filename, step, species) == 0:
            return None, None, None, None

        try:
            with adios2.Stream(filename, "r") as f:
                for _ in f.steps():
                    base = f"/data/{step}/particles/{species}"
                    try:
                        px = f.read(f"{base}/momentum/x")
                        py = f.read(f"{base}/momentum/y")
                        pz = f.read(f"{base}/momentum/z")
                        w = f.read(f"{base}/weighting")
                    except Exception:
                        return None, None, None, None

                    try:
                        unit_p = f.read_attribute(f"{base}/momentum/x/unitSI")
                    except Exception:
                        unit_p = 1.0
                    px = px * unit_p
                    py = py * unit_p
                    pz = pz * unit_p
                    return px, py, pz, w
        except Exception:
            return None, None, None, None

        return None, None, None, None


    def _read_positions_and_weight(self, filename: str, step: int, species: str):
        """
        Read particle positions in SI meters plus macro-particle weights.

        PIConGPU/openPMD usually stores particle coordinates as:
            position/*       fractional position inside the cell
            positionOffset/* integer cell index

        This method reconstructs absolute SI positions as
            (positionOffset + position) * cell_size

        It also falls back to older/simple outputs where positionOffset is absent.
        If the species is empty or missing, returns (None, None, None, None).
        """
        if self._num_particles(filename, step, species) == 0:
            return None, None, None, None

        self._load_meta()
        dx, dy, dz = float(self.Dx_SI), float(self.Dy_SI), float(self.Dz_SI)

        try:
            with adios2.Stream(filename, "r") as f:
                for _ in f.steps():
                    base = f"/data/{step}/particles/{species}"
                    try:
                        x = np.asarray(f.read(f"{base}/position/x"), dtype=np.float64)
                        y = np.asarray(f.read(f"{base}/position/y"), dtype=np.float64)
                        z = np.asarray(f.read(f"{base}/position/z"), dtype=np.float64)
                        w = np.asarray(f.read(f"{base}/weighting"), dtype=np.float64)
                    except Exception:
                        return None, None, None, None

                    if x.size == 0 or y.size == 0 or z.size == 0:
                        return None, None, None, None

                    try:
                        xo = np.asarray(f.read(f"{base}/positionOffset/x"), dtype=np.int64)
                        yo = np.asarray(f.read(f"{base}/positionOffset/y"), dtype=np.int64)
                        zo = np.asarray(f.read(f"{base}/positionOffset/z"), dtype=np.int64)

                        # Standard PIConGPU case: position is fractional inside a cell.
                        if np.nanmax(np.abs(x)) <= 1.5 and np.nanmax(np.abs(y)) <= 1.5 and np.nanmax(np.abs(z)) <= 1.5:
                            x = (xo + x) * dx
                            y = (yo + y) * dy
                            z = (zo + z) * dz
                        else:
                            # Fallback: position was already cell-like but offsets exist.
                            x = x * dx
                            y = y * dy
                            z = z * dz
                    except Exception:
                        # No positionOffset. Try unitSI first; otherwise infer cell units.
                        try:
                            unit_x = f.read_attribute(f"{base}/position/x/unitSI")
                        except Exception:
                            unit_x = None

                        if unit_x is not None:
                            x = x * float(unit_x)
                            y = y * float(unit_x)
                            z = z * float(unit_x)
                        elif np.nanmax(np.abs(x)) > 2.0 and np.nanmax(np.abs(x)) < 1e7:
                            x = x * dx
                            y = y * dy
                            z = z * dz
                        # else: assume already SI

                    return x, y, z, w
        except Exception:
            return None, None, None, None

        return None, None, None, None

    # -------------------------
    # Mandatory plots
    # -------------------------
    def plot_particle_number(self, species: str = "He_e"):
        """
        Particle number vs time-step from macroParticlesCount.dat (as in notebook).
        """
        self._load_meta()
        fn = f"{self.path}/{species}_macroParticlesCount.dat"
        nt, ne = np.loadtxt(fn, usecols=(0, 1), unpack=True)
        t_fs = (nt - self.laser_peak_at_target) * self.dt_fs

        plt.figure(figsize=(12,4))
        plt.plot(t_fs[1:], ne[1:]/ne[-1], "o", ms=4)
        plt.xlabel("time - t_peak (fs)")
        plt.ylabel("normalized macro-particle count")
        plt.title(f"Particle number evolution ({species})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_charge_state_evolution(
        self,
        ion_species: str = "He_i",
        Zmax: int = 2,
        *,
        show_laser_envelope: bool = True,
        tau_fwhm_fs: float = 40.0,
        I0_Wcm2: float = 4e14,
        lambda_um: float = 0.8,
    ):
        """Charge-state evolution from boundElectrons (Z = Zmax - boundElectrons).

        If show_laser_envelope=True, overlays the laser envelope as Up(t) (ponderomotive energy)
        on a secondary axis, matching the style used in the project notebooks.
        """
        self._load_meta()

        Z_vals = np.arange(0, Zmax + 1)
        counts = np.zeros((len(self.files), len(Z_vals)), dtype=float)
        t_files = np.zeros(len(self.files), dtype=float)

        bins = np.arange(-0.5, Zmax + 1.5, 1.0)

        for i, fn in enumerate(self.files):
            step = extract_step_from_filename(fn)
            t_files[i] = self.time_fs_from_step(step)

            with adios2.Stream(fn, "r") as f:
                for _ in f.steps():
                    be = f.read(f"/data/{step}/particles/{ion_species}/boundElectrons")
                    w = f.read(f"/data/{step}/particles/{ion_species}/weighting")

            Z = Zmax - be
            h, _ = np.histogram(Z, bins=bins, weights=w)
            counts[i, :] = h[: len(Z_vals)]

        denom = np.sum(counts, axis=1)
        denom[denom == 0] = np.nan
        frac = counts / denom[:, None]

        order = np.argsort(t_files)
        t_files = t_files[order]
        frac = frac[order, :]

        fig, ax = plt.subplots(figsize=(12, 5))

        for j, Z in enumerate(Z_vals):
            ax.plot(t_files, frac[:, j], "o-", ms=3, label=f"Z={Z}")

        ax.set_xlabel("time - t_peak (fs)")
        ax.set_ylabel("fraction")
        ax.set_title(f"Charge state evolution ({ion_species})")
        ax.grid(True)
        ax.legend(loc="best")

        if show_laser_envelope:
            # Laser envelope as ponderomotive energy Up(t) in eV.
            # Use a dense time grid for a smooth curve (independent of diagnostic cadence).
            t_dense = np.linspace(np.nanmin(t_files), np.nanmax(t_files), 1500)
            I_dense = I0_Wcm2 * np.exp(-4 * np.log(2) * (t_dense / tau_fwhm_fs) ** 2)
            Up_dense = 9.33e-14 * I_dense * (lambda_um ** 2)

            ax2 = ax.twinx()
            ax2.plot(t_dense, Up_dense, lw=2.0, alpha=0.55)
            ax2.fill_between(t_dense, Up_dense, 0, alpha=0.15)
            ax2.set_ylabel(r"$U_p$ (eV)")
            ax2.set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_field_e_i_maps(
        self,
        file_index: int = -1,
        y_nm: float = 300.0,
        electron_density_field: str = "He_e_all_density",
        ion_charge_density_field: str = "He_i_all_density",
        I0_Wcm2: float = 4e14,
        *,
        zoom_ions: bool = True,
        ion_zoom_pad_nm: float = 50.0,
        lognorm_vmin: float = 1e-6,
        show: bool = True,
        savepath: str | None = None,
    ):
        """3-panel plot: field (normalized), electron density slice, ion *charge* density slice.

        - Field: Ex normalized to E0 from I0_Wcm2 (same as notebook)
        - e-panel: He_e_all_density (typically charge density) shown in x–z at fixed y
        - ion-panel: He_i_all_density shown as **charge density**, so it reflects ionization level
          (higher charge state -> higher charge density for same number density).

        If zoom_ions=True, the ion panel auto-zooms to the region where the ion signal lives,
        since ions move much less than electrons.
        """
        self._load_meta()
        fn = self.files[file_index]
        step = extract_step_from_filename(fn)
        t_fs = self.time_fs_from_step(step)

        with adios2.Stream(fn, "r") as f:
            for _ in f.steps():
                # Field (slice at z = Nz//2)
                unit_fieldE = f.read_attribute(f"/data/{step}/fields/E/x/unitSI")
                Ex = f.read(f"/data/{step}/fields/E/x", start=[int(self.Nz // 2), 0, 0], count=[1, self.Ny, self.Nx]) * unit_fieldE
                Ey = f.read(f"/data/{step}/fields/E/y", start=[int(self.Nz // 2), 0, 0], count=[1, self.Ny, self.Nx]) * unit_fieldE
                Ez = f.read(f"/data/{step}/fields/E/z", start=[int(self.Nz // 2), 0, 0], count=[1, self.Ny, self.Nx]) * unit_fieldE

                # density slices (x–z plane at y = y_nm)
                y_idx = int((y_nm * 1e-9) / self.Dy_SI)
                y_idx = max(0, min(self.Ny - 1, y_idx))

                unit_rho_e = f.read_attribute(f"/data/{step}/fields/{electron_density_field}/unitSI")
                rho_e = f.read(f"/data/{step}/fields/{electron_density_field}", start=[0, y_idx, 0], count=[self.Nz, 1, self.Nx]) * unit_rho_e

                rho_i = None
                if f"/data/{step}/fields/{ion_charge_density_field}" in f.available_variables():
                    unit_rho_i = f.read_attribute(f"/data/{step}/fields/{ion_charge_density_field}/unitSI")
                    rho_i = f.read(f"/data/{step}/fields/{ion_charge_density_field}", start=[0, y_idx, 0], count=[self.Nz, 1, self.Nx]) * unit_rho_i

        # normalize field to E0 corresponding to I0
        E0 = np.sqrt(2 * I0_Wcm2 * 1e4 / (sc.c * sc.epsilon_0))
        plot_field = Ex[0, :, :] / E0

        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        # 1) field
        im0 = axes[0].imshow(
            plot_field,
            cmap="RdBu_r",
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            extent=(0, self.Nx * self.Dx_SI * 1e9, 0, self.Ny * self.Dy_SI * 1e9),
            vmin=-2.0,
            vmax=2.0,
        )
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title(f"Ex/E0 @ t={t_fs:.2f} fs")
        axes[0].set_xlabel("x (nm)")
        axes[0].set_ylabel("y (nm)")

        # 2) electron density (x–z)
        rho_e2 = rho_e[:, 0, :]
        im1 = axes[1].imshow(
            rho_e2 / (np.max(rho_e2) + 1e-300),
            cmap="plasma",
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            extent=(0, self.Nx * self.Dx_SI * 1e9, 0, self.Nz * self.Dz_SI * 1e9),
            norm=plt.matplotlib.colors.LogNorm(vmin=lognorm_vmin, vmax=1.0),
        )
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(f"e- density (slice y={y_nm:.0f} nm)")
        axes[1].set_xlabel("x (nm)")
        axes[1].set_ylabel("z (nm)")

        # 3) ion charge density (x–z) with optional zoom
        if rho_i is not None:
            rho_i2 = rho_i[:, 0, :]
            im2 = axes[2].imshow(
                rho_i2 / (np.max(rho_i2) + 1e-300),
                cmap="plasma",
                origin="lower",
                interpolation="nearest",
                aspect="equal",
                extent=(0, self.Nx * self.Dx_SI * 1e9, 0, self.Nz * self.Dz_SI * 1e9),
                norm=plt.matplotlib.colors.LogNorm(vmin=lognorm_vmin, vmax=1.0),
            )
            fig.colorbar(im2, ax=axes[2])
            axes[2].set_title(f"ion charge density (slice y={y_nm:.0f} nm)")
            axes[2].set_xlabel("x (nm)")
            axes[2].set_ylabel("z (nm)")

            if zoom_ions:
                # auto-zoom to where ions are (based on threshold)
                norm_i = rho_i2 / (np.max(rho_i2) + 1e-300)
                thr = 1e-3
                idx = np.argwhere(norm_i > thr)
                if idx.size > 0:
                    zmin, xmin = idx.min(axis=0)
                    zmax, xmax = idx.max(axis=0)

                    pad_x = int((ion_zoom_pad_nm * 1e-9) / self.Dx_SI)
                    pad_z = int((ion_zoom_pad_nm * 1e-9) / self.Dz_SI)

                    xmin = max(0, xmin - pad_x)
                    xmax = min(self.Nx - 1, xmax + pad_x)
                    zmin = max(0, zmin - pad_z)
                    zmax = min(self.Nz - 1, zmax + pad_z)

                    axes[2].set_xlim(xmin * self.Dx_SI * 1e9, xmax * self.Dx_SI * 1e9)
                    axes[2].set_ylim(zmin * self.Dz_SI * 1e9, zmax * self.Dz_SI * 1e9)
        else:
            axes[2].axis("off")
            axes[2].text(0.5, 0.5, f"'{ion_charge_density_field}' not found", ha="center", va="center")

        plt.suptitle("Field / electron / ion maps", y=1.02)
        plt.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        
    def make_field_e_i_gif(
        self,
        out_gif: str,
        *,
        y_nm: float = 300.0,
        I0_Wcm2: float = 4e14,
        zoom_ions: bool = True,
        ion_zoom_pad_nm: float = 50.0,
        ion_charge_density_field: str | None = "He_i_all_density",
        electron_density_field: str = "He_e_all_density",
        fps: int = 10,
        every: int = 1,
        ):
        """Create an animated GIF over time from plot_field_e_i_maps.
    
        This function is robust against older/newer signatures of plot_field_e_i_maps by
        only passing arguments that it is guaranteed to accept in the current script.
    
        Parameters
        ----------
        out_gif:
            Output gif filename.
        every:
            Use every Nth file to reduce gif size (e.g. every=2).
        fps:
            Frames per second in the gif.
        """
        self._load_meta()
    
        out_gif = os.path.abspath(out_gif)
        os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    
        with tempfile.TemporaryDirectory() as td:
            frame_paths = []
            for idx in range(0, len(self.files), max(1, int(every))):
                png_path = os.path.join(td, f"frame_{idx:05d}.png")
    
                # IMPORTANT: do NOT pass ion_density_field here (signature mismatch caused your error)
                self.plot_field_e_i_maps(
                    file_index=idx,
                    y_nm=y_nm,
                    electron_density_field=electron_density_field,
                    ion_charge_density_field=ion_charge_density_field,
                    zoom_ions=zoom_ions,
                    ion_zoom_pad_nm=ion_zoom_pad_nm,
                    I0_Wcm2=I0_Wcm2,
                    show=False,
                    savepath=png_path,
                )
                frame_paths.append(png_path)
    
            images = [imageio.imread(fp) for fp in frame_paths]
            imageio.mimsave(out_gif, images, fps=fps)

        return out_gif

    def plot_laser_envelope_and_electron_yield(
        self,
        species: str = "He_e",
        tau_fwhm_fs: float = 40.0,
        I0_Wcm2: float = 4e14,
        lambda_um: float = 0.8,
        t_ref_fs: float = 0.0,
    ):
        """
        2-panel plot (same as notebook): ponderomotive energy Up(t) and electron yield.
        Electron yield is taken from {species}_macroParticlesCount.dat.
        """
        self._load_meta()
        fn = f"{self.path}/{species}_macroParticlesCount.dat"
        nt, ne = np.loadtxt(fn, usecols=(0, 1), unpack=True)
        t_fs = (nt - self.laser_peak_at_target) * self.dt_fs
        ne_norm = ne / ne[-1]

        I_t = I0_Wcm2 * np.exp(-4 * np.log(2) * (t_fs / tau_fwhm_fs) ** 2)
        Up_t = 9.33e-14 * I_t * (lambda_um ** 2)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(16, 7), sharex=True,
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05}
        )

        ax_top.plot(t_fs, Up_t, lw=2)
        ax_top.fill_between(t_fs, Up_t, 0, alpha=0.25)
        ax_top.set_ylabel(r"$U_p$ (eV)")
        ax_top.set_title("Laser envelope (ponderomotive energy) + electron yield")
        Up_ref = np.interp(t_ref_fs, t_fs, Up_t)
        ax_top.axhline(Up_ref, color="black", linestyle="--", linewidth=1.2)
        ax_top.axvline(t_ref_fs, color="black", linestyle=":", linewidth=1.2)
        ax_top.grid(True)

        ax_bot.plot(t_fs, ne_norm, "o", ms=4)
        ax_bot.set_xlabel("time - t_peak (fs)")
        ax_bot.set_ylabel(r"$e^-$ yield (normalized)")
        ax_bot.axvline(t_ref_fs, color="black", linestyle=":", linewidth=1.2)
        ax_bot.grid(True)

        plt.tight_layout()
        plt.show()
        plt.close()


    def plot_kinetic_energy_spectra(
        self,
        species: str = "He_e",
        file_indices=None,
        bins: tuple[float, float, int] = (0.0, 20.0, 100),
        normalize: bool = True,
        logy: bool = False,
        *,
        label_every: int = 1,
        lw: float = 2.0,
        alpha: float = 0.85,
        mass_kg: float | None = None,
    ):
        """Kinetic energy spectra dN/dE for many timesteps.

        - If file_indices is None or "all": plot all outputs.
        - You can also pass a slice (e.g. slice(None, None, 2)) or an explicit list of indices.
        - Robust to empty species: skips those frames.

        Parameters
        ----------
        mass_kg:
            Optional explicit mass (kg). Strongly recommended for non-standard species.
            If omitted, uses _mass_kg(species).
        """
        self._load_meta()

        if file_indices is None or file_indices == "all":
            indices = list(range(len(self.files)))
        elif isinstance(file_indices, slice):
            indices = list(range(len(self.files)))[file_indices]
        else:
            indices = list(file_indices)

        E0, E1, Nb = bins
        edges = np.linspace(E0, E1, Nb + 1)
        Emid = 0.5 * (edges[1:] + edges[:-1])

        m_kg = self._mass_kg(species, mass_kg=mass_kg)

        plt.figure(figsize=(16, 4))

        line_count = 0
        for idx in indices:
            fn = self.files[idx]
            step = extract_step_from_filename(fn)

            px, py, pz, w = self._read_momentum_and_weight(fn, step, species)
            if px is None:
                continue

            E_eV = (px * px + py * py + pz * pz) / (2.0 * m_kg) / sc.e

            hist, _ = np.histogram(E_eV, bins=edges, weights=w)
            dNdE = hist / np.diff(edges)

            if normalize:
                s = max(dNdE)
                if s > 0:
                    dNdE = dNdE / s

            lbl = None
            if (label_every is not None) and (line_count % max(1, label_every) == 0):
                lbl = f"t={self.time_fs_from_step(step):.1f} fs"

            plt.plot(Emid, dNdE, lw=lw, alpha=alpha, label=lbl)
            line_count += 1

        plt.xlabel("E (eV)")
        plt.ylabel("normalized dN/dE" if normalize else "dN/dE")
        plt.title(f"Kinetic energy spectra ({species})")
        if logy:
            plt.yscale("log")
        plt.grid(True)

        if label_every is not None:
            plt.legend(ncol=2, fontsize=8)

        plt.tight_layout()
        plt.show()
        plt.close()


    def plot_tail_decay_vs_time(
        self,
        species: str = "He_e",
        Emin_eV: float = 1.0,
        Emax_eV: float = 25.0,
        Ebins_max_eV: float = 200.0,
        Nbins: int = 400,
        *,
        plot_characteristic_energy: bool = False,
        min_bins: int = 5,
        mass_kg: float | None = None,
    ):
        """Fit an exponential tail f(E) ~ exp(-k E) at each output and plot k(t).

        Uses:
          log(dN/dE) = a + m E   with   k = -m   (units 1/eV)

        Optionally plot 1/k (characteristic energy scale in eV).

        Robust to empty species: skips those frames.

        Parameters
        ----------
        mass_kg:
            Optional explicit mass (kg). If omitted, uses _mass_kg(species).
            IMPORTANT for ions because mass attribute is often missing in the output.
        """
        self._load_meta()

        Ebins = np.linspace(0.0, Ebins_max_eV, Nbins + 1)
        Emid = 0.5 * (Ebins[1:] + Ebins[:-1])

        k_decay = np.full(len(self.files), np.nan)
        t_fs = np.full(len(self.files), np.nan)

        m_kg = self._mass_kg(species, mass_kg=mass_kg)

        for i, fn in enumerate(self.files):
            step = extract_step_from_filename(fn)
            t_fs[i] = self.time_fs_from_step(step)

            px, py, pz, w = self._read_momentum_and_weight(fn, step, species)
            if px is None:
                continue

            E_eV = (px * px + py * py + pz * pz) / (2.0 * m_kg) / sc.e

            hist, _ = np.histogram(E_eV, bins=Ebins, weights=w)
            dNdE = hist / np.diff(Ebins)

            mask = (Emid >= Emin_eV) & (Emid <= Emax_eV) & (dNdE > 0)
            if np.sum(mask) < max(2, int(min_bins)):
                continue

            x = Emid[mask]
            y = np.log(dNdE[mask])

            m, a = np.polyfit(x, y, 1)
            k = -m
            if k > 0:
                k_decay[i] = k

        # clean + sort
        valid = np.isfinite(t_fs) & np.isfinite(k_decay)
        t = t_fs[valid]
        k = k_decay[valid]
        order = np.argsort(t)
        t = t[order]
        k = k[order]

        plt.figure(figsize=(10, 4))
        if plot_characteristic_energy:
            plt.plot(t, 1.0 / (k + 1e-300), "o-", ms=4)
            plt.ylabel(r"characteristic energy $1/k$ (eV)")
        else:
            plt.plot(t, k, "o-", ms=4)
            plt.ylabel(r"decay rate $k$ (1/eV)")
        plt.xlabel("time - t_peak (fs)")
        plt.title(f"Exponential tail decay vs time ({species}), fit {Emin_eV}-{Emax_eV} eV")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        return t, (1.0 / (k + 1e-300) if plot_characteristic_energy else k)


    def plot_vmi(
        self,
        species: str = "He_e",
        file_index: int = -1,
        plane: tuple[str, str] = ("x", "y"),
        Nbins: int = 300,
    ):
        """
        Synthetic VMI: 2D histogram of momentum components (px vs py by default).
        Robust to empty species: prints a message and returns None.
        """
        self._load_meta()
        fn = self.files[file_index]
        step = extract_step_from_filename(fn)

        px, py, pz, w = self._read_momentum_and_weight(fn, step, species)
        if px is None:
            print(f"[plot_vmi] No particles for species='{species}' at step {step}.")
            return None

        comp = {"x": px, "y": py, "z": pz}
        a, b = plane
        H, xed, yed = np.histogram2d(comp[a], comp[b], bins=Nbins, weights=w)

        plt.figure(figsize=(6, 6))
        plt.imshow(H.T, origin="lower", extent=(xed[0], xed[-1], yed[0], yed[-1]), aspect="equal")
        plt.xlabel(fr"$p_{a}$ (SI)")
        plt.ylabel(fr"$p_{b}$ (SI)")
        plt.title(f"Synthetic VMI ({species}) at t={self.time_fs_from_step(step):.0f} fs")
        plt.colorbar(label="counts (arb.)")
        plt.tight_layout()
        plt.show()
        plt.close()

        return H, xed, yed


    def plot_angular_distribution(
        self,
        species: str = "He_e",
        file_index: int = -1,
        axis: str = "x",
        n_mu: int = 120,
    ):
        """
        Angular distribution for one time: histogram of mu = p_axis/|p|.
        Robust to empty species.
        """
        fn = self.files[file_index]
        step = extract_step_from_filename(fn)

        px, py, pz, w = self._read_momentum_and_weight(fn, step, species)
        if px is None:
            print(f"[plot_angular_distribution] No particles for species='{species}' at step {step}.")
            return None

        p = np.sqrt(px * px + py * py + pz * pz) + 1e-300
        comp = {"x": px, "y": py, "z": pz}
        mu = comp[axis] / p

        mu_edges = np.linspace(-1, 1, n_mu + 1)
        mu_mid = 0.5 * (mu_edges[1:] + mu_edges[:-1])

        hist, _ = np.histogram(mu, bins=mu_edges, weights=w)
        hist = hist / (np.sum(hist) + 1e-300)

        plt.figure(figsize=(7, 4))
        plt.plot(mu_mid, hist, "o-", ms=3)
        plt.xlabel(fr"$\mu=\cos	heta$ (w.r.t. {axis}-axis)")
        plt.ylabel("normalized counts")
        plt.title(f"Angular distribution ({species}) at t={self.time_fs_from_step(step):.0f} fs")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        return mu_mid, hist


    def plot_angular_distribution_vs_time(
        self,
        species: str = "He_e",
        axis: str = "x",
        n_mu: int = 120,
        *,
        file_indices=None,                 # "all", list, slice, or None
        t_range_fs: tuple[float, float] | None = None,
        skip_before_fs: float | None = None,
    ):
        """
        Heatmap of angular distribution vs time.
    
        Improvements vs old version:
        - robust to empty species (skips those frames)
        - can restrict to a time window (t_range_fs) or skip early times (skip_before_fs)
        - can select outputs via file_indices="all"/list/slice, like kinetic plots
    
        Normalization: each timestep histogram is normalized to 1 (shape only).
        """
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
    
        # Select file indices
        if file_indices is None or file_indices == "all":
            indices = list(range(len(self.files)))
        elif isinstance(file_indices, slice):
            indices = list(range(len(self.files)))[file_indices]
        else:
            indices = list(file_indices)
    
        mu_edges = np.linspace(-1, 1, n_mu + 1)
        mu_mid   = 0.5 * (mu_edges[1:] + mu_edges[:-1])
    
        H = []
        t_list = []
    
        for idx in indices:
            fn = self.files[idx]
            step = extract_step_from_filename(fn)
            t_fs = self.time_fs_from_step(step)
    
            # time filtering
            if skip_before_fs is not None and t_fs < skip_before_fs:
                continue
            if t_range_fs is not None:
                tmin, tmax = t_range_fs
                if (t_fs < tmin) or (t_fs > tmax):
                    continue
    
            px, py, pz, w = self._read_momentum_and_weight(fn, step, species)
            if px is None or w is None:
                continue
    
            p = np.sqrt(px*px + py*py + pz*pz) + 1e-300
            comp = {"x": px, "y": py, "z": pz}[axis]
            mu = comp / p
    
            hist, _ = np.histogram(mu, bins=mu_edges, weights=w)
            hist = hist / (np.sum(hist) + 1e-30)  # normalize per time
            H.append(hist)
            t_list.append(t_fs)
    
        if len(H) == 0:
            print(f"[plot_angular_distribution_vs_time] No valid frames for species='{species}'.")
            return None
    
        H = np.array(H).T
        t = np.array(t_list)
        order = np.argsort(t)
        t = t[order]
        H = H[:, order]
    
        plt.figure(figsize=(10,3))
        plt.imshow(
            H,
            origin="lower",
            aspect="auto",
            extent=[t.min(), t.max(), mu_mid[0], mu_mid[-1]],
            #interpolation="none",
        )
        plt.xlabel("time (fs) (0 fs = laser peak)")
        plt.ylabel(r"$\cos\theta$")
        plt.title(f"{species}: angular distribution vs time (axis={axis})")
        plt.colorbar(label="normalized counts")
        plt.tight_layout()
        plt.show()
        plt.close()
    
        return t, H

    def plot_asymmetry_vs_time(
    
        self,
        species: str = "He_e",
        axis: str = "x",
        *,
        file_indices=None,                     # "all", list, slice, or None
        t_range_fs: tuple[float, float] | None = None,
        skip_before_fs: float | None = None,
        center_component: bool = True,         # subtract weighted mean to remove tiny drift
    
        # --- NEW: laser overlay options ---
        overlay_laser: bool = False,
        overlay_what: str = "carrier+envelope",   # "carrier", "envelope", "carrier+envelope"
        envelope_model: str = "gaussian",         # "gaussian" or "sin2"
        tau_fwhm_fs: float = 7.0,                 # intensity FWHM (fs)
        lambda_um: float = 0.8,
        I0_Wcm2: float = 6e13,                    # only used for correct scaling (we normalize anyway)
        phase_rad: float = 0.0,                   # phase shift of the carrier (rad)
        overlay_alpha: float = 0.35,
        overlay_lw: float = 1.5,
    ):
        """
        Asymmetry vs time:
            A(t) = (N+ - N-) / (N+ + N-)
    
        NEW:
        Optional laser overlay (normalized) to compare A(t) with the laser oscillation:
          - carrier: E(t)/E0
          - envelope: |E|_env/E0  (amplitude envelope)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.constants as sc
    
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
    
        # Select file indices
        if file_indices is None or file_indices == "all":
            indices = list(range(len(self.files)))
        elif isinstance(file_indices, slice):
            indices = list(range(len(self.files)))[file_indices]
        else:
            indices = list(file_indices)
    
        t_list = []
        A_list = []
    
        for idx in indices:
            fn = self.files[idx]
            step = extract_step_from_filename(fn)
            t_fs = self.time_fs_from_step(step)
    
            # time filtering
            if skip_before_fs is not None and t_fs < skip_before_fs:
                continue
            if t_range_fs is not None:
                tmin, tmax = t_range_fs
                if (t_fs < tmin) or (t_fs > tmax):
                    continue
    
            # skip if total particles = 0
            if hasattr(self, "_num_particles"):
                if self._num_particles(fn, step, species) == 0:
                    continue
    
            px, py, pz, w = self._read_momentum_and_weight(fn, step, species)
            if px is None or w is None:
                continue
    
            comp = {"x": px, "y": py, "z": pz}[axis]
    
            if center_component:
                wsum = np.sum(w) + 1e-30
                comp = comp - (np.sum(w * comp) / wsum)
    
            Np = np.sum(w[comp > 0])
            Nm = np.sum(w[comp < 0])
            denom = (Np + Nm) + 1e-30
            A = (Np - Nm) / denom
    
            t_list.append(t_fs)
            A_list.append(A)
    
        if len(t_list) == 0:
            print(f"[plot_asymmetry_vs_time] No valid frames for species='{species}'.")
            return None
    
        t = np.array(t_list)
        A = np.array(A_list)
        order = np.argsort(t)
        t, A = t[order], A[order]
    
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(t, A, "o-", ms=4, label="Asymmetry A(t)")
        ax.axhline(0, lw=1)
        ax.set_xlabel("time (fs) (0 fs = laser peak)")
        ax.set_ylabel("asymmetry A")
        ax.set_title(f"Asymmetry vs time ({species}, axis={axis})")
        ax.grid(True)
    
        # ---- NEW: laser overlay on secondary axis ----
        if overlay_laser:
            # carrier frequency
            omega = 2.0 * np.pi * sc.c / (lambda_um * 1e-6)  # rad/s
    
            # build a dense time grid spanning the plotted range (smooth carrier)
            t_dense = np.linspace(float(np.min(t)), float(np.max(t)), 4000)
            t_dense_s = t_dense * 1e-15
    
            # intensity envelope (normalized to 1 at peak)
            if envelope_model == "gaussian":
                I_norm = np.exp(-4.0 * np.log(2.0) * (t_dense / tau_fwhm_fs) ** 2)
            elif envelope_model == "sin2":
                # sin^2 intensity envelope with same FWHM approx:
                # define half-duration T such that intensity FWHM ~ tau_fwhm_fs
                # For I = sin^2(pi t / (2T)) on [-T, T], FWHM occurs at sin^2 = 1/2 -> |t| = T/2
                # so FWHM ~ T. Use T = tau_fwhm_fs.
                T = float(tau_fwhm_fs)
                I_norm = np.zeros_like(t_dense)
                inside = np.abs(t_dense) <= T
                I_norm[inside] = np.sin(0.5 * np.pi * (t_dense[inside] / T + 1.0)) ** 2
            else:
                raise ValueError("envelope_model must be 'gaussian' or 'sin2'.")
    
            # amplitude envelope ~ sqrt(I)
            Eenv_norm = np.sqrt(I_norm)
    
            # carrier (normalized)
            Ecar_norm = Eenv_norm * np.cos(omega * t_dense_s + float(phase_rad))
    
            ax2 = ax.twinx()
            ax2.set_ylabel("laser (normalized)")
            # keep the overlay visually light
            if overlay_what in ("carrier", "carrier+envelope"):
                ax2.plot(t_dense, Ecar_norm, lw=overlay_lw, alpha=overlay_alpha, label="laser carrier (norm)")
            if overlay_what in ("envelope", "carrier+envelope"):
                ax2.plot(t_dense, +Eenv_norm, lw=overlay_lw, alpha=overlay_alpha, label="envelope (norm)")
                ax2.plot(t_dense, -Eenv_norm, lw=overlay_lw, alpha=overlay_alpha)
    
            # nice limits for normalized field
            ax2.set_ylim(-1.05, 1.05)
    
            # combined legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
        else:
            ax.legend(loc="best", fontsize=8)
    
        plt.tight_layout()
        plt.show()
        plt.close()
    
        return t, A


    def plot_relative_electron_ion_asymmetry_vs_time(
        self,
        electron_species: str = "He_e",
        ion_species: str = "He_i",
        axis: str = "x",
    ):
        """
        Electron position asymmetry RELATIVE to the ion center-of-mass.

        For each timestep:
          r_ion_COM = sum(w_i * r_i) / sum(w_i)
          r_rel = r_e - r_ion_COM
          A_rel = (N_rel_plus - N_rel_minus)/(N_rel_plus + N_rel_minus) along 'axis'.

        Robust to empty species: returns NaN for steps without ions or electrons.
        """
        t_list = []
        A_list = []

        for fn in self.files:
            step = extract_step_from_filename(fn)
            t_list.append(self.time_fs_from_step(step))

            xi, yi, zi, wi = self._read_positions_and_weight(fn, step, ion_species)
            if xi is None:
                A_list.append(np.nan)
                continue

            denom_i = np.sum(wi)
            if denom_i <= 0:
                A_list.append(np.nan)
                continue
            rcom = np.array([np.sum(wi * xi) / denom_i, np.sum(wi * yi) / denom_i, np.sum(wi * zi) / denom_i])

            xe, ye, ze, we = self._read_positions_and_weight(fn, step, electron_species)
            if xe is None:
                A_list.append(np.nan)
                continue

            rel = {"x": xe - rcom[0], "y": ye - rcom[1], "z": ze - rcom[2]}[axis]

            Np = np.sum(we[rel > 0])
            Nm = np.sum(we[rel < 0])
            denom = (Np + Nm)
            A = (Np - Nm) / denom if denom > 0 else np.nan
            A_list.append(A)

        t = np.array(t_list)
        A = np.array(A_list)
        order = np.argsort(t)

        plt.figure(figsize=(10, 4))
        plt.plot(t[order], A[order], "o-", ms=4)
        plt.xlabel("time - t_peak (fs)")
        plt.ylabel("relative-position asymmetry A_rel")
        plt.title(f"Electron position asymmetry relative to ion COM (axis={axis})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        return t[order], A[order]
        
    def plot_ion_radial_density_vs_time(
        self,
        ion_species: str = "He_i",
        *,
        r_max_nm: float = 2000.0,
        n_r: int = 250,
        center_mode: str = "first_com",
        center_xyz_m: tuple[float, float, float] | None = None,
        normalize_each_time: bool = True,
        show_speed: bool = True,
    ):
        """Ion radial density vs time, plus expansion metrics.

        Computes r = |r - r0| for ions each output and builds a heatmap dN/dr (or normalized per time),
        where r0 is chosen by:

        center_mode:
          - "first_com": ion center-of-mass from the first non-empty frame (recommended)
          - "each_com":  ion center-of-mass separately at each time (removes drift)
          - "given":     use center_xyz_m (meters)

        Also plots <r>(t) and optionally dr/dt as an estimate of expansion speed.

        Returns
        -------
        t_fs, r_mid_nm, H  where H has shape (n_r, n_t).
        """
        self._load_meta()

        # Determine reference center
        r0 = None
        if center_mode == "given":
            if center_xyz_m is None:
                raise ValueError("center_mode='given' requires center_xyz_m=(x0,y0,z0) in meters.")
            r0 = np.array(center_xyz_m, dtype=float)

        # Prepare bins
        r_edges_nm = np.linspace(0.0, float(r_max_nm), int(n_r) + 1)
        r_mid_nm = 0.5 * (r_edges_nm[1:] + r_edges_nm[:-1])

        H_rows = []
        t_list = []
        mean_r_nm = []
        rms_r_nm = []

        # First pass to find r0 for "first_com"
        if center_mode == "first_com":
            for fn in self.files:
                step = extract_step_from_filename(fn)
                xi, yi, zi, wi = self._read_positions_and_weight(fn, step, ion_species)
                if xi is None:
                    continue
                denom = np.sum(wi)
                if denom <= 0:
                    continue
                r0 = np.array([np.sum(wi * xi) / denom, np.sum(wi * yi) / denom, np.sum(wi * zi) / denom], dtype=float)
                break
            if r0 is None:
                print(f"[plot_ion_radial_density_vs_time] No ions found for species='{ion_species}'.")
                return None

        for fn in self.files:
            step = extract_step_from_filename(fn)
            t_list.append(self.time_fs_from_step(step))

            xi, yi, zi, wi = self._read_positions_and_weight(fn, step, ion_species)
            if xi is None:
                H_rows.append(np.full(int(n_r), np.nan))
                mean_r_nm.append(np.nan)
                rms_r_nm.append(np.nan)
                continue

            # choose center
            if center_mode == "each_com":
                denom = np.sum(wi)
                if denom <= 0:
                    H_rows.append(np.full(int(n_r), np.nan))
                    mean_r_nm.append(np.nan)
                    rms_r_nm.append(np.nan)
                    continue
                r0_t = np.array([np.sum(wi * xi) / denom, np.sum(wi * yi) / denom, np.sum(wi * zi) / denom], dtype=float)
            else:
                r0_t = r0

            dx = xi - r0_t[0]
            dy = yi - r0_t[1]
            dz = zi - r0_t[2]
            r_nm = np.sqrt(dx * dx + dy * dy + dz * dz) * 1e9

            hist, _ = np.histogram(r_nm, bins=r_edges_nm, weights=wi)
            dNdr = hist / np.diff(r_edges_nm)

            if normalize_each_time:
                s = np.nansum(dNdr)
                if s > 0:
                    dNdr = dNdr / s

            # metrics
            wsum = np.sum(wi)
            if wsum > 0:
                r_mean = np.sum(wi * r_nm) / wsum
                r_rms = np.sqrt(np.sum(wi * (r_nm ** 2)) / wsum)
            else:
                r_mean = np.nan
                r_rms = np.nan

            H_rows.append(dNdr)
            mean_r_nm.append(r_mean)
            rms_r_nm.append(r_rms)

        t = np.array(t_list, dtype=float)
        H = np.array(H_rows, dtype=float).T  # [r, t]
        mean_r_nm = np.array(mean_r_nm, dtype=float)
        rms_r_nm = np.array(rms_r_nm, dtype=float)

        # sort by time
        order = np.argsort(t)
        t = t[order]
        H = H[:, order]
        mean_r_nm = mean_r_nm[order]
        rms_r_nm = rms_r_nm[order]

        # plot heatmap + <r>
        fig, axes = plt.subplots(2 if show_speed else 1, 1, figsize=(12, 6 if show_speed else 4), sharex=True,
                                 gridspec_kw={"height_ratios": ([3, 1] if show_speed else [1])})

        ax0 = axes[0] if show_speed else axes
        im = ax0.imshow(
            H,
            origin="lower",
            aspect="auto",
            extent=[np.nanmin(t), np.nanmax(t), r_mid_nm[0], r_mid_nm[-1]],
            interpolation="none",
        )
        plt.colorbar(im, ax=ax0, label=("normalized dN/dr" if normalize_each_time else "dN/dr"))
        ax0.set_ylabel("r (nm)")
        ax0.set_title(f"{ion_species}: radial density vs time (center_mode={center_mode})")

        # overlay mean radius
        ax0.plot(t, mean_r_nm, "w-", lw=2, alpha=0.8, label=r"$\langle r \rangle$")
        ax0.plot(t, rms_r_nm, "w--", lw=1.5, alpha=0.8, label=r"$r_\mathrm{rms}$")
        ax0.legend(loc="upper left")

        if show_speed:
            ax1 = axes[1]
            # speed nm/fs from derivative of mean radius
            speed = np.full_like(mean_r_nm, np.nan)
            valid = np.isfinite(t) & np.isfinite(mean_r_nm)
            if np.sum(valid) >= 3:
                speed[valid] = np.gradient(mean_r_nm[valid], t[valid])
            ax1.plot(t, speed, "o-", ms=3)
            ax1.set_ylabel("d<r>/dt (nm/fs)")
            ax1.set_xlabel("time - t_peak (fs)")
            ax1.grid(True)
        else:
            ax0.set_xlabel("time - t_peak (fs)")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

        return t, r_mid_nm, H
    def plot_mean_temperature_vs_time(
        self,
        species: str = "He_e",
        ):
        """
        Compute and plot mean kinetic temperature vs time using:
    
            (3/2) k_B T = <E>
    
        Since we work in eV:
            T[eV] = (2/3) * <E>[eV]
    
        Robust:
        - skips frames with zero particles
        - uses correct mass via self._mass_kg(species)
    
        Returns:
            t (fs), T_mean (eV)
        """
    
        self._load_meta()
    
        t_list = []
        T_list = []
    
        # get correct particle mass
        m_kg = self._mass_kg(species)
    
        for fn in self.files:
            step = extract_step_from_filename(fn)
    
            # skip empty species
            if hasattr(self, "_num_particles"):
                if self._num_particles(fn, step, species) == 0:
                    continue
    
            px, py, pz, w = self._read_momentum_and_weight(fn, step, species)
            if px is None or w is None:
                continue
    
            p2 = px*px + py*py + pz*pz
            E_J = p2 / (2.0 * m_kg)
            E_eV = E_J / sc.e
    
            wsum = np.sum(w)
            if wsum == 0:
                continue
    
            meanE_eV = np.sum(w * E_eV) / wsum
            T_mean_eV = (2.0/3.0) * meanE_eV
    
            t_list.append(self.time_fs_from_step(step))
            T_list.append(T_mean_eV)
    
        if len(t_list) == 0:
            print(f"[plot_mean_temperature_vs_time] No valid frames for species='{species}'.")
            return None
    
        t = np.array(t_list)
        T = np.array(T_list)
    
        order = np.argsort(t)
        t = t[order]
        T = T[order]
    
        plt.figure(figsize=(10,4))
        plt.plot(t, T, "o-", ms=4)
        plt.xlabel("time (fs) (0 fs = laser peak)")
        plt.ylabel("mean temperature (eV)")
        plt.title(f"Mean kinetic temperature vs time ({species})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()
    
        return t, T
        
    def plot_axial_field_evolution(
        self,      
        # selection
        file_indices=None,                 # "all", list, slice, or None
        t_range_fs: tuple[float, float] | None = None,
        every: int = 1,
    
        # geometry: lineout position
        z_index: int | None = None,        # default Nz//2
        y_nm: float | None = None,         # direct y position (nm)
        y_target_nm: float = 300.0,        # if y_nm is None -> search around this
        y_search_halfwidth_nm: float = 50.0,
        y_pick_mode: str = "auto_peak",    # "auto_peak" or "closest"
    
        # crop in x (ABSOLUTE nm in the simulation box)
        x_range_nm: tuple[float, float] | None = None,
    
        # fields in openPMD
        electron_density_field: str = "He_e_all_density",
        ion_charge_density_field: str = "He_i_all_density",
    
        # normalization
        charge_norm: str = "ion_global_max",   # "ion_global_max" or "ion_per_time_max"
        field_norm: str = "E0",                # "E0" (laser peak) or "none"
        I0_Wcm2: float = 4e14,                 # used if field_norm=="E0"
    
        # scaling (colormap normalization)
        charge_scale: str = "symlog",          # "linear", "log_abs", "symlog"
        field_scale: str = "linear",           # "linear", "symlog", "log_abs"
        symlog_linthresh_charge: float = 1e-3,
        symlog_linthresh_field: float = 1e-2,
    
        # plotting options
        cmap_charge: str = "RdBu_r",
        cmap_field: str = "RdBu_r",
        vlim_charge: float | None = None,      # symmetric +/- limit for linear/symlog (normalized units)
        vlim_field: float | None = None,       # symmetric +/- limit for linear/symlog (normalized units)
        interpolation: str = "none",           # "none", "nearest", "bilinear", "bicubic", ...
        show: bool = True,
        savepath: str | None = None,
    ):
        from matplotlib import colors as mcolors
        self._load_meta()
    
        # Select file indices
        if file_indices is None or file_indices == "all":
            indices = list(range(len(self.files)))
        elif isinstance(file_indices, slice):
            indices = list(range(len(self.files)))[file_indices]
        else:
            indices = list(file_indices)
        indices = indices[:: max(1, int(every))]
    
        # z slice
        z0 = int(self.Nz // 2) if z_index is None else int(z_index)
        z0 = max(0, min(self.Nz - 1, z0))
    
        # x coordinate (nm), cell-centered
        x_nm = (np.arange(self.Nx) + 0.5) * self.Dx_SI * 1e9
    
        # x cropping mask
        if x_range_nm is not None:
            xmin_nm, xmax_nm = float(x_range_nm[0]), float(x_range_nm[1])
            mask_x = (x_nm >= xmin_nm) & (x_nm <= xmax_nm)
            if not np.any(mask_x):
                raise ValueError(f"x_range_nm={x_range_nm} selects no cells. "
                                 f"Domain is roughly [{x_nm.min():.1f}, {x_nm.max():.1f}] nm.")
        else:
            mask_x = slice(None)
    
        x_nm_used = x_nm[mask_x] if not isinstance(mask_x, slice) else x_nm
    
        # helper: y index
        def y_nm_to_idx(yval_nm: float) -> int:
            yi = int((float(yval_nm) * 1e-9) / self.Dy_SI)
            return max(0, min(self.Ny - 1, yi))
    
        # choose y index
        if y_nm is not None:
            y0 = y_nm_to_idx(y_nm)
            y0_nm_used = (y0 + 0.5) * self.Dy_SI * 1e9
        else:
            y_center = y_nm_to_idx(y_target_nm)
            dy_idx = int((float(y_search_halfwidth_nm) * 1e-9) / self.Dy_SI)
            y_min = max(0, y_center - dy_idx)
            y_max = min(self.Ny - 1, y_center + dy_idx)
            cand = list(range(y_min, y_max + 1))
    
            if y_pick_mode == "closest":
                y0 = y_center
            elif y_pick_mode == "auto_peak":
                # pick y that maximizes total |ion| signal along x for the first usable frame
                best_y = y_center
                best_val = -np.inf
    
                fn0 = None
                step0 = None
                for idx in indices:
                    fn_try = self.files[idx]
                    st_try = extract_step_from_filename(fn_try)
                    with adios2.Stream(fn_try, "r") as f:
                        for _ in f.steps():
                            if f"/data/{st_try}/fields/{ion_charge_density_field}" in f.available_variables():
                                fn0, step0 = fn_try, st_try
                                break
                    if fn0 is not None:
                        break
    
                if fn0 is None:
                    y0 = y_center
                else:
                    with adios2.Stream(fn0, "r") as f:
                        for _ in f.steps():
                            unit_rho_i = f.read_attribute(f"/data/{step0}/fields/{ion_charge_density_field}/unitSI")
                            for yy in cand:
                                rho_i_line = (
                                    f.read(
                                        f"/data/{step0}/fields/{ion_charge_density_field}",
                                        start=[z0, yy, 0],
                                        count=[1, 1, self.Nx],
                                    )
                                    * unit_rho_i
                                )[0, 0, :]
                                rho_i_line = rho_i_line[mask_x]
                                val = float(np.sum(np.abs(rho_i_line)))
                                if val > best_val:
                                    best_val = val
                                    best_y = yy
                    y0 = best_y
            else:
                raise ValueError("y_pick_mode must be 'auto_peak' or 'closest'.")
    
            y0_nm_used = (y0 + 0.5) * self.Dy_SI * 1e9
    
        # Field normalization E0
        if field_norm == "E0":
            E0 = np.sqrt(2.0 * float(I0_Wcm2) * 1e4 / (sc.c * sc.epsilon_0))
        elif field_norm == "none":
            E0 = 1.0
        else:
            raise ValueError("field_norm must be 'E0' or 'none'.")
    
        # gather time series of lineouts
        t_list = []
        Ex_rows = []
        rho_rows = []
        ion_peak_list = []
    
        for idx in indices:
            fn = self.files[idx]
            step = extract_step_from_filename(fn)
            t_fs = self.time_fs_from_step(step)
    
            if t_range_fs is not None:
                tmin, tmax = t_range_fs
                if (t_fs < tmin) or (t_fs > tmax):
                    continue
    
            with adios2.Stream(fn, "r") as f:
                for _ in f.steps():
                    # Ex line
                    unit_E = f.read_attribute(f"/data/{step}/fields/E/x/unitSI")
                    Ex_line = (
                        f.read(
                            f"/data/{step}/fields/E/x",
                            start=[z0, y0, 0],
                            count=[1, 1, self.Nx],
                        )
                        * unit_E
                    )[0, 0, :]
                    Ex_line = (Ex_line[mask_x]) / (E0 + 1e-300)
    
                    # charge density fields line
                    unit_rho_e = f.read_attribute(f"/data/{step}/fields/{electron_density_field}/unitSI")
                    rho_e_line = (
                        f.read(
                            f"/data/{step}/fields/{electron_density_field}",
                            start=[z0, y0, 0],
                            count=[1, 1, self.Nx],
                        )
                        * unit_rho_e
                    )[0, 0, :]
                    rho_e_line = rho_e_line[mask_x]
    
                    if f"/data/{step}/fields/{ion_charge_density_field}" in f.available_variables():
                        unit_rho_i = f.read_attribute(f"/data/{step}/fields/{ion_charge_density_field}/unitSI")
                        rho_i_line = (
                            f.read(
                                f"/data/{step}/fields/{ion_charge_density_field}",
                                start=[z0, y0, 0],
                                count=[1, 1, self.Nx],
                            )
                            * unit_rho_i
                        )[0, 0, :]
                        rho_i_line = rho_i_line[mask_x]
                    else:
                        rho_i_line = np.zeros_like(rho_e_line)
    
            rho_net = rho_i_line + rho_e_line  # signed net charge density
    
            t_list.append(float(t_fs))
            Ex_rows.append(Ex_line)
            rho_rows.append(rho_net)
            ion_peak_list.append(float(np.max(np.abs(rho_i_line)) + 1e-300))
    
        if len(t_list) == 0:
            print("[plot_peltz_fig5_xt_vertical] No frames matched your selection.")
            return None
    
        # sort by time, build arrays as (Nt, Nx)
        t = np.array(t_list)
        order = np.argsort(t)
        t = t[order]
        Ex_xt = np.array(Ex_rows, dtype=float)[order, :]     # (Nt, Nx)
        rho_xt = np.array(rho_rows, dtype=float)[order, :]   # (Nt, Nx)
        ion_peaks = np.array(ion_peak_list, dtype=float)[order]
    
        # normalize charge
        if charge_norm == "ion_global_max":
            denom = float(np.nanmax(ion_peaks))
            rho_xt_n = rho_xt / (denom + 1e-300)
        elif charge_norm == "ion_per_time_max":
            rho_xt_n = rho_xt / (ion_peaks[:, None] + 1e-300)
        else:
            raise ValueError("charge_norm must be 'ion_global_max' or 'ion_per_time_max'.")
    
        # colormap norms
        def make_norm(scale: str, vlim: float | None, linthresh: float):
            if scale == "linear":
                return None if vlim is None else mcolors.Normalize(vmin=-vlim, vmax=+vlim)
            if scale == "symlog":
                return mcolors.SymLogNorm(linthresh=linthresh, vmin=-vlim, vmax=+vlim) if vlim is not None \
                    else mcolors.SymLogNorm(linthresh=linthresh)
            if scale == "log_abs":
                return mcolors.LogNorm()
            raise ValueError("scale must be 'linear', 'symlog', or 'log_abs'.")
    
        # what to plot
        if charge_scale == "log_abs":
            rho_plot = np.abs(rho_xt_n) + 1e-300
            norm_rho = mcolors.LogNorm()
            cmap_rho = "viridis"
        else:
            rho_plot = rho_xt_n
            norm_rho = make_norm(charge_scale, vlim_charge, symlog_linthresh_charge)
            cmap_rho = cmap_charge
    
        if field_scale == "log_abs":
            Ex_plot = np.abs(Ex_xt) + 1e-300
            norm_Ex = mcolors.LogNorm()
            cmap_Ex = "viridis"
        else:
            Ex_plot = Ex_xt
            norm_Ex = make_norm(field_scale, vlim_field, symlog_linthresh_field)
            cmap_Ex = cmap_field
    
        # extent: [xmin, xmax, tmin, tmax] because X is x_nm, Y is time
        extent = [x_nm_used[0], x_nm_used[-1], t.min(), t.max()]
    
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
        im0 = ax0.imshow(
            rho_plot,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap_rho,
            norm=norm_rho,
            interpolation=interpolation,
        )
        ax0.set_title(f"(a) net charge density / ion scale\n(z={z0}, y≈{y0_nm_used:.1f} nm)")
        ax0.set_xlabel("x (nm)")
        ax0.set_ylabel("time (fs) (0 fs = laser peak)")
        cb0 = fig.colorbar(im0, ax=ax0)
        cb0.set_label(r"$\rho_\mathrm{net}$ / ion scale")
    
        im1 = ax1.imshow(
            Ex_plot,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap_Ex,
            norm=norm_Ex,
            interpolation=interpolation,
        )
        ax1.set_title("(b) $E_x/E_0$" if field_norm == "E0" else "(b) $E_x$ (SI)")
        ax1.set_xlabel("x (nm)")
        cb1 = fig.colorbar(im1, ax=ax1)
        cb1.set_label(r"$E_x/E_0$" if field_norm == "E0" else r"$E_x$ (SI)")
    
        plt.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, dpi=200)
        if show:
            plt.show()
        plt.close(fig)
    
        meta = dict(z_index=z0, y_index=y0, y_nm_used=y0_nm_used, x_range_nm=x_range_nm)
        return t, x_nm_used, rho_xt_n, Ex_xt, meta


    def plot_energy_partition_vs_time(
        self,
        electron_species="He_e",
        ion_species="He_i",
        show_laser=False,
        tau_fwhm_fs=40,
        I0_Wcm2=4e14,
        lambda_um=0.8,
    ):
        import scipy.constants as sc
    
        self._load_meta()
    
        t_list = []
        Ee_list = []
        Ei_list = []
    
        me = self._mass_kg(electron_species)
        mi = self._mass_kg(ion_species)
    
        for fn in self.files:
            step = extract_step_from_filename(fn)
            t_fs = self.time_fs_from_step(step)
    
            # electrons
            px, py, pz, we = self._read_momentum_and_weight(fn, step, electron_species)
            Ee = 0
            if px is not None:
                Ee = np.sum(we * (px**2 + py**2 + pz**2) / (2*me))
    
            # ions
            px, py, pz, wi = self._read_momentum_and_weight(fn, step, ion_species)
            Ei = 0
            if px is not None:
                Ei = np.sum(wi * (px**2 + py**2 + pz**2) / (2*mi))
    
            t_list.append(t_fs)
            Ee_list.append(Ee / sc.e)  # to eV
            Ei_list.append(Ei / sc.e)
    
        t = np.array(t_list)
        Ee = np.array(Ee_list)
        Ei = np.array(Ei_list)
        Etot = Ee + Ei
    
        order = np.argsort(t)
        t, Ee, Ei, Etot = t[order], Ee[order], Ei[order], Etot[order]
    
        fig, ax = plt.subplots(figsize=(10,4))
    
        ax.plot(t, Ee, label="electrons")
        ax.plot(t, Ei, label="ions")
        ax.plot(t, Etot, label="total", linestyle="--")
    
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("kinetic energy (eV)")
        ax.legend()
        ax.grid(True)
    
        if show_laser:
            I_t = I0_Wcm2 * np.exp(-4*np.log(2)*(t/tau_fwhm_fs)**2)
            Up_t = 9.33e-14 * I_t * lambda_um**2
            ax2 = ax.twinx()
            ax2.plot(t, Up_t, color="gray", alpha=0.4)
            ax2.set_ylabel("Up (eV)")
    
        plt.tight_layout()
        plt.show()


    def plot_core_electron_density_vs_time(
        self,
        *,
        electron_density_field: str = "He_e_all_density",
        mode: str = "max_slice_y",      # "max_slice_y" (fast) or "max3d" (best, heavier)
        y_nm: float = 300.0,            # used for max_slice_y
        assume_charge_density: bool = True,  # if the field is charge density (C/m^3), convert to number density
        lambda_um: float = 0.8,
    ):
        """
        Plot core electron density vs time, normalized to critical density ncrit.
    
        'core density' here is taken as:
          - mode="max_slice_y": max of density in an x–z slice at fixed y (fast; matches your page-1 slices)
          - mode="max3d":       global max over full 3D field (more correct but can be heavier)
    
        If assume_charge_density=True, converts rho (C/m^3) -> ne (1/m^3) via ne = |rho|/e.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.constants as sc
    
        self._load_meta()
    
        # ncrit for given wavelength
        lam = float(lambda_um) * 1e-6
        omega = 2.0 * np.pi * sc.c / lam
        ncrit = sc.epsilon_0 * sc.m_e * omega**2 / (sc.e**2)  # 1/m^3
    
        t_list = []
        ne_core_list = []
    
        for fn in self.files:
            step = extract_step_from_filename(fn)
            t_fs = self.time_fs_from_step(step)
    
            with adios2.Stream(fn, "r") as f:
                for _ in f.steps():
                    base = f"/data/{step}/fields/{electron_density_field}"
                    if base not in f.available_variables():
                        ne_core = np.nan
                        break
    
                    unit = 1.0
                    try:
                        unit = f.read_attribute(f"{base}/unitSI")
                    except Exception:
                        pass
    
                    if mode == "max3d":
                        rho = f.read(base) * unit  # full 3D array
                        rho = np.asarray(rho)
                        ne_core = np.nanmax(np.abs(rho))
                    elif mode == "max_slice_y":
                        y_idx = int((float(y_nm) * 1e-9) / self.Dy_SI)
                        y_idx = max(0, min(self.Ny - 1, y_idx))
                        rho = f.read(base, start=[0, y_idx, 0], count=[self.Nz, 1, self.Nx]) * unit
                        rho = np.asarray(rho)[:, 0, :]
                        ne_core = np.nanmax(np.abs(rho))
                    else:
                        raise ValueError("mode must be 'max_slice_y' or 'max3d'")
    
            # convert to number density if needed
            if assume_charge_density:
                ne_core = ne_core / sc.e  # (C/m^3)/e -> 1/m^3
    
            t_list.append(t_fs)
            ne_core_list.append(ne_core)
    
        t = np.array(t_list, dtype=float)
        ne_core = np.array(ne_core_list, dtype=float)
        order = np.argsort(t)
        t, ne_core = t[order], ne_core[order]
    
        ratio = ne_core / ncrit
    
        plt.figure(figsize=(9, 4))
        plt.plot(t, ratio, "o-", ms=4)
        plt.axhline(1.0, lw=1.2, linestyle="--")
        plt.xlabel("time - t_peak (fs)")
        plt.ylabel(r"$n_e^\mathrm{core}/n_\mathrm{crit}$")
        plt.title(f"Core electron density vs time (field='{electron_density_field}', mode={mode})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()
    
        return t, ne_core, ncrit











        
