"""Utilities for reduced HDF5 nanoplasma analysis archives.

This module contains small reusable helpers extracted from the H5 plotting notebooks.
Heavy figure-layout cells stay in notebooks until explicitly refactored.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def h5_inventory(path: str | Path) -> dict[str, list[str]]:
    """Return top-level HDF5 groups and immediate children."""
    with h5py.File(path, "r") as f:
        return {k: list(f[k].keys()) if isinstance(f[k], h5py.Group) else [] for k in f.keys()}


def h5_read(path: str | Path, dataset: str) -> np.ndarray:
    """Read a dataset from a reduced HDF5 archive into a NumPy array."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[dataset])


def h5_has(path: str | Path, dataset: str) -> bool:
    """Return True if a dataset/group exists in the archive."""
    with h5py.File(path, "r") as f:
        return dataset in f


def load_axes(path: str | Path) -> dict[str, np.ndarray]:
    """Load all datasets under /axes."""
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f["axes"][k]) for k in f["axes"].keys()}


def load_timeseries(path: str | Path) -> dict[str, np.ndarray]:
    """Load all datasets under /timeseries."""
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f["timeseries"][k]) for k in f["timeseries"].keys()}
