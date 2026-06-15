"""
Backend abstraction for raw MS data, modelled on the ``MsBackend`` class of
the R *Spectra* package.

A backend is the *storage engine* behind a :class:`PteroSpectra` object. The
user-facing object always exposes the same interface (``mz``, ``intensity``,
``peaks_data``, ``spectra_data``, filters); the backend decides whether the
data lives in RAM, in an mzML file read on demand, or in a chunked Zarr /
columnar Parquet store on disk. New formats are added by subclassing
:class:`MsBackend` and implementing a handful of methods -- no user code
changes.

Peak storage convention (CSR / ragged layout)
----------------------------------------------
Spectra have variable numbers of peaks, so peaks are stored as flat arrays plus
an ``offsets`` (index-pointer) array, exactly like a compressed-sparse-row
matrix: the peaks of spectrum ``i`` are ``mz[offsets[i]:offsets[i+1]]`` (and
likewise ``intensity`` and, for 4D data, ``ion_mobility``). This is the layout
alphatims uses for Bruker timsTOF data and is what makes O(1) random access
into any spectrum possible without loading the whole run.
"""
from __future__ import annotations

import abc
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Core spectrum-level variables (mirrors Spectra's "core spectra variables").
# These are guaranteed to be returned by spectra_data (NA-filled if unknown).
CORE_SPECTRA_VARIABLES: dict[str, object] = {
    "ms_level": "Int64",
    "rtime": "float64",
    "acquisition_num": "Int64",
    "scan_index": "Int64",
    "data_origin": "string",
    "data_storage": "string",
    "centroided": "boolean",
    "polarity": "Int64",
    "precursor_mz": "float64",
    "precursor_charge": "Int64",
    "precursor_intensity": "float64",
    "collision_energy": "float64",
    "ion_mobility": "float64",  # spectrum-level (4D); per-peak IM is a peaks var
}

DEFAULT_PEAKS_VARIABLES = ("mz", "intensity")
PEAKS_VARIABLES_4D = ("mz", "intensity", "ion_mobility")


class MsBackend(abc.ABC):
    """Abstract storage backend for a collection of mass spectra.

    Subclasses must implement :meth:`__len__`, :meth:`spectra_data`,
    :meth:`peaks_data`, and :meth:`subset`. Everything else has a default.
    """

    #: whether peak (m/z, intensity) values can be modified in place
    read_only: bool = False
    #: whether the backend carries per-peak ion mobility (4D data)
    has_4d: bool = False

    # ── required interface ───────────────────────────────────────────────
    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of spectra."""

    @abc.abstractmethod
    def spectra_data(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Return spectrum-level metadata as a DataFrame (one row per spectrum).

        Always includes the core variables in :data:`CORE_SPECTRA_VARIABLES`
        (filled with NA when unknown) so downstream code can rely on them.
        """

    @abc.abstractmethod
    def peaks_data(
        self,
        indices: Optional[Sequence[int]] = None,
        columns: Sequence[str] = DEFAULT_PEAKS_VARIABLES,
    ) -> list[np.ndarray]:
        """Return a list of 2D arrays, one per requested spectrum.

        Each array has shape ``(n_peaks_i, len(columns))`` with columns in the
        requested order. ``indices=None`` means all spectra, in storage order.
        """

    @abc.abstractmethod
    def subset(self, indices: Sequence[int]) -> "MsBackend":
        """Return a backend restricted/reordered to ``indices``.

        Implementations should make this cheap (a view over metadata + offsets)
        rather than copying peak data where possible.
        """

    # ── conveniences with defaults ───────────────────────────────────────
    @property
    def peaks_variables(self) -> tuple[str, ...]:
        return PEAKS_VARIABLES_4D if self.has_4d else DEFAULT_PEAKS_VARIABLES

    def data_storage(self) -> list[str]:
        sd = self.spectra_data(["data_storage"])
        return list(sd["data_storage"].values)

    def mz(self, indices: Optional[Sequence[int]] = None) -> list[np.ndarray]:
        return [p[:, 0] for p in self.peaks_data(indices, ("mz",))]

    def intensity(self, indices: Optional[Sequence[int]] = None) -> list[np.ndarray]:
        return [p[:, 0] for p in self.peaks_data(indices, ("intensity",))]

    def ion_mobility(self, indices: Optional[Sequence[int]] = None) -> list[np.ndarray]:
        if not self.has_4d:
            raise ValueError("This backend has no per-peak ion mobility (not 4D).")
        return [p[:, 0] for p in self.peaks_data(indices, ("ion_mobility",))]

    # ── helpers for subclasses ───────────────────────────────────────────
    @staticmethod
    def _normalize_indices(indices, n: int) -> np.ndarray:
        if indices is None:
            return np.arange(n, dtype=np.int64)
        return np.asarray(indices, dtype=np.int64)

    @classmethod
    def _ensure_core_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add any missing core spectra variables (NA-filled) with right dtype."""
        df = df.copy()
        for col, dtype in CORE_SPECTRA_VARIABLES.items():
            if col not in df.columns:
                fill = np.nan if dtype == "float64" else pd.NA
                df[col] = pd.Series([fill] * len(df), dtype=dtype)
        return df

    def __repr__(self) -> str:
        kind = "4D" if self.has_4d else "3D"
        return f"{type(self).__name__}: {len(self)} spectra ({kind}, read_only={self.read_only})"
