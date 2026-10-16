"""In-memory backend (analog of Spectra's ``MsBackendMemory``/``MsBackendDataFrame``).

Holds everything in RAM: spectrum metadata as a DataFrame and peaks as flat
CSR arrays. Fast, but its memory footprint scales with the total peak count, so
it is intended for small/processed data -- large runs should use the Zarr or
Parquet backend.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._base import MsBackend, DEFAULT_PEAKS_VARIABLES


class MsBackendMemory(MsBackend):
    read_only = False

    def __init__(self, meta, mz, intensity, offsets, ion_mobility=None,
                 data_storage="<memory>"):
        self._meta = meta.reset_index(drop=True)
        self._mz = np.asarray(mz, dtype=np.float64)
        self._intensity = np.asarray(intensity, dtype=np.float32)
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._im = None if ion_mobility is None else np.asarray(ion_mobility, dtype=np.float32)
        self.has_4d = self._im is not None
        if "data_storage" not in self._meta.columns:
            self._meta["data_storage"] = data_storage

    # ── factories ────────────────────────────────────────────────────────
    @classmethod
    def from_spectra_list(cls, meta: pd.DataFrame, peaks: list[dict]):
        """Build from per-spectrum dicts: ``{'mz':..., 'intensity':..., 'ion_mobility':...}``."""
        has_4d = any("ion_mobility" in p and p["ion_mobility"] is not None for p in peaks)
        mz_parts, int_parts, im_parts, offs = [], [], [], [0]
        for p in peaks:
            m = np.asarray(p["mz"], dtype=np.float64)
            i = np.asarray(p["intensity"], dtype=np.float32)
            mz_parts.append(m)
            int_parts.append(i)
            if has_4d:
                im = p.get("ion_mobility")
                im_parts.append(np.full(len(m), np.nan, np.float32) if im is None
                                else np.asarray(im, dtype=np.float32))
            offs.append(offs[-1] + len(m))
        cat = lambda parts: (np.concatenate(parts) if parts else np.array([], np.float64))
        return cls(
            meta=meta,
            mz=cat(mz_parts),
            intensity=cat(int_parts),
            offsets=np.asarray(offs, np.int64),
            ion_mobility=(cat(im_parts) if has_4d else None),
        )

    # ── required interface ───────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._meta)
    @property
    def n_peaks(self) -> int:
        """Return the total number of peaks stored in this backend."""
        return int(self._offsets[-1])
    def spectra_data(self, columns=None) -> pd.DataFrame:
        df = self._ensure_core_columns(self._meta)
        return df if columns is None else df[list(columns)]

    def peaks_data(self, indices=None, columns=DEFAULT_PEAKS_VARIABLES):
        idx = self._normalize_indices(indices, len(self))
        cols = list(columns)
        out = []
        for i in idx:
            s, e = self._offsets[i], self._offsets[i + 1]
            arrs = []
            for c in cols:
                if c == "mz":
                    arrs.append(self._mz[s:e])
                elif c == "intensity":
                    arrs.append(self._intensity[s:e].astype(np.float64))
                elif c == "ion_mobility":
                    if self._im is None:
                        raise KeyError("ion_mobility not available (3D data)")
                    arrs.append(self._im[s:e].astype(np.float64))
                else:
                    raise KeyError(f"unknown peaks variable {c!r}")
            out.append(np.column_stack(arrs) if arrs else np.empty((0, 0)))
        return out

    def subset(self, indices: Sequence[int]) -> "MsBackendMemory":
        idx = np.asarray(indices, dtype=np.int64)
        # rebuild compact CSR for the chosen spectra (keeps it simple + correct)
        mz_parts, int_parts, im_parts, offs = [], [], [], [0]
        for i in idx:
            s, e = self._offsets[i], self._offsets[i + 1]
            mz_parts.append(self._mz[s:e])
            int_parts.append(self._intensity[s:e])
            if self._im is not None:
                im_parts.append(self._im[s:e])
            offs.append(offs[-1] + (e - s))
        cat = lambda parts, dt: (np.concatenate(parts) if parts else np.array([], dt))
        return MsBackendMemory(
            meta=self._meta.iloc[idx].reset_index(drop=True),
            mz=cat(mz_parts, np.float64),
            intensity=cat(int_parts, np.float32),
            offsets=np.asarray(offs, np.int64),
            ion_mobility=(cat(im_parts, np.float32) if self._im is not None else None),
        )
