"""
Zarr backend for raw MS data (3D and 4D) -- the chunked, cloud-native store.

A lazy, chunked on-disk backend for raw spectra conforming to PTERO's common backend interface. The design supports runs larger than available RAM.

On-disk layout (a ``.zarr`` directory / store)
-----------------------------------------------
::

    experiment.zarr/
      .zattrs                 global metadata (n_spectra, has_4d, peaks_variables)
      peaks/
        mz          (1D chunked float64)   flat m/z for ALL spectra concatenated
        intensity   (1D chunked float32)
        ion_mobility(1D chunked float32)   only if 4D
      offsets       (1D int64, length n+1) CSR index-pointer into the flat arrays
      spectra.parquet                      spectrum-level metadata (columnar)

Peaks are stored as flat chunked arrays + a CSR ``offsets`` array (NOT one file
per spectrum -- that would create millions of tiny objects and is pathological
on real / cloud filesystems). Reading spectrum ``i`` slices
``mz[offsets[i]:offsets[i+1]]``, and Zarr only fetches the chunks that overlap
that slice. Metadata goes to Parquet because it is tabular and benefits from
columnar compression and projection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._base import MsBackend, DEFAULT_PEAKS_VARIABLES
from .._schema import (
    ATTR_ACQUISITION_MODE, ATTR_FORMAT_VERSION, ATTR_HAS_4D, ATTR_N_SPECTRA,
    ATTR_PEAKS_VARIABLES, OFFSETS_ARRAY, PEAKS_GROUP, PEAK_INTENSITY,
    PEAK_ION_MOBILITY, PEAK_MZ, PTERO_FORMAT_VERSION, SPECTRA_METADATA_FILE,
    store_attrs,
)


def _require_zarr():
    try:
        import zarr  # noqa
    except Exception as e:  # pragma: no cover
        raise ImportError("MsBackendZarr requires `zarr`. Install with pip install ptero[zarr].") from e


class MsBackendZarr(MsBackend):
    """Lazy, chunked Zarr-backed store for raw spectra."""

    read_only = False  # peaks are immutable once written; metadata editable in memory

    def __init__(self, path: str, _idx: Optional[np.ndarray] = None):
        _require_zarr()
        import zarr

        self.path = str(path)
        self._root = zarr.open_group(self.path, mode="r")
        attrs = dict(self._root.attrs)
        self.format_version = attrs.get(ATTR_FORMAT_VERSION)
        self.has_4d = bool(attrs.get(ATTR_HAS_4D, False))
        #: "dda" | "dia" | "unknown" -- the acquisition mode recorded at write time
        self.acquisition_mode = str(attrs.get(ATTR_ACQUISITION_MODE, "unknown"))
        self._n_total = int(attrs.get(ATTR_N_SPECTRA, 0))
        self._offsets = self._root[OFFSETS_ARRAY][:]  # small (n+1,), cheap to hold
        # view index (subsetting without rewriting the store)
        self._idx = np.arange(self._n_total, dtype=np.int64) if _idx is None else np.asarray(_idx, np.int64)
        # metadata lives in a Parquet sidecar
        self._meta_path = Path(self.path) / SPECTRA_METADATA_FILE

    # ── writing ──────────────────────────────────────────────────────────
    @classmethod
    def write(cls, path, meta: pd.DataFrame, mz, intensity, offsets,
              ion_mobility=None, chunk_size: int = 1_000_000,
              compressor="zstd", acquisition_mode: str = "unknown",
               *, peak_ion_mobility_type=None,
               peak_ion_mobility_unit=None,
               spectrum_ion_mobility_type=None,
               spectrum_ion_mobility_unit=None) -> "MsBackendZarr":
        """Create a Zarr store from CSR peak arrays + metadata, return the backend."""
        # MZZARR_MOBILITY_WRITER_V1
        from .._schema import mobility_semantic_attrs
        _semantic_attrs = mobility_semantic_attrs(
            has_peak_mobility=ion_mobility is not None,
            has_spectrum_mobility=(
                "ion_mobility" in meta.columns
                and bool(meta["ion_mobility"].notna().any())
            ),
            strict=True,
            peak_ion_mobility_type=peak_ion_mobility_type,
            peak_ion_mobility_unit=peak_ion_mobility_unit,
            spectrum_ion_mobility_type=spectrum_ion_mobility_type,
            spectrum_ion_mobility_unit=spectrum_ion_mobility_unit,
        )
        # MZZARR_MS_LEVEL_WRITER_V1
        import pyarrow as _mzzarr_pa
        _mzzarr_meta = meta
        if "ms_level" not in _mzzarr_meta.columns:
            raise ValueError("mzZarr v1.0 requires ms_level column")
        if _mzzarr_meta["ms_level"].isna().any():
            raise ValueError("mzZarr v1.0 ms_level contains null values")
        _mzzarr_type = _mzzarr_pa.array(_mzzarr_meta["ms_level"]).type
        if not _mzzarr_pa.types.is_int64(_mzzarr_type):
            raise ValueError("mzZarr v1.0 ms_level must be int64")
        if (_mzzarr_meta["ms_level"] <= 0).any():
            raise ValueError("mzZarr v1.0 ms_level must contain positive integers")
        _require_zarr()
        import zarr

        path = str(path)
        mz = np.asarray(mz, np.float64)
        intensity = np.asarray(intensity, np.float32)
        offsets = np.asarray(offsets, np.int64)
        has_4d = ion_mobility is not None

        if offsets.ndim != 1 or len(offsets) == 0:
            raise ValueError("offsets must be a non-empty 1D array")
        if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("offsets must start at 0 and be monotonically non-decreasing")
        if int(offsets[-1]) != len(mz) or len(intensity) != len(mz):
            raise ValueError("offsets[-1], len(mz), and len(intensity) must agree")
        if has_4d and len(np.asarray(ion_mobility)) != len(mz):
            raise ValueError("ion_mobility must have one value per peak")
        if len(meta) != len(offsets) - 1:
            raise ValueError("metadata must contain one row per spectrum")

        from ._zarr_qc import ZarrQCAccumulator

        qc = ZarrQCAccumulator()
        qc.add_batch(meta, mz, intensity, offsets, ion_mobility)

        root = zarr.open_group(path, mode="w")
        root.attrs.update(store_attrs(
            n_spectra=len(offsets) - 1, has_4d=has_4d,
            acquisition_mode=acquisition_mode,
        ))
        root.attrs.update(_semantic_attrs)
        peaks = root.create_group(PEAKS_GROUP)

        def _write_array(group, name, data):
            n = max(1, len(data))
            arr = group.create_array(
                name, shape=(len(data),), chunks=(min(chunk_size, n),), dtype=data.dtype
            )
            if len(data):
                arr[:] = data

        _write_array(peaks, PEAK_MZ, mz)
        _write_array(peaks, PEAK_INTENSITY, intensity)
        if has_4d:
            _write_array(peaks, PEAK_ION_MOBILITY, np.asarray(ion_mobility, np.float32))
        off_arr = root.create_array(OFFSETS_ARRAY, shape=(len(offsets),),
                                    chunks=(len(offsets),), dtype="i8")
        off_arr[:] = offsets

        root.attrs["ptero_qc_v1"] = qc.result()

        # metadata -> parquet sidecar
        meta = meta.reset_index(drop=True).copy()
        if "data_storage" not in meta.columns:
            meta["data_storage"] = path
        meta.to_parquet(Path(path) / SPECTRA_METADATA_FILE, index=False)
        return cls(path)

    # ── required interface ───────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._idx)

    def spectra_data(self, columns=None) -> pd.DataFrame:
        cols = None if columns is None else list(columns)
        # Parquet projection: read only requested columns when possible
        df = pd.read_parquet(self._meta_path, columns=cols) if cols else pd.read_parquet(self._meta_path)
        df = df.iloc[self._idx].reset_index(drop=True)
        return self._ensure_core_columns(df) if columns is None else df

    def peaks_data(self, indices=None, columns=DEFAULT_PEAKS_VARIABLES):
        view = self._normalize_indices(indices, len(self))
        backend_idx = self._idx[view]  # map view positions -> storage rows
        cols = list(columns)
        mz_a = self._root["peaks"]["mz"]
        int_a = self._root["peaks"]["intensity"]
        im_a = self._root["peaks"]["ion_mobility"] if self.has_4d and "ion_mobility" in cols else None
        out = []
        for bi in backend_idx:
            s, e = int(self._offsets[bi]), int(self._offsets[bi + 1])
            arrs = []
            for c in cols:
                if c == "mz":
                    arrs.append(mz_a[s:e].astype(np.float64))
                elif c == "intensity":
                    arrs.append(int_a[s:e].astype(np.float64))
                elif c == "ion_mobility":
                    if not self.has_4d:
                        raise KeyError("ion_mobility not available (3D data)")
                    arrs.append(im_a[s:e].astype(np.float64))
                else:
                    raise KeyError(f"unknown peaks variable {c!r}")
            out.append(np.column_stack(arrs) if arrs else np.empty((0, 0)))
        return out

    def subset(self, indices: Sequence[int]) -> "MsBackendZarr":
        idx = self._idx[np.asarray(indices, dtype=np.int64)]
        return MsBackendZarr(self.path, _idx=idx)

    def iter_peaks_blocked(self, indices=None, columns=DEFAULT_PEAKS_VARIABLES,
                           block_size: int = 4_000_000):
        """Yield ``(view_position, peaks)`` for the requested spectra, reading the
        flat ``peaks/*`` arrays in large **contiguous blocks** rather than one slice
        per spectrum.

        Reading spectrum-by-spectrum re-fetches and re-decompresses the same Zarr
        chunk once per spectrum that overlaps it -- on a run with tens of thousands
        of spectra that is the dominant cost. Here we walk the requested spectra in
        storage order, read up to ``block_size`` peaks at a time in a single slice
        (so each chunk is decompressed once), and slice each spectrum out of the
        in-memory block. Memory is bounded to one block; results are yielded in
        storage order with their original view position so callers can place them.
        """
        view = self._normalize_indices(indices, len(self))
        rows = self._idx[view]                       # storage rows
        n = len(rows)
        if n == 0:
            return
        cols = list(columns)
        mz_a = self._root["peaks"]["mz"]
        int_a = self._root["peaks"]["intensity"]
        im_a = self._root["peaks"]["ion_mobility"] if (self.has_4d and "ion_mobility" in cols) else None
        offs = self._offsets
        order = np.argsort(rows, kind="stable")      # storage order -> contiguous reads
        sr = rows[order]
        i = 0
        while i < n:
            bstart = int(offs[sr[i]])
            j = i
            # extend the block while it stays within the peak budget
            while j + 1 < n and (int(offs[sr[j + 1] + 1]) - bstart) <= block_size:
                j += 1
            bend = int(offs[sr[j] + 1])
            mzb = mz_a[bstart:bend]
            intb = int_a[bstart:bend]
            imb = im_a[bstart:bend] if im_a is not None else None
            for k in range(i, j + 1):
                r = sr[k]; s = int(offs[r]) - bstart; e = int(offs[r + 1]) - bstart
                # Construct the final float64 matrix directly, avoiding
                # intermediate per-column copies. Preserve requested column
                # order and the original reader's float64 output.
                P = np.empty((e - s, len(cols)) if cols else (0, 0), dtype=np.float64)
                for ci, c in enumerate(cols):
                    if c == "mz":
                        P[:, ci] = mzb[s:e]
                    elif c == "intensity":
                        P[:, ci] = intb[s:e]
                    elif c == "ion_mobility":
                        if imb is None:
                            raise KeyError("ion_mobility not available (3D data)")
                        P[:, ci] = imb[s:e]
                    else:
                        raise KeyError(f"unknown peaks variable {c!r}")
                yield int(order[k]), P
            i = j + 1


class ZarrStreamWriter:
    """Write a Zarr store incrementally, one batch of spectra at a time.

    This is the memory-safe path for large runs. Only the *current* batch's
    peaks are ever in RAM; the flat ``peaks/*`` arrays are grown on disk with
    ``resize`` + slice-assign, while the small ``offsets`` index and the
    spectrum metadata accumulate and are written once in :meth:`finalize`.

    Produces a store byte-for-byte compatible with :meth:`MsBackendZarr.write`,
    so anything that opens a Zarr store reads a streamed one identically.

    Usage::

        w = ZarrStreamWriter(path, has_4d=True)
        for mem in batches:          # each `mem` is an MsBackendMemory
            w.append_backend(mem)
        backend = w.finalize()
    """

    def __init__(self, path, has_4d: bool, chunk_size: int = 250_000,
                 acquisition_mode: str = "unknown",
                 *, peak_ion_mobility_type=None,
                 peak_ion_mobility_unit=None,
                 spectrum_ion_mobility_type=None,
                 spectrum_ion_mobility_unit=None):
        from .._schema import mobility_semantic_attrs
        self._semantic_inputs = dict(
            peak_ion_mobility_type=peak_ion_mobility_type,
            peak_ion_mobility_unit=peak_ion_mobility_unit,
            spectrum_ion_mobility_type=spectrum_ion_mobility_type,
            spectrum_ion_mobility_unit=spectrum_ion_mobility_unit,
        )
        # Validate the peak-level declaration before creating a store.
        mobility_semantic_attrs(
            has_peak_mobility=bool(has_4d),
            has_spectrum_mobility=False,
            strict=True,
            peak_ion_mobility_type=peak_ion_mobility_type,
            peak_ion_mobility_unit=peak_ion_mobility_unit,
        )
        _require_zarr()
        import zarr

        self.path = str(path)
        self.has_4d = bool(has_4d)
        self.acquisition_mode = str(acquisition_mode)
        self._root = zarr.open_group(self.path, mode="w-")
        peaks = self._root.create_group(PEAKS_GROUP)
        self._mz = peaks.create_array(PEAK_MZ, shape=(0,), chunks=(chunk_size,), dtype="f8")
        self._int = peaks.create_array(PEAK_INTENSITY, shape=(0,), chunks=(chunk_size,), dtype="f4")
        self._im = (peaks.create_array(PEAK_ION_MOBILITY, shape=(0,), chunks=(chunk_size,), dtype="f4")
                    if self.has_4d else None)
        self._offsets = [0]        # CSR index-pointer (small: one int per spectrum)
        self._meta_parts = []      # list of metadata DataFrames (small)
        from ._zarr_qc import ZarrQCAccumulator
        self._qc = ZarrQCAccumulator()

    @staticmethod
    def _grow(arr, data):
        """Append a 1D array to a resizable Zarr array via resize + slice-assign."""
        old = arr.shape[0]
        n = len(data)
        if n:
            arr.resize((old + n,))
            arr[old:old + n] = data

    def append_backend(self, mem) -> None:
        """Append one in-memory batch (an ``MsBackendMemory``) to the store."""
        # MZZARR_MS_LEVEL_STREAM_V1
        import pyarrow as _mzzarr_pa
        _mzzarr_meta = mem._meta
        if "ms_level" not in _mzzarr_meta.columns:
            raise ValueError("mzZarr v1.0 requires ms_level column")
        if _mzzarr_meta["ms_level"].isna().any():
            raise ValueError("mzZarr v1.0 ms_level contains null values")
        _mzzarr_type = _mzzarr_pa.array(_mzzarr_meta["ms_level"]).type
        if not _mzzarr_pa.types.is_int64(_mzzarr_type):
            raise ValueError("mzZarr v1.0 ms_level must be int64")
        if (_mzzarr_meta["ms_level"] <= 0).any():
            raise ValueError("mzZarr v1.0 ms_level must contain positive integers")
        n_new = int(mem._offsets[-1])          # total peaks in this batch
        # MZZARR_STREAM_COMPLETE_MOBILITY_V1
        if self.has_4d and n_new > 0 and mem._im is None:
            raise ValueError(
                'Nonempty 4D batch lacks per-peak ion mobility; '
                'refusing to fabricate NaN mobility values'
            )
        if n_new:
            self._grow(self._mz, np.asarray(mem._mz, np.float64))
            self._grow(self._int, np.asarray(mem._intensity, np.float32))
            if self.has_4d:
                im = mem._im if mem._im is not None else np.full(n_new, np.nan, np.float32)
                self._grow(self._im, np.asarray(im, np.float32))
        self._qc.add_batch(
            mem._meta,
            np.asarray(mem._mz, np.float64),
            np.asarray(mem._intensity, np.float32),
            mem._offsets,
            (
                np.asarray(mem._im, np.float32)
                if mem._im is not None
                else np.full(n_new, np.nan, np.float32)
            ) if self.has_4d else None,
        )

        # extend the CSR offsets, shifting this batch's local ends by the running total
        base = self._offsets[-1]
        self._offsets.extend((base + np.asarray(mem._offsets[1:], np.int64)).tolist())
        self._meta_parts.append(mem._meta.reset_index(drop=True))

    def finalize(self) -> "MsBackendZarr":
        """Write offsets, attributes, and the metadata sidecar; return the backend."""
        from .._schema import mobility_semantic_attrs
        _has_spectrum_im = any(
            "ion_mobility" in part.columns
            and bool(part["ion_mobility"].notna().any())
            for part in self._meta_parts
        )
        _semantic_attrs = mobility_semantic_attrs(
            has_peak_mobility=self.has_4d,
            has_spectrum_mobility=_has_spectrum_im,
            strict=True,
            **self._semantic_inputs,
        )
        offsets = np.asarray(self._offsets, np.int64)
        off_arr = self._root.create_array(OFFSETS_ARRAY, shape=(len(offsets),),
                                          chunks=(len(offsets),), dtype="i8")
        off_arr[:] = offsets
        self._root.attrs.update(store_attrs(
            n_spectra=len(offsets) - 1, has_4d=self.has_4d,
            acquisition_mode=self.acquisition_mode,
        ))
        self._root.attrs.update(_semantic_attrs)
        self._root.attrs["ptero_qc_v1"] = self._qc.result()

        meta = (pd.concat(self._meta_parts, ignore_index=True)
                if self._meta_parts else pd.DataFrame())
        meta["scan_index"] = np.arange(len(meta), dtype=np.int64)   # global, unique across batches
        if "data_storage" not in meta.columns:
            meta["data_storage"] = self.path
        meta.to_parquet(Path(self.path) / SPECTRA_METADATA_FILE, index=False)
        return MsBackendZarr(self.path)


# Backward-compatible private alias. Internal code should use ZarrStreamWriter.
_ZarrStreamWriter = ZarrStreamWriter
