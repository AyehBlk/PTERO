"""
Parquet backend for raw MS data -- columnar storage for analytical queries.

Peaks are stored as one long, columnar table (``spectrum, mz, intensity[,
ion_mobility]``) sorted by spectrum, with row-group statistics. This is ideal
for *analytical* access patterns: a metadata filter (e.g. by retention time)
touches only the small ``spectra.parquet`` file, and an m/z-range filter can use
Parquet row-group min/max statistics to skip row groups (predicate pushdown)
rather than scanning every peak.

Layout::

    experiment_pq/
      peaks.parquet     long table: spectrum,int  mz,float64  intensity,float32  [ion_mobility,float32]
      spectra.parquet   spectrum-level metadata
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._base import MsBackend, DEFAULT_PEAKS_VARIABLES


class MsBackendParquet(MsBackend):
    read_only = False

    def __init__(self, path: str, _idx: Optional[np.ndarray] = None):
        self.path = str(path)
        self._peaks_path = Path(self.path) / "peaks.parquet"
        self._meta_path = Path(self.path) / "spectra.parquet"
        meta = pd.read_parquet(self._meta_path, columns=["scan_index"]) \
            if self._meta_path.exists() else pd.DataFrame()
        self._n_total = len(meta)
        self.has_4d = self._peaks_has_im()
        self._idx = np.arange(self._n_total, dtype=np.int64) if _idx is None else np.asarray(_idx, np.int64)

    def _peaks_has_im(self) -> bool:
        import pyarrow.parquet as pq
        schema = pq.read_schema(self._peaks_path)
        return "ion_mobility" in schema.names

    @classmethod
    def write(cls, path, meta: pd.DataFrame, mz, intensity, offsets,
              ion_mobility=None) -> "MsBackendParquet":
        path = str(path)
        Path(path).mkdir(parents=True, exist_ok=True)
        offsets = np.asarray(offsets, np.int64)
        n = len(offsets) - 1
        spectrum_col = np.repeat(np.arange(n, dtype=np.int64), np.diff(offsets))
        cols = {
            "spectrum": spectrum_col,
            "mz": np.asarray(mz, np.float64),
            "intensity": np.asarray(intensity, np.float32),
        }
        if ion_mobility is not None:
            cols["ion_mobility"] = np.asarray(ion_mobility, np.float32)
        pd.DataFrame(cols).to_parquet(Path(path) / "peaks.parquet", index=False,
                                      row_group_size=500_000)
        meta = meta.reset_index(drop=True).copy()
        if "data_storage" not in meta.columns:
            meta["data_storage"] = path
        meta.to_parquet(Path(path) / "spectra.parquet", index=False)
        return cls(path)

    # ── required interface ───────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._idx)

    def spectra_data(self, columns=None) -> pd.DataFrame:
        cols = None if columns is None else list(columns)
        df = pd.read_parquet(self._meta_path, columns=cols)
        df = df.iloc[self._idx].reset_index(drop=True)
        return self._ensure_core_columns(df) if columns is None else df

    def peaks_data(self, indices=None, columns=DEFAULT_PEAKS_VARIABLES):
        import pyarrow.parquet as pq
        import pyarrow.compute as pc

        view = self._normalize_indices(indices, len(self))
        backend_idx = self._idx[view]
        want = set(backend_idx.tolist())
        read_cols = ["spectrum"] + [c for c in columns if c != "spectrum"]
        # Predicate pushdown: only read rows whose spectrum is requested.
        tbl = pq.read_table(
            self._peaks_path, columns=read_cols,
            filters=[("spectrum", "in", list(want))],
        )
        df = tbl.to_pandas()
        grouped = {sp: g for sp, g in df.groupby("spectrum", sort=False)}
        cols = list(columns)
        out = []
        for bi in backend_idx:
            g = grouped.get(int(bi))
            if g is None or len(g) == 0:
                out.append(np.empty((0, len(cols))))
                continue
            arrs = [g[c].to_numpy(dtype=np.float64) for c in cols]
            out.append(np.column_stack(arrs))
        return out

    def subset(self, indices: Sequence[int]) -> "MsBackendParquet":
        idx = self._idx[np.asarray(indices, dtype=np.int64)]
        return MsBackendParquet(self.path, _idx=idx)
