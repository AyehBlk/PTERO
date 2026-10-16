"""Exact, incremental QC summaries for Zarr conversion."""

import numpy as np


class ZarrQCAccumulator:
    """Accumulate full-data QC by MS level without retaining peak arrays."""

    def __init__(self):
        self._groups = {}

    def add_batch(self, meta, mz, intensity, offsets, ion_mobility=None):
        mz = np.asarray(mz)
        intensity = np.asarray(intensity)
        offsets = np.asarray(offsets, dtype=np.int64)
        im = None if ion_mobility is None else np.asarray(ion_mobility)

        if len(meta) != len(offsets) - 1:
            raise ValueError("QC metadata and offsets disagree")
        if len(mz) != len(intensity) or offsets[-1] != len(mz):
            raise ValueError("QC peak arrays and offsets disagree")
        if im is not None and len(im) != len(mz):
            raise ValueError("QC ion-mobility array length disagrees")

        levels = (
            meta["ms_level"].to_numpy()
            if "ms_level" in meta.columns
            else np.full(len(meta), -1)
        )

        for i, level in enumerate(levels):
            start, end = int(offsets[i]), int(offsets[i + 1])
            level = int(level) if np.isfinite(level) else -1

            for key in ("all", str(level)):
                group = self._groups.setdefault(key, {
                    "n_spectra": 0,
                    "n_peaks": 0,
                    "mz_min": None,
                    "mz_max": None,
                    "mobility_min": None,
                    "mobility_max": None,
                    "tic_sum": 0.0,
                    "bpc_max": None,
                    "nonfinite_mz": 0,
                    "nonfinite_intensity": 0,
                    "nonfinite_mobility": 0,
                })

                group["n_spectra"] += 1
                group["n_peaks"] += end - start

                if end == start:
                    continue

                m = mz[start:end]
                v = intensity[start:end]

                finite_m = m[np.isfinite(m)]
                finite_v = v[np.isfinite(v)]

                group["nonfinite_mz"] += len(m) - len(finite_m)
                group["nonfinite_intensity"] += len(v) - len(finite_v)

                if len(finite_m):
                    lo, hi = float(finite_m.min()), float(finite_m.max())
                    group["mz_min"] = lo if group["mz_min"] is None else min(group["mz_min"], lo)
                    group["mz_max"] = hi if group["mz_max"] is None else max(group["mz_max"], hi)

                if len(finite_v):
                    group["tic_sum"] += float(np.sum(finite_v, dtype=np.float64))
                    peak_max = float(finite_v.max())
                    group["bpc_max"] = (
                        peak_max if group["bpc_max"] is None
                        else max(group["bpc_max"], peak_max)
                    )

                if im is not None:
                    mob = im[start:end]
                    finite = mob[np.isfinite(mob)]
                    group["nonfinite_mobility"] += len(mob) - len(finite)

                    if len(finite):
                        lo, hi = float(finite.min()), float(finite.max())
                        group["mobility_min"] = (
                            lo if group["mobility_min"] is None
                            else min(group["mobility_min"], lo)
                        )
                        group["mobility_max"] = (
                            hi if group["mobility_max"] is None
                            else max(group["mobility_max"], hi)
                        )

    def result(self):
        return {
            "version": 1,
            "scope": "full_data",
            "groups": self._groups,
        }
