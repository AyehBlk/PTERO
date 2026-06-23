"""
Tests for the centroiding transform (ptero.process.make_centroid_transform), the
first real processing step. Skipped if pyOpenMS (the `process` extra) is absent.

Covers:
* the full profile -> centroid pipeline through process_store: detection flips from
  profile to centroid and the peak count collapses to one centroid per profile peak;
* the 4D guard: the (m/z-only) transform refuses ion-mobility data rather than
  silently dropping mobility.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyopenms")        # the `process` extra; skip cleanly if missing

from ptero.spectra import from_dataframe, open_zarr, detect_centroided
from ptero.process import process_store, make_centroid_transform


def _profile_peaks(centers, npts=21, width=0.01):
    mz, inten = [], []
    for c in centers:
        g = np.linspace(c - 3 * width, c + 3 * width, npts)
        mz.append(g)
        inten.append(1e5 * np.exp(-0.5 * ((g - c) / width) ** 2))
    return {"mz": np.concatenate(mz),
            "intensity": np.concatenate(inten).astype(np.float32)}


def _profile_store(tmp_path: Path, n_centers_ms1=40, n_centers_ms2=15) -> str:
    rng = np.random.default_rng(0)
    meta, peaks = [], []
    for i in range(6):
        lvl = 1 if i % 2 == 0 else 2
        k = n_centers_ms1 if lvl == 1 else n_centers_ms2
        centers = np.sort(rng.uniform(300.0, 1200.0, k))
        meta.append({"ms_level": lvl, "rtime": float(i * 5), "scan_index": i,
                     "precursor_mz": (np.nan if lvl == 1 else float(centers[0] + 200))})
        peaks.append(_profile_peaks(centers))
    sp = from_dataframe(pd.DataFrame(meta), peaks)
    store = str(tmp_path / "profile.zarr")
    sp.set_backend("zarr", path=store)
    return store


def _median_peaks(store: str, ms_level: int) -> float:
    s = open_zarr(store).filter_ms_level(ms_level)
    return float(np.median([len(s.mz([j])[0]) for j in range(len(s))]))


def test_profile_is_centroided_by_the_pipeline(tmp_path):
    profile = _profile_store(tmp_path)

    # before: profile (dense, sub-0.01 Da spacing)
    p = open_zarr(profile)
    assert p.has_4d is False
    assert detect_centroided(p, ms_level=1)["centroided"] is False
    assert _median_peaks(profile, 1) == 40 * 21      # 40 peaks x 21 profile points
    assert _median_peaks(profile, 2) == 15 * 21

    # centroid it
    cent = str(tmp_path / "centroided.zarr")
    process_store(profile, cent, transform=make_centroid_transform(),
                  steps=[{"name": "centroid", "params": {}, "library": "pyopenms",
                          "library_version": "PeakPickerHiRes"}])

    # after: centroid (sparse), one peak per profile peak
    c = open_zarr(cent)
    assert detect_centroided(c, ms_level=1)["centroided"] is True
    assert detect_centroided(c, ms_level=2)["centroided"] is True
    assert _median_peaks(cent, 1) == 40
    assert _median_peaks(cent, 2) == 15


def test_centroid_transform_refuses_4d():
    t = make_centroid_transform()
    peaks_4d = {"mz": np.array([400.0, 400.01, 400.02]),
                "intensity": np.array([10.0, 20.0, 10.0]),
                "ion_mobility": np.array([0.9, 0.9, 0.9])}
    with pytest.raises(NotImplementedError, match="4D"):
        t(peaks_4d)