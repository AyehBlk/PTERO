"""
Tests for centroid/profile detection (ptero.spectra._centroid + the fixed
PteroSpectra.centroid_check).

Covers:
* within-slice detection on hand-built 4D data with DISCRETE mobility scans
  (profile -> not centroided; centroided -> centroided), the case the old pooled
  check got wrong;
* 3D data measured on the pooled m/z array;
* graceful pooled-fallback for 4D data with continuous mobility (synthetic), which
  keeps the old verdict so existing behaviour is preserved;
* detect_centroided preferring a self-consistent file 'centroided' flag over the
  empirical measurement;
* annotate_store_centroided stamping the per-level verdict into the store and being
  idempotent unless force=True.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptero.spectra import (
    open_zarr, from_dataframe, datasets,
    detect_centroided, annotate_store_centroided,
)
from ptero.spectra._centroid import measure_centroided, CENTROID_CUTOFF_DA


def _build_4d(kind: str, n_scans: int = 4, n_peaks_per_scan: int = 25, ms_level: int = 1):
    """One 4D spectrum with DISCRETE mobility scans. kind='profile' gives each peak
    a cluster of adjacent m/z points; kind='centroid' gives one point per peak."""
    centers = np.linspace(200.0, 1600.0, n_peaks_per_scan)
    scans = np.round(np.linspace(0.80, 1.20, n_scans), 4)
    mzs, ints, ims = [], [], []
    for s in scans:
        for c in centers:
            pts = (c + np.arange(-3, 4) * 0.006) if kind == "profile" else np.array([c])
            mzs.append(pts)
            ints.append(np.full(len(pts), 1.0e4, dtype=np.float32))
            ims.append(np.full(len(pts), s))
    meta = pd.DataFrame([{"ms_level": ms_level, "rtime": 1.0, "scan_index": 0}])
    peaks = [{"mz": np.concatenate(mzs),
              "intensity": np.concatenate(ints).astype(np.float32),
              "ion_mobility": np.concatenate(ims).astype(np.float32)}]
    return from_dataframe(meta, peaks)


@pytest.mark.parametrize("kind,expect_centroided", [("profile", False), ("centroid", True)])
def test_within_slice_4d(kind, expect_centroided):
    sp = _build_4d(kind)
    r = sp.centroid_check(n_check=1)
    assert r["basis"] == "within_slice_1overK0"
    assert r["centroided"] is expect_centroided
    if kind == "profile":
        assert r["median_mz_spacing"] < CENTROID_CUTOFF_DA
    else:
        assert r["median_mz_spacing"] > CENTROID_CUTOFF_DA


def test_pooled_check_would_be_wrong_for_4d_profile_centroided_mix():
    """A centroided 4D spectrum, pooled across scans, still reads centroided here
    only because the scans share m/z; the point of the within-slice basis is that
    it is the right axis. Sanity: within-slice basis is used for 4D."""
    sp = _build_4d("profile")
    assert sp.has_4d
    assert sp.centroid_check(n_check=1)["basis"] == "within_slice_1overK0"


def test_3d_uses_pooled_basis():
    sp = datasets.simulate_run(n_features=80, n_ms1_scans=15, with_ion_mobility=False,
                               with_ms2=True, seed=1)
    r = sp.centroid_check(n_check=8)
    assert r["basis"] == "pooled_mz"
    # synthetic data is sparse -> centroided True (unchanged from old behaviour)
    assert r["centroided"] is True


def test_4d_continuous_mobility_falls_back_to_pooled():
    """When per-peak mobility is continuous (every peak its own value, no real TIMS
    scans), no single slice has enough peaks, so the measure falls back to the pooled
    m/z array instead of failing -- flagged in 'basis' so the fallback is never silent."""
    n = 200
    mz = np.sort(np.random.default_rng(0).uniform(300.0, 1500.0, n))   # sparse -> centroided
    im = np.linspace(0.70, 1.30, n).astype(np.float32)                 # all distinct
    meta = pd.DataFrame([{"ms_level": 1, "rtime": 1.0, "scan_index": 0}])
    sp = from_dataframe(meta, [{"mz": mz, "intensity": np.full(n, 1e4, np.float32),
                                "ion_mobility": im}])
    r = sp.centroid_check(n_check=1)
    assert r["basis"] == "pooled_mz_fallback"
    assert r["centroided"] is True


def test_synthetic_4d_verdict_unchanged_backward_compat():
    """The fix must not change the verdict on existing synthetic 4D data: it was
    centroided before, and must stay centroided (only the basis label is new)."""
    sp = datasets.simulate_run(n_features=80, n_ms1_scans=15, with_ion_mobility=True,
                               with_ms2=True, seed=1)
    assert sp.is_centroided() is True


def test_detect_prefers_consistent_file_metadata():
    # build a profile 4D spectrum but DECLARE it centroided in metadata
    sp = _build_4d("profile")
    sp._backend._meta["centroided"] = True            # simulate an mzML-declared flag
    d = detect_centroided(sp)
    assert d["source"] == "file_metadata"
    assert d["centroided"] is True                    # trusts metadata over the data
    # with prefer_metadata=False it measures instead -> profile -> not centroided
    m = detect_centroided(sp, prefer_metadata=False)
    assert m["source"] == "measured"
    assert m["centroided"] is False


def test_annotate_store_stamps_and_is_idempotent(tmp_path):
    # a Bruker-like store: 4D, MS1 (profile) + MS2 (centroid), NO centroided flag
    ms1 = _build_4d("profile", ms_level=1)
    ms2 = _build_4d("centroid", ms_level=2)
    meta = pd.concat([ms1.spectra_data(), ms2.spectra_data()], ignore_index=True)
    peaks = (ms1.peaks_data(None, ("mz", "intensity", "ion_mobility"))
             + ms2.peaks_data(None, ("mz", "intensity", "ion_mobility")))
    peaks = [{"mz": p[:, 0], "intensity": p[:, 1], "ion_mobility": p[:, 2]} for p in peaks]
    combined = from_dataframe(meta, peaks)
    store = str(tmp_path / "bruker_like.zarr")
    combined.set_backend("zarr", path=store, peak_ion_mobility_type="inverse_reduced_ion_mobility", peak_ion_mobility_unit="V.s/cm2")

    # before: centroided column is empty (NaN-filled)
    before = pd.read_parquet(Path(store) / "spectra.parquet")
    assert before["centroided"].isna().all()

    res = annotate_store_centroided(store, n_check=2)
    assert res["action"] == "annotated"
    assert res["levels"][1]["centroided"] is False     # MS1 profile
    assert res["levels"][2]["centroided"] is True       # MS2 centroid

    after = pd.read_parquet(Path(store) / "spectra.parquet")
    ms1_flags = after.loc[after["ms_level"] == 1, "centroided"]
    ms2_flags = after.loc[after["ms_level"] == 2, "centroided"]
    assert bool(ms1_flags.iloc[0]) is False
    assert bool(ms2_flags.iloc[0]) is True

    # the verdict is now readable as authoritative metadata
    sp2 = open_zarr(store)
    assert detect_centroided(sp2, ms_level=2)["source"] == "file_metadata"

    # idempotent: a second call keeps the existing flag rather than re-measuring
    res2 = annotate_store_centroided(store, n_check=2)
    assert res2["action"] == "kept_existing"
    # ... unless forced
    res3 = annotate_store_centroided(store, n_check=2, force=True)
    assert res3["action"] == "annotated"