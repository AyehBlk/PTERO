"""
Tests for ptero.spectra._dia (diaPASEF window-scheme reading from SQLite).

These build a synthetic ``analysis.tdf``-shaped SQLite database in a temp ``.d``
folder -- using the real schema and real window numbers observed in a HeLa
diaPASEF run -- so the metadata reader is exercised against realistic data with
no instrument file and no vendor library.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptero.spectra import _dia as dia


# real group-1 and group-2 windows from the observed scheme (scans, m/z, CE)
_GROUP1 = [(125, 566, 812.5, 25.0, 45.36), (566, 725, 612.5, 25.0, 32.24),
           (725, 926, 412.5, 25.0, 24.37)]
_GROUP2 = [(125, 539, 837.5, 25.0, 45.93), (539, 708, 637.5, 25.0, 33.20),
           (708, 926, 437.5, 25.0, 24.72)]


def _make_dia_tdf(d_dir: Path, nested: bool = False) -> Path:
    """Create a minimal but schema-faithful diaPASEF analysis.tdf. Return the .d dir."""
    inner = d_dir / d_dir.name if nested else d_dir
    inner.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(inner / "analysis.tdf")
    cur = con.cursor()
    # note the real-world capitalisation 'GlobalMetadata' (not 'GlobalMetaData')
    cur.execute("CREATE TABLE GlobalMetadata (Key TEXT, Value TEXT)")
    cur.execute("CREATE TABLE Frames (Id INTEGER PRIMARY KEY, Time REAL, MsMsType INTEGER, NumScans INTEGER)")
    cur.execute("CREATE TABLE DiaFrameMsMsWindows "
                "(WindowGroup INTEGER, ScanNumBegin INTEGER, ScanNumEnd INTEGER, "
                "IsolationMz REAL, IsolationWidth REAL, CollisionEnergy REAL)")
    cur.execute("CREATE TABLE DiaFrameMsMsInfo (Frame INTEGER, WindowGroup INTEGER)")
    # frames: 1 & 10 are MS1 (type 0); 2 & 3 are MS2 (type 9) for groups 1 & 2
    cur.executemany("INSERT INTO Frames VALUES (?,?,?,?)",
                    [(1, 0.5, 0, 927), (2, 0.6, 9, 927), (3, 0.7, 9, 927), (10, 1.5, 0, 927)])
    for g, rows in ((1, _GROUP1), (2, _GROUP2)):
        cur.executemany("INSERT INTO DiaFrameMsMsWindows VALUES (?,?,?,?,?,?)",
                        [(g, *r) for r in rows])
    cur.executemany("INSERT INTO DiaFrameMsMsInfo VALUES (?,?)", [(2, 1), (3, 2)])
    con.commit()
    con.close()
    return d_dir


def _make_dda_tdf(d_dir: Path) -> Path:
    d_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(d_dir / "analysis.tdf")
    cur = con.cursor()
    cur.execute("CREATE TABLE Frames (Id INTEGER PRIMARY KEY, MsMsType INTEGER)")
    cur.execute("CREATE TABLE Precursors (Id INTEGER, MonoisotopicMz REAL)")
    cur.execute("CREATE TABLE PasefFrameMsMsInfo (Frame INTEGER, Precursor INTEGER, IsolationMz REAL)")
    cur.executemany("INSERT INTO Frames VALUES (?,?)", [(1, 0), (2, 8)])
    cur.execute("INSERT INTO Precursors VALUES (1, 543.21)")
    cur.execute("INSERT INTO PasefFrameMsMsInfo VALUES (2, 1, 543.5)")
    con.commit()
    con.close()
    return d_dir


# ── detection ────────────────────────────────────────────────────────────────

def test_detect_dia(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    assert dia.detect_acquisition_mode(d) == "dia"


def test_detect_dda(tmp_path):
    d = _make_dda_tdf(tmp_path / "run.d")
    assert dia.detect_acquisition_mode(d) == "dda"


def test_locate_tdf_nested(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d", nested=True)
    # pointing at the OUTER folder must still find the inner analysis.tdf
    assert dia.locate_tdf(d).exists()
    assert dia.detect_acquisition_mode(d) == "dia"


def test_connect_ro_missing_file_raises(tmp_path):
    # the bug that produced "no such table": a missing tdf must raise, not create
    with pytest.raises(Exception):
        dia._connect_ro(tmp_path / "does_not_exist.tdf").execute("SELECT 1 FROM Frames")


# ── scheme ───────────────────────────────────────────────────────────────────

def test_window_scheme_shape_and_bounds(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s = dia.read_window_scheme(d)
    assert len(s) == 6
    assert s["window_group"].nunique() == 2
    assert sorted(s["sub_window"].unique()) == [0, 1, 2]
    assert list(s["window_id"]) == [0, 1, 2, 3, 4, 5]  # unique, ordered
    # mz bounds = centre +/- width/2
    row = s[(s["window_group"] == 1) & (s["isolation_mz"] == 812.5)].iloc[0]
    assert row["mz_lower"] == pytest.approx(800.0)
    assert row["mz_upper"] == pytest.approx(825.0)
    # sub_window ordered by ascending scan_begin within the group
    g1 = s[s["window_group"] == 1].sort_values("sub_window")
    assert list(g1["scan_begin"]) == [125, 566, 725]


def test_mz_unique(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s = dia.read_window_scheme(d)
    assert dia.windows_have_unique_mz(s) is True


def test_frame_window_map(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    fm = dia.read_frame_window_map(d)
    assert dict(zip(fm["frame"], fm["window_group"])) == {2: 1, 3: 2}


def test_expand_frame_windows(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    ex = dia.expand_frame_windows(d)
    assert len(ex) == 6  # 2 MS2 frames x 3 sub-windows
    assert set(ex["frame"]) == {2, 3}
    assert (ex.groupby("frame").size() == 3).all()


# ── event assignment (the read-time core) ────────────────────────────────────

def test_assign_by_scan_half_open(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s = dia.read_window_scheme(d)
    fm = dia.read_frame_window_map(d)
    # frame 2 -> group 1: windows [125,566)->812.5, [566,725)->612.5, [725,926)->412.5
    frames = np.array([2, 2, 2, 2, 2, 2], dtype=np.int64)
    scans = np.array([124, 125, 565, 566, 725, 926], dtype=np.int64)
    wid = dia.assign_windows_by_scan(frames, scans, s, fm)
    g1 = s[s["window_group"] == 1].sort_values("scan_begin")
    w0, w1, w2 = g1["window_id"].tolist()  # 812.5, 612.5, 412.5
    assert wid[0] == -1            # 124: below first window
    assert wid[1] == w0            # 125: inclusive begin -> window 0
    assert wid[2] == w0            # 565: still in window 0
    assert wid[3] == w1            # 566: half-open -> window 1, not 0
    assert wid[4] == w2            # 725: -> window 2
    assert wid[5] == -1            # 926: exclusive end -> none


def test_assign_by_scan_ms1_frame_unassigned(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s, fm = dia.read_window_scheme(d), dia.read_frame_window_map(d)
    # frame 1 is MS1 (not in the frame->group map) -> all events unassigned
    wid = dia.assign_windows_by_scan(np.array([1, 1]), np.array([300, 600]), s, fm)
    assert (wid == -1).all()


def test_assign_by_quad_matches_scan(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s, fm = dia.read_window_scheme(d), dia.read_frame_window_map(d)
    # build events with both scan and quad info, then check the two paths agree
    frames = np.array([2, 2, 2, 3, 3, 3], dtype=np.int64)
    scans = np.array([300, 600, 800, 300, 600, 800], dtype=np.int64)
    by_scan = dia.assign_windows_by_scan(frames, scans, s, fm)
    # corresponding quad centres for those (group, scan) picks
    centers = []
    for f, sc in zip(frames, scans):
        g = fm.set_index("frame").loc[f, "window_group"]
        gdf = s[s["window_group"] == g].sort_values("scan_begin")
        row = gdf[(gdf["scan_begin"] <= sc) & (sc < gdf["scan_end"])].iloc[0]
        centers.append(row["isolation_mz"])
    centers = np.array(centers)
    quad_low = centers - 12.5
    quad_high = centers + 12.5
    by_quad = dia.assign_windows_by_quad(quad_low, quad_high, s)
    assert np.array_equal(by_scan, by_quad)


def test_assign_by_quad_ms1_sentinel(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s = dia.read_window_scheme(d)
    # quad_low < 0 marks an MS1/no-isolation event -> -1
    wid = dia.assign_windows_by_quad(np.array([-1.0, 812.5 - 12.5]),
                                     np.array([-1.0, 812.5 + 12.5]), s)
    assert wid[0] == -1
    assert wid[1] == s[s["isolation_mz"] == 812.5]["window_id"].iloc[0]


# ── summary ──────────────────────────────────────────────────────────────────

def test_scheme_summary(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    summ = dia.dia_scheme_summary(d)
    assert summ["mode"] == "dia"
    assert summ["n_window_groups"] == 2
    assert summ["n_windows_total"] == 6
    assert summ["sub_windows_per_group"] == [3]
    assert summ["isolation_widths_da"] == [25.0]
    assert summ["mz_unique_per_window"] is True
    assert summ["n_ms2_frames"] == 2


# ── long-table enrichment + DIA grouping (the reader core) ───────────────────

def _toy_long_table():
    """A small renamed-alphatims-style long table: 1 MS1 frame + 2 MS2 frames."""
    rows = []
    # MS1 frame 1: 3 survey peaks (quad = -1)
    for sc, mz, it in [(200, 500.0, 10.0), (400, 700.0, 20.0), (600, 900.0, 30.0)]:
        rows.append((1, 0.5, sc, mz, it, 0.9, -1.0, -1.0))
    # MS2 frame 2 (group 1): events in each of the 3 sub-windows + 1 unassigned
    for sc, mz, it in [(300, 150.0, 5.0),   # scan 300 -> window [125,566) (812.5)
                       (600, 160.0, 6.0),   # scan 600 -> window [566,725) (612.5)
                       (800, 170.0, 7.0),   # scan 800 -> window [725,926) (412.5)
                       (50, 180.0, 8.0)]:   # scan 50  -> no window -> dropped
        rows.append((2, 0.6, sc, mz, it, 1.0, np.nan, np.nan))
    # MS2 frame 3 (group 2): one event in the first sub-window
    rows.append((3, 0.7, 300, 200.0, 9.0, 1.1, np.nan, np.nan))
    return pd.DataFrame(rows, columns=[
        "frame", "rt", "scan", "mz", "intensity", "mobility",
        "quad_low_mz_values", "quad_high_mz_values"])


def test_assign_windows_to_long_table_ms_level_from_frame(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s, fm = dia.read_window_scheme(d), dia.read_frame_window_map(d)
    long = dia.assign_windows_to_long_table(_toy_long_table(), s, fm, method="scan")
    # frame 1 -> MS1, frames 2 & 3 -> MS2 (because they are in DiaFrameMsMsInfo)
    lev = long.set_index("frame")["ms_level"]
    assert set(np.atleast_1d(lev.loc[1])) == {1}
    assert set(np.atleast_1d(lev.loc[2])) == {2}
    assert set(np.atleast_1d(lev.loc[3])) == {2}
    # MS1 events have no window; the scan-50 MS2 event is unassigned
    assert (long[long["frame"] == 1]["window_id"] == -1).all()
    assert (long[(long["frame"] == 2) & (long["scan"] == 50)]["window_id"] == -1).all()


def test_group_dia_spectra(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s, fm = dia.read_window_scheme(d), dia.read_frame_window_map(d)
    long = dia.assign_windows_to_long_table(_toy_long_table(), s, fm)
    with pytest.warns(UserWarning, match="no isolation window"):
        meta, peaks = dia.group_dia_spectra(long)

    # 1 MS1 spectrum (frame 1) + 3 MS2 (frame 2, 3 windows) + 1 MS2 (frame 3) = 5
    assert len(meta) == 5
    assert len(peaks) == 5
    assert (meta["ms_level"] == 1).sum() == 1
    assert (meta["ms_level"] == 2).sum() == 4

    # the dropped scan-50 event is gone: frame-2 MS2 spectra hold 1 peak each
    ms2_f2 = meta[(meta["ms_level"] == 2) & (meta["isolation_mz"].notna())]
    # every MS2 spectrum carries real isolation metadata, never a precursor m/z
    assert meta.loc[meta["ms_level"] == 2, "precursor_mz"].isna().all()
    assert meta.loc[meta["ms_level"] == 2, "isolation_width"].eq(25.0).all()
    # MS1 spectrum has NaN isolation metadata
    ms1 = meta[meta["ms_level"] == 1].iloc[0]
    assert pd.isna(ms1["isolation_mz"])
    # peak counts: MS1 frame had 3 peaks; each MS2 window had 1
    n_peaks = [len(p["mz"]) for p in peaks]
    assert sorted(n_peaks) == [1, 1, 1, 1, 3]
    # 4D carried through
    assert all("ion_mobility" in p for p in peaks)


def test_group_dia_spectra_isolation_bounds(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s, fm = dia.read_window_scheme(d), dia.read_frame_window_map(d)
    long = dia.assign_windows_to_long_table(_toy_long_table(), s, fm)
    meta, _ = dia.group_dia_spectra(long)
    # the 812.5 window in group 1 must come out as [800, 825]
    w = meta[(meta["isolation_mz"] == 812.5)]
    assert len(w) == 1
    assert w["isolation_mz_lower"].iloc[0] == pytest.approx(800.0)
    assert w["isolation_mz_upper"].iloc[0] == pytest.approx(825.0)


# ── per-window mobility (1/K0) bounds ────────────────────────────────────────

def _mobility_model(scan):
    """A monotonic-decreasing scan->1/K0 map, like a real TIMS ramp."""
    return 1.5 - np.asarray(scan, dtype=float) * 0.001


def _long_table_with_mobility():
    """Toy table whose 'mobility' follows _mobility_model, covering scans 50..900."""
    scans = np.array([50, 125, 300, 565, 600, 725, 800, 900])
    rows = []
    for sc in scans:
        rows.append((2, 0.6, int(sc), 150.0, 5.0, float(_mobility_model(sc))))
    return pd.DataFrame(rows, columns=["frame", "rt", "scan", "mz", "intensity", "mobility"])


def test_build_scan_mobility_lookup(tmp_path):
    df = _long_table_with_mobility()
    scans, mob = dia.build_scan_mobility_lookup(df)
    assert list(scans) == sorted(scans.tolist())          # ascending
    assert mob[0] == pytest.approx(_mobility_model(50))    # matches the model


def test_add_mobility_bounds(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s = dia.read_window_scheme(d)
    lookup = dia.build_scan_mobility_lookup(_long_table_with_mobility())
    s2 = dia.add_mobility_bounds(s, lookup)
    assert "im_lower" in s2.columns and "im_upper" in s2.columns
    assert (s2["im_lower"] <= s2["im_upper"]).all()        # always low<=high
    # window scans [125,566): 1/K0 spans model(125)..model(565); since 1/K0 falls
    # with scan, im_lower≈model(565), im_upper≈model(125)
    w = s2[(s2["window_group"] == 1) & (s2["scan_begin"] == 125)].iloc[0]
    assert w["im_upper"] == pytest.approx(_mobility_model(125), abs=1e-3)
    assert w["im_lower"] == pytest.approx(_mobility_model(565), abs=2e-3)


def test_mobility_bounds_flow_to_spectra(tmp_path):
    d = _make_dia_tdf(tmp_path / "run.d")
    s, fm = dia.read_window_scheme(d), dia.read_frame_window_map(d)
    s = dia.add_mobility_bounds(s, dia.build_scan_mobility_lookup(_long_table_with_mobility()))
    long = dia.assign_windows_to_long_table(_toy_long_table(), s, fm)
    meta, _ = dia.group_dia_spectra(long)
    ms2 = meta[meta["ms_level"] == 2]
    # every MS2 spectrum now carries finite mobility bounds, low <= high
    assert ms2["isolation_im_lower"].notna().all()
    assert ms2["isolation_im_upper"].notna().all()
    assert (ms2["isolation_im_lower"] <= ms2["isolation_im_upper"]).all()
    # MS1 spectrum has NaN mobility bounds
    ms1 = meta[meta["ms_level"] == 1].iloc[0]
    assert pd.isna(ms1["isolation_im_lower"])


# ── storage round-trip: a DIA batch -> Zarr -> reopened ──────────────────────

def test_dia_zarr_roundtrip(tmp_path):
    pytest.importorskip("zarr")
    import warnings
    from ptero.spectra.backends.memory import MsBackendMemory
    from ptero.spectra.backends.zarr_backend import _ZarrStreamWriter, MsBackendZarr

    d = _make_dia_tdf(tmp_path / "run.d")
    s, fm = dia.read_window_scheme(d), dia.read_frame_window_map(d)
    s = dia.add_mobility_bounds(s, dia.build_scan_mobility_lookup(_long_table_with_mobility()))
    long = dia.assign_windows_to_long_table(_toy_long_table(), s, fm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta, peaks = dia.group_dia_spectra(long)

    mem = MsBackendMemory.from_spectra_list(meta, peaks)
    store = str(tmp_path / "out.zarr")
    w = _ZarrStreamWriter(store, has_4d=True, acquisition_mode="dia", peak_ion_mobility_type="inverse_reduced_ion_mobility", peak_ion_mobility_unit="V.s/cm2")
    w.append_backend(mem)
    w.finalize()

    b = MsBackendZarr(store)
    assert b.acquisition_mode == "dia"                 # mode survives the write
    sd = b.spectra_data()
    for c in ["window_id", "isolation_mz", "isolation_mz_lower", "isolation_im_lower"]:
        assert c in sd.columns                         # window metadata survives
    ms2 = sd[sd["ms_level"] == 2]
    assert len(ms2) >= 1
    assert ms2["isolation_im_lower"].notna().all()
