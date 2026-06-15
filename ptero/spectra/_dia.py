"""
diaPASEF acquisition-scheme reading for PTERO (Bruker timsTOF ``.d``).

Why this is separate from the alphatims peak reader
---------------------------------------------------
The diaPASEF *window scheme* is acquisition **metadata**, not detector data.
Bruker stores it explicitly in the ``analysis.tdf`` SQLite database, in two
small tables:

    DiaFrameMsMsWindows   one row per (window group x mobility sub-window):
                          WindowGroup, ScanNumBegin, ScanNumEnd, IsolationMz,
                          IsolationWidth, CollisionEnergy
    DiaFrameMsMsInfo      one row per MS2 frame:  Frame -> WindowGroup

PTERO therefore reads the scheme **directly from SQLite with the standard
library only** (no vendor library, every OS). Consequences:

* it is the *authoritative* source -- exact integer scan ranges and exact
  isolation m/z, not values reverse-engineered from per-event quad bounds;
* assigning an MS2 detector event to a window is **exact integer logic**
  (frame -> window group, scan number -> sub-window), never a fragile
  float-equality grouping on a computed window centre;
* it is fully unit-testable against a synthetic ``.tdf``-shaped database, with
  no real instrument file required.

alphatims is still required to read the actual peaks (m/z, intensity, mobility)
and to convert TIMS scan numbers to 1/K0; that lives in ``_bruker.py``. This
module reads metadata only and never imports a vendor library.

Scan ranges are **half-open** ``[ScanNumBegin, ScanNumEnd)``: the end of one
sub-window equals the begin of the next, so the boundary scan belongs to the
*next* sub-window. Scans outside every sub-window's range (e.g. the high-
mobility head before the first window) are assigned window id ``-1``.

Observed schema (HeLa 21 min diaPASEF): 8 window groups x 3 sub-windows = 24
windows; isolation width 25 Da; ``Frames.MsMsType`` 9 = diaPASEF MS2, 0 = MS1.
The integer code 9 is **not** hard-coded for detection below -- detection keys
off which tables are populated, which is robust across acquisition-software
versions.
"""
from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, "Path"]

# Canonical table names. Detection compares case-insensitively against the real
# table list, so capitalisation differences across versions do not matter.
_DIA_SCHEME_TABLE = "DiaFrameMsMsWindows"
_DIA_INFO_TABLE = "DiaFrameMsMsInfo"
_DDA_TABLES = ("Precursors", "PasefFrameMsMsInfo")


# ── locating analysis.tdf (tolerates a nested .d, a stub, or a .tsf outer) ───

def _tdf_is_real(tdf: Path) -> bool:
    """True if ``tdf`` is a usable Bruker metadata database.

    A nested or freshly-copied ``.d`` can leave a 0-byte (or otherwise broken)
    ``analysis.tdf`` stub at the outer level while the real database sits one
    folder down. A real database opens read-only and contains the core tables
    (``GlobalMetadata`` / ``Frames``); a stub does not. Cheap and case-insensitive
    (SQLite table names are case-insensitive).
    """
    try:
        if not tdf.exists() or tdf.stat().st_size == 0:
            return False
        con = sqlite3.connect(f"file:{tdf}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                "AND lower(name) IN ('globalmetadata', 'frames')"
            ).fetchone()
            return bool(row and row[0] >= 1)
        finally:
            con.close()
    except Exception:
        return False


def resolve_run_dir(path: PathLike) -> str:
    """Return the directory that holds a *usable* Bruker metadata database.

    Descends one level into a nested ``.d`` and skips an outer folder whose
    ``analysis.tdf`` is a stub (or which only has a ``.tsf``) in favour of a
    child with a real ``analysis.tdf``. Falls back to a stub/``.tsf`` folder only
    if no real database is found anywhere, so the opener can raise a clear error
    rather than mis-opening the wrong folder.
    """
    p = Path(str(path))
    if p.is_file():  # an analysis.tdf/.tsf passed directly
        return str(p.parent)
    candidates = [p]
    if p.is_dir():
        candidates += sorted(c for c in p.iterdir() if c.is_dir())
    # 1) a real, openable analysis.tdf wins (this is the fix for the stub/.tsf outer)
    for d in candidates:
        if _tdf_is_real(d / "analysis.tdf"):
            return str(d)
    # 2) any analysis.tdf (even a stub) -> let the opener raise a clear error
    for d in candidates:
        if (d / "analysis.tdf").exists():
            return str(d)
    # 3) a .tsf-only run (non-TIMS Bruker)
    for d in candidates:
        if (d / "analysis.tsf").exists():
            return str(d)
    return str(p)


def locate_tdf(path: PathLike) -> Path:
    """Return the usable ``analysis.tdf`` for a ``.d`` run (see :func:`resolve_run_dir`).

    Accepts the ``.d`` folder, a nested ``.d`` (an outer ``.d`` whose data lives
    in an inner ``.d`` of the same name), or the ``analysis.tdf`` file itself.
    """
    p = Path(str(path))
    if p.is_file() and p.name.lower() == "analysis.tdf":
        return p
    tdf = Path(resolve_run_dir(p)) / "analysis.tdf"
    if not tdf.exists():
        raise FileNotFoundError(
            f"No real analysis.tdf found in {p} (or one level below). Point this "
            "at the .d folder that directly contains a usable analysis.tdf."
        )
    return tdf


def _connect_ro(tdf: Path) -> sqlite3.Connection:
    """Open the SQLite database read-only (never creates a file)."""
    # uri=True + mode=ro: if the file is missing this raises instead of silently
    # creating an empty database (the failure mode that produced the misleading
    # "no such table: GlobalMetaData" error).
    return sqlite3.connect(f"file:{tdf}?mode=ro", uri=True)


def _real_table_names(con: sqlite3.Connection) -> dict[str, str]:
    """Map lower-cased table name -> actual name, for case-insensitive lookup."""
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0].lower(): r[0] for r in rows}


def _resolve_table(con: sqlite3.Connection, wanted: str) -> Optional[str]:
    return _real_table_names(con).get(wanted.lower())


def _row_count(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


# ── acquisition-mode detection ───────────────────────────────────────────────

def detect_acquisition_mode(path: PathLike) -> str:
    """Return ``"dia"``, ``"dda"``, ``"ambiguous"``, or ``"unknown"``.

    Decided by which MS/MS metadata tables are *populated*, which is the robust
    signal -- not by window widths or frame-type codes.
    """
    tdf = locate_tdf(path)
    con = _connect_ro(tdf)
    try:
        names = _real_table_names(con)

        def populated(wanted: str) -> bool:
            real = names.get(wanted.lower())
            return real is not None and _row_count(con, real) > 0

        dia = populated(_DIA_SCHEME_TABLE)
        dda = any(populated(t) for t in _DDA_TABLES)
    finally:
        con.close()

    if dia and not dda:
        return "dia"
    if dda and not dia:
        return "dda"
    if dia and dda:
        return "ambiguous"
    return "unknown"


# ── the window scheme ────────────────────────────────────────────────────────

def read_window_scheme(path: PathLike) -> pd.DataFrame:
    """Authoritative diaPASEF window scheme from ``DiaFrameMsMsWindows``.

    Returns one row per window, sorted by ``(window_group, scan_begin)``, with:

    ===================  =======  ======================================================
    column               dtype    meaning
    ===================  =======  ======================================================
    window_group         int64    group id as stored in the file (1-based)
    sub_window           int64    0-based index of the sub-window within its group,
                                  ordered by ascending scan_begin
    window_id            int64    0-based unique id across the whole scheme
    scan_begin           int64    TIMS scan-number range start (inclusive)
    scan_end             int64    TIMS scan-number range end (exclusive; half-open)
    isolation_mz         float64  quadrupole centre m/z
    isolation_width      float64  full isolation width (Da)
    mz_lower             float64  isolation_mz - isolation_width / 2
    mz_upper             float64  isolation_mz + isolation_width / 2
    collision_energy     float64  collision energy (eV)
    ===================  =======  ======================================================

    Mobility (1/K0) bounds are intentionally **not** computed here: converting
    scan numbers to 1/K0 needs the TIMS calibration, which is read with the peak
    data in ``_bruker.py``. ``scan_begin``/``scan_end`` are the exact, sufficient
    keys for event-to-window assignment.
    """
    tdf = locate_tdf(path)
    con = _connect_ro(tdf)
    try:
        tbl = _resolve_table(con, _DIA_SCHEME_TABLE)
        if tbl is None:
            raise ValueError(
                f"{_DIA_SCHEME_TABLE} not present in {tdf}. This run is not "
                "diaPASEF (check detect_acquisition_mode first)."
            )
        df = pd.read_sql(f"SELECT * FROM {tbl}", con)
    finally:
        con.close()

    if len(df) == 0:
        raise ValueError(
            f"{_DIA_SCHEME_TABLE} is empty in {tdf}; the run is not diaPASEF "
            "(or the scheme was not written)."
        )

    out = pd.DataFrame({
        "window_group": df["WindowGroup"].astype("int64"),
        "scan_begin": df["ScanNumBegin"].astype("int64"),
        "scan_end": df["ScanNumEnd"].astype("int64"),
        "isolation_mz": df["IsolationMz"].astype("float64"),
        "isolation_width": df["IsolationWidth"].astype("float64"),
        "collision_energy": df["CollisionEnergy"].astype("float64"),
    })
    out["mz_lower"] = out["isolation_mz"] - out["isolation_width"] / 2.0
    out["mz_upper"] = out["isolation_mz"] + out["isolation_width"] / 2.0

    out = out.sort_values(["window_group", "scan_begin"], kind="stable").reset_index(drop=True)
    out["sub_window"] = out.groupby("window_group").cumcount().astype("int64")
    out["window_id"] = np.arange(len(out), dtype="int64")

    return out[[
        "window_group", "sub_window", "window_id",
        "scan_begin", "scan_end",
        "isolation_mz", "isolation_width", "mz_lower", "mz_upper",
        "collision_energy",
    ]]


def read_frame_window_map(path: PathLike) -> pd.DataFrame:
    """``DiaFrameMsMsInfo`` as a tidy table: ``frame`` (int) -> ``window_group`` (int)."""
    tdf = locate_tdf(path)
    con = _connect_ro(tdf)
    try:
        tbl = _resolve_table(con, _DIA_INFO_TABLE)
        if tbl is None:
            raise ValueError(f"{_DIA_INFO_TABLE} not present in {tdf} (not diaPASEF).")
        df = pd.read_sql(f"SELECT * FROM {tbl}", con)
    finally:
        con.close()
    return pd.DataFrame({
        "frame": df["Frame"].astype("int64"),
        "window_group": df["WindowGroup"].astype("int64"),
    })


def expand_frame_windows(path: PathLike) -> pd.DataFrame:
    """Join the frame->group map with the scheme: one row per (MS2 frame, sub-window).

    This is the per-spectrum table for DIA MS2: each (frame, window) pair is one
    output spectrum. Columns: ``frame`` plus all scheme columns.
    """
    scheme = read_window_scheme(path)
    fmap = read_frame_window_map(path)
    merged = fmap.merge(scheme, on="window_group", how="left")
    return merged.sort_values(["frame", "scan_begin"], kind="stable").reset_index(drop=True)


# ── event-to-window assignment (the read-time core; pure & testable) ─────────

def windows_have_unique_mz(scheme: pd.DataFrame) -> bool:
    """True if every window has a distinct isolation centre.

    When true, an MS2 event can be assigned to its window from quadrupole bounds
    alone. When false (staggered/overlapping schemes that reuse an m/z window at
    different mobility ranges), scan-number assignment is required.
    """
    return scheme["isolation_mz"].round(4).nunique() == len(scheme)


def assign_windows_by_scan(
    frames: np.ndarray,
    scans: np.ndarray,
    scheme: pd.DataFrame,
    frame_map: pd.DataFrame,
) -> np.ndarray:
    """Assign each MS2 detector event to a ``window_id`` from (frame, scan number).

    This is the general, always-correct path: exact integer logic, independent of
    any m/z uniqueness in the scheme. Events whose frame is not an MS2 frame, or
    whose scan number falls in no sub-window, get ``-1``.

    Parameters
    ----------
    frames, scans : 1-D int arrays of equal length, one entry per detector event.
    scheme : output of :func:`read_window_scheme`.
    frame_map : output of :func:`read_frame_window_map`.
    """
    frames = np.asarray(frames, dtype=np.int64)
    scans = np.asarray(scans, dtype=np.int64)
    if frames.shape != scans.shape:
        raise ValueError("frames and scans must have the same shape.")
    out = np.full(frames.shape, -1, dtype=np.int64)

    # dense frame -> window_group lookup (frame ids are small, ~1e4)
    max_frame = int(frames.max()) if frames.size else 0
    max_frame = max(max_frame, int(frame_map["frame"].max()) if len(frame_map) else 0)
    grp_of_frame = np.full(max_frame + 1, -1, dtype=np.int64)
    fm = frame_map[frame_map["frame"] <= max_frame]
    grp_of_frame[fm["frame"].to_numpy()] = fm["window_group"].to_numpy()
    event_group = np.where(frames <= max_frame, grp_of_frame[np.clip(frames, 0, max_frame)], -1)

    # assign per group (a handful of iterations, each vectorised over its events)
    for g, gdf in scheme.groupby("window_group", sort=True):
        gdf = gdf.sort_values("scan_begin")
        begins = gdf["scan_begin"].to_numpy(np.int64)
        ends = gdf["scan_end"].to_numpy(np.int64)
        wids = gdf["window_id"].to_numpy(np.int64)
        # sub-windows are contiguous & half-open: edges = [b0, b1, ..., end_last]
        edges = np.concatenate([begins, ends[-1:]])
        sel = np.where(event_group == int(g))[0]
        if sel.size == 0:
            continue
        s = scans[sel]
        pos = np.searchsorted(edges, s, side="right") - 1
        in_range = (pos >= 0) & (pos < len(wids)) & (s >= edges[0]) & (s < edges[-1])
        assigned = np.where(in_range, wids[np.clip(pos, 0, len(wids) - 1)], -1)
        out[sel] = assigned
    return out


def assign_windows_by_quad(
    quad_low: np.ndarray,
    quad_high: np.ndarray,
    scheme: pd.DataFrame,
    atol: float = 1e-3,
) -> np.ndarray:
    """Assign events to ``window_id`` by matching quadrupole bounds to the scheme.

    Faster and frame-map-free, but only valid when :func:`windows_have_unique_mz`
    is true (raises otherwise). Intended as the validation cross-check against
    :func:`assign_windows_by_scan`, and as a fast path when applicable. Events
    with no isolation (MS1, ``quad_low < 0``) or no match get ``-1``.
    """
    if not windows_have_unique_mz(scheme):
        raise ValueError(
            "Scheme reuses isolation m/z across windows; quad-based assignment is "
            "ambiguous. Use assign_windows_by_scan instead."
        )
    quad_low = np.asarray(quad_low, dtype=np.float64)
    quad_high = np.asarray(quad_high, dtype=np.float64)
    center = (quad_low + quad_high) / 2.0
    out = np.full(center.shape, -1, dtype=np.int64)

    centers = scheme["isolation_mz"].to_numpy(np.float64)
    wids = scheme["window_id"].to_numpy(np.int64)
    order = np.argsort(centers)
    cs, ws = centers[order], wids[order]
    valid = quad_low >= 0
    idx = np.searchsorted(cs, center)
    idx = np.clip(idx, 0, len(cs) - 1)
    # check nearest of idx and idx-1
    for cand in (idx, np.clip(idx - 1, 0, len(cs) - 1)):
        hit = valid & (np.abs(center - cs[cand]) <= atol) & (out < 0)
        out[hit] = ws[cand][hit]
    return out


# ── compact QC summary ───────────────────────────────────────────────────────

def dia_scheme_summary(path: PathLike) -> dict:
    """One-glance QC of the acquisition scheme (safe to show in a dashboard)."""
    scheme = read_window_scheme(path)
    fmap = read_frame_window_map(path)
    widths = sorted(scheme["isolation_width"].round(4).unique().tolist())
    per_group = scheme.groupby("window_group").size()
    return {
        "mode": "dia",
        "n_window_groups": int(scheme["window_group"].nunique()),
        "n_windows_total": int(len(scheme)),
        "sub_windows_per_group": sorted(per_group.unique().tolist()),
        "isolation_widths_da": widths,
        "mz_range": (float(scheme["mz_lower"].min()), float(scheme["mz_upper"].max())),
        "mz_unique_per_window": bool(windows_have_unique_mz(scheme)),
        "n_ms2_frames": int(len(fmap)),
        "frames_per_group": fmap.groupby("window_group").size().to_dict(),
    }


# ── turning a raw detector-event table into DIA spectra ──────────────────────
#
# These two functions are the DIA analogue of _bruker._timstof_to_spectra. They
# are pure pandas/numpy (no vendor library, no PteroSpectra), so the reader's
# data-shaping logic is unit-tested here with synthetic data, exactly like the
# DDA core. _bruker.py calls them and wraps the result with from_dataframe.

# Per-spectrum window-metadata columns added to the output (NaN for MS1 spectra).
# The isolation_im_* (1/K0) bounds are present only when the scheme has been
# enriched with mobility bounds (see add_mobility_bounds); they are NaN otherwise.
DIA_WINDOW_META = [
    "window_id", "window_group", "isolation_mz", "isolation_width",
    "isolation_mz_lower", "isolation_mz_upper",
    "isolation_im_lower", "isolation_im_upper", "collision_energy",
]


# ── per-window ion-mobility (1/K0) bounds ────────────────────────────────────
#
# A diaPASEF window is a box in BOTH m/z and ion mobility. The scheme table gives
# the m/z bounds directly and the mobility sub-range as TIMS *scan numbers*
# (ScanNumBegin/End). Converting scan number -> 1/K0 needs the run's mobility
# axis. Rather than depend on a particular alphatims attribute name (which has
# shifted between releases), we reconstruct that axis from the detector events
# themselves: within a run, 1/K0 is a function of scan number alone, so every
# event at a given scan carries the same mobility. This is exact and vendor-
# library-independent.

def build_scan_mobility_lookup(df: pd.DataFrame):
    """Build a (scans, mobilities) lookup from a long event table.

    Returns two parallel arrays sorted by ascending scan number: the distinct
    scan numbers seen and the 1/K0 at each. Use with :func:`add_mobility_bounds`.
    A single batch of frames is enough, since the scan->1/K0 map is run-constant.
    """
    sub = df[["scan", "mobility"]].dropna()
    if len(sub) == 0:
        return np.array([], np.int64), np.array([], np.float64)
    g = sub.groupby("scan", sort=True)["mobility"].median()
    return g.index.to_numpy(np.int64), g.to_numpy(np.float64)


def _interp_mobility(lookup, scans_query) -> np.ndarray:
    scans, mob = lookup
    q = np.asarray(scans_query, dtype=np.float64)
    if len(scans) == 0:
        return np.full(q.shape, np.nan)
    # np.interp needs ascending x (scans are sorted); mobility may decrease in
    # scan, which np.interp handles. Queries outside the range clamp to the ends.
    return np.interp(q, scans.astype(np.float64), mob)


def add_mobility_bounds(scheme: pd.DataFrame, lookup) -> pd.DataFrame:
    """Add ``im_lower``/``im_upper`` (1/K0) to the window scheme using a scan->
    mobility lookup. ``im_lower``/``im_upper`` are ordered low->high regardless of
    whether 1/K0 rises or falls with scan number.
    """
    scheme = scheme.copy()
    b = _interp_mobility(lookup, scheme["scan_begin"].to_numpy())
    e = _interp_mobility(lookup, (scheme["scan_end"].to_numpy() - 1))
    scheme["im_lower"] = np.minimum(b, e)
    scheme["im_upper"] = np.maximum(b, e)
    return scheme


def assign_windows_to_long_table(
    df: pd.DataFrame,
    scheme: pd.DataFrame,
    frame_map: pd.DataFrame,
    method: str = "scan",
) -> pd.DataFrame:
    """Add ``ms_level``, ``window_id`` and window-metadata columns to a long
    detector-event table.

    The input ``df`` is one row per detector event with at least ``frame``,
    ``scan``, ``mz``, ``intensity`` (and ``mobility`` for 4D). ``rt`` is expected
    too (one value per frame). It is the renamed alphatims output.

    Decisions, all exact (no float grouping):

    * **MS level comes from the frame, not the event.** A frame is MS2 iff it is
      listed in ``DiaFrameMsMsInfo`` (``frame_map``); everything else is MS1.
      This correctly treats the rare "in an MS2 frame but isolated by no window"
      events as MS2 events that simply get dropped later, rather than mislabeling
      them MS1.
    * **Window assignment** uses scan number + the scheme (``method="scan"``,
      the general path) or quadrupole bounds (``method="quad"``, valid only when
      every window has a unique isolation m/z). MS1 events get ``window_id = -1``.

    Returns a new DataFrame (the input is not modified).
    """
    df = df.copy()
    ms2_frames = frame_map["frame"].to_numpy()
    df["ms_level"] = np.where(np.isin(df["frame"].to_numpy(), ms2_frames), 2, 1).astype("int64")

    if method == "scan":
        wid = assign_windows_by_scan(
            df["frame"].to_numpy(), df["scan"].to_numpy(), scheme, frame_map)
    elif method == "quad":
        wid = assign_windows_by_quad(
            df["quad_low_mz_values"].to_numpy(), df["quad_high_mz_values"].to_numpy(), scheme)
    else:
        raise ValueError("method must be 'scan' or 'quad'")
    # MS1 events never belong to a window, regardless of what assignment returned
    wid = np.where(df["ms_level"].to_numpy() == 1, -1, wid)
    df["window_id"] = wid.astype("int64")

    # merge the per-window metadata in by window_id (unmatched -> NaN, e.g. MS1)
    keep = ["window_id", "window_group", "isolation_mz", "isolation_width",
            "mz_lower", "mz_upper", "collision_energy"]
    ren = {"mz_lower": "isolation_mz_lower", "mz_upper": "isolation_mz_upper"}
    if "im_lower" in scheme.columns:           # mobility bounds were added
        keep += ["im_lower", "im_upper"]
        ren.update({"im_lower": "isolation_im_lower", "im_upper": "isolation_im_upper"})
    cols = scheme[keep].rename(columns=ren)
    df = df.merge(cols, on="window_id", how="left")
    return df


def group_dia_spectra(df: pd.DataFrame):
    """Group an enriched long table (output of :func:`assign_windows_to_long_table`)
    into DIA spectra, returning ``(meta, peaks)`` for :func:`from_dataframe`.

    * MS1 events -> one spectrum per frame (whole survey scan).
    * MS2 events -> one spectrum per ``(frame, window_id)``; events not assigned
      to any window (``window_id == -1``) are dropped (a count is warned).

    Each peak dict carries ``mz``, ``intensity`` and ``ion_mobility`` (4D).
    Spectra are ordered by ``(rtime, ms_level, window_id)`` and given a fresh
    ``scan_index``.
    """
    required = {"frame", "rt", "mz", "intensity", "ms_level", "window_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DIA long table missing columns: {sorted(missing)}")
    has_mobility = "mobility" in df.columns

    is_ms1 = df["ms_level"].to_numpy() == 1
    ms1 = df[is_ms1]
    ms2 = df[~is_ms1]

    n_unassigned = int((ms2["window_id"].to_numpy() == -1).sum())
    if n_unassigned:
        warnings.warn(
            f"{n_unassigned} MS2 detector event(s) fell in no isolation window "
            f"(scan outside the scheme) and were dropped.",
            stacklevel=2,
        )
    ms2 = ms2[ms2["window_id"].to_numpy() != -1]

    records = []  # (sort_key, meta_row, peak_dict)

    def _peak_dict(g):
        pk = {"mz": g["mz"].to_numpy(np.float64),
              "intensity": g["intensity"].to_numpy(np.float64)}
        if has_mobility:
            pk["ion_mobility"] = g["mobility"].to_numpy(np.float64)
        return pk

    # MS1: one spectrum per frame
    for frame, g in ms1.groupby("frame", sort=True):
        rt = float(g["rt"].iloc[0])
        meta = {"ms_level": 1, "rtime": rt, "precursor_mz": np.nan, "window_id": -1}
        for c in DIA_WINDOW_META[1:]:  # window_group, isolation_*, collision_energy
            meta[c] = np.nan
        records.append(((rt, 1, -1), meta, _peak_dict(g)))

    # MS2: one spectrum per (frame, window_id)
    for (frame, wid), g in ms2.groupby(["frame", "window_id"], sort=True):
        rt = float(g["rt"].iloc[0])
        row0 = g.iloc[0]
        meta = {
            "ms_level": 2, "rtime": rt, "precursor_mz": np.nan,
            "window_id": int(wid),
            "window_group": _as_int(row0.get("window_group")),
            "isolation_mz": _as_float(row0.get("isolation_mz")),
            "isolation_width": _as_float(row0.get("isolation_width")),
            "isolation_mz_lower": _as_float(row0.get("isolation_mz_lower")),
            "isolation_mz_upper": _as_float(row0.get("isolation_mz_upper")),
            "isolation_im_lower": _as_float(row0.get("isolation_im_lower")),
            "isolation_im_upper": _as_float(row0.get("isolation_im_upper")),
            "collision_energy": _as_float(row0.get("collision_energy")),
        }
        records.append(((rt, 2, int(wid)), meta, _peak_dict(g)))

    records.sort(key=lambda r: r[0])
    meta_rows = [r[1] for r in records]
    peaks = [r[2] for r in records]
    meta = pd.DataFrame(meta_rows)
    meta["scan_index"] = np.arange(len(meta), dtype="int64")
    return meta, peaks


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1
