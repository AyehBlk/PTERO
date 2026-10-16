"""
Bruker timsTOF (``.d``) support for PTERO.

Design (read this before editing)
----------------------------------
Reading a Bruker ``.d`` folder requires a vendor library that only ships for
Windows and Linux, and the Python readers for it (alphatims / alpharaw) expose
an API that has been shifting between releases. To keep PTERO correct and
testable despite that, the logic is split in two:

1. :func:`_timstof_to_spectra` / :func:`from_timstof_dataframe` -- the **tested
   core**. They consume a plain, well-defined long table (one row per detector
   event) and build 4D spectra. No vendor library involved, so this is unit
   tested with synthetic timsTOF-shaped data and is the stable contract. For
   DIA the data-shaping core lives in :mod:`ptero.spectra._dia` (also tested).

2. :func:`from_bruker_d` -- a **thin adapter** that opens a ``.d`` with the
   vendor reader, produces that long table, and hands it to the core. This part
   cannot be tested without a real ``.d`` + Bruker library, so validate it on
   Windows with a real file (see the package docs / validation snippet).

DDA vs DIA frame -> spectrum mapping
------------------------------------
* **ddaPASEF**: each MS1 frame -> one 4D spectrum; PASEF MS2 frames hold several
  precursor windows, grouped by (frame, precursor) -- one spectrum per precursor.
* **diaPASEF**: each MS1 frame -> one 4D spectrum; each MS2 frame steps the
  quadrupole through a fixed set of wide isolation **windows** as the mobility
  ramp proceeds, so MS2 is grouped by (frame, window) -- one spectrum per window.
  The window scheme is authoritative metadata read from the ``analysis.tdf``
  SQLite tables (see :mod:`ptero.spectra._dia`); each MS2 event is assigned to a
  window by exact (frame -> window group, scan number -> sub-window) lookup.

Required long-table columns (the DDA contract)
----------------------------------------------
``frame``      int   -- groups detector events into spectra
``rt``         float -- retention time in seconds (one value per frame)
``mobility``   float -- 1/K0 per detector event (becomes per-peak ion_mobility)
``mz``         float -- per detector event
``intensity``  float -- per detector event
Optional: ``ms_level`` (default 1), ``precursor_mz`` (NaN for MS1).
For DIA, a ``window_id`` column (added by :func:`_dia.assign_windows_to_long_table`)
switches :func:`from_timstof_dataframe` to the DIA grouping path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._spectra import PteroSpectra
from .io import from_dataframe
from . import _dia


# ── tested core (DDA) ─────────────────────────────────────────────────────────

_BRUKER_ACQUISITION_FIELDS = (
    "bruker_frame_id",
    "bruker_precursor_id",
    "bruker_scan_begin",
    "bruker_scan_end",
)


def _bruker_group_metadata(group: pd.DataFrame) -> dict:
    """Preserve optional acquisition IDs only when constant within a group."""
    extra = {}

    for column in _BRUKER_ACQUISITION_FIELDS:
        if column not in group.columns:
            continue

        values = group[column].drop_duplicates()

        if len(values) != 1:
            raise ValueError(
                f"Non-unique {column} within one Bruker spectrum group"
            )

        extra[column] = values.iloc[0]

    return extra


def _load_bruker_pasef_events(path) -> pd.DataFrame:
    """Read authoritative ddaPASEF event identifiers from analysis.tdf."""
    from pathlib import Path
    import sqlite3

    database = Path(path) / "analysis.tdf"
    if not database.is_file():
        raise FileNotFoundError(database)

    uri = database.resolve().as_uri() + "?mode=ro"

    with sqlite3.connect(uri, uri=True) as connection:
        events = pd.read_sql_query(
            """
            SELECT
                Frame AS bruker_frame_id,
                Precursor AS bruker_precursor_id,
                ScanNumBegin AS bruker_scan_begin,
                ScanNumEnd AS bruker_scan_end,
                IsolationMz AS isolation_mz
            FROM PasefFrameMsMsInfo
            """,
            connection,
        )

    if events.empty:
        raise ValueError("No Bruker PASEF acquisition events found")

    identity = [
        "bruker_frame_id",
        "bruker_scan_begin",
        "bruker_scan_end",
    ]

    if events.duplicated(identity).any():
        raise ValueError("Duplicate Bruker PASEF acquisition intervals")

    return events


def _attach_bruker_acquisition_ids(
    df: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Assign original Bruker IDs to detector events without guessing.

    Each MS2 detector event must map to exactly one original PASEF
    acquisition event using its frame, scan position, and isolation m/z.
    MS1 detector events retain their frame ID and nullable MS2 fields.
    """
    required = {"frame", "scan", "ms_level", "precursor_mz"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Bruker detector table lacks acquisition fields: "
            f"{sorted(missing)}"
        )

    output = df.copy()
    output["bruker_frame_id"] = output["frame"].astype("int64")

    for field in (
        "bruker_precursor_id",
        "bruker_scan_begin",
        "bruker_scan_end",
    ):
        output[field] = pd.Series(
            pd.array([pd.NA] * len(output), dtype="Int64"),
            index=output.index,
        )

    if output.empty:
        return output

    ms2 = output["ms_level"].to_numpy() == 2

    if not ms2.any():
        return output

    # Group by original frame and isolation center to avoid a costly
    # full event-table comparison for every detector peak.
    subset = output.loc[ms2]

    for (frame, center), group in subset.groupby(
        ["frame", "precursor_mz"], sort=False
    ):
        candidates = events.loc[
            (events["bruker_frame_id"] == int(frame))
            & np.isclose(
                events["isolation_mz"].to_numpy(dtype=float),
                float(center),
                rtol=0,
                atol=1e-4,
            )
        ]

        scans = group["scan"].to_numpy(dtype=np.int64)
        assignments = np.full(len(group), -1, dtype=np.int64)

        for event_index, event in candidates.iterrows():
            inside = (
                (scans >= int(event["bruker_scan_begin"]))
                & (scans < int(event["bruker_scan_end"]))
            )

            if (assignments[inside] != -1).any():
                raise ValueError(
                    f"Ambiguous Bruker event: frame={frame}, "
                    f"precursor_mz={center}"
                )

            assignments[inside] = int(event_index)

        if (assignments < 0).any():
            raise ValueError(
                f"Unmatched Bruker detector events: frame={frame}, "
                f"precursor_mz={center}, count={(assignments < 0).sum()}"
            )

        matched = events.loc[assignments]

        for field in (
            "bruker_precursor_id",
            "bruker_scan_begin",
            "bruker_scan_end",
        ):
            output.loc[group.index, field] = (
                matched[field].to_numpy(dtype=np.int64)
            )

    return output


def _normalize_bruker_id_dtypes(metadata: pd.DataFrame) -> pd.DataFrame:
    """Keep original Bruker acquisition identifiers as nullable integers."""
    for column in _BRUKER_ACQUISITION_FIELDS:
        if column in metadata.columns:
            metadata[column] = metadata[column].astype("Int64")
    return metadata


def _timstof_to_spectra_pandas(df: pd.DataFrame):
    """Turn a long timsTOF detector-event table into (meta, peaks).

    Groups rows into spectra by ``frame`` (and ``precursor_mz`` when present, so
    PASEF MS2 windows become separate spectra). Returns a ``(meta, peaks)`` pair
    suitable for :func:`from_dataframe`: each peak dict carries ``ion_mobility``,
    making every spectrum 4D.
    """
    required = {"frame", "rt", "mobility", "mz", "intensity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"timsTOF table is missing required columns: {sorted(missing)}")

    has_prec = "precursor_mz" in df.columns
    has_lvl = "ms_level" in df.columns
    group_keys = ["frame", "precursor_mz"] if has_prec else ["frame"]

    # stable order: by frame, then precursor (NaN precursors -> MS1 -> sort first)
    df = df.sort_values(group_keys, kind="stable")

    meta_rows, peaks = [], []
    scan_index = 0
    for keys, g in df.groupby(group_keys, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        prec = keys[1] if has_prec else np.nan
        if has_lvl:
            lvl = int(g["ms_level"].iloc[0])
        else:
            lvl = 1 if (not has_prec or pd.isna(prec)) else 2
        mob = g["mobility"].to_numpy(dtype=np.float64)
        meta_rows.append({
            "ms_level": lvl,
            "rtime": float(g["rt"].iloc[0]),
            "scan_index": scan_index,
            "precursor_mz": float(prec) if (has_prec and not pd.isna(prec)) else np.nan,
            "ion_mobility": float(np.median(mob)) if mob.size else np.nan,
            **_bruker_group_metadata(g),
        })
        peaks.append({
            "mz": g["mz"].to_numpy(dtype=np.float64),
            "intensity": g["intensity"].to_numpy(dtype=np.float64),
            "ion_mobility": g["mobility"].to_numpy(dtype=np.float64),
        })
        scan_index += 1

    return _normalize_bruker_id_dtypes(pd.DataFrame(meta_rows)), peaks

def _timstof_to_spectra(df: pd.DataFrame):
    """Group DDA timsTOF events into spectra using contiguous NumPy slices."""
    required = {"frame", "rt", "mobility", "mz", "intensity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"timsTOF table is missing required columns: {sorted(missing)}")

    has_prec = "precursor_mz" in df.columns
    has_lvl = "ms_level" in df.columns
    group_keys = ["frame", "precursor_mz"] if has_prec else ["frame"]

    # Preserve pandas' established group ordering and stable peak ordering.
    s = df.sort_values(group_keys, kind="stable")

    if s.empty:
        return pd.DataFrame(), []

    # Keep unusual group-key dtypes on the original, tested pandas path.
    if not pd.api.types.is_numeric_dtype(s["frame"].dtype):
        return _timstof_to_spectra_pandas(df)
    if has_prec and not pd.api.types.is_numeric_dtype(s["precursor_mz"].dtype):
        return _timstof_to_spectra_pandas(df)

    frame = s["frame"].to_numpy()
    prec = (
        s["precursor_mz"].to_numpy()
        if has_prec else np.full(len(s), np.nan)
    )

    # Nullable extension arrays may not support vectorized comparisons.
    try:
        same_prec = (
            (prec[1:] == prec[:-1])
            | (pd.isna(prec[1:]) & pd.isna(prec[:-1]))
        )
        changes = (frame[1:] != frame[:-1]) | ~same_prec
        boundaries = np.r_[
            0, np.flatnonzero(changes) + 1, len(s)
        ]
    except (TypeError, ValueError):
        return _timstof_to_spectra_pandas(df)

    rt = s["rt"].to_numpy()
    mz = s["mz"].to_numpy(dtype=np.float64)
    intensity = s["intensity"].to_numpy(dtype=np.float64)
    mobility = s["mobility"].to_numpy(dtype=np.float64)
    levels = s["ms_level"].to_numpy() if has_lvl else None

    meta_rows, peaks = [], []

    for i, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        p = prec[start]
        mob = mobility[start:end]

        if has_lvl:
            lvl = int(levels[start])
        else:
            lvl = 1 if (not has_prec or pd.isna(p)) else 2

        meta_rows.append({
            "ms_level": lvl,
            "rtime": float(rt[start]),
            "scan_index": i,
            "precursor_mz": (
                float(p) if has_prec and not pd.isna(p) else np.nan
            ),
            "ion_mobility": float(np.median(mob)) if mob.size else np.nan,
            **_bruker_group_metadata(s.iloc[start:end]),
        })

        peaks.append({
            "mz": mz[start:end],
            "intensity": intensity[start:end],
            "ion_mobility": mob,
        })

    return _normalize_bruker_id_dtypes(pd.DataFrame(meta_rows)), peaks



def from_timstof_dataframe(df: pd.DataFrame, ms_level: Optional[int] = None) -> PteroSpectra:
    """Build a 4D :class:`PteroSpectra` from a long timsTOF detector-event table.

    Vendor-independent entry point (the data-shaping cores are fully tested).
    If the table carries a ``window_id`` column it is treated as **diaPASEF** and
    grouped per (frame, window) by :func:`ptero.spectra._dia.group_dia_spectra`;
    otherwise it follows the DDA grouping in :func:`_timstof_to_spectra`.
    """
    if "window_id" in df.columns:
        meta, peaks = _dia.group_dia_spectra(df)
    else:
        meta, peaks = _timstof_to_spectra(df)
    if len(meta) == 0:
        raise ValueError(
            "No detector events in the selected data — nothing to build. "
            "If you passed a frame range, check it is within the run."
        )
    sp = from_dataframe(meta, peaks)
    if ms_level is not None:
        sp = sp.filter_ms_level(ms_level)
    return sp


def _bruker_slice(ms_level: int, frames=None):
    """Build the alphatims index tuple ``(frame_sel, scan_sel, precursor_sel)``.

    Precursor index 0 == MS1 (unfragmented); indices >= 1 == PASEF MS2 windows.
    ``frames`` is an optional ``(start, stop)`` frame range (stop exclusive).
    Isolated so the slice logic is unit-testable without a real ``.d`` file.
    """
    frame_sel = slice(None) if frames is None else slice(int(frames[0]), int(frames[1]))
    if ms_level == 1:
        prec_sel = 0
    elif ms_level == 2:
        prec_sel = slice(1, None)
    else:
        prec_sel = slice(None)
    return (frame_sel, slice(None), prec_sel)


# ── thin vendor adapter (validate on Windows with a real .d) ──────────────────

def _import_timstof_reader():
    """Return a TimsTOF reader class, or raise a clear, actionable error."""
    try:
        from alphatims.bruker import TimsTOF  # canonical reader with slicing API
        return TimsTOF
    except Exception:
        pass
    try:
        from alpharaw.bruker.timstof import TimsTOFBase
        return TimsTOFBase
    except Exception as e:
        raise ImportError(
            "Reading Bruker .d needs a timsTOF reader. Install one with:\n"
            "    pip install alphatims    (recommended; provides data[...] slicing)\n"
            "or  pip install alpharaw\n"
            "Note: reading m/z and mobility from .d requires Windows or Linux "
            "(Bruker libraries are unavailable on macOS, where values are only estimated)."
        ) from e


def _resolve_bruker_dir(path):
    """Return the directory that actually contains the Bruker metadata DB.

    ``.d`` folders are sometimes **nested** -- an outer ``<name>.d`` holding an
    inner ``<name>.d`` that has the real data -- which is common after copying or
    unzipping a run. If the given folder has no ``analysis.tdf`` directly, look one
    level down for the child folder that does, so callers may pass either the outer
    or the inner path and it just works.
    """
    from pathlib import Path
    p = Path(str(path))

    def has_meta(d):
        return (d / "analysis.tdf").exists() or (d / "analysis.tsf").exists()

    if has_meta(p) or not p.is_dir():
        return str(p)
    try:
        children = sorted(c for c in p.iterdir() if c.is_dir())
    except OSError:
        return str(p)
    for c in children:                       # any direct subfolder that holds it
        if has_meta(c):
            return str(c)
    return str(p)                            # not found -> let _open_timstof raise



def _alpharaw_framezero_reader(original):
    """Correct AlphaRaw's synthetic frame zero under pandas Copy-on-Write."""
    def compatible(*args, **kwargs):
        import warnings
        import pandas as pd

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=pd.errors.ChainedAssignmentError
            )
            result = original(*args, **kwargs)

        add_zero = (
            args[1] if len(args) > 1
            else kwargs.get("add_zeroth_frame", True)
        )
        if not add_zero:
            return result

        parts = list(result)
        frames = parts[2].copy()
        fields = (
            "Id", "Time", "MaxIntensity",
            "SummedIntensities", "NumPeaks", "MsMsType",
        )
        frames.loc[0, list(fields)] = 0
        parts[2] = frames
        return tuple(parts)

    return compatible


def _open_timstof(path, mz_estimation_from_frame: int = 1,
                  mobility_estimation_from_frame: int = 1):
    """Open a ``.d`` once and return the vendor reader object (the slow step:
    indexing happens here, so callers that read many slices should open once)."""
    from pathlib import Path
    path = _resolve_bruker_dir(path)
    if not ((Path(path) / "analysis.tdf").exists() or (Path(path) / "analysis.tsf").exists()):
        raise FileNotFoundError(
            f"No Bruker metadata file (analysis.tdf) was found in {path!r}. "
            "If the .d folder is nested (a .d inside a .d), point at the folder that "
            "directly contains analysis.tdf. If it was copied, check the copy finished "
            "and the file is not 0 bytes."
        )
    TimsTOF = _import_timstof_reader()

    # AlphaRaw uses chained assignments for its synthetic frame zero.
    # Pandas 3 Copy-on-Write ignores those assignments, corrupting
    # the frame-zero detector-event count and downstream offsets.
    namespace = TimsTOF._import_data_from_d_folder.__globals__
    original_reader = namespace["read_bruker_sql"]
    namespace["read_bruker_sql"] = _alpharaw_framezero_reader(original_reader)
    try:
        try:
            data = TimsTOF(
                str(path),
                mz_estimation_from_frame=mz_estimation_from_frame,
                mobility_estimation_from_frame=mobility_estimation_from_frame,
            )
        except TypeError:
            data = TimsTOF(str(path))  # readers with a simpler signature
    finally:
        namespace["read_bruker_sql"] = original_reader
    if not hasattr(data, "__getitem__"):
        raise RuntimeError(
            "The installed timsTOF reader does not expose the data[...] -> DataFrame "
            "slicing API this adapter needs. Install the `alphatims` package "
            "(`pip install alphatims`) and retry, or build the long table yourself "
            "and call ptero.spectra.from_timstof_dataframe(df)."
        )
    return data


def _timstof_n_frames(data) -> Optional[int]:
    """Best-effort total frame count (a *stop-exclusive* bound), or ``None``."""
    v = getattr(data, "frame_max_index", None)
    if v is not None:
        try:
            return int(v) + 1          # frames are 1..frame_max_index inclusive
        except Exception:
            pass
    frames_tbl = getattr(data, "frames", None)
    if frames_tbl is not None:
        try:
            return int(len(frames_tbl)) + 1
        except Exception:
            pass
    return None


def _read_timstof_frames(data, ms_level: Optional[int], frames=None) -> pd.DataFrame:
    """Slice an already-open reader into a standardized long detector-event table.

    ``ms_level`` is 1 (MS1), 2 (PASEF MS2), or ``None`` to read **both** levels
    in one pass. Returns an *empty* DataFrame (no exception) when the slice has
    no events, so a batch walker can detect the end of the run cleanly.
    """
    # alphatims convention: precursor index 0 == MS1 (unfragmented)
    df = data[_bruker_slice(ms_level, frames)]
    df = df.rename(columns={
        "frame_indices": "frame",
        "scan_indices": "scan",        # TIMS scan number -> window assignment (DIA)
        "rt_values": "rt",
        "mobility_values": "mobility",
        "mz_values": "mz",
        "intensity_values": "intensity",
    })
    # precursor m/z from the quadrupole isolation window (NaN for MS1 survey scans)
    if "precursor_mz" not in df.columns and {"quad_low_mz_values", "quad_high_mz_values"} <= set(df.columns):
        center = (df["quad_low_mz_values"] + df["quad_high_mz_values"]) / 2.0
        df["precursor_mz"] = center.where(df["quad_low_mz_values"] >= 0, np.nan)
    # MS level: forced when a single level was requested; otherwise derived PER
    # EVENT so a combined read labels each detector event correctly. An event is
    # MS2 iff it falls inside a quadrupole isolation window (has a precursor m/z);
    # MS1 survey events have none. (For DIA this per-event guess is later replaced
    # by the exact frame-based level in _dia.assign_windows_to_long_table.)
    if ms_level in (1, 2):
        df["ms_level"] = ms_level
    elif "precursor_mz" in df.columns:
        pm = df["precursor_mz"].to_numpy(dtype="float64")
        df["ms_level"] = np.where(np.isnan(pm), 1, 2).astype("int64")
    elif "precursor_indices" in df.columns:
        df["ms_level"] = np.where(df["precursor_indices"].to_numpy() == 0, 1, 2).astype("int64")
    else:
        df["ms_level"] = 1
    return df


def from_bruker_d(
    path,
    ms_level: int = 1,
    frames: Optional[tuple] = None,
    mz_estimation_from_frame: int = 1,
    mobility_estimation_from_frame: int = 1,
    mode: str = "auto",
) -> PteroSpectra:
    """Read a Bruker timsTOF ``.d`` folder into a 4D :class:`PteroSpectra`.

    .. important::
       This adapter opens the ``.d`` with the vendor reader and is **validated
       on Windows/Linux with a real file**, not in PTERO's test suite. The
       data-shaping logic it calls (:func:`from_timstof_dataframe` and, for DIA,
       :mod:`ptero.spectra._dia`) *is* tested.

    Memory note: without ``frames`` this pulls every selected detector event into
    one DataFrame, which can be several GB for a long run. Use ``frames`` to read
    a slice, or use :func:`ptero.dashboard.convert_to_zarr`, which walks the run
    in frame batches and never holds it all in RAM.

    Parameters
    ----------
    path : str | Path
    ms_level : int, default 1
        1 reads MS1 frames; 2 reads MS2; any other value reads both. For DIA this
        is applied as a metadata filter *after* windows are assigned, so MS2
        spectra keep their window metadata.
    frames : (start, stop), optional
        Frame range to read (stop exclusive). ``None`` reads the whole run.
    mode : {'auto', 'dda', 'dia'}
        ``'auto'`` detects the acquisition mode from the ``analysis.tdf`` tables
        (:func:`ptero.spectra._dia.detect_acquisition_mode`). Override only to
        force a path.
    mz_estimation_from_frame, mobility_estimation_from_frame : int
        Passed to the reader; control calibrated vs estimated m/z and mobility.
    """
    data = _open_timstof(path, mz_estimation_from_frame, mobility_estimation_from_frame)

    resolved = mode
    if resolved == "auto":
        try:
            resolved = _dia.detect_acquisition_mode(path)
        except Exception:
            resolved = "dda"

    if resolved == "dia":
        scheme = _dia.read_window_scheme(path)
        frame_map = _dia.read_frame_window_map(path)
        # read both levels in one pass; window assignment needs MS1+MS2 together
        df = _read_timstof_frames(data, None, frames)
        df = _dia.assign_windows_to_long_table(df, scheme, frame_map, method="scan")
        keep = ms_level if ms_level in (1, 2) else None
        return from_timstof_dataframe(df, ms_level=keep)

    # ddaPASEF (and anything not detected as DIA) -- unchanged behaviour
    df = _read_timstof_frames(data, ms_level, frames)
    return from_timstof_dataframe(df, ms_level=None)
