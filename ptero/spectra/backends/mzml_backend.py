"""
Lazy mzML backend (Python analog of R Spectra's ``MsBackendMzR``).

Keeps only spectrum-level **metadata** in memory and reads peak arrays (m/z +
intensity [+ ion mobility]) **on demand** from the original mzML file, so opening
a file that is larger than RAM is cheap and you only pay for the spectra you
actually touch.

How the laziness works (verified against pyteomics)
---------------------------------------------------
pyteomics builds a byte-offset index (``HierarchicalOffsetIndex``) mapping each
spectrum's ``id`` to its position in the file. ``reader.get_by_id(id)`` then
*seeks* to that offset and parses a single spectrum -- it does not rescan the
file. Measured: ~0.3 ms per random spectrum read, independent of file size.

At init this backend does two cheap one-time passes over the file:

1. a ``decode_binary=False`` metadata pass that records, per spectrum, its id +
   core variables (MS level, retention time, precursor, **isolation window**,
   scan-level ion mobility) WITHOUT decoding the (large) m/z and intensity
   arrays;
2. construction of a persistent, indexed reader used later for ``get_by_id``.

What it reads beyond the basic 3D case
--------------------------------------
mzML can carry ion mobility and DIA isolation windows; this backend reads both
where present, so a Thermo/Waters/Sciex run converted to mzML (e.g. via
MSConvert) keeps that information rather than silently dropping to 3D:

* **Per-peak ion mobility** -- a third binary data array alongside m/z and
  intensity (e.g. "mean inverse reduced ion mobility array"). When found, the
  backend reports ``has_4d = True`` and ``peaks_data`` returns the mobility
  column, exactly like the Bruker path. This is the encoding that maps onto
  PTERO's 4D model.
* **Scan-level ion mobility** -- a single drift time / 1/K0 per spectrum (the
  "stacked", one-spectrum-per-mobility-bin encoding). This is recorded as the
  spectrum-level ``ion_mobility`` metadata column; it is NOT treated as 4D,
  because the peaks themselves carry no per-peak mobility. (Restructuring that
  encoding into per-peak 4D is a separate, larger job and is not done here.)
* **DIA isolation windows** -- each MS2 spectrum's ``isolationWindow`` (target
  m/z + lower/upper offsets) becomes ``isolation_mz`` / ``isolation_mz_lower`` /
  ``isolation_mz_upper`` / ``isolation_width``, matching the schema produced by
  :mod:`ptero.spectra._dia`. Because mzML does not enumerate the windows in a
  scheme table (unlike Bruker's ``analysis.tdf``), a ``window_id`` /
  ``window_group`` is synthesised by grouping spectra with identical isolation
  bounds, so the dashboard's window-scheme view populates for mzML DIA too.

Note that ordinary (non-mobility, DDA) mzML reads exactly as before: no mobility
array -> ``has_4d = False``; no isolation window -> no extra columns.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._base import MsBackend, DEFAULT_PEAKS_VARIABLES


def _require_pyteomics():
    try:
        from pyteomics import mzml, mzxml  # noqa
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MsBackendMzML requires `pyteomics` (and `psims` for mzML). "
            "Install with: pip install pyteomics psims"
        ) from e


# ── per-spectrum metadata extraction (shared with io.from_mzml) ──────────────

def _scan_start_time(spec) -> float:
    """Retention time in seconds (mzML may store it in minutes)."""
    try:
        scan = spec["scanList"]["scan"][0]
        if "scan start time" in scan:
            t = scan["scan start time"]
            unit = getattr(t, "unit_info", "")
            val = float(t)
            return val * 60.0 if unit and "minute" in str(unit).lower() else val
    except Exception:
        pass
    return float(spec["scan start time"]) if "scan start time" in spec else np.nan


def _scan_mobility(spec) -> float:
    """Scan-level ion mobility (1/K0 or drift time), or NaN.

    This is the single value attached to a whole spectrum in the 'stacked'
    IMS encoding (one spectrum per mobility bin), as opposed to a per-peak
    mobility array.
    """
    try:
        scan = spec["scanList"]["scan"][0]
        for key in ("inverse reduced ion mobility", "ion mobility drift time"):
            if key in scan:
                return float(scan[key])
    except Exception:
        pass
    return np.nan


def _spectrum_mobility(spec) -> float:
    """Per-spectrum 1/K0 for an ion-mobility spectrum that carries its mobility
    *per peak* rather than as a scan-level scalar (e.g. a timsTOF run converted
    with "Combine ion mobility scans": each MS/MS is one summed spectrum whose
    peaks span a narrow 1/K0 window).

    Such spectra expose ``ion mobility lower limit`` / ``ion mobility upper
    limit`` cvParams (available in the metadata pass, no binary decoding). Their
    midpoint is the precursor's representative 1/K0 -- the value the ID linker
    uses to separate co-eluting precursors. Returns NaN when no limits are
    present (an ordinary 3D mzML), leaving the spectrum 3D as before.
    """
    lo = spec.get("ion mobility lower limit")
    hi = spec.get("ion mobility upper limit")
    if lo is not None and hi is not None:
        try:
            return (float(lo) + float(hi)) / 2.0
        except (TypeError, ValueError):
            return np.nan
    return np.nan


def _precursor(spec) -> dict:
    out = {}
    try:
        pre = spec["precursorList"]["precursor"][0]
        ion = pre["selectedIonList"]["selectedIon"][0]
        if "selected ion m/z" in ion:
            out["mz"] = float(ion["selected ion m/z"])
        if "charge state" in ion:
            out["charge"] = int(ion["charge state"])
        if "peak intensity" in ion:
            out["intensity"] = float(ion["peak intensity"])
    except Exception:
        pass
    return out


def _isolation_window(spec) -> dict:
    """DIA isolation window as target m/z + lower/upper offsets, if present.

    Returns ``isolation_mz`` (centre), ``isolation_mz_lower``/``upper`` (edges)
    and ``isolation_width`` (Da). Empty when the spectrum has no isolation
    window (e.g. MS1, or a DDA file without the cvParams).
    """
    out = {}
    try:
        iw = spec["precursorList"]["precursor"][0].get("isolationWindow", {})
        tgt = iw.get("isolation window target m/z")
        if tgt is None:
            return out
        tgt = float(tgt)
        out["isolation_mz"] = tgt
        lo = iw.get("isolation window lower offset")
        hi = iw.get("isolation window upper offset")
        if lo is not None and hi is not None:
            lo = float(lo); hi = float(hi)
            out["isolation_mz_lower"] = tgt - lo
            out["isolation_mz_upper"] = tgt + hi
            out["isolation_width"] = lo + hi
    except Exception:
        pass
    return out


# Names (not accessions) of binary arrays that hold per-peak ion mobility. We
# match by name so the several ontology variants (mean/raw inverse-reduced-IM,
# drift-time arrays) are all caught without hard-coding accession numbers.
def _find_mobility_array_key(spec) -> Optional[str]:
    for k in spec.keys():
        kl = str(k).lower()
        if "mobility" in kl and kl.endswith("array"):
            return k
    return None


def _meta_row(spec, index: int, origin: str) -> dict:
    lvl = int(spec.get("ms level", spec.get("msLevel", 1)))
    prec = _precursor(spec)
    # Per-spectrum 1/K0: prefer a scan-level scalar (stacked-IMS encoding); if
    # absent, fall back to the midpoint of the spectrum's ion-mobility limits
    # (combined-mobility timsTOF mzML, where mobility is otherwise per-peak).
    mob = _scan_mobility(spec)
    if mob != mob:                                # NaN -> try IM-limit midpoint
        mob = _spectrum_mobility(spec)
    native_id = str(spec.get("id", ""))
    scan_match = re.search(r"(?:^|\s)scan=(\d+)(?=\s|$)", native_id)
    acquisition_num = int(scan_match.group(1)) if scan_match else pd.NA
    row = {
        "native_spectrum_id": native_id,
        "acquisition_num": acquisition_num,
        "ms_level": lvl,
        "rtime": _scan_start_time(spec),
        "scan_index": index,
        "data_origin": origin,
        "data_storage": origin,
        "centroided": ("centroid spectrum" in spec) or ("profile spectrum" not in spec),
        "precursor_mz": prec.get("mz", np.nan),
        "precursor_charge": prec.get("charge", pd.NA),
        "precursor_intensity": prec.get("intensity", np.nan),
        "ion_mobility": mob,                       # scan-level, else IM-limit midpoint, else NaN
    }
    row.update(_isolation_window(spec))           # isolation_mz[_lower/_upper], width
    return row


# ── the backend ──────────────────────────────────────────────────────────────

class MsBackendMzML(MsBackend):
    """Read-only, lazy backend over an mzML (or mzXML) file."""

    read_only = True        # the file's peaks are not modifiable in place
    has_4d = False          # set True at init if a per-peak mobility array is found

    def __init__(self, path, _idx=None, _reader=None, _meta=None, _ids=None,
                 _has_4d=None):
        self.path = str(path)
        if _reader is not None:
            # subset/view: share the open reader, metadata, and id list
            self._reader = _reader
            self._meta = _meta
            self._ids = _ids
            self.has_4d = bool(_has_4d)
        else:
            _require_pyteomics()
            from pyteomics import mzml, mzxml

            is_xml = self.path.lower().endswith(".mzxml")
            # (1) persistent, indexed reader for on-demand get_by_id (seeks)
            self._reader = (mzxml.MzXML if is_xml else mzml.MzML)(self.path)
            # (2) one metadata-only pass (does NOT decode m/z / intensity arrays)
            read = (mzxml.read if is_xml else mzml.read)
            rows, ids = [], []
            for i, spec in enumerate(read(self.path, decode_binary=False)):
                rows.append(_meta_row(spec, i, self.path))
                ids.append(spec["id"])
            self._meta = pd.DataFrame(rows)
            self._ids = np.array(ids, dtype=object)
            self._synthesize_windows()
            self.has_4d = self._detect_per_peak_mobility()

        n = len(self._ids)
        self._idx = np.arange(n, dtype=np.int64) if _idx is None else np.asarray(_idx, np.int64)

    # ── one-time post-processing of the metadata pass ────────────────────
    def _synthesize_windows(self) -> None:
        """Assign window_id / window_group to DIA spectra by grouping identical
        isolation bounds (mzML does not enumerate windows in a scheme table)."""
        m = self._meta
        if "isolation_mz_lower" not in m.columns or not m["isolation_mz_lower"].notna().any():
            return
        has_win = m["isolation_mz_lower"].notna().to_numpy()
        key = (m["isolation_mz_lower"].round(4).astype("string") + "_"
               + m["isolation_mz_upper"].round(4).astype("string"))
        codes = pd.factorize(key.where(has_win))[0]   # NaN (non-window) -> -1
        m["window_id"] = codes.astype("int64")
        m["window_group"] = np.where(codes >= 0, codes + 1, np.nan)

    def _detect_per_peak_mobility(self) -> bool:
        """Decode the first spectrum once to see if peaks carry a mobility array."""
        if len(self._ids) == 0:
            return False
        try:
            s0 = self._reader.get_by_id(str(self._ids[0]))
            return _find_mobility_array_key(s0) is not None
        except Exception:
            return False

    # ── required interface ───────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._idx)

    def spectra_data(self, columns=None) -> pd.DataFrame:
        df = self._ensure_core_columns(self._meta).iloc[self._idx].reset_index(drop=True)
        return df if columns is None else df[list(columns)]

    def peaks_data(self, indices=None, columns=DEFAULT_PEAKS_VARIABLES):
        view = self._normalize_indices(indices, len(self))
        ids = self._ids[self._idx[view]]            # view position -> storage row -> spectrum id
        cols = list(columns)
        want_im = "ion_mobility" in cols
        if want_im and not self.has_4d:
            # match the memory/zarr backends: a 3D file has no per-peak mobility
            raise KeyError("ion_mobility not available (3D data)")
        out = []
        for sid in ids:
            spec = self._reader.get_by_id(str(sid))  # SEEK + parse one spectrum
            mz = np.asarray(spec.get("m/z array", []), dtype=np.float64)
            inten = np.asarray(spec.get("intensity array", []), dtype=np.float64)
            mob = None
            if want_im:
                mk = _find_mobility_array_key(spec)
                # has_4d is True here; a stray spectrum lacking the array -> NaN
                mob = (np.asarray(spec[mk], dtype=np.float64) if mk is not None
                       else np.full(len(mz), np.nan, dtype=np.float64))
            arrs = []
            for c in cols:
                if c == "mz":
                    arrs.append(mz)
                elif c == "intensity":
                    arrs.append(inten)
                elif c == "ion_mobility":
                    arrs.append(mob)
                else:
                    raise KeyError(f"unknown peaks variable {c!r}")
            out.append(np.column_stack(arrs) if arrs else np.empty((0, len(cols))))
        return out

    def subset(self, indices: Sequence[int]) -> "MsBackendMzML":
        idx = self._idx[np.asarray(indices, dtype=np.int64)]
        return MsBackendMzML(self.path, _idx=idx, _reader=self._reader,
                             _meta=self._meta, _ids=self._ids, _has_4d=self.has_4d)

    # ── resource management ──────────────────────────────────────────────
    def close(self):
        """Close the open file handle held by the reader."""
        try:
            self._reader.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        kind = "4D" if self.has_4d else "3D"
        return f"MsBackendMzML: {len(self)} spectra (lazy, read_only, {kind}) <- {self.path}"
