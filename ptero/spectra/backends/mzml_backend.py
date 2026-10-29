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
import xml.etree.ElementTree as ET
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

# MZZARR_MZML_SEMANTIC_EXTRACTION_V1
_MZML_MOBILITY_QUANTITIES = {
    "inverse reduced ion mobility": (
        "inverse_reduced_ion_mobility", "V.s/cm2"
    ),
    "ion mobility drift time": (
        "ion_mobility_drift_time", "millisecond"
    ),
}


def _mzml_mobility_semantic(name, value):
    """Identify a mobility quantity only when its source unit is verified.

    Returns (quantity, canonical_unit), or None for unknown or
    unsupported units. No numerical conversion is performed.
    """
    name = str(name).strip().lower()
    if "inverse reduced ion mobility" in name:
        pair = _MZML_MOBILITY_QUANTITIES[
            "inverse reduced ion mobility"
        ]
        expected = (
            "volt-second per square centimeter",
            "volt second per square centimeter",
            "v.s/cm2",
            "v s/cm2",
        )
    elif "ion mobility drift time" in name:
        pair = _MZML_MOBILITY_QUANTITIES[
            "ion mobility drift time"
        ]
        expected = ("millisecond", "milliseconds", "ms")
    else:
        return None

    unit = getattr(value, "unit_info", None)
    if unit is None:
        return None

    unit = str(unit).strip().lower()
    if unit not in expected:
        return None

    return pair


def _mzml_scan_mobility_semantic(spec):
    """Return the scan-level mobility semantic, if unambiguous."""
    try:
        scan = spec["scanList"]["scan"][0]
    except (KeyError, IndexError, TypeError):
        return None

    matches = []
    for name in _MZML_MOBILITY_QUANTITIES:
        if name in scan:
            matches.append(
                _mzml_mobility_semantic(name, scan[name])
            )

    if len(matches) != 1:
        return None
    return matches[0]


def _mzml_peak_mobility_semantic(spec):
    """Return the semantic of a unique mobility binary array."""
    names = [
        key for key in spec
        if "mobility" in str(key).lower()
        and str(key).lower().endswith("array")
    ]
    if len(names) != 1:
        return None

    key = names[0]
    return _mzml_mobility_semantic(key, spec[key])


def _mzml_common_semantic(values):
    """Return one verified semantic shared by all nonempty entries.

    Unknown entries prevent a global declaration.
    """
    values = list(values)
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values) else None




# MZZARR_MZML_XML_MOBILITY_SEMANTICS_V2
_MZML_XML_MOBILITY_ACCESSIONS = {
    "MS:1003006": (
        "inverse_reduced_ion_mobility",
        "MS:1002814",
        "V.s/cm2",
    ),
    "MS:1003008": (
        "inverse_reduced_ion_mobility",
        "MS:1002814",
        "V.s/cm2",
    ),
}

_MZML_XML_MOBILITY_NAMES = {
    "mean inverse reduced ion mobility array",
    "raw inverse reduced ion mobility array",
    "ion mobility drift time array",
}


def _mzml_xml_local_name(tag):
    """Strip an XML namespace from an element tag."""
    return tag.rsplit("}", 1)[-1]


def _mzml_xml_cv_params(element, groups):
    """Resolve direct and referenced CV parameters.

    Raises ValueError for unresolved references or reference cycles.
    """
    result = []

    def visit(node, active):
        for child in node:
            name = _mzml_xml_local_name(child.tag)

            if name == "cvParam":
                result.append(child.attrib)

            elif name == "referenceableParamGroupRef":
                ref = child.attrib.get("ref")
                if not ref or ref not in groups:
                    raise ValueError(
                        f"Unresolved referenceableParamGroupRef: {ref}"
                    )
                if ref in active:
                    raise ValueError(
                        f"Cyclic referenceableParamGroupRef: {ref}"
                    )
                visit(groups[ref], active | {ref})

    visit(element, set())
    return result


def _mzml_xml_is_mobility_param(param):
    accession = param.get("accession", "")
    name = param.get("name", "").strip().lower()
    return (
        accession in _MZML_XML_MOBILITY_ACCESSIONS
        or name in _MZML_XML_MOBILITY_NAMES
        or (
            "mobility" in name
            and name.endswith("array")
        )
    )


def _mzml_xml_array_semantic(array_element, groups=None):
    """Identify one mobility array from verified CV metadata.

    Returns a (quantity, canonical_unit) pair or None.
    An unrecognized or conflicting mobility declaration is unknown.
    """
    groups = {} if groups is None else groups
    params = _mzml_xml_cv_params(array_element, groups)
    mobility = [
        p for p in params
        if _mzml_xml_is_mobility_param(p)
    ]

    if len(mobility) != 1:
        return None

    param = mobility[0]
    accession = param.get("accession", "")
    known = _MZML_XML_MOBILITY_ACCESSIONS.get(accession)

    if known is None:
        return None

    quantity, expected_unit, canonical_unit = known

    # A recognized accession with a contradictory known name
    # is not trustworthy metadata.
    name = param.get("name", "").strip().lower()
    if name in _MZML_XML_MOBILITY_NAMES:
        if (
            quantity == "inverse_reduced_ion_mobility"
            and "drift time" in name
        ):
            return None
        if (
            quantity == "ion_mobility_drift_time"
            and "inverse reduced" in name
        ):
            return None

    if param.get("unitAccession") != expected_unit:
        return None

    return quantity, canonical_unit


def _mzml_xml_peak_mobility_semantics(path):
    """Map mzML spectrum IDs to verified peak mobility semantics.

    A missing key means no mobility array was identified.
    A None value means mobility metadata exists but is ambiguous.
    """
    path = str(path)
    if not path.lower().endswith(".mzml"):
        raise ValueError("XML mobility extractor requires .mzML")

    result = {}
    groups = {}
    context = ET.iterparse(path, events=("start", "end"))
    spectrum_id = None
    arrays = []
    in_spectrum = False

    for event, element in context:
        name = _mzml_xml_local_name(element.tag)

        if event == "start":
            if name == "spectrum":
                spectrum_id = element.attrib.get("id")
                arrays = []
                in_spectrum = True
            continue

        if name == "referenceableParamGroup":
            ref = element.attrib.get("id")
            if not ref:
                raise ValueError(
                    "referenceableParamGroup without ID"
                )
            if ref in groups:
                raise ValueError(
                    f"Duplicate referenceableParamGroup: {ref}"
                )
            # Keep a detached copy: the streaming parser may
            # clear the original XML tree later.
            groups[ref] = ET.fromstring(
                ET.tostring(element, encoding="utf-8")
            )
            element.clear()

        elif name == "binaryDataArray" and in_spectrum:
            params = _mzml_xml_cv_params(element, groups)
            if any(
                _mzml_xml_is_mobility_param(p)
                for p in params
            ):
                arrays.append(
                    _mzml_xml_array_semantic(
                        element, groups
                    )
                )
            element.clear()

        elif name == "spectrum" and in_spectrum:
            if spectrum_id is None:
                raise ValueError("Spectrum without an ID")
            if spectrum_id in result:
                raise ValueError(
                    f"Duplicate spectrum ID: {spectrum_id}"
                )
            if arrays:
                result[spectrum_id] = (
                    arrays[0] if len(arrays) == 1 else None
                )
            spectrum_id = None
            arrays = []
            in_spectrum = False
            element.clear()

        elif name in (
            "binaryDataArrayList",
            "spectrumList",
            "referenceableParamGroupList",
            "run",
        ):
            element.clear()

    return result



# MZZARR_MZML_SEMANTIC_ROUTING_V1

def _mzml_array_key_quantity(key):
    """Classify a Pyteomics mobility array key by known name."""
    name = str(key).strip().lower()

    if name in (
        "mean inverse reduced ion mobility array",
        "raw inverse reduced ion mobility array",
    ):
        return "inverse_reduced_ion_mobility"

    if name == "ion mobility drift time array":
        return "ion_mobility_drift_time"

    return None


def _mzml_route_peak_semantics(
    selected_ids,
    xml_semantics,
    array_keys,
):
    """Resolve one safe peak mobility declaration.

    Parameters
    ----------
    selected_ids:
        Spectrum IDs in the selected conversion order.
    xml_semantics:
        Mapping from spectrum ID to a verified semantic pair,
        None for ambiguous mobility metadata, or absent for
        no mobility array.
    array_keys:
        Mapping from spectrum ID to the mobility array key
        actually selected by the Pyteomics backend, or None
        when no mobility array is selected.

    Returns
    -------
    tuple | None:
        One (quantity, unit) pair, or None if no selected
        spectrum has a mobility array.

    Raises
    ------
    ValueError:
        On ambiguous metadata, mismatched array identity,
        or mixed physical quantities.
    """
    selected_ids = [str(sid) for sid in selected_ids]

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(
            "Selected mzML spectrum IDs are not unique"
        )

    values = []

    for sid in selected_ids:
        key = array_keys.get(sid)
        has_xml = sid in xml_semantics

        if key is None and not has_xml:
            continue

        if key is None and has_xml:
            raise ValueError(
                f"mzML spectrum {sid!r}: XML describes a "
                "mobility array but Pyteomics selected none"
            )

        if key is not None and not has_xml:
            raise ValueError(
                f"mzML spectrum {sid!r}: Pyteomics selected "
                "a mobility array without verified XML metadata"
            )

        semantic = xml_semantics[sid]
        if semantic is None:
            raise ValueError(
                f"mzML spectrum {sid!r}: ambiguous or "
                "unsupported mobility metadata"
            )

        if (
            not isinstance(semantic, tuple)
            or len(semantic) != 2
        ):
            raise ValueError(
                f"mzML spectrum {sid!r}: invalid semantic pair"
            )

        quantity = _mzml_array_key_quantity(key)
        if quantity is None:
            raise ValueError(
                f"mzML spectrum {sid!r}: unsupported "
                f"Pyteomics mobility array key {key!r}"
            )

        if quantity != semantic[0]:
            raise ValueError(
                f"mzML spectrum {sid!r}: mobility array "
                "quantity conflicts with XML declaration"
            )

        expected_units = {
            "inverse_reduced_ion_mobility": "V.s/cm2",
            "ion_mobility_drift_time": "millisecond",
        }
        if semantic[1] != expected_units[quantity]:
            raise ValueError(
                f"mzML spectrum {sid!r}: unsupported "
                "mobility unit"
            )

        values.append(semantic)

    if not values:
        return None

    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(
            "Selected mzML spectra mix incompatible "
            "peak mobility quantities or units"
        )

    return first


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
