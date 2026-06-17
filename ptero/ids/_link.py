"""
ptero.ids._link -- locate search-engine PSMs in the raw spectra store, and
annotate the matched MS2 spectrum.

The link is PROPERTY-BASED, not by scan number (MaxQuant accumulates several
PASEF events into one searched spectrum, so there is no 1:1 scan key to PTERO's
per-(frame, precursor) spectra). A PSM is located in a PTERO raw store by
physical coordinates:

* retention time   -- MaxQuant reports MINUTES; the store is in SECONDS (x60).
* precursor m/z     -- the store holds the isolation-WINDOW CENTRE, MaxQuant the
                      monoisotopic m/z, so the tolerance must be WIDE (~1.5 Th).
* ion mobility 1/K0 -- the disambiguator for co-eluting precursors that RT + m/z
                      alone cannot separate. OPTIONAL: it is NOT in msms.txt;
                      attach it from accumulatedMsmsScans.txt with
                      :func:`attach_mobility_from_maxquant` first.

When more than one candidate survives (e.g. mobility unavailable), candidates are
ranked by MS2 FRAGMENT match (b/y ions via spectrum_utils) -- which both
disambiguates and produces the annotated-spectrum object for the flagship plot.

This contract was verified on real timsTOF data; the concept is encoded in
``tests/test_link.py`` and these functions in ``tests/test_ids.py``.

Design note: the PteroSpectra import is LAZY (inside :class:`RunRegistry`), so
this module imports without the ``raw`` extra installed -- only the matcher,
which actually reads a store, needs PTERO's spectra layer. psm_utils and
spectrum_utils (the ``ids`` extra) are hard imports: the module is useless
without them.
"""
from __future__ import annotations

import csv
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from psm_utils import PSM, PSMList
import psm_utils.io as psm_io
from spectrum_utils.spectrum import MsmsSpectrum
from spectrum_utils import proforma as _sus_proforma
from spectrum_utils import fragment_annotation as _sus_fragment


# ── reading PSMs ─────────────────────────────────────────────────────────────

def _strip_bom_to_temp(path: Path) -> Optional[Path]:
    """If *path* begins with a UTF-8 BOM, write a BOM-free copy to a temp file and
    return its path; otherwise return ``None``.

    psm_utils' MaxQuant reader opens the file with a plain ``open()`` and does an
    exact (case-folded) column check, so a leading BOM makes the first column
    (``Raw file``) unrecognised and the read fails with
    ``Missing columns: ['Raw file']``. MaxQuant on Windows writes that BOM.
    """
    with open(path, "rb") as fh:
        head = fh.read(3)
    if head != b"\xef\xbb\xbf":
        return None
    text = path.read_text(encoding="utf-8-sig")          # decodes and drops the BOM
    tmp = Path(tempfile.gettempdir()) / f"ptero_nobom_{path.name}"
    tmp.write_text(text, encoding="utf-8")
    return tmp


def read_psms(path, filetype: str = "msms") -> PSMList:
    """Read a search-engine result file into a psm_utils :class:`PSMList`.

    Transparently strips a leading UTF-8 BOM (common in MaxQuant output on
    Windows) that would otherwise make psm_utils' MaxQuant reader fail.

    ``filetype`` is passed to :func:`psm_utils.io.read_file` (``'msms'`` for
    MaxQuant msms.txt; ``'diann'``, ``'sage'``, ``'fragpipe'``, ... otherwise).

    Notes
    -----
    For MaxQuant msms.txt the resulting PSMs carry: ``run`` (raw-file basename,
    the registry key), ``spectrum_id`` (MaxQuant's sequential scan number -- NOT a
    key into PTERO spectra), ``precursor_mz`` (monoisotopic), ``retention_time``
    (in MINUTES, verbatim), and ``peptidoform``. ``ion_mobility`` is ``None``
    (MaxQuant does not write it to msms.txt) -- see
    :func:`attach_mobility_from_maxquant`.
    """
    path = Path(path)
    cleaned = _strip_bom_to_temp(path)
    try:
        return psm_io.read_file(str(cleaned or path), filetype=filetype)
    finally:
        if cleaned is not None:
            cleaned.unlink(missing_ok=True)


def attach_mobility_from_maxquant(
    psms: PSMList,
    accumulated_path,
    *,
    scan_col: str = "Scan number",
    mobility_col: str = "Precursor ion mobility",
) -> int:
    """Populate ``PSM.ion_mobility`` (1/K0) from MaxQuant ``accumulatedMsmsScans.txt``.

    MaxQuant does not write ion mobility into msms.txt; the per-PSM 1/K0 lives in
    accumulatedMsmsScans.txt, keyed by the same ``Scan number`` as msms.txt
    (i.e. ``PSM.spectrum_id``). This reads that file (BOM-safe) and sets
    ``psm.ion_mobility`` on every PSM whose scan number is found.

    Returns the number of PSMs updated. PSMs whose scan number is absent keep
    ``ion_mobility = None`` and the matcher falls back to fragment ranking.
    """
    accumulated_path = Path(accumulated_path)
    mob: dict[str, float] = {}
    with open(accumulated_path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        if scan_col not in fields or mobility_col not in fields:
            raise KeyError(
                f"{accumulated_path.name} is missing required column(s) "
                f"{scan_col!r}/{mobility_col!r}. Found: {fields}"
            )
        for row in reader:
            raw = row.get(mobility_col, "")
            if raw in ("", "NaN", "nan", None):
                continue
            try:
                mob[str(row[scan_col])] = float(raw)
            except (TypeError, ValueError):
                continue

    if mob and max(mob.values()) > 100:                 # 1/K0 ~0.5-2.0; CCS is hundreds
        warnings.warn(
            f"{mobility_col!r} values exceed 100 -- these look like CCS, not 1/K0. "
            "The store mobility is 1/K0; check you selected the right column.",
            stacklevel=2,
        )

    n = 0
    for psm in psms:
        im = mob.get(str(psm.spectrum_id))
        if im is not None:
            psm.ion_mobility = im
            n += 1
    return n


# ── run registry (run_id -> opened raw store) ────────────────────────────────

class RunRegistry:
    """Map a ``PSM.run`` (raw-file basename) to a lazy raw store.

    Register a run either by its ``.zarr`` path (opened lazily on first
    :meth:`get`) or by handing over an already-open ``PteroSpectra``.

    >>> reg = RunRegistry()
    >>> reg.register("20201204_..._22476", r"C:/data/hela_ddamob.zarr")
    >>> run = reg.get("20201204_..._22476")          # opens lazily, instantly
    """

    def __init__(self) -> None:
        self._paths: dict[str, str] = {}
        self._opened: dict[str, object] = {}

    def register(self, run_id: str, store) -> None:
        """Register ``run_id`` -> a ``.zarr`` path (str/Path) or open PteroSpectra."""
        run_id = str(run_id)
        if isinstance(store, (str, Path)):
            self._paths[run_id] = str(store)
            self._opened.pop(run_id, None)            # drop any stale handle
        else:                                          # assume a PteroSpectra-like object
            self._opened[run_id] = store
            self._paths.setdefault(run_id, "<provided>")

    def __contains__(self, run_id) -> bool:
        run_id = str(run_id)
        return run_id in self._paths or run_id in self._opened

    def runs(self) -> list[str]:
        return sorted(set(self._paths) | set(self._opened))

    def get(self, run_id: str):
        """Return the opened ``PteroSpectra`` for ``run_id`` (opening it if needed)."""
        run_id = str(run_id)
        if run_id in self._opened:
            return self._opened[run_id]
        if run_id not in self._paths:
            raise KeyError(
                f"run {run_id!r} is not registered. Call "
                f"register({run_id!r}, <.zarr path>) first. Known runs: {self.runs()}"
            )
        from ptero.spectra import open_zarr            # lazy: keep PTERO out of import time
        self._opened[run_id] = open_zarr(self._paths[run_id])
        return self._opened[run_id]


# ── annotation core (spectrum_utils) ─────────────────────────────────────────

def _is_annotated(entry) -> bool:
    """True if a spectrum_utils annotation entry holds >=1 fragment assignment.

    annotate_proforma fills every peak with a ``PeakInterpretation``; unannotated
    peaks carry the ``?`` unknown marker with an empty ``fragment_annotations``.
    """
    if entry is None:
        return False
    return bool(getattr(entry, "fragment_annotations", None))


def _annotate_spectrum(mz, intensity, proforma, precursor_mz, precursor_charge,
                       *, fragment_tol=0.02, tol_mode="Da", ion_types="by") -> MsmsSpectrum:
    """Build a spectrum_utils MsmsSpectrum and annotate b/y ions of *proforma*.

    Peaks are sorted by m/z (spectrum_utils requires ascending m/z); store peaks
    are not guaranteed sorted.
    """
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if mz.size:
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
    sp = MsmsSpectrum(
        identifier=str(proforma),
        precursor_mz=float(precursor_mz) if precursor_mz is not None else float("nan"),
        precursor_charge=int(precursor_charge),
        mz=mz,
        intensity=intensity,
    )
    return sp.annotate_proforma(
        str(proforma), fragment_tol_mass=fragment_tol,
        fragment_tol_mode=tol_mode, ion_types=ion_types,
    )


def _peptide_length(proforma: str) -> int:
    """Number of residues in a ProForma string (strip charge suffix and [mods])."""
    seq = str(proforma).split("/")[0]
    seq = re.sub(r"\[[^\]]*\]", "", seq)          # drop [Oxidation] / [+15.99] etc.
    seq = re.sub(r"[^A-Za-z]", "", seq)            # drop N-/C-term dashes, digits
    return len(seq)


def _theoretical_fragments(proforma: str, max_charge: int, ion_types: str = "by"):
    """Theoretical (monoisotopic) b/y fragment m/z for *proforma*, charges 1..max_charge.

    Returns a list of ``(series, position, mz)`` where ``series`` is the ion-series
    letter ('b','y',...) and ``position`` the residue index. Uses spectrum_utils'
    own fragment generator so modifications shift masses correctly.
    """
    try:
        proteoform = _sus_proforma.parse(str(proforma))[0]
    except Exception:
        return []
    frags = _sus_fragment.get_theoretical_fragments(
        proteoform, ion_types=ion_types, max_isotope=0, max_charge=max(1, int(max_charge)))
    out = []
    for ann, mz in frags:
        it = getattr(ann, "ion_type", "")         # e.g. 'b3', 'y10'
        if not it or not it[1:].isdigit():
            continue
        out.append((it[0], int(it[1:]), float(mz)))
    return out


def _match_fragments(mz, proforma, precursor_charge,
                     *, fragment_tol=0.02, tol_mode="Da", ion_types="by") -> dict:
    """Match a peak list against a peptide's THEORETICAL b/y ions, one-to-one.

    This is the disambiguation primitive. Unlike walking every peak with
    ``annotate_proforma`` (which, on a raw/un-deisotoped spectrum, tags nearly
    every peak at some charge/isotope and massively over-counts), this iterates
    over the bounded set of theoretical ions and lets each claim at most one peak.
    The result is therefore bounded by the number of theoretical ions, and
    ``coverage`` is bounded by ``peptide_length - 1``.

    Returns a dict with:
      ``matched_ions``    -- distinct theoretical ions with a matching peak
      ``coverage``        -- distinct backbone bonds explained (b_i and y_(n-i)
                             count as the same bond)
      ``n_theoretical``   -- number of theoretical ions considered
      ``max_coverage``    -- peptide_length - 1
    """
    mz = np.asarray(mz, dtype=float)
    n = _peptide_length(proforma)
    theo = _theoretical_fragments(proforma, precursor_charge, ion_types=ion_types)
    result = {"matched_ions": 0, "coverage": 0,
              "n_theoretical": len(theo), "max_coverage": max(0, n - 1)}
    if mz.size == 0 or not theo:
        return result
    order = np.argsort(mz)
    mz_sorted = mz[order]
    used = np.zeros(mz_sorted.size, dtype=bool)
    bonds = set()
    matched = 0
    for series, pos, t in theo:
        if tol_mode == "ppm":
            tol = t * fragment_tol * 1e-6
        else:
            tol = fragment_tol
        j = int(np.searchsorted(mz_sorted, t))
        # nearest of the two neighbours around the insertion point
        best, best_d = -1, np.inf
        for k in (j - 1, j):
            if 0 <= k < mz_sorted.size and not used[k]:
                d = abs(mz_sorted[k] - t)
                if d < best_d:
                    best, best_d = k, d
        if best >= 0 and best_d <= tol:
            used[best] = True
            matched += 1
            bonds.add(pos if series in "abc" else (n - pos))
    result["matched_ions"] = matched
    result["coverage"] = len(bonds)
    return result


def _fragment_score(mz, intensity, proforma, precursor_mz, precursor_charge,
                    *, fragment_tol=0.02, tol_mode="Da", ion_types="by") -> int:
    """Disambiguation score: backbone-bond COVERAGE of *proforma* by this spectrum.

    Coverage (distinct backbone bonds explained) is bounded by peptide length and
    is a robust discriminator -- a true match explains most positions; a wrong
    co-eluting precursor explains few. (``intensity`` is accepted for signature
    compatibility but not used; ranking is by which peptide the peaks fit, not how
    intense they are.)
    """
    return _match_fragments(
        mz, proforma, precursor_charge,
        fragment_tol=fragment_tol, tol_mode=tol_mode, ion_types=ion_types)["coverage"]


def peak_annotations(mz, intensity, proforma, precursor_charge,
                     *, fragment_tol=0.02, tol_mode="Da", ion_types="by") -> pd.DataFrame:
    """Per-peak b/y ion labels for plotting an annotated spectrum.

    For each peak returns its m/z, intensity, a ``label`` (``"b5"`` / ``"y8"`` /
    ... for a matched theoretical ion, else ``""``), the ``series`` (``"b"`` /
    ``"y"`` / ``""``), and a boolean ``matched``. Matching is the same bounded,
    one-to-one, charge≤precursor scheme as :func:`_match_fragments` (each
    theoretical ion claims at most one peak), so the labelled peaks are the real
    fragments -- NOT the isotopolog/multi-charge over-count that
    ``annotate_proforma`` produces on a raw spectrum.

    This is the data a stick plot needs: matched peaks coloured by series and
    labelled, the rest drawn grey.

    Returns a DataFrame with columns ``mz``, ``intensity``, ``label``,
    ``series``, ``matched`` -- one row per input peak, in input order.
    """
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    n = mz.size
    out = pd.DataFrame({
        "mz": mz, "intensity": intensity,
        "label": [""] * n, "series": [""] * n, "matched": [False] * n,
    })
    theo = _theoretical_fragments(proforma, precursor_charge, ion_types=ion_types)
    if n == 0 or not theo:
        return out
    order = np.argsort(mz)
    mz_sorted = mz[order]
    used = np.zeros(n, dtype=bool)
    label = [""] * n
    series_col = [""] * n
    matched = [False] * n
    for s, pos, t in theo:
        tol = (t * fragment_tol * 1e-6) if tol_mode == "ppm" else fragment_tol
        j = int(np.searchsorted(mz_sorted, t))
        best, best_d = -1, np.inf
        for k in (j - 1, j):
            if 0 <= k < n and not used[k]:
                d = abs(mz_sorted[k] - t)
                if d < best_d:
                    best, best_d = k, d
        if best >= 0 and best_d <= tol:
            used[best] = True
            orig = int(order[best])          # sorted index -> original peak index
            label[orig] = f"{s}{pos}"
            series_col[orig] = s
            matched[orig] = True
    out["label"] = label
    out["series"] = series_col
    out["matched"] = matched
    return out


# ── the property-based link ──────────────────────────────────────────────────

@dataclass
class PsmMatch:
    """Result of locating a PSM in a raw store.

    Attributes
    ----------
    psm : PSM
        The query PSM.
    store_index : int | None
        Position in the store (usable as ``run[store_index]``), or ``None`` if
        unmatched / left ambiguous.
    spectrum : PteroSpectra | None
        Single-spectrum view of the matched spectrum, or ``None``.
    candidates : pandas.DataFrame
        Every spectrum that passed RT + m/z (+ mobility), with coordinate deltas
        (``d_rt``, ``d_mz``, ``d_im``). Empty if nothing matched.
    method : str
        How the result was reached: ``'rt+mz'``, ``'rt+mz+im'``,
        ``'fragment-ranked'``, ``'ambiguous'``, or ``'none'``.
    fragment_scores : pandas.DataFrame | None
        Per-candidate backbone-coverage scores, set only when fragment ranking
        ran (``store_index``, ``coverage``; best first).
    """
    psm: PSM
    store_index: Optional[int]
    spectrum: object
    candidates: pd.DataFrame
    method: str
    fragment_scores: Optional[pd.DataFrame] = None

    @property
    def resolved(self) -> bool:
        return self.store_index is not None

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def _select_candidates(ms2: pd.DataFrame, rt_sec, mz, im, *, rt_tol, mz_tol, im_tol):
    """Pure candidate selection -- identical logic to ``tests/test_link.py._match``.

    ``ms2`` is the MS2 rows of ``spectra_data`` (index == store position). Returns
    ``(candidates_df, used_im)``.
    """
    cand = ms2[(np.abs(ms2["rtime"] - rt_sec) <= rt_tol)
               & (np.abs(ms2["precursor_mz"] - mz) <= mz_tol)]
    used_im = False
    if im is not None and len(cand) > 1:
        cand = cand[np.abs(cand["ion_mobility"] - im) <= im_tol]
        used_im = True
    return cand, used_im


def _candidate_frame(cand: pd.DataFrame, rt_sec, mz, im) -> pd.DataFrame:
    """Attach coordinate deltas and surface the store index as a column."""
    out = cand.copy()
    out.insert(0, "store_index", out.index.to_numpy())
    out["d_rt"] = out["rtime"] - rt_sec
    out["d_mz"] = out["precursor_mz"] - mz
    if "ion_mobility" in out.columns:
        out["d_im"] = out["ion_mobility"] - im if im is not None else np.nan
    return out.reset_index(drop=True)


def spectrum_for_psm(
    psm: PSM,
    registry: RunRegistry,
    *,
    rt_tol: float = 3.0,
    mz_tol: float = 1.5,
    im_tol: float = 0.03,
    disambiguate: str = "fragments",
    fragment_tol: float = 0.02,
    tol_mode: str = "Da",
    ion_types: str = "by",
) -> PsmMatch:
    """Locate the raw MS2 spectrum for *psm* by (RT, precursor m/z, ion mobility).

    RT is converted from MaxQuant minutes to store seconds (x60). Mobility is used
    only when ``psm.ion_mobility`` is set (see
    :func:`attach_mobility_from_maxquant`) AND more than one candidate remains.
    If still ambiguous and ``disambiguate='fragments'``, candidates are ranked by
    b/y fragment match and the best is chosen. Candidate selection is identical to
    the verified ``tests/test_link.py._match``.

    Parameters
    ----------
    rt_tol : float
        RT tolerance in seconds (compared after the x60 conversion).
    mz_tol : float
        Precursor m/z tolerance in Th (wide: store holds the isolation-window
        centre, not the monoisotopic m/z).
    im_tol : float
        1/K0 tolerance (the empirically justified disambiguation window).
    disambiguate : {'fragments', 'none'}
        ``'fragments'`` ranks remaining candidates by b/y match; ``'none'`` leaves
        an ambiguous result unresolved (``store_index = None``).
    """
    run = registry.get(psm.run)
    sd = run.spectra_data(["ms_level", "rtime", "precursor_mz", "ion_mobility"])
    ms2 = sd[sd["ms_level"] == 2]

    rt_sec = psm.retention_time * 60.0 if psm.retention_time is not None else np.nan
    mz = psm.precursor_mz
    im = psm.ion_mobility

    cand, used_im = _select_candidates(
        ms2, rt_sec, mz, im, rt_tol=rt_tol, mz_tol=mz_tol, im_tol=im_tol)
    base_method = "rt+mz+im" if used_im else "rt+mz"
    cframe = _candidate_frame(cand, rt_sec, mz, im)

    if len(cand) == 0:
        return PsmMatch(psm, None, None, cframe, "none")
    if len(cand) == 1:
        idx = int(cand.index[0])
        return PsmMatch(psm, idx, run[idx], cframe, base_method)

    # still ambiguous -> rank by MS2 fragment match
    if disambiguate == "fragments" and psm.peptidoform is not None:
        proforma = psm.peptidoform.proforma
        charge = psm.peptidoform.precursor_charge
        scores = []
        for pos in cand.index:
            pos = int(pos)
            m = run.mz([pos])[0]
            it = run.intensity([pos])[0]
            scores.append((pos, _fragment_score(
                m, it, proforma, mz, charge,
                fragment_tol=fragment_tol, tol_mode=tol_mode, ion_types=ion_types)))
        sframe = (pd.DataFrame(scores, columns=["store_index", "coverage"])
                  .sort_values("coverage", ascending=False, kind="stable")
                  .reset_index(drop=True))
        best = int(sframe.iloc[0]["store_index"])
        return PsmMatch(psm, best, run[best], cframe, "fragment-ranked", sframe)

    return PsmMatch(psm, None, None, cframe, "ambiguous")


def annotate_match(
    match: PsmMatch,
    *,
    fragment_tol: float = 0.02,
    tol_mode: str = "Da",
    ion_types: str = "by",
) -> MsmsSpectrum:
    """Annotate the matched MS2 spectrum's b/y ions -- the flagship annotated
    spectrum (input to a mirror/annotation plot).

    Raises if the match is unresolved (no unique spectrum).
    """
    if match.store_index is None or match.spectrum is None:
        raise ValueError("match is unresolved (no unique spectrum); nothing to annotate.")
    sp = match.spectrum
    mz = sp.mz([0])[0]
    it = sp.intensity([0])[0]
    pep = match.psm.peptidoform
    return _annotate_spectrum(
        mz, it, pep.proforma, match.psm.precursor_mz, pep.precursor_charge,
        fragment_tol=fragment_tol, tol_mode=tol_mode, ion_types=ion_types)