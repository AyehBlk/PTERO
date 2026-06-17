"""Tests for the mzML backend's per-spectrum ion-mobility derivation.

These guard the behaviour added for combined-mobility timsTOF mzML (e.g. a
``.d`` converted with MSConvert "Combine ion mobility scans"): each MS/MS is one
summed spectrum that carries ``ion mobility lower/upper limit`` cvParams rather
than a scan-level 1/K0 scalar. The backend must report the midpoint of those
limits as the spectrum-level ``ion_mobility`` (the coordinate the ID linker uses
to separate co-eluting precursors), while leaving ordinary 3D mzML at NaN and
still preferring a genuine scan-level mobility scalar when one is present.

Synthetic-first: tiny mzML files are written with ``psims`` into ``tmp_path``; no
real instrument data. ``psims`` ships with the project's mzML support (it is the
writer pyteomics' reader round-trips), so this needs no extra dependency.
"""
import numpy as np
import pytest

psims_mzml = pytest.importorskip("psims.mzml")
from psims.mzml import MzMLWriter

from ptero.spectra.backends.mzml_backend import MsBackendMzML


# ── helper: write a tiny mzML with configurable mobility encoding ────────────

def _write_mzml(path, spectra):
    """Write *spectra* to an mzML at *path*.

    Each entry of *spectra* is a dict:
      ``ms_level``     -- 1 or 2
      ``mz``,``inten`` -- peak arrays
      ``rt``           -- scan start time (seconds; written with a seconds unit)
      ``im_limits``    -- (lower, upper) -> 'ion mobility lower/upper limit' params
      ``im_scan``      -- scalar -> scan-level 'inverse reduced ion mobility'
      ``precursor_mz`` -- for MS2
    """
    with MzMLWriter(open(path, "wb"), close=True) as w:
        w.controlled_vocabularies()
        with w.run(id="run1"):
            with w.spectrum_list(count=len(spectra)):
                for i, s in enumerate(spectra):
                    params = [{"ms level": s["ms_level"]},
                              "MS1 spectrum" if s["ms_level"] == 1 else "MSn spectrum"]
                    for lim, nm in zip(s.get("im_limits", (None, None)),
                                       ("ion mobility lower limit",
                                        "ion mobility upper limit")):
                        if lim is not None:
                            params.append({"name": nm, "value": float(lim),
                                           "unit_name": "volt-second per square centimeter"})
                    scan_params = []
                    if s.get("im_scan") is not None:
                        scan_params.append(
                            {"name": "inverse reduced ion mobility", "value": float(s["im_scan"]),
                             "unit_name": "volt-second per square centimeter"})
                    kw = dict(
                        id=f"scan={i + 1}",
                        params=params,
                        scan_start_time=s["rt"],
                    )
                    if scan_params:
                        kw["scan_params"] = scan_params
                    if s["ms_level"] == 2:
                        kw["precursor_information"] = {
                            "mz": s.get("precursor_mz", 500.0), "charge": 2, "scan_id": "scan=1"}
                    w.write_spectrum(list(s["mz"]), list(s["inten"]), **kw)
    return path


def _ms2(sd):
    return sd[sd["ms_level"] == 2].reset_index(drop=True)


# ── tests ─────────────────────────────────────────────────────────────────

def test_mobility_from_im_limits_midpoint(tmp_path):
    """Combined-mobility mzML: ion_mobility is the midpoint of the IM limits."""
    p = _write_mzml(tmp_path / "im.mzML", [
        {"ms_level": 1, "mz": [100.0, 200.0], "inten": [9.0, 8.0], "rt": 10.0},
        {"ms_level": 2, "mz": [150.0, 250.0], "inten": [5.0, 6.0], "rt": 11.0,
         "im_limits": (0.80, 0.90), "precursor_mz": 478.78},
    ])
    be = MsBackendMzML(str(p))
    sd = be.spectra_data(["ms_level", "rtime", "precursor_mz", "ion_mobility"])
    ms2 = _ms2(sd)
    assert ms2.loc[0, "ion_mobility"] == pytest.approx(0.85)      # (0.80 + 0.90) / 2


def test_mobility_nan_when_no_limits(tmp_path):
    """Ordinary 3D mzML (no IM limits, no scan-level IM): ion_mobility stays NaN."""
    p = _write_mzml(tmp_path / "plain.mzML", [
        {"ms_level": 1, "mz": [100.0], "inten": [9.0], "rt": 10.0},
        {"ms_level": 2, "mz": [150.0, 250.0], "inten": [5.0, 6.0], "rt": 11.0,
         "precursor_mz": 300.0},
    ])
    be = MsBackendMzML(str(p))
    sd = be.spectra_data(["ms_level", "ion_mobility"])
    ms2 = _ms2(sd)
    assert np.isnan(ms2.loc[0, "ion_mobility"])


def test_scan_level_mobility_takes_precedence(tmp_path):
    """A genuine scan-level 1/K0 wins over the IM-limit midpoint when both exist."""
    p = _write_mzml(tmp_path / "both.mzML", [
        {"ms_level": 1, "mz": [100.0], "inten": [9.0], "rt": 10.0},
        {"ms_level": 2, "mz": [150.0], "inten": [5.0], "rt": 11.0,
         "im_limits": (0.80, 0.90), "im_scan": 1.234, "precursor_mz": 478.78},
    ])
    be = MsBackendMzML(str(p))
    sd = be.spectra_data(["ms_level", "ion_mobility"])
    ms2 = _ms2(sd)
    assert ms2.loc[0, "ion_mobility"] == pytest.approx(1.234)     # scan-level, not 0.85


def test_one_sided_limit_is_nan(tmp_path):
    """If only one IM limit is present, no midpoint is derivable -> NaN."""
    p = _write_mzml(tmp_path / "onesided.mzML", [
        {"ms_level": 1, "mz": [100.0], "inten": [9.0], "rt": 10.0},
        {"ms_level": 2, "mz": [150.0], "inten": [5.0], "rt": 11.0,
         "im_limits": (0.80, None), "precursor_mz": 478.78},
    ])
    be = MsBackendMzML(str(p))
    sd = be.spectra_data(["ms_level", "ion_mobility"])
    ms2 = _ms2(sd)
    assert np.isnan(ms2.loc[0, "ion_mobility"])