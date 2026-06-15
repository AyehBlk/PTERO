"""
Tests for MsBackendMzML reading ion mobility + DIA isolation windows.

There is no real 4D/DIA mzML on hand, so these build a small synthetic mzML that
contains the exact constructs a vendor mzML would (per-peak ion-mobility binary
arrays and per-MS2 isolation windows) and check the backend reads them. This
validates the *parsing*, not a specific instrument's file.
"""
from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyteomics")
pytest.importorskip("psims")

from ptero.spectra.backends.mzml_backend import MsBackendMzML


def _enc(vals) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    return base64.b64encode(zlib.compress(struct.pack("<%dd" % len(vals), *vals))).decode()


def _bda(vals, accession, name) -> str:
    return (f'<binaryDataArray encodedLength="0">'
            f'<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            f'<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression"/>'
            f'<cvParam cvRef="MS" accession="{accession}" name="{name}"/>'
            f'<binary>{_enc(vals)}</binary></binaryDataArray>')


def _spectrum(i, lvl, rt, mz, inten, mob=None, iso_center=None,
              iso_lo=12.5, iso_hi=12.5) -> str:
    arrays = [_bda(mz, "MS:1000514", "m/z array"),
              _bda(inten, "MS:1000515", "intensity array")]
    if mob is not None:
        arrays.append(_bda(mob, "MS:1002816", "mean inverse reduced ion mobility array"))
    prec = ""
    if iso_center is not None:
        # DIA: an isolation window but NO selected ion (no single precursor m/z)
        prec = (f'<precursorList count="1"><precursor><isolationWindow>'
                f'<cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="{iso_center}"/>'
                f'<cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="{iso_lo}"/>'
                f'<cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="{iso_hi}"/>'
                f'</isolationWindow></precursor></precursorList>')
    return (f'<spectrum index="{i}" id="scan={i+1}" defaultArrayLength="{len(mz)}">'
            f'<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{lvl}"/>'
            f'<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum"/>'
            f'<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" '
            f'name="scan start time" value="{rt}" unitName="second"/></scan></scanList>'
            f'{prec}<binaryDataArrayList count="{len(arrays)}">{"".join(arrays)}'
            f'</binaryDataArrayList></spectrum>')


def _write_mzml(path, specs) -> str:
    doc = ('<?xml version="1.0" encoding="utf-8"?>'
           '<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">'
           '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">'
           f'<run id="synthetic"><spectrumList count="{len(specs)}">{"".join(specs)}'
           '</spectrumList></run></mzML></indexedmzML>')
    Path(path).write_text(doc, encoding="utf-8")
    return str(path)


def _dia_im_mzml(path) -> str:
    """1 MS1 (with mobility array) + 4 MS2 at two distinct isolation windows."""
    specs = [
        _spectrum(0, 1, 10.0, [400., 600., 800.], [1e4, 2e4, 3e4], mob=[0.8, 1.0, 1.2]),
        _spectrum(1, 2, 10.1, [120., 250.], [500., 700.], mob=[0.81, 0.82], iso_center=500.0),
        _spectrum(2, 2, 10.2, [130., 260.], [600., 800.], mob=[1.01, 1.02], iso_center=700.0),
        _spectrum(3, 2, 20.0, [121., 251.], [510., 710.], mob=[0.80, 0.83], iso_center=500.0),
        _spectrum(4, 2, 20.1, [131., 261.], [610., 810.], mob=[1.00, 1.03], iso_center=700.0),
    ]
    return _write_mzml(path, specs)


def _dda_plain_mzml(path) -> str:
    """Ordinary 3D DDA: no mobility array, MS2 has a selected-ion precursor."""
    s0 = _spectrum(0, 1, 5.0, [300., 500.], [1e3, 2e3])
    # MS2 with a real selected ion (DDA)
    s1 = (f'<spectrum index="1" id="scan=2" defaultArrayLength="2">'
          f'<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/>'
          f'<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" '
          f'name="scan start time" value="5.1" unitName="second"/></scan></scanList>'
          f'<precursorList count="1"><precursor><selectedIonList count="1"><selectedIon>'
          f'<cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="455.2"/>'
          f'</selectedIon></selectedIonList></precursor></precursorList>'
          f'<binaryDataArrayList count="2">{_bda([100.,200.],"MS:1000514","m/z array")}'
          f'{_bda([10.,20.],"MS:1000515","intensity array")}</binaryDataArrayList></spectrum>')
    return _write_mzml(path, [s0, s1])


# ── ion mobility (4D) ────────────────────────────────────────────────────────

def test_mzml_detects_4d(tmp_path):
    be = MsBackendMzML(_dia_im_mzml(tmp_path / "dia.mzML"))
    assert be.has_4d is True


def test_mzml_reads_per_peak_mobility(tmp_path):
    be = MsBackendMzML(_dia_im_mzml(tmp_path / "dia.mzML"))
    P = be.peaks_data([0], ("mz", "intensity", "ion_mobility"))[0]
    assert P.shape == (3, 3)
    np.testing.assert_allclose(P[:, 2], [0.8, 1.0, 1.2])   # the mobility column


def test_mzml_dda_stays_3d(tmp_path):
    be = MsBackendMzML(_dda_plain_mzml(tmp_path / "dda.mzML"))
    assert be.has_4d is False
    # a plain DDA file carries no isolation-window columns
    assert "isolation_mz_lower" not in be.spectra_data().columns


# ── DIA isolation windows ────────────────────────────────────────────────────

def test_mzml_reads_isolation_windows(tmp_path):
    be = MsBackendMzML(_dia_im_mzml(tmp_path / "dia.mzML"))
    sd = be.spectra_data()
    ms2 = sd[sd["ms_level"] == 2]
    assert len(ms2) == 4
    # DIA: no single precursor m/z, but an isolation window centre + edges
    assert ms2["precursor_mz"].isna().all()
    centers = sorted(ms2["isolation_mz"].dropna().unique().tolist())
    assert centers == [500.0, 700.0]
    w = ms2[ms2["isolation_mz"] == 500.0].iloc[0]
    assert w["isolation_mz_lower"] == pytest.approx(487.5)
    assert w["isolation_mz_upper"] == pytest.approx(512.5)
    assert w["isolation_width"] == pytest.approx(25.0)


def test_mzml_synthesizes_window_ids(tmp_path):
    be = MsBackendMzML(_dia_im_mzml(tmp_path / "dia.mzML"))
    sd = be.spectra_data()
    ms2 = sd[sd["ms_level"] == 2]
    # two distinct windows -> two window ids; each used by two MS2 spectra
    assert ms2["window_id"].nunique() == 2
    assert set(ms2["window_group"].dropna().unique()) == {1.0, 2.0}
    # MS1 row is not assigned to a window
    ms1 = sd[sd["ms_level"] == 1].iloc[0]
    assert ms1["window_id"] == -1


def test_mzml_dda_no_window_synthesis(tmp_path):
    be = MsBackendMzML(_dda_plain_mzml(tmp_path / "dda.mzML"))
    sd = be.spectra_data()
    assert "window_id" not in sd.columns          # nothing to synthesise
    # DDA precursor m/z is still read as before
    assert sd[sd["ms_level"] == 2]["precursor_mz"].iloc[0] == pytest.approx(455.2)


def test_mzml_subset_preserves_4d_and_columns(tmp_path):
    be = MsBackendMzML(_dia_im_mzml(tmp_path / "dia.mzML"))
    sub = be.subset([1, 2])                        # two MS2 spectra
    assert sub.has_4d is True
    assert len(sub) == 2
    assert "isolation_mz" in sub.spectra_data().columns
