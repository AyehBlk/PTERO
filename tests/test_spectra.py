"""Tests for ptero.spectra: PteroSpectra, backends, lazy queue, 4D, mzML."""
import base64
import struct
import zlib

import numpy as np
import pandas as pd
import pytest

from ptero.spectra import (
    from_dataframe, from_mzml, PteroSpectra,
    MsBackendMemory, MsBackendZarr, MsBackendParquet,
)


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sp4d():
    """5-spectrum synthetic 4D object with known peaks."""
    peaks = [
        {"mz": np.array([150., 300., 500., 950.]),
         "intensity": np.array([50., 5000., 80000., 200.], "f4"),
         "ion_mobility": np.array([0.8, 1.0, 1.2, 0.95], "f4")},
        {"mz": np.array([120., 480.]),
         "intensity": np.array([900., 12000.], "f4"),
         "ion_mobility": np.array([0.7, 1.1], "f4")},
        {"mz": np.array([200., 600., 800.]),
         "intensity": np.array([3000., 4000., 100.], "f4"),
         "ion_mobility": np.array([0.85, 1.05, 1.3], "f4")},
        {"mz": np.array([333.]),
         "intensity": np.array([7777.], "f4"),
         "ion_mobility": np.array([0.99], "f4")},
        {"mz": np.array([250., 750., 900., 999.]),
         "intensity": np.array([10., 20000., 50., 60000.], "f4"),
         "ion_mobility": np.array([0.75, 1.0, 1.2, 1.25], "f4")},
    ]
    meta = pd.DataFrame({
        "ms_level": [1, 2, 1, 2, 1],
        "rtime": [0., 10., 20., 30., 40.],
        "precursor_mz": [np.nan, 400., np.nan, 333., np.nan],
    })
    return from_dataframe(meta, peaks)


@pytest.fixture
def tiny_mzml(tmp_path):
    def b64(vals):
        comp = zlib.compress(struct.pack("<%dd" % len(vals), *vals))
        return base64.b64encode(comp).decode()

    def bda(vals, kind):
        acc = "MS:1000514" if kind == "mz" else "MS:1000515"
        nm = "m/z array" if kind == "mz" else "intensity array"
        return (f'<binaryDataArray encodedLength="0">'
                f'<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
                f'<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression"/>'
                f'<cvParam cvRef="MS" accession="{acc}" name="{nm}"/>'
                f'<binary>{b64(vals)}</binary></binaryDataArray>')

    def spec(idx, sid, lvl, rt, mz, it, pmz=None):
        prec = ""
        if pmz is not None:
            prec = (f'<precursorList count="1"><precursor><selectedIonList count="1">'
                    f'<selectedIon><cvParam cvRef="MS" accession="MS:1000744" '
                    f'name="selected ion m/z" value="{pmz}"/></selectedIon>'
                    f'</selectedIonList></precursor></precursorList>')
        return (f'<spectrum index="{idx}" id="{sid}" defaultArrayLength="{len(mz)}">'
                f'<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{lvl}"/>'
                f'<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum"/>'
                f'<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" '
                f'name="scan start time" value="{rt}" unitName="second"/></scan></scanList>'
                f'{prec}<binaryDataArrayList count="2">{bda(mz,"mz")}{bda(it,"int")}'
                f'</binaryDataArrayList></spectrum>')

    s1 = spec(0, "scan=1", 1, 12.5, [100.1, 200.2, 300.3], [1000., 2000., 3000.])
    s2 = spec(1, "scan=2", 2, 13.0, [55.5, 66.6], [500., 600.], pmz=400.4)
    doc = ('<?xml version="1.0" encoding="utf-8"?>'
           '<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">'
           '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">'
           f'<run id="r1"><spectrumList count="2">{s1}{s2}</spectrumList></run>'
           '</mzML></indexedmzML>')
    p = tmp_path / "tiny.mzML"
    p.write_text(doc, encoding="utf-8")
    return p


# ── basics ──────────────────────────────────────────────────────────────────

class TestBasics:
    def test_len_and_4d(self, sp4d):
        assert len(sp4d) == 5
        assert sp4d.has_4d
        assert sp4d.peaks_variables == ("mz", "intensity", "ion_mobility")

    def test_core_variable_access(self, sp4d):
        assert list(sp4d.ms_level) == [1, 2, 1, 2, 1]
        np.testing.assert_array_equal(sp4d.rtime, [0, 10, 20, 30, 40])

    def test_peaks_data_shapes(self, sp4d):
        pk = sp4d.peaks_data([0], ("mz", "intensity", "ion_mobility"))[0]
        assert pk.shape == (4, 3)

    def test_missing_core_columns_filled(self, sp4d):
        sd = sp4d.spectra_data()
        for col in ("collision_energy", "polarity", "centroided"):
            assert col in sd.columns


# ── lazy queue ────────────────────────────────────────────────────────────

class TestLazyQueue:
    def test_filter_intensity(self, sp4d):
        out = sp4d.filter_intensity(low=1000).mz()[0]
        np.testing.assert_array_equal(out, [300., 500.])

    def test_filter_mz_range(self, sp4d):
        out = sp4d.filter_mz_range(200, 900).mz()[0]
        np.testing.assert_array_equal(out, [300., 500.])

    def test_filter_mobility(self, sp4d):
        out = sp4d.filter_mobility(0.9, 1.1).mz()[0]
        np.testing.assert_array_equal(out, [300., 950.])

    def test_chaining_and_queue_length(self, sp4d):
        f = sp4d.filter_intensity(low=1000).filter_mz_range(200, 900)
        assert len(f.processing_queue) == 2
        np.testing.assert_array_equal(f.mz()[0], [300., 500.])

    def test_reset_undoes_queue(self, sp4d):
        f = sp4d.filter_intensity(low=1000)
        assert len(f.reset().processing_queue) == 0
        assert len(f.reset().mz()[0]) == 4

    def test_original_unchanged(self, sp4d):
        _ = sp4d.filter_intensity(low=1000).mz()
        assert len(sp4d.mz()[0]) == 4  # lazy filters return new objects


# ── spectrum filters (view-based) ──────────────────────────────────────────

class TestSpectrumFilters:
    def test_filter_ms_level(self, sp4d):
        assert len(sp4d.filter_ms_level(1)) == 3
        assert len(sp4d.filter_ms_level(2)) == 2

    def test_filter_rt(self, sp4d):
        sub = sp4d.filter_rt(5, 25)
        assert len(sub) == 2
        np.testing.assert_array_equal(sub.rtime, [10, 20])

    def test_getitem(self, sp4d):
        assert len(sp4d[1:3]) == 2
        np.testing.assert_array_equal(sp4d[0].peaks_data()[0][:, 0], [150, 300, 500, 950])

    def test_subset_preserves_peaks(self, sp4d):
        np.testing.assert_array_equal(sp4d[3].mz()[0], [333.])


# ── backends + conversion ───────────────────────────────────────────────────

def _peaks_equal(a, b):
    cols = ("mz", "intensity", "ion_mobility") if a.has_4d else ("mz", "intensity")
    A, B = a.peaks_data(None, cols), b.peaks_data(None, cols)
    return len(A) == len(B) and all(np.allclose(x, y, equal_nan=True) for x, y in zip(A, B))


class TestBackends:
    def test_zarr_roundtrip(self, sp4d, tmp_path):
        z = sp4d.set_backend("zarr", path=str(tmp_path / "e.zarr"))
        assert isinstance(z.backend, MsBackendZarr)
        assert _peaks_equal(sp4d, z)
        np.testing.assert_array_equal(z.rtime, sp4d.rtime)

    def test_parquet_roundtrip(self, sp4d, tmp_path):
        p = sp4d.set_backend("parquet", path=str(tmp_path / "e_pq"))
        assert isinstance(p.backend, MsBackendParquet)
        assert _peaks_equal(sp4d, p)

    def test_zarr_has_4d_flag_persisted(self, sp4d, tmp_path):
        z = sp4d.set_backend("zarr", path=str(tmp_path / "e.zarr"))
        z2 = MsBackendZarr(str(tmp_path / "e.zarr"))
        assert z2.has_4d

    def test_zarr_random_access(self, sp4d, tmp_path):
        z = sp4d.set_backend("zarr", path=str(tmp_path / "e.zarr"))
        np.testing.assert_array_equal(z[4].mz()[0], sp4d[4].mz()[0])

    def test_lazy_queue_materialized_on_convert(self, sp4d, tmp_path):
        filtered = sp4d.filter_intensity(low=1000)
        z = filtered.set_backend("zarr", path=str(tmp_path / "f.zarr"))
        assert len(z.processing_queue) == 0  # queue applied during write
        np.testing.assert_array_equal(z.mz()[0], [300., 500.])

    def test_zarr_csr_layout_on_disk(self, sp4d, tmp_path):
        import os
        sp4d.set_backend("zarr", path=str(tmp_path / "e.zarr"))
        contents = os.listdir(tmp_path / "e.zarr")
        assert "offsets" in contents and "peaks" in contents
        assert "spectra.parquet" in contents  # metadata in Parquet sidecar

    def test_iter_peaks_matches_peaks_data(self, sp4d, tmp_path):
        # blocked reader must return byte-identical per-spectrum arrays; force many
        # small blocks so the multi-block path is exercised
        z = sp4d.set_backend("zarr", path=str(tmp_path / "e.zarr"))
        ref = z.peaks_data(columns=("mz", "intensity", "ion_mobility"))
        got = [None] * len(ref)
        for pos, P in z.iter_peaks(columns=("mz", "intensity", "ion_mobility"), block_size=3):
            got[pos] = P
        assert all(g is not None for g in got)
        for a, b in zip(ref, got):
            np.testing.assert_array_equal(a, b)

    def test_iter_peaks_applies_queue(self, sp4d, tmp_path):
        # queued filters must be applied identically through the blocked path
        z = sp4d.set_backend("zarr", path=str(tmp_path / "e.zarr"))
        q = z.filter_mz_range(250, 1200).filter_mobility(0.5, 1.5)
        ref = q.peaks_data(columns=("mz", "intensity"))
        got = [None] * len(ref)
        for pos, P in q.iter_peaks(columns=("mz", "intensity"), block_size=5):
            got[pos] = P
        for a, b in zip(ref, got):
            np.testing.assert_array_equal(a, b)

    def test_memory_backend_public_spectrum_count(self, sp4d):
        mem = sp4d.backend
        assert isinstance(mem, MsBackendMemory)
        assert len(mem) == 5

    def test_memory_backend_public_peak_count(self, sp4d):
        mem = sp4d.backend
        assert isinstance(mem, MsBackendMemory)
        assert mem.n_peaks == 14

# ── 3D (no ion mobility) ────────────────────────────────────────────────────

class Test3D:
    def test_3d_object(self):
        peaks = [{"mz": np.array([1., 2.]), "intensity": np.array([3., 4.], "f4")}]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [1.]}), peaks)
        assert not sp.has_4d
        assert sp.peaks_variables == ("mz", "intensity")
        with pytest.raises(ValueError):
            sp.filter_mobility(0.9, 1.1)


# ── mzML import ──────────────────────────────────────────────────────────────

class TestMzML:
    def test_read(self, tiny_mzml):
        sp = from_mzml(tiny_mzml)
        assert len(sp) == 2
        assert list(sp.ms_level) == [1, 2]
        np.testing.assert_allclose(sp.rtime, [12.5, 13.0])

    def test_peaks(self, tiny_mzml):
        sp = from_mzml(tiny_mzml)
        np.testing.assert_allclose(sp.mz()[0], [100.1, 200.2, 300.3])
        np.testing.assert_allclose(sp.intensity()[1], [500., 600.])

    def test_precursor(self, tiny_mzml):
        sp = from_mzml(tiny_mzml)
        np.testing.assert_allclose(sp.precursor_mz[1], 400.4)

    def test_ms_level_filter(self, tiny_mzml):
        assert len(from_mzml(tiny_mzml, ms_level=2)) == 1

    def test_mzml_to_zarr(self, tiny_mzml, tmp_path):
        sp = from_mzml(tiny_mzml)
        z = sp.set_backend("zarr", path=str(tmp_path / "m.zarr"))
        np.testing.assert_allclose(z.mz()[0], sp.mz()[0])


# ── lazy mzML backend ────────────────────────────────────────────────────────

class TestLazyMzML:
    def test_default_is_lazy(self, tiny_mzml):
        from ptero.spectra import MsBackendMzML
        sp = from_mzml(tiny_mzml)               # lazy=True is the default
        assert isinstance(sp.backend, MsBackendMzML)
        assert sp.backend.read_only is True
        assert sp.has_4d is False

    def test_eager_is_memory(self, tiny_mzml):
        from ptero.spectra import MsBackendMemory
        sp = from_mzml(tiny_mzml, lazy=False)
        assert isinstance(sp.backend, MsBackendMemory)

    def test_lazy_equals_eager(self, tiny_mzml):
        lazy = from_mzml(tiny_mzml)
        eager = from_mzml(tiny_mzml, lazy=False)
        assert len(lazy) == len(eager)
        for a, b in zip(lazy.mz(), eager.mz()):
            np.testing.assert_allclose(a, b)
        np.testing.assert_allclose(lazy.rtime, eager.rtime)

    def test_metadata_without_peaks(self, tiny_mzml):
        sp = from_mzml(tiny_mzml)
        # spectra_data must work from the in-memory metadata, no peak reads
        np.testing.assert_array_equal(sp.ms_level, [1, 2])

    def test_random_access_single_spectrum(self, tiny_mzml):
        sp = from_mzml(tiny_mzml)
        np.testing.assert_allclose(sp[1].mz()[0], [55.5, 66.6])

    def test_ms_level_filter(self, tiny_mzml):
        assert len(from_mzml(tiny_mzml, ms_level=2)) == 1

    def test_lazy_to_zarr(self, tiny_mzml, tmp_path):
        sp = from_mzml(tiny_mzml)
        z = sp.set_backend("zarr", path=str(tmp_path / "lz.zarr"))
        np.testing.assert_allclose(z.mz()[0], sp.mz()[0])

    def test_no_ion_mobility(self, tiny_mzml):
        sp = from_mzml(tiny_mzml)
        with pytest.raises(KeyError):
            sp.backend.peaks_data([0], ("mz", "intensity", "ion_mobility"))


# ── to_points (flatten to long peak table) ───────────────────────────────────

class TestToPoints:
    def test_columns_4d(self, sp4d):
        pts = sp4d.to_points()
        assert list(pts.columns) == ["rt", "mz", "intensity", "ion_mobility", "ms_level", "scan_index"]

    def test_columns_3d(self):
        peaks = [{"mz": np.array([1., 2.]), "intensity": np.array([3., 4.], "f4")}]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [5.0]}), peaks)
        pts = sp.to_points()
        assert "ion_mobility" not in pts.columns
        assert list(pts["rt"]) == [5.0, 5.0]

    def test_row_count_equals_total_peaks(self, sp4d):
        assert len(sp4d.to_points()) == sum(len(m) for m in sp4d.mz())

    def test_rt_repeated_per_peak(self, sp4d):
        pts = sp4d.to_points()
        # scan 0 has 4 peaks all at rtime 0.0
        assert (pts[pts.scan_index == 0]["rt"] == 0.0).all()
        assert (pts.scan_index == 0).sum() == 4

    def test_ms_level_arg(self, sp4d):
        pts = sp4d.to_points(ms_level=1)
        assert (pts["ms_level"] == 1).all()

    def test_queue_applied(self, sp4d):
        full = len(sp4d.to_points())
        filt = len(sp4d.filter_intensity(low=1000).to_points())
        assert filt < full

    def test_empty(self):
        peaks = [{"mz": np.array([]), "intensity": np.array([], "f4")}]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [1.0]}), peaks)
        pts = sp.to_points()
        assert len(pts) == 0 and "rt" in pts.columns

    def test_backend_independent(self, sp4d, tmp_path):
        a = sp4d.to_points()
        z = sp4d.set_backend("zarr", path=str(tmp_path / "tp.zarr")).to_points()
        assert len(a) == len(z)


# ── Bruker timsTOF core (vendor-independent; .d adapter validated on Windows) ──

@pytest.fixture
def tims_events():
    """A synthetic timsTOF detector-event table: 2 MS1 frames + 1 PASEF MS2 frame."""
    rows = []
    for mob, mzs in [(1.10, [400.1, 500.2]), (0.95, [600.3]), (0.80, [700.4, 800.5, 900.6])]:
        for mz in mzs:
            rows.append(dict(frame=1, rt=100.0, mobility=mob, mz=mz, intensity=1000.0 + mz,
                             ms_level=1, precursor_mz=np.nan))
    for mob, mzs in [(1.05, [410.0]), (0.85, [710.0, 720.0])]:
        for mz in mzs:
            rows.append(dict(frame=2, rt=101.0, mobility=mob, mz=mz, intensity=2000.0 + mz,
                             ms_level=1, precursor_mz=np.nan))
    for prec, mzs in [(550.0, [120.0, 250.0]), (650.0, [300.0])]:
        for mz in mzs:
            rows.append(dict(frame=3, rt=102.0, mobility=0.9, mz=mz, intensity=500.0 + mz,
                             ms_level=2, precursor_mz=prec))
    return pd.DataFrame(rows)


class TestBrukerTimsTOF:
    def test_frame_grouping_counts(self, tims_events):
        from ptero.spectra._bruker import _timstof_to_spectra
        meta, peaks = _timstof_to_spectra(tims_events)
        assert len(meta) == 4                       # 2 MS1 frames + 2 MS2 precursors
        assert list(meta["ms_level"]) == [1, 1, 2, 2]

    def test_ms1_frame_combines_mobility_scans(self, tims_events):
        from ptero.spectra import from_timstof_dataframe
        sp = from_timstof_dataframe(tims_events)
        # frame 1 had 6 peaks across 3 mobility scans -> one 4D spectrum
        assert len(sp.mz([0])[0]) == 6
        assert sp.has_4d
        im = sp.ion_mobility_peaks([0])[0]
        assert im.min() >= 0.79 and im.max() <= 1.11

    def test_ms2_split_by_precursor(self, tims_events):
        from ptero.spectra import from_timstof_dataframe
        sp = from_timstof_dataframe(tims_events)
        ms2 = sp.filter_ms_level(2)
        assert len(ms2) == 2
        assert sorted(ms2.spectra_data(["precursor_mz"]).precursor_mz.tolist()) == [550.0, 650.0]

    def test_ms_level_filter(self, tims_events):
        from ptero.spectra import from_timstof_dataframe
        assert len(from_timstof_dataframe(tims_events, ms_level=1)) == 2

    def test_points_roundtrip(self, tims_events):
        from ptero.spectra import from_timstof_dataframe
        pts = from_timstof_dataframe(tims_events).to_points()
        assert "ion_mobility" in pts.columns
        assert len(pts) == len(tims_events)        # every detector event preserved

    def test_missing_columns_raises(self):
        from ptero.spectra._bruker import _timstof_to_spectra
        with pytest.raises(KeyError):
            _timstof_to_spectra(pd.DataFrame({"frame": [1], "mz": [100.0]}))

    def test_bruker_slice_logic(self):
        from ptero.spectra._bruker import _bruker_slice
        # MS1, whole run
        assert _bruker_slice(1, None) == (slice(None), slice(None), 0)
        # MS1, frame range (stop exclusive)
        assert _bruker_slice(1, (100, 110)) == (slice(100, 110), slice(None), 0)
        # MS2 selects precursor windows >= 1
        assert _bruker_slice(2, None) == (slice(None), slice(None), slice(1, None))

    def test_bruker_slice_both_levels(self):
        from ptero.spectra._bruker import _bruker_slice
        # ms_level=None reads ALL precursor indices (MS1 survey + every PASEF window)
        assert _bruker_slice(None, None) == (slice(None), slice(None), slice(None))
        assert _bruker_slice(None, (10, 20)) == (slice(10, 20), slice(None), slice(None))

    def test_read_frames_derives_ms_level(self):
        """Combined read (ms_level=None) labels each event from its quad window;
        a forced single level overrides that. Uses a fake reader (no vendor lib)."""
        from ptero.spectra._bruker import _read_timstof_frames

        class FakeReader:
            def __getitem__(self, key):
                return pd.DataFrame({
                    "frame_indices":       [1, 1, 2, 2],
                    "rt_values":           [10., 10., 11., 11.],
                    "mobility_values":     [1.0, 1.1, 1.0, 1.1],
                    "mz_values":           [400., 500., 120., 150.],
                    "intensity_values":    [1e3, 1e3, 5e2, 3e2],
                    "quad_low_mz_values":  [-1., -1., 499., 649.],   # -1 => MS1 (no isolation)
                    "quad_high_mz_values": [-1., -1., 502., 652.],
                })
        df = _read_timstof_frames(FakeReader(), None)
        assert df["ms_level"].tolist() == [1, 1, 2, 2]
        assert df["precursor_mz"].isna().tolist() == [True, True, False, False]
        assert set(_read_timstof_frames(FakeReader(), 1)["ms_level"]) == {1}

    def test_read_frames_combined_to_spectra(self):
        """End-to-end (no vendor lib): a combined read builds mixed MS1+MS2 spectra."""
        from ptero.spectra._bruker import _read_timstof_frames, from_timstof_dataframe

        class FakeReader:
            def __getitem__(self, key):
                return pd.DataFrame({
                    "frame_indices":       [1, 1, 2, 2, 2],
                    "rt_values":           [10., 10., 11., 11., 11.],
                    "mobility_values":     [1.0, 1.1, 1.0, 1.0, 1.1],
                    "mz_values":           [400., 500., 120., 230., 150.],
                    "intensity_values":    [1e3, 2e3, 5e2, 5e2, 3e2],
                    "quad_low_mz_values":  [-1., -1., 499., 499., 649.],
                    "quad_high_mz_values": [-1., -1., 502., 502., 652.],
                })
        sp = from_timstof_dataframe(_read_timstof_frames(FakeReader(), None))
        md = sp.spectra_data(["ms_level", "precursor_mz"])
        assert int((md.ms_level == 1).sum()) == 1      # one MS1 frame
        assert int((md.ms_level == 2).sum()) == 2      # two precursor windows
        assert sorted(round(x, 1) for x in md.precursor_mz.dropna()) == [500.5, 650.5]

    def test_empty_range_raises(self):
        from ptero.spectra import from_timstof_dataframe
        empty = pd.DataFrame({c: pd.Series(dtype="float64")
                              for c in ["frame", "rt", "mobility", "mz", "intensity"]})
        with pytest.raises(ValueError):
            from_timstof_dataframe(empty)


# ── chromatograms (TIC / BPC / XIC) ──────────────────────────────────────────

class TestChromatograms:
    def test_tic_shape_and_values(self, sp4d):
        ms1 = sp4d.filter_ms_level(1)
        tic = ms1.tic()
        assert list(tic.columns) == ["rt", "intensity"]
        assert len(tic) == len(ms1)
        # matches summed peaks via the independent to_points path
        g = ms1.to_points().groupby("scan_index")["intensity"].sum()
        np.testing.assert_allclose(sorted(tic["intensity"]), sorted(g.values))

    def test_bpc_le_tic(self, sp4d):
        ms1 = sp4d.filter_ms_level(1)
        assert (ms1.bpc()["intensity"] <= ms1.tic()["intensity"] + 1e-6).all()

    def test_bpc_matches_max(self, sp4d):
        ms1 = sp4d.filter_ms_level(1)
        g = ms1.to_points().groupby("scan_index")["intensity"].max()
        np.testing.assert_allclose(sorted(ms1.bpc()["intensity"]), sorted(g.values))

    def test_xic_hits_present_mz(self, sp4d):
        P0 = sp4d.peaks_data([0])[0]
        target = float(P0[0, 0])
        assert sp4d.xic(target, mz_tol=0.01)["intensity"].sum() > 0

    def test_xic_absent_mz_is_zero(self, sp4d):
        assert sp4d.xic(99999.0, mz_tol=0.01)["intensity"].sum() == 0

    def test_xic_ppm(self, sp4d):
        target = float(sp4d.peaks_data([0])[0][0, 0])
        assert sp4d.xic(target, ppm=50)["intensity"].sum() > 0

    def test_xic_mobility_window_narrows(self, sp4d):
        P = sp4d.peaks_data([0])[0]
        tmz, tim = float(P[0, 0]), float(sp4d.ion_mobility_peaks([0])[0][0])
        full = sp4d.xic(tmz, mz_tol=0.5)["intensity"].sum()
        narrow = sp4d.xic(tmz, mz_tol=0.5, mobility=(tim - 0.02, tim + 0.02))["intensity"].sum()
        assert narrow <= full + 1e-6

    def test_mobility_window_requires_4d(self):
        peaks = [{"mz": np.array([100.0]), "intensity": np.array([5.0], "f4")}]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [1.0]}), peaks)
        with pytest.raises(ValueError):
            sp.xic(100.0, mobility=(0.8, 1.0))

    def test_backend_independent(self, sp4d, tmp_path):
        ms1 = sp4d.filter_ms_level(1)
        a = ms1.tic()
        z = ms1.set_backend("zarr", path=str(tmp_path / "chrom.zarr")).tic()
        np.testing.assert_allclose(sorted(a["intensity"]), sorted(z["intensity"]))


# ── processing: normalize + m/z binning (queued, composable) ─────────────────

class TestProcessing:
    @pytest.fixture
    def tiny(self):
        peaks = [{"mz": np.array([100., 200., 300.]),
                  "intensity": np.array([10., 40., 50.])}]
        return from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [1.0]}), peaks)

    def test_normalize_max(self, tiny):
        n = tiny.normalize("max")
        np.testing.assert_allclose(n.intensity()[0], [0.2, 0.8, 1.0])

    def test_normalize_sum(self, tiny):
        assert np.isclose(tiny.normalize("sum").intensity()[0].sum(), 1.0)

    def test_normalize_target(self, tiny):
        assert np.isclose(tiny.normalize("max", target=100.0).intensity()[0].max(), 100.0)

    def test_normalize_leaves_mz_and_backend(self, tiny):
        n = tiny.normalize("max")
        np.testing.assert_allclose(n.mz()[0], [100., 200., 300.])
        assert tiny.intensity()[0].max() == 50.0          # backend not mutated

    def test_normalize_zero_intensity_safe(self):
        peaks = [{"mz": np.array([100.]), "intensity": np.array([0.])}]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [1.0]}), peaks)
        assert sp.normalize("max").intensity()[0][0] == 0.0   # no div-by-zero

    def test_normalize_bad_method(self, tiny):
        with pytest.raises(ValueError):
            tiny.normalize("median")

    def test_bin_merges_and_sums(self):
        peaks = [{"mz": np.array([100.1, 100.4, 100.9, 205.0]),
                  "intensity": np.array([10., 30., 20., 5.])}]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [1.0]}), peaks)
        b = sp.bin_mz(1.0, mz_min=100.0)
        assert len(b.mz()[0]) == 2
        np.testing.assert_allclose(sorted(b.intensity()[0]), [5.0, 60.0])
        exp = (100.1*10 + 100.4*30 + 100.9*20) / 60
        assert np.isclose(b.mz()[0][0], exp)               # intensity-weighted m/z

    def test_bin_4d_weighted_mobility(self):
        peaks = [{"mz": np.array([100.2, 100.7, 300.0]),
                  "intensity": np.array([10., 30., 5.]),
                  "ion_mobility": np.array([0.8, 1.2, 0.9])}]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [1.0]}), peaks)
        b = sp.bin_mz(1.0, mz_min=100.0)
        im = b.ion_mobility_peaks([0])[0]
        assert len(im) == len(b.mz()[0])
        assert np.isclose(im[0], (0.8*10 + 1.2*30) / 40)

    def test_bin_bad_width(self, tiny):
        with pytest.raises(ValueError):
            tiny.bin_mz(0.0)

    def test_compose_and_reset(self, sp4d):
        chain = sp4d.filter_ms_level(1).filter_intensity(low=500).normalize("max").bin_mz(0.5)
        assert chain.processing_queue == [
            "filter_intensity(500,inf)", "normalize(max,1.0)", "bin_mz(0.5)"]
        assert chain.reset().processing_queue == []

    def test_normalize_sum_makes_tic_one(self, sp4d):
        tic = sp4d.filter_ms_level(1).normalize("sum").tic()
        np.testing.assert_allclose(tic[tic.intensity > 0]["intensity"], 1.0)

    def test_bin_backend_independent(self, sp4d, tmp_path):
        ms1 = sp4d.filter_ms_level(1)
        a = [len(m) for m in ms1.bin_mz(1.0).mz()]
        z = [len(m) for m in ms1.set_backend("zarr", path=str(tmp_path / "bp.zarr")).bin_mz(1.0).mz()]
        assert a == z


# ── combine spectra + is_centroided ──────────────────────────────────────────

class TestCombine:
    @pytest.fixture
    def four_ms1(self):
        meta = pd.DataFrame({"ms_level": [1, 1, 1, 1], "rtime": [10., 11., 12., 13.]})
        peaks = [{"mz": np.array([500.0]), "intensity": np.array([10.0])} for _ in range(4)]
        return from_dataframe(meta, peaks)

    def test_n_scans_count_and_sum(self, four_ms1):
        c = four_ms1.combine(n_scans=2, agg="sum")
        assert len(c) == 2
        assert [float(x[0]) for x in c.intensity()] == [20.0, 20.0]
        np.testing.assert_allclose(c.rtime, [10.5, 12.5])

    def test_agg_mean(self, four_ms1):
        c = four_ms1.combine(n_scans=2, agg="mean")
        assert [float(x[0]) for x in c.intensity()] == [10.0, 10.0]

    def test_rt_window_count(self, four_ms1):
        assert len(four_ms1.combine(rt_window=2.0)) == 2

    def test_ms_level_and_precursor_kept_separate(self):
        meta = pd.DataFrame({"ms_level": [1, 1, 2, 2, 2], "rtime": [10.] * 5,
                             "precursor_mz": [np.nan, np.nan, 500.0, 500.0, 600.0]})
        peaks = [{"mz": np.array([100.0]), "intensity": np.array([5.0])} for _ in range(5)]
        c = from_dataframe(meta, peaks).combine(rt_window=100.0)
        sd = c.spectra_data()[["ms_level", "precursor_mz"]]
        assert len(c) == 3                              # 1 MS1 group + 2 MS2 precursors
        assert sorted(sd[sd.ms_level == 2]["precursor_mz"]) == [500.0, 600.0]

    def test_4d_merges_co_mobility_peak(self):
        # regression: float32 mobility 0.90 must still merge with 0.901 (round, not floor)
        meta = pd.DataFrame({"ms_level": [1, 1], "rtime": [5., 6.]})
        peaks = [{"mz": np.array([400.001]), "intensity": np.array([7.0]), "ion_mobility": np.array([0.90])},
                 {"mz": np.array([400.004]), "intensity": np.array([9.0]), "ion_mobility": np.array([0.901])}]
        c = from_dataframe(meta, peaks).combine(rt_window=100.0, mz_tol=0.01, mobility_tol=0.01)
        assert len(c.mz()[0]) == 1
        assert np.isclose(c.intensity()[0][0], 16.0)
        assert c.has_4d

    def test_4d_keeps_distinct_mobility_separate(self):
        meta = pd.DataFrame({"ms_level": [1, 1], "rtime": [5., 6.]})
        peaks = [{"mz": np.array([400.0]), "intensity": np.array([7.0]), "ion_mobility": np.array([0.90])},
                 {"mz": np.array([400.0]), "intensity": np.array([9.0]), "ion_mobility": np.array([1.30])}]
        c = from_dataframe(meta, peaks).combine(rt_window=100.0, mz_tol=0.01, mobility_tol=0.01)
        assert len(c.mz()[0]) == 2

    def test_requires_exactly_one_grouping(self, four_ms1):
        with pytest.raises(ValueError):
            four_ms1.combine()
        with pytest.raises(ValueError):
            four_ms1.combine(n_scans=2, rt_window=2.0)

    def test_bad_agg(self, four_ms1):
        with pytest.raises(ValueError):
            four_ms1.combine(n_scans=2, agg="median")

    def test_backend_independent(self, sp4d, tmp_path):
        ms1 = sp4d.filter_ms_level(1)
        a = ms1.combine(n_scans=3)
        z = ms1.set_backend("zarr", path=str(tmp_path / "cmb.zarr")).combine(n_scans=3)
        assert len(a) == len(z)
        np.testing.assert_allclose(sorted(a.tic()["intensity"]), sorted(z.tic()["intensity"]))


class TestCentroidCheck:
    def test_sparse_reads_as_centroided(self):
        mz = np.array([200., 250., 350., 480., 510., 620., 700.9,
                       815., 900.1, 1100., 1300., 1500.])   # sparse, >10 peaks
        peaks = [{"mz": mz, "intensity": np.ones(mz.size)} for _ in range(3)]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1, 1, 1], "rtime": [1., 2., 3.]}), peaks)
        assert sp.is_centroided() is True

    def test_dense_reads_as_profile(self):
        dense = np.arange(400, 460, 0.002)        # 0.002 Da spacing == profile-like
        peaks = [{"mz": dense, "intensity": np.ones(dense.size)} for _ in range(3)]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1, 1, 1], "rtime": [1., 2., 3.]}), peaks)
        assert sp.is_centroided() is False

    def test_empty_defaults_true(self):
        peaks = [{"mz": np.array([]), "intensity": np.array([])}]
        sp = from_dataframe(pd.DataFrame({"ms_level": [1], "rtime": [1.0]}), peaks)
        assert sp.is_centroided() is True


# ── raw-data QC summary ──────────────────────────────────────────────────────

class TestQC:
    @pytest.fixture
    def run(self):
        meta = pd.DataFrame({"ms_level": [1, 2, 1, 2, 1], "rtime": [0., 5., 10., 15., 20.],
                             "precursor_mz": [np.nan, 500., np.nan, 600., np.nan]})
        peaks = [
            {"mz": np.array([100., 200., 300.]), "intensity": np.array([10., 20., 30.])},
            {"mz": np.array([150.]),             "intensity": np.array([5.])},
            {"mz": np.array([400., 500.]),       "intensity": np.array([40., 50.])},
            {"mz": np.array([250., 350.]),       "intensity": np.array([7., 8.])},
            {"mz": np.array([600., 700., 800., 900.]), "intensity": np.array([1., 2., 3., 4.])},
        ]
        return from_dataframe(meta, peaks)

    def test_one_row(self, run):
        q = run.qc()
        assert q.shape[0] == 1

    def test_scan_counts(self, run):
        q = run.qc()
        assert int(q["n_spectra"][0]) == 5
        assert int(q["n_ms1"][0]) == 3
        assert int(q["n_ms2"][0]) == 2
        assert np.isclose(q["ms2_per_ms1"][0], 2 / 3)

    def test_tic_and_peaks(self, run):
        q = run.qc()
        assert q["total_ion_current"][0] == 180.0          # 60+5+90+15+10
        assert q["median_peaks_per_scan"][0] == 2.0        # [3,1,2,2,4] -> 2

    def test_cycle_time(self, run):
        assert run.qc()["median_ms1_cycle_time_s"][0] == 10.0   # MS1 at 0,10,20

    def test_mz_range(self, run):
        q = run.qc()
        assert q["mz_min"][0] == 100.0 and q["mz_max"][0] == 900.0

    def test_3d_has_no_mobility_columns(self, run):
        assert "mobility_min" not in run.qc().columns

    def test_4d_has_mobility_range(self, sp4d):
        q = sp4d.qc()
        assert "mobility_min" in q.columns
        assert q["mobility_min"][0] <= q["mobility_max"][0]

    def test_filter_aware(self, run):
        assert int(run.filter_ms_level(1).qc()["n_ms2"][0]) == 0

    def test_runs_stack(self, run, sp4d):
        table = pd.concat([run.qc(), sp4d.qc()], ignore_index=True)
        assert table.shape[0] == 2

    def test_backend_independent(self, sp4d, tmp_path):
        a = sp4d.qc()
        z = sp4d.set_backend("zarr", path=str(tmp_path / "qc.zarr")).qc()
        assert int(a["n_spectra"][0]) == int(z["n_spectra"][0])
        assert np.isclose(a["total_ion_current"][0], z["total_ion_current"][0])


class TestOpenZarr:
    def test_open_zarr_roundtrip(self, tmp_path):
        import numpy as np
        import pytest
        pytest.importorskip("zarr")
        from ptero.spectra import datasets as ds, open_zarr
        zr = str(tmp_path / "run.zarr")
        original = ds.make_4d(n_features=50, n_ms1_scans=30, with_ms2=True, seed=7)
        original.set_backend("zarr", path=zr)
        run = open_zarr(zr)
        assert len(run) == len(original)
        assert run.has_4d
        # qc reads back consistently
        assert int(run.qc()["n_spectra"][0]) == len(original)

    def test_open_zarr_missing_path(self, tmp_path):
        import pytest
        from ptero.spectra import open_zarr
        with pytest.raises(FileNotFoundError):
            open_zarr(str(tmp_path / "does_not_exist.zarr"))
