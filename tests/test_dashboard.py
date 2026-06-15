"""Smoke tests for the dashboard. Skipped if the viz stack isn't installed."""
import os
from pathlib import Path
import pytest

pytest.importorskip("panel")
pytest.importorskip("datashader")
pytest.importorskip("holoviews")

from ptero.spectra import datasets as ds, PteroSpectra, MsBackendZarr
from ptero.dashboard import build_app, convert_to_zarr, Dashboard


def test_convert_mzml_to_zarr(tmp_path):
    mz = str(tmp_path / "r.mzML"); zr = str(tmp_path / "r.zarr")
    ds.write_mzml(mz, ds.make_3d(n_features=40, n_ms1_scans=20, with_ms2=True))
    out = convert_to_zarr(mz, zr, ms_level=None)
    sp = PteroSpectra(MsBackendZarr(out))
    assert len(sp) > 0
    assert sp.tic().shape[1] == 2


def test_build_app_constructs():
    app = build_app()
    assert app is not None


def test_dashboard_loads_and_renders_4d(tmp_path):
    zr = str(tmp_path / "r4.zarr")
    ds.make_4d(n_features=60, n_ms1_scans=40, with_ms2=True, seed=1).set_backend("zarr", path=zr)
    dash = Dashboard()
    dash._set_data(PteroSpectra(MsBackendZarr(zr)), "test")   # also triggers _render
    assert len(dash.plot_panel) > 0
    sel = dash._current_selection()
    assert len(sel) > 0


def _stores_equal(a_path, b_path):
    import numpy as np
    from ptero.spectra import PteroSpectra, MsBackendZarr
    A = PteroSpectra(MsBackendZarr(a_path)); B = PteroSpectra(MsBackendZarr(b_path))
    assert len(A) == len(B)
    pa = A.to_points().sort_values(["rt", "mz"]).reset_index(drop=True)
    pb = B.to_points().sort_values(["rt", "mz"]).reset_index(drop=True)
    assert len(pa) == len(pb)
    cols = ["rt", "mz", "intensity"] + (["ion_mobility"] if "ion_mobility" in pa.columns else [])
    for c in cols:
        assert np.allclose(pa[c], pb[c], rtol=1e-5, atol=1e-3)


def test_stream_writer_matches_oneshot_4d(tmp_path):
    from ptero.spectra import datasets as ds, PteroSpectra, MsBackendZarr
    from ptero.spectra.backends.zarr_backend import _ZarrStreamWriter
    sp = ds.make_4d(n_features=120, n_ms1_scans=60, with_ms2=True, seed=3)
    a = str(tmp_path / "oneshot.zarr"); sp.set_backend("zarr", path=a)
    b = str(tmp_path / "streamed.zarr")
    w = _ZarrStreamWriter(b, has_4d=True); n = len(sp)
    for s in range(0, n, 17):                     # odd batch size stresses boundaries
        w.append_backend(sp.backend.subset(range(s, min(s + 17, n))))
    w.finalize()
    _stores_equal(a, b)


def test_convert_mzml_streaming_matches_oneshot(tmp_path):
    from ptero.spectra import datasets as ds, from_mzml
    from ptero.dashboard import convert_to_zarr
    mz = str(tmp_path / "r.mzML")
    ds.write_mzml(mz, ds.make_3d(n_features=80, n_ms1_scans=40, with_ms2=True))
    a = str(tmp_path / "oneshot.zarr"); from_mzml(mz, lazy=False).set_backend("zarr", path=a)
    b = str(tmp_path / "streamed.zarr"); convert_to_zarr(mz, b, ms_level=None, spectrum_batch=13)
    _stores_equal(a, b)


# ── batch conversion: input-resolution + output-planning (fast, no subprocess) ──

def test_resolve_batch_inputs_list_and_delimited():
    from ptero.dashboard.convert import resolve_batch_inputs
    assert resolve_batch_inputs(["a.d", " b.mzML "]) == ["a.d", "b.mzML"]
    assert resolve_batch_inputs("a.d, b.d") == ["a.d", "b.d"]
    assert resolve_batch_inputs("a.d\nb.d\n\n") == ["a.d", "b.d"]
    assert resolve_batch_inputs("   ") == []


def test_resolve_batch_inputs_folder_scan(tmp_path):
    from ptero.dashboard.convert import resolve_batch_inputs
    (tmp_path / "s1.mzML").write_text("x", encoding="utf-8")
    (tmp_path / "s2.mzXML").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    d = tmp_path / "run3.d"; d.mkdir(); (d / "analysis.tdf").write_text("x", encoding="utf-8")
    found = {Path(p).name for p in resolve_batch_inputs(str(tmp_path))}
    assert found == {"s1.mzML", "s2.mzXML", "run3.d"}      # .txt excluded, .d folder included


def test_resolve_batch_inputs_single_d_folder(tmp_path):
    # a path that *is* a .d run must be treated as one input, not scanned into
    from ptero.dashboard.convert import resolve_batch_inputs
    d = tmp_path / "run.d"; d.mkdir(); (d / "analysis.tdf").write_text("x", encoding="utf-8")
    assert resolve_batch_inputs(str(d)) == [str(d)]


def test_plan_outputs_unique_names(tmp_path):
    from ptero.dashboard.convert import _plan_outputs
    plan = _plan_outputs(["/x/run.d", "/y/run.d", "/z/sample.mzML"], tmp_path)
    outs = [Path(o).name for _, o in plan]
    assert outs == ["run.zarr", "run_2.zarr", "sample.zarr"]   # collision disambiguated
