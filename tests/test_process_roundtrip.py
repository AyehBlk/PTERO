"""
Round-trip plumbing test for the processed-data Zarr container (Scope A, Step 1).

Asserts the three acceptance criteria from NEXT_STEP_preprocessing.md:

1. the processed store opens via ``ptero.spectra.open_zarr``;
2. for a 4D source it is STILL 4D with per-peak ion mobility intact (the
   non-negotiable constraint), and all peaks + the metadata the rest of PTERO
   relies on round-trip unchanged under the identity transform;
3. the raw<->ID link runs against the processed store and resolves the same
   spectrum, by the same method, as against the raw store.

No real algorithm is exercised here -- only the container + the
read/transform/write pipeline. The mobility-*disambiguation* path of the link is
covered on real data in tests/test_link.py; this test only needs the link to
READ a processed store and behave identically to the raw one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptero.spectra import open_zarr, datasets
from ptero.process import (
    process_store,
    is_processed_store,
    read_processing_provenance,
)


def _make_raw_store(tmp_path: Path, with_im: bool, seed: int = 0) -> str:
    """A small synthetic raw run written to a real PTERO Zarr store."""
    sp = datasets.simulate_run(
        n_features=60, n_ms1_scans=20,
        with_ion_mobility=with_im, with_ms2=True, seed=seed,
    )
    raw = tmp_path / ("raw_4d.zarr" if with_im else "raw_3d.zarr")
    mobility_kwargs = ({
        "peak_ion_mobility_type": "inverse_reduced_ion_mobility",
        "peak_ion_mobility_unit": "V.s/cm2",
    } if with_im else {})
    sp.set_backend("zarr", path=str(raw), **mobility_kwargs)
    return str(raw)


@pytest.mark.parametrize("with_im", [True, False])
def test_identity_roundtrip_preserves_everything(tmp_path, with_im):
    raw = _make_raw_store(tmp_path, with_im)
    proc = str(tmp_path / "processed.zarr")

    out = process_store(raw, proc)                      # default = identity transform
    assert out == proc

    r = open_zarr(raw)
    p = open_zarr(proc)                                 # (1) it opens via open_zarr

    # same size and dimensionality
    assert len(p) == len(r)
    assert p.has_4d == r.has_4d == with_im
    assert tuple(p.peaks_variables) == tuple(r.peaks_variables)

    # (2) every peak round-trips exactly: mz, intensity, and (4D) ion mobility
    cols = list(r.peaks_variables)
    rp = r.peaks_data(None, cols)
    pp = p.peaks_data(None, cols)
    assert len(rp) == len(pp)
    for a, b in zip(rp, pp):
        assert a.shape == b.shape
        np.testing.assert_array_equal(a, b)

    # per-peak mobility specifically intact for 4D (the non-negotiable constraint)
    if with_im:
        rim = r.ion_mobility_peaks()
        pim = p.ion_mobility_peaks()
        assert sum(len(x) for x in pim) > 0             # there really is mobility data
        for mr, mp in zip(rim, pim):
            np.testing.assert_array_equal(mr, mp)

    # metadata round-trips on the columns the rest of PTERO relies on
    keep = ["ms_level", "rtime", "precursor_mz", "ion_mobility"]
    rmeta = r.spectra_data(keep).reset_index(drop=True)
    pmeta = p.spectra_data(keep).reset_index(drop=True)
    pd.testing.assert_frame_equal(rmeta, pmeta, check_dtype=False)


def test_provenance_marker_and_record(tmp_path):
    raw = _make_raw_store(tmp_path, with_im=True)
    proc = str(tmp_path / "processed.zarr")
    process_store(raw, proc)

    # the processed store is flagged; a raw store is not (backward compatible)
    assert is_processed_store(proc) is True
    assert is_processed_store(raw) is False

    prov = read_processing_provenance(proc)
    assert prov is not None
    assert prov["schema_version"] >= 1
    assert Path(prov["source_store"]) == Path(raw)
    assert prov["source_n_spectra"] == len(open_zarr(raw))
    assert [s["name"] for s in prov["steps"]] == ["identity"]

    assert read_processing_provenance(raw) is None


def test_overwrite_guard(tmp_path):
    raw = _make_raw_store(tmp_path, with_im=True)
    proc = str(tmp_path / "processed.zarr")
    process_store(raw, proc)
    with pytest.raises(FileExistsError):
        process_store(raw, proc)                        # refuses to clobber
    process_store(raw, proc, overwrite=True)            # ... unless told to


def test_transform_dropping_mobility_is_rejected_for_4d(tmp_path):
    """A 4D source must stay 4D: a transform that drops mobility fails loudly."""
    raw = _make_raw_store(tmp_path, with_im=True)
    proc = str(tmp_path / "processed.zarr")

    def drop_mobility(peaks):
        return {"mz": peaks["mz"], "intensity": peaks["intensity"]}

    with pytest.raises(ValueError, match="ion_mobility"):
        process_store(raw, proc, transform=drop_mobility)


def test_id_link_runs_against_processed_store(tmp_path):
    raw = _make_raw_store(tmp_path, with_im=True)
    proc = str(tmp_path / "processed.zarr")
    process_store(raw, proc)

    # target a real MS2 spectrum from the store
    r = open_zarr(raw)
    sd = r.spectra_data(["ms_level", "rtime", "precursor_mz"]).reset_index(drop=True)
    ms2 = sd[sd["ms_level"] == 2]
    assert len(ms2) > 0
    row = ms2.iloc[len(ms2) // 2]

    from psm_utils import PSM
    from ptero.ids import RunRegistry, spectrum_for_psm

    psm = PSM(
        peptidoform="PEPTIDEK/2",                       # only used if fragment ranking runs
        spectrum_id="1",
        run="syn",
        retention_time=float(row["rtime"]) / 60.0,      # the link multiplies by 60
        precursor_mz=float(row["precursor_mz"]),
        # ion_mobility left unset (synthetic store has no per-spectrum 1/K0):
        # resolve on RT + m/z, which is all we need to prove the link reads it.
    )

    reg_raw = RunRegistry(); reg_raw.register("syn", raw)
    reg_proc = RunRegistry(); reg_proc.register("syn", proc)

    m_raw = spectrum_for_psm(psm, reg_raw)
    m_proc = spectrum_for_psm(psm, reg_proc)

    # the link resolves the processed store identically to the raw store
    assert m_raw.resolved and m_proc.resolved
    assert m_raw.store_index == m_proc.store_index
    assert m_raw.method == m_proc.method

    # and it resolved to an MS2 spectrum whose peaks match between the two stores
    idx = m_proc.store_index
    assert int(p_ms_level(proc, idx)) == 2
    np.testing.assert_array_equal(
        m_raw.spectrum.mz([0])[0], m_proc.spectrum.mz([0])[0]
    )


def p_ms_level(store_path: str, idx: int) -> int:
    return int(open_zarr(store_path).spectra_data(["ms_level"]).iloc[idx]["ms_level"])