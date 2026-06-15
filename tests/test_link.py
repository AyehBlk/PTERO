"""Property-based raw<->ID link: a PSM is located in the raw store by
(retention time, precursor m/z, ion mobility). This encodes the contract
verified on real MaxQuant timsTOF data: RT + m/z are not unique on a dense
run (co-eluting precursors), and ion mobility (1/K0) is the coordinate that
disambiguates to the correct spectrum.
"""
import numpy as np
import pandas as pd

from ptero.spectra._bruker import from_timstof_dataframe


def _synthetic_store():
    """Build a tiny 4D timsTOF PteroSpectra with two co-eluting MS2 precursors
    at the SAME rt and nearly the same precursor m/z, separated only by 1/K0.
    """
    rows = []
    # MS1 frame
    for mz in (400.0, 500.0, 600.0):
        rows.append(dict(frame=1, rt=300.0, mobility=0.90, mz=mz, intensity=1000.0))
    # Two MS2 precursors in frame 2: same rt, m/z 0.1 Th apart, 1/K0 well separated
    for mz in (150.0, 250.0, 350.0):                       # precursor A: low mobility
        rows.append(dict(frame=2, rt=300.5, mobility=0.80, mz=mz, intensity=500.0, precursor_mz=620.50))
    for mz in (180.0, 280.0, 380.0):                       # precursor B: high mobility
        rows.append(dict(frame=2, rt=300.5, mobility=1.15, mz=mz, intensity=400.0, precursor_mz=620.60))
    return from_timstof_dataframe(pd.DataFrame(rows))


def _match(ms2, rt_sec, mz, im, rt_tol=3.0, mz_tol=1.5, im_tol=0.05):
    cand = ms2[(np.abs(ms2["rtime"] - rt_sec) <= rt_tol)
               & (np.abs(ms2["precursor_mz"] - mz) <= mz_tol)]
    if im is not None and len(cand) > 1:
        cand = cand[np.abs(cand["ion_mobility"] - im) <= im_tol]
    return cand


def test_store_has_ms2_mobility():
    """Every MS2 spectrum carries a spectrum-level ion_mobility (the link's 3rd coordinate)."""
    df = _synthetic_store().spectra_data()
    ms2 = df[df.ms_level == 2]
    assert len(ms2) == 2
    assert ms2["ion_mobility"].notna().all()
    # the median-1/K0 we store equals the precursor's mobility
    got = sorted(round(v, 2) for v in ms2["ion_mobility"])
    assert got == [0.80, 1.15]


def test_rt_mz_alone_is_ambiguous():
    """RT + m/z cannot separate the two co-eluting precursors; this is WHY mobility is needed."""
    ms2 = _synthetic_store().spectra_data().query("ms_level == 2")
    cand = _match(ms2, rt_sec=300.5, mz=620.55, im=None)   # no mobility filter
    assert len(cand) == 2                                   # ambiguous, as on real data


def test_mobility_disambiguates_to_one():
    """Adding the 1/K0 coordinate resolves to exactly the right spectrum."""
    ms2 = _synthetic_store().spectra_data().query("ms_level == 2")
    # PSM measured at 1/K0 = 0.80 must resolve to precursor A (mz 620.50)
    a = _match(ms2, rt_sec=300.5, mz=620.55, im=0.80)
    assert len(a) == 1
    assert abs(a.iloc[0]["precursor_mz"] - 620.50) < 0.01
    # PSM measured at 1/K0 = 1.15 must resolve to precursor B (mz 620.60)
    b = _match(ms2, rt_sec=300.5, mz=620.55, im=1.15)
    assert len(b) == 1
    assert abs(b.iloc[0]["precursor_mz"] - 620.60) < 0.01