"""
Tests for the mzQC-aligned ID-free QC metrics (ptero.spectra._qc_metrics),
computed per-sample by compute_mzqc_metrics(). Expected values are derived by
following the MsQuality reference implementation's algorithm (source: GitHub
tnaake/MsQuality, R/function_Spectra_metrics.R) on a small controlled run, so
PTERO is checked for agreement with the reference, not just internal sanity.
CV accessions are the confirmed HUPO-PSI mzQC identifiers from that source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptero.spectra import from_dataframe
from ptero.spectra._qc_metrics import (
    compute_mzqc_metrics, accession_for, MZQC_ACCESSIONS,
)


@pytest.fixture
def controlled():
    # 4 MS1 (rt 0,10,20,30; TIC 100,200,300,400) + 4 MS2 (rt 5,15,25,35).
    # MS2 precursors: m/z 400,500,600,700; charge 2,2,3,1; intensity 1000,2000,3000,4000.
    # The last MS2 has no peaks -> one empty MS2 scan.
    meta, peaks = [], []
    for rt, tic in [(0, 100.0), (10, 200.0), (20, 300.0), (30, 400.0)]:
        meta.append({"ms_level": 1, "rtime": float(rt), "scan_index": len(meta),
                     "precursor_mz": np.nan, "precursor_charge": pd.NA,
                     "precursor_intensity": np.nan})
        peaks.append({"mz": np.array([400., 500.]),
                      "intensity": np.array([tic * 0.4, tic * 0.6], dtype="f4")})
    for j, (rt, pmz, ch, pint) in enumerate(
            [(5, 400., 2, 1000.), (15, 500., 2, 2000.), (25, 600., 3, 3000.), (35, 700., 1, 4000.)]):
        meta.append({"ms_level": 2, "rtime": float(rt), "scan_index": len(meta),
                     "precursor_mz": pmz, "precursor_charge": ch, "precursor_intensity": pint})
        peaks.append({"mz": np.array([]), "intensity": np.array([], dtype="f4")} if j == 3
                     else {"mz": np.array([120., 140.]), "intensity": np.array([5., 7.], dtype="f4")})
    sp = from_dataframe(pd.DataFrame(meta), peaks)
    tbl = compute_mzqc_metrics(sp)
    return tbl.set_index("metric")["value"]


def test_tic_and_chromatography_metrics(controlled):
    g = controlled
    assert g["areaUnderTic"] == pytest.approx(1000.0)
    assert g["chromatographyDuration"] == pytest.approx(35.0)        # whole-run span (MS1+MS2)
    # MsQuality bins by rtime QUANTILES and excludes the min-RT scan (rt > 0%):
    # quantile([0,10,20,30]) = [0,7.5,15,22.5,30]; the rt=0 scan (TIC 100) drops out.
    assert [g[f"areaUnderTicRtQuantiles_{q}"] for q in ("Q1", "Q2", "Q3", "Q4")] == \
        pytest.approx([0.0, 200.0, 300.0, 400.0])
    # rtOverMsQuarters: end-of-quarter RT over the WHOLE-run duration (35), not MS1 span.
    assert [g[f"rtOverMsQuarters_{q}"] for q in ("Q1", "Q2", "Q3", "Q4")] == \
        pytest.approx([0.0, 10 / 35, 20 / 35, 30 / 35])


def test_count_and_precursor_metrics(controlled):
    g = controlled
    assert g["numberEmptyScansMS1"] == 0
    assert g["numberEmptyScansMS2"] == 1
    assert g["medianPrecursorMz"] == pytest.approx(550.0)
    assert g["precursorIntensityMean"] == pytest.approx(2500.0)
    assert [g[f"precursorIntensityQuartiles_{q}"] for q in ("Q1", "Q2", "Q3")] == \
        pytest.approx([1750.0, 2500.0, 3250.0])


def test_charge_metrics(controlled):
    g = controlled
    assert g["meanCharge"] == pytest.approx(2.0)
    assert g["medianCharge"] == pytest.approx(2.0)
    assert g["ratioCharge1over2"] == pytest.approx(0.5)   # one z=1 over two z=2
    assert g["ratioCharge3over2"] == pytest.approx(0.5)   # one z=3 over two z=2


def test_cv_accessions_attached():
    meta, peaks = [], []
    for rt in (0.0, 10.0, 20.0, 30.0):                     # 4 MS1 -> quartiles defined
        meta.append({"ms_level": 1, "rtime": rt, "scan_index": len(meta),
                     "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan})
        peaks.append({"mz": np.array([400., 500.]), "intensity": np.array([1., 2.], dtype="f4")})
    for rt, pmz, ch, pint in [(33.0, 500.0, 2, 1000.0), (35.0, 600.0, 1, 2000.0)]:
        meta.append({"ms_level": 2, "rtime": rt, "scan_index": len(meta),
                     "precursor_mz": pmz, "precursor_charge": ch, "precursor_intensity": pint})
        peaks.append({"mz": np.array([100.]), "intensity": np.array([3.], dtype="f4")})
    tbl = compute_mzqc_metrics(from_dataframe(pd.DataFrame(meta), peaks))
    acc = dict(zip(tbl["metric"], tbl["accession"]))
    assert acc["areaUnderTic"] == "MS:4000155"
    assert acc["areaUnderTicRtQuantiles_Q1"] == "MS:4000156"
    assert acc["rtOverMsQuarters_Q1"] == "MS:4000055"
    assert acc["chromatographyDuration"] == "MS:4000053"
    assert acc["mzAcquisitionRange_min"] == "MS:4000069"
    assert acc["rtAcquisitionRange_min"] == "MS:4000070"
    assert acc["numberEmptyScansMS1"] == "MS:4000099"
    assert acc["numberEmptyScansMS2"] == "MS:4000100"
    assert acc["precursorIntensityMean"] == "MS:4000117"
    assert acc["precursorIntensitySd"] == "MS:4000118"
    assert acc["precursorIntensityRange_min"] == "MS:4000160"
    assert acc["precursorIntensityQuartiles_Q1"] == "MS:4000116"
    assert acc["meanCharge"] == "MS:4000173"
    assert acc["medianCharge"] == "MS:4000175"
    assert acc["ratioCharge1over2"] == "MS:4000167"
    # n-tuple suffix resolves to the base metric's accession
    assert accession_for("areaUnderTicRtQuantiles_Q3") == "MS:4000156"
    assert accession_for("mzPeakRange_min") == ""        # PTERO extension, no CV term


def test_no_guessed_accessions():
    for v in MZQC_ACCESSIONS.values():
        assert v == "" or v.startswith("MS:4000")


def test_respects_view_ms1_only():
    meta = [{"ms_level": 1, "rtime": 0.0, "scan_index": 0,
             "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan},
            {"ms_level": 2, "rtime": 1.0, "scan_index": 1,
             "precursor_mz": 500.0, "precursor_charge": 2, "precursor_intensity": 1000.0}]
    peaks = [{"mz": np.array([400., 500.]), "intensity": np.array([1., 2.], dtype="f4")},
             {"mz": np.array([100.]), "intensity": np.array([3.], dtype="f4")}]
    sp = from_dataframe(pd.DataFrame(meta), peaks)
    g = compute_mzqc_metrics(sp.filter_ms_level(1)).set_index("metric")["value"]
    assert "areaUnderTic" in g.index            # MS1 metric present
    assert "medianCharge" not in g.index        # no MS2 -> charge/precursor absent
    assert "precursorIntensityMean" not in g.index


def test_sample_column_and_stacking():
    meta = [{"ms_level": 1, "rtime": 0.0, "scan_index": 0,
             "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan},
            {"ms_level": 2, "rtime": 1.0, "scan_index": 1,
             "precursor_mz": 500.0, "precursor_charge": 2, "precursor_intensity": 1000.0}]
    peaks = [{"mz": np.array([400., 500.]), "intensity": np.array([1., 2.], dtype="f4")},
             {"mz": np.array([100.]), "intensity": np.array([3.], dtype="f4")}]
    sp = from_dataframe(pd.DataFrame(meta), peaks)
    a = compute_mzqc_metrics(sp, sample_name="a")
    b = compute_mzqc_metrics(sp, sample_name="b")
    assert a.columns[0] == "sample"
    wide = pd.concat([a, b], ignore_index=True).pivot_table(
        index="sample", columns="metric", values="value")
    assert wide.shape[0] == 2 and "areaUnderTic" in wide.columns


# ── cross-sample comparison ────────────────────────────────────────────────

def _sample(seed, scale):
    """Small DDA-like memory store; `scale` multiplies MS1 intensities."""
    rng = np.random.default_rng(seed)
    meta, peaks = [], []
    for i in range(12):
        lvl = 1 if i % 3 == 0 else 2
        rt = float(i * 4)
        if lvl == 1:
            meta.append({"ms_level": 1, "rtime": rt, "scan_index": i,
                         "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan})
            mz = np.sort(rng.uniform(400, 1200, 80))
            inten = (rng.uniform(1e3, 1e6, 80) * scale).astype("f4")
        else:
            meta.append({"ms_level": 2, "rtime": rt, "scan_index": i,
                         "precursor_mz": float(rng.uniform(400, 1000)),
                         "precursor_charge": int(rng.choice([2, 2, 3, 1])),
                         "precursor_intensity": float(rng.uniform(1e4, 1e7))})
            mz = np.sort(rng.uniform(200, 1400, 20))
            inten = rng.uniform(1e2, 5e4, 20).astype("f4")
        peaks.append({"mz": mz, "intensity": inten})
    return from_dataframe(pd.DataFrame(meta), peaks)


def test_compare_columns_and_order():
    from ptero.spectra._qc_metrics import compare_samples
    opened = {"b": _sample(1, 1.0), "a": _sample(2, 1.0)}   # deliberately not alphabetical
    out = compare_samples(opened)
    assert list(out.columns)[:3] == ["metric", "accession", "ms_level"]
    assert list(out.columns)[3:5] == ["b", "a"]             # input order preserved
    assert out.columns[-1] == "cv_pct"
    assert out["metric"].iloc[0] == "areaUnderTic"          # first-seen metric order


def test_compare_cv_flags_intensity_outlier():
    from ptero.spectra._qc_metrics import compare_samples
    out = compare_samples({"x": _sample(1, 1.0), "y": _sample(2, 1.0), "z": _sample(3, 5.0)})
    cv = out.set_index("metric")["cv_pct"]
    assert cv["areaUnderTic"] > 50            # one sample 5x more intense -> high spread
    # an acquisition-shape metric stays ~invariant across the three runs
    assert pd.isna(cv["rtOverMsQuarters_Q4"]) or cv["rtOverMsQuarters_Q4"] < 1


def test_compare_single_sample_has_no_cv():
    from ptero.spectra._qc_metrics import compare_samples
    out = compare_samples({"only": _sample(1, 1.0)})
    assert "cv_pct" not in out.columns
    assert "only" in out.columns


# ── MS1 peak-intensity distribution (PTERO extension; no CV accession) ──────

def test_ms1_intensity_distribution_values():
    # two MS1 scans; pooled MS1 intensities = 1..8 -> known stats
    meta = [{"ms_level": 1, "rtime": 0.0, "scan_index": 0,
             "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan},
            {"ms_level": 1, "rtime": 10.0, "scan_index": 1,
             "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan}]
    peaks = [{"mz": np.array([100., 200, 300, 400]), "intensity": np.array([1, 2, 3, 4], dtype="f4")},
             {"mz": np.array([100., 200, 300, 400]), "intensity": np.array([5, 6, 7, 8], dtype="f4")}]
    g = compute_mzqc_metrics(from_dataframe(pd.DataFrame(meta), peaks)).set_index("metric")["value"]
    assert g["ms1IntensityMean"] == pytest.approx(4.5)
    assert g["ms1IntensityRange_min"] == pytest.approx(1.0)
    assert g["ms1IntensityRange_max"] == pytest.approx(8.0)
    assert [g[f"ms1IntensityQuartiles_{q}"] for q in ("Q1", "Q2", "Q3")] == \
        pytest.approx([2.75, 4.5, 6.25])
    assert g["ms1IntensitySd"] == pytest.approx(np.std(np.arange(1, 9)), abs=1e-3)  # population sd


def test_ms1_intensity_is_extension_without_cv_accession():
    assert accession_for("ms1IntensityMean") == ""
    assert accession_for("ms1IntensityQuartiles_Q1") == ""
    meta = [{"ms_level": 1, "rtime": 0.0, "scan_index": 0,
             "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan}]
    peaks = [{"mz": np.array([100., 200.]), "intensity": np.array([10., 20.], dtype="f4")}]
    tbl = compute_mzqc_metrics(from_dataframe(pd.DataFrame(meta), peaks))
    fam = tbl[tbl["metric"].str.startswith("ms1Intensity")]
    assert len(fam) >= 4
    assert (fam["accession"] == "").all()          # never a guessed CV id
    assert (fam["ms_level"] == 1).all()


# ── per-sample outlier flagging ─────────────────────────────────────────────

def _cmp_matrix():
    return pd.DataFrame({
        "metric":    ["mA", "mB", "mC"],
        "accession": ["MS:1", "", ""],
        "ms_level":  [1, 2, 1],
        "s1": [100.0, 100.0, 50.0],
        "s2": [102.0, 100.0, 51.0],
        "s3": [98.0, 100.0, 49.0],
        "s4": [400.0, 130.0, 50.5],     # mA huge; mB off vs 3 tied; mC normal
        "cv_pct": [1.0, 1.0, 1.0],
    })


def test_flag_detects_clear_outlier_by_both():
    from ptero.spectra._qc_metrics import flag_outliers
    fl = flag_outliers(_cmp_matrix(), ["s1", "s2", "s3", "s4"])
    row = fl[(fl["metric"] == "mA") & (fl["sample"] == "s4")].iloc[0]
    assert row["flag_z"] and row["flag_iqr"] and row["direction"] == "high"


def test_flag_mad_zero_uses_meanad_fallback_no_inf():
    from ptero.spectra._qc_metrics import flag_outliers
    fl = flag_outliers(_cmp_matrix(), ["s1", "s2", "s3", "s4"])
    assert np.isfinite(fl["robust_z"]).all()          # MAD==0 -> meanAD, never inf
    mb = fl[(fl["metric"] == "mB") & (fl["sample"] == "s4")].iloc[0]
    assert mb["flag_iqr"]                              # IQR catches the 3-tied + 1-off case
    assert not mb["flag_z"]                            # z (meanAD fallback) is conservative here


def test_flag_no_false_positive_on_tight_metric():
    from ptero.spectra._qc_metrics import flag_outliers
    fl = flag_outliers(_cmp_matrix(), ["s1", "s2", "s3", "s4"])
    assert (fl["metric"] == "mC").sum() == 0


def test_flag_needs_min_samples():
    from ptero.spectra._qc_metrics import flag_outliers
    fl = flag_outliers(_cmp_matrix()[["metric", "accession", "ms_level", "s1", "s2"]], ["s1", "s2"])
    assert fl.empty                                    # cannot flag an outlier among two


def test_flag_summary_ranks_worst_first():
    from ptero.spectra._qc_metrics import flag_outliers, flag_summary
    samples = ["s1", "s2", "s3", "s4"]
    summary = flag_summary(flag_outliers(_cmp_matrix(), samples), samples)
    assert summary.iloc[0]["sample"] == "s4"          # most-flagged first
    assert summary.iloc[0]["n_flag_iqr"] >= 2


# ── MsQuality-alignment regressions (found via source cross-check) ──────────

def test_mz_acquisition_range_is_precursor_not_peaks():
    # MsQuality's mzAcquisitionRange = range(precursorMz) of MS2; the peak m/z
    # extent is a separate PTERO extension (mzPeakRange).
    meta = [{"ms_level": 1, "rtime": 0., "scan_index": 0,
             "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan},
            {"ms_level": 2, "rtime": 1., "scan_index": 1,
             "precursor_mz": 500., "precursor_charge": 2, "precursor_intensity": 1e3},
            {"ms_level": 2, "rtime": 2., "scan_index": 2,
             "precursor_mz": 900., "precursor_charge": 2, "precursor_intensity": 2e3}]
    peaks = [{"mz": np.array([100., 1800.]), "intensity": np.array([5., 5.], dtype="f4")},
             {"mz": np.array([300.]), "intensity": np.array([1.], dtype="f4")},
             {"mz": np.array([400.]), "intensity": np.array([1.], dtype="f4")}]
    g = compute_mzqc_metrics(from_dataframe(pd.DataFrame(meta), peaks)).set_index("metric")["value"]
    assert g["mzAcquisitionRange_min"] == pytest.approx(500.)   # precursor range
    assert g["mzAcquisitionRange_max"] == pytest.approx(900.)
    assert g["mzPeakRange_min"] == pytest.approx(100.)          # peak extent (extension)
    assert g["mzPeakRange_max"] == pytest.approx(1800.)


def test_charge_ratio_omitted_when_charge_absent():
    # all precursors charge 2 -> no 1/3/4 present -> ratios omitted (MsQuality NaN),
    # never reported as 0.
    meta = [{"ms_level": 2, "rtime": float(i), "scan_index": i,
             "precursor_mz": 500., "precursor_charge": 2, "precursor_intensity": 1e3}
            for i in range(3)]
    peaks = [{"mz": np.array([100.]), "intensity": np.array([1.], dtype="f4")} for _ in range(3)]
    g = compute_mzqc_metrics(from_dataframe(pd.DataFrame(meta), peaks)).set_index("metric")["value"]
    assert "ratioCharge1over2" not in g.index
    assert "ratioCharge3over2" not in g.index
    assert g["medianCharge"] == pytest.approx(2.0)


def test_empty_scan_counts_zero_tic_scan():
    # a scan with peaks but all-zero intensity is empty by MsQuality's sum()==0 rule.
    meta = [{"ms_level": 1, "rtime": 0., "scan_index": 0,
             "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan},
            {"ms_level": 1, "rtime": 1., "scan_index": 1,
             "precursor_mz": np.nan, "precursor_charge": pd.NA, "precursor_intensity": np.nan}]
    peaks = [{"mz": np.array([100., 200.]), "intensity": np.array([5., 7.], dtype="f4")},
             {"mz": np.array([100., 200.]), "intensity": np.array([0., 0.], dtype="f4")}]  # zero-TIC
    g = compute_mzqc_metrics(from_dataframe(pd.DataFrame(meta), peaks)).set_index("metric")["value"]
    assert g["numberEmptyScansMS1"] == 1


# ── effect-size gate on flagging (near-constant-metric artifact) ────────────

def _near_constant_matrix():
    # an acquisition setting: 5 runs identical, 1 differs by 0.08% (real data
    # artifact -> MAD collapses, |z| explodes); plus a real ~12%-high metric and
    # an empty-scan jump from a zero median.
    return pd.DataFrame({
        "metric":    ["mzAcquisitionRange_min", "ms1IntensitySd", "numberEmptyScansMS1"],
        "accession": ["MS:4000069", "", "MS:4000099"],
        "ms_level":  [2, 1, 1],
        "s1": [398.2, 6.09e6, 0.0],     # ms1IntensitySd ~12% high (real signal)
        "s2": [398.2, 5.46e6, 0.0],
        "s3": [398.2, 5.47e6, 0.0],
        "s4": [398.2, 5.45e6, 0.0],
        "s5": [398.2, 5.42e6, 5000.0],  # empty-scan jump from a zero median
        "s6": [397.9, 5.44e6, 0.0],     # 0.08% setting difference (artifact)
    })


def test_flag_gate_suppresses_near_constant_setting():
    from ptero.spectra._qc_metrics import flag_outliers
    cols = ["s1", "s2", "s3", "s4", "s5", "s6"]
    cmp = _near_constant_matrix()
    # default gate (5%): the 0.08% setting difference must NOT be flagged
    fl = flag_outliers(cmp, cols)
    assert (fl["metric"] == "mzAcquisitionRange_min").sum() == 0
    # without the gate it WOULD flag -> proves the gate is what suppresses it
    raw = flag_outliers(cmp, cols, min_rel_dev=0.0)
    assert (raw["metric"] == "mzAcquisitionRange_min").sum() > 0


def test_flag_gate_keeps_real_deviation():
    from ptero.spectra._qc_metrics import flag_outliers
    cols = ["s1", "s2", "s3", "s4", "s5", "s6"]
    fl = flag_outliers(_near_constant_matrix(), cols)
    hit = fl[(fl["metric"] == "ms1IntensitySd") & (fl["sample"] == "s1")]
    assert len(hit) == 1 and hit.iloc[0]["rel_dev"] > 0.05   # ~12% deviation survives


def test_flag_gate_catches_zero_median_jump():
    from ptero.spectra._qc_metrics import flag_outliers
    cols = ["s1", "s2", "s3", "s4", "s5", "s6"]
    # median empty-scan count is 0, so a jump to 5000 has effectively infinite
    # relative deviation -> the gate keeps it (a CV/relative filter must not hide this)
    fl = flag_outliers(_near_constant_matrix(), cols)
    hit = fl[(fl["metric"] == "numberEmptyScansMS1") & (fl["sample"] == "s5")]
    assert len(hit) == 1 and np.isfinite(hit.iloc[0]["rel_dev"])


def test_flag_output_has_rel_dev_column():
    from ptero.spectra._qc_metrics import flag_outliers
    fl = flag_outliers(_cmp_matrix(), ["s1", "s2", "s3", "s4"])
    assert "rel_dev" in fl.columns
    assert (fl["rel_dev"] >= 0.05).all()      # everything that survives clears the gate


def test_flag_summary_ranks_iqr_only_above_clean():
    from ptero.spectra._qc_metrics import flag_outliers, flag_summary
    # one IQR-only outlier (16% low) among otherwise clean samples must rank first
    cmp = pd.DataFrame({
        "metric": ["precursorIntensityRange_min"],
        "accession": ["MS:4000160"], "ms_level": [2],
        "A1": [22000.0], "A2": [23300.0], "A3": [24400.0], "A4": [23700.0],
        "C1": [19900.0], "C2": [24800.0], "C3": [23600.0], "C4": [24500.0],
    })
    cols = ["A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4"]
    summ = flag_summary(flag_outliers(cmp, cols), cols)
    assert summ.iloc[0]["sample"] == "C1"          # the lone IQR flag rises to the top
    assert summ.iloc[0]["n_flag_iqr"] == 1