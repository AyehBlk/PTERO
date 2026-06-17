"""Tests for ptero.ids -- reading PSMs, the run registry, the property-based
raw<->ID match (spectrum_for_psm), and MS2 fragment annotation.

Synthetic-first: raw stores are built with ``from_timstof_dataframe`` (same as
tests/test_link.py); PSMs are constructed directly. No real files or instrument.
The reading/mobility/scoring logic here was cross-checked against psm_utils and
spectrum_utils directly; these tests wire it through the real PteroSpectra store.
"""
import numpy as np
import pandas as pd
import pytest

from pyteomics import mass
from psm_utils import PSM

from ptero.spectra._bruker import from_timstof_dataframe
from ptero import ids


# ── helpers ──────────────────────────────────────────────────────────────────

def _by_ions(seq, charge=1):
    """Theoretical singly-charged b/y fragment m/z for an unmodified peptide."""
    out = []
    for i in range(1, len(seq)):
        out.append(mass.fast_mass(seq[:i], ion_type="b", charge=charge))
        out.append(mass.fast_mass(seq[i:], ion_type="y", charge=charge))
    return np.array(sorted(out))


def _two_precursor_store(*, mz_a=620.50, mz_b=620.60, im_a=0.80, im_b=1.15,
                         peaks_a=None, peaks_b=None, rt=300.5):
    """One MS1 frame + two co-eluting MS2 precursors (A, B) at the same RT.

    Each MS2's peaks / mobility / precursor m/z is configurable, so a test can
    separate A and B by mobility OR by fragment content.
    """
    rows = []
    for mz in (400.0, 500.0, 600.0):                       # MS1 frame -> precursor_mz NaN
        rows.append(dict(frame=1, rt=rt - 0.5, mobility=0.90, mz=mz, intensity=1000.0))
    peaks_a = np.array([150.0, 250.0, 350.0]) if peaks_a is None else np.asarray(peaks_a)
    peaks_b = np.array([180.0, 280.0, 380.0]) if peaks_b is None else np.asarray(peaks_b)
    for mz in peaks_a:
        rows.append(dict(frame=2, rt=rt, mobility=im_a, mz=float(mz),
                         intensity=500.0, precursor_mz=mz_a))
    for mz in peaks_b:
        rows.append(dict(frame=2, rt=rt, mobility=im_b, mz=float(mz),
                         intensity=400.0, precursor_mz=mz_b))
    return from_timstof_dataframe(pd.DataFrame(rows))


def _ms2_index_near(store, precursor_mz):
    """Store position of the MS2 spectrum whose precursor m/z is closest."""
    sd = store.spectra_data(["ms_level", "precursor_mz"])
    ms2 = sd[sd["ms_level"] == 2]
    return int(ms2.index[int(np.argmin(np.abs(ms2["precursor_mz"] - precursor_mz)))])


def _write_tsv(path, columns, rows):
    with open(path, "w", encoding="utf-8-sig") as fh:      # WITH BOM, like MaxQuant on Windows
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def _psm(run="synth", mz=620.55, rt_sec=300.5, im=None, pep="PEPTIDEK/2"):
    # retention_time is stored in MINUTES (MaxQuant convention); spectrum_for_psm x60s it
    return PSM(peptidoform=pep, spectrum_id="1", run=run,
               precursor_mz=mz, retention_time=rt_sec / 60.0, ion_mobility=im)


# ── reading PSMs (no store needed) ─────────────────────────────────────────────

def test_read_psms_strips_bom_and_reads_minutes(tmp_path):
    cols = ["Raw file", "Scan number", "Charge", "Modified sequence", "Proteins",
            "m/z", "Reverse", "Retention time", "PEP", "Score"]
    _write_tsv(tmp_path / "msms.txt", cols,
               [["synth", "101", "2", "_PEPTIDEK_", "P1", "620.55", "", "5.0083", "0.001", "99.0"]])
    psms = ids.read_psms(tmp_path / "msms.txt")
    assert len(psms) == 1
    p = psms[0]
    assert p.run == "synth"
    assert p.peptidoform.proforma == "PEPTIDEK/2"
    assert p.precursor_mz == pytest.approx(620.55)
    assert p.retention_time == pytest.approx(5.0083)       # MINUTES, verbatim
    assert p.ion_mobility is None                          # not present in msms.txt


def test_attach_mobility_from_maxquant(tmp_path):
    cols = ["Raw file", "Scan number", "Charge", "Modified sequence", "Proteins",
            "m/z", "Reverse", "Retention time", "PEP", "Score"]
    _write_tsv(tmp_path / "msms.txt", cols,
               [["synth", "101", "2", "_PEPTIDEK_", "P1", "620.55", "", "5.0", "0.001", "99"],
                ["synth", "102", "3", "_NDVPLVGGK_", "P2", "301.51", "", "5.0", "0.002", "80"]])
    psms = ids.read_psms(tmp_path / "msms.txt")

    acc = ["Raw file", "Scan number", "Sequence", "m/z", "Precursor ion mobility"]
    _write_tsv(tmp_path / "accumulatedMsmsScans.txt", acc,
               [["synth", "101", "PEPTIDEK", "620.55", "0.913"],
                ["synth", "102", "NDVPLVGGK", "301.51", "1.207"]])
    n = ids.attach_mobility_from_maxquant(psms, tmp_path / "accumulatedMsmsScans.txt")
    assert n == 2
    assert psms[0].ion_mobility == pytest.approx(0.913)
    assert psms[1].ion_mobility == pytest.approx(1.207)


def test_attach_mobility_missing_scan_left_none(tmp_path):
    cols = ["Raw file", "Scan number", "Charge", "Modified sequence", "Proteins",
            "m/z", "Reverse", "Retention time", "PEP", "Score"]
    _write_tsv(tmp_path / "msms.txt", cols,
               [["synth", "999", "2", "_PEPTIDEK_", "P1", "620.55", "", "5.0", "0.001", "99"]])
    psms = ids.read_psms(tmp_path / "msms.txt")
    _write_tsv(tmp_path / "accumulatedMsmsScans.txt",
               ["Raw file", "Scan number", "Precursor ion mobility"],
               [["synth", "101", "0.913"]])
    n = ids.attach_mobility_from_maxquant(psms, tmp_path / "accumulatedMsmsScans.txt")
    assert n == 0
    assert psms[0].ion_mobility is None


def test_attach_mobility_missing_column_raises(tmp_path):
    cols = ["Raw file", "Scan number", "Charge", "Modified sequence", "Proteins",
            "m/z", "Reverse", "Retention time", "PEP", "Score"]
    _write_tsv(tmp_path / "msms.txt", cols,
               [["synth", "101", "2", "_PEPTIDEK_", "P1", "620.55", "", "5.0", "0.001", "99"]])
    psms = ids.read_psms(tmp_path / "msms.txt")
    _write_tsv(tmp_path / "acc.txt", ["Raw file", "Scan number"], [["synth", "101"]])
    with pytest.raises(KeyError):
        ids.attach_mobility_from_maxquant(psms, tmp_path / "acc.txt")


# ── run registry ───────────────────────────────────────────────────────────────

def test_run_registry_open_spectra_roundtrip():
    store = _two_precursor_store()
    reg = ids.RunRegistry()
    reg.register("synth", store)
    assert "synth" in reg
    assert reg.get("synth") is store
    with pytest.raises(KeyError):
        reg.get("missing")


# ── the property-based match ────────────────────────────────────────────────────

def test_spectrum_for_psm_mobility_disambiguates():
    store = _two_precursor_store(im_a=0.80, im_b=1.15)
    reg = ids.RunRegistry(); reg.register("synth", store)
    idx_a = _ms2_index_near(store, 620.50)
    idx_b = _ms2_index_near(store, 620.60)

    m_a = ids.spectrum_for_psm(_psm(im=0.80), reg)
    assert m_a.resolved
    assert m_a.method == "rt+mz+im"
    assert m_a.store_index == idx_a

    m_b = ids.spectrum_for_psm(_psm(im=1.15), reg)
    assert m_b.store_index == idx_b


def test_spectrum_for_psm_ambiguous_without_mobility():
    store = _two_precursor_store()
    reg = ids.RunRegistry(); reg.register("synth", store)
    m = ids.spectrum_for_psm(_psm(im=None), reg, disambiguate="none")
    assert m.method == "ambiguous"
    assert m.store_index is None
    assert m.n_candidates == 2


def test_spectrum_for_psm_fragment_ranks_when_mobility_absent():
    # A carries the true b/y ions of PEPTIDEK; B carries random peaks; same mobility,
    # so only the fragment content can separate them.
    peaks_a = _by_ions("PEPTIDEK")
    peaks_b = np.random.RandomState(2).uniform(100, 900, len(peaks_a))
    store = _two_precursor_store(im_a=0.90, im_b=0.90, peaks_a=peaks_a, peaks_b=peaks_b)
    reg = ids.RunRegistry(); reg.register("synth", store)
    idx_a = _ms2_index_near(store, 620.50)

    m = ids.spectrum_for_psm(_psm(im=None), reg, disambiguate="fragments")
    assert m.method == "fragment-ranked"
    assert m.store_index == idx_a
    assert m.fragment_scores is not None
    top = m.fragment_scores.iloc[0]
    assert int(top["store_index"]) == idx_a
    # coverage = backbone bonds explained; PEPTIDEK has 8 residues -> 7 bonds,
    # and peaks_a are its full b/y ladder, so the winner covers all 7. The random
    # candidate B scores far lower, which is what makes the pick unambiguous.
    assert int(top["coverage"]) == len("PEPTIDEK") - 1
    assert int(m.fragment_scores.iloc[-1]["coverage"]) < int(top["coverage"])


def test_annotate_match_produces_annotations():
    peaks_a = _by_ions("PEPTIDEK")
    peaks_b = np.random.RandomState(3).uniform(100, 900, len(peaks_a))
    store = _two_precursor_store(im_a=0.90, im_b=0.90, peaks_a=peaks_a, peaks_b=peaks_b)
    reg = ids.RunRegistry(); reg.register("synth", store)
    m = ids.spectrum_for_psm(_psm(im=None), reg)
    annotated = ids.annotate_match(m)
    n = sum(1 for x in annotated.annotation
            if x is not None and getattr(x, "fragment_annotations", None))
    assert n >= len(peaks_a)


def test_annotate_match_unresolved_raises():
    store = _two_precursor_store()
    reg = ids.RunRegistry(); reg.register("synth", store)
    m = ids.spectrum_for_psm(_psm(im=None), reg, disambiguate="none")
    with pytest.raises(ValueError):
        ids.annotate_match(m)