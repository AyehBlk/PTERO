"""Tests for the ID-annotation dashboard page and shared state.

The viz-dependent tests are skipped if the Panel/HoloViews stack isn't installed
(same pattern as test_dashboard.py). The `peak_annotations` data test needs only
spectrum_utils/pyteomics and so runs anywhere those are present.
"""
import numpy as np
import pandas as pd
import pytest


# ── peak_annotations: the plot-data pipeline (no Panel needed) ───────────────

def test_peak_annotations_labels_only_real_fragments():
    pytest.importorskip("spectrum_utils")
    from pyteomics import mass
    from ptero.ids import peak_annotations

    seq = "PEPTIDEK"
    true = []
    for i in range(1, len(seq)):
        true.append(mass.fast_mass(seq[:i], ion_type="b", charge=1))
        true.append(mass.fast_mass(seq[i:], ion_type="y", charge=1))
    noise = np.random.RandomState(0).uniform(100, 900, 50)
    mz = np.concatenate([np.array(true), noise])
    inten = np.concatenate([np.full(len(true), 8.0), np.full(50, 0.4)])

    ann = peak_annotations(mz, inten, "PEPTIDEK/2", 2)
    assert list(ann.columns) == ["mz", "intensity", "label", "series", "matched"]
    assert len(ann) == len(mz)
    # exactly the 14 b/y ions of PEPTIDEK are labelled, not the noise
    assert int(ann["matched"].sum()) == 2 * (len(seq) - 1)
    labelled = set(ann.loc[ann["matched"], "label"])
    assert {"b1", "b7", "y1", "y7"} <= labelled
    # series colours map cleanly
    assert set(ann.loc[ann["matched"], "series"]) == {"b", "y"}


def test_peak_annotations_empty_spectrum():
    pytest.importorskip("spectrum_utils")
    from ptero.ids import peak_annotations
    ann = peak_annotations(np.array([]), np.array([]), "PEPTIDEK/2", 2)
    assert len(ann) == 0
    assert list(ann.columns) == ["mz", "intensity", "label", "series", "matched"]


# ── shared state ─────────────────────────────────────────────────────────────

def test_appstate_set_and_notify():
    from ptero.dashboard import AppState
    st = AppState()
    seen = []
    st.on_change(lambda key: seen.append(key))
    st.set("psms", [1, 2, 3])
    st.set("id_store", "store")
    assert st.psms == [1, 2, 3]
    assert st.id_store == "store"
    assert seen == ["psms", "id_store"]


def test_appstate_listener_error_does_not_break():
    from ptero.dashboard import AppState
    st = AppState()
    st.on_change(lambda key: 1 / 0)          # misbehaving listener
    st.set("psms", "x")                       # must not raise
    assert st.psms == "x"


# ── construction (needs the viz stack) ──────────────────────────────────────

def test_ids_view_constructs():
    pytest.importorskip("panel")
    pytest.importorskip("holoviews")
    from ptero.dashboard import IdsView, AppState
    v = IdsView(AppState())
    layout = v.view()
    assert layout is not None


def test_build_app_is_multipage():
    pytest.importorskip("panel")
    pytest.importorskip("holoviews")
    pytest.importorskip("datashader")
    from ptero.dashboard import build_app
    app = build_app()
    assert app is not None


# ── _ids_data: protein/peptide read + join + unique-razor classification ─────

def _write_mq(tmp_path):
    pg = tmp_path / "proteinGroups.txt"
    pep = tmp_path / "peptides.txt"
    pg.write_text("\t".join([
        "Gene names", "Protein names", "Peptide counts (all)", "Peptide counts (unique)",
        "Razor + unique peptides", "Sequence coverage [%]", "MS/MS count", "Score", "id"]) + "\n"
        + "\t".join(["PABPN1", "PolyA-binding protein 2", "2", "1", "2", "23.5", "41", "201.3", "2020"]) + "\n"
        + "\t".join(["ALB", "Serum albumin", "3", "3", "3", "45.0", "60", "305.0", "100"]) + "\n"
        + "\t".join(["", "", "1", "1", "1", "2.1", "3", "12.3", "0"]) + "\n",   # empty-gene row
        encoding="utf-8")
    pep.write_text("\t".join([
        "Sequence", "Proteins", "Leading razor protein", "Gene names", "Protein names",
        "Unique (Groups)", "Charges", "Score", "PEP", "Protein group IDs", "id"]) + "\n"
        + "\t".join(["AAAAAAAAAAGAAGGR", "Q86U42", "Q86U42", "PABPN1", "PolyA-binding protein 2",
                     "yes", "2", "144.77", "7.5e-09", "2020", "0"]) + "\n"
        + "\t".join(["SHAREDPEPK", "Q86U42;P02768", "Q86U42", "PABPN1", "PolyA-binding protein 2",
                     "no", "2", "79.7", "3e-05", "2020;100", "1"]) + "\n"
        + "\t".join(["ALBUNIQUEK", "P02768", "P02768", "ALB", "Serum albumin",
                     "yes", "3", "150.0", "1e-08", "100", "2"]) + "\n",
        encoding="utf-8")
    return str(pg), str(pep)


def test_ids_data_protein_table(tmp_path):
    from ptero.dashboard._ids_data import IdentificationData
    pg, pep = _write_mq(tmp_path)
    idd = IdentificationData(pg, pep)
    pt = idd.protein_table
    # sorted by score desc -> ALB first
    assert pt.iloc[0]["gene"] == "ALB"
    assert pt.iloc[0]["score"] == 305.0
    assert pt.iloc[0]["coverage_pct"] == 45.0
    # the empty-gene contaminant row stays blank (no peptides give it a name)
    assert "" in set(pt["gene"])


def test_ids_data_peptides_and_razor(tmp_path):
    from ptero.dashboard._ids_data import IdentificationData
    pg, pep = _write_mq(tmp_path)
    idd = IdentificationData(pg, pep)
    # ALB group (100): a unique peptide and the shared one (razor here)
    alb = idd.peptides_for_group("100")
    classes = dict(zip(alb["peptide"], alb["unique"]))
    assert classes["ALBUNIQUEK"] == "unique"
    assert classes["SHAREDPEPK"] == "razor"     # shared but leading razor protein matches
    # the hidden plain-sequence column is present for PSM lookup
    assert "_seq" in alb.columns
    # PABPN1 group (2020) also contains the shared peptide
    pabpn1 = idd.peptides_for_group("2020")
    assert "SHAREDPEPK" in set(pabpn1["peptide"])


# ── _fasta: sequence reading + peptide-to-protein mapping (Stage 2) ──────────

def test_fasta_read_and_locate(tmp_path):
    from ptero.dashboard._fasta import ProteinSequences, locate_peptide
    fa = tmp_path / "db.fasta"
    fa.write_text(
        ">sp|P06733|ENOA_HUMAN Alpha-enolase OS=Homo sapiens\n"
        "MSILKIHAREIFDSRGNPTVEVDLFTSKGLFRAAVPSGASTGIYEALELRDNDKTRYMGKGVSKAVEHINKTIAPALVSKK\n"
        ">sp|Q86U42|PABP2_HUMAN PolyA-binding protein 2\n"
        "MAAAAAAAAAAGAAGGRGSGPGRRR\n", encoding="utf-8")
    db = ProteinSequences(str(fa))
    assert len(db) == 2
    seq = db.get("P06733")
    assert seq is not None and len(seq) == 81
    # isoform id falls back to the base accession
    assert db.get("P06733-2") == seq
    # locate a known peptide
    spans = locate_peptide("AAVPSGASTGIYEALELR", seq)
    assert spans == [(32, 50)]
    # a peptide that isn't there
    assert locate_peptide("NOTHEREK", seq) == []


def test_fasta_map_peptides_coverage(tmp_path):
    from ptero.dashboard._fasta import ProteinSequences, map_peptides
    fa = tmp_path / "db.fasta"
    fa.write_text(
        ">sp|P06733|ENOA_HUMAN Alpha-enolase\n"
        "MSILKIHAREIFDSRGNPTVEVDLFTSKGLFRAAVPSGASTGIYEALELRDNDKTRYMGKGVSKAVEHINKTIAPALVSKK\n",
        encoding="utf-8")
    db = ProteinSequences(str(fa))
    peps = [{"peptide": "AAVPSGASTGIYEALELR", "unique": "razor"},
            {"peptide": "GVSKAVEHINK", "unique": "unique"},
            {"peptide": "NOTHEREK", "unique": "unique"}]
    m = map_peptides("P06733", db.get("P06733"), peps)
    assert len(m.located) == 2
    assert m.not_located == ["NOTHEREK"]
    assert 0 < m.coverage_pct < 100
    # classes carried through for colouring
    assert {d["unique"] for d in m.located} == {"razor", "unique"}


# ── _mods_sites: generic *Sites.txt reading (Stage 3) ───────────────────────

def test_mod_sites_reads_generic(tmp_path):
    from ptero.dashboard._mods_sites import ModificationSites
    # oxidation sites file with the real MaxQuant columns
    (tmp_path / "Oxidation (M)Sites.txt").write_text("\t".join([
        "Proteins", "Positions within proteins", "Leading proteins", "Amino acid",
        "Localization prob", "Protein group IDs", "Reverse", "Potential contaminant", "id"]) + "\n"
        + "\t".join(["P06733", "152", "P06733", "M", "0.98", "11", "", "", "0"]) + "\n"
        + "\t".join(["P06733", "305", "P06733", "M", "0.42", "11", "", "", "1"]) + "\n"
        + "\t".join(["REV__X", "9", "REV__X", "M", "1.0", "999", "+", "", "2"]) + "\n",  # decoy
        encoding="utf-8")
    # a phospho file proves genericity
    (tmp_path / "Phospho (STY)Sites.txt").write_text("\t".join([
        "Positions within proteins", "Amino acid", "Localization prob",
        "Protein group IDs", "id"]) + "\n"
        + "\t".join(["210", "S", "0.85", "11", "0"]) + "\n", encoding="utf-8")

    ms = ModificationSites(str(tmp_path))
    assert set(ms.mod_types) == {"Oxidation (M)", "Phospho (STY)"}
    sites = ms.for_group("11")
    assert len(sites) == 3                       # 2 oxidation + 1 phospho, decoy excluded
    mods = {(s["mod"], s["position"]) for s in sites}
    assert ("Oxidation (M)", 152) in mods
    assert ("Phospho (STY)", 210) in mods
    # localisation parsed as float
    ox152 = next(s for s in sites if s["position"] == 152)
    assert abs(ox152["localization"] - 0.98) < 1e-6


def test_mod_sites_empty_when_none(tmp_path):
    from ptero.dashboard._mods_sites import ModificationSites
    ms = ModificationSites(str(tmp_path))   # no *Sites.txt present
    assert len(ms) == 0
    assert ms.for_group("anything") == []

def test_ids_view_does_not_depend_on_private_ids_link():
    from pathlib import Path
    import ptero.dashboard.ids_view as ids_view

    source = Path(ids_view.__file__).read_text(encoding="utf-8")

    assert "ids._link" not in source