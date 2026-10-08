"""Tests for ptero.ids._mods -- the MaxQuant-label -> mass-offset ProForma
converter that makes modified peptides annotatable offline.

String-rewrite tests are pure (no psm_utils / spectrum_utils needed). Two further
tests confirm (a) the rewritten ProForma actually yields theoretical fragments
offline via spectrum_utils, and (b) ``convert_psm_modifications`` rewrites a
psm_utils PSM in place; these import those libraries and are skipped if absent.
"""
import pytest

from ptero.ids import (
    maxquant_to_mass_offset,
    UnknownModificationError,
)
from ptero.ids._mods import OXIDATION, ACETYL, CARBAMIDOMETHYL


# ── pure string rewrites ─────────────────────────────────────────────────────

def test_oxidation_in_place():
    assert (maxquant_to_mass_offset("AAALATVNAWAEQTGM[Oxidation (M)]K/2")
            == "AAALATVNAWAEQTGM[+15.994915]K/2")


def test_acetyl_n_term_prefix():
    assert (maxquant_to_mass_offset("[Acetyl (Protein N-term)]-AAAAAAAAAAGAAGGR/2")
            == "[+42.010565]-AAAAAAAAAAGAAGGR/2")


def test_unmodified_unchanged():
    assert maxquant_to_mass_offset("PEPTIDEK/2") == "PEPTIDEK/2"


def test_charge_preserved_and_absent_ok():
    assert maxquant_to_mass_offset("AAM[Oxidation (M)]K/3").endswith("/3")
    assert maxquant_to_mass_offset("AAM[Oxidation (M)]K") == "AAM[+15.994915]K"


def test_fixed_mod_added_to_each_residue():
    # no fixed mod by default (matches the no-alkylation HeLa run)
    assert maxquant_to_mass_offset("PEPCIDECK/2") == "PEPCIDECK/2"
    # with a fixed Carbamidomethyl, every C gets the offset
    out = maxquant_to_mass_offset("PEPCIDECK/2", fixed_mods={"C": CARBAMIDOMETHYL})
    assert out == "PEPC[+57.021464]IDEC[+57.021464]K/2"


def test_variable_and_fixed_combined():
    out = maxquant_to_mass_offset(
        "[Acetyl (Protein N-term)]-PEPCM[Oxidation (M)]K/2",
        fixed_mods={"C": CARBAMIDOMETHYL})
    assert out == "[+42.010565]-PEPC[+57.021464]M[+15.994915]K/2"


def test_unknown_modification_raises():
    with pytest.raises(UnknownModificationError):
        maxquant_to_mass_offset("PEPS[Phospho (STY)]K/2")


def test_extra_var_mods_recognised():
    out = maxquant_to_mass_offset(
        "PEPS[Phospho (STY)]K/2", extra_var_mods={"Phospho (STY)": 79.966331})
    assert out == "PEPS[+79.966331]K/2"


def test_idempotent():
    once = maxquant_to_mass_offset("AAM[Oxidation (M)]K/2")
    assert maxquant_to_mass_offset(once) == once       # numeric offsets are skipped


def test_masses_are_the_unimod_values():
    assert OXIDATION == pytest.approx(15.994915)
    assert ACETYL == pytest.approx(42.010565)
    assert CARBAMIDOMETHYL == pytest.approx(57.021464)


# ── integration: rewritten ProForma annotates offline ───────────────────────

def test_converted_proforma_annotates_offline():
    """The rewritten string must yield theoretical fragments without a network
    lookup (the whole point of the converter)."""
    pytest.importorskip("spectrum_utils")
    from spectrum_utils import proforma as P, fragment_annotation as FA

    pf = maxquant_to_mass_offset("[Acetyl (Protein N-term)]-PEPM[Oxidation (M)]IDEK/2")
    proteoform = P.parse(pf)[0]
    frags = FA.get_theoretical_fragments(proteoform, ion_types="by", max_isotope=0, max_charge=2)
    assert len(frags) > 0

    # N-terminal acetyl must shift the b-ion ladder by +42.01
    b_plain = dict((a.ion_type, mz) for a, mz in FA.get_theoretical_fragments(
        P.parse("PEPM[+15.994915]IDEK/2")[0], ion_types="b", max_charge=1))
    b_acet = dict((a.ion_type, mz) for a, mz in FA.get_theoretical_fragments(
        P.parse(pf)[0], ion_types="b", max_charge=1))
    assert b_acet["b1"] - b_plain["b1"] == pytest.approx(ACETYL, abs=1e-4)


def test_convert_psm_modifications_in_place():
    pytest.importorskip("psm_utils")
    from psm_utils import PSM

    psms = [
        PSM(peptidoform="AAM[Oxidation (M)]K/2", spectrum_id="1"),
        PSM(peptidoform="PEPTIDEK/2", spectrum_id="2"),                # no mod
    ]
    from ptero.ids import convert_psm_modifications
    n = convert_psm_modifications(psms)
    assert n == 1                                                      # only the modified one changed
    assert psms[0].peptidoform.proforma == "AAM[+15.994915]K/2"
    assert psms[1].peptidoform.proforma == "PEPTIDEK/2"


# ── reading the search's modifications from summary.txt ──────────────────────

def _write_summary(path, variable, fixed):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Raw file\tVariable modifications\tFixed modifications\n")
        fh.write(f"run1\t{variable}\t{fixed}\n")
        fh.write("\t\t\n")          # MaxQuant often writes a near-empty total row


def test_read_search_modifications_parses_fixed_and_variable(tmp_path):
    from ptero.ids import read_search_modifications
    p = tmp_path / "summary.txt"
    _write_summary(p, "Oxidation (M);Acetyl (Protein N-term)", "Carbamidomethyl (C)")
    res = read_search_modifications(p)
    assert res["fixed"] == {"C": pytest.approx(CARBAMIDOMETHYL)}
    assert "Oxidation (M)" in res["variable"]
    assert "Acetyl (Protein N-term)" in res["variable"]
    assert res["fixed_labels"] == ["Carbamidomethyl (C)"]


def test_read_search_modifications_no_fixed(tmp_path):
    from ptero.ids import read_search_modifications
    p = tmp_path / "summary.txt"
    _write_summary(p, "Oxidation (M)", "")          # a search with no fixed mods
    res = read_search_modifications(p)
    assert res["fixed"] == {}
    assert res["variable"] == ["Oxidation (M)"]


def test_read_search_modifications_unknown_fixed_raises(tmp_path):
    from ptero.ids import read_search_modifications
    p = tmp_path / "summary.txt"
    _write_summary(p, "", "Frobnicate (C)")          # unknown fixed mod name
    with pytest.raises(UnknownModificationError):
        read_search_modifications(p)


def test_read_search_modifications_bad_file_raises(tmp_path):
    from ptero.ids import read_search_modifications
    p = tmp_path / "notsummary.txt"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("Some\tOther\tColumns\n1\t2\t3\n")
    with pytest.raises(KeyError):
        read_search_modifications(p)


def test_convert_psm_modifications_with_summary_path(tmp_path):
    pytest.importorskip("psm_utils")
    from psm_utils import PSM
    from ptero.ids import convert_psm_modifications

    summ = tmp_path / "summary.txt"
    _write_summary(summ, "Oxidation (M)", "Carbamidomethyl (C)")

    # a cysteine-containing peptide must pick up +57.021464 from summary.txt
    psms = [PSM(peptidoform="PEPCM[Oxidation (M)]IDEK/2", spectrum_id="1")]
    n = convert_psm_modifications(psms, summary_path=str(summ))
    assert n == 1
    assert psms[0].peptidoform.proforma == "PEPC[+57.021464]M[+15.994915]IDEK/2"


def test_explicit_fixed_mods_override_summary(tmp_path):
    pytest.importorskip("psm_utils")
    from psm_utils import PSM
    from ptero.ids import convert_psm_modifications

    summ = tmp_path / "summary.txt"
    _write_summary(summ, "", "Carbamidomethyl (C)")
    # explicit fixed_mods={} should override the file -> no +57 added
    psms = [PSM(peptidoform="PEPCK/2", spectrum_id="1")]
    n = convert_psm_modifications(psms, fixed_mods={}, summary_path=str(summ))
    assert psms[0].peptidoform.proforma == "PEPCK/2"
    assert n == 0

def test_read_search_modifications_unsupported_fixed_target_raises(tmp_path):
    from ptero.ids import read_search_modifications

    p = tmp_path / "summary.txt"
    _write_summary(p, "", "Acetyl (Protein N-term)")

    with pytest.raises(ValueError, match="unsupported target"):
        read_search_modifications(p)