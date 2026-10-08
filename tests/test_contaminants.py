"""Tests for ptero.quant.annotate_contaminants (configurable contaminant tags)."""
import pandas as pd

from ptero.quant import annotate_contaminants


def _prot(genes, *, protein_ids=None, flag=None):
    idx = pd.Index(range(len(genes)), name="id")
    d = {"gene": genes}
    if protein_ids is not None:
        d["protein_ids"] = protein_ids
    if flag is not None:
        d["is_potential_contaminant"] = flag
    return pd.DataFrame(d, index=idx)


def test_tags_muscle_keratin_blood_and_ok():
    p = _prot(["Myh4", "Krtap1-1", "Hba-a1", "Gbp2b", "Stat1"])
    cls = annotate_contaminants(p)["contaminant_class"].tolist()
    assert cls == ["muscle", "keratin", "blood", "ok", "ok"]


def test_crap_from_flag_and_accession():
    p = _prot(["Gene1", "Gene2"],
              protein_ids=["P1", "CON__P2"],
              flag=[True, False])
    out = annotate_contaminants(p)
    assert out["contaminant_class"].tolist() == ["cRAP", "cRAP"]
    assert out["is_crap"].tolist() == [True, True]


def test_crap_overrides_gene_pattern():
    # a muscle gene that is also a database contaminant -> cRAP wins (applied last)
    p = _prot(["Myh1"], protein_ids=["CON__X"])
    assert annotate_contaminants(p)["contaminant_class"].tolist() == ["cRAP"]


def test_toggles_disable_pattern_sets():
    p = _prot(["Myh4", "Krt8", "Hbb-b1"])
    cls = annotate_contaminants(p, flag_muscle=False, flag_keratin=False,
                                flag_blood=False)["contaminant_class"].tolist()
    assert cls == ["ok", "ok", "ok"]


def test_custom_patterns():
    p = _prot(["Foo1", "Bar2"])
    cls = annotate_contaminants(p, muscle=(r"^Foo\d",))["contaminant_class"].tolist()
    assert cls == ["muscle", "ok"]


def test_missing_gene_is_ok():
    p = _prot([None, "Myh4"])
    assert annotate_contaminants(p)["contaminant_class"].tolist() == ["ok", "muscle"]

def test_crap_accession_marker_must_be_prefix():
    p = _prot(
        ["Gene1", "Gene2", "Gene3"],
        protein_ids=["CON__P1", "cRAP_P2", "SCRAP1"],
    )

    out = annotate_contaminants(p)

    assert out["contaminant_class"].tolist() == [
        "cRAP",
        "cRAP",
        "ok",
    ]
    assert out["is_crap"].tolist() == [
        True,
        True,
        False,
    ]