"""CV accession and referenceable group robustness."""

import pytest

from ptero.spectra.backends.mzml_backend import (
    _mzml_xml_peak_mobility_semantics,
)

INVERSE = (
    "MS:1003006",
    "mean inverse reduced ion mobility array",
    "MS:1002814",
)
DRIFT = (
    "MS:1002816",
    "ion mobility drift time array",
    "UO:0000028",
)


def cv(accession, name, unit):
    unit_attr = (
        f' unitAccession="{unit}"'
        if unit is not None else ""
    )
    return (
        f'<cvParam accession="{accession}"'
        f' name="{name}"{unit_attr}/>'
    )


def write_xml(path, groups="", arrays=""):
    path.write_text(
        '<?xml version="1.0"?>'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml">'
        f'<referenceableParamGroupList>{groups}'
        '</referenceableParamGroupList>'
        '<run><spectrumList>'
        '<spectrum id="scan=1" index="0">'
        '<binaryDataArrayList>'
        f'{arrays}'
        '</binaryDataArrayList>'
        '</spectrum>'
        '</spectrumList></run></mzML>',
        encoding="utf-8",
    )
    return path


def array(body):
    return (
        '<binaryDataArray>'
        + body
        + '<binary></binary>'
        + '</binaryDataArray>'
    )


def test_accession_is_primary_identity(tmp_path):
    accession, _, unit = INVERSE
    path = write_xml(
        tmp_path / "accession.mzML",
        arrays=array(cv(accession, "vendor label", unit)),
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": (
            "inverse_reduced_ion_mobility", "V.s/cm2"
        )
    }


def test_conflicting_known_name_rejected(tmp_path):
    accession, _, unit = INVERSE
    path = write_xml(
        tmp_path / "conflict.mzML",
        arrays=array(
            cv(accession, DRIFT[1], unit)
        ),
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": None
    }


def test_referenced_inverse_array(tmp_path):
    accession, name, unit = INVERSE
    groups = (
        '<referenceableParamGroup id="im">'
        + cv(accession, name, unit)
        + '</referenceableParamGroup>'
    )
    path = write_xml(
        tmp_path / "referenced.mzML",
        groups=groups,
        arrays=array(
            '<referenceableParamGroupRef ref="im"/>'
        ),
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": (
            "inverse_reduced_ion_mobility", "V.s/cm2"
        )
    }


def test_referenced_drift_array(tmp_path):
    accession, name, unit = DRIFT
    groups = (
        '<referenceableParamGroup id="dt">'
        + cv(accession, name, unit)
        + '</referenceableParamGroup>'
    )
    path = write_xml(
        tmp_path / "drift_ref.mzML",
        groups=groups,
        arrays=array(
            '<referenceableParamGroupRef ref="dt"/>'
        ),
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": None
    }


def test_conflicting_direct_and_referenced(tmp_path):
    groups = (
        '<referenceableParamGroup id="im">'
        + cv(*INVERSE)
        + '</referenceableParamGroup>'
    )
    path = write_xml(
        tmp_path / "double.mzML",
        groups=groups,
        arrays=array(
            '<referenceableParamGroupRef ref="im"/>'
            + cv(*DRIFT)
        ),
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": None
    }


def test_unsupported_unit_rejected(tmp_path):
    accession, name, _ = DRIFT
    path = write_xml(
        tmp_path / "wrong_unit.mzML",
        arrays=array(
            cv(accession, name, "UO:0000010")
        ),
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": None
    }


def test_unresolved_group_rejected(tmp_path):
    path = write_xml(
        tmp_path / "missing_group.mzML",
        arrays=array(
            '<referenceableParamGroupRef ref="missing"/>'
        ),
    )
    with pytest.raises(ValueError, match="Unresolved"):
        _mzml_xml_peak_mobility_semantics(path)


def test_duplicate_groups_rejected(tmp_path):
    groups = (
        '<referenceableParamGroup id="same"/>'
        '<referenceableParamGroup id="same"/>'
    )
    path = write_xml(
        tmp_path / "duplicate_group.mzML",
        groups=groups,
        arrays=array(cv(*INVERSE)),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        _mzml_xml_peak_mobility_semantics(path)


def test_non_mobility_array_ignored(tmp_path):
    path = write_xml(
        tmp_path / "plain.mzML",
        arrays=array(
            cv("MS:1000514", "m/z array", None)
        ),
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {}
