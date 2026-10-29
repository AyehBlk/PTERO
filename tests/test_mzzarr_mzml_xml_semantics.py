"""XML-level mobility semantics: no binary decoding required."""

import pytest

from ptero.spectra.backends.mzml_backend import (
    _mzml_xml_peak_mobility_semantics,
)


def write_mzml(path, arrays_by_spectrum):
    spectra = []
    for i, arrays in enumerate(arrays_by_spectrum):
        binary_arrays = []
        for name, accession, unit in arrays:
            unit_attr = (
                f' unitAccession="{unit}"'
                if unit is not None else ""
            )
            binary_arrays.append(
                '<binaryDataArray>'
                f'<cvParam cvRef="MS" accession="{accession}"'
                f' name="{name}"{unit_attr}/>'
                '<binary></binary>'
                '</binaryDataArray>'
            )
        spectra.append(
            f'<spectrum id="scan={i + 1}" index="{i}">'
            f'<binaryDataArrayList count="{len(arrays)}">'
            + "".join(binary_arrays)
            + '</binaryDataArrayList></spectrum>'
        )

    xml = (
        '<?xml version="1.0"?>'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml">'
        '<run><spectrumList>'
        + "".join(spectra)
        + '</spectrumList></run></mzML>'
    )
    path.write_text(xml, encoding="utf-8")
    return path


INVERSE = (
    "mean inverse reduced ion mobility array",
    "MS:1003006",
    "MS:1002814",
)
DRIFT = (
    "ion mobility drift time array",
    "MS:1002816",
    "UO:0000028",
)


def test_inverse_reduced_array(tmp_path):
    path = write_mzml(tmp_path / "inverse.mzML", [[INVERSE]])
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": (
            "inverse_reduced_ion_mobility", "V.s/cm2"
        )
    }


def test_drift_time_array(tmp_path):
    path = write_mzml(tmp_path / "drift.mzML", [[DRIFT]])
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": None
    }


def test_missing_unit_is_unknown(tmp_path):
    array = (INVERSE[0], INVERSE[1], None)
    path = write_mzml(tmp_path / "missing.mzML", [[array]])
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": None
    }


def test_wrong_unit_is_unknown(tmp_path):
    array = (INVERSE[0], INVERSE[1], "UO:0000028")
    path = write_mzml(tmp_path / "wrong.mzML", [[array]])
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": None
    }


def test_multiple_mobility_arrays_are_ambiguous(tmp_path):
    path = write_mzml(
        tmp_path / "multiple.mzML", [[INVERSE, DRIFT]]
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": None
    }


def test_no_mobility_array_has_no_entry(tmp_path):
    path = write_mzml(
        tmp_path / "plain.mzML",
        [[("m/z array", "MS:1000514", None)]]
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {}


def test_mixed_spectra_remain_distinguishable(tmp_path):
    path = write_mzml(
        tmp_path / "mixed.mzML",
        [[INVERSE], [], [DRIFT]]
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": (
            "inverse_reduced_ion_mobility", "V.s/cm2"
        ),
        "scan=3": None,
    }


def test_non_mzml_rejected(tmp_path):
    path = tmp_path / "sample.mzXML"
    path.write_text("<root/>", encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        _mzml_xml_peak_mobility_semantics(path)


def test_duplicate_spectrum_ids_rejected(tmp_path):
    path = write_mzml(
        tmp_path / "duplicate.mzML",
        [[INVERSE], [DRIFT]]
    )
    text = path.read_text(encoding="utf-8")
    text = text.replace('id="scan=2"', 'id="scan=1"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        _mzml_xml_peak_mobility_semantics(path)
