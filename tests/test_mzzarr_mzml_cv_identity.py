"""PSI-MS CV identity checks for mzML peak mobility."""

import pytest

from ptero.spectra.backends.mzml_backend import (
    _mzml_xml_peak_mobility_semantics,
)


def make_fixture(path, accession, name, unit):
    path.write_text(
        '<?xml version="1.0"?>'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml">'
        '<run><spectrumList>'
        '<spectrum id="scan=1" index="0">'
        '<binaryDataArrayList>'
        '<binaryDataArray>'
        f'<cvParam accession="{accession}" '
        f'name="{name}" unitAccession="{unit}"/>'
        '<binary></binary>'
        '</binaryDataArray>'
        '</binaryDataArrayList>'
        '</spectrum>'
        '</spectrumList></run></mzML>',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "accession,name",
    [
        ("MS:1003006", "mean inverse reduced ion mobility array"),
        ("MS:1003008", "raw inverse reduced ion mobility array"),
    ],
)
def test_valid_inverse_accessions(tmp_path, accession, name):
    path = make_fixture(
        tmp_path / "valid.mzML",
        accession,
        name,
        "MS:1002814",
    )
    assert _mzml_xml_peak_mobility_semantics(path) == {
        "scan=1": (
            "inverse_reduced_ion_mobility", "V.s/cm2"
        )
    }


@pytest.mark.parametrize(
    "accession,name,unit",
    [
        (
            "MS:1002816",
            "mean inverse reduced ion mobility array",
            "MS:1002814",
        ),
        (
            "MS:1003008",
            "ion mobility drift time array",
            "UO:0000028",
        ),
        (
            "MS:1003006",
            "mean inverse reduced ion mobility array",
            "UO:0000268",
        ),
    ],
)
def test_invalid_cv_identity_rejected(
    tmp_path, accession, name, unit
):
    path = make_fixture(
        tmp_path / "invalid.mzML",
        accession,
        name,
        unit,
    )
    result = _mzml_xml_peak_mobility_semantics(path)
    assert result in ({}, {"scan=1": None})
