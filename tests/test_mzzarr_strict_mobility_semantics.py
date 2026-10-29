"""Focused strict/legacy mobility semantic regression tests."""

import pytest

from ptero.spectra._schema import validate_mobility_semantics


@pytest.mark.parametrize(
    "peak_present,spectrum_present",
    [(True, False), (False, True), (True, True)],
)
def test_strict_requires_scope_declarations(
    peak_present, spectrum_present
):
    errors = validate_mobility_semantics(
        {},
        has_peak_mobility=peak_present,
        has_spectrum_mobility=spectrum_present,
        strict=True,
    )
    if peak_present:
        assert any("peak ion mobility requires both" in e for e in errors)
    if spectrum_present:
        assert any("spectrum ion mobility requires both" in e for e in errors)


def test_legacy_mode_preserves_undeclared_mobility():
    errors = validate_mobility_semantics(
        {},
        has_peak_mobility=True,
        has_spectrum_mobility=True,
        strict=False,
    )
    assert errors == []


def test_strict_accepts_valid_peak_declaration():
    errors = validate_mobility_semantics(
        {
            "peak_ion_mobility_type": "inverse_reduced_ion_mobility",
            "peak_ion_mobility_unit": "V.s/cm2",
        },
        has_peak_mobility=True,
        has_spectrum_mobility=False,
        strict=True,
    )
    assert errors == []


def test_strict_rejects_wrong_unit():
    errors = validate_mobility_semantics(
        {
            "peak_ion_mobility_type": "inverse_reduced_ion_mobility",
            "peak_ion_mobility_unit": "millisecond",
        },
        has_peak_mobility=True,
        has_spectrum_mobility=False,
        strict=True,
    )
    assert any("invalid peak_ion_mobility_unit" in e for e in errors)
