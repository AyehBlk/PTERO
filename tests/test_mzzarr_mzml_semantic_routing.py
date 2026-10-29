"""Selected-spectrum mzML mobility routing tests."""

import pytest

from ptero.spectra.backends.mzml_backend import (
    _mzml_route_peak_semantics,
)

IM = ("inverse_reduced_ion_mobility", "V.s/cm2")
DT = ("ion_mobility_drift_time", "millisecond")
IM_KEY = "mean inverse reduced ion mobility array"
DT_KEY = "ion mobility drift time array"


def route(ids, xml, keys):
    return _mzml_route_peak_semantics(ids, xml, keys)


def test_empty_selection():
    assert route([], {}, {}) is None


def test_no_mobility_arrays():
    assert route(["s1", "s2"], {}, {}) is None


def test_verified_inverse_mobility():
    assert route(
        ["s1"], {"s1": IM}, {"s1": IM_KEY}
    ) == IM


def test_verified_drift_time():
    assert route(
        ["s1"], {"s1": DT}, {"s1": DT_KEY}
    ) == DT


def test_selected_subset_ignores_unselected_conflict():
    assert route(
        ["s1"],
        {"s1": IM, "s2": DT},
        {"s1": IM_KEY, "s2": DT_KEY},
    ) == IM


def test_mixed_selected_quantities_rejected():
    with pytest.raises(ValueError, match="mix"):
        route(
            ["s1", "s2"],
            {"s1": IM, "s2": DT},
            {"s1": IM_KEY, "s2": DT_KEY},
        )


def test_ambiguous_xml_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        route(
            ["s1"], {"s1": None}, {"s1": IM_KEY}
        )


def test_missing_xml_rejected():
    with pytest.raises(ValueError, match="without verified"):
        route(["s1"], {}, {"s1": IM_KEY})


def test_missing_pyteomics_array_rejected():
    with pytest.raises(ValueError, match="selected none"):
        route(["s1"], {"s1": IM}, {})


def test_conflicting_array_key_rejected():
    with pytest.raises(ValueError, match="conflicts"):
        route(
            ["s1"], {"s1": IM}, {"s1": DT_KEY}
        )


def test_unknown_array_key_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        route(
            ["s1"],
            {"s1": IM},
            {"s1": "vendor ion mobility array"},
        )


def test_wrong_unit_rejected():
    with pytest.raises(ValueError, match="unit"):
        route(
            ["s1"],
            {
                "s1": (
                    "inverse_reduced_ion_mobility",
                    "millisecond",
                )
            },
            {"s1": IM_KEY},
        )


def test_duplicate_selected_ids_rejected():
    with pytest.raises(ValueError, match="not unique"):
        route(
            ["s1", "s1"],
            {"s1": IM},
            {"s1": IM_KEY},
        )


def test_missing_mobility_array_between_valid_arrays():
    assert route(
        ["s1", "s2", "s3"],
        {"s1": IM, "s3": IM},
        {"s1": IM_KEY, "s3": IM_KEY},
    ) == IM


def test_raw_inverse_array_supported():
    assert route(
        ["s1"],
        {"s1": IM},
        {"s1": "raw inverse reduced ion mobility array"},
    ) == IM
