"""Conservative mzML mobility-semantic extraction tests."""

from types import SimpleNamespace

from ptero.spectra.backends.mzml_backend import (
    _mzml_mobility_semantic,
    _mzml_scan_mobility_semantic,
    _mzml_peak_mobility_semantic,
    _mzml_common_semantic,
)


class UnitValue(float):
    def __new__(cls, value, unit):
        obj = float.__new__(cls, value)
        obj.unit_info = unit
        return obj


def test_inverse_reduced_mobility():
    value = UnitValue(
        1.12, "volt-second per square centimeter"
    )
    assert _mzml_mobility_semantic(
        "inverse reduced ion mobility", value
    ) == ("inverse_reduced_ion_mobility", "V.s/cm2")


def test_drift_time():
    value = UnitValue(4.5, "millisecond")
    assert _mzml_mobility_semantic(
        "ion mobility drift time", value
    ) == ("ion_mobility_drift_time", "millisecond")


def test_missing_unit_is_unknown():
    assert _mzml_mobility_semantic(
        "inverse reduced ion mobility", 1.12
    ) is None


def test_wrong_unit_is_unknown():
    assert _mzml_mobility_semantic(
        "inverse reduced ion mobility",
        UnitValue(1.12, "millisecond")
    ) is None


def test_scan_level_semantic():
    spec = {
        "scanList": {
            "scan": [{
                "ion mobility drift time":
                    UnitValue(5.0, "millisecond")
            }]
        }
    }
    assert _mzml_scan_mobility_semantic(spec) == (
        "ion_mobility_drift_time", "millisecond"
    )


def test_scan_conflicting_quantities_are_unknown():
    spec = {
        "scanList": {
            "scan": [{
                "inverse reduced ion mobility":
                    UnitValue(
                        1.0, "volt-second per square centimeter"
                    ),
                "ion mobility drift time":
                    UnitValue(5.0, "millisecond"),
            }]
        }
    }
    assert _mzml_scan_mobility_semantic(spec) is None


def test_peak_array_semantic():
    spec = {
        "mean inverse reduced ion mobility array":
            UnitValue(
                1.0, "volt-second per square centimeter"
            )
    }
    assert _mzml_peak_mobility_semantic(spec) == (
        "inverse_reduced_ion_mobility", "V.s/cm2"
    )


def test_peak_conflicting_arrays_are_unknown():
    spec = {
        "mean inverse reduced ion mobility array":
            UnitValue(
                1.0, "volt-second per square centimeter"
            ),
        "ion mobility drift time array":
            UnitValue(5.0, "millisecond"),
    }
    assert _mzml_peak_mobility_semantic(spec) is None


def test_common_semantic():
    inverse = ("inverse_reduced_ion_mobility", "V.s/cm2")
    drift = ("ion_mobility_drift_time", "millisecond")

    assert _mzml_common_semantic([inverse, inverse]) == inverse
    assert _mzml_common_semantic([inverse, drift]) is None
    assert _mzml_common_semantic([inverse, None]) is None
    assert _mzml_common_semantic([]) is None
