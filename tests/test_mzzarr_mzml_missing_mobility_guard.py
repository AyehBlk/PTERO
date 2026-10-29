"""Behavioral tests for mzML peak mobility completeness."""

import pytest

from ptero.dashboard.convert import (
    _mzzarr_require_complete_peak_mobility,
)

IM = "mean inverse reduced ion mobility array"


def test_nonempty_missing_array_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        _mzzarr_require_complete_peak_mobility(
            ["s1", "s2", "s3"],
            {"s1": IM, "s2": None, "s3": IM},
            {"s1": 3, "s2": 2, "s3": 4},
        )


def test_empty_missing_array_allowed():
    _mzzarr_require_complete_peak_mobility(
        ["s1", "empty", "s3"],
        {"s1": IM, "empty": None, "s3": IM},
        {"s1": 3, "empty": 0, "s3": 4},
    )


def test_complete_arrays_allowed():
    _mzzarr_require_complete_peak_mobility(
        ["s1", "s2"],
        {"s1": IM, "s2": IM},
        {"s1": 3, "s2": 2},
    )


def test_all_empty_allowed():
    _mzzarr_require_complete_peak_mobility(
        ["empty"], {"empty": None}, {"empty": 0}
    )


def test_missing_id_in_error():
    with pytest.raises(ValueError, match="scan=42"):
        _mzzarr_require_complete_peak_mobility(
            ["scan=1", "scan=42"],
            {"scan=1": IM, "scan=42": None},
            {"scan=1": 3, "scan=42": 1},
        )
