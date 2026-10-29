"""Regression tests for Bruker ddaPASEF metadata preservation."""

import numpy as np
import pandas as pd
import pytest

from ptero.spectra._bruker import (
    _timstof_to_spectra,
    _timstof_to_spectra_pandas,
)


@pytest.mark.parametrize(
    "grouping",
    [_timstof_to_spectra, _timstof_to_spectra_pandas],
    ids=["optimized", "pandas"],
)
def test_bruker_enriched_metadata_preserved(grouping):
    events = pd.DataFrame(
        {
            "frame": [1, 1, 2, 2, 3],
            "rt": [1.0, 1.0, 2.0, 2.0, 3.0],
            "ms_level": [1, 1, 2, 2, 2],
            "precursor_mz": [
                np.nan, np.nan, 500.0, 500.0, 600.0
            ],
            "mz": [100.0, 101.0, 200.0, 201.0, 300.0],
            "intensity": [10.0, 11.0, 20.0, 21.0, 30.0],
            "mobility": [1.0, 1.1, 1.2, 1.3, 1.4],
            "bruker_frame_id": [1, 1, 2, 2, 3],
            "bruker_precursor_id": [
                np.nan, np.nan, 101, 101, 102
            ],
            "bruker_scan_begin": [
                np.nan, np.nan, 10, 10, 20
            ],
            "bruker_scan_end": [
                np.nan, np.nan, 15, 15, 25
            ],
            "isolation_width": [
                np.nan, np.nan, 2.0, 2.0, 3.0
            ],
            "collision_energy": [
                np.nan, np.nan, 25.0, 25.0, 30.0
            ],
            "precursor_charge": [
                np.nan, np.nan, 2, 2, np.nan
            ],
            "monoisotopic_mz": [
                np.nan, np.nan, 499.5, 499.5, np.nan
            ],
            "average_mz": [
                np.nan, np.nan, 500.1, 500.1, 600.1
            ],
        }
    )

    metadata, peaks = grouping(events)

    assert len(metadata) == 3
    assert len(peaks) == 3
    assert metadata["ms_level"].tolist() == [1, 2, 2]

    for field in (
        "isolation_width",
        "collision_energy",
        "precursor_charge",
        "monoisotopic_mz",
        "average_mz",
    ):
        assert field in metadata.columns

    assert str(metadata["precursor_charge"].dtype) == "Int64"

    ms1 = metadata.iloc[0]
    assert pd.isna(ms1["precursor_charge"])
    assert pd.isna(ms1["isolation_width"])

    first_ms2 = metadata.iloc[1]
    assert first_ms2["bruker_precursor_id"] == 101
    assert first_ms2["isolation_width"] == 2.0
    assert first_ms2["collision_energy"] == 25.0
    assert first_ms2["precursor_charge"] == 2
    assert first_ms2["monoisotopic_mz"] == 499.5
    assert first_ms2["average_mz"] == 500.1

    second_ms2 = metadata.iloc[2]
    assert second_ms2["bruker_precursor_id"] == 102
    assert second_ms2["isolation_width"] == 3.0
    assert second_ms2["collision_energy"] == 30.0
    assert pd.isna(second_ms2["precursor_charge"])
    assert pd.isna(second_ms2["monoisotopic_mz"])
    assert second_ms2["average_mz"] == 600.1

    assert [len(p["mz"]) for p in peaks] == [2, 2, 1]
    np.testing.assert_array_equal(
        peaks[1]["mz"], np.array([200.0, 201.0])
    )


@pytest.mark.parametrize(
    "grouping",
    [_timstof_to_spectra, _timstof_to_spectra_pandas],
    ids=["optimized", "pandas"],
)
def test_bruker_rejects_conflicting_metadata(grouping):
    events = pd.DataFrame(
        {
            "frame": [2, 2],
            "rt": [2.0, 2.0],
            "ms_level": [2, 2],
            "precursor_mz": [500.0, 500.0],
            "mz": [200.0, 201.0],
            "intensity": [20.0, 21.0],
            "mobility": [1.2, 1.3],
            "collision_energy": [25.0, 30.0],
        }
    )

    with pytest.raises(
        ValueError, match="Non-unique collision_energy"
    ):
        grouping(events)
