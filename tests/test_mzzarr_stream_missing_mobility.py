import numpy as np
import pandas as pd
import pytest

from ptero.spectra.backends.zarr_backend import ZarrStreamWriter


class Batch:
    def __init__(self, count, mobility):
        self._meta = pd.DataFrame({
            "ms_level": pd.Series([1], dtype="int64"),
        })
        self._mz = np.arange(count, dtype=np.float64)
        self._intensity = np.ones(count, dtype=np.float32)
        self._im = mobility
        self._offsets = np.array([0, count], dtype=np.int64)


def writer(path):
    return ZarrStreamWriter(
        path,
        has_4d=True,
        peak_ion_mobility_type="inverse_reduced_ion_mobility",
        peak_ion_mobility_unit="V.s/cm2",
    )


def test_nonempty_missing_mobility_rejected_before_growth(tmp_path):
    output = writer(tmp_path / "reject.zarr")
    with pytest.raises(ValueError, match="refusing to fabricate"):
        output.append_backend(Batch(3, None))
    assert output._mz.shape == (0,)
    assert output._int.shape == (0,)
    assert output._im.shape == (0,)
    assert output._offsets == [0]


def test_empty_missing_mobility_allowed(tmp_path):
    output = writer(tmp_path / "empty.zarr")
    output.append_backend(Batch(0, None))
    assert output._offsets == [0, 0]
    assert output._im.shape == (0,)


def test_present_mobility_preserved(tmp_path):
    output = writer(tmp_path / "valid.zarr")
    mobility = np.array([0.8, 1.0, 1.2], dtype=np.float32)
    output.append_backend(Batch(3, mobility))
    assert np.array_equal(output._im[:], mobility)
    assert output._offsets == [0, 3]
