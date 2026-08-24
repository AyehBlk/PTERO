"""Tests for load_quant(quantity=...) -- raw-Intensity fallback when a MaxQuant
search was run without LFQ (and a clear error instead of a crash)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptero.quant import load_quant

SAMPLES = ["S_A_1", "S_A_2", "S_B_1"]


def _write_no_lfq(path):
    """A proteinGroups.txt with Intensity (+ Identification type) but NO LFQ."""
    rows = [
        {"id": 0, "Protein IDs": "sp|P1|P1_X", "Majority protein IDs": "sp|P1|P1_X",
         "Fasta headers": "sp|P1|P1_X X OS=Homo sapiens OX=9606 GN=Gene1 PE=1 SV=1",
         "Reverse": "", "Potential contaminant": "", "Only identified by site": "",
         "Intensity S_A_1": "2000", "Intensity S_A_2": "2200", "Intensity S_B_1": "0",
         "Identification type S_A_1": "By MS/MS", "Identification type S_A_2": "By matching",
         "Identification type S_B_1": ""},
        {"id": 1, "Protein IDs": "sp|P2|P2_X", "Majority protein IDs": "sp|P2|P2_X",
         "Fasta headers": "sp|P2|P2_X X OS=Homo sapiens OX=9606 GN=Gene2 PE=1 SV=1",
         "Reverse": "", "Potential contaminant": "", "Only identified by site": "",
         "Intensity S_A_1": "300", "Intensity S_A_2": "900", "Intensity S_B_1": "1500",
         "Identification type S_A_1": "By MS/MS", "Identification type S_A_2": "By MS/MS",
         "Identification type S_B_1": "By MS/MS"},
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, encoding="utf-8-sig")


@pytest.fixture()
def no_lfq_path(tmp_path):
    p = tmp_path / "proteinGroups.txt"
    _write_no_lfq(p)
    return p


def test_intensity_fallback_loads(no_lfq_path):
    qd = load_quant(no_lfq_path, quantity="intensity")
    assert qd.samples == SAMPLES
    assert list(qd.lfq.index) == [0, 1]
    # 0 -> NaN, raw Intensity values land in the quantification matrix
    assert qd.lfq.loc[0, "S_A_1"] == 2000.0
    assert np.isnan(qd.lfq.loc[0, "S_B_1"])
    # lfq and intensity are the same matrix under quantity='intensity'
    pd.testing.assert_frame_equal(qd.lfq, qd.intensity)


def test_lfq_request_without_lfq_raises_helpfully(no_lfq_path):
    with pytest.raises(ValueError, match="quantity='intensity'"):
        load_quant(no_lfq_path, quantity="lfq")


def test_bad_quantity_raises(no_lfq_path):
    with pytest.raises(ValueError, match="quantity must be"):
        load_quant(no_lfq_path, quantity="foo")