"""Tests for ptero.quant._missing -- the six-state model, summaries, on/off table.

Synthetic-first: a tiny BOM-encoded proteinGroups.txt with Identification type
columns covering every state, plus two planted group-exclusive proteins (one with
a clean 'off' side, one whose 'off' side still carries detected signal).
"""
from __future__ import annotations

import pandas as pd
import pytest
from matplotlib.figure import Figure

from ptero.quant import (
    load_quant, cell_state, state_counts, missingness_summary, group_exclusive,
    plot_missingness_vs_abundance, plot_cell_state_composition,
    QUANTIFIED_MSMS, QUANTIFIED_MBR, DETECTED_MSMS, DETECTED_MBR,
    MATCHED_EMPTY, ABSENT,
)

SAMPLES = ["S_A_1", "S_A_2", "S_B_1", "S_B_2"]
GROUPS = {"S_A_1": "A", "S_A_2": "A", "S_B_1": "B", "S_B_2": "B"}
MS, MB = "By MS/MS", "By matching"


def _row(pid, lfq, inten, idt, gene="GeneX"):
    r = {"id": pid, "Protein IDs": f"sp|P{pid}|P{pid}_X",
         "Majority protein IDs": f"sp|P{pid}|P{pid}_X",
         "Fasta headers": f"sp|P{pid}|P{pid}_X X OS=H OX=9606 GN={gene} PE=1 SV=1",
         "Species": "H", "Q-value": "0", "Score": "100",
         "Razor + unique peptides": "5",
         "Reverse": "", "Potential contaminant": "", "Only identified by site": ""}
    for s, l, i, t in zip(SAMPLES, lfq, inten, idt):
        r[f"LFQ intensity {s}"] = l
        r[f"Intensity {s}"] = i
        r[f"Identification type {s}"] = t
    return r


def _write(path):
    rows = [
        # p0: all quantified by MS/MS
        _row(0, ["1000", "1100", "1050", "1080"], ["2000", "2200", "2100", "2160"], [MS, MS, MS, MS]),
        # p1: present in all of A, NOT observed in all of B  -> clean on/off
        _row(1, ["900", "950", "0", "0"], ["1800", "1900", "0", "0"], [MS, MS, "", ""]),
        # p2: present in all of A, but B is detected-no-LFQ  -> on/off with signal off
        _row(2, ["800", "820", "0", "0"], ["1600", "1640", "300", "320"], [MS, MS, MS, MS]),
        # p3: the MBR states (quantified_mbr / detected_mbr / matched_empty / absent)
        _row(3, ["500", "0", "0", "0"], ["900", "300", "0", "0"], [MB, MB, MB, ""]),
        # p4: never LFQ-quantified (mix of detected + absent); not an on/off
        _row(4, ["0", "0", "0", "0"], ["100", "0", "0", "50"], [MS, "", "", MB]),
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, encoding="utf-8-sig")


@pytest.fixture()
def pg(tmp_path):
    p = tmp_path / "proteinGroups.txt"
    _write(p)
    return p


def test_id_type_loaded(pg):
    qd = load_quant(pg, groups=GROUPS)
    assert list(qd.id_type.columns) == SAMPLES
    assert qd.id_type.loc[0, "S_A_1"] == "By MS/MS"
    assert qd.id_type.loc[3, "S_A_1"] == "By matching"
    assert qd.id_type.loc[1, "S_B_1"] is None        # blank -> None
    # to_long now carries id_type
    assert "id_type" in qd.to_long().columns


def test_cell_state(pg):
    st = cell_state(load_quant(pg, groups=GROUPS))
    assert st.loc[0, "S_A_1"] == QUANTIFIED_MSMS
    assert st.loc[3, "S_A_1"] == QUANTIFIED_MBR
    assert st.loc[2, "S_B_1"] == DETECTED_MSMS
    assert st.loc[3, "S_A_2"] == DETECTED_MBR
    assert st.loc[3, "S_B_1"] == MATCHED_EMPTY
    assert st.loc[1, "S_B_1"] == ABSENT


def test_state_counts_total(pg):
    qd = load_quant(pg, groups=GROUPS)
    assert int(state_counts(qd).sum()) == qd.n_proteins * len(qd.samples)   # 20


def test_missingness_summary(pg):
    s = missingness_summary(load_quant(pg, groups=GROUPS))
    assert s.loc[0, "n_present"] == 4 and s.loc[0, "n_missing"] == 0
    assert s.loc[1, "present_A"] == 2 and s.loc[1, "present_B"] == 0
    assert s.loc[1, "n_present"] == 2


def test_group_exclusive(pg):
    ex = group_exclusive(load_quant(pg, groups=GROUPS))
    ids = list(ex["id"])
    assert 1 in ids and 2 in ids            # both present-in-A / absent-in-B
    assert 0 not in ids and 4 not in ids    # 0 present everywhere; 4 never quantified
    by_id = ex.set_index("id")
    assert by_id.loc[1, "on_group"] == "A" and by_id.loc[1, "off_groups"] == "B"
    assert bool(by_id.loc[1, "off_clean"]) is True      # B all not-observed
    assert bool(by_id.loc[2, "off_clean"]) is False     # B carries detected signal
    assert ids.index(1) < ids.index(2)                  # clean candidate sorts first


def test_group_exclusive_needs_groups(pg):
    with pytest.raises(ValueError):
        group_exclusive(load_quant(pg))                 # ungrouped


def test_plots_build_headless(pg):
    qd = load_quant(pg, groups=GROUPS)
    for f in (plot_missingness_vs_abundance(qd), plot_cell_state_composition(qd)):
        assert isinstance(f, Figure)