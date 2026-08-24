"""Synthetic validation of the Fisher on/off track (ptero.quant._de).

Planted on/off proteins (both directions), a condition-dependent skew, a
near-floor vs above-floor on/off, and a both-present non-candidate. Checks the
Fisher p / conditional OR, BH over the detected family, the detection-limited
log2FC sign, the enriched-group direction, and the near_floor refinement.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptero.quant import (
    QuantData,
    classify_missingness,
    plot_onoff,
    presence_absence,
)
from matplotlib.figure import Figure

A = [f"A{i}" for i in range(5)]   # group "PS19" (sorts first -> A side)
B = [f"B{i}" for i in range(5)]   # group "WT"
GROUPS = {**{c: "PS19" for c in A}, **{c: "WT" for c in B}}
MID = 22.0


def _make_qd(lfq: pd.DataFrame) -> QuantData:
    lfq = lfq.replace(0.0, np.nan)
    proteins = pd.DataFrame(index=lfq.index)
    proteins.index.name = "id"
    proteins["gene"] = [str(p) for p in lfq.index]
    proteins["label"] = proteins["gene"].to_numpy()
    proteins["majority_protein_ids"] = proteins["gene"].to_numpy()
    empty = pd.DataFrame(index=lfq.index, columns=A + B, dtype=object)
    return QuantData(lfq=lfq, intensity=lfq.copy(), id_type=empty, proteins=proteins,
                     samples=A + B, groups=dict(GROUPS), source=Path("syn"))


def _row(a_n, b_n, *, a_lvl=26.0, b_lvl=26.0):
    v = {c: 0.0 for c in A + B}
    for c in A[:a_n]:
        v[c] = 2.0 ** a_lvl
    for c in B[:b_n]:
        v[c] = 2.0 ** b_lvl
    return v


def _pa():
    rng = np.random.default_rng(1)
    rows = {}
    for j in range(300):
        L = rng.uniform(15, 30)
        p = 1.0 / (1.0 + np.exp(-1.3 * (L - MID)))
        rows[f"bg{j}"] = {c: (2.0 ** (L + rng.normal(0, 0.2)) if rng.random() < p else 0.0)
                          for c in A + B}
    rows["ps_clean"] = _row(5, 0, a_lvl=26.0)            # PS19-specific, above floor
    rows["wt_clean"] = _row(0, 5, b_lvl=26.0)            # WT-specific, above floor
    rows["ps_floor"] = _row(5, 0, a_lvl=21.5)            # PS19-specific, at the floor
    rows["cd_skew"] = _row(4, 1, a_lvl=26.0, b_lvl=26.0)  # condition_dependent
    rows["both"] = _row(5, 5, a_lvl=26.0)                # present both -> NOT a candidate
    lfq = pd.DataFrame(rows).T
    lfq.index.name = "id"
    qd = _make_qd(lfq)
    diag = classify_missingness(qd, on_min=4)
    return qd, diag, presence_absence(qd, diag)


def test_candidate_set():
    _, diag, pa = _pa()
    # the four on/off-type planted proteins are candidates; the both-present one is not
    assert {"ps_clean", "wt_clean", "ps_floor", "cd_skew"} <= set(pa.index)
    assert "both" not in pa.index
    assert pa.loc["both" if "both" in pa.index else "ps_clean", "mechanism"]  # smoke
    assert pa.attrs["bh_family_size"] >= len(pa)        # family >= candidates


def test_direction_and_detlim_sign():
    _, _, pa = _pa()
    assert pa.loc["ps_clean", "enriched_in"] == "PS19"
    assert pa.loc["wt_clean", "enriched_in"] == "WT"
    assert pa.loc["ps_clean", "detlim_log2FC"] > 0      # + = higher in PS19 (group A)
    assert pa.loc["wt_clean", "detlim_log2FC"] < 0
    assert bool(pa.loc["ps_clean", "is_on_off"]) is True
    assert bool(pa.loc["cd_skew", "is_on_off"]) is False  # both groups > 0


def test_odds_ratio_direction():
    _, _, pa = _pa()
    assert np.isinf(pa.loc["ps_clean", "OR"]) or pa.loc["ps_clean", "OR"] > 1
    assert pa.loc["wt_clean", "OR"] == 0 or pa.loc["wt_clean", "OR"] < 1


def test_fisher_p_and_bh():
    _, _, pa = _pa()
    # 5/5 vs 0/5 is the most extreme split available with n=5+5
    assert pa.loc["ps_clean", "fisher_p"] == pytest.approx(2 / 252, rel=1e-3)
    # padj is BH-adjusted -> >= raw p, and within [0, 1]
    assert pa.loc["ps_clean", "fisher_padj"] >= pa.loc["ps_clean", "fisher_p"]
    assert (pa["fisher_padj"].dropna() <= 1.0).all()


def test_near_floor_refinement():
    _, _, pa = _pa()
    # a true on/off whose present group sits at LOD50 is flagged; one well above is not
    assert bool(pa.loc["ps_floor", "near_floor"]) is True
    assert bool(pa.loc["ps_clean", "near_floor"]) is False


def test_only_ok_excludes_contaminants():
    qd, diag, _ = _pa()
    # tag one candidate as a contaminant; with only_ok it drops from the table
    cc = pd.Series("ok", index=diag.index)
    cc.loc["wt_clean"] = "cRAP"
    pa = presence_absence(qd, diag.assign(contaminant_class=cc), only_ok=True)
    assert "wt_clean" not in pa.index
    assert "ps_clean" in pa.index


def test_plot_builds():
    _, _, pa = _pa()
    assert isinstance(plot_onoff(pa), Figure)