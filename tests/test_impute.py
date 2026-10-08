"""Synthetic validation of mechanism-routed imputation (ptero.quant._impute).

Checks the routing (only high_confidence / MAR_scattered / low_abundance_MNAR get
imputed; on/off, too_sparse, undetected are excluded), that imputation only fills
originally-missing cells and leaves observed values untouched, that the matrix
comes out complete, and that QRILC (MNAR) and MinProb produce left-shifted draws.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptero.quant import (
    GENOTYPE_RESTRICTED,
    LOW_ABUNDANCE_MNAR,
    TOO_SPARSE,
    UNDETECTED,
    QuantData,
    classify_missingness,
    impute_matrix,
    impute_minprob,
    impute_qrilc,
    log2_median_normalize,
)
from ptero.quant._impute import IMPUTABLE

A = [f"A{i}" for i in range(5)]   # PS19
B = [f"B{i}" for i in range(5)]   # WT
GROUPS = {**{c: "PS19" for c in A}, **{c: "WT" for c in B}}


def _make_qd(lfq):
    lfq = lfq.replace(0.0, np.nan)
    prot = pd.DataFrame(index=lfq.index)
    prot.index.name = "id"
    prot["gene"] = [str(p) for p in lfq.index]
    prot["label"] = prot["gene"].to_numpy()
    prot["majority_protein_ids"] = prot["gene"].to_numpy()
    empty = pd.DataFrame(index=lfq.index, columns=A + B, dtype=object)
    return QuantData(lfq=lfq, intensity=lfq.copy(), id_type=empty, proteins=prot,
                     samples=A + B, groups=dict(GROUPS), source=Path("syn"))


def _row(a_n, b_n, *, a_lvl, b_lvl):
    v = {c: 0.0 for c in A + B}
    for c in A[:a_n]:
        v[c] = 2.0 ** a_lvl
    for c in B[:b_n]:
        v[c] = 2.0 ** b_lvl
    return v


def _data():
    rng = np.random.default_rng(2)
    rows = {}
    for j in range(400):
        L = rng.uniform(12, 26)
        p = 0.15 + 0.85 / (1.0 + np.exp(-1.0 * (L - 18.0)))
        rows[f"bg{j}"] = {c: (2.0 ** (L + rng.normal(0, 0.2)) if rng.random() < p else 0.0)
                          for c in A + B}
    rows["highc"] = _row(5, 4, a_lvl=24, b_lvl=24)     # 9/10 present -> high_confidence
    rows["mar"] = _row(4, 3, a_lvl=27, b_lvl=27)       # 7/10, high -> MAR_scattered
    rows["mnar"] = _row(4, 3, a_lvl=13, b_lvl=13)      # 7/10, low  -> low_abundance_MNAR
    rows["onoff"] = _row(5, 0, a_lvl=24, b_lvl=24)     # 5/0 -> genotype_restricted (never impute)
    rows["sparse"] = _row(2, 1, a_lvl=24, b_lvl=24)    # 3/10 -> too_sparse (never impute)
    rows["none"] = _row(0, 0, a_lvl=24, b_lvl=24)      # undetected
    lfq = pd.DataFrame(rows).T
    lfq.index.name = "id"
    qd = _make_qd(lfq)
    diag = classify_missingness(qd, on_min=4)
    return qd, diag


def test_routing_keep_set():
    qd, diag = _data()
    imp, was_missing = impute_matrix(qd, diag, method="mechanism")
    keep_ids = set(diag.index[(diag["contaminant_class"] == "ok")
                              & diag["mechanism"].isin(IMPUTABLE)])
    assert set(imp.index) == keep_ids
    # the excluded classes must not appear
    for name, mech in [("onoff", GENOTYPE_RESTRICTED), ("sparse", TOO_SPARSE), ("none", UNDETECTED)]:
        assert diag.loc[name, "mechanism"] == mech
        assert name not in imp.index


def test_complete_and_observed_unchanged():
    qd, diag = _data()
    imp, was_missing = impute_matrix(qd, diag, method="mechanism")
    assert not imp.isna().to_numpy().any()            # fully imputed
    # observed cells are identical to the normalized input (only NaNs were filled)
    M = log2_median_normalize(qd.lfq).loc[imp.index]
    obs = ~was_missing.to_numpy()
    assert np.allclose(imp.to_numpy()[obs], M.to_numpy()[obs], equal_nan=False)
    # the mask matches the input missingness
    assert (was_missing.to_numpy() == M.isna().to_numpy()).all()


def test_qrilc_left_shifts_mnar():
    qd, diag = _data()
    imp, was_missing = impute_matrix(qd, diag, method="mechanism")
    # QRILC imputes per sample (column): an imputed cell is drawn from the low tail
    # of THAT column, so MNAR imputed cells sit below their column's observed median.
    mnar_ids = diag.index[(diag["mechanism"] == LOW_ABUNDANCE_MNAR)
                          & (diag["contaminant_class"] == "ok")]
    M = log2_median_normalize(qd.lfq)
    col_median = M.median(axis=0, skipna=True)
    vals, refs = [], []
    for pid in mnar_ids:
        if pid not in imp.index:
            continue
        miss = was_missing.loc[pid].to_numpy()
        for col in M.columns[miss]:
            vals.append(imp.loc[pid, col])
            refs.append(col_median[col])
    assert len(vals) > 0
    assert np.mean(vals) < np.mean(refs)        # imputed below the column medians


def test_minprob_downshifts():
    qd, diag = _data()
    imp, was_missing = impute_matrix(qd, diag, method="minprob",
                                     minprob_downshift=1.8, minprob_width=0.3)
    assert not imp.isna().to_numpy().any()
    M = log2_median_normalize(qd.lfq)
    # imputed values sit below the observed values of their column, on average
    filled = imp.to_numpy()[was_missing.to_numpy()]
    obs_all = M.to_numpy()[~np.isnan(M.to_numpy())]
    assert filled.mean() < np.mean(obs_all)


def test_qrilc_direct_complete_and_low():
    rng = np.random.default_rng(0)
    # left-censored columns: high observed values, ~30% missing
    X = 2.0 ** rng.normal(24, 1.5, size=(300, 4))
    M = np.log2(X)
    mask = rng.random(M.shape) < 0.3
    Md = pd.DataFrame(M, columns=list("abcd"))
    Md[mask] = np.nan
    out = impute_qrilc(Md, seed=0)
    assert not out.isna().to_numpy().any()
    # imputed values are lower than observed (left tail)
    assert out.to_numpy()[mask].mean() < np.nanmean(Md.to_numpy())


def test_requires_two_groups():
    qd, diag = _data()
    qd.groups = {c: "ALL" for c in A + B}
    with pytest.raises(ValueError, match="2 groups"):
        impute_matrix(qd, diag)

def test_minprob_rejects_all_missing_sample():
    M = pd.DataFrame({
        "A": [20.0, 21.0, 22.0],
        "B": [np.nan, np.nan, np.nan],
    })

    with pytest.raises(ValueError, match="no observed values"):
        impute_minprob(M)


def test_qrilc_rejects_all_missing_sample():
    M = pd.DataFrame({
        "A": [20.0, 21.0, 22.0],
        "B": [np.nan, np.nan, np.nan],
    })

    with pytest.raises(ValueError, match="no observed values"):
        impute_qrilc(M)