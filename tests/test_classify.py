"""
Synthetic validation of the missingness MECHANISM classifier (ptero.quant._missing).

Different data than the reference R analysis (R = DIA-NN; PTERO here = MaxQuant),
so these tests validate the *logic*: planted proteins must land in the mechanism
class the 03c cascade specifies, LOD50 must be recovered from a known detection
gradient, Fisher must match scipy, and the routing flags (near_floor, call_ready)
must behave. Real-data fidelity is checked separately by sanity (Gbp2b, class
counts, LOD50, plot shape) on the user's MaxQuant output.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptero.quant import (
    CONDITION_DEPENDENT,
    GENOTYPE_RESTRICTED,
    HIGH_CONFIDENCE,
    LOW_ABUNDANCE_MNAR,
    MAR_SCATTERED,
    TOO_SPARSE,
    UNDETECTED,
    QuantData,
    classify_missingness,
    fisher_2x2,
    log2_median_normalize,
)
from ptero.quant._missing import _irls_binomial, _logistic_lod50

A_COLS = [f"A{i}" for i in range(10)]
B_COLS = [f"B{i}" for i in range(10)]
GROUPS = {**{c: "A" for c in A_COLS}, **{c: "B" for c in B_COLS}}
MID = 22.0          # log2 intensity at 50% detection in the background gradient


def _make_qd(lfq: pd.DataFrame, groups=GROUPS) -> QuantData:
    lfq = lfq.replace(0.0, np.nan)          # mirror load_quant: 0 == not quantified
    samples = list(lfq.columns)
    proteins = pd.DataFrame(index=lfq.index)
    proteins.index.name = "id"
    proteins["gene"] = [f"G{p}" for p in lfq.index]
    proteins["label"] = proteins["gene"].to_numpy()
    proteins["majority_protein_ids"] = [f"P{p}" for p in lfq.index]
    empty = pd.DataFrame(index=lfq.index, columns=samples, dtype=object)
    return QuantData(lfq=lfq, intensity=lfq.copy(), id_type=empty, proteins=proteins,
                     samples=samples, groups=dict(groups), source=Path("synthetic"))


def _row(a_present, b_present, log2_level, *, a_level=None, b_level=None, rng=None):
    """Build one protein row: `a_present`/`b_present` samples carry intensity at
    `log2_level` (group-specific overrides via a_level/b_level), the rest are 0."""
    vals = {c: 0.0 for c in A_COLS + B_COLS}
    la = a_level if a_level is not None else log2_level
    lb = b_level if b_level is not None else log2_level
    for c in A_COLS[:a_present]:
        vals[c] = 2.0 ** la
    for c in B_COLS[:b_present]:
        vals[c] = 2.0 ** lb
    return vals


def _background(rng, n=160):
    """Proteins with a clean intensity->detection gradient so LOD50 ~ MID."""
    rows = {}
    for j in range(n):
        Lj = rng.uniform(16.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-1.3 * (Lj - MID)))
        vals = {}
        for c in A_COLS + B_COLS:
            vals[c] = 2.0 ** (Lj + rng.normal(0, 0.2)) if rng.random() < p else 0.0
        rows[f"bg{j}"] = vals
    return rows


def _classified():
    rng = np.random.default_rng(1)
    rows = _background(rng)
    # planted proteins, one per target mechanism (high, far-from-MID levels)
    rows["undetected"] = _row(0, 0, 0.0)
    rows["restricted_ready"] = _row(9, 0, 26.0)          # 0 in B, 9 in A   -> call_ready
    rows["restricted_notready"] = _row(4, 0, 26.0)       # 0 in B, 4 in A   -> not call_ready
    rows["cd_near_floor"] = _row(9, 1, 26.0, b_level=20.0)   # A 90%, B 10% low  -> near_floor
    rows["cd_above_floor"] = _row(9, 2, 26.0, b_level=28.0)  # A 90%, B 20% high -> not near_floor
    rows["high_conf"] = _row(10, 9, 26.0)                # 19/20 present
    rows["too_sparse"] = _row(4, 4, 26.0)               # 8/20, symmetric, high
    rows["mnar"] = _row(7, 7, 20.0)                     # 14/20, symmetric, LOW
    rows["mar"] = _row(7, 7, 28.0)                     # 14/20, symmetric, HIGH
    lfq = pd.DataFrame(rows).T
    lfq.index.name = "id"
    qd = _make_qd(lfq)
    return classify_missingness(qd)


def test_mechanism_assignments():
    d = _classified()
    m = d["mechanism"]
    assert m["undetected"] == UNDETECTED
    assert m["restricted_ready"] == GENOTYPE_RESTRICTED
    assert m["restricted_notready"] == GENOTYPE_RESTRICTED
    assert m["cd_near_floor"] == CONDITION_DEPENDENT
    assert m["cd_above_floor"] == CONDITION_DEPENDENT
    assert m["high_conf"] == HIGH_CONFIDENCE
    assert m["too_sparse"] == TOO_SPARSE
    assert m["mnar"] == LOW_ABUNDANCE_MNAR
    assert m["mar"] == MAR_SCATTERED


def test_call_ready_needs_on_min():
    d = _classified()
    assert bool(d.loc["restricted_ready", "call_ready"]) is True
    assert bool(d.loc["restricted_notready", "call_ready"]) is False
    # call_ready is only ever set for genotype_restricted
    assert not bool(d.loc["high_conf", "call_ready"])


def test_near_floor_flag():
    d = _classified()
    assert d.loc["cd_near_floor", "near_floor"] is True
    assert d.loc["cd_above_floor", "near_floor"] is False
    # near_floor is None for non condition-dependent rows
    assert d.loc["high_conf", "near_floor"] is None


def test_mnar_mar_split_consistent_with_lod():
    d = _classified()
    lod = d.attrs["lod50"]
    assert d.loc["mnar", "min_log2_intensity"] <= lod      # reached the floor
    assert d.loc["mar", "min_log2_intensity"] > lod        # above the floor


def test_detection_fisher_only_for_candidates():
    d = _classified()
    # restricted / condition-dependent rows get a Fisher detection p
    assert np.isfinite(d.loc["restricted_ready", "detection_fisher_p"])
    assert np.isfinite(d.loc["cd_near_floor", "detection_fisher_p"])
    # a symmetric high-confidence row is not a candidate -> NaN
    assert pd.isna(d.loc["high_conf", "detection_fisher_p"])
    # OR points the right way: restricted (present in A) -> enriched in A (OR>1 or inf)
    assert d.loc["restricted_ready", "detection_OR"] > 1


def test_lod50_recovers_known_midpoint():
    rng = np.random.default_rng(0)
    x = np.linspace(MID - 6, MID + 6, 60)
    p = 1.0 / (1.0 + np.exp(-1.2 * (x - MID)))
    n_trials = 100
    succ = np.round(p * n_trials)
    fail = n_trials - succ
    L = _logistic_lod50(x, succ, succ + fail, 0.5)
    assert L["source"] == "logistic_LOD50"
    assert abs(L["lod"] - MID) < 0.4


def test_lod50_fallback_when_no_slope():
    # detection independent of intensity -> non-positive slope -> quantile fallback
    rng = np.random.default_rng(2)
    x = rng.uniform(16, 30, 80)
    succ = np.full(80, 10.0)
    fail = np.full(80, 10.0)
    L = _logistic_lod50(x, succ, succ + fail, 0.5)
    assert L["source"] == "fallback_quantile"
    assert abs(L["lod"] - np.quantile(x, 0.5)) < 1e-6


def test_fisher_2x2_matches_scipy():
    from scipy.stats import fisher_exact
    from scipy.stats.contingency import odds_ratio
    table = [[8, 2], [2, 8]]
    p, OR, lo, hi = fisher_2x2(*[v for row in table for v in row])
    assert p == pytest.approx(fisher_exact(table, alternative="two-sided")[1])
    ref = odds_ratio(table, kind="conditional")
    assert OR == pytest.approx(ref.statistic)
    ci = ref.confidence_interval(0.95)
    assert (lo, hi) == pytest.approx((ci.low, ci.high))


def test_requires_exactly_two_groups():
    lfq = pd.DataFrame({c: [2.0 ** 25] * 3 for c in A_COLS + B_COLS},
                       index=pd.Index([0, 1, 2], name="id"))
    # one group
    qd1 = _make_qd(lfq, groups={c: "A" for c in A_COLS + B_COLS})
    with pytest.raises(ValueError, match="exactly 2 groups"):
        classify_missingness(qd1)
    # three groups
    g3 = dict(GROUPS); g3["A0"] = "C"; g3["A1"] = "C"
    qd3 = _make_qd(lfq, groups=g3)
    with pytest.raises(ValueError, match="exactly 2 groups"):
        classify_missingness(qd3)


def test_log2_median_normalize_aligns_medians():
    rng = np.random.default_rng(3)
    base = rng.uniform(20, 28, size=(100, 6))
    offsets = np.array([0.0, 1.0, -1.0, 2.0, -0.5, 0.3])
    raw = 2.0 ** (base + offsets)            # per-sample shifts in log2 space
    df = pd.DataFrame(raw, columns=[f"s{i}" for i in range(6)])
    norm = log2_median_normalize(df, method="median")
    meds = norm.median(axis=0, skipna=True).to_numpy()
    assert np.allclose(meds, meds[0])        # all per-sample medians equal
    # 'none' is just log2
    assert np.allclose(log2_median_normalize(df, method="none").to_numpy(),
                       np.log2(raw))


def test_irls_matches_closed_form_direction():
    # monotone increasing detection -> positive slope
    x = np.array([10.0, 14.0, 18.0, 22.0, 26.0, 30.0])
    succ = np.array([1.0, 3.0, 6.0, 14.0, 18.0, 20.0])
    fail = 20.0 - succ
    b0, b1 = _irls_binomial(x, succ, fail)
    assert b1 > 0