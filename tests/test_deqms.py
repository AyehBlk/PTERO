"""Validation of the DEqMS port (ptero.quant._de.deqms_de).

DEqMS = limma's linear fit plus a count-dependent variance prior. The tests pin
the behaviour that distinguishes it from limma: it rewards proteins quantified
from many peptides (tighter variance -> more power), leaves low-count proteins
about where limma had them, stays calibrated under the null, and collapses to
limma when the variance does not depend on count. The loess trend and the
IBMT prior-df grid-search are ported from DEqMS::spectraCounteBayes; numbers are
validated against R DEqMS with tolerance separately.
"""
import numpy as np
import pandas as pd
import pytest

from ptero.quant import deqms_de, limma_de
from ptero.quant._de import _deqms_prior_df, _loess_predict

A = [f"A{i}" for i in range(5)]   # PS
B = [f"B{i}" for i in range(5)]   # WT
GROUPS = {**{c: "PS" for c in A}, **{c: "WT" for c in B}}


def test_loess_recovers_smooth_trend():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 6, 400)
    truth = -0.5 * x + 2.0
    y = truth + rng.normal(0, 0.3, 400)
    pred = _loess_predict(x, y, span=0.75, degree=2)
    assert np.corrcoef(pred, truth)[0, 1] > 0.98


def test_prior_df_matches_trigamma():
    from scipy.special import polygamma
    # if mean_myfct == trigamma(d/2), the grid search should recover ~d
    for d_true in (2.0, 6.0, 20.0):
        target = float(polygamma(1, d_true / 2.0))
        d0 = _deqms_prior_df(target, numgenes=2000)
        assert abs(d0 - d_true) <= 0.6


def _matrix_counts(n_prot=3000, n_de=300, shift=1.2, var_depends_on_count=True, seed=0):
    rng = np.random.default_rng(seed)
    count = rng.integers(1, 60, n_prot)
    sd = (0.25 + 2.0 / np.sqrt(count)) if var_depends_on_count else np.full(n_prot, 0.7)
    base = rng.normal(20, 1, (n_prot, 1))
    X = base + rng.normal(0, 1, (n_prot, 10)) * sd[:, None]
    X[:n_de, 5:] += shift                                   # WT shifted up -> DE
    M = pd.DataFrame(X, columns=A + B)
    M.index.name = "id"
    return M, pd.Series(count, index=M.index, name="count")


def test_output_sane_and_log2fc_matches_limma():
    M, counts = _matrix_counts()
    deq = deqms_de(M, GROUPS, counts, ref_group="WT")
    lim = limma_de(M, GROUPS, ref_group="WT")
    assert not deq[["log2FC", "t", "p", "padj"]].isna().to_numpy().any()
    assert np.allclose(deq["log2FC"].to_numpy(), lim["log2FC"].to_numpy())   # same fit
    assert (deq["p"] >= 0).all() and (deq["p"] <= 1).all()
    assert deq.attrs["contrast"] == "PS_vs_WT"


def test_rewards_high_count_proteins():
    """The DEqMS signature: at equal effect size, high-count DE proteins get
    smaller p than under limma; low-count proteins are not inflated."""
    M, counts = _matrix_counts(n_de=300, shift=1.2)
    lim = limma_de(M, GROUPS, ref_group="WT")
    deq = deqms_de(M, GROUPS, counts, ref_group="WT")
    de = slice(0, 300)
    hi = counts.to_numpy()[de] >= 30
    lo = counts.to_numpy()[de] < 30
    lp, dp = lim["p"].to_numpy()[de], deq["p"].to_numpy()[de]
    # high-count DE proteins: DEqMS more significant for the large majority
    assert np.mean(dp[hi] < lp[hi]) > 0.85
    # low-count DE proteins: DEqMS does not systematically inflate them
    assert np.mean(dp[lo] < lp[lo]) < 0.6


def test_calibrated_under_null():
    M, counts = _matrix_counts(n_de=300)
    deq = deqms_de(M, GROUPS, counts, ref_group="WT")
    null = deq["p"].to_numpy()[300:]
    assert 0.02 < (null < 0.05).mean() < 0.10


def test_collapses_to_limma_when_variance_flat():
    M, counts = _matrix_counts(var_depends_on_count=False, seed=1)
    lim = limma_de(M, GROUPS, ref_group="WT")
    deq = deqms_de(M, GROUPS, counts, ref_group="WT")
    rho = pd.Series(deq["p"].to_numpy()).rank().corr(
        pd.Series(lim["p"].to_numpy()).rank(), method="spearman")
    assert rho > 0.99


def test_pseudocount_when_min_count_zero():
    M, counts = _matrix_counts(n_prot=500)
    counts.iloc[:5] = 0                       # force a zero count
    deq = deqms_de(M, GROUPS, counts, ref_group="WT")
    assert not deq["p"].isna().any()
    assert (deq["count"] >= 1).all()          # pseudocount applied


def test_requires_complete_matrix():
    M, counts = _matrix_counts(n_prot=200)
    M.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete matrix"):
        deqms_de(M, GROUPS, counts, ref_group="WT")


def test_fit_methods_run():
    M, counts = _matrix_counts(n_prot=400)
    for fm in ("loess", "spline", "nls"):
        deq = deqms_de(M, GROUPS, counts, ref_group="WT", fit_method=fm)
        assert not deq["p"].isna().any(), fm
