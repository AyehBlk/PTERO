"""Validation of the limma eBayes port (ptero.quant._de.limma_de).

The load-bearing test is prior recovery: simulate per-protein variances from the
exact scaled-F model limma assumes and check fitFDist recovers (s2_0, df0). The
rest check that log2FC is the group-mean difference, that real DE is detected, that
null p-values are calibrated, and that a complete matrix is required. Numbers are
validated against R limma with tolerance separately; these pin the math.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptero.quant import QuantData, complete_case_matrix, limma_de
from ptero.quant._de import _fit_fdist, _squeeze_var, _trigamma_inverse

A = [f"A{i}" for i in range(5)]   # PS19
B = [f"B{i}" for i in range(5)]   # WT
GROUPS = {**{c: "PS19" for c in A}, **{c: "WT" for c in B}}


def test_trigamma_inverse_roundtrip():
    from scipy.special import polygamma
    x = np.array([0.1, 0.5, 1.0, 3.0, 10.0])
    y = _trigamma_inverse(polygamma(1, x))
    assert np.allclose(y, x, rtol=1e-4)


def test_fitfdist_recovers_known_prior():
    rng = np.random.default_rng(0)
    n = 8000
    df_resid, df0_true, s20_true = 8, 4.0, 0.5
    sigma2 = s20_true * df0_true / rng.chisquare(df0_true, n)      # scaled inv-chi2 prior
    s2 = sigma2 * rng.chisquare(df_resid, n) / df_resid            # observed variances
    s20_hat, df0_hat = _fit_fdist(s2, df_resid)
    assert abs(df0_hat - df0_true) < 1.5
    assert abs(s20_hat - s20_true) / s20_true < 0.2


def test_squeeze_var_between_obs_and_prior():
    rng = np.random.default_rng(1)
    s2 = rng.chisquare(8, 2000) / 8 * 0.5
    s2_post, s2_0, df0 = _squeeze_var(s2, 8)
    # posterior variance is a convex blend of observed and prior -> bounded between
    lo = np.minimum(s2, s2_0)
    hi = np.maximum(s2, s2_0)
    assert np.all(s2_post >= lo - 1e-9) and np.all(s2_post <= hi + 1e-9)
    assert np.var(s2_post) < np.var(s2)            # shrinkage reduces spread


def _de_matrix(n_prot=2000, n_de=200, shift=2.0, seed=0):
    rng = np.random.default_rng(seed)
    df0_true, s20_true = 4.0, 0.4
    prot_sd = np.sqrt(s20_true * df0_true / rng.chisquare(df0_true, (n_prot, 1)))
    base = rng.normal(20, 1, (n_prot, 1))
    X = base + rng.normal(0, 1, (n_prot, 10)) * prot_sd
    X[:n_de, 5:] += shift                          # shift WT (cols 5-9) up for DE proteins
    M = pd.DataFrame(X, columns=A + B)
    M.index.name = "id"
    return M


def test_log2fc_is_group_mean_difference():
    M = _de_matrix()
    res = limma_de(M, GROUPS, ref_group="WT")
    manual = M[A].mean(axis=1) - M[B].mean(axis=1)     # PS19 - WT
    assert np.allclose(res["log2FC"].to_numpy(), manual.to_numpy())
    assert res.attrs["contrast"] == "PS19_vs_WT"


def test_detects_de_and_is_calibrated():
    M = _de_matrix(n_prot=2000, n_de=200, shift=2.0)
    res = limma_de(M, GROUPS, ref_group="WT")
    de = res.iloc[:200]
    null = res.iloc[200:]
    # DE proteins: WT shifted up -> log2FC (PS19-WT) negative, and significant
    assert de["log2FC"].mean() < -1.0
    assert (de["padj"] < 0.05).mean() > 0.7
    # null proteins: p-values roughly uniform -> ~5% below 0.05
    assert 0.02 < (null["p"] < 0.05).mean() < 0.10
    # moderation augments the residual df by the (finite) prior df
    assert np.isfinite(res.attrs["df_prior"])
    assert (res["df_total"] > res.attrs["df_residual"]).all()


def test_moderated_t_tracks_ordinary_t_sign():
    from scipy import stats
    M = _de_matrix(seed=2)
    res = limma_de(M, GROUPS, ref_group="WT")
    t_ord = stats.ttest_ind(M[A], M[B], axis=1, equal_var=True).statistic
    # same direction as the ordinary t, and strongly correlated
    same_sign = np.mean(np.sign(res["t"].to_numpy()) == np.sign(t_ord))
    assert same_sign > 0.98
    assert np.corrcoef(res["t"].to_numpy(), t_ord)[0, 1] > 0.95


def test_requires_complete_matrix():
    M = _de_matrix()
    M.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete matrix"):
        limma_de(M, GROUPS, ref_group="WT")


def test_runs_with_covariate():
    M = _de_matrix()
    batch = {**{c: ("1" if i < 3 else "2") for i, c in enumerate(A)},
             **{c: ("1" if i < 3 else "2") for i, c in enumerate(B)}}
    res = limma_de(M, GROUPS, ref_group="WT", covariates=batch)
    assert {"log2FC", "p", "padj"} <= set(res.columns)
    assert len(res) == len(M)


def test_complete_case_matrix():
    rng = np.random.default_rng(0)
    lfq = pd.DataFrame(2.0 ** rng.normal(22, 1, (6, 10)),
                       columns=A + B, index=pd.Index(range(6), name="id"))
    lfq.iloc[0, 0] = np.nan            # row 0 has a missing value -> excluded
    qd = QuantData(lfq=lfq, intensity=lfq.copy(),
                   id_type=pd.DataFrame(index=lfq.index, columns=A + B, dtype=object),
                   proteins=pd.DataFrame({"gene": list("abcdef")}, index=lfq.index),
                   samples=A + B, groups=dict(GROUPS), source=Path("syn"))
    diag = pd.DataFrame({"contaminant_class": ["ok"] * 5 + ["cRAP"]}, index=lfq.index)
    cc = complete_case_matrix(qd, diag)
    assert 0 not in cc.index           # missing-value row dropped
    assert 5 not in cc.index           # contaminant dropped
    assert not cc.isna().to_numpy().any()


# ── Welch and Wilcoxon ───────────────────────────────────────────────────────

def test_welch_matches_scipy_and_log2fc():
    from scipy import stats
    from ptero.quant import welch_de
    M = _de_matrix(seed=3)
    res = welch_de(M, GROUPS, ref_group="WT")
    ref = stats.ttest_ind(M[A], M[B], axis=1, equal_var=False)
    assert np.allclose(res["t"].to_numpy(), ref.statistic)
    assert np.allclose(res["p"].to_numpy(), ref.pvalue)
    assert np.allclose(res["log2FC"].to_numpy(),
                       (M[A].mean(axis=1) - M[B].mean(axis=1)).to_numpy())


def test_welch_detects_and_calibrated():
    M = _de_matrix(n_prot=2000, n_de=200, shift=2.0)
    from ptero.quant import welch_de
    res = welch_de(M, GROUPS, ref_group="WT")
    assert (res.iloc[:200]["p"] < 0.05).mean() > 0.7
    assert 0.02 < (res.iloc[200:]["p"] < 0.05).mean() < 0.10


def test_wilcoxon_matches_scipy_and_floored():
    from scipy import stats
    from ptero.quant import wilcoxon_de
    M = _de_matrix(seed=4)
    res = wilcoxon_de(M, GROUPS, ref_group="WT")
    ref = stats.mannwhitneyu(M[A], M[B], axis=1, alternative="two-sided")
    assert np.allclose(res["p"].to_numpy(), ref.pvalue)
    # two-sided U-test floor at 5 vs 5 is 2 / C(10,5) ~ 0.0079
    assert res["p"].min() >= 2.0 / 252.0 - 1e-9
    assert np.allclose(res["log2FC"].to_numpy(),
                       (M[A].mean(axis=1) - M[B].mean(axis=1)).to_numpy())


def test_welch_wilcoxon_need_complete_matrix():
    from ptero.quant import welch_de, wilcoxon_de
    M = _de_matrix()
    M.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete matrix"):
        welch_de(M, GROUPS, ref_group="WT")
    with pytest.raises(ValueError, match="complete matrix"):
        wilcoxon_de(M, GROUPS, ref_group="WT")