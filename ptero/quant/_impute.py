"""
ptero.quant._impute -- mechanism-routed missing-value imputation for the DE
"imputed" track.

A point imputation is only valid when the mechanism is known, so the missingness
classifier (:func:`ptero.quant.classify_missingness`) decides what each protein
gets, and this module applies it:

* ``low_abundance_MNAR``               -> QRILC (left-censored draw)
* ``high_confidence`` / ``MAR_scattered`` -> within-group kNN
* on/off (``genotype_restricted`` / ``condition_dependent``), ``too_sparse``,
  ``undetected``                       -> NEVER imputed (excluded from the matrix)

That routing is the reference ``impute_known_mechanism`` logic. Two other whole-
matrix strategies are provided for the imputation-sensitivity comparison the user
asked for: ``minprob`` (a Perseus-style downshifted-normal draw, applied to every
missing value in the imputable set) and single-method ``knn`` / ``qrilc``.

Fidelity: QRILC and within-group kNN have no exact Python equivalent of the R
(imputeLCMD / impute) implementations, so the cores are ported and validated by
behaviour (imputed values are left-shifted / neighbour-consistent, and the matrix
comes out complete). Numbers are validated against R outputs with tolerance, not
expected to match bit-for-bit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._missing import (
    HIGH_CONFIDENCE,
    LOW_ABUNDANCE_MNAR,
    MAR_SCATTERED,
    log2_median_normalize,
)

if TYPE_CHECKING:
    from ._quant import QuantData

# classes that are eligible for imputation (everything else is excluded)
IMPUTABLE = (HIGH_CONFIDENCE, MAR_SCATTERED, LOW_ABUNDANCE_MNAR)


def impute_qrilc(M: pd.DataFrame, *, tune_sigma: float = 1.0, seed: int = 1) -> pd.DataFrame:
    """QRILC-style left-censored imputation, per sample (column).

    For each sample the observed values are treated as the upper part of a normal
    whose lower tail was lost to the detection limit. The underlying mean/sd are
    estimated by regressing observed quantiles on the matching standard-normal
    quantiles (a QQ fit, accounting for the missing fraction), and the missing
    values are drawn from that normal truncated at the detection threshold (the
    missing-fraction quantile). Mirrors ``imputeLCMD::impute.QRILC`` in behaviour.
    """
    from scipy.stats import norm, truncnorm

    rng = np.random.default_rng(seed)
    X = M.to_numpy(dtype=float).copy()
    n, m = X.shape
    u = np.linspace(0.05, 0.95, 50)
    for j in range(m):
        col = X[:, j]
        miss = np.isnan(col)
        nm = int(miss.sum())
        if nm == 0:
            continue
        obs = col[~miss]
        if obs.size == 0:
            raise ValueError(
            f"sample {M.columns[j]!r} has no observed values; "
            "QRILC imputation is undefined"
            )
        if obs.size < 3:
            X[miss, j] = np.nanmin(col)
            continue
        pNA = nm / n
        full_p = pNA + u * (1.0 - pNA)
        qn = norm.ppf(full_p)
        qo = np.quantile(obs, u)
        A = np.column_stack([np.ones_like(qn), qn])
        coef, *_ = np.linalg.lstsq(A, qo, rcond=None)
        mean_cdd, sd_cdd = float(coef[0]), float(coef[1])
        if not np.isfinite(sd_cdd) or sd_cdd <= 0:
            mean_cdd, sd_cdd = float(np.mean(obs)), float(np.std(obs) or 1.0)
        sd_draw = sd_cdd * tune_sigma
        b = (norm.ppf(pNA, loc=mean_cdd, scale=sd_cdd) - mean_cdd) / sd_draw
        X[miss, j] = truncnorm.rvs(-np.inf, b, loc=mean_cdd, scale=sd_draw,
                                   size=nm, random_state=rng)
    return pd.DataFrame(X, index=M.index, columns=M.columns)


def impute_minprob(M: pd.DataFrame, *, width: float = 0.3, downshift: float = 1.8,
                   seed: int = 1) -> pd.DataFrame:
    """MinProb (Perseus-style) imputation, per sample (column).

    Missing values are drawn from a narrow normal shifted below the sample's
    observed distribution: ``N(mean_obs - downshift*sd_obs, (width*sd_obs)^2)``.
    A simple, widely used MNAR imputation; here it serves as the sensitivity
    alternative to the mechanism-routed scheme.
    """
    rng = np.random.default_rng(seed)
    X = M.to_numpy(dtype=float).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        nm = int(miss.sum())
        if nm == 0:
            continue
        obs = col[~miss]
        if obs.size == 0:
            raise ValueError(
            f"sample {M.columns[j]!r} has no observed values; "
            "MinProb imputation is undefined"
            )
        mu, sd = float(np.mean(obs)), float(np.std(obs) or 1.0)
        X[miss, j] = rng.normal(mu - downshift * sd, max(width * sd, 1e-9), size=nm)
    return pd.DataFrame(X, index=M.index, columns=M.columns)


def _knn_within_groups(M: pd.DataFrame, members, groups, k: int) -> pd.DataFrame:
    """Within-group gene-wise kNN (sklearn KNNImputer per group submatrix).

    KNNImputer treats matrix rows (proteins) as samples and columns (MS-samples)
    as features, so it imputes a missing cell from the k nearest *proteins* by
    their pattern over the other samples -- the same grain as R ``impute.knn``.
    Run separately on each group's columns so genotype signal is not used to fill.
    """
    from sklearn.impute import KNNImputer

    out = M.copy()
    for g in groups:
        cols = members[g]
        if len(cols) < 2:
            continue
        block = M[cols].to_numpy(dtype=float)
        nbrs = max(1, min(k, block.shape[0] - 1))
        out[cols] = KNNImputer(n_neighbors=nbrs).fit_transform(block)
    return out


def impute_matrix(
    qd: "QuantData",
    diag: pd.DataFrame,
    *,
    method: str = "mechanism",
    normalization: str = "median",
    knn_k: int = 10,
    qrilc_tune_sigma: float = 1.0,
    minprob_width: float = 0.3,
    minprob_downshift: float = 1.8,
    seed: int = 1,
):
    """Impute the imputable proteins of a two-group matrix for the DE imputed track.

    Restricts to contaminant-free proteins whose mechanism is in
    :data:`IMPUTABLE` (high_confidence / MAR_scattered / low_abundance_MNAR) --
    on/off, too_sparse and undetected are excluded, never imputed.

    ``method``:
      * ``"mechanism"`` (default) -- QRILC for ``low_abundance_MNAR``,
        within-group kNN for the rest (the reference routing).
      * ``"minprob"`` -- MinProb downshift for every missing value (sensitivity).
      * ``"knn"`` / ``"qrilc"`` -- a single method applied throughout.

    Returns ``(imputed, was_missing)``: a complete (no-NaN) normalized log2 matrix
    over the imputable proteins, and a boolean mask of which cells were originally
    missing (i.e. imputed). ``imputed.attrs`` records the method, group labels, and
    imputed row/cell counts.
    """
    members = qd.group_members()
    groups = sorted(members)
    if len(groups) != 2:
        raise ValueError(f"impute_matrix needs exactly 2 groups (got {groups or 'none'}).")

    M = log2_median_normalize(qd.lfq, method=normalization)
    cc = pd.Series(diag["contaminant_class"]).reindex(M.index).to_numpy()
    mech = pd.Series(diag["mechanism"]).reindex(M.index).to_numpy()
    keep = (cc == "ok") & np.isin(mech, IMPUTABLE)
    Mk = M.loc[keep]
    was_missing = Mk.isna()

    if method == "mechanism":
        imp = _knn_within_groups(Mk, members, groups, knn_k)
        q = impute_qrilc(Mk, tune_sigma=qrilc_tune_sigma, seed=seed)
        mnar_idx = Mk.index[mech[keep] == LOW_ABUNDANCE_MNAR]
        imp.loc[mnar_idx] = q.loc[mnar_idx]
    elif method == "minprob":
        imp = impute_minprob(Mk, width=minprob_width, downshift=minprob_downshift, seed=seed)
    elif method == "knn":
        imp = _knn_within_groups(Mk, members, groups, knn_k)
    elif method == "qrilc":
        imp = impute_qrilc(Mk, tune_sigma=qrilc_tune_sigma, seed=seed)
    else:
        raise ValueError(f"method must be 'mechanism' | 'minprob' | 'knn' | 'qrilc', got {method!r}.")

    # guarantee completeness: any cell a method couldn't fill (e.g. kNN with no
    # informative neighbour) is back-filled with a MinProb draw.
    if imp.isna().to_numpy().any():
        fb = impute_minprob(imp, width=minprob_width, downshift=minprob_downshift, seed=seed + 1)
        imp = imp.where(imp.notna(), fb)

    imp.attrs["method"] = method
    imp.attrs["groups"] = tuple(groups)
    imp.attrs["n_imputed_rows"] = int(keep.sum())
    imp.attrs["n_imputed_cells"] = int(was_missing.to_numpy().sum())
    return imp, was_missing