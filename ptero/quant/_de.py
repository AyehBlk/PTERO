"""
ptero.quant._de -- differential detection / abundance tracks.

Step 2 of the arc: the PRESENCE/ABSENCE (on/off) track. The missingness
classifier (:func:`ptero.quant.classify_missingness`) routes ``genotype_restricted``
(0 in one group) and ``condition_dependent`` (strongly skewed) proteins here
instead of imputing them, because their missing intensities are unidentifiable.
This module gives each such candidate a Fisher's exact test on the detection 2x2
plus a conditional-MLE odds ratio + 95% CI (matching R ``fisher.test``), with BH
computed over the whole detected-in->=1 family (the conservative correction).

Two refinements over a bare on/off list:
* **detection-limited log2FC** for the volcano "diamonds": median of the present
  group minus a region detection floor (1% quantile of observed intensities) when
  the other group is fully absent -- a left-censored lower bound, clearly flagged.
* **near_floor** is extended to ``genotype_restricted``: a true on/off whose
  PRESENT group sits at the detection floor (mean obs <= LOD50 + margin) may be
  low-abundance dropout rather than true absence (the Camkmt-at-LOD50 case). For
  ``condition_dependent`` the classifier's own near_floor (sparse-group mean) is
  carried through. Either way it marks candidates needing a second look (the PTM
  / proteoform check is the other half of that disambiguation).

Sign / direction convention: groups are sorted; the first is ``A``. A positive
``detlim_log2FC`` and an odds ratio > 1 both mean *higher / more-detected in A*.
The (complete-case) quantitative volcano body is step 3 (limma); this module is
the diamonds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ._plots import _style_axes
from ._missing import (
    CONDITION_DEPENDENT,
    GENOTYPE_RESTRICTED,
    fisher_2x2,
    log2_median_normalize,
)

if TYPE_CHECKING:
    from ._quant import QuantData

# direction colours (A = first sorted group; e.g. PS19 orange, WT blue)
_DIR_COLORS = ("#DE8F05", "#0173B2")


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN-safe over the input array)."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def presence_absence(
    qd: "QuantData",
    diag: pd.DataFrame,
    *,
    normalization: str = "median",
    floor_quantile: float = 0.01,
    cd_floor_margin: float = 1.0,
    only_ok: bool = True,
) -> pd.DataFrame:
    """Fisher on/off detection track for the classifier's on/off candidates.

    Parameters
    ----------
    qd : QuantData
        The same two-group data passed to :func:`classify_missingness` (used for
        the detection-limited log2FC, computed from the normalized matrix).
    diag : DataFrame
        Output of :func:`classify_missingness` (its ``.attrs`` carry the learned
        LOD50 and group labels). ``normalization`` must match what produced it.
    floor_quantile : float
        Quantile of observed log2 intensities used as the detection floor for the
        absent group's level in ``detlim_log2FC`` (default 0.01).
    cd_floor_margin : float
        Margin above LOD50 for the ``near_floor`` flag (default 1.0 log2).
    only_ok : bool
        If True (default), restrict both the BH family and the candidate set to
        contaminant-free (``contaminant_class == "ok"``) proteins.

    Returns
    -------
    DataFrame indexed by protein-group ``id``, one row per on/off candidate
    (mechanism ``genotype_restricted`` or ``condition_dependent``), sorted by
    Fisher p then by |detlim_log2FC|. Columns include the per-group observed
    counts/fractions, ``enriched_in``, ``is_on_off``, ``detlim_log2FC``,
    ``fisher_p`` / ``fisher_padj`` (BH over the whole detected family),
    ``OR`` / ``OR_lo`` / ``OR_hi`` (conditional MLE + 95% CI), ``near_floor``,
    ``call_ready``, ``mechanism``, ``contaminant_class``. Run-level provenance
    (floor value, LOD50, groups, BH family size) is attached in ``.attrs``.
    """
    from scipy.stats import fisher_exact

    members = qd.group_members()
    groups = sorted(members)
    if len(groups) != 2:
        raise ValueError(f"presence_absence needs exactly 2 groups (got {groups or 'none'}).")
    ga, gb = groups
    nA, nB = len(members[ga]), len(members[gb])
    lod = diag.attrs.get("lod50", float("nan"))

    oA = diag[f"obs_{ga}"].to_numpy()
    oB = diag[f"obs_{gb}"].to_numpy()
    cc = diag["contaminant_class"].to_numpy()
    mech = diag["mechanism"].to_numpy()
    detected = (oA + oB) > 0
    family = detected & (cc == "ok") if only_ok else detected

    # Fisher p over the whole detected family (cheap), then BH over that family
    p = np.full(len(diag), np.nan)
    for i in np.nonzero(family)[0]:
        a = int(oA[i]); b = int(oB[i])
        p[i] = fisher_exact([[a, nA - a], [b, nB - b]], alternative="two-sided")[1]
    padj = _bh(p)  # NaN outside the family; BH only over finite (family) entries

    # detection-limited log2FC from the normalized matrix
    M = log2_median_normalize(qd.lfq, method=normalization)
    floor_val = float(np.nanquantile(M.to_numpy(), floor_quantile))
    medA = M[members[ga]].median(axis=1, skipna=True).to_numpy()
    medB = M[members[gb]].median(axis=1, skipna=True).to_numpy()
    levelA = np.where(np.isfinite(medA), medA, floor_val)
    levelB = np.where(np.isfinite(medB), medB, floor_val)
    detlim = levelA - levelB

    meanA = diag[f"mean_{ga}_obs"].to_numpy()
    meanB = diag[f"mean_{gb}_obs"].to_numpy()
    diag_nf = diag["near_floor"].to_numpy()
    has_gene = "gene" in diag.columns
    has_label = "label" in diag.columns
    call_ready = diag["call_ready"].to_numpy()

    cand = family & np.isin(mech, [GENOTYPE_RESTRICTED, CONDITION_DEPENDENT])
    rows = []
    for i in np.nonzero(cand)[0]:
        a, b = int(oA[i]), int(oB[i])
        _, OR, lo, hi = fisher_2x2(a, nA - a, b, nB - b)
        fA, fB = a / nA, b / nB
        enriched = ga if fA >= fB else gb
        is_onoff = (a == 0) or (b == 0)
        if mech[i] == GENOTYPE_RESTRICTED:
            present_mean = meanA[i] if a > 0 else meanB[i]
            nf = bool(np.isfinite(present_mean) and present_mean <= lod + cd_floor_margin)
        else:
            v = diag_nf[i]
            nf = bool(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else False
        rows.append({
            "id": diag.index[i],
            "gene": diag["gene"].to_numpy()[i] if has_gene else None,
            "label": diag["label"].to_numpy()[i] if has_label else None,
            "mechanism": mech[i],
            f"obs_{ga}": a, f"obs_{gb}": b,
            f"frac_{ga}": round(fA, 2), f"frac_{gb}": round(fB, 2),
            "enriched_in": enriched,
            "is_on_off": is_onoff,
            "detlim_log2FC": round(float(detlim[i]), 3),
            "fisher_p": float(p[i]),
            "fisher_padj": float(padj[i]),
            "OR": OR, "OR_lo": lo, "OR_hi": hi,
            "near_floor": nf,
            "call_ready": bool(call_ready[i]),
            "contaminant_class": cc[i],
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.set_index("id")
        out = out.sort_values(["fisher_p", "detlim_log2FC"],
                              key=lambda s: s.abs() if s.name == "detlim_log2FC" else s,
                              ascending=[True, False])
    out.attrs["floor_value"] = floor_val
    out.attrs["lod50"] = lod
    out.attrs["groups"] = (ga, gb)
    out.attrs["n_per_group"] = {ga: nA, gb: nB}
    out.attrs["bh_family_size"] = int(family.sum())
    out.attrs["n_candidates"] = int(len(out))
    return out


def plot_onoff(pa: pd.DataFrame, *, which: str = "call_ready", top: int = 25,
               figsize=None) -> Figure:
    """Detection-pattern plot for on/off candidates: a back-to-back bar of how
    many samples in each group detected the protein.

    With small n the Fisher p is discrete (a volcano just stacks points on a few
    lines), so this shows the actual evidence instead: each protein is a bar
    extending right by its detected-sample count in group A and left by its count
    in group B. Colour = group; **hatched/faded = ``near_floor``** (the signal sits
    at the detection limit -- possibly low-abundance rather than a true on/off).

    ``which`` selects the rows: ``"call_ready"`` (default; robust true on/off,
    >= on_min in the present group), ``"on_off"`` (all true on/off), or ``"all"``
    (every candidate). At most ``top`` rows are shown, ranked by how clean the
    split is (|count difference|). The detection-limited log2FC and full stats
    stay in the table; the combined volcano (with the limma body) is step 3.
    """
    ga, gb = pa.attrs.get("groups", ("A", "B"))
    npg = pa.attrs.get("n_per_group", {})
    nA = int(npg.get(ga, 5)); nB = int(npg.get(gb, 5))
    cA, cB = f"obs_{ga}", f"obs_{gb}"

    if which == "on_off":
        sub = pa[pa["is_on_off"]].copy()
    elif which == "all":
        sub = pa.copy()
    else:
        sub = pa[pa["call_ready"]].copy()

    if sub.empty:
        fig = Figure(figsize=figsize or (7, 3), layout="constrained")
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"no '{which}' on/off candidates", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        _style_axes(ax)
        return fig

    # rank by how clean the split is, keep the top, then order for display
    sub["_clean"] = (sub[cA] - sub[cB]).abs()
    sub = sub.sort_values(["_clean"], ascending=False).head(top)
    sub["_net"] = sub[cA] / nA - sub[cB] / nB
    sub = sub.sort_values("_net")                 # WT-enriched at the bottom, A at top

    n = len(sub)
    fig = Figure(figsize=figsize or (8.2, max(3.0, 0.34 * n + 1.6)), layout="constrained")
    ax = fig.add_subplot(111)
    cA_col, cB_col = _DIR_COLORS
    names, ticks_nf = [], []
    for i, (idx, r) in enumerate(sub.iterrows()):
        a, b = int(r[cA]), int(r[cB])
        nf = bool(r["near_floor"])
        common = dict(height=0.66, alpha=0.35 if nf else 0.9,
                      hatch="////" if nf else None, zorder=2)
        if a:
            ax.barh(i, a, color=cA_col, edgecolor=cA_col, **common)
            ax.text(a + 0.08, i, str(a), va="center", ha="left", fontsize=7.5, color="#333")
        if b:
            ax.barh(i, -b, color=cB_col, edgecolor=cB_col, **common)
            ax.text(-b - 0.08, i, str(b), va="center", ha="right", fontsize=7.5, color="#333")
        name = r.get("gene") or r.get("label") or str(idx)
        names.append(name if isinstance(name, str) and name else str(idx))
        ticks_nf.append(nf)

    ax.axvline(0, color="#444444", linewidth=1.0, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    for tick, nf in zip(ax.get_yticklabels(), ticks_nf):
        if nf:
            tick.set_color("#7a7a7a")              # near-floor names greyed
    span = max(nA, nB)
    ax.set_xlim(-(nB + 0.8), nA + 0.8)
    xt = list(range(-nB, nA + 1))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(abs(t)) for t in xt])
    ax.set_xlabel(f"samples with the protein detected     (\u2190 {gb}      |      {ga} \u2192)")
    ax.set_title("Presence/absence (on/off) candidates")
    _style_axes(ax)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=cA_col, markersize=9, label=f"{ga}-enriched"),
        Line2D([0], [0], marker="s", linestyle="", color=cB_col, markersize=9, label=f"{gb}-enriched"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="white",
               markeredgecolor="#888888", markersize=9, label="near floor (faded/hatched)"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative DE: limma's moderated t-test (empirical-Bayes variance shrinkage)
#
# A faithful port of the limma eBayes core (Smyth 2004): per-protein linear model,
# residual variances shrunk toward a prior fitted across all proteins (fitFDist +
# trigammaInverse), then a moderated t with augmented degrees of freedom. Runs on a
# COMPLETE matrix (no NaN) -- the complete-case or an imputed track. Validate
# against R limma with tolerance; the variance machinery is unit-tested by
# recovering a known prior from simulated variances.
# ─────────────────────────────────────────────────────────────────────────────


def _trigamma_inverse(x):
    """Invert the trigamma function (limma's trigammaInverse), elementwise."""
    from scipy.special import polygamma

    x = np.asarray(x, dtype=float)
    y = np.where(x > 1e7, 1.0 / np.sqrt(np.maximum(x, 1e-300)),
                 np.where(x < 1e-6, 1.0 / x, 0.5 + 1.0 / x))
    y = y.astype(float)
    for _ in range(50):
        tri = polygamma(1, y)
        with np.errstate(invalid="ignore", divide="ignore"):
            dif = tri * (1.0 - tri / x) / polygamma(2, y)
        y = y + dif
        if np.nanmax(np.abs(-dif / y)) < 1e-8:
            break
    return y


def _fit_fdist(s2, df1):
    """Fit a scaled F to the variances (limma fitFDist): returns (s2_0, df0)."""
    from scipy.special import digamma, polygamma

    s2 = np.asarray(s2, dtype=float)
    ok = np.isfinite(s2) & (s2 > 0)
    x = s2[ok]
    n = x.size
    if n <= 1:
        return float(np.nanmean(s2)) if n else np.nan, 0.0
    z = np.log(x)
    e = z - digamma(df1 / 2.0) + np.log(df1 / 2.0)
    emean = float(np.mean(e))
    evar = float(np.sum((e - emean) ** 2) / (n - 1)) - float(polygamma(1, df1 / 2.0))
    if evar > 0:
        df0 = 2.0 * float(_trigamma_inverse(np.array([evar]))[0])
        s2_0 = float(np.exp(emean + digamma(df0 / 2.0) - np.log(df0 / 2.0)))
    else:
        df0 = np.inf
        s2_0 = float(np.exp(emean))
    return s2_0, df0


def _squeeze_var(s2, df_resid):
    """Posterior variances by EB shrinkage toward the fitted prior (limma squeezeVar)."""
    s2 = np.asarray(s2, dtype=float)
    s2_0, df0 = _fit_fdist(s2, df_resid)
    if np.isinf(df0):
        s2_post = np.full_like(s2, s2_0)
    else:
        s2_post = (df_resid * s2 + df0 * s2_0) / (df_resid + df0)
    return s2_post, s2_0, df0


def _design_matrix(samples, groups, ref_group, covariates=None):
    g = {s: groups[s] for s in samples}
    levels = sorted(set(g.values()))

    if ref_group not in levels:
        raise ValueError(
            f"ref_group {ref_group!r} not among groups {levels}."
        )
    if len(levels) != 2:
        raise ValueError(
            f"limma_de supports a 2-level group factor; got {levels}."
        )

    non_ref = next(l for l in levels if l != ref_group)

    cols = [np.ones(len(samples))]
    names = ["(Intercept)", f"{non_ref}_vs_{ref_group}"]

    cols.append(
        np.array([1.0 if g[s] == non_ref else 0.0 for s in samples])
    )
    coef_index = 1

    if covariates:
        cov = {s: str(covariates[s]) for s in samples}
        for lv in sorted(set(cov.values()))[1:]:
            cols.append(
                np.array([1.0 if cov[s] == lv else 0.0 for s in samples])
            )
            names.append(f"cov_{lv}")

    design = np.column_stack(cols)

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "design matrix is not full rank; group and covariate effects "
            "cannot be estimated independently"
        )

    return design, coef_index, names[1]

def limma_de(matrix: pd.DataFrame, groups, *, ref_group: str, covariates=None) -> pd.DataFrame:
    """limma moderated t-test for a two-group contrast on a complete matrix.

    Parameters
    ----------
    matrix : DataFrame
        proteins x samples, **no missing values** (a complete-case or imputed
        matrix). Values are normalized log2 intensities.
    groups : mapping {sample -> group}
        Two-level group factor (e.g. genotype).
    ref_group : str
        Reference level; a positive log2FC means higher in the *other* group.
    covariates : mapping {sample -> value}, optional
        One categorical covariate (e.g. batch) added to the design as
        ``~ covariate + group``. For a single-batch dataset leave as ``None``.

    Returns
    -------
    DataFrame indexed by protein id with ``log2FC``, ``AveExpr``, ``t`` (moderated),
    ``p``, ``padj`` (BH), and the moderated ``df_total``. ``.attrs`` records the
    prior variance/df and the contrast name.
    """
    from scipy import stats

    samples = list(matrix.columns)
    X, ci, coef_name = _design_matrix(samples, groups, ref_group, covariates)

    Y = matrix.to_numpy(dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError(
            "limma_de requires a complete matrix with finite values "
            "(no NaN or infinity); impute or use the complete-case matrix first."
        )

    n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y.T
    resid = Y.T - X @ beta

    df_resid = n - np.linalg.matrix_rank(X)
    if df_resid <= 0:
        raise ValueError(
            "limma_de requires positive residual degrees of freedom; "
            "the design has too few samples for the fitted model"
        )

    s2 = np.sum(resid ** 2, axis=0) / df_resid
    stdev_unscaled = float(np.sqrt(XtX_inv[ci, ci]))
    coef = beta[ci, :]

    s2_post, s2_0, df0 = _squeeze_var(s2, df_resid)
    se = stdev_unscaled * np.sqrt(s2_post)

    with np.errstate(invalid="ignore", divide="ignore"):
        t = coef / se

    df_total = df_resid + df0

    if np.isinf(df_total):
        p = 2.0 * stats.norm.sf(np.abs(t))
    else:
        p = 2.0 * stats.t.sf(np.abs(t), df_total)

    out = pd.DataFrame(index=matrix.index)
    out.index.name = matrix.index.name or "id"
    out["log2FC"] = coef
    out["AveExpr"] = Y.mean(axis=1)
    out["t"] = t
    out["p"] = p
    out["padj"] = _bh(p)
    out["df_total"] = df_total

    out.attrs["method"] = "limma"
    out.attrs["contrast"] = coef_name
    out.attrs["var_prior"] = s2_0
    out.attrs["df_prior"] = df0
    out.attrs["df_residual"] = df_resid

    return out

def complete_case_matrix(qd: "QuantData", diag: pd.DataFrame = None, *,
                         normalization: str = "median", only_ok: bool = True) -> pd.DataFrame:
    """The no-imputation DE matrix: normalized log2, rows with zero missing.

    If ``diag`` is given and ``only_ok``, contaminant rows are dropped first (so the
    complete-case set matches the proteins the DE family is computed over).
    """
    M = log2_median_normalize(qd.lfq, method=normalization)
    if only_ok and diag is not None and "contaminant_class" in diag.columns:
        ok = pd.Series(diag["contaminant_class"]).reindex(M.index).fillna("ok").to_numpy() == "ok"
        M = M.loc[ok]
    return M.loc[M.notna().all(axis=1)]


# ─────────────────────────────────────────────────────────────────────────────
# Non-parametric / unmoderated DE: Welch t and Wilcoxon (Mann-Whitney) per protein.
# Both take the same complete matrix + group factor as limma and return the same
# schema (log2FC, AveExpr, statistic, p, padj). They ignore covariates by design.
# ─────────────────────────────────────────────────────────────────────────────


def _two_group_cols(matrix, groups, ref_group):
    samples = list(matrix.columns)
    g = {s: groups[s] for s in samples}
    levels = sorted(set(g.values()))
    if ref_group not in levels:
        raise ValueError(f"ref_group {ref_group!r} not among groups {levels}.")
    if len(levels) != 2:
        raise ValueError(f"two-group test needs a 2-level factor; got {levels}.")
    non_ref = next(l for l in levels if l != ref_group)
    a_cols = [s for s in samples if g[s] == non_ref]   # numerator (positive log2FC)
    b_cols = [s for s in samples if g[s] == ref_group]
    return non_ref, a_cols, b_cols


def welch_de(matrix: pd.DataFrame, groups, *, ref_group: str) -> pd.DataFrame:
    """Welch's t-test (unequal variance) per protein on a complete matrix."""
    from scipy import stats

    Y = matrix.to_numpy(dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError(
            "welch_de requires a complete matrix with finite values "
            "(no NaN or infinity)."
        )

    non_ref, a_cols, b_cols = _two_group_cols(matrix, groups, ref_group)

    if len(a_cols) < 2 or len(b_cols) < 2:
        raise ValueError(
            "welch_de requires at least 2 samples in each group."
        )

    Acols = matrix[a_cols].to_numpy(dtype=float)
    Bcols = matrix[b_cols].to_numpy(dtype=float)

    r = stats.ttest_ind(
        Acols,
        Bcols,
        axis=1,
        equal_var=False,
    )

    out = pd.DataFrame(index=matrix.index)
    out.index.name = matrix.index.name or "id"
    out["log2FC"] = Acols.mean(axis=1) - Bcols.mean(axis=1)
    out["AveExpr"] = Y.mean(axis=1)
    out["t"] = r.statistic
    out["p"] = r.pvalue
    out["padj"] = _bh(np.asarray(r.pvalue, dtype=float))

    out.attrs["method"] = "welch"
    out.attrs["contrast"] = f"{non_ref}_vs_{ref_group}"

    return out

def wilcoxon_de(matrix: pd.DataFrame, groups, *, ref_group: str) -> pd.DataFrame:
    """Mann-Whitney U (rank-sum) per protein on a complete matrix.

    Reports the mean-difference log2FC as the effect size (shared x-axis with the
    other methods). Note: with small n per group the U-test p-value is discrete and
    floored (e.g. ~0.0079 two-sided at 5 vs 5), so it rarely survives multiple-
    testing correction on thousands of proteins -- read it as a robustness check.
    """
    from scipy import stats

    Y = matrix.to_numpy(dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError(
            "wilcoxon_de requires a complete matrix with finite values "
            "(no NaN or infinity)."
        )

    non_ref, a_cols, b_cols = _two_group_cols(matrix, groups, ref_group)

    Acols = matrix[a_cols].to_numpy(dtype=float)
    Bcols = matrix[b_cols].to_numpy(dtype=float)

    r = stats.mannwhitneyu(
        Acols,
        Bcols,
        axis=1,
        alternative="two-sided",
    )

    out = pd.DataFrame(index=matrix.index)
    out.index.name = matrix.index.name or "id"
    out["log2FC"] = Acols.mean(axis=1) - Bcols.mean(axis=1)
    out["AveExpr"] = Y.mean(axis=1)
    out["U"] = r.statistic
    out["p"] = r.pvalue
    out["padj"] = _bh(np.asarray(r.pvalue, dtype=float))

    out.attrs["method"] = "wilcoxon"
    out.attrs["contrast"] = f"{non_ref}_vs_{ref_group}"

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher: run a set of DE methods across the complete-case and imputed tracks.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# DEqMS: limma fit + count-dependent variance moderation (Zhu et al. 2020).
#
# A faithful port of DEqMS::spectraCounteBayes. Same linear fit as limma, but the
# variance prior is a function of the per-protein peptide/PSM count: fit a trend of
# log(residual variance) vs log2(count), then estimate the prior df by Sartor's
# IBMT moment-matching and form a per-protein posterior variance from the trend.
# A protein quantified from many peptides gets a tighter, more confident variance.
# Runs on a COMPLETE matrix (complete-case or an imputed track), like limma.
# ─────────────────────────────────────────────────────────────────────────────


def _loess_predict(x, y, *, span: float = 0.75, degree: int = 2):
    """Local weighted polynomial regression (R loess: degree-2, tricube weights,
    nearest-neighbour bandwidth = span fraction of points). Fitted values at ``x``.

    Peptide counts are integers, so ``x = log2(count)`` takes few distinct values;
    we fit once per distinct x and map back -- exact and fast (no interpolation).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    q = int(min(max(int(np.ceil(span * n)), degree + 1), n))
    xu = np.unique(x)
    pred_u = np.empty(xu.size)
    for k, x0 in enumerate(xu):
        d = np.abs(x - x0)
        h = np.partition(d, q - 1)[q - 1]
        h = h if h > 0 else (d.max() if d.max() > 0 else 1.0)
        u = np.clip(d / h, 0.0, 1.0)
        w = (1.0 - u ** 3) ** 3
        nz = w > 0
        Xp = np.vander(x[nz] - x0, degree + 1, increasing=True)
        W = w[nz]
        XtWX = Xp.T @ (W[:, None] * Xp)
        XtWy = Xp.T @ (W * y[nz])
        try:
            beta = np.linalg.solve(XtWX, XtWy)
            pred_u[k] = beta[0]                      # intercept = value at x0
        except np.linalg.LinAlgError:
            pred_u[k] = float(np.average(y[nz], weights=W))
    return pred_u[np.searchsorted(xu, x)]


def _deqms_prior_df(mean_myfct: float, numgenes: int):
    """Grid-search the prior df d0 so trigamma(d0/2) matches the mean trend-residual
    (DEqMS spectraCounteBayes: step d0 by 0.1, stop once the objective turns up)."""
    from scipy.special import polygamma

    priordf = []
    for i in range(1, numgenes * 10 + 1):
        d0i = i / 10.0
        priordf.append(abs(mean_myfct - float(polygamma(1, d0i / 2.0))))
        if i > 2 and priordf[i - 3] < priordf[i - 2]:
            break
    j = int(np.argmin(priordf))
    return (j + 1) / 10.0

def deqms_de(matrix: pd.DataFrame, groups, counts, *, ref_group: str,
             covariates=None, fit_method: str = "loess") -> pd.DataFrame:
    """DEqMS moderated t-test for a two-group contrast on a complete matrix.

    Same as :func:`limma_de` plus a count-dependent variance prior.

    Parameters
    ----------
    matrix : DataFrame
        proteins x samples, no missing values (complete-case or imputed).
    groups : mapping {sample -> group}; ``ref_group`` sets the contrast direction.
    counts : Series/array
        Per-protein peptide (or PSM) count, aligned to (or indexable by)
        ``matrix.index`` -- e.g. ``qd.proteins['razor_unique_peptides']`` or the
        per-sample minimum. A count of 0 gets a pseudocount of 1 (DEqMS).
    covariates : mapping {sample -> value}, optional -- one categorical covariate.
    fit_method : {"loess", "spline", "nls"} -- the variance-vs-count trend.

    Returns
    -------
    DataFrame indexed by protein id with ``log2FC``, ``AveExpr``, ``t`` (count-
    moderated), ``p``, ``padj`` (BH), ``count`` and ``df_total``. ``.attrs`` records
    the prior df, contrast and fit method.
    """
    from scipy import stats
    from scipy.special import digamma, polygamma

    samples = list(matrix.columns)
    X, ci, coef_name = _design_matrix(samples, groups, ref_group, covariates)

    Y = matrix.to_numpy(dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError(
            "deqms_de requires a complete matrix with finite values "
            "(no NaN or infinity); impute or use the complete-case matrix first."
        )

    cnt = pd.Series(counts).reindex(matrix.index).to_numpy(dtype=float)

    if not np.isfinite(cnt).all():
        raise ValueError(
            "counts must contain finite values for every protein in the matrix "
            "(no NaN or infinity)."
        )

    if np.any(cnt < 0):
        raise ValueError(
            "counts must be non-negative peptide/PSM counts."
        )

    if cnt.min() == 0:
        cnt = cnt + 1.0  # DEqMS pseudocount

    n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y.T
    resid = Y.T - X @ beta

    df = n - np.linalg.matrix_rank(X)
    if df <= 0:
        raise ValueError(
            "deqms_de requires positive residual degrees of freedom; "
            "the design has too few samples for the fitted model"
        )

    sigma2 = np.sum(resid ** 2, axis=0) / df
    stdev_unscaled = float(np.sqrt(XtX_inv[ci, ci]))
    coef = beta[ci, :]

    # Guard exact-zero variance (identical rows) before the log.
    floor = (
        np.nanmin(sigma2[sigma2 > 0]) * 1e-6
        if np.any(sigma2 > 0)
        else 1e-12
    )
    logVAR = np.log(np.maximum(sigma2, floor))

    if fit_method == "loess":
        y_pred = _loess_predict(
            np.log2(cnt),
            logVAR,
            span=0.75,
            degree=2,
        )

    elif fit_method == "spline":
        from scipy.interpolate import UnivariateSpline

        x = np.log2(cnt)
        o = np.argsort(x)
        y_pred = np.empty_like(logVAR)
        y_pred[o] = UnivariateSpline(
            x[o],
            logVAR[o],
            k=3,
        )(x[o])

    elif fit_method == "nls":
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            lambda xx, a, b: a + b / xx,
            cnt,
            sigma2,
            p0=[0.1, 0.05],
            maxfev=10000,
        )
        y_pred = np.log(
            np.maximum(popt[0] + popt[1] / cnt, floor)
        )

    else:
        raise ValueError(
            f"fit_method must be 'loess' | 'spline' | 'nls', "
            f"got {fit_method!r}."
        )

    # Sartor IBMT: prior df from the trend residuals,
    # count-dependent prior variance.
    eg = logVAR - digamma(df / 2.0) + np.log(df / 2.0)
    egpred = y_pred - digamma(df / 2.0) + np.log(df / 2.0)

    myfct = (
        (eg - egpred) ** 2
        - float(polygamma(1, df / 2.0))
    )
    mean_myfct = float(np.nanmean(myfct))

    d0 = _deqms_prior_df(
        mean_myfct,
        int(np.isfinite(logVAR).sum()),
    )

    s2_0 = np.exp(
        egpred
        + digamma(d0 / 2.0)
        - np.log(d0 / 2.0)
    )

    post_var = (d0 * s2_0 + df * sigma2) / (d0 + df)
    post_df = df + d0

    with np.errstate(invalid="ignore", divide="ignore"):
        t = coef / (
            stdev_unscaled * np.sqrt(post_var)
        )

    p = 2.0 * stats.t.sf(np.abs(t), post_df)

    out = pd.DataFrame(index=matrix.index)
    out.index.name = matrix.index.name or "id"
    out["log2FC"] = coef
    out["AveExpr"] = Y.mean(axis=1)
    out["t"] = t
    out["p"] = p
    out["padj"] = _bh(p)
    out["count"] = cnt
    out["df_total"] = post_df

    out.attrs["method"] = "deqms"
    out.attrs["contrast"] = coef_name
    out.attrs["df_prior"] = d0
    out.attrs["df_residual"] = df
    out.attrs["fit_method"] = fit_method

    return out

_DE_METHODS = ("limma", "welch", "wilcoxon")              # default suite
_DE_METHODS_ALL = ("limma", "deqms", "welch", "wilcoxon")  # deqms is opt-in (needs counts)
_DE_TRACKS = ("complete_case", "imputed_mechanism", "imputed_minprob")


def differential_abundance(
    qd: "QuantData",
    diag: pd.DataFrame,
    *,
    methods=_DE_METHODS,
    tracks=_DE_TRACKS,
    ref_group: str,
    covariates=None,
    normalization: str = "median",
    knn_k: int = 10,
    qrilc_tune_sigma: float = 1.0,
    minprob_width: float = 0.3,
    minprob_downshift: float = 1.8,
    counts=None,
    deqms_fit_method: str = "loess",
    seed: int = 1,
) -> pd.DataFrame:
    """Run several DE methods across the DE tracks and return one tidy table.

    Tracks: ``complete_case`` (no imputation), ``imputed_mechanism`` (QRILC + kNN),
    ``imputed_minprob`` (downshift). Methods: ``limma`` (moderated t), ``welch``,
    ``wilcoxon``. Within a track every method shares the same mean-difference
    ``log2FC``; only the p-values differ. The imputed tracks test more proteins
    than complete-case (the recovered MAR/MNAR/high-confidence rows).

    ``covariates`` (one categorical, e.g. batch) is applied to **limma only**;
    welch and wilcoxon ignore it by construction.

    Returns a long DataFrame with one row per (protein, track, method) and columns
    ``id, label, track, method, log2FC, AveExpr, p, padj``. ``.attrs`` records the
    methods, tracks, ref_group and contrast name.
    """
    methods = tuple(methods)
    tracks = tuple(tracks)
    bad_m = set(methods) - set(_DE_METHODS_ALL)
    bad_t = set(tracks) - set(_DE_TRACKS)
    if bad_m:
        raise ValueError(f"unknown method(s) {sorted(bad_m)}; choose from {_DE_METHODS_ALL}.")
    if bad_t:
        raise ValueError(f"unknown track(s) {sorted(bad_t)}; choose from {_DE_TRACKS}.")

    from ._impute import impute_matrix

    mats = {}
    if "complete_case" in tracks:
        mats["complete_case"] = complete_case_matrix(qd, diag, normalization=normalization)
    if "imputed_mechanism" in tracks:
        mats["imputed_mechanism"] = impute_matrix(
            qd, diag, method="mechanism", normalization=normalization,
            knn_k=knn_k, qrilc_tune_sigma=qrilc_tune_sigma, seed=seed)[0]
    if "imputed_minprob" in tracks:
        mats["imputed_minprob"] = impute_matrix(
            qd, diag, method="minprob", normalization=normalization,
            minprob_width=minprob_width, minprob_downshift=minprob_downshift, seed=seed)[0]

    label = (qd.proteins["label"] if "label" in qd.proteins.columns
             else qd.proteins.get("gene", pd.Series(qd.proteins.index, index=qd.proteins.index)))
    if counts is None and "deqms" in methods:
        if "razor_unique_peptides" not in qd.proteins.columns:
            raise ValueError("deqms needs a per-protein peptide/PSM count, but "
                             "qd.proteins has no 'razor_unique_peptides'; pass counts= explicitly.")
        counts = qd.proteins["razor_unique_peptides"]
    runners = {"limma": lambda M: limma_de(M, qd.groups, ref_group=ref_group, covariates=covariates),
               "deqms": lambda M: deqms_de(M, qd.groups, counts, ref_group=ref_group,
                                           covariates=covariates, fit_method=deqms_fit_method),
               "welch": lambda M: welch_de(M, qd.groups, ref_group=ref_group),
               "wilcoxon": lambda M: wilcoxon_de(M, qd.groups, ref_group=ref_group)}

    parts, contrast = [], None
    for track in tracks:
        M = mats[track]
        for method in methods:
            r = runners[method](M)
            contrast = r.attrs.get("contrast", contrast)
            parts.append(pd.DataFrame({
                "id": r.index.to_numpy(),
                "label": pd.Series(label).reindex(r.index).to_numpy(),
                "track": track,
                "method": method,
                "log2FC": r["log2FC"].to_numpy(),
                "AveExpr": r["AveExpr"].to_numpy(),
                "p": r["p"].to_numpy(),
                "padj": r["padj"].to_numpy(),
            }))
    long = pd.concat(parts, ignore_index=True)
    long.attrs["methods"] = methods
    long.attrs["tracks"] = tracks
    long.attrs["ref_group"] = ref_group
    long.attrs["contrast"] = contrast
    return long


def de_concordance(long: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Per-track concordance: n significant per method, and n significant in *all*
    methods of that track (BH-padj < ``alpha``)."""
    methods = list(long.attrs.get("methods", sorted(long["method"].unique())))
    tracks = list(long.attrs.get("tracks", sorted(long["track"].unique())))
    rows = []
    for track in tracks:
        sub = long[long["track"] == track]
        row = {m: int((sub.loc[sub["method"] == m, "padj"] < alpha).sum()) for m in methods}
        wide = sub.pivot_table(index="id", columns="method", values="padj")
        present = [m for m in methods if m in wide.columns]
        row["all_methods"] = int((wide[present] < alpha).all(axis=1).sum()) if present else 0
        row["n_proteins"] = int(sub["id"].nunique())
        rows.append(pd.Series(row, name=track))
    return pd.DataFrame(rows)