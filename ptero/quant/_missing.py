"""
ptero.quant._missing -- missing-value analysis (A3 of the multi-sample arc).

The cross-tab of LFQ state x MaxQuant identification type is not noise -- it is
the data model. Every protein x sample cell is one of six states, and the three
"tiers" of missingness roll straight up from them:

    quantified (MS/MS)            LFQ present, MS/MS-backed        -- gold
    quantified (MBR)              LFQ present, match-transferred   -- borrowed quantity
    detected, no LFQ (MS/MS)      Int>0/LFQ=0, MS/MS here          -.
    detected, no LFQ (MBR)        Int>0/LFQ=0, transferred          | tier 1: recoverable
    ID only, no intensity (MBR)   both 0, MBR ID, no signal        -- tier 2
    not observed                  both 0, never seen               -- tier 3: MNAR candidate

This module derives those states (:func:`cell_state`), summarises missingness per
protein (:func:`missingness_summary`), surfaces group-exclusive on/off CANDIDATES
ranked by how clean the "off" side is (:func:`group_exclusive` -- evidence never
delivers a verdict, it surfaces candidates to investigate), and plots the
left-censoring check (:func:`plot_missingness_vs_abundance`) and the per-sample
state composition (:func:`plot_cell_state_composition`).

Plotting reuses the house style from ``ptero.quant._plots`` (Figure object API,
no pyplot), so the look stays consistent across the subpackage.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ._plots import _ordered_samples, _positions, _style_axes

if TYPE_CHECKING:
    from ._quant import QuantData

# ── the six states (+ a guard bucket) ────────────────────────────────────────
QUANTIFIED_MSMS = "quantified (MS/MS)"
QUANTIFIED_MBR = "quantified (MBR)"
DETECTED_MSMS = "detected, no LFQ (MS/MS)"
DETECTED_MBR = "detected, no LFQ (MBR)"
MATCHED_EMPTY = "ID only, no intensity (MBR)"
ABSENT = "not observed"
OTHER = "other"          # should be empty on real MaxQuant output; a guard

STATE_ORDER = [QUANTIFIED_MSMS, QUANTIFIED_MBR, DETECTED_MSMS, DETECTED_MBR,
               MATCHED_EMPTY, ABSENT, OTHER]
STATE_COLORS = {
    QUANTIFIED_MSMS: "#2C7FB8",   # strong blue   -- LFQ, MS/MS
    QUANTIFIED_MBR:  "#7FB9E6",   # light blue    -- LFQ, transferred
    DETECTED_MSMS:   "#E0982E",   # amber         -- detected, no LFQ, MS/MS
    DETECTED_MBR:    "#F3D08A",   # light amber   -- detected, no LFQ, transferred
    MATCHED_EMPTY:   "#B7A6CE",   # muted purple  -- ID only
    ABSENT:          "#DCDCDC",   # light grey    -- not observed
    OTHER:           "#9E9E9E",
}
# the states that make an "off" group a CLEAN absence (no signal at all)
_CLEAN_OFF = (ABSENT, MATCHED_EMPTY)


def cell_state(qd: "QuantData") -> pd.DataFrame:
    """Per protein x sample state (one of the six above), as a string DataFrame.

    Built from LFQ presence, raw-intensity presence, and identification type:
    ``quantified`` (LFQ present) / ``detected, no LFQ`` (Int>0, LFQ=0) /
    ``not observed`` (both 0), each split by ``MS/MS`` vs ``MBR`` provenance,
    plus ``ID only`` (MBR identification but zero intensity).
    """
    lfq = qd.lfq
    inten = qd.intensity.reindex(columns=lfq.columns)
    idt = qd.id_type.reindex(columns=lfq.columns)

    msms = idt == "By MS/MS"
    mbr = idt == "By matching"
    present = lfq.notna()
    detected = lfq.isna() & inten.notna()
    absent_q = lfq.isna() & inten.isna()

    state = pd.DataFrame(OTHER, index=lfq.index, columns=lfq.columns)
    state = state.mask(present & msms, QUANTIFIED_MSMS)
    state = state.mask(present & mbr, QUANTIFIED_MBR)
    state = state.mask(detected & msms, DETECTED_MSMS)
    state = state.mask(detected & mbr, DETECTED_MBR)
    state = state.mask(absent_q & mbr, MATCHED_EMPTY)
    state = state.mask(absent_q & ~mbr & ~msms, ABSENT)
    return state


def state_counts(qd: "QuantData") -> pd.Series:
    """Total protein x sample cells in each state, in canonical order."""
    s = cell_state(qd).melt(value_name="state")["state"]
    counts = s.value_counts()
    order = [x for x in STATE_ORDER if x in counts.index]
    return counts.reindex(order)


def missingness_summary(qd: "QuantData") -> pd.DataFrame:
    """Per-protein LFQ presence/absence summary (indexed by id).

    Columns: ``label``, ``gene`` (if known), ``n_present`` / ``n_missing`` over
    all samples, and -- when grouped -- ``present_<group>`` / ``size_<group>`` for
    each group (the basis for the on/off table).
    """
    present = qd.lfq.notna()
    n = qd.lfq.shape[1]
    out = pd.DataFrame(index=qd.lfq.index)
    out.index.name = "id"
    if "label" in qd.proteins.columns:
        out["label"] = qd.proteins["label"]
    if "gene" in qd.proteins.columns:
        out["gene"] = qd.proteins["gene"]
    out["n_present"] = present.sum(axis=1).astype(int)
    out["n_missing"] = (n - out["n_present"]).astype(int)
    if qd.grouped:
        for g, members in qd.group_members().items():
            out[f"present_{g}"] = present[members].sum(axis=1).astype(int)
            out[f"size_{g}"] = len(members)
    return out


def group_exclusive(qd: "QuantData") -> pd.DataFrame:
    """Group-exclusive on/off CANDIDATES: LFQ present in all of one group, absent
    in every other group.

    Not a verdict -- a ranked list to investigate. ``off_clean`` is True when the
    off side is entirely ``not observed`` / ``ID only`` (a strong on/off
    candidate); False when the off side carries ``detected, no LFQ`` signal (weak
    -- the protein was seen there, LFQ just didn't form). Sorted strongest first.
    """
    members = qd.group_members()
    gnames = list(members)
    if not qd.grouped or len(gnames) < 2:
        raise ValueError("group_exclusive needs >= 2 groups; pass groups= to load_quant.")

    present = qd.lfq.notna()
    pres = {g: present[members[g]].sum(axis=1) for g in gnames}
    size = {g: len(members[g]) for g in gnames}
    state = cell_state(qd)
    have_label = "label" in qd.proteins.columns
    have_gene = "gene" in qd.proteins.columns

    rows = []
    for pid in qd.lfq.index:
        on = [g for g in gnames if size[g] > 0 and pres[g][pid] == size[g]]
        off = [g for g in gnames if pres[g][pid] == 0]
        if len(on) != 1:
            continue
        off_gs = [g for g in off if g != on[0]]
        if not off_gs:
            continue
        off_samples = [s for g in off_gs for s in members[g]]
        off_states = state.loc[pid, off_samples]
        rows.append({
            "id": pid,
            "label": qd.proteins.loc[pid, "label"] if have_label else pid,
            "gene": qd.proteins.loc[pid, "gene"] if have_gene else None,
            "on_group": on[0],
            "off_groups": ",".join(off_gs),
            "off_dominant_state": off_states.value_counts().idxmax(),
            "off_clean": bool(off_states.isin(_CLEAN_OFF).all()),
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["off_clean", "on_group", "label"],
                              ascending=[False, True, True]).reset_index(drop=True)
    return res


# ── plots (Figure object API; house style from ._plots) ──────────────────────

def plot_missingness_vs_abundance(qd: "QuantData", *, log2: bool = True,
                                  figsize=(8.5, 4.8)) -> Figure:
    """Mean LFQ (where present) vs number of samples missing -- the MNAR check.

    If higher-missingness proteins skew to lower abundance, missing values are
    left-censored (MNAR), which justifies a left-shifted imputation later. A flat
    trend means missingness is closer to random.
    """
    present = qd.lfq.notna()
    n = qd.lfq.shape[1]
    n_missing = (n - present.sum(axis=1)).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ab = (np.log2(qd.lfq) if log2 else qd.lfq).mean(axis=1, skipna=True)

    mask = present.sum(axis=1) > 0          # need >=1 present to have an abundance
    nm, ab = n_missing[mask], mean_ab[mask]
    xs_k = sorted(int(k) for k in nm.unique())
    data = [ab[nm == k].dropna().to_numpy() for k in xs_k]
    pos = np.arange(len(xs_k))

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    bp = ax.boxplot(
        data, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="#888888", linewidth=0.9),
        capprops=dict(color="#888888", linewidth=0.9),
        boxprops=dict(linewidth=0.8, edgecolor="#888888"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#7FB9E6")
        patch.set_alpha(0.65)
    meds = [float(np.median(d)) if len(d) else np.nan for d in data]
    ax.plot(pos, meds, color="#C44E52", linewidth=1.6, marker="o", markersize=3.5,
            zorder=4, label="median trend")
    for x, d in zip(pos, data):                 # n proteins per bin
        ax.text(x, ax.get_ylim()[0], f"n={len(d)}", ha="center", va="bottom",
                fontsize=6.5, color="#999999")

    ax.set_xticks(pos)
    ax.set_xticklabels(xs_k)
    ax.set_xlabel(f"number of samples with LFQ missing (of {n})")
    ax.set_ylabel("mean log2 LFQ (where present)" if log2 else "mean LFQ (where present)")
    if not log2:
        ax.set_yscale("log")
    ax.set_title("Missingness vs abundance", pad=22)
    ax.text(0.5, 1.012,
            "lower abundance at higher missingness => left-censored (MNAR)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#777777")
    _style_axes(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    return fig


def plot_cell_state_composition(qd: "QuantData", *, figsize=(9.5, 5.2)) -> Figure:
    """Per-sample stacked composition of the six cell states."""
    state = cell_state(qd)
    samples = _ordered_samples(qd)
    xs = _positions(qd, samples)
    present_states = [s for s in STATE_ORDER if (state == s).to_numpy().any()]

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    bottom = np.zeros(len(samples))
    for st in present_states:
        vals = np.array([int((state[s] == st).sum()) for s in samples], dtype=float)
        ax.bar(xs, vals, bottom=bottom, width=0.72, color=STATE_COLORS[st],
               label=st, zorder=2)
        bottom += vals

    ax.set_xticks(xs)
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylabel("protein groups")
    ax.set_title("Per-sample cell-state composition")
    _style_axes(ax)
    # legend outside right, top-of-stack first
    handles = [Line2D([0], [0], marker="s", linestyle="", color=STATE_COLORS[s], label=s)
               for s in reversed(present_states)]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, title="cell state")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Missingness MECHANISM classifier  (faithful port of the reference 03c logic)
#
# Per-protein: WHY are values missing, and what may be done about them. A point
# imputation is only valid when the mechanism is identifiable; non-random /
# condition-dependent missingness is routed to a presence/absence (Fisher) test
# instead of being imputed, because the missing intensities are unidentifiable
# (low biology, dropped-but-normal, or a proteoform shift).
#
# Cascade (ORDER MATTERS), for a two-group contrast (e.g. genotype A vs B):
#   1. 0 in BOTH groups                              -> undetected          (drop)
#   2. 0 in ONE group (>=1 in the other)             -> genotype_restricted (Fisher)
#   3. det >= cd_present_min in one AND <= cd_absent_max in the other (both >0)
#                                                     -> condition_dependent (Fisher)
#        (+ near_floor: missing-group obs mean <= LOD50 + cd_floor_margin)
#   4. pct_missing <= hi_conf_max                    -> high_confidence     (kNN)
#   5. det_rate < min_detect_overall (symmetric)     -> too_sparse          (not imputed)
#   6. symmetric & min_obs <= LOD50                  -> low_abundance_MNAR  (QRILC)
#   7. otherwise (symmetric, above floor)            -> MAR_scattered       (kNN)
#
# LOD50 is learned from the data: a binomial logistic GLM of detection count vs
# mean observed intensity (detection ~ intensity); LOD50 = -b0/b1 when the slope
# is positive, else a fallback intensity quantile -- matching the reference glm().
# ─────────────────────────────────────────────────────────────────────────────

UNDETECTED = "undetected"
GENOTYPE_RESTRICTED = "genotype_restricted"
CONDITION_DEPENDENT = "condition_dependent"
HIGH_CONFIDENCE = "high_confidence"
TOO_SPARSE = "too_sparse"
LOW_ABUNDANCE_MNAR = "low_abundance_MNAR"
MAR_SCATTERED = "MAR_scattered"

MECHANISM_ORDER = [HIGH_CONFIDENCE, MAR_SCATTERED, LOW_ABUNDANCE_MNAR,
                   CONDITION_DEPENDENT, GENOTYPE_RESTRICTED, TOO_SPARSE, UNDETECTED]

# routing per mechanism (what the next arc will do; this module only labels)
_IMPUTE_METHOD = {
    UNDETECTED: "drop",
    GENOTYPE_RESTRICTED: "Fisher (presence/absence)",
    CONDITION_DEPENDENT: "Fisher (presence/absence)",
    HIGH_CONFIDENCE: "kNN (within group)",
    TOO_SPARSE: "not imputed (majority missing)",
    LOW_ABUNDANCE_MNAR: "QRILC",
    MAR_SCATTERED: "kNN (within group)",
}
DIAGNOSE_COLORS = {
    HIGH_CONFIDENCE: "#9ca3af", MAR_SCATTERED: "#d97706", LOW_ABUNDANCE_MNAR: "#1f4e79",
    CONDITION_DEPENDENT: "#7c3aed", GENOTYPE_RESTRICTED: "#b91c1c",
    TOO_SPARSE: "#f4d03f", UNDETECTED: "#dddddd",
}


def log2_median_normalize(lfq: pd.DataFrame, *, method: str = "median") -> pd.DataFrame:
    """log2-transform an LFQ matrix (``NaN`` == missing) and normalize observed values.

    Mirrors the reference ``normalize_log``:

    * ``"median"`` (default) -- shift each sample so its observed median equals
      the global median-of-medians (NA-aware; only observed values move).
    * ``"none"`` -- log2 only.
    * ``"quantile"`` -- not yet ported (raises); the reference analysis used
      median.

    The classifier and downstream DE operate on this normalized log2 matrix, so
    LOD50 and the abundance axis are on a comparable per-sample scale.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.log2(lfq)
    if method == "none":
        return L
    if method == "median":
        med = L.median(axis=0, skipna=True)
        shift = med - med.median()
        return L.subtract(shift, axis=1)
    if method == "quantile":
        raise NotImplementedError(
            "quantile normalization is not yet ported; use method='median' "
            "(the reference default) or 'none'.")
    raise ValueError(f"unknown normalization {method!r} (use 'median' | 'none').")


def _spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _expit(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _irls_binomial(x, succ, fail, *, max_iter: int = 100, tol: float = 1e-10):
    """Binomial logistic GLM by IRLS (numpy only): logit(p) = b0 + b1*x, with
    per-row trial counts succ+fail. Matches R ``glm(cbind(succ,fail) ~ x, binomial)``.
    Raises on divergence so callers can fall back."""
    x = np.asarray(x, float); succ = np.asarray(succ, float); fail = np.asarray(fail, float)
    nt = succ + fail
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ b
        p = _expit(eta)
        w = np.clip(nt * p * (1.0 - p), 1e-9, None)
        z = eta + (succ - nt * p) / w
        XtWX = X.T @ (X * w[:, None])
        XtWz = X.T @ (w * z)
        b_new = np.linalg.solve(XtWX, XtWz)
        if not np.all(np.isfinite(b_new)):
            raise FloatingPointError("IRLS diverged")
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    return float(b[0]), float(b[1])


def _logistic_lod50(mean_int, n_obs, n_tot, fallback_q):
    """Learn LOD50 = intensity at 50% detection from a binomial logistic fit of
    detection-count vs mean intensity. Falls back to a quantile of observed means
    if the slope is non-positive / non-finite (matching the reference)."""
    mean_int = np.asarray(mean_int, float); n_obs = np.asarray(n_obs, float)
    ok = np.isfinite(mean_int) & (n_obs > 0)
    x = mean_int[ok]; succ = n_obs[ok]; fail = (n_tot - n_obs)[ok]
    lod = float(np.nanquantile(x, fallback_q)) if x.size else float("nan")
    src = "fallback_quantile"; slope = float("nan")
    if x.size >= 3 and np.unique(x).size >= 2:
        try:
            b0, b1 = _irls_binomial(x, succ, fail)
            slope = b1
            if np.isfinite(b1) and b1 > 0:
                x0 = -b0 / b1
                if np.isfinite(x0):
                    lod = x0; src = "logistic_LOD50"
        except Exception:
            pass
    return {"lod": lod, "slope": slope, "source": src}


def fisher_2x2(a, b, c, d, *, confidence: float = 0.95):
    """Fisher's exact test on a 2x2 ``[[a, b], [c, d]]``.

    Returns ``(p, odds_ratio, ci_low, ci_high)`` where ``p`` is the two-sided
    exact p-value and the odds ratio is the **conditional MLE** with an exact
    confidence interval (matching R ``fisher.test``). If the conditional OR is
    unavailable, falls back to the sample OR ``ad/bc`` with NaN bounds.

    With rows = (group A, group B) and cols = (observed, missing), ``OR > 1`` means
    the feature is detected more in group A. Shared by the missingness classifier
    and the presence/absence DE track.
    """
    from scipy.stats import fisher_exact
    table = [[int(a), int(b)], [int(c), int(d)]]
    p = float(fisher_exact(table, alternative="two-sided")[1])
    OR = lo = hi = float("nan")
    try:
        from scipy.stats.contingency import odds_ratio
        res = odds_ratio(table, kind="conditional")
        OR = float(res.statistic)
        ci = res.confidence_interval(confidence_level=confidence)
        lo, hi = float(ci.low), float(ci.high)
    except Exception:
        denom = b * c
        OR = float(a * d / denom) if denom else float("inf")
    return p, OR, lo, hi


def classify_missingness(
    qd: "QuantData",
    *,
    normalization: str = "median",
    on_min: int = 7,
    hi_conf_max: float = 10.0,
    min_detect_overall: float = 0.50,
    cd_present_min: float = 0.70,
    cd_absent_max: float = 0.30,
    cd_floor_margin: float = 1.0,
    mnar_quantile: float = 0.50,
    contaminant_class=None,
) -> pd.DataFrame:
    """Classify each protein's missingness mechanism for a two-group contrast.

    Faithful port of the reference 03c cascade (see the section banner above).
    Requires exactly two groups in ``qd`` (e.g. genotype); their labels, sorted,
    name the per-group columns (``obs_<A>``, ``frac_obs_<A>``, ...).

    Parameters mirror the reference ``cfg`` (``on_min``, ``hi_conf_max`` in
    percent, ``min_detect_overall``, ``cd_present_min``, ``cd_absent_max``,
    ``cd_floor_margin``, ``mnar_quantile``). ``contaminant_class`` is an optional
    per-protein Series (from :func:`ptero.quant.annotate_contaminants`) carried
    into the output for the downstream Fisher / impute steps; ``None`` -> all
    ``"ok"``.

    Returns a per-protein DataFrame indexed by protein-group ``id`` with the
    mechanism, the Fisher detection p / conditional OR (for restricted +
    condition-dependent candidates), ``near_floor``, ``call_ready``
    (genotype_restricted with >= ``on_min`` in the present group), the routed
    ``imputation_method``, and a ``reason``. Run-level provenance (learned LOD50,
    its source, the global Spearman, the group labels, n samples) is attached in
    ``.attrs`` for plotting / reporting.
    """
    members = qd.group_members()
    groups = sorted(members)
    if len(groups) != 2:
        raise ValueError(
            f"classify_missingness needs exactly 2 groups (got {groups or 'none'}). "
            "Pass a 2-level groups= to load_quant, e.g. genotype.")
    ga, gb = groups
    sa, sb = members[ga], members[gb]
    nA, nB = len(sa), len(sb)
    n = nA + nB

    M = log2_median_normalize(qd.lfq, method=normalization)
    P = int(M.shape[0])
    A, B = M[sa], M[sb]
    oA = A.notna().sum(axis=1).to_numpy()
    oB = B.notna().sum(axis=1).to_numpy()
    n_obs = oA + oB
    pct_miss = 100.0 * (1.0 - n_obs / n)
    det_rate = n_obs / n
    fA = oA / nA if nA else np.zeros(P)
    fB = oB / nB if nB else np.zeros(P)
    mean_int = M.mean(axis=1, skipna=True).to_numpy()
    meanA = A.mean(axis=1, skipna=True).to_numpy()
    meanB = B.mean(axis=1, skipna=True).to_numpy()
    min_obs = M.min(axis=1, skipna=True).to_numpy()   # NaN where a row is all-missing

    okc = np.isfinite(mean_int) & (pct_miss < 100)
    gcorr = _spearman(mean_int[okc], pct_miss[okc]) if okc.any() else float("nan")
    L = _logistic_lod50(mean_int, n_obs, n, mnar_quantile)
    lod = L["lod"]
    int_pctile = np.full(P, np.nan)
    if okc.any():
        int_pctile[okc] = pd.Series(mean_int[okc]).rank().to_numpy() / int(okc.sum())

    # Fisher detection test (p + conditional OR) for restricted + CD candidates
    det_p = np.full(P, np.nan)
    det_or = np.full(P, np.nan)
    cand = (n_obs > 0) & (
        (oA == 0) | (oB == 0)
        | ((fA >= cd_present_min) & (fB <= cd_absent_max))
        | ((fB >= cd_present_min) & (fA <= cd_absent_max))
    )
    for i in np.nonzero(cand)[0]:
        p_i, or_i, _, _ = fisher_2x2(int(oA[i]), nA - int(oA[i]),
                                     int(oB[i]), nB - int(oB[i]))
        det_p[i] = p_i
        det_or[i] = or_i

    mech = np.empty(P, dtype=object)
    reason = np.empty(P, dtype=object)
    near_floor = np.array([None] * P, dtype=object)
    for i in range(P):
        a, b = int(oA[i]), int(oB[i])
        if a == 0 and b == 0:
            mech[i] = UNDETECTED
            reason[i] = "never detected"
        elif a == 0 or b == 0:
            mech[i] = GENOTYPE_RESTRICTED
            if a == 0:
                zero_g, on_g, on_n = ga, gb, b
            else:
                zero_g, on_g, on_n = gb, ga, a
            reason[i] = (f"0 in {zero_g}, {on_n} in {on_g} "
                         f"(Fisher presence/absence; not imputed)")
        elif (fA[i] >= cd_present_min and fB[i] <= cd_absent_max) or \
             (fB[i] >= cd_present_min and fA[i] <= cd_absent_max):
            if fA[i] >= cd_present_min:
                pres_g, miss_g, mu_miss = ga, gb, meanB[i]
            else:
                pres_g, miss_g, mu_miss = gb, ga, meanA[i]
            nf = bool(np.isfinite(mu_miss) and mu_miss <= lod + cd_floor_margin)
            near_floor[i] = nf
            mech[i] = CONDITION_DEPENDENT
            head = (f"{pres_g} det>={cd_present_min*100:.0f}%, "
                    f"{miss_g} det<={cd_absent_max*100:.0f}% "
                    f"-> Fisher presence/absence (not imputed)")
            if np.isfinite(mu_miss):
                tail = ("near floor (low-abundance-like)" if nf
                        else "above floor (heterogeneity/technical-like)")
                reason[i] = f"{head}; {miss_g} obs mean {mu_miss:.2f} {tail}"
            else:
                reason[i] = head
        elif pct_miss[i] <= hi_conf_max:
            mech[i] = HIGH_CONFIDENCE
            reason[i] = f"detected in {det_rate[i]*100:.0f}% of samples"
        elif det_rate[i] < min_detect_overall:
            mech[i] = TOO_SPARSE
            reason[i] = f"detected in only {det_rate[i]*100:.0f}% (majority missing; NOT imputed)"
        elif np.isfinite(min_obs[i]) and min_obs[i] <= lod:
            mech[i] = LOW_ABUNDANCE_MNAR
            reason[i] = f"symmetric; observed reach floor (min {min_obs[i]:.2f} <= LOD50 {lod:.2f})"
        else:
            mech[i] = MAR_SCATTERED
            reason[i] = f"symmetric sporadic; observed above floor (min {min_obs[i]:.2f} > LOD50 {lod:.2f})"

    call_ready = (mech == GENOTYPE_RESTRICTED) & (np.maximum(oA, oB) >= on_min)
    method = np.array([_IMPUTE_METHOD[m] for m in mech], dtype=object)

    if contaminant_class is None:
        cc = pd.Series("ok", index=M.index, dtype=object)
    else:
        cc = pd.Series(contaminant_class).reindex(M.index).fillna("ok")

    out = pd.DataFrame(index=M.index)
    out.index.name = "id"
    if "gene" in qd.proteins.columns:
        out["gene"] = qd.proteins["gene"].reindex(M.index).to_numpy()
    if "label" in qd.proteins.columns:
        out["label"] = qd.proteins["label"].reindex(M.index).to_numpy()
    if "majority_protein_ids" in qd.proteins.columns:
        out["majority_protein_ids"] = qd.proteins["majority_protein_ids"].reindex(M.index).to_numpy()
    out["contaminant_class"] = cc.to_numpy()
    out["n_obs"] = n_obs.astype(int)
    out["pct_missing"] = np.round(pct_miss, 1)
    out["mean_log2_intensity"] = np.round(mean_int, 2)
    out["min_log2_intensity"] = np.round(min_obs, 2)
    out[f"mean_{ga}_obs"] = np.round(meanA, 2)
    out[f"mean_{gb}_obs"] = np.round(meanB, 2)
    out["intensity_percentile"] = np.round(int_pctile, 3)
    out[f"obs_{ga}"] = oA.astype(int)
    out[f"obs_{gb}"] = oB.astype(int)
    out[f"frac_obs_{ga}"] = np.round(fA, 2)
    out[f"frac_obs_{gb}"] = np.round(fB, 2)
    out["near_floor"] = near_floor
    out["detection_fisher_p"] = det_p
    out["detection_OR"] = det_or
    out["mechanism"] = mech
    out["call_ready"] = call_ready
    out["imputation_method"] = method
    out["reason"] = reason

    out.attrs["lod50"] = lod
    out.attrs["lod50_source"] = L["source"]
    out.attrs["lod50_slope"] = L["slope"]
    out.attrs["spearman_missing_intensity"] = gcorr
    out.attrs["groups"] = (ga, gb)
    out.attrs["n_samples"] = n
    return out


def mechanism_counts(diag: pd.DataFrame) -> pd.Series:
    """Mechanism class counts in canonical order (impute / Fisher / excluded)."""
    c = diag["mechanism"].value_counts()
    order = [m for m in MECHANISM_ORDER if m in c.index]
    return c.reindex(order)


def plot_missingness_diagnose(diag: pd.DataFrame, *, min_detect_overall: float = 0.50,
                              figsize=(8.8, 5.4)) -> Figure:
    """Detection rate vs mean observed intensity, coloured by mechanism.

    The LOD50 vline and the min-detection hline are the two decision boundaries;
    a left-leaning cloud (low intensity -> low detection) is the MNAR fingerprint.
    Reads the learned LOD50 / Spearman / group labels from ``diag.attrs``.
    """
    lod = diag.attrs.get("lod50", float("nan"))
    gcorr = diag.attrs.get("spearman_missing_intensity", float("nan"))
    n = diag.attrs.get("n_samples", None)

    d = diag[np.isfinite(diag["mean_log2_intensity"].to_numpy())
             & (diag["mechanism"] != UNDETECTED)].copy()
    denom = n if n else float(d["n_obs"].max() or 1)
    d["detrate"] = d["n_obs"] / denom

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    present = [m for m in MECHANISM_ORDER if (d["mechanism"] == m).any()]
    for m in present:
        sub = d[d["mechanism"] == m]
        ax.scatter(sub["mean_log2_intensity"], sub["detrate"], s=7, alpha=0.45,
                   color=DIAGNOSE_COLORS[m], edgecolors="none", zorder=2)
    ax.axhline(min_detect_overall, linestyle=":", color="#666666", linewidth=1.0, zorder=1)
    if np.isfinite(lod):
        ax.axvline(lod, linestyle="--", color="black", linewidth=1.0, zorder=1)
        ax.text(lod, 0.02, f" LOD50 = {lod:.2f}", ha="left", va="bottom", fontsize=8)

    ax.set_xlabel("mean observed log2 intensity")
    ax.set_ylabel("detection rate (obs / n)")
    title = "Missingness mechanism"
    if np.isfinite(gcorr):
        title += f"  (Spearman %miss~intensity = {gcorr:.2f})"
    ax.set_title(title)
    _style_axes(ax)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=DIAGNOSE_COLORS[m], label=m)
               for m in present]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, title="mechanism")
    return fig