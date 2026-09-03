"""
ptero.quant._deviz -- visualisation of differential-expression results.

Built on the object-API ``Figure`` (never pyplot), same discipline and house style
as ``ptero.quant._plots`` (it reuses ``_style_axes``), so the same functions embed
in a Panel pane and back the dashboard DE page.

:func:`volcano` takes a single method x track result -- a ``limma_de`` / ``deqms_de``
/ ``welch_de`` output, or one ``(track, method)`` slice of the
:func:`ptero.quant.differential_abundance` long table -- and draws log2FC vs
-log10(p), coloured by direction (read from the ``contrast`` attr), with the BH
threshold marked. The presence/absence on/off candidates have no p-value, so when
passed they ride the top of the plot as diamonds, placed on x by their
detection-limit log2FC and coloured by the group they are present in.
"""
from __future__ import annotations

import re
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ._plots import _style_axes

_UP = "#DE8F05"       # positive log2FC -> higher in the non-reference group
_DOWN = "#0173B2"     # negative log2FC -> higher in the reference group
_NS = "#C9C9C9"       # not significant
_GUIDE = "#999999"
_ZERO = "#DDDDDD"


def _short_label(s, maxlen: int = 16) -> str:
    """Gene symbol only (drop trailing accessions after ';'), length-capped."""
    s = str(s).split(";")[0].strip()
    return s if len(s) <= maxlen else s[: maxlen - 1] + "\u2026"


def _contrast_labels(de: pd.DataFrame):
    """Direction labels + group names parsed from the ``contrast`` attr (``A_vs_B``)."""
    c = de.attrs.get("contrast")
    if isinstance(c, str) and "_vs_" in c:
        up, ref = c.split("_vs_", 1)
        return f"higher in {up}", f"higher in {ref}", up, ref
    return "up (log2FC > 0)", "down (log2FC < 0)", None, None


def volcano(
    de: pd.DataFrame,
    *,
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    label_top: int = 10,
    labels=None,
    title: Optional[str] = None,
    figsize=(8.8, 7.2),
) -> Figure:
    """Volcano plot for one method x track DE result (purely quantitative).

    On/off (presence/absence) candidates are deliberately **not** shown here -- they
    have no p-value and live in a different measurement space; use
    :func:`detection_matrix` for those.

    Parameters
    ----------
    de : DataFrame
        Per-protein DE result with ``log2FC`` and ``p`` (``padj`` computed via BH
        if absent). A ``label`` column is used for annotation when present. Reads
        ``de.attrs['contrast']`` / ``['method']`` for the axis/title text.
    alpha : float
        BH-padj significance threshold (default 0.05).
    lfc_thresh : float
        |log2FC| threshold for the highlight + guide lines (default 1.0).
    label_top : int
        Annotate this many proteins (significant first, then lowest-p) with labels.
    labels : mapping, optional
        id -> gene label, used when ``de`` has no ``label`` column.
    """
    if "log2FC" not in de.columns or "p" not in de.columns:
        raise ValueError("volcano needs 'log2FC' and 'p' columns.")
    d = de.reset_index(drop=False)
    lfc = d["log2FC"].to_numpy(dtype=float)
    p = d["p"].to_numpy(dtype=float)
    if "padj" in d.columns:
        padj = d["padj"].to_numpy(dtype=float)
    else:
        from ._de import _bh
        padj = _bh(p)

    if "label" in d.columns:
        lab = d["label"].astype(str).to_numpy()
    elif labels is not None:
        lab = pd.Series(labels).reindex(de.index).astype(str).to_numpy()
    else:
        lab = de.index.astype(str).to_numpy()

    with np.errstate(divide="ignore"):
        y = -np.log10(np.clip(p, 1e-300, 1.0))
    sig = (padj < alpha) & (np.abs(lfc) >= lfc_thresh)
    up = sig & (lfc > 0)
    down = sig & (lfc < 0)
    ns = ~sig

    up_lab, down_lab, up_grp, ref_grp = _contrast_labels(de)

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)

    ax.scatter(lfc[ns], y[ns], s=10, c=_NS, alpha=0.5, linewidths=0, zorder=1)
    ax.scatter(lfc[down], y[down], s=36, c=_DOWN, alpha=0.9,
               edgecolor="white", linewidths=0.3, zorder=3)
    ax.scatter(lfc[up], y[up], s=36, c=_UP, alpha=0.9,
               edgecolor="white", linewidths=0.3, zorder=3)

    # guides: zero, +/- lfc threshold, BH p threshold
    ax.axvline(0, color=_ZERO, lw=0.8, zorder=0)
    if lfc_thresh > 0:
        for xg in (-lfc_thresh, lfc_thresh):
            ax.axvline(xg, color=_GUIDE, ls="--", lw=0.9, zorder=0)

    xmax = float(np.nanmax(np.abs(lfc))) * 1.15 if lfc.size else 1.0
    ax.set_xlim(-xmax, xmax)
    ymax = float(np.nanmax(y)) * 1.12 if y.size and np.isfinite(y).any() else 1.0

    passed = p[padj < alpha]
    if passed.size:
        pthr = float(passed.max())
        yln = -np.log10(pthr)
        ax.axhline(yln, color=_GUIDE, ls=":", lw=1.0, zorder=0)
        ax.text(xmax, yln, f"BH {alpha:g} ", va="bottom", ha="right",
                fontsize=7, color=_GUIDE)

    # on/off candidates are shown separately (detection_matrix), not here
    ax.set_ylim(0, ymax)

    # annotate top proteins (significant first, then lowest-p so the lead hit shows);
    # greedily skip a label that would collide with one already placed
    order = list(np.argsort(p))
    picked = [i for i in order if sig[i]][:label_top]
    if len(picked) < label_top:
        picked += [i for i in order if i not in picked][:label_top - len(picked)]
    placed = []
    xr, yr = 2 * xmax, max(ymax, 1e-9)
    for i in picked:
        xi, yi = lfc[i], y[i]
        if any(abs(xi - px) < 0.055 * xr and abs(yi - py) < 0.045 * yr for px, py in placed):
            continue
        placed.append((xi, yi))
        ax.annotate(_short_label(lab[i]), (xi, yi),
                    xytext=(3 if xi >= 0 else -3, 3), textcoords="offset points",
                    ha="left" if xi >= 0 else "right", va="bottom",
                    fontsize=7, color="#333333")

    xlab = (f"log2 fold change    (<- {down_lab}   |   {up_lab} ->)"
            if up_grp else "log2 fold change")
    ax.set_xlabel(xlab)
    ax.set_ylabel("-log10 p")
    method = de.attrs.get("method", "")
    contrast = de.attrs.get("contrast", "")
    ax.set_title(title or f"Volcano  {contrast}" + (f"  ({method})" if method else ""))
    _style_axes(ax)

    handles = [Line2D([0], [0], marker="o", ls="", color=_UP, label=up_lab),
               Line2D([0], [0], marker="o", ls="", color=_DOWN, label=down_lab),
               Line2D([0], [0], marker="o", ls="", color=_NS, label="not significant")]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    ax.text(0.99, 0.01,
            f"{int(up.sum())} up / {int(down.sum())} down "
            f"(padj<{alpha:g}, |log2FC|>={lfc_thresh:g})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#555555")
    return fig


def detection_matrix(
    qd,
    pa: pd.DataFrame,
    *,
    which: str = "call_ready",
    ref_group=None,
    max_proteins: int = 40,
    cmap: str = "viridis",
    title=None,
    figsize=None,
) -> Figure:
    """Per-sample detection pattern for the presence/absence (on/off) candidates.

    On/off proteins are categorical detection evidence with no p-value, so this
    shows the raw truth instead of a fake coordinate: one row per protein, one cell
    per sample, split into two group blocks. A **filled** dot means the protein was
    detected in that sample (coloured by its log2 intensity); an **open ring** means
    it was not. Rows are sorted so proteins present in the non-reference group
    cluster at the top and reference-group ones at the bottom -- the on/off split
    reads straight down the middle. Gene labels are coloured by which group the
    protein is present in (matching the volcano); ``near_floor`` (at the detection
    limit, ambiguous) rows are faded and flagged.

    Parameters
    ----------
    qd : QuantData -- supplies per-sample LFQ and group membership.
    pa : DataFrame -- a :func:`presence_absence` table (indexed like ``qd.lfq``).
    which : {"call_ready", "is_on_off", "all"} -- which candidates to show.
    ref_group : str, optional -- placed as the left block (default: sorted first).
    max_proteins : int -- cap on rows (strongest kept), to keep the figure legible.
    cmap : str -- colormap for the detected-intensity fill.
    """
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sel = pa if which == "all" else (pa[pa[which]] if which in pa.columns else pa)
    if not len(sel):
        raise ValueError(f"no proteins to show for which={which!r}.")

    groups = sorted(set(qd.groups.values()))
    if len(groups) != 2:
        raise ValueError(f"detection_matrix needs exactly 2 groups; got {groups}.")
    if ref_group and ref_group in groups:
        left, right = ref_group, next(g for g in groups if g != ref_group)
    else:
        left, right = groups[0], groups[1]
    left_s = [s for s in qd.samples if qd.groups.get(s) == left]
    right_s = [s for s in qd.samples if qd.groups.get(s) == right]
    samples = left_s + right_s
    nL = len(left_s)

    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log2(qd.lfq.loc[sel.index, samples].astype(float).to_numpy())
    detected = ~np.isnan(L)

    enr = sel["enriched_in"].to_numpy() if "enriched_in" in sel.columns else np.array([None] * len(sel))
    nl_cnt = detected[:, :nL].sum(1)
    nr_cnt = detected[:, nL:].sum(1)
    with np.errstate(invalid="ignore"):
        meanint = np.nanmean(np.where(detected, L, np.nan), axis=1)
    meanint = np.nan_to_num(meanint, nan=np.nanmin(L) if np.isfinite(L).any() else 0.0)

    # right-enriched on top; then by detection count in the present group, then intensity
    order = sorted(range(len(sel)), key=lambda i: (
        0 if enr[i] == right else 1,
        -(nr_cnt[i] if enr[i] == right else nl_cnt[i]),
        -meanint[i]))[:max_proteins]
    L, detected, sel = L[order], detected[order], sel.iloc[order]
    nl_cnt, nr_cnt = nl_cnt[order], nr_cnt[order]
    n = len(order)

    # column x-positions with a gap between the two blocks
    gap = 1.4
    xs = np.array(list(range(nL)) + [nL - 1 + gap + 1 + i for i in range(len(right_s))], dtype=float)
    div = (xs[nL - 1] + xs[nL]) / 2 if len(right_s) else xs[nL - 1] + 0.5

    vals = L[detected]
    vmin = float(np.nanpercentile(vals, 5)) if vals.size else 0.0
    vmax = float(np.nanpercentile(vals, 95)) if vals.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = colormaps[cmap]

    if figsize is None:
        figsize = (max(6.5, 0.52 * len(samples) + 5.0), max(3.0, 0.36 * n + 2.0))
    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)

    # block tints + divider
    ax.axvspan(xs[0] - 0.5, xs[nL - 1] + 0.5, color=_DOWN, alpha=0.05, zorder=0)
    if len(right_s):
        ax.axvspan(xs[nL] - 0.5, xs[-1] + 0.5, color=_UP, alpha=0.05, zorder=0)
    ax.axvline(div, color="#CCCCCC", lw=1.0, zorder=1)

    for r in range(n):
        for c in range(len(samples)):
            if detected[r, c]:
                ax.scatter(xs[c], r, s=135, color=cmap_obj(norm(L[r, c])),
                           edgecolor="white", linewidths=0.6, zorder=3)
            else:
                ax.scatter(xs[c], r, s=66, facecolor="white", edgecolor="#D2D2D2",
                           linewidths=1.0, zorder=2)

    has_nf = "near_floor" in sel.columns
    for r in range(n):
        e = sel["enriched_in"].iloc[r] if "enriched_in" in sel.columns else None
        nf = bool(sel["near_floor"].iloc[r]) if has_nf else False
        col = _UP if e == right else (_DOWN if e == left else "#555555")
        ax.text(xs[0] - 1.1, r, _short_label(sel["label"].iloc[r]),
                ha="right", va="center", fontsize=8.5,
                color=col, alpha=0.5 if nf else 1.0,
                fontstyle="italic" if nf else "normal")
        # per-group detection count, left/right = block order, coloured by group
        ax.text(xs[-1] + 1.0, r, str(int(nl_cnt[r])), ha="left", va="center",
                fontsize=8, color=_DOWN)
        ax.text(xs[-1] + 1.5, r, "/", ha="center", va="center", fontsize=8, color="#999999")
        ax.text(xs[-1] + 1.9, r, str(int(nr_cnt[r])), ha="left", va="center",
                fontsize=8, color=_UP)
        if nf:
            ax.scatter(xs[-1] + 3.0, r, marker="D", s=28, facecolor="none",
                       edgecolor="#999999", linewidths=1.0, zorder=3)

    # group headers (coloured) above the blocks
    ax.text(float(np.mean(xs[:nL])), -1.3, left, ha="center", va="bottom",
            fontsize=12, color=_DOWN, fontweight="bold")
    if len(right_s):
        ax.text(float(np.mean(xs[nL:])), -1.3, right, ha="center", va="bottom",
                fontsize=12, color=_UP, fontweight="bold")
    ax.text(xs[-1] + 1.9, -1.3, f"{left}/{right}", ha="center", va="bottom",
            fontsize=7.5, color="#888888")

    ax.set_xticks(xs)
    ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_xlim(xs[0] - 4.0, xs[-1] + 4.0)
    ax.set_ylim(n - 0.4, -2.4)                     # inverts: row 0 at top, headers above
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(length=0)
    ax.set_title(title or f"On/off detection matrix  ({left} vs {right})", fontsize=12)

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("log2 intensity (when detected)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor="#9E9E9E",
                      markeredgecolor="white", label="detected"),
               Line2D([0], [0], marker="o", ls="", markerfacecolor="white",
                      markeredgecolor="#D2D2D2", label="not detected")]
    if has_nf and bool(sel["near_floor"].any()):
        handles.append(Line2D([0], [0], marker="D", ls="", markerfacecolor="none",
                              markeredgecolor="#999999", label="near detection floor"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=len(handles), fontsize=8, frameon=False)
    return fig


def heatmap_top(
    de: pd.DataFrame,
    matrix: pd.DataFrame,
    groups,
    *,
    top: int = 30,
    by: str = "p",
    ref_group=None,
    zscore: bool = True,
    cmap: str = "RdBu_r",
    title=None,
    figsize=None,
) -> Figure:
    """Heatmap of the top differential proteins across samples.

    Rows are the top ``top`` proteins from ``de`` (by ``by`` = ``"p"`` / ``"padj"`` /
    ``"log2FC"``), ordered by log2FC (up-regulated at the top); columns are the
    samples, split into two group blocks. Values are per-protein z-scored by default
    (so the diverging colour shows the relative pattern, not absolute abundance).

    Parameters
    ----------
    de : DataFrame -- a DE result (``log2FC``, ``p``/``padj``, optional ``label``).
    matrix : DataFrame -- proteins x samples log2 intensities the DE was run on
        (complete-case or imputed); indexed by protein id like ``de``.
    groups : mapping {sample -> group}.
    ref_group : str, optional -- placed as the left block (default: sorted first).
    """
    from matplotlib.cm import ScalarMappable

    key = by if by in ("p", "padj", "log2FC") else "p"
    if key == "log2FC":
        idx = de["log2FC"].abs().sort_values(ascending=False).index
    else:
        idx = de[key].sort_values().index
    ids = [i for i in idx if i in matrix.index][:top]
    if not ids:
        raise ValueError("no top proteins found in the matrix.")

    grps = sorted(set(groups.values()))
    if len(grps) != 2:
        raise ValueError(f"heatmap_top needs exactly 2 groups; got {grps}.")
    left, right = (ref_group, next(g for g in grps if g != ref_group)) \
        if (ref_group and ref_group in grps) else (grps[0], grps[1])
    left_s = [s for s in matrix.columns if groups.get(s) == left]
    right_s = [s for s in matrix.columns if groups.get(s) == right]
    samples = left_s + right_s

    X = matrix.loc[ids, samples].to_numpy(dtype=float)
    if "log2FC" in de.columns:
        lfc = de.loc[ids, "log2FC"].to_numpy(dtype=float)
        ro = np.argsort(-lfc)
        X, ids = X[ro], [ids[i] for i in ro]
    if zscore:
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
    else:
        Z = X

    labels = (de["label"] if "label" in de.columns else pd.Series(matrix.index, index=matrix.index))
    ylab = [_short_label(labels.get(i, i)) for i in ids]
    n, m = len(ids), len(samples)

    if figsize is None:
        figsize = (max(6.0, 0.52 * m + 3.8), max(3.0, 0.32 * n + 2.0))
    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)

    vmax = float(np.nanpercentile(np.abs(Z), 98)) if zscore else None
    im = ax.imshow(Z, aspect="auto", cmap=cmap,
                   vmin=-vmax if zscore else None, vmax=vmax if zscore else None)

    # cell gridlines + group divider
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    div = len(left_s) - 0.5
    ax.axvline(div, color="white", lw=3.2, zorder=4)
    ax.axvline(div, color="#666666", lw=1.0, zorder=4)

    ax.set_xticks(range(m))
    ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ylab, fontsize=8)
    ax.text(float(np.mean(range(len(left_s)))), -0.9, left, ha="center", va="bottom",
            color=_DOWN, fontweight="bold", fontsize=11)
    ax.text(float(np.mean(range(len(left_s), m))), -0.9, right, ha="center", va="bottom",
            color=_UP, fontweight="bold", fontsize=11)
    ax.set_ylim(n - 0.5, -1.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title or f"Top {n} differential proteins  ({left} vs {right})", fontsize=12)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("row z-score" if zscore else "log2 intensity", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return fig


def plot_de_boxplot(
    matrix: pd.DataFrame,
    protein_id,
    groups,
    *,
    de: pd.DataFrame = None,
    label=None,
    ref_group=None,
    seed: int = 0,
    title=None,
    figsize=(5.2, 5.0),
) -> Figure:
    """Grouped intensity box + points for one protein, with the DE stats annotated.

    One box per group (distribution across that group's samples) with the individual
    sample values jittered on top, coloured by group (matching the volcano). If a DE
    result ``de`` is given, the protein's log2FC / p / padj are annotated.
    """
    if protein_id not in matrix.index:
        raise KeyError(f"protein id {protein_id!r} not in matrix.")
    grps = sorted(set(groups.values()))
    order = ([ref_group] + [g for g in grps if g != ref_group]) if (ref_group and ref_group in grps) else grps
    color = {g: (_DOWN if (ref_group and g == ref_group) else _UP) for g in order}
    if not ref_group and len(order) == 2:
        color = {order[0]: _DOWN, order[1]: _UP}

    row = matrix.loc[protein_id]
    vals = {g: [] for g in order}
    for s in matrix.columns:
        g = groups.get(s)
        v = row[s]
        if g in vals and np.isfinite(v):
            vals[g].append(float(v))

    rng = np.random.default_rng(seed)
    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    data = [vals[g] for g in order]
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#333333", linewidth=1.4),
                    whiskerprops=dict(color="#999999", linewidth=1.0),
                    capprops=dict(color="#999999", linewidth=1.0),
                    boxprops=dict(linewidth=0.9, edgecolor="#999999"))
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(color[g])
        patch.set_alpha(0.22)
    for i, g in enumerate(order):
        ys = vals[g]
        xs = rng.normal(i, 0.055, len(ys))
        ax.scatter(xs, ys, color=color[g], s=46, zorder=3, edgecolor="white", linewidths=0.7)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("normalized log2 intensity")
    _style_axes(ax)

    name = label
    if name is None and de is not None and "label" in de.columns and protein_id in de.index:
        name = de.loc[protein_id, "label"]
    name = _short_label(name if name is not None else protein_id)
    sub = ""
    if de is not None and protein_id in de.index:
        lfc = de.loc[protein_id, "log2FC"]
        p = de.loc[protein_id, "p"]
        padj = de.loc[protein_id, "padj"]
        method = de.attrs.get("method", "")
        sub = f"log2FC {lfc:+.2f}   p {p:.2g}   padj {padj:.2g}" + (f"   ({method})" if method else "")
    ax.set_title(name + (("\n" + sub) if sub else ""), fontsize=12)
    return fig


def _safe_name(s, maxlen: int = 40) -> str:
    """Filesystem-safe short gene name (drop accessions, strip illegal chars)."""
    s = str(s).split(";")[0].strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s[:maxlen] or "protein"


def save_de_boxplots(
    matrix: pd.DataFrame,
    de: pd.DataFrame,
    groups,
    out_dir,
    *,
    which: str = "significant",
    alpha: float = 0.05,
    lfc_thresh: float = 0.0,
    top: int = 20,
    proteins=None,
    ref_group=None,
    dpi: int = 200,
    fmt: str = "png",
    seed: int = 0,
):
    """Save one boxplot per protein for a selected set; returns the written paths.

    A thin batch wrapper over :func:`plot_de_boxplot` (which still draws a single
    protein). Selection is either an explicit ``proteins`` list or, failing that,
    ``which``:

    * ``"significant"`` -- ``padj < alpha`` (and ``|log2FC| >= lfc_thresh`` if set),
    * ``"top"`` -- the ``top`` proteins by p-value,
    * ``"all"`` -- every protein in ``de``.

    ``proteins`` accepts protein ids or gene labels (exact, then prefix match).
    Files are written to ``out_dir`` as ``boxplot_<gene>_<id>.<fmt>`` with
    filesystem-safe names. Everything experiment-specific (group names, reference
    group, colours, annotated stats) comes from ``groups`` / ``ref_group`` / ``de``.

    Returns
    -------
    list[pathlib.Path] -- the files written (empty if the selection is empty).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lab = de["label"].astype(str) if "label" in de.columns else None

    if proteins is not None:
        ids = []
        for pr in proteins:
            if pr in matrix.index:
                ids.append(pr)
            elif lab is not None:
                hits = list(de.index[lab == str(pr)]) or list(de.index[lab.str.startswith(str(pr))])
                ids.extend(hits)
    elif which == "significant":
        mask = de["padj"] < alpha
        if lfc_thresh and "log2FC" in de.columns:
            mask = mask & (de["log2FC"].abs() >= lfc_thresh)
        ids = list(de.index[mask])
    elif which == "top":
        ids = list(de["p"].sort_values().index[:top])
    elif which == "all":
        ids = list(de.index)
    else:
        raise ValueError("which must be 'significant' | 'top' | 'all', or pass proteins=.")

    ids = [i for i in dict.fromkeys(ids) if i in matrix.index]     # dedupe, keep drawable
    written = []
    for pid in ids:
        fig = plot_de_boxplot(matrix, pid, groups, de=de, ref_group=ref_group, seed=seed)
        gene = (lab.get(pid, pid) if lab is not None else pid)
        path = out / f"boxplot_{_safe_name(gene)}_{pid}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written