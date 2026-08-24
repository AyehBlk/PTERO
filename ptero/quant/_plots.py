"""
ptero.quant._plots -- quantification display, A2 of the multi-sample arc.

Pure plotting: every function builds a Matplotlib ``Figure`` via the object API
(``Figure()`` + ``add_subplot``), never ``pyplot`` -- the same discipline as
``ptero.dashboard._qc_plots``, so nothing leaks global state in the long-running
Panel server, and the figures embed straight into a Panel ``Matplotlib`` pane.

No statistics here (a later arc) -- three views:

* :func:`plot_protein_quant` -- one protein's LFQ across samples, coloured by
  group, with a per-group mean line; **missing shown at the baseline, never zero**.
* :func:`plot_lfq_distributions` -- per-sample LFQ box plots; because LFQ is
  normalised, the medians should line up. The read's sanity check on a dataset
  with no spike-in ground truth.
* :func:`plot_detection_counts` -- proteins quantified per sample, with the mean.

A shared :func:`_style_axes` keeps a consistent, de-cluttered look (no top/right
spines, light y-grid); colours are assigned per group from a fixed palette in
group order -- no hard-coded condition names, so it works for any experiment.
Groups are separated along x by a small gap (:func:`_positions`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from ._quant import QuantData

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
_MISSING = "#BFBFBF"
_FALLBACK = "#4C72B0"
_TEXT = "#222222"
_GRID = "#E8E8E8"
_SPINE = "#BBBBBB"
_REF = "#555555"          # reference lines (global median / mean)
_GROUP_GAP = 0.8          # extra x-spacing between groups


def _ordered_samples(qd: "QuantData") -> list[str]:
    if qd.groups:
        return sorted(qd.samples, key=lambda s: (qd.groups.get(s, ""), s))
    return list(qd.samples)


def _group_colors(qd: "QuantData") -> dict[str, str]:
    order: list[str] = []
    for s in _ordered_samples(qd):
        g = qd.groups.get(s) if qd.groups else None
        if g is not None and g not in order:
            order.append(g)
    return {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(order)}


def _sample_color(qd, sample, gcolors) -> str:
    g = qd.groups.get(sample) if qd.groups else None
    return gcolors.get(g, _FALLBACK)


def _positions(qd: "QuantData", samples, gap: float = _GROUP_GAP) -> np.ndarray:
    """x positions with an extra gap inserted between groups (consistent across plots)."""
    xs, cur, prev = [], 0.0, None
    for s in samples:
        g = qd.groups.get(s) if qd.groups else None
        if prev is not None and g != prev:
            cur += gap
        xs.append(cur)
        cur += 1.0
        prev = g
    return np.asarray(xs, dtype=float)


def _style_axes(ax) -> None:
    """Shared house style: drop top/right spines, soften the rest, light y-grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_SPINE)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=_TEXT, labelsize=8, length=3, width=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.title.set_color(_TEXT)
    ax.title.set_fontsize(11.5)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.xaxis.label.set_fontsize(9.5)
    ax.yaxis.label.set_fontsize(9.5)


def _group_legend(qd, gcolors, *, missing: bool):
    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=g)
               for g, c in gcolors.items()]
    if missing:
        handles.append(Line2D([0], [0], marker="v", linestyle="", color=_MISSING,
                              label="missing (not quantified)"))
    return handles


def plot_protein_quant(qd: "QuantData", protein_id, *, log2: bool = True,
                       figsize=(8.5, 4.6)) -> Figure:
    """One protein's per-sample LFQ, with per-group mean lines.

    Missing samples are drawn as grey markers at the baseline -- never as zero.
    """
    if protein_id not in qd.lfq.index:
        raise KeyError(f"protein id {protein_id!r} not in quant data "
                       f"(e.g. ids {list(qd.lfq.index[:5])}).")
    samples = _ordered_samples(qd)
    xs = _positions(qd, samples)
    vals = qd.lfq.loc[protein_id, samples].astype(float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log2(vals) if log2 else vals
    gcolors = _group_colors(qd)

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    finite = y[np.isfinite(y)]
    baseline = (float(finite.min()) - 0.6) if finite.size else 0.0

    # per-group mean line (over present values only)
    if qd.groups:
        for g in dict.fromkeys(qd.groups.get(s) for s in samples):
            sel = [i for i, s in enumerate(samples) if qd.groups.get(s) == g]
            gy = y[sel][np.isfinite(y[sel])]
            if gy.size:
                gx = xs[sel]
                ax.hlines(float(gy.mean()), gx.min() - 0.35, gx.max() + 0.35,
                          color=gcolors[g], linewidth=2.2, alpha=0.45, zorder=2)

    for i, s in enumerate(samples):
        if np.isfinite(y[i]):
            ax.scatter(xs[i], y[i], color=_sample_color(qd, s, gcolors),
                       s=72, zorder=3, edgecolor="white", linewidth=0.7)
    miss = [i for i in range(len(samples)) if not np.isfinite(y[i])]
    if miss:
        ax.scatter(xs[miss], [baseline] * len(miss), marker="v", color=_MISSING,
                   s=55, zorder=3, edgecolor="white", linewidth=0.4)

    ax.set_xticks(xs)
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylabel("log2 LFQ intensity" if log2 else "LFQ intensity")
    if not log2:
        ax.set_yscale("log")
    label = qd.proteins.loc[protein_id, "label"] \
        if "label" in qd.proteins.columns else protein_id
    ax.set_title(f"{label}   (id {protein_id})")
    _style_axes(ax)

    handles = _group_legend(qd, gcolors, missing=bool(miss))
    if qd.groups:
        handles.append(Line2D([0], [0], color="#999999", linewidth=2.2,
                              alpha=0.6, label="group mean"))
    if handles:
        ax.legend(handles=handles, title="group" if qd.groups else None, fontsize=8,
                  frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0)
    return fig


def plot_lfq_distributions(qd: "QuantData", *, log2: bool = True,
                           figsize=(9.2, 5.0)) -> Figure:
    """Per-sample LFQ box plots. Medians should align if LFQ is normalised."""
    samples = _ordered_samples(qd)
    xs = _positions(qd, samples)
    gcolors = _group_colors(qd)
    data = []
    for s in samples:
        v = qd.lfq[s].dropna().astype(float).to_numpy()
        if v.size:
            data.append(np.log2(v) if log2 else v)
        else:
            data.append(np.array([np.nan]))        # keep box positions aligned

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    bp = ax.boxplot(
        data, positions=xs, widths=0.62, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(color="#888888", linewidth=0.9),
        capprops=dict(color="#888888", linewidth=0.9),
        boxprops=dict(linewidth=0.8, edgecolor="#888888"),
    )
    for patch, s in zip(bp["boxes"], samples):
        patch.set_facecolor(_sample_color(qd, s, gcolors))
        patch.set_alpha(0.65)

    nonempty = [d for d in data if np.isfinite(d).any()]
    if nonempty:
        ax.axhline(float(np.nanmedian(np.concatenate(nonempty))),
                   color=_REF, linestyle="--", linewidth=1, alpha=0.7, zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylabel("log2 LFQ intensity" if log2 else "LFQ intensity")
    if not log2:
        ax.set_yscale("log")
    ax.set_title("Per-sample LFQ distributions", pad=22)
    ax.text(0.5, 1.012, "medians should align if LFQ is correctly normalised",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#777777")
    _style_axes(ax)

    handles = _group_legend(qd, gcolors, missing=False)
    handles.append(Line2D([0], [0], linestyle="--", color=_REF, label="global median"))
    ax.legend(handles=handles, title="group" if qd.groups else None, fontsize=8,
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    return fig


def plot_detection_counts(qd: "QuantData", *, figsize=(9.2, 4.7)) -> Figure:
    """Proteins quantified (non-missing LFQ) per sample, with the mean line."""
    samples = _ordered_samples(qd)
    xs = _positions(qd, samples)
    gcolors = _group_colors(qd)
    counts = [int(qd.lfq[s].notna().sum()) for s in samples]

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    ax.bar(xs, counts, width=0.72,
           color=[_sample_color(qd, s, gcolors) for s in samples],
           alpha=0.85, zorder=2)
    mean_c = float(np.mean(counts)) if counts else 0.0
    ax.axhline(mean_c, color=_REF, linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    for x, c in zip(xs, counts):
        ax.text(x, c, f"{c:,}", ha="center", va="bottom", fontsize=7.5, color="#333333")
    if counts:
        ax.set_ylim(0, max(counts) * 1.12)         # headroom for labels + legend

    ax.set_xticks(xs)
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylabel("proteins quantified (non-missing LFQ)")
    ax.set_title("Per-sample quantification depth")
    _style_axes(ax)

    handles = _group_legend(qd, gcolors, missing=False)
    handles.append(Line2D([0], [0], linestyle="--", color=_REF, label=f"mean {mean_c:,.0f}"))
    ax.legend(handles=handles, title="group" if qd.groups else None, fontsize=8,
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    return fig