"""
QC comparison plots for the cross-sample view.

Three matplotlib figures built from the same objects the dashboard already
makes -- the ``compare_samples`` matrix, ``flag_outliers`` output, and
``flag_summary`` -- so they drop straight into the comparison tab via
``pn.pane.Matplotlib(fig)`` and also work standalone (e.g. in a notebook).

* ``plot_zscore_heatmap`` -- metric x sample heatmap of the robust z-score
  (the overview); flagged cells are boxed; near-constant cells are muted.
* ``plot_flag_tally``     -- per-sample stacked bar of flag counts by method.
* ``plot_metric_strips``  -- per-metric value-across-samples dot plots (default:
  only the flagged metrics) so the *magnitude* of each deviation is visible.

Figures are created with the ``Figure()`` object API (no pyplot global state) so
they don't leak in a long-running server; Panel attaches its own canvas, and for
file output you can attach ``FigureCanvasAgg(fig)`` before ``fig.savefig``.

The z-score here mirrors ``flag_outliers`` exactly (median / MAD with a mean-AD
fallback) so the colours and the flags always agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# flag colours (a little more saturated than the table highlights, for ink-on-white)
_FLAG_COLORS = {"both": "#e06666", "z": "#f6b26b", "iqr": "#ffd966"}
_PT_BLUE, _PT_RED = "#4a6fa5", "#e06666"


def _robust_z_matrix(comparison, sample_cols, *, min_samples: int = 3,
                     min_rel_dev: float | None = None) -> pd.DataFrame:
    """Per-metric modified z-score across samples (mirrors flag_outliers).

    If ``min_rel_dev`` is given, cells whose value is within that fraction of the
    metric median are muted to 0 -- so near-constant 'settings' metrics (a 0.08%
    difference scoring |z|=80) don't dominate the colour, matching the flag gate.
    """
    idx, data = [], []
    for _, r in comparison.iterrows():
        v = pd.to_numeric(pd.Series({c: r.get(c) for c in sample_cols}), errors="coerce")
        vv = v.dropna()
        if vv.size < min_samples:
            continue
        med = float(vv.median())
        mad = float((vv - med).abs().median())
        if mad > 0:
            scale = 1.4826 * mad
        else:
            mean_ad = float((vv - med).abs().mean())
            scale = 1.253 * mean_ad if mean_ad > 0 else 0.0
        z = (v - med) / scale if scale > 0 else v * 0.0
        if min_rel_dev is not None:
            rel = (v - med).abs() / max(abs(med), 1e-12)
            z = z.where(rel >= min_rel_dev, 0.0)        # mute practically-negligible cells
        idx.append(r["metric"])
        data.append(z.reindex(sample_cols))
    return pd.DataFrame(data, index=idx, columns=sample_cols)


def plot_zscore_heatmap(comparison, sample_cols, flags=None, *,
                        min_samples: int = 3, min_rel_dev: float = 0.05,
                        vmax: float = 5.0, figsize=None):
    """Heatmap of per-metric robust z-score across samples; flagged cells boxed.

    Colour is gated by ``min_rel_dev`` (default 5%) to match the flagging, so
    near-constant-metric artifacts stay neutral instead of dominating the view.
    """
    Z = _robust_z_matrix(comparison, sample_cols, min_samples=min_samples,
                         min_rel_dev=min_rel_dev)
    if Z.empty:
        fig = Figure(figsize=(6, 1.5))
        ax = fig.subplots()
        ax.text(0.5, 0.5, "Not enough samples to compute z-scores", ha="center", va="center")
        ax.axis("off")
        return fig
    n_m, n_s = Z.shape
    fig = Figure(figsize=figsize or (1.0 * n_s + 4.0, 0.30 * n_m + 1.6))
    ax = fig.subplots()
    im = ax.imshow(Z.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_s))
    ax.set_xticklabels(sample_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(Z.index, fontsize=7)
    if flags is not None and not flags.empty:
        pm = {m: i for i, m in enumerate(Z.index)}
        ps = {s: j for j, s in enumerate(sample_cols)}
        for _, f in flags.iterrows():
            if f["metric"] in pm and f["sample"] in ps:
                ax.add_patch(Rectangle((ps[f["sample"]] - 0.5, pm[f["metric"]] - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=1.8))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("robust z-score")
    ax.set_title("Per-metric robust z-score across samples  (boxed = flagged; "
                 "near-constant cells muted)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_flag_tally(summary, *, figsize=None):
    """Per-sample stacked bar of flagged-metric counts, split by method."""
    if (summary is None or summary.empty
            or summary[["n_flag_both", "n_flag_z", "n_flag_iqr"]].to_numpy().sum() == 0):
        fig = Figure(figsize=(6, 1.5))
        ax = fig.subplots()
        ax.text(0.5, 0.5, "No samples flagged on any metric", ha="center", va="center")
        ax.axis("off")
        return fig
    s = summary.copy()
    both = s["n_flag_both"].to_numpy(dtype=float)
    z_only = (s["n_flag_z"] - s["n_flag_both"]).clip(lower=0).to_numpy(dtype=float)
    iqr_only = (s["n_flag_iqr"] - s["n_flag_both"]).clip(lower=0).to_numpy(dtype=float)
    y = np.arange(len(s))[::-1]                      # worst-first at the top
    fig = Figure(figsize=figsize or (7.0, 0.42 * len(s) + 1.4))
    ax = fig.subplots()
    ax.barh(y, both, color=_FLAG_COLORS["both"], label="both methods")
    ax.barh(y, z_only, left=both, color=_FLAG_COLORS["z"], label="z only")
    ax.barh(y, iqr_only, left=both + z_only, color=_FLAG_COLORS["iqr"], label="IQR only")
    ax.set_yticks(y)
    ax.set_yticklabels(s["sample"], fontsize=8)
    ax.set_xlabel("number of flagged metrics")
    ax.set_title("Per-sample outlier tally", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_metric_strips(comparison, sample_cols, flags=None, *,
                       metrics=None, ncols: int = 4, figsize=None):
    """Per-metric dot plot of values across samples (default: flagged metrics only)."""
    present = list(dict.fromkeys(comparison["metric"]))
    if metrics is None:
        metrics = (list(dict.fromkeys(flags["metric"]))
                   if (flags is not None and not flags.empty) else present)
    metrics = [m for m in metrics if m in set(present)]
    if not metrics:
        fig = Figure(figsize=(6, 1.5))
        ax = fig.subplots()
        ax.text(0.5, 0.5, "No flagged metrics to plot", ha="center", va="center")
        ax.axis("off")
        return fig
    flag_set = set() if (flags is None or flags.empty) else set(zip(flags["metric"], flags["sample"]))
    row = comparison.drop_duplicates("metric").set_index("metric")
    n = len(metrics)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig = Figure(figsize=figsize or (3.3 * ncols, 2.5 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    x = np.arange(len(sample_cols))
    for k, m in enumerate(metrics):
        ax = axes[k // ncols][k % ncols]
        vals = pd.to_numeric(row.loc[m, sample_cols], errors="coerce")
        med = np.nanmedian(vals.to_numpy(dtype=float))
        ax.axhline(med, color="grey", lw=0.8, ls="--")
        for xi, sc in zip(x, sample_cols):
            on = (m, sc) in flag_set
            ax.scatter([xi], [vals[sc]], s=46 if on else 24, zorder=3,
                       color=_PT_RED if on else _PT_BLUE,
                       edgecolor="black" if on else "none", linewidths=0.6)
        ax.set_title(m, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sample_cols, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.margins(x=0.15)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Flagged metrics — values across samples  (red = flagged, dashed = median)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig