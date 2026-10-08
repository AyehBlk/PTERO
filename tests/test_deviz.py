"""Tests for the DE visualisation module (ptero.quant._deviz).

volcano() is quantitative-only (on/off candidates are shown by detection_matrix,
not as diamonds). Plotting is validated at the smoke level: the functions build a
Figure via the object API without error, handle the edge cases, and the
label/contrast helpers behave.
"""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ptero.quant import (QuantData, detection_matrix, heatmap_top,
                         plot_de_boxplot, volcano)
from ptero.quant._deviz import _contrast_labels, _short_label

A = [f"A{i}" for i in range(3)]     # PS
B = [f"B{i}" for i in range(3)]     # WT


def _de(n=400, seed=0):
    rng = np.random.default_rng(seed)
    lfc = rng.normal(0, 1.2, n)
    p = rng.uniform(0, 1, n)
    p[:20] = rng.uniform(1e-8, 1e-4, 20)
    lfc[:20] = rng.choice([-1, 1], 20) * rng.uniform(1.5, 3.0, 20)
    d = pd.DataFrame({"log2FC": lfc, "p": p,
                      "padj": np.clip(p * n / (np.argsort(np.argsort(p)) + 1), 0, 1),
                      "label": [f"G{i}" for i in range(n)]})
    d.index.name = "id"
    d.attrs["contrast"] = "PS_vs_WT"
    d.attrs["method"] = "limma"
    return d


def _qd_and_pa():
    idx = pd.Index([f"p{i}" for i in range(6)], name="id")
    lfq = pd.DataFrame(index=idx, columns=A + B, dtype=float)
    rng = np.random.default_rng(0)

    def fill(ids, cols, lvl):
        for i in ids:
            for c in cols:
                lfq.loc[i, c] = 2.0 ** (lvl + rng.normal(0, 0.2))

    fill(["p0", "p1"], A, 24)          # PS-on
    fill(["p2", "p3"], B, 22)          # WT-on
    fill(["p4", "p5"], A + B, 20)      # detected in both
    qd = QuantData(lfq=lfq, intensity=lfq.copy(),
                   id_type=pd.DataFrame(index=idx, columns=A + B, dtype=object),
                   proteins=pd.DataFrame({"gene": [f"G{i}" for i in range(6)],
                                          "label": [f"G{i}" for i in range(6)]}, index=idx),
                   samples=A + B, groups={**{c: "PS" for c in A}, **{c: "WT" for c in B}},
                   source="syn")
    pa = pd.DataFrame({
        "label": [f"G{i}" for i in range(4)],
        "enriched_in": ["PS", "PS", "WT", "WT"],
        "near_floor": [False, True, False, True],
        "call_ready": [True, True, True, True],
    }, index=pd.Index([f"p{i}" for i in range(4)], name="id"))
    return qd, pa


# ── volcano ──────────────────────────────────────────────────────────────────

def test_volcano_returns_figure_with_point_groups():
    fig = volcano(_de())
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].collections) >= 3     # up, down, not-significant


def test_volcano_computes_padj_if_absent():
    assert isinstance(volcano(_de().drop(columns=["padj"])), Figure)


def test_volcano_with_labels_arg_when_no_label_column():
    d = _de().drop(columns=["label"])
    assert isinstance(volcano(d, labels={i: f"P{i}" for i in d.index}, label_top=5), Figure)


def test_volcano_all_nonsignificant_ok():
    d = _de()
    d["padj"] = 1.0
    assert isinstance(volcano(d), Figure)


def test_volcano_missing_columns_raises():
    with pytest.raises(ValueError, match="log2FC"):
        volcano(pd.DataFrame({"p": [0.1, 0.2]}))


# ── detection_matrix ─────────────────────────────────────────────────────────

def test_detection_matrix_returns_figure():
    qd, pa = _qd_and_pa()
    fig = detection_matrix(qd, pa, which="call_ready", ref_group="WT")
    assert isinstance(fig, Figure)
    # a colorbar axis is added alongside the main axes
    assert len(fig.axes) >= 2
    assert len(fig.axes[0].collections) > 0


def test_detection_matrix_respects_ref_group_order():
    qd, pa = _qd_and_pa()
    # both orderings render without error
    assert isinstance(detection_matrix(qd, pa, ref_group="WT"), Figure)
    assert isinstance(detection_matrix(qd, pa, ref_group="PS"), Figure)


def test_detection_matrix_max_proteins_caps_rows():
    qd, pa = _qd_and_pa()
    fig = detection_matrix(qd, pa, which="all", max_proteins=2)
    assert isinstance(fig, Figure)


def test_detection_matrix_empty_selection_raises():
    qd, pa = _qd_and_pa()
    pa2 = pa.copy()
    pa2["call_ready"] = False
    with pytest.raises(ValueError, match="no proteins"):
        detection_matrix(qd, pa2, which="call_ready")


def test_detection_matrix_needs_two_groups():
    qd, pa = _qd_and_pa()
    qd.groups = {s: "ALL" for s in qd.samples}
    with pytest.raises(ValueError, match="2 groups"):
        detection_matrix(qd, pa)


# ── helpers ──────────────────────────────────────────────────────────────────

def test_short_label():
    assert _short_label("Mapt;tr|A0A0A0MQC7|A0A0A0MQC7_MOUSE") == "Mapt"
    assert _short_label("A" * 40).endswith("\u2026")
    assert len(_short_label("A" * 40)) <= 16


def test_contrast_labels():
    up, down, up_grp, ref = _contrast_labels(_de())
    assert up_grp == "PS" and ref == "WT"
    plain = pd.DataFrame({"log2FC": [0.0], "p": [1.0]})
    _, _, g2, r2 = _contrast_labels(plain)
    assert g2 is None and r2 is None


# ── heatmap_top + plot_de_boxplot ────────────────────────────────────────────

A3 = [f"A{i}" for i in range(3)]     # PS
B3 = [f"B{i}" for i in range(3)]     # WT
G3 = {**{c: "PS" for c in A3}, **{c: "WT" for c in B3}}


def _mat_de(n=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"p{i}" for i in range(n)], name="id")
    X = rng.normal(20, 1, (n, 6))
    X[:n // 2, :3] += 2.0            # up in PS (A cols)
    X[n // 2:, 3:] += 2.0            # up in WT (B cols)
    M = pd.DataFrame(X, index=idx, columns=A3 + B3)
    lfc = np.concatenate([np.full(n // 2, 2.0), np.full(n - n // 2, -2.0)])
    de = pd.DataFrame({"log2FC": lfc, "p": rng.uniform(1e-4, 0.5, n),
                       "padj": rng.uniform(0.01, 0.9, n),
                       "label": [f"G{i}" for i in range(n)]}, index=idx)
    de.attrs["method"] = "limma"
    de.attrs["contrast"] = "PS_vs_WT"
    return M, de


def test_heatmap_returns_figure():
    M, de = _mat_de()
    fig = heatmap_top(de, M, G3, top=10, ref_group="WT")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2         # main + colorbar


def test_heatmap_by_options():
    M, de = _mat_de()
    for by in ("p", "padj", "log2FC"):
        assert isinstance(heatmap_top(de, M, G3, top=8, by=by, ref_group="WT"), Figure)


def test_heatmap_no_zscore():
    M, de = _mat_de()
    assert isinstance(heatmap_top(de, M, G3, top=6, zscore=False), Figure)


def test_heatmap_needs_two_groups():
    M, de = _mat_de()
    g = {s: "ALL" for s in M.columns}
    with pytest.raises(ValueError, match="2 groups"):
        heatmap_top(de, M, g)


def test_boxplot_returns_figure():
    M, de = _mat_de()
    assert isinstance(plot_de_boxplot(M, "p0", G3, de=de, ref_group="WT"), Figure)


def test_boxplot_without_de():
    M, de = _mat_de()
    assert isinstance(plot_de_boxplot(M, "p0", G3, label="Foo"), Figure)


def test_boxplot_missing_protein_raises():
    M, de = _mat_de()
    with pytest.raises(KeyError):
        plot_de_boxplot(M, "nope", G3)


# ── save_de_boxplots (batch) ─────────────────────────────────────────────────

def test_save_boxplots_top(tmp_path):
    from ptero.quant import save_de_boxplots
    M, de = _mat_de()
    paths = save_de_boxplots(M, de, G3, tmp_path, which="top", top=5, ref_group="WT")
    assert len(paths) == 5
    assert all(p.exists() and p.suffix == ".png" and p.name.startswith("boxplot_") for p in paths)


def test_save_boxplots_by_list(tmp_path):
    from ptero.quant import save_de_boxplots
    M, de = _mat_de()
    paths = save_de_boxplots(M, de, G3, tmp_path, proteins=["G0", "G3"], ref_group="WT")
    assert len(paths) == 2


def test_save_boxplots_significant_empty(tmp_path):
    from ptero.quant import save_de_boxplots
    M, de = _mat_de()
    de["padj"] = 1.0                       # nothing significant
    paths = save_de_boxplots(M, de, G3, tmp_path, which="significant")
    assert paths == []


def test_save_boxplots_filename_safe(tmp_path):
    from ptero.quant import save_de_boxplots
    M, de = _mat_de()
    de.loc["p0", "label"] = "Ma;pt|A0/B1"  # illegal filename chars
    paths = save_de_boxplots(M, de, G3, tmp_path, proteins=["p0"], ref_group="WT")
    assert len(paths) == 1
    assert not any(c in paths[0].name for c in ';|/\\')
    assert paths[0].exists()


# ── highlight + imputed-mask + plot_imputation ───────────────────────────────

def test_volcano_highlight():
    d = _de()
    fig = volcano(d, highlight=d.index[0])
    assert isinstance(fig, Figure)


def test_heatmap_highlight():
    M, de = _mat_de()
    fig = heatmap_top(de, M, G3, top=10, ref_group="WT", highlight=de.index[0])
    assert isinstance(fig, Figure)


def test_boxplot_imputed_mask_marks_points():
    from ptero.quant import plot_de_boxplot
    M, de = _mat_de()
    # mark two samples of p0 as imputed
    mask = pd.Series(False, index=M.columns)
    mask[[A3[0], B3[0]]] = True
    fig = plot_de_boxplot(M, "p0", G3, de=de, ref_group="WT", imputed_mask=mask)
    assert isinstance(fig, Figure)
    # legend (observed/imputed) is drawn only when imputed points exist
    assert fig.axes[0].get_legend() is not None


def test_plot_imputation_runs():
    import numpy as np
    from ptero.quant import (QuantData, classify_missingness, plot_imputation)
    A = [f"A{i}" for i in range(5)]
    B = [f"B{i}" for i in range(5)]
    rng = np.random.default_rng(0)
    rows = {}
    for j in range(300):
        L = rng.uniform(12, 26)
        p = 0.15 + 0.85 / (1 + np.exp(-(L - 18)))
        rows[f"bg{j}"] = {c: (2.0 ** (L + rng.normal(0, 0.2)) if rng.random() < p else 0.0)
                          for c in A + B}
    lfq = pd.DataFrame(rows).T.replace(0.0, np.nan)
    lfq.index.name = "id"
    prot = pd.DataFrame({"gene": lfq.index, "label": lfq.index}, index=lfq.index)
    qd = QuantData(lfq=lfq, intensity=lfq.copy(),
                   id_type=pd.DataFrame(index=lfq.index, columns=A + B, dtype=object),
                   proteins=prot, samples=A + B,
                   groups={**{c: "PS" for c in A}, **{c: "WT" for c in B}}, source="syn")
    diag = classify_missingness(qd, on_min=4)
    assert isinstance(plot_imputation(qd, diag), Figure)

def test_volcano_rejects_invalid_p_values():
    d = _de()
    d.loc[d.index[0], "p"] = -0.1

    with pytest.raises(ValueError, match="p-value"):
        volcano(d)

def test_volcano_rejects_invalid_adjusted_p_values():
    d = _de()
    d.loc[d.index[0], "padj"] = 1.1

    with pytest.raises(ValueError, match="adjusted p-value"):
        volcano(d)