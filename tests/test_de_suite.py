"""Validation of the DE dispatcher (differential_abundance + de_concordance)."""
from pathlib import Path

import numpy as np
import pandas as pd

from ptero.quant import (
    QuantData,
    classify_missingness,
    complete_case_matrix,
    de_concordance,
    differential_abundance,
    limma_de,
)

A = [f"A{i}" for i in range(5)]
B = [f"B{i}" for i in range(5)]
GROUPS = {**{c: "PS19" for c in A}, **{c: "WT" for c in B}}


def _qd():
    rng = np.random.default_rng(3)
    rows = {}
    for j in range(500):
        L = rng.uniform(12, 26)
        p = 0.15 + 0.85 / (1.0 + np.exp(-(L - 18.0)))
        rows[f"bg{j}"] = {c: (2.0 ** (L + rng.normal(0, 0.2)) if rng.random() < p else 0.0)
                          for c in A + B}
    # a few on/off and sparse so the classifier exercises exclusion
    rows["onoff"] = {**{c: 2.0 ** 24 for c in A}, **{c: 0.0 for c in B}}
    lfq = pd.DataFrame(rows).T.replace(0.0, np.nan)
    lfq.index.name = "id"
    prot = pd.DataFrame(index=lfq.index)
    prot.index.name = "id"
    prot["gene"] = lfq.index
    prot["label"] = lfq.index
    prot["majority_protein_ids"] = lfq.index
    qd = QuantData(lfq=lfq, intensity=lfq.copy(),
                   id_type=pd.DataFrame(index=lfq.index, columns=A + B, dtype=object),
                   proteins=prot, samples=A + B, groups=dict(GROUPS), source=Path("syn"))
    return qd, classify_missingness(qd, on_min=4)


def test_dispatcher_shape_and_attrs():
    qd, diag = _qd()
    long = differential_abundance(qd, diag, ref_group="WT")
    assert set(long.columns) == {"id", "label", "track", "method", "log2FC", "AveExpr", "p", "padj"}
    assert set(long["method"]) == {"limma", "welch", "wilcoxon"}
    assert set(long["track"]) == {"complete_case", "imputed_mechanism", "imputed_minprob"}
    assert long.attrs["ref_group"] == "WT"
    assert long.attrs["contrast"] == "PS19_vs_WT"


def test_dispatcher_matches_standalone_limma():
    qd, diag = _qd()
    long = differential_abundance(qd, diag, methods=("limma",), tracks=("complete_case",),
                                  ref_group="WT")
    M = complete_case_matrix(qd, diag)
    direct = limma_de(M, qd.groups, ref_group="WT")
    j = long.set_index("id").loc[direct.index]
    assert np.allclose(j["log2FC"].to_numpy(), direct["log2FC"].to_numpy())
    assert np.allclose(j["padj"].to_numpy(), direct["padj"].to_numpy())


def test_imputed_tracks_test_more_proteins():
    qd, diag = _qd()
    long = differential_abundance(qd, diag, methods=("limma",), ref_group="WT")
    n = long.groupby("track")["id"].nunique()
    assert n["imputed_mechanism"] >= n["complete_case"]
    assert n["imputed_minprob"] >= n["complete_case"]
    # on/off protein is never in any DE track
    assert "onoff" not in set(long["id"])


def test_concordance_table():
    qd, diag = _qd()
    long = differential_abundance(qd, diag, ref_group="WT")
    summ = de_concordance(long, alpha=0.05)
    assert list(summ.index) == ["complete_case", "imputed_mechanism", "imputed_minprob"]
    assert {"limma", "welch", "wilcoxon", "all_methods", "n_proteins"} <= set(summ.columns)
    assert (summ["all_methods"] <= summ[["limma", "welch", "wilcoxon"]].min(axis=1)).all()


def test_bad_method_and_track_rejected():
    import pytest
    qd, diag = _qd()
    with pytest.raises(ValueError, match="unknown method"):
        differential_abundance(qd, diag, methods=("limma", "deseq"), ref_group="WT")
    with pytest.raises(ValueError, match="unknown track"):
        differential_abundance(qd, diag, tracks=("raw",), ref_group="WT")