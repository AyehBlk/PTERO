"""Tests for the proDA subprocess wrapper (ptero.quant._proda).

The wrapper calls the real R proDA; these tests validate the Python side -- input
guards, the missing-R error, and results parsing -- by mocking the subprocess, so
they need no R installed. The numeric acceptance test (PTERO's output == a direct R
proDA run) is done by the user with R available.
"""
import numpy as np
import pandas as pd
import pytest

from ptero.quant import _proda


def _matrix():
    idx = pd.Index([f"p{i}" for i in range(20)], name="id")
    A = [f"A{i}" for i in range(4)]
    B = [f"B{i}" for i in range(4)]
    rng = np.random.default_rng(0)
    X = rng.normal(20, 1, (20, 8))
    X[rng.random((20, 8)) < 0.1] = np.nan          # missing kept (proDA models it)
    M = pd.DataFrame(X, index=idx, columns=A + B)
    return M, {**{c: "PS" for c in A}, **{c: "WT" for c in B}}


def test_two_group_guard():
    M, g = _matrix()
    with pytest.raises(ValueError, match="2 groups"):
        _proda.proda_de(M, {**g, "A0": "Z"}, ref_group="WT", rscript="Rscript")


def test_ref_group_guard():
    M, g = _matrix()
    with pytest.raises(ValueError, match="ref_group"):
        _proda.proda_de(M, g, ref_group="NOPE", rscript="Rscript")


def test_missing_r_raises_clear_error(monkeypatch):
    monkeypatch.setattr(_proda, "_resolve_rscript", lambda rscript="Rscript": None)
    M, g = _matrix()
    with pytest.raises(RuntimeError, match="not found on PATH"):
        _proda.proda_de(M, g, ref_group="WT")


def test_r_available_false_when_missing(monkeypatch):
    monkeypatch.setattr(_proda, "_resolve_rscript", lambda rscript="Rscript": None)
    ok, msg = _proda.r_available()
    assert ok is False and "proDA" in msg


def test_parses_results_into_schema(monkeypatch):
    M, g = _matrix()
    monkeypatch.setattr(_proda.shutil, "which", lambda x: "/usr/bin/Rscript")

    def fake_run(cmd, capture_output, text, timeout):
        out_file = cmd[-2]
        ids = list(M.index)
        res = pd.DataFrame({
            "name": ids, "diff": np.linspace(-2, 2, len(ids)),
            "t_statistic": np.linspace(-3, 3, len(ids)), "se": np.full(len(ids), 0.3),
            "pval": np.linspace(1e-3, 0.9, len(ids)), "adj_pval": np.linspace(1e-2, 0.95, len(ids)),
            "df": np.full(len(ids), 6.0), "avg_abundance": np.full(len(ids), 20.0)})
        res.to_csv(out_file, sep="\t", index=False)

        class P:
            stdout = "PRODA_OK coef=conditionPS rver=4.4.1 pver=1.20.0 seed=1\n"
            stderr = ""
            returncode = 0
        return P()

    monkeypatch.setattr(_proda.subprocess, "run", fake_run)
    res = _proda.proda_de(M, g, ref_group="WT")
    assert list(res.columns) == ["log2FC", "AveExpr", "t", "se", "p", "padj", "df_total"]
    assert res.attrs["contrast"] == "PS_vs_WT"
    assert res.attrs["proda_version"] == "1.20.0"
    assert res.attrs["r_version"] == "4.4.1"
    assert len(res) == len(M)
    assert np.isclose(res["log2FC"].iloc[0], -2.0)


def test_r_error_surfaced(monkeypatch):
    M, g = _matrix()
    monkeypatch.setattr(_proda.shutil, "which", lambda x: "/usr/bin/Rscript")

    def fake_run(cmd, capture_output, text, timeout):
        class P:
            stdout = "PRODA_ERROR proDA package not installed\n"
            stderr = ""
            returncode = 3
        return P()

    monkeypatch.setattr(_proda.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="proDA .R. failed"):
        _proda.proda_de(M, g, ref_group="WT")

def test_infinite_matrix_value_rejected(monkeypatch):
    M, g = _matrix()
    M.iloc[0, 0] = np.inf

    # R should never be needed for an invalid input matrix.
    monkeypatch.setattr(
        _proda,
        "_resolve_rscript",
        lambda rscript="Rscript": "/usr/bin/Rscript",
    )

    with pytest.raises(ValueError, match="infinite"):
        _proda.proda_de(M, g, ref_group="WT")

def test_rejects_missing_protein_rows_from_r(monkeypatch):
    M, g = _matrix()
    monkeypatch.setattr(
        _proda,
        "_resolve_rscript",
        lambda rscript="Rscript": "/usr/bin/Rscript",
    )

    def fake_run(cmd, capture_output, text, timeout):
        out_file = cmd[-2]

        # Deliberately omit the final protein returned by R.
        ids = list(M.index[:-1])
        n = len(ids)

        res = pd.DataFrame({
            "name": ids,
            "diff": np.zeros(n),
            "t_statistic": np.zeros(n),
            "se": np.ones(n),
            "pval": np.ones(n),
            "adj_pval": np.ones(n),
            "df": np.full(n, 6.0),
            "avg_abundance": np.full(n, 20.0),
        })
        res.to_csv(out_file, sep="\t", index=False)

        class P:
            stdout = (
                "PRODA_OK coef=conditionPS "
                "rver=4.4.1 pver=1.20.0 seed=1\n"
            )
            stderr = ""
            returncode = 0

        return P()

    monkeypatch.setattr(_proda.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="protein"):
        _proda.proda_de(M, g, ref_group="WT")

def test_rejects_nonfinite_numeric_results_from_r(monkeypatch):
    M, g = _matrix()
    monkeypatch.setattr(
        _proda,
        "_resolve_rscript",
        lambda rscript="Rscript": "/usr/bin/Rscript",
    )

    def fake_run(cmd, capture_output, text, timeout):
        out_file = cmd[-2]
        ids = list(M.index)
        n = len(ids)

        res = pd.DataFrame({
            "name": ids,
            "diff": np.zeros(n),
            "t_statistic": np.zeros(n),
            "se": np.ones(n),
            "pval": np.ones(n),
            "adj_pval": np.ones(n),
            "df": np.full(n, 6.0),
            "avg_abundance": np.full(n, 20.0),
        })

        # Deliberately corrupt one returned statistic.
        res.loc[0, "diff"] = np.inf
        res.to_csv(out_file, sep="\t", index=False)

        class P:
            stdout = (
                "PRODA_OK coef=conditionPS "
                "rver=4.4.1 pver=1.20.0 seed=1\n"
            )
            stderr = ""
            returncode = 0

        return P()

    monkeypatch.setattr(_proda.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="non-finite"):
        _proda.proda_de(M, g, ref_group="WT")