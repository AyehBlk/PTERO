"""
ptero.quant._proda -- model-based differential abundance via the real R proDA.

proDA (Ahlmann-Eltze & Anders 2020) fits a probabilistic-dropout linear model:
missing values are not imputed, they enter the likelihood through an intensity-
dependent dropout probability, with empirical-Bayes variance moderation. It is the
principled, peer-reviewed answer to missingness in label-free proteomics.

Rather than reimplement proDA (which would need its own validation and could not be
cited as proDA), PTERO **calls the canonical R package** through a small, robust
subprocess bridge: it writes the matrix + design to temp files, runs a bundled R
script with ``Rscript``, and parses the results back into PTERO's DE schema. Your
numbers are exactly proDA's, and PTERO records the R and proDA versions used.

This makes proDA an **optional** capability: it is available only if R and the
proDA package are installed (:func:`r_available`). Everything else in
``ptero.quant`` stays pure-Python; if R is absent these functions raise a clear,
actionable error and the pure-Python methods remain the baseline.

Note proDA consumes the **full normalized log2 matrix, missing values included** --
it is *not* run on a complete-case or imputed matrix. It replaces the impute-then-
test workflow; do not impute or drop before calling it.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _script_path() -> Path:
    return Path(__file__).with_name("proda_fit.R")


def _r_search_roots():
    """Standard Windows R install roots to search when Rscript isn't on PATH."""
    roots = []
    if os.name == "nt":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        roots += [os.path.join(pf, "R"), os.path.join(pf86, "R")]
        if local:
            roots.append(os.path.join(local, "Programs", "R"))
    return roots


def _rver_key(path):
    m = re.search(r"R-(\d+)\.(\d+)\.(\d+)", path)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def _find_rscript_windows(roots=None):
    """Newest Rscript.exe found under the given (or standard) Windows R roots."""
    roots = _r_search_roots() if roots is None else roots
    cands = []
    for root in roots:
        for pat in ("R-*/bin/x64/Rscript.exe", "R-*/bin/Rscript.exe",
                    "*/bin/x64/Rscript.exe", "*/bin/Rscript.exe"):
            cands += glob.glob(os.path.join(root, pat))
    cands = sorted(set(cands), key=_rver_key, reverse=True)
    return cands[0] if cands else None


def _resolve_rscript(rscript: str = "Rscript"):
    """Locate the R front-end: explicit path -> PTERO_RSCRIPT env -> PATH -> Windows
    standard install locations. Returns the executable path, or None."""
    if rscript and (os.sep in rscript or "/" in rscript) and os.path.isfile(rscript):
        return rscript
    env = os.environ.get("PTERO_RSCRIPT")
    if env and os.path.isfile(env):
        return env
    found = shutil.which(rscript)
    if found:
        return found
    if os.name == "nt":
        return _find_rscript_windows()
    return None


def r_available(rscript: str = "Rscript") -> tuple[bool, str]:
    """Return ``(ok, message)``: whether ``Rscript`` and the proDA package are usable.

    ``ok`` is True only if the R executable is found *and* ``library(proDA)`` loads.
    On failure ``message`` explains what's missing (for the UI / error text).
    """
    exe = _resolve_rscript(rscript)
    if not exe:
        return False, ("R (Rscript) not found on PATH or in the standard install locations. "
                       "Install R (https://www.r-project.org) with the Bioconductor package "
                       "proDA, or set the PTERO_RSCRIPT environment variable to your Rscript "
                       "executable.")
    try:
        r = subprocess.run(
            [exe, "--vanilla", "-e", 'cat(requireNamespace("proDA", quietly=TRUE))'],
            capture_output=True, text=True, timeout=120)
    except Exception as e:                                       # noqa: BLE001
        return False, f"could not run '{rscript}': {e}"
    if "TRUE" in r.stdout:
        return True, ""
    return False, ("R was found but the proDA package is not installed. In R run: "
                   'if (!require("BiocManager")) install.packages("BiocManager"); '
                   'BiocManager::install("proDA")')


def proda_version(rscript: str = "Rscript") -> str | None:
    """The installed proDA package version string, or None if unavailable."""
    exe = _resolve_rscript(rscript)
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--vanilla", "-e",
             'if(requireNamespace("proDA",quietly=TRUE)) cat(as.character(packageVersion("proDA")))'],
            capture_output=True, text=True, timeout=120)
        v = r.stdout.strip()
        return v or None
    except Exception:                                           # noqa: BLE001
        return None

def proda_de(
    matrix: pd.DataFrame,
    groups,
    *,
    ref_group: str,
    rscript: str = "Rscript",
    seed: int = 1,
    timeout: int = 1800,
) -> pd.DataFrame:
    """Differential abundance with the real R proDA (probabilistic dropout model).

    Parameters
    ----------
    matrix : DataFrame
        proteins x samples, **normalized log2 intensities with missing values kept
        as NaN** (e.g. ``log2_median_normalize(qd.lfq)``). Do not impute or drop --
        proDA models the missingness.
    groups : mapping {sample -> group}
        Two-level group factor. ``ref_group`` is the baseline; a positive log2FC
        means higher in the other group (matching the rest of PTERO).
    ref_group : str
    rscript : str
        Name/path of the R front-end executable (default ``"Rscript"``).
    seed : int
        Passed to R ``set.seed`` before fitting. proDA initialises with random
        draws, so a fixed seed makes results reproducible run-to-run (default 1).
    timeout : int
        Seconds before the R call is aborted.

    Returns
    -------
    DataFrame indexed by protein id with ``log2FC``, ``AveExpr``, ``t``, ``se``,
    ``p``, ``padj`` (BH, from proDA), ``df_total``. ``.attrs`` records the method,
    contrast, and the R / proDA versions used (for the methods section).

    Raises
    ------
    ValueError
        If the input matrix contains infinite values, the group factor does not
        have exactly two levels, or ``ref_group`` is not one of those levels.
    RuntimeError
        If R or proDA is unavailable, if the R call fails, if the returned protein
        identities do not match the input matrix, or if proDA returns infinite
        numerical results.
    """
    samples = list(matrix.columns)
    g = {s: str(groups[s]) for s in samples}
    uniq = list(dict.fromkeys(g.values()))

    if len(uniq) != 2:
        raise ValueError(f"proda_de needs exactly 2 groups; got {uniq}.")
    if ref_group not in uniq:
        raise ValueError(f"ref_group {ref_group!r} not among groups {uniq}.")

    Y = matrix.to_numpy(dtype=float)
    if np.isinf(Y).any():
        raise ValueError(
            "proda_de allows NaN values because proDA models missingness, "
            "but infinite values are not allowed."
        )

    exe = _resolve_rscript(rscript)
    if not exe:
        raise RuntimeError(r_available(rscript)[1])

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mfile, dfile, ofile = td / "matrix.tsv", td / "design.tsv", td / "out.tsv"

        # index (protein id) becomes the first column; NaN -> "NA" for R
        matrix.to_csv(mfile, sep="\t", na_rep="NA")
        pd.DataFrame(
            {
                "sample": samples,
                "group": [g[s] for s in samples],
            }
        ).to_csv(dfile, sep="\t", index=False)

        proc = subprocess.run(
            [
                exe,
                "--vanilla",
                str(_script_path()),
                str(mfile),
                str(dfile),
                ref_group,
                str(ofile),
                str(int(seed)),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        tail = (
            proc.stdout.strip() + "\n" + proc.stderr.strip()
        ).strip()

        if (
            "PRODA_ERROR" in proc.stdout
            or not ofile.exists()
            or proc.returncode != 0
        ):
            raise RuntimeError(
                f"proDA (R) failed:\n{tail[:1500]}"
            )

        res = pd.read_csv(ofile, sep="\t")

    if res.empty:
        raise RuntimeError("proDA returned no rows.")

    if "name" not in res.columns:
        raise RuntimeError(
            "proDA result is missing the protein identifier column 'name'."
        )

    input_names = pd.Index(matrix.index.astype(str))
    result_names = pd.Index(res["name"].astype(str))

    if (
        len(result_names) != len(input_names)
        or result_names.has_duplicates
        or set(result_names) != set(input_names)
    ):
        raise RuntimeError(
            "proDA returned a protein set that does not match the input matrix."
        )

    numeric_columns = [
        "diff",
        "avg_abundance",
        "t_statistic",
        "se",
        "pval",
        "adj_pval",
        "df",
    ]

    try:
        numeric_results = res[numeric_columns].to_numpy(dtype=float)
    except (KeyError, ValueError, TypeError) as e:
        raise RuntimeError(
            "proDA returned an invalid result schema."
        ) from e

    if np.isinf(numeric_results).any():
        raise RuntimeError(
            "proDA returned non-finite infinite values in its numerical results."
        )

    out = pd.DataFrame(
        {
            "log2FC": res["diff"].to_numpy(dtype=float),
            "AveExpr": res["avg_abundance"].to_numpy(dtype=float),
            "t": res["t_statistic"].to_numpy(dtype=float),
            "se": res["se"].to_numpy(dtype=float),
            "p": res["pval"].to_numpy(dtype=float),
            "padj": res["adj_pval"].to_numpy(dtype=float),
            "df_total": res["df"].to_numpy(dtype=float),
        }
    )

    # align the index dtype to the input matrix where possible
    names = res["name"].to_numpy()
    try:
        names = names.astype(matrix.index.dtype)
    except (ValueError, TypeError):
        pass

    out.index = pd.Index(
        names,
        name=matrix.index.name or "id",
    )

    non_ref = next(x for x in uniq if x != ref_group)
    out.attrs["method"] = "proda"
    out.attrs["contrast"] = f"{non_ref}_vs_{ref_group}"

    for line in proc.stdout.splitlines():
        if line.startswith("PRODA_OK"):
            for tok in line.split()[1:]:
                if tok.startswith("rver="):
                    out.attrs["r_version"] = tok[5:]
                elif tok.startswith("pver="):
                    out.attrs["proda_version"] = tok[5:]
                elif tok.startswith("seed="):
                    out.attrs["seed"] = int(tok[5:])

    return out