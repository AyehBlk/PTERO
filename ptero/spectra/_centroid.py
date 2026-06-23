"""
ptero.spectra._centroid -- detect whether spectra are centroided or profile, and
record that verdict authoritatively in a store.

Why this is not trivial for 4D data
------------------------------------
A naive "median m/z spacing" check works for an ordinary (3D) spectrum, but a PTERO
4D spectrum is a whole frame (MS1) or a (frame, precursor) group (MS2): every
detector event across MANY ion-mobility (1/K0) scans is pooled into one m/z array.
Measuring spacing on that POOLED array conflates "how many mobility scans got
pooled" with "is each scan profile or centroided" -- an MS1 frame pools hundreds of
scans, so its m/z axis looks dense even when each scan is centroided.

So for 4D data the trustworthy measurement is WITHIN A SINGLE MOBILITY SCAN: fix one
1/K0 value (one TOF push) and look at the m/z spacing of just those peaks. Profile
data has a tiny, regular spacing (adjacent TOF bins, ~0.001-0.01 Da); centroided
data is sparse (one point per peak, larger spacing). PTERO stores per-peak ion
mobility, so a single scan is isolated exactly by grouping on the mobility value.

For 3D data (no per-peak mobility, e.g. an mzML / Orbitrap run) the spectrum's own
m/z array is the right thing to measure, so the pooled measurement is used directly.
When per-peak mobility is effectively continuous (e.g. synthetic data that does not
model discrete TIMS scans), no slice has enough peaks and the measurement falls back
to the pooled array -- flagged in ``basis`` so the fallback is never silent.

The 0.01 Da cutoff is a heuristic; the definitive status is the instrument/file
metadata. :func:`detect_centroided` therefore prefers a file-declared flag (e.g. an
mzML ``centroid spectrum`` cvParam, which PTERO records in the ``centroided``
column) and only measures when none is available.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

#: median m/z spacing (Da) above which a spectrum looks centroided (sparse).
CENTROID_CUTOFF_DA = 0.01


def _pooled_median_spacing(mz: np.ndarray, min_peaks: int) -> float:
    s = np.unique(mz[mz > 0])
    if s.size < min_peaks:
        return float("nan")
    return float(np.median(np.diff(s)))


def _within_slice_median_spacing(mz: np.ndarray, im: np.ndarray, min_peaks: int) -> float:
    """Median consecutive-m/z spacing within the most-populated single 1/K0 slice.

    Groups peaks by exact mobility value -- exact for real timsTOF (all events in a
    TIMS scan share one 1/K0). Returns NaN when the densest slice has too few peaks
    (e.g. continuous synthetic mobility, where every peak is its own 'slice')."""
    if im is None or im.size == 0:
        return float("nan")
    vals, inv, counts = np.unique(im, return_inverse=True, return_counts=True)
    densest = int(np.argmax(counts))
    s = np.unique(mz[inv == densest])
    s = s[s > 0]
    if s.size < min_peaks:
        return float("nan")
    return float(np.median(np.diff(s)))


def measure_centroided(sp, *, ms_level: Optional[int] = None, n_check: int = 8,
                       min_peaks_slice: int = 10, min_peaks_pooled: int = 10) -> dict:
    """Empirically classify a view as centroided vs profile from m/z spacing.

    For 4D data measures WITHIN a single 1/K0 slice (the correct basis), falling back
    to the pooled m/z array when slices are too sparse; for 3D data measures the
    pooled array directly.

    Returns ``{centroided, median_mz_spacing, basis, n_spectra_checked, ms_level}``
    where ``basis`` is one of ``within_slice_1overK0`` / ``pooled_mz`` /
    ``pooled_mz_fallback`` / ``insufficient_data``. With no usable data the verdict
    defaults to centroided=True (vacuously), matching prior behaviour.
    """
    view = sp if ms_level is None else sp.filter_ms_level(ms_level)
    n = len(view)
    result = {"median_mz_spacing": float("nan"), "centroided": True,
              "basis": "insufficient_data", "n_spectra_checked": 0, "ms_level": ms_level}
    if n == 0:
        return result

    idx = np.unique(np.linspace(0, n - 1, min(n_check, n)).astype(int))
    use_im = bool(view.has_4d)
    within, pooled = [], []
    for i in idx:
        if use_im:
            P = view.peaks_data([int(i)], ("mz", "ion_mobility"))[0]
            mz, im = P[:, 0], P[:, 1]
            w = _within_slice_median_spacing(mz, im, min_peaks_slice)
            if np.isfinite(w):
                within.append(w)
        else:
            mz = view.mz([int(i)])[0]
        p = _pooled_median_spacing(mz, min_peaks_pooled)
        if np.isfinite(p):
            pooled.append(p)

    if use_im and len(within) >= max(1, len(idx) // 2):
        med, basis, nchk = float(np.median(within)), "within_slice_1overK0", len(within)
    elif pooled:
        med = float(np.median(pooled))
        basis = "pooled_mz_fallback" if use_im else "pooled_mz"
        nchk = len(pooled)
    else:
        return result  # nothing measurable

    result.update(median_mz_spacing=med, centroided=bool(med > CENTROID_CUTOFF_DA),
                  basis=basis, n_spectra_checked=nchk)
    return result


def _metadata_verdict(sp, ms_level: Optional[int]):
    """Return True/False from a consistent file-declared 'centroided' flag, or None."""
    try:
        col = sp.spectra_data(["centroided"])["centroided"]
    except Exception:
        return None
    if ms_level is not None:
        try:
            lvl = sp.spectra_data(["ms_level"])["ms_level"]
            col = col[lvl.to_numpy() == ms_level]
        except Exception:
            return None
    col = col.dropna()
    if len(col) == 0:
        return None
    vals = col.astype(bool)
    if vals.nunique() == 1:                      # all spectra agree -> trust it
        return bool(vals.iloc[0])
    return None                                   # mixed -> fall through to measuring


def detect_centroided(sp, *, ms_level: Optional[int] = None, n_check: int = 8,
                      prefer_metadata: bool = True) -> dict:
    """Best available centroided/profile answer for a view.

    Trusts the file's own ``centroided`` metadata when present and self-consistent
    (e.g. an mzML cvParam, which the mzML backend records), and only measures the
    data empirically when no such flag exists (e.g. a Bruker ``.d``-derived store).
    The returned dict adds ``source`` = ``file_metadata`` or ``measured``.
    """
    if prefer_metadata:
        v = _metadata_verdict(sp, ms_level)
        if v is not None:
            return {"centroided": v, "median_mz_spacing": float("nan"),
                    "basis": "file_metadata", "source": "file_metadata",
                    "n_spectra_checked": 0, "ms_level": ms_level}
    out = measure_centroided(sp, ms_level=ms_level, n_check=n_check)
    out["source"] = "measured"
    return out


def annotate_store_centroided(store_path, *, n_check: int = 8, force: bool = False) -> dict:
    """Detect centroided/profile per MS level and stamp it into the store once.

    Writes the per-spectrum ``centroided`` column in ``spectra.parquet`` (per level)
    and a ``centroid_detection`` provenance attribute on the store. Peaks are not
    touched. Idempotent: a store that already carries a self-consistent ``centroided``
    flag (e.g. from mzML metadata) is left as-is unless ``force=True``.

    Parameters
    ----------
    store_path : str | Path
        A PTERO Zarr store directory.
    n_check : int
        Spectra sampled per level for the empirical measurement.
    force : bool
        Re-measure and overwrite even if a usable flag is already present.

    Returns
    -------
    dict
        ``{"action": "annotated"|"kept_existing", "store": ..., "levels": {level: ...}}``.
    """
    from .io import open_zarr
    import zarr

    store_path = Path(str(store_path))
    parquet = store_path / "spectra.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"No spectra.parquet in {store_path}. Is this a PTERO Zarr store?"
        )
    df = pd.read_parquet(parquet)

    has_usable = ("centroided" in df.columns and df["centroided"].notna().any()
                  and df["centroided"].dropna().astype(bool).nunique() >= 1)
    if has_usable and not force:
        per = {}
        if "ms_level" in df.columns:
            for lvl, g in df.dropna(subset=["ms_level"]).groupby("ms_level"):
                vals = g["centroided"].dropna()
                per[int(lvl)] = None if not len(vals) else bool(vals.astype(bool).mode().iloc[0])
        return {"action": "kept_existing", "store": str(store_path), "levels": per}

    sp = open_zarr(str(store_path))
    if "ms_level" not in df.columns:
        df["ms_level"] = sp.spectra_data(["ms_level"])["ms_level"].to_numpy()
    if "centroided" not in df.columns:
        df["centroided"] = pd.NA

    levels = sorted(int(x) for x in pd.unique(df["ms_level"].dropna()))
    detection = {}
    for lvl in levels:
        d = measure_centroided(sp, ms_level=lvl, n_check=n_check)
        detection[lvl] = d
        if d["basis"] != "insufficient_data" and d["centroided"] is not None:
            df.loc[df["ms_level"] == lvl, "centroided"] = bool(d["centroided"])

    df["centroided"] = df["centroided"].astype("boolean")
    df.to_parquet(parquet, index=False)

    root = zarr.open_group(str(store_path), mode="r+")
    root.attrs["centroid_detection"] = {
        "cutoff_da": CENTROID_CUTOFF_DA,
        "n_check": int(n_check),
        "measured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "levels": {str(k): {"centroided": v["centroided"],
                            "median_mz_spacing": v["median_mz_spacing"],
                            "basis": v["basis"],
                            "n_spectra_checked": v["n_spectra_checked"]}
                   for k, v in detection.items()},
    }
    return {"action": "annotated", "store": str(store_path),
            "levels": {k: {"centroided": v["centroided"],
                           "median_mz_spacing": v["median_mz_spacing"],
                           "basis": v["basis"]} for k, v in detection.items()}}