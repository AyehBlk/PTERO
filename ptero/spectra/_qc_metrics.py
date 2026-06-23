"""
ptero.spectra._qc_metrics -- ID-free quality-control metrics aligned with the
HUPO-PSI mzQC controlled vocabulary (https://hupo-psi.github.io/mzQC).

Every metric here is computed from low-level data PTERO already holds: retention
time, per-scan total ion current (TIC), per-scan peak counts, and MS2 precursor
m/z / intensity / charge. None require pooling per-peak intensities, so they are
cheap and memory-safe even on profile data.

Definitions follow the MsQuality reference implementation (Naake et al., 2023,
Bioinformatics) so PTERO's numbers can be cross-checked against it. Each metric
carries its mzQC CV accession in ``MZQC_ACCESSIONS`` where confirmed; an empty
string means the accession still needs to be read from the PSI-MS CV. Accessions
are never guessed -- a wrong identifier would misrepresent the metric.

Scope: these cover DDA LC-MS performance and are vendor/acquisition agnostic.
DIA needs a parallel metric set (cf. iDIA-QC); ion-mobility (4D) metrics are a
PTERO extension with no mzQC term yet.
"""
from __future__ import annotations

import numpy as np

# metric base-name -> mzQC CV accession.  "" = confirmed-pending (do NOT guess).
MZQC_ACCESSIONS: dict[str, str] = {
    "areaUnderTic": "MS:4000155",
    "areaUnderTicRtQuantiles": "MS:4000156",
    "rtOverMsQuarters": "MS:4000055",
    "chromatographyDuration": "MS:4000053",
    "mzAcquisitionRange": "MS:4000069",  # range of MS2 *precursor* m/z (per the CV)
    "mzPeakRange": "",                   # PTERO extension: m/z extent of all peaks (no CV term)
    "rtAcquisitionRange": "MS:4000070",  # retention time acquisition range
    "numberEmptyScansMS1": "MS:4000099",
    "numberEmptyScansMS2": "MS:4000100", # number of empty MS2 scans
    "medianPrecursorMz": "",             # MS:4000152 only applies to identified data; "" for all-MS2
    "precursorIntensityMean": "MS:4000117",
    "precursorIntensitySd": "MS:4000118",
    "precursorIntensityRange": "MS:4000160",
    "precursorIntensityQuartiles": "MS:4000116",
    "meanCharge": "MS:4000173",
    "medianCharge": "MS:4000175",
    "ratioCharge1over2": "MS:4000167",
    "ratioCharge3over2": "MS:4000169",
    "ratioCharge4over2": "MS:4000171",
    # MS1 peak-intensity distribution -- PTERO extension, NOT an mzQC CV metric.
    # (MsQuality's intensityMean/Quartiles/Range/Sd are Chromatograms metrics with
    #  no CV accession; the only CV-defined intensity family is the precursor one
    #  above. These stay "" -- a CV term is never guessed.)
    "ms1IntensityMean": "",
    "ms1IntensitySd": "",
    "ms1IntensityRange": "",
    "ms1IntensityQuartiles": "",
}

_NTUPLE_SUFFIXES = ("_Q1", "_Q2", "_Q3", "_Q4", "_min", "_max")


def accession_for(column: str) -> str:
    """CV accession for a metric column, stripping any n-tuple suffix (_Q1, _min…)."""
    base = column
    for suf in _NTUPLE_SUFFIXES:
        if column.endswith(suf):
            base = column[: -len(suf)]
            break
    return MZQC_ACCESSIONS.get(base, "")


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    return a[np.isfinite(a)]


def tic_shape_metrics(rt_ms1, tic_ms1, rt_duration=None) -> dict:
    """Chromatographic-shape metrics from per-MS1-scan (retention time, TIC).

    Matches the MsQuality reference implementation:
    * areaUnderTicRtQuantiles bins by the **quantiles of the rtime values**
      (R ``quantile`` type 7 == numpy default) with half-open ``(q, q]``
      intervals; the first interval is ``rt > 0%``, so the minimum-RT scan is
      excluded (a quirk of MsQuality we replicate for comparability).
    * rtOverMsQuarters partitions scans into four near-equal groups via
      ``sort(rep(1:4, length.out=n))`` and divides by ``rt_duration`` -- the
      **whole-run** RT span (MS1+MS2), not the MS1-only span. Pass it in;
      falls back to the MS1 span if not given.
    """
    out: dict = {}
    rt = np.asarray(rt_ms1, "float64")
    tic = np.asarray(tic_ms1, "float64")
    m = np.isfinite(rt) & np.isfinite(tic)
    rt, tic = rt[m], tic[m]
    if rt.size == 0:
        return out
    order = np.argsort(rt, kind="stable")
    rt, tic = rt[order], tic[order]
    rmin, rmax = float(rt[0]), float(rt[-1])
    dur = rmax - rmin
    rtd = float(rt_duration) if (rt_duration is not None and np.isfinite(rt_duration)) else dur
    out["areaUnderTic"] = float(tic.sum())              # MS:4000155 (MS1)
    # MS:4000156 -- TIC summed within each rtime-quantile interval (min scan excluded)
    if dur > 0:
        q = np.quantile(rt, [0.0, .25, .5, .75, 1.0])
        for k, tag in enumerate(("Q1", "Q2", "Q3", "Q4")):
            sel = (rt > q[k]) & (rt <= q[k + 1])
            out[f"areaUnderTicRtQuantiles_{tag}"] = float(tic[sel].sum())
    # MS:4000055 -- RT fraction elapsed at the end of each quarter of MS1 events
    if rt.size >= 4 and rtd > 0:
        ind = np.sort(np.resize(np.arange(1, 5), rt.size))      # rep(1:4, length.out=n), sorted
        idx = np.where(np.append(np.diff(ind), 1) == 1)[0]      # last index of each group
        for tag, i in zip(("Q1", "Q2", "Q3", "Q4"), idx):
            out[f"rtOverMsQuarters_{tag}"] = float((rt[i] - rmin) / rtd)
    return out


def range_metrics(rt_all, mz_lo, mz_hi) -> dict:
    """RT acquisition range (CV) and the peak m/z extent (PTERO extension).

    Note: the CV ``mzAcquisitionRange`` (MS:4000069) is the range of MS2
    *precursor* m/z and is computed in :func:`precursor_metrics`. The m/z extent
    of all recorded peaks is a different quantity, returned here as the
    PTERO-only ``mzPeakRange`` (no CV term).
    """
    out: dict = {}
    rt = _finite(rt_all)
    if rt.size:
        out["chromatographyDuration"] = float(rt.max() - rt.min())  # MS:4000053
        out["rtAcquisitionRange_min"] = float(rt.min())             # MS:4000070
        out["rtAcquisitionRange_max"] = float(rt.max())
    if mz_lo is not None and np.isfinite(mz_lo):
        out["mzPeakRange_min"] = float(mz_lo)   # PTERO extension (no CV term)
        out["mzPeakRange_max"] = float(mz_hi)
    return out


def empty_scan_metrics(npks, lvl_s, tics=None) -> dict:
    """numberEmptyScans per MS level. A scan is empty if it has no peaks OR its
    intensities sum to zero (MsQuality treats ``sum(intensity) == 0`` as empty)."""
    out: dict = {}
    npks = np.asarray(npks)
    lvl = np.asarray(lvl_s)
    tic = np.asarray(tics, "float64") if tics is not None else None
    for lv, tag in ((1, "MS1"), (2, "MS2")):
        mask = lvl == lv
        if mask.any():
            empty = npks[mask] == 0
            if tic is not None:
                empty = empty | (tic[mask] == 0)
            out[f"numberEmptyScans{tag}"] = int(empty.sum())
    return out


def precursor_metrics(prec_mz, prec_int) -> dict:
    """MS2 precursor m/z range + median, and precursor-intensity distribution.

    Matches MsQuality: ``mzAcquisitionRange`` is ``range(precursorMz)`` of MS2;
    intensity metrics use all *finite* precursor intensities (NA removed only --
    zeros are kept, unlike a peak-intensity filter)."""
    out: dict = {}
    mz = _finite(prec_mz)
    if mz.size:
        out["mzAcquisitionRange_min"] = float(mz.min())             # MS:4000069
        out["mzAcquisitionRange_max"] = float(mz.max())
        out["medianPrecursorMz"] = float(np.median(mz))             # "" (all-MS2: no CV term)
    it = _finite(prec_int)
    if it.size:
        out["precursorIntensityMean"] = float(it.mean())            # MS:4000117
        if it.size > 1:
            out["precursorIntensitySd"] = float(it.std(ddof=1))     # MS:4000118
        out["precursorIntensityRange_min"] = float(it.min())        # MS:4000160
        out["precursorIntensityRange_max"] = float(it.max())
        q1, q2, q3 = np.quantile(it, [.25, .5, .75])                # MS:4000116
        out["precursorIntensityQuartiles_Q1"] = float(q1)
        out["precursorIntensityQuartiles_Q2"] = float(q2)
        out["precursorIntensityQuartiles_Q3"] = float(q3)
    return out


def charge_metrics(prec_charge) -> dict:
    """MS2 precursor charge distribution (all precursors).

    Matches MsQuality: a ratio ``X over 2`` is returned only when **both** charge
    X and charge 2 occur (otherwise MsQuality yields NaN -- here the key is simply
    omitted, never reported as 0)."""
    out: dict = {}
    c = _finite(prec_charge)
    if c.size == 0:
        return out
    out["meanCharge"] = float(c.mean())                             # MS:4000173
    out["medianCharge"] = float(np.median(c))                       # MS:4000175
    n2 = int((c == 2).sum())
    for z, name in ((1, "ratioCharge1over2"), (3, "ratioCharge3over2"), (4, "ratioCharge4over2")):
        nz = int((c == z).sum())
        if n2 and nz:                                               # both charges must be present
            out[name] = float(nz / n2)
    return out


class _Ms1IntensityAccumulator:
    """Streaming MS1 peak-intensity stats: exact mean/sd/min/max + reservoir quartiles.

    Memory is bounded to ``cap`` values (default 2,000,000 -> ~16 MB) regardless of
    how many MS1 peaks the run has, so this is safe on a 200M-peak profile run.
    Mean/sd/min/max are exact; quartiles come from a uniform reservoir sample
    (block-vectorised Algorithm R, so it stays fast on huge runs).
    """

    def __init__(self, cap: int = 2_000_000, seed: int = 0):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.mn = np.inf
        self.mx = -np.inf
        self._cap = int(cap)
        self._res = np.empty(self._cap, dtype="float64")
        self._fill = 0
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def update(self, vals) -> None:
        v = np.asarray(vals, dtype="float64")
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        self.n += v.size
        self.s += float(v.sum())
        self.ss += float(np.dot(v, v))
        self.mn = min(self.mn, float(v.min()))
        self.mx = max(self.mx, float(v.max()))
        # fill the reservoir first
        if self._fill < self._cap:
            t = min(self._cap - self._fill, v.size)
            self._res[self._fill:self._fill + t] = v[:t]
            self._fill += t
            self._seen += t
            v = v[t:]
            if v.size == 0:
                return
        # reservoir full: item with global rank g replaces a random slot w.p. cap/g
        m = v.size
        g = self._seen + np.arange(1, m + 1)
        keep = self._rng.random(m) < (self._cap / g)
        k = int(keep.sum())
        if k:
            slots = self._rng.integers(0, self._cap, size=k)
            self._res[slots] = v[keep]
        self._seen += m

    def summary(self):
        if self.n == 0:
            return None
        mean = self.s / self.n
        var = max(self.ss / self.n - mean * mean, 0.0)
        q1, q2, q3 = np.quantile(self._res[:self._fill], [0.25, 0.5, 0.75])
        return dict(mean=mean, sd=float(np.sqrt(var)), mn=self.mn, mx=self.mx,
                    q1=float(q1), q2=float(q2), q3=float(q3))


def ms1_intensity_metrics(summary) -> dict:
    """MS1 peak-intensity distribution (PTERO extension; no mzQC CV term).

    A dynamic-range descriptor of the MS1 signal, from the accumulator summary.
    On *profile* MS1 data the distribution is dominated by baseline points, so it
    is most meaningful on centroided MS1 (PTERO records centroid status per store).
    """
    if not summary:
        return {}
    return {
        "ms1IntensityMean": summary["mean"],
        "ms1IntensitySd": summary["sd"],
        "ms1IntensityRange_min": summary["mn"],
        "ms1IntensityRange_max": summary["mx"],
        "ms1IntensityQuartiles_Q1": summary["q1"],
        "ms1IntensityQuartiles_Q2": summary["q2"],
        "ms1IntensityQuartiles_Q3": summary["q3"],
    }


def all_mzqc_metrics(*, rt_ms1, tic_ms1, rt_all, mz_lo, mz_hi,
                     npks, lvl_s, prec_mz, prec_int, prec_charge,
                     tics=None, rt_duration=None) -> dict:
    """Compute every implemented mzQC ID-free metric and return one flat dict.

    ``rt_duration`` is the whole-run RT span (MS1+MS2); it is the denominator for
    rtOverMsQuarters. ``tics`` (per scan, aligned with ``npks``/``lvl_s``) lets
    empty-scan counting also catch zero-TIC scans, as MsQuality does."""
    m: dict = {}
    m.update(tic_shape_metrics(rt_ms1, tic_ms1, rt_duration=rt_duration))
    m.update(range_metrics(rt_all, mz_lo, mz_hi))
    m.update(empty_scan_metrics(npks, lvl_s, tics=tics))
    m.update(precursor_metrics(prec_mz, prec_int))
    m.update(charge_metrics(prec_charge))
    return m


# ── per-sample driver ──────────────────────────────────────────────────────

# Coarse MS level each metric family pertains to (for the tidy output table).
_METRIC_LEVEL: dict[str, object] = {
    "areaUnderTic": 1, "areaUnderTicRtQuantiles": 1, "rtOverMsQuarters": 1,
    "chromatographyDuration": None, "rtAcquisitionRange": None,
    "mzAcquisitionRange": 2, "mzPeakRange": None,
    "numberEmptyScansMS1": 1, "numberEmptyScansMS2": 2,
    "medianPrecursorMz": 2, "precursorIntensityMean": 2, "precursorIntensitySd": 2,
    "precursorIntensityRange": 2, "precursorIntensityQuartiles": 2,
    "meanCharge": 2, "medianCharge": 2,
    "ratioCharge1over2": 2, "ratioCharge3over2": 2, "ratioCharge4over2": 2,
    "ms1IntensityMean": 1, "ms1IntensitySd": 1,
    "ms1IntensityRange": 1, "ms1IntensityQuartiles": 1,
}


def _level_for(column: str):
    base = column
    for suf in _NTUPLE_SUFFIXES:
        if column.endswith(suf):
            base = column[: -len(suf)]
            break
    return _METRIC_LEVEL.get(base)


def compute_mzqc_metrics(sp, *, sample_name=None):
    """Compute the implemented mzQC ID-free metrics for a ``PteroSpectra``.

    Reads scan metadata and makes one memory-safe peak pass (per-scan TIC, peak
    count, m/z extent); precursor m/z / intensity / charge come from metadata.
    Respects the current view (e.g. ``sp.filter_ms_level(1).``). Returns a tidy
    DataFrame ``[metric, value, accession, ms_level]`` (+ ``sample`` if named).
    Stack per-sample tables to build the cross-sample QC comparison.

    The MS1 peak-intensity distribution (mean / sd / range / quartiles) is also
    included, as a PTERO extension with no mzQC CV term; it is accumulated in the
    same peak pass with bounded memory.
    """
    import pandas as pd

    sd = sp.spectra_data(
        ["rtime", "ms_level", "precursor_mz", "precursor_intensity", "precursor_charge"]
    ).reset_index(drop=True)
    lvl = sd["ms_level"].to_numpy()
    rt = sd["rtime"].to_numpy(dtype="float64")
    n = len(sd)

    npks = np.zeros(n, dtype="int64")
    tics = np.zeros(n, dtype="float64")
    mz_lo, mz_hi = np.inf, -np.inf
    backend = sp._backend
    cols = ("mz", "intensity")
    acc = _Ms1IntensityAccumulator()                  # MS1 peak-intensity distribution
    if hasattr(backend, "iter_peaks_blocked"):
        for pos, P in backend.iter_peaks_blocked(None, columns=cols):
            if P.size:
                npks[pos] = P.shape[0]
                tics[pos] = float(P[:, 1].sum())
                mz_lo = min(mz_lo, float(P[:, 0].min()))
                mz_hi = max(mz_hi, float(P[:, 0].max()))
                if lvl[pos] == 1:
                    acc.update(P[:, 1])
    else:
        for i in range(n):
            P = sp.peaks_data([i], cols)[0]
            if P.size:
                npks[i] = P.shape[0]
                tics[i] = float(P[:, 1].sum())
                mz_lo = min(mz_lo, float(P[:, 0].min()))
                mz_hi = max(mz_hi, float(P[:, 0].max()))
                if lvl[i] == 1:
                    acc.update(P[:, 1])

    ms1 = lvl == 1
    ms2 = lvl == 2
    rt_finite = rt[np.isfinite(rt)]
    rt_duration = float(rt_finite.max() - rt_finite.min()) if rt_finite.size else None
    flat = all_mzqc_metrics(
        rt_ms1=rt[ms1], tic_ms1=tics[ms1],
        rt_all=rt, mz_lo=mz_lo, mz_hi=mz_hi,
        npks=npks, lvl_s=lvl, tics=tics, rt_duration=rt_duration,
        prec_mz=sd["precursor_mz"].to_numpy(dtype="float64")[ms2],
        prec_int=sd["precursor_intensity"].to_numpy(dtype="float64")[ms2],
        prec_charge=pd.to_numeric(sd["precursor_charge"], errors="coerce").to_numpy(dtype="float64")[ms2],
    )
    flat.update(ms1_intensity_metrics(acc.summary()))   # PTERO extension (no CV term)
    tidy = pd.DataFrame(
        [(k, float(v), accession_for(k), _level_for(k)) for k, v in flat.items()],
        columns=["metric", "value", "accession", "ms_level"],
    )
    if sample_name is not None:
        tidy.insert(0, "sample", sample_name)
    return tidy


def compare_samples(opened, *, add_spread: bool = True):
    """Build a cross-sample QC matrix from several samples.

    Parameters
    ----------
    opened : dict
        ``{sample_name: PteroSpectra}`` -- one (lazily opened) store per sample.
    add_spread : bool
        If True and there are >= 2 samples, append a ``cv_pct`` column: the
        coefficient of variation (100 * sd / |mean|) of each metric across
        samples -- a quick read of which metrics vary most between runs. It is
        descriptive only; per-sample outlier flagging (z-score / IQR) is a
        separate step.

    Returns
    -------
    pandas.DataFrame
        One row per metric, columns ``[metric, accession, ms_level,
        <sample1>, <sample2>, ...]`` (+ ``cv_pct``). Samples keep their input
        order; metrics keep first-seen order. Metrics a sample lacks are NaN.
    """
    import pandas as pd

    names = list(opened)
    if not names:
        return pd.DataFrame(columns=["metric", "accession", "ms_level"])
    frames = [compute_mzqc_metrics(sp, sample_name=n) for n, sp in opened.items()]
    long = pd.concat(frames, ignore_index=True)

    metric_order = list(dict.fromkeys(long["metric"]))          # first-seen order
    meta = (long.drop_duplicates("metric").set_index("metric")[["accession", "ms_level"]]
                .reindex(metric_order))
    wide = (long.pivot_table(index="metric", columns="sample", values="value", sort=False)
                .reindex(index=metric_order, columns=names))
    out = meta.join(wide).reset_index()

    if add_spread and len(names) >= 2:
        vals = out[names].to_numpy(dtype="float64")
        mean = np.nanmean(vals, axis=1)
        sd = np.nanstd(vals, axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(np.abs(mean) > 0, 100.0 * sd / np.abs(mean), np.nan)
        out["cv_pct"] = cv
    return out


# ── per-sample outlier flagging (two robust methods, side by side) ──────────

def flag_outliers(comparison, sample_cols, *, z_thresh: float = 3.5,
                  iqr_k: float = 1.5, min_samples: int = 3,
                  min_rel_dev: float = 0.05):
    """Flag, per metric, which samples are outliers -- by two robust methods.

    Each metric (one row of the ``compare_samples`` matrix) is screened across
    its sample values independently by:

    * **Modified z-score** (Iglewicz & Hoaglin): ``z = (x - median) / (1.4826 *
      MAD)``; if ``MAD == 0`` fall back to ``1.253 * meanAD``; if both are 0
      (all values identical) ``z = 0``. Flag ``|z| > z_thresh`` (default 3.5).
    * **Tukey fences**: flag ``x < Q1 - iqr_k*IQR`` or ``x > Q3 + iqr_k*IQR``
      (``iqr_k`` default 1.5).

    Both use medians / quartiles, so neither is dragged around by the very
    outlier it is trying to find. Metrics with fewer than ``min_samples`` finite
    values are skipped (you cannot meaningfully call a sample an outlier among
    two).

    A statistical flag only stands if the value also differs from the metric's
    median by at least ``min_rel_dev`` (default 5%). This removes the
    near-constant-metric artifact -- e.g. an acquisition setting where six of
    eight runs share an identical value, so the MAD is ~0 and a trivial 0.08%
    difference scores ``|z|`` of 80 -- while still catching genuine jumps such
    as empty-scan counts going 0 -> 5000 (median ~0, so the relative deviation
    is effectively infinite and the flag is kept). Returns a tidy DataFrame of
    FLAGGED cells only: ``[metric, sample, value, robust_z, lower_fence,
    upper_fence, flag_z, flag_iqr, direction, rel_dev]``.
    """
    import pandas as pd

    rows = []
    for _, r in comparison.iterrows():
        metric = r["metric"]
        x = pd.to_numeric(pd.Series({c: r.get(c) for c in sample_cols}), errors="coerce")
        v = x.dropna()
        if v.size < min_samples:
            continue
        med = float(v.median())
        mad = float((v - med).abs().median())
        if mad > 0:
            scale = 1.4826 * mad
        else:
            mean_ad = float((v - med).abs().mean())
            scale = 1.253 * mean_ad if mean_ad > 0 else 0.0
        q1, q3 = (float(q) for q in np.quantile(v.to_numpy(), [0.25, 0.75]))
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        for s in sample_cols:
            val = x.get(s)
            if val is None or not np.isfinite(val):
                continue
            val = float(val)
            if scale > 0:
                z = (val - med) / scale
            else:
                z = 0.0                       # all values identical
            fz = abs(z) > z_thresh
            fi = (val < lo) or (val > hi)
            rel_dev = abs(val - med) / max(abs(med), 1e-12)
            if (fz or fi) and rel_dev >= min_rel_dev:
                rows.append((metric, s, val, z, lo, hi, bool(fz), bool(fi),
                             "high" if val > med else "low", float(rel_dev)))
    return pd.DataFrame(rows, columns=["metric", "sample", "value", "robust_z",
                                       "lower_fence", "upper_fence",
                                       "flag_z", "flag_iqr", "direction", "rel_dev"])


def flag_summary(flags, sample_cols):
    """Per-sample tally of flagged metrics (by method) from ``flag_outliers``.

    Sorted so the most-flagged sample (by agreement of both methods, then z) is
    first -- the prime technical-outlier suspect.
    """
    import pandas as pd

    def n(mask_sample, *masks):
        sub = flags[flags["sample"] == mask_sample]
        for m in masks:
            sub = sub[sub[m]]
        return int(len(sub))

    recs = []
    for s in sample_cols:
        metrics = flags[flags["sample"] == s]["metric"].tolist() if not flags.empty else []
        recs.append({
            "sample": s,
            "n_flag_both": 0 if flags.empty else n(s, "flag_z", "flag_iqr"),
            "n_flag_z": 0 if flags.empty else n(s, "flag_z"),
            "n_flag_iqr": 0 if flags.empty else n(s, "flag_iqr"),
            "flagged_metrics": ", ".join(metrics),
        })
    out = pd.DataFrame(recs)
    return out.sort_values(["n_flag_both", "n_flag_z", "n_flag_iqr"],
                           ascending=False).reset_index(drop=True)