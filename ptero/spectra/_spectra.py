"""
PteroSpectra -- user-facing raw MS data object (Python analog of R's ``Spectra``).

Wraps a backend and provides one consistent interface regardless of where the
data lives. Two ideas are ported directly from ``Spectra``:

* **Backend independence** -- the same methods work on the in-memory, mzML,
  Zarr, or Parquet backend; :meth:`set_backend` converts between them.
* **Lazy evaluation queue** -- peak-modifying operations (e.g.
  ``filter_intensity``) are *queued*, not applied immediately, and run only when
  peak data is accessed. This lets them work on read-only backends and be
  undone with :meth:`reset`.

Spectrum-subsetting operations (``[]``, ``filter_rt``, ``filter_ms_level``) act
on metadata and produce a cheap view (no peak copying); they don't go on the
queue because they don't touch peaks.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .backends._base import MsBackend, DEFAULT_PEAKS_VARIABLES
from .backends.memory import MsBackendMemory


def _assemble_qc_row(n, n_ms1, n_ms2, ms1_rt, rt, npks, tics,
                     mz_lo, mz_hi, im_lo, im_hi, has_im) -> pd.DataFrame:
    """Build the one-row QC DataFrame from already-computed per-scan arrays.

    Shared by :meth:`PteroSpectra.qc` and the dashboard's single-pass overview
    so the metric formulas live in exactly one place. Counts and RT/cycle-time
    come from full metadata (``n``, ``ms1_rt``, ``rt``); the peak-derived arrays
    (``npks``, ``tics``) and m/z / mobility extents may cover only a sampled
    subset of scans -- if so, ``total_ion_current`` is scaled to the full run and
    an ``n_scans_sampled`` column is added.
    """
    sampled = len(npks)
    fin = lambda x: float(x) if (x is not None and np.isfinite(x)) else np.nan
    tic_total = float(np.sum(tics)) * (n / sampled if sampled else 1.0)
    row = {
        "n_spectra": n,
        "n_ms1": n_ms1,
        "n_ms2": n_ms2,
        "ms2_per_ms1": (n_ms2 / n_ms1) if n_ms1 else np.nan,
        "median_peaks_per_scan": float(np.median(npks)) if sampled else np.nan,
        "total_ion_current": tic_total,
        "median_scan_tic": float(np.median(tics)) if sampled else np.nan,
        "median_ms1_cycle_time_s": float(np.median(np.diff(ms1_rt))) if ms1_rt.size > 1 else np.nan,
        "rt_min_s": fin(rt.min()) if n else np.nan,
        "rt_max_s": fin(rt.max()) if n else np.nan,
        "mz_min": fin(mz_lo),
        "mz_max": fin(mz_hi),
    }
    if has_im:
        row["mobility_min"] = fin(im_lo)
        row["mobility_max"] = fin(im_hi)
    if sampled < n:
        row["n_scans_sampled"] = sampled
    return pd.DataFrame([row])


class PteroSpectra:
    def __init__(self, backend: MsBackend, processing_queue: Optional[list] = None,
                 processing_log: Optional[list] = None):
        self._backend = backend
        self._queue: list[tuple[str, Callable]] = list(processing_queue or [])
        self._log: list[str] = list(processing_log or [])

    # ── basic info ───────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._backend)

    @property
    def backend(self) -> MsBackend:
        return self._backend

    @property
    def has_4d(self) -> bool:
        return self._backend.has_4d

    @property
    def peaks_variables(self) -> tuple[str, ...]:
        return self._backend.peaks_variables

    # ── spectrum-level metadata access ───────────────────────────────────
    def spectra_data(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        return self._backend.spectra_data(columns)

    def spectra_variables(self) -> list[str]:
        return list(self._backend.spectra_data().columns)

    def __getattr__(self, name):
        # expose core spectra variables as attributes, e.g. sp.rtime, sp.ms_level
        if name.startswith("_"):
            raise AttributeError(name)
        from .backends._base import CORE_SPECTRA_VARIABLES
        if name in CORE_SPECTRA_VARIABLES:
            return self._backend.spectra_data([name])[name].to_numpy()
        raise AttributeError(name)

    # ── peak data access (applies the lazy queue) ────────────────────────
    def peaks_data(self, indices=None, columns=DEFAULT_PEAKS_VARIABLES) -> list[np.ndarray]:
        cols = list(columns)
        # always read mz+intensity (+im) so queued ops can act, then project
        read_cols = list(self.peaks_variables)
        raw = self._backend.peaks_data(indices, read_cols)
        col_pos = {c: i for i, c in enumerate(read_cols)}
        processed = []
        for arr in raw:
            for _name, op in self._queue:
                arr = op(arr, col_pos)
            take = [col_pos[c] for c in cols]
            processed.append(arr[:, take] if arr.size else np.empty((0, len(cols))))
        return processed

    def iter_peaks(self, indices=None, columns=DEFAULT_PEAKS_VARIABLES,
                   block_size: int = 4_000_000):
        """Generator form of :meth:`peaks_data`, yielding ``(position, peaks)`` one
        spectrum at a time. ``position`` is the index into the requested spectra
        (0..n-1 when ``indices`` is None).

        When the backend supports blocked reads (the Zarr backend), the flat peak
        arrays are read in large contiguous blocks so each chunk is fetched and
        decompressed once -- far cheaper than one read per spectrum on a large run
        -- while memory stays bounded to a single block. The queued peak filters
        (mz-range, mobility, intensity, ...) are applied to every spectrum exactly
        as in :meth:`peaks_data`, so the per-spectrum result is identical; only the
        read path and the yield order differ. Backends without blocked reads fall
        back to a correct per-spectrum path.
        """
        cols = list(columns)
        read_cols = list(self.peaks_variables)
        col_pos = {c: i for i, c in enumerate(read_cols)}
        take = [col_pos[c] for c in cols]
        if hasattr(self._backend, "iter_peaks_blocked"):
            src = self._backend.iter_peaks_blocked(indices, read_cols, block_size)
        else:
            src = enumerate(self._backend.peaks_data(indices, read_cols))
        for pos, arr in src:
            for _name, op in self._queue:
                arr = op(arr, col_pos)
            if not arr.size:
                result = np.empty((0, len(cols)))
            elif take == list(range(arr.shape[1])):
                result = arr
            else:
                result = arr[:, take]
            yield pos, result

    def mz(self, indices=None) -> list[np.ndarray]:
        return [p[:, 0] for p in self.peaks_data(indices, ("mz",))]

    def intensity(self, indices=None) -> list[np.ndarray]:
        return [p[:, 0] for p in self.peaks_data(indices, ("intensity",))]

    def ion_mobility_peaks(self, indices=None) -> list[np.ndarray]:
        if not self.has_4d:
            raise ValueError("Not 4D data: no per-peak ion mobility.")
        return [p[:, 0] for p in self.peaks_data(indices, ("ion_mobility",))]

    # ── subsetting (cheap, view-based) ───────────────────────────────────
    def __getitem__(self, key) -> "PteroSpectra":
        n = len(self)
        if isinstance(key, (int, np.integer)):
            idx = [int(key)]
        elif isinstance(key, slice):
            idx = list(range(*key.indices(n)))
        else:
            key = np.asarray(key)
            idx = np.where(key)[0].tolist() if key.dtype == bool else key.astype(int).tolist()
        return PteroSpectra(self._backend.subset(idx), self._queue, self._log)

    def _subset_by_mask(self, mask: np.ndarray, note: str) -> "PteroSpectra":
        idx = np.where(np.asarray(mask, dtype=bool))[0]
        new = PteroSpectra(self._backend.subset(idx), self._queue, self._log + [note])
        return new

    # ── spectrum filters (operate on metadata; eager view) ───────────────
    def filter_ms_level(self, level: int) -> "PteroSpectra":
        ms = self._backend.spectra_data(["ms_level"])["ms_level"].to_numpy()
        return self._subset_by_mask(ms == level, f"filter_ms_level({level})")

    def filter_rt(self, rt_min: float, rt_max: float, ms_level: Optional[int] = None) -> "PteroSpectra":
        sd = self._backend.spectra_data(["rtime", "ms_level"])
        rt = sd["rtime"].to_numpy(dtype="float64")
        mask = (rt >= rt_min) & (rt <= rt_max)
        if ms_level is not None:
            mask &= sd["ms_level"].to_numpy() == ms_level
        return self._subset_by_mask(mask, f"filter_rt({rt_min},{rt_max})")

    def filter_precursor_mz(self, mz_min: float, mz_max: float) -> "PteroSpectra":
        pmz = self._backend.spectra_data(["precursor_mz"])["precursor_mz"].to_numpy(dtype="float64")
        return self._subset_by_mask((pmz >= mz_min) & (pmz <= mz_max),
                                    f"filter_precursor_mz({mz_min},{mz_max})")

    def filter_empty(self) -> "PteroSpectra":
        n_peaks = np.array([len(m) for m in self._backend.mz()])
        return self._subset_by_mask(n_peaks > 0, "filter_empty")

    # ── peak filters (lazy; queued) ──────────────────────────────────────
    def _queue_op(self, name: str, op: Callable) -> "PteroSpectra":
        return PteroSpectra(self._backend, self._queue + [(name, op)], self._log + [name])

    def filter_intensity(self, low: float = 0.0, high: float = np.inf) -> "PteroSpectra":
        def op(arr, pos):
            if not arr.size:
                return arr
            ints = arr[:, pos["intensity"]]
            return arr[(ints >= low) & (ints <= high)]
        return self._queue_op(f"filter_intensity({low},{high})", op)

    def filter_mz_range(self, mz_min: float, mz_max: float) -> "PteroSpectra":
        def op(arr, pos):
            if not arr.size:
                return arr
            m = arr[:, pos["mz"]]
            mask = (m >= mz_min) & (m <= mz_max)
            return arr if mask.all() else arr[mask]
        return self._queue_op(f"filter_mz_range({mz_min},{mz_max})", op)

    def filter_mobility(self, im_min: float, im_max: float) -> "PteroSpectra":
        if not self.has_4d:
            raise ValueError("Not 4D data: cannot filter by ion mobility.")
        def op(arr, pos):
            if not arr.size:
                return arr
            im = arr[:, pos["ion_mobility"]]
            mask = (im >= im_min) & (im <= im_max)
            return arr if mask.all() else arr[mask]
        return self._queue_op(f"filter_mobility({im_min},{im_max})", op)

    def normalize(self, method: str = "max", target: float = 1.0) -> "PteroSpectra":
        """Rescale each spectrum's intensities (queued, per-spectrum).

        ``method='max'`` divides by the base peak (tallest peak -> ``target``);
        ``method='sum'`` divides by the total ion current (intensities sum to
        ``target``). Makes spectra comparable across scans/runs. Spectra with
        zero total intensity are left unchanged.
        """
        if method not in ("max", "sum"):
            raise ValueError("method must be 'max' or 'sum'")

        def op(arr, pos):
            if not arr.size:
                return arr
            ii = pos["intensity"]
            inten = arr[:, ii]
            denom = inten.max() if method == "max" else inten.sum()
            if denom <= 0:
                return arr
            arr = arr.copy()                       # don't mutate the backend's array
            arr[:, ii] = inten * (target / denom)
            return arr
        return self._queue_op(f"normalize({method},{target})", op)

    def bin_mz(self, bin_width: float = 1.0, mz_min: Optional[float] = None) -> "PteroSpectra":
        """Bin the m/z axis (queued, per-spectrum): peaks within ``bin_width``
        Da are merged -- intensities summed, m/z reported as the
        intensity-weighted centroid of the bin.

        Down-samples for speed/denoising. For 4D data the merged peak's mobility
        is the intensity-weighted mean within the bin, so binning is **m/z-only
        and merges across mobility** -- don't bin if you need mobility resolved.
        """
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")

        def op(arr, pos):
            if arr.shape[0] == 0:
                return arr
            mzi, ii = pos["mz"], pos["intensity"]
            mz, inten = arr[:, mzi], arr[:, ii]
            lo = mz.min() if mz_min is None else mz_min
            bins = np.floor((mz - lo) / bin_width + 1e-9).astype(np.int64)
            _, inv = np.unique(bins, return_inverse=True)
            sum_int = np.bincount(inv, weights=inten)
            safe = np.where(sum_int > 0, sum_int, 1.0)
            wmean_mz = np.bincount(inv, weights=inten * mz) / safe
            out = np.empty((sum_int.size, arr.shape[1]), dtype=arr.dtype)
            out[:, mzi] = wmean_mz
            out[:, ii] = sum_int
            if "ion_mobility" in pos:                # weighted-mean mobility per bin
                im = arr[:, pos["ion_mobility"]]
                out[:, pos["ion_mobility"]] = np.bincount(inv, weights=inten * im) / safe
            return out
        return self._queue_op(f"bin_mz({bin_width})", op)

    @property
    def processing_queue(self) -> list[str]:
        return [n for n, _ in self._queue]

    def reset(self) -> "PteroSpectra":
        """Drop the lazy processing queue (undo queued peak operations)."""
        return PteroSpectra(self._backend, [], self._log + ["reset"])

    # ── flatten to a long peak table (bridge to viz / dashboard) ─────────
    def to_points(self, ms_level: Optional[int] = None,
                  include_mobility: bool = True,
                  max_points: Optional[int] = None,
                  max_spectra: Optional[int] = None) -> pd.DataFrame:
        """Flatten the run into a long table with **one row per peak**.

        Columns: ``rt``, ``mz``, ``intensity`` (+ ``ion_mobility`` if 4D),
        plus ``ms_level`` and ``scan_index``. The lazy processing queue is
        applied, so any queued peak filters take effect here.

        This is the bridge to visualization (feature maps, chromatograms) and
        the dashboard, which consume exactly these columns.

        Big runs: a full timsTOF run has hundreds of millions of peaks, far more
        than fits in one DataFrame (or than a rasterized plot can use). Two
        optional caps keep it bounded **without ever holding the whole run in
        memory** -- peaks are streamed one spectrum at a time and subsampled:

        * ``max_spectra`` -- read at most this many spectra, chosen at random
          (keeps retention-time coverage while reading far fewer peaks).
        * ``max_points`` -- keep at most ~this many peaks total (uniform random
          thinning). For a density plot like a feature map, a couple of million
          points is visually indistinguishable from all of them.

        With neither cap set, behaviour is unchanged (every peak is returned).
        Narrowing the view first (``filter_rt`` / ``filter_mz_range``) and then
        calling without caps gives full detail on a small slice.
        """
        view = self.filter_ms_level(ms_level) if ms_level is not None else self
        want_im = include_mobility and view.has_4d
        peak_cols = ("mz", "intensity", "ion_mobility") if want_im else ("mz", "intensity")
        empty_cols = ["rt", "mz", "intensity"] + (["ion_mobility"] if want_im else []) + ["ms_level", "scan_index"]

        # ── fast path: no caps -> original all-at-once behaviour ──
        if max_points is None and max_spectra is None:
            peaks = view.peaks_data(None, peak_cols)          # applies the lazy queue
            sd = view.spectra_data(["rtime", "ms_level", "scan_index"]).reset_index(drop=True)
            counts = np.array([p.shape[0] for p in peaks], dtype=np.int64)
            if counts.sum() == 0:
                return pd.DataFrame({c: np.array([], dtype="float64") for c in empty_cols})
            stacked = np.vstack([p for p in peaks if p.shape[0] > 0])
            rt = np.repeat(sd["rtime"].to_numpy(dtype="float64"), counts)
            ms = np.repeat(sd["ms_level"].to_numpy(), counts)
            scan = np.repeat(sd["scan_index"].to_numpy(), counts)
            out = {"rt": rt, "mz": stacked[:, 0], "intensity": stacked[:, 1]}
            if want_im:
                out["ion_mobility"] = stacked[:, 2]
            out["ms_level"] = ms
            out["scan_index"] = scan
            return pd.DataFrame(out)

        # ── bounded path: stream + subsample (memory stays ~max_points) ──
        sd = view.spectra_data(["rtime", "ms_level", "scan_index"]).reset_index(drop=True)
        n = len(sd)
        sel = np.arange(n)
        if max_spectra is not None and n > max_spectra:
            # random (not evenly-spaced) subset: avoids regular gaps that alias
            # into vertical stripes when the points are rasterized
            sel = np.sort(np.random.default_rng(0).choice(n, int(max_spectra), replace=False))

        # choose a keep-fraction from a cheap upper-bound on the peak count
        frac = 1.0
        if max_points is not None:
            est = view._estimate_peak_count(sel)
            if est and est > max_points:
                frac = float(max_points) / float(est)
        rng = np.random.default_rng(0)

        rt_all = sd["rtime"].to_numpy(dtype="float64")
        ms_all = sd["ms_level"].to_numpy()
        scan_all = sd["scan_index"].to_numpy()
        mz_p, int_p, im_p, rt_p, ms_p, sc_p = [], [], [], [], [], []
        for i in sel:
            P = view.peaks_data([int(i)], peak_cols)[0]      # one spectrum, queue applied
            k = P.shape[0]
            if k == 0:
                continue
            if frac < 1.0:
                keep = rng.random(k) < frac
                if not keep.any():
                    continue
                P = P[keep]; k = P.shape[0]
            mz_p.append(P[:, 0]); int_p.append(P[:, 1])
            if want_im:
                im_p.append(P[:, 2])
            rt_p.append(np.full(k, rt_all[i], dtype="float64"))
            ms_p.append(np.full(k, ms_all[i]))
            sc_p.append(np.full(k, scan_all[i]))

        if not mz_p:
            return pd.DataFrame({c: np.array([], dtype="float64") for c in empty_cols})
        out = {"rt": np.concatenate(rt_p), "mz": np.concatenate(mz_p),
               "intensity": np.concatenate(int_p)}
        if want_im:
            out["ion_mobility"] = np.concatenate(im_p)
        out["ms_level"] = np.concatenate(ms_p)
        out["scan_index"] = np.concatenate(sc_p)
        return pd.DataFrame(out)

    def _estimate_peak_count(self, sel: Optional[np.ndarray] = None) -> Optional[int]:
        """Cheap upper-bound peak count (from CSR offsets if the backend has
        them), used only to pick a subsample fraction. Returns None if unknown.
        Ignores queued peak filters, so it's an over-estimate -- which makes the
        resulting subsample conservative (never more than the budget)."""
        be = self._backend
        offs = getattr(be, "_offsets", None)
        if offs is None:
            return None
        offs = np.asarray(offs)
        idx = getattr(be, "_idx", None)               # zarr view-index, if any
        if idx is not None:
            storage = np.asarray(idx)
            storage = storage if sel is None else storage[sel]
            return int(np.sum(offs[storage + 1] - offs[storage]))
        # memory backend: offsets already match this object's spectra
        if sel is None:
            return int(offs[-1])
        return int(np.sum(offs[sel + 1] - offs[sel]))

    # ── chromatograms (collapse m/z -> intensity vs retention time) ──────
    def _chromatogram(self, kind, mz=None, mz_tol=0.01, ppm=None,
                      mobility=None, agg="sum", ms_level=None,
                      max_spectra=None) -> pd.DataFrame:
        """Shared engine for TIC/BPC/XIC. Streams one spectrum at a time so the
        whole run never has to be in memory at once. Returns rt/intensity.

        Reads only the peak columns each chromatogram needs (TIC/BPC need just
        intensity), and with ``max_spectra`` evaluates an evenly-spaced subset of
        scans -- enough for a faithful overview of a very large run, while a
        narrowed view (few scans) is computed exactly."""
        view = self.filter_ms_level(ms_level) if ms_level is not None else self
        n = len(view)
        rt_all = view.spectra_data(["rtime"])["rtime"].to_numpy(dtype="float64")

        sel = np.arange(n)
        if max_spectra is not None and n > max_spectra:
            sel = np.unique(np.linspace(0, n - 1, int(max_spectra)).round().astype(np.int64))

        need_im = mobility is not None
        if need_im and not view.has_4d:
            raise ValueError("a mobility window requires 4D data")
        if kind == "xic":
            half = (mz * ppm / 1e6) if ppm is not None else mz_tol
            lo, hi = mz - half, mz + half
            cols = ("mz", "intensity", "ion_mobility") if need_im else ("mz", "intensity")
            mz_i, int_i, im_i = 0, 1, 2
        else:                                   # tic / bpc need only intensity
            cols = ("intensity",)
            int_i = 0
        aggfun = np.sum if agg == "sum" else np.max

        rt = rt_all[sel]
        out = np.zeros(len(sel), dtype="float64")
        for j, i in enumerate(sel):
            P = view.peaks_data([int(i)], cols)[0]      # one spectrum (queue applied)
            if P.shape[0] == 0:
                continue
            if kind == "tic":
                out[j] = P[:, int_i].sum()
            elif kind == "bpc":
                out[j] = P[:, int_i].max()
            else:  # xic
                mask = (P[:, mz_i] >= lo) & (P[:, mz_i] <= hi)
                if need_im:
                    mask &= (P[:, im_i] >= mobility[0]) & (P[:, im_i] <= mobility[1])
                out[j] = aggfun(P[:, int_i][mask]) if mask.any() else 0.0

        return (pd.DataFrame({"rt": rt, "intensity": out})
                .sort_values("rt", kind="stable").reset_index(drop=True))

    def tic(self, ms_level: Optional[int] = None, max_spectra: Optional[int] = None) -> pd.DataFrame:
        """Total Ion Chromatogram: summed intensity per scan vs retention time."""
        return self._chromatogram("tic", ms_level=ms_level, max_spectra=max_spectra)

    def bpc(self, ms_level: Optional[int] = None, max_spectra: Optional[int] = None) -> pd.DataFrame:
        """Base Peak Chromatogram: most intense peak per scan vs retention time."""
        return self._chromatogram("bpc", ms_level=ms_level, max_spectra=max_spectra)

    def xic(self, mz: float, mz_tol: float = 0.01, ppm: Optional[float] = None,
            mobility: Optional[tuple] = None, agg: str = "sum",
            ms_level: Optional[int] = None) -> pd.DataFrame:
        """Extracted Ion Chromatogram for a target m/z.

        Sums (``agg='sum'``) or takes the max (``agg='max'``) intensity of peaks
        within ``mz ± mz_tol`` (Da) -- or ``± mz*ppm/1e6`` if ``ppm`` is given --
        per scan, vs retention time. ``mobility=(low, high)`` additionally
        restricts to a 1/K0 window (4D data only), which separates co-eluting
        isobaric ions -- the timsTOF advantage.
        """
        return self._chromatogram("xic", mz=mz, mz_tol=mz_tol, ppm=ppm,
                                  mobility=mobility, agg=agg, ms_level=ms_level)

    # ── combine spectra (S/N boost; non-lazy -> returns a NEW PteroSpectra) ──
    def combine(self, rt_window: Optional[float] = None, n_scans: Optional[int] = None,
                agg: str = "sum", mz_tol: float = 0.01,
                mobility_tol: float = 0.01) -> "PteroSpectra":
        """Merge groups of spectra into fewer, higher-S/N spectra.

        Grouping (choose one): ``rt_window`` (seconds) groups scans by
        retention-time block; ``n_scans`` groups every N consecutive scans.
        MS levels are always kept separate, and MS2 spectra are additionally
        grouped by precursor m/z (different precursors never merge) -- matching
        OpenMS's SpectraMerger semantics.

        Within a group, peaks are merged on an ``mz_tol`` grid (and
        ``mobility_tol`` for 4D, so peaks at different mobilities stay separate):
        intensities are summed (``agg='sum'``) or averaged over the group
        (``agg='mean'``); m/z and mobility become intensity-weighted means.

        This materializes the lazy queue and returns a new in-memory
        PteroSpectra (the spectrum count changes, so it can't be a queued op).
        """
        if (rt_window is None) == (n_scans is None):
            raise ValueError("pass exactly one of rt_window or n_scans")
        if agg not in ("sum", "mean"):
            raise ValueError("agg must be 'sum' or 'mean'")

        md = self.spectra_data(["rtime", "ms_level", "scan_index", "precursor_mz"]).reset_index(drop=True)
        peaks = self.peaks_data(None, self.peaks_variables)      # queue applied here
        has_im = self.has_4d
        n_pv = len(self.peaks_variables)

        rt = md["rtime"].to_numpy(dtype="float64")
        lvl = md["ms_level"].to_numpy()
        prec = md["precursor_mz"].to_numpy(dtype="float64")
        pk = np.where(np.isnan(prec), np.nan, np.round(prec, 2))  # precursor group key

        info = pd.DataFrame({"rt": rt, "lvl": lvl, "pk": pk})
        if rt_window is not None:
            info["tg"] = np.floor((rt - np.nanmin(rt)) / rt_window).astype("int64")
        else:
            info = info.sort_values(["lvl", "pk", "rt"], kind="stable")
            info["tg"] = (info.groupby(["lvl", "pk"], dropna=False).cumcount() // n_scans).astype("int64")
            info = info.sort_index()

        meta_rows, out_peaks = [], []
        new_idx = 0
        for (glvl, gpk, _gtg), gdf in info.groupby(["lvl", "pk", "tg"], dropna=False, sort=True):
            pos = gdf.index.to_numpy()
            parts = [peaks[i] for i in pos if peaks[i].shape[0] > 0]
            if parts:
                P = np.vstack(parts)
                mz, inten = P[:, 0], P[:, 1]
                # round (not floor) to the tolerance grid: robust to float32
                # storage error near round values (e.g. mobility 0.90 -> 0.8999998)
                mz_bin = np.round(mz / mz_tol).astype("int64")
                if has_im:
                    im = P[:, 2]
                    keys = np.stack([mz_bin, np.round(im / mobility_tol).astype("int64")], axis=1)
                    _, inv = np.unique(keys, axis=0, return_inverse=True)
                else:
                    _, inv = np.unique(mz_bin, return_inverse=True)
                sum_int = np.bincount(inv, weights=inten)
                safe = np.where(sum_int > 0, sum_int, 1.0)
                pd_ = {"mz": np.bincount(inv, weights=inten * mz) / safe,
                       "intensity": sum_int if agg == "sum" else sum_int / len(pos)}
                if has_im:
                    pd_["ion_mobility"] = np.bincount(inv, weights=inten * im) / safe
            else:
                pd_ = {"mz": np.array([]), "intensity": np.array([])}
                if has_im:
                    pd_["ion_mobility"] = np.array([])
            out_peaks.append(pd_)
            meta_rows.append({"ms_level": int(glvl), "rtime": float(gdf["rt"].mean()),
                              "scan_index": new_idx,
                              "precursor_mz": (np.nan if pd.isna(gpk) else float(gpk))})
            new_idx += 1

        be = MsBackendMemory.from_spectra_list(pd.DataFrame(meta_rows), out_peaks)
        return PteroSpectra(be)

    def centroid_check(self, n_check: int = 5, ms_level=None) -> dict:
        """Inspect whether peaks look centroided, and *show the basis*.

        For 4D data this measures m/z spacing WITHIN a single ion-mobility (1/K0)
        slice; pooling a whole frame across mobility scans would make even
        centroided data look densely sampled, so the pooled spacing is not a valid
        4D verdict (it is reported separately by the diagnostic). For 3D data it
        measures the spectrum's m/z array directly. Profile data has a tiny, regular
        spacing (~0.001-0.01 Da); centroided data is sparse (> ~0.01 Da). Returns
        ``{'median_mz_spacing', 'centroided', 'basis', 'n_spectra_checked',
        'ms_level'}``; the 0.01 Da cutoff is heuristic and the definitive status
        comes from instrument/file metadata. ``ms_level`` restricts the check to one
        level.
        """
        from ._centroid import measure_centroided
        return measure_centroided(self, ms_level=ms_level, n_check=n_check)

    def is_centroided(self, n_check: int = 5, ms_level=None) -> bool:
        """Heuristic bool: do the peaks look centroided (sparse) vs profile?

        Thin wrapper over :meth:`centroid_check` -- call that to also see the
        measured m/z spacing and the ``basis`` the verdict is based on. Heuristic,
        not authoritative; the definitive status comes from the instrument/file.
        """
        return bool(self.centroid_check(n_check, ms_level)["centroided"])

    # ── raw-data QC (per-run quality summary) ────────────────────────────
    def qc(self, max_spectra: Optional[int] = None) -> pd.DataFrame:
        """One-row summary of *acquisition* quality, computed from the raw data.

        Columns: scan counts (``n_spectra/n_ms1/n_ms2``), acquisition efficiency
        (``ms2_per_ms1``), ``median_peaks_per_scan``, ``total_ion_current`` and
        ``median_scan_tic``, ``median_ms1_cycle_time_s`` (median gap between
        consecutive MS1 scans), the RT span, and the m/z (and 1/K0, if 4D) range
        actually covered. Returns a one-row DataFrame so multiple runs can be
        ``pd.concat``-ed into a QC table.

        Respects the current view/queue, so e.g. ``sp.filter_ms_level(1).qc()``
        summarises just MS1. Covers instrument/acquisition quality only -- it
        does not compute mass accuracy or identification rates (those need
        reference masses or search results, i.e. later modules).

        ``max_spectra`` caps how many (randomly chosen) scans are read for the
        *peak-derived* metrics on a very large run -- scan counts, RT span and
        cycle time stay exact (they come from metadata), peak metrics become a
        faithful estimate, ``total_ion_current`` is scaled to the full run, and a
        ``n_scans_sampled`` column records the sample size. With it unset, every
        scan is read (exact).
        """
        sd = self.spectra_data(["rtime", "ms_level"]).reset_index(drop=True)
        n = len(sd)
        lvl = sd["ms_level"].to_numpy()
        rt = sd["rtime"].to_numpy(dtype="float64")
        cols = self.peaks_variables
        has_im = self.has_4d

        sel = np.arange(n)
        if max_spectra is not None and n > max_spectra:
            sel = np.sort(np.random.default_rng(0).choice(n, int(max_spectra), replace=False))
        sampled = len(sel)

        npks = np.zeros(sampled, dtype="int64")
        tics = np.zeros(sampled, dtype="float64")
        mz_lo, mz_hi, im_lo, im_hi = np.inf, -np.inf, np.inf, -np.inf
        for j, i in enumerate(sel):                     # stream: one spectrum at a time
            p = self.peaks_data([int(i)], cols)[0]
            npks[j] = p.shape[0]
            if p.shape[0]:
                tics[j] = p[:, 1].sum()
                mz_lo = min(mz_lo, p[:, 0].min()); mz_hi = max(mz_hi, p[:, 0].max())
                if has_im:
                    im_lo = min(im_lo, p[:, 2].min()); im_hi = max(im_hi, p[:, 2].max())

        ms1_rt = np.sort(rt[lvl == 1])
        n_ms1 = int((lvl == 1).sum())
        n_ms2 = int((lvl == 2).sum())
        return _assemble_qc_row(n, n_ms1, n_ms2, ms1_rt, rt, npks, tics,
                                mz_lo, mz_hi, im_lo, im_hi, has_im)

    # ── backend conversion (materializes the queue) ──────────────────────
    def set_backend(self, target: str, path: Optional[str] = None, **kw) -> "PteroSpectra":
        """Convert to another backend ('memory', 'zarr', 'parquet').

        Reads all (queue-applied) peaks + metadata from the current backend and
        writes them into the target, returning a new PteroSpectra with an empty
        queue. For 'zarr'/'parquet' a ``path`` is required.
        """
        meta = self.spectra_data().reset_index(drop=True)
        read_cols = list(self.peaks_variables)
        peaks = self.peaks_data(None, read_cols)  # applies queue
        pos = {c: i for i, c in enumerate(read_cols)}
        mz_parts = [p[:, pos["mz"]] for p in peaks]
        int_parts = [p[:, pos["intensity"]] for p in peaks]
        offsets = np.concatenate([[0], np.cumsum([len(m) for m in mz_parts])]).astype(np.int64)
        cat = lambda parts, dt: (np.concatenate(parts) if parts else np.array([], dt))
        mz = cat(mz_parts, np.float64)
        intensity = cat(int_parts, np.float64).astype(np.float32)
        im = None
        if self.has_4d:
            im = cat([p[:, pos["ion_mobility"]] for p in peaks], np.float64).astype(np.float32)

        if target == "memory":
            from .backends.memory import MsBackendMemory
            be = MsBackendMemory(meta, mz, intensity, offsets, ion_mobility=im)
        elif target == "zarr":
            if path is None:
                raise ValueError("set_backend('zarr') requires a path=")
            from .backends.zarr_backend import MsBackendZarr
            be = MsBackendZarr.write(path, meta, mz, intensity, offsets, ion_mobility=im, **kw)
        elif target == "parquet":
            if path is None:
                raise ValueError("set_backend('parquet') requires a path=")
            from .backends.parquet_backend import MsBackendParquet
            be = MsBackendParquet.write(path, meta, mz, intensity, offsets, ion_mobility=im)
        else:
            raise ValueError(f"unknown backend {target!r}; use 'memory', 'zarr', or 'parquet'")
        return PteroSpectra(be, [], self._log + [f"set_backend({target})"])

    # ── display ──────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        kind = "4D (RT x m/z x ion mobility)" if self.has_4d else "3D (RT x m/z)"
        q = f" | lazy queue: {len(self._queue)} step(s)" if self._queue else ""
        return (f"PteroSpectra: {len(self):,} spectra | {kind} | "
                f"backend: {type(self._backend).__name__}{q}")