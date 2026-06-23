"""
Interactive raw-data dashboard (Panel + HoloViews + Datashader).

Big-data strategy: data stays in Python. Density maps are built by reading the
selection in batches and accumulating each batch into a fixed-size Datashader
pixel grid, so the whole run never has to fit in memory yet every peak is shown.
Chromatogram, diagnostics and QC are computed exactly in the same single pass.
On-demand views (a clicked spectrum, an extracted-ion chromatogram) read only
what they need.

Views
-----
* Chromatogram (TIC/BPC) -- click it to show that scan's spectrum below.
* Feature map (RT x m/z), Mobility (m/z x 1/K0), RT x 1/K0 -- the three faces
  of the 4D data cube, as intensity-density heatmaps.
* XIC -- target m/z (+ tolerance) -> elution trace, plus its mobilogram (4D).
* Diagnostics -- peaks-per-scan and MS1 cycle-time across the run.
* Precursor / Window scheme -- mode-aware single tab. DDA: precursor m/z
  distribution + m/z-vs-RT density. DIA: the isolation windows drawn as boxes
  in m/z x 1/K0 (the acquisition scheme).
* QC -- one-row acquisition-quality summary.

Run it on your machine (not inside a sandbox):

    panel serve dashboard.py --show
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv

from ..spectra import (PteroSpectra, MsBackendZarr, open_zarr,
                       annotate_store_centroided, detect_centroided, from_mzml)
from ..spectra._spectra import _assemble_qc_row
from ..spectra._qc_metrics import compare_samples, flag_outliers, flag_summary
from ._qc_plots import plot_zscore_heatmap, plot_flag_tally, plot_metric_strips
from ..process import make_centroid_transform
from .convert import convert_to_zarr, convert_many, resolve_batch_inputs

pn.extension(notifications=True)
hv.extension("bokeh")

# shown under each density map so users can read the colour scale
_COLOR_HELP = (
    "**Colour = summed ion intensity** in each pixel — dark → bright means "
    "low → high. The brightest spots are the most abundant ions. Colouring is "
    "histogram-equalised (it shows *relative* structure, not a linear ruler); "
)

_FLAG_GUIDE = (
    "**How to read this.** A flag means *worth a look*, not *a bad run* \u2014 confirm against the value in the table above.\n\n"
    "- **Trust the per-sample tally, not single cells.** Several *related* flags on one sample (e.g. its intensity metrics) is a real signal; a lone flag usually is not.\n"
    "- **Near-constant metrics:** a flag only fires when a value differs from the median by \u22655%, so trivial differences in settings-like metrics (m/z & RT acquisition ranges) are not flagged. If one of these *does* differ it means the **acquisition method changed** \u2014 the runs are not directly comparable, which is different from low quality.\n"
    "- **Grouped designs:** if your samples span conditions (e.g. two groups), a metric that differs *by group* will surface the extremes as \u2018outliers\u2019 \u2014 that is biology, not a fault. Compare within each group for technical QC.\n"
    "- `ms1Intensity*` is a PTERO extension computed on centroided MS1 and is the least standardised signal here."
)


class Dashboard:
    OVERVIEW_MAX_SPECTRA = 600          # scans sampled only to set initial slider ranges

    def __init__(self):
        self.sp: PteroSpectra | None = None

        # ── data-source widgets ──
        # ── Section 1a: open an existing .zarr (one store OR a folder of stores) ──
        self._samples: dict = {}                 # display name -> .zarr path
        self._suspend_sample_watch = False       # guard so populating the selector doesn't auto-open
        self.zarr_path = pn.widgets.TextInput(
            name="A .zarr store, or a folder of .zarr stores (several paths also OK — one per line / comma-separated)",
            placeholder=r"C:\data\run.zarr    — or —    C:\data\stores", width=620)
        self.open_btn = pn.widgets.Button(name="Open", button_type="primary", width=120)

        # ── Section 1b: convert raw data to .zarr (one file OR many, auto-detected) ──
        self.conv_input = pn.widgets.TextInput(
            name="A raw file (.d / .mzML), a folder of raw files, or several paths (one per line / comma-separated)",
            placeholder=r"C:\data\run.d    — or —    C:\data\my_runs    — or —    C:\data\a.d, C:\data\b.d",
            width=620)
        self.conv_outdir = pn.widgets.TextInput(
            name="Output folder for the .zarr store(s)", placeholder=r"C:\data\stores", width=320)
        self.conv_ms = pn.widgets.Select(
            name="MS level", options={"MS1": 1, "MS2": 2, "MS1 + MS2 (both)": None}, value=1, width=150)
        self.conv_frames = pn.widgets.TextInput(
            name="Frames (.d only; e.g. 0-3000; blank = all)", placeholder="0-3000", width=240)
        self.conv_batch = pn.widgets.IntInput(name="Frames/batch", value=500, start=50, step=50, width=120)
        self.conv_workers = pn.widgets.IntInput(
            name="Parallel workers (many files)", value=2, start=1, end=8, step=1, width=180)
        self.conv_overwrite = pn.widgets.Checkbox(name="Overwrite existing", value=False, width=150)
        self.conv_btn = pn.widgets.Button(name="Convert → .zarr", button_type="primary", width=160)

        # the sample switcher (shown on the View panel only when 2+ samples are loaded)
        self.sample_sel = pn.widgets.Select(name="Sample (view)", options=[], visible=False, width=320)

        self.status = pn.pane.Markdown("**No data loaded.** Open a Zarr store or convert a raw file.")

        # ── view controls ──
        self.ms_sel = pn.widgets.Select(name="MS level", options=["all"], value="all", width=120)
        self.rt = pn.widgets.RangeSlider(name="RT (s)", start=0, end=1, value=(0, 1), step=0.1, width=300)
        self.mz = pn.widgets.RangeSlider(name="m/z", start=0, end=1, value=(0, 1), step=1.0, width=300)
        self.im = pn.widgets.RangeSlider(name="1/K0 (mobility)", start=0, end=1, value=(0, 1),
                                         step=0.01, width=300, disabled=True)
        self.update_btn = pn.widgets.Button(name="Update plots", button_type="success", width=150)
        self.plot_panel = pn.Column(
            pn.pane.Markdown("_Plots appear here once data is loaded._"), sizing_mode="stretch_width")

        # ── XIC widgets (independent of the heavy overview render) ──
        self.xic_mz = pn.widgets.TextInput(
            name="Target m/z (comma-separated; optional label as label:mz)",
            placeholder="500.25, 712.40   or   pepA:500.25, pepB:712.40", width=460)
        self.xic_tol = pn.widgets.FloatInput(name="± tolerance (Da)", value=0.01, step=0.01, width=150)
        self.xic_btn = pn.widgets.Button(name="Extract XIC", button_type="primary", width=140)
        self.xic_panel = pn.Column(pn.pane.Markdown("_Enter one or more m/z and click Extract._"),
                                   sizing_mode="stretch_width")

        # ── wiring ──
        self.open_btn.on_click(self._open)
        self.conv_btn.on_click(self._convert)
        self.sample_sel.param.watch(self._on_sample_change, "value")
        self.update_btn.on_click(self._on_update)
        self.xic_btn.on_click(self._on_xic)

        # ── cross-sample QC comparison (mzQC ID-free metrics across loaded samples) ──
        self.cmp_btn = pn.widgets.Button(name="Compute comparison", button_type="primary", width=200)
        self.cmp_panel = pn.Column(
            pn.pane.Markdown("_Click **Compute comparison** to build the table._"),
            sizing_mode="stretch_width")
        self.cmp_btn.on_click(self._render_comparison)

    async def _show_loading(self, msg):
        """Show a spinner over the plot area and a status message, and *flush*
        them to the browser before the (blocking) heavy work begins. Without the
        await, Panel would only send these once the whole callback finished."""
        self.status.object = msg
        self.plot_panel.loading = True
        await asyncio.sleep(0.05)

    # ── loading ──────────────────────────────────────────────────────────
    def _set_data(self, sp: PteroSpectra, msg: str):
        self.sp = sp
        lvls = [int(x) for x in sorted(sp.spectra_data(["ms_level"])["ms_level"].unique().tolist())]
        self.ms_sel.options = ["all"] + lvls
        # Default to MS1 when present: it's the fast, standard precursor view. A
        # ddaPASEF run has few MS1 frames but tens of thousands of MS2 spectra,
        # and reading every one for the "all" overview is very slow -- so MS2 /
        # "all" are opt-in via the selector rather than the default.
        default_lvl = 1 if 1 in lvls else "all"
        self.ms_sel.value = default_lvl

        # ranges come from the level we'll actually show by default
        range_sp = sp.filter_ms_level(1) if default_lvl == 1 else sp
        sd = range_sp.spectra_data(["rtime"])
        rtmin, rtmax = float(sd["rtime"].min()), float(sd["rtime"].max())
        q = range_sp.qc(max_spectra=self.OVERVIEW_MAX_SPECTRA)
        mzmin, mzmax = float(q["mz_min"][0]), float(q["mz_max"][0])
        mzpad = max((mzmax - mzmin) * 0.02, 1.0)
        self.rt.param.update(start=rtmin, end=max(rtmax, rtmin + 1e-6), value=(rtmin, rtmax))
        self.mz.param.update(start=mzmin - mzpad, end=mzmax + mzpad,
                             value=(mzmin - mzpad, mzmax + mzpad))
        if sp.has_4d:
            immin, immax = float(q["mobility_min"][0]), float(q["mobility_max"][0])
            impad = max((immax - immin) * 0.02, 0.01)
            self.im.param.update(start=immin - impad, end=immax + impad,
                                 value=(immin - impad, immax + impad),
                                 disabled=False, visible=True)
        else:
            # 3D data (e.g. Thermo) has no ion mobility — hide the slider entirely
            self.im.param.update(disabled=True, visible=False)

        hint = ""
        if 2 in lvls:
            hint = (" — showing **MS1** by default. Switch the MS level selector to MS2 or "
                    "'all' for fragment data (note: 'all' reads every MS2 scan and can be slow "
                    "on a large run — narrow the RT slider first).")
        self.status.object = f"✅ {msg}{hint}"
        self._render()

    async def _open(self, event=None):
        spec = self.zarr_path.value.strip()
        if not spec:
            self.status.object = ("⚠️ Enter a `.zarr` store, or a folder of `.zarr` stores "
                                  "(several paths also OK, one per line / comma-separated).")
            return
        paths = self._resolve_sample_paths(spec)
        if not paths:
            self.status.object = f"⚠️ No `.zarr` store found from {spec!r}."
            return
        names = self._register_samples(paths)
        await self._activate_sample(names[0])
        if len(names) > 1:
            shown = ", ".join(f"`{n}`" for n in names[:6]) + (f" +{len(names) - 6} more"
                                                              if len(names) > 6 else "")
            self.status.object = (f"✅ Loaded {len(names)} samples: {shown}. Showing `{names[0]}` "
                                  "— switch with the **Sample** selector on the View panel.")

    def _parse_frames(self):
        txt = self.conv_frames.value.strip()
        if not txt:
            return None
        sep = "-" if "-" in txt else ("," if "," in txt else None)
        if sep is None:
            raise ValueError("Frame range must look like '0-3000'.")
        a, b = txt.split(sep, 1)
        return (int(a), int(b))

    @staticmethod
    def _have_pyopenms() -> bool:
        try:
            import pyopenms  # noqa: F401
            return True
        except Exception:
            return False

    def _convert_with_centroiding(self, src, out, *, ms_level, frames, frame_batch):
        """Convert ``src`` -> one centroid store at ``out``.

        Detects profile vs centroid; for a profile mzML it centroids inline during
        conversion (so the profile data is never written to disk). A ``.d`` is left
        as-is (timsTOF is already centroided), and an already-centroid level is never
        re-picked (the peak-picker would wipe it). Returns ``(PteroSpectra, msg)``.
        """
        is_mzml = src.lower().endswith((".mzml", ".mzxml"))
        if not is_mzml:
            convert_to_zarr(src, out, ms_level=ms_level, frames=frames, frame_batch=frame_batch)
            note = "timsTOF .d — already centroided by the instrument"
        else:
            # detect from the source (reads only a few spectra per level)
            probe = from_mzml(src, ms_level=ms_level, lazy=True)
            levels = sorted(int(x) for x in
                            probe.spectra_data(["ms_level"])["ms_level"].dropna().unique())
            verdict = {lvl: detect_centroided(probe, ms_level=lvl)["centroided"] for lvl in levels}
            all_profile = bool(verdict) and all(v is False for v in verdict.values())
            mixed = (any(v is True for v in verdict.values())
                     and not all(bool(v) for v in verdict.values()))
            if all_profile and self._have_pyopenms():
                convert_to_zarr(src, out, ms_level=ms_level, frames=frames,
                                frame_batch=frame_batch, transform=make_centroid_transform())
                note = "mzML was profile → PTERO centroided it during conversion"
            else:
                convert_to_zarr(src, out, ms_level=ms_level, frames=frames, frame_batch=frame_batch)
                if all_profile:
                    note = ("mzML is profile and pyOpenMS isn't installed — stored as profile; "
                            "MS2 annotation will be poor. Run `pip install pyopenms` and re-convert "
                            "to centroid")
                elif mixed:
                    note = ("mixed levels (" + ", ".join(
                        f"MS{l}:{'centroid' if v else 'profile'}" for l, v in verdict.items())
                        + ") — stored as-is; PTERO did not centroid, to avoid wiping the "
                        "already-centroid level")
                else:
                    note = "mzML already centroid — no processing needed"
        res = annotate_store_centroided(out, force=True)   # record the FINAL status
        sp = open_zarr(out)
        levels = res.get("levels", {})
        lv = ", ".join(
            f"MS{l}:{'centroid' if (v.get('centroided') if isinstance(v, dict) else v) else 'profile'}"
            for l, v in levels.items()) if levels else ""
        from pathlib import Path as _P
        msg = (f"Converted `{_P(out).name}` — {len(sp)} spectra. {note}."
               + (f" Final: [{lv}]." if lv else ""))
        return sp, msg

    async def _convert(self, event=None):
        """Convert raw data to .zarr. One input file -> one centroid-aware store
        (profile mzML is centroided inline). Many files / a folder -> parallel batch
        (each -> its own .zarr). Either way the result(s) load as switchable samples."""
        spec = self.conv_input.value.strip()
        outdir = self.conv_outdir.value.strip()
        if not spec or not outdir:
            self.status.object = "⚠️ Fill in the raw input and the output folder."
            return
        inputs = resolve_batch_inputs(spec)
        if not inputs:
            self.status.object = ("⚠️ No `.d` or `.mzML` inputs found. Give a raw file, a folder of "
                                  "raw runs, or list the paths (one per line / comma-separated).")
            return
        try:
            frames = self._parse_frames()
        except Exception as e:
            self.status.object = f"⚠️ {e}"
            return
        from pathlib import Path as _P
        _P(outdir).mkdir(parents=True, exist_ok=True)

        # ── one file: centroid-aware single conversion ──
        if len(inputs) == 1:
            src = inputs[0]
            stem = _P(src).name
            for suf in (".d", ".mzml", ".mzxml"):
                if stem.lower().endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            out = str(_P(outdir) / (stem + ".zarr"))
            scope = f" frames {frames[0]}-{frames[1]}" if frames else " (full run)"
            await self._show_loading(
                f"⏳ Converting `{src}`{scope} → `{out}`: detecting profile/centroid and, if profile, "
                "centroiding during conversion. Progress prints in the server terminal — please wait…")
            try:
                sp, msg = await asyncio.to_thread(
                    self._convert_with_centroiding, src, out,
                    ms_level=self.conv_ms.value, frames=frames,
                    frame_batch=int(self.conv_batch.value))
                self._register_samples([out])
                self._set_data(sp, msg)
            except Exception as e:
                self.status.object = f"❌ **Conversion failed:** {e}"
            finally:
                self.plot_panel.loading = False
            return

        # ── many files: parallel batch (note: the batch path does not centroid
        #    profile mzML — single-file conversion above does) ──
        workers = int(self.conv_workers.value)
        await self._show_loading(
            f"⏳ Converting {len(inputs)} files with {workers} worker(s) → `{outdir}`. Each uses the "
            "**MS level** & **frames/batch** above. Per-file progress prints in the server terminal…")
        try:
            results = await asyncio.to_thread(
                convert_many, inputs, outdir,
                ms_level=self.conv_ms.value, frame_batch=int(self.conv_batch.value),
                max_workers=workers, overwrite=bool(self.conv_overwrite.value))
        except Exception as e:
            self.status.object = f"❌ **Batch failed:** {e}"
            return
        finally:
            self.plot_panel.loading = False

        ok = [r for r in results if r["status"] == "ok"]
        err = [r for r in results if r["status"] == "error"]
        skip = [r for r in results if r["status"] == "skipped"]
        avail = [r["output"] for r in results if r["status"] in ("ok", "skipped")]
        if not avail:
            self.status.object = "❌ Batch produced no stores. " + "; ".join(
                f"`{_P(e['input']).name}` ({e['error']})" for e in err[:4])
            return
        names = self._register_samples(avail)
        await self._activate_sample(names[0])
        parts = [f"**{len(ok)} converted**"]
        if skip:
            parts.append(f"{len(skip)} skipped (already existed)")
        if err:
            parts.append(f"{len(err)} failed")
        msg = "✅ Batch done: " + ", ".join(parts) + f" → `{outdir}`."
        if err:
            msg += " Failed: " + "; ".join(
                f"`{_P(e['input']).name}` ({e['error']})" for e in err[:4])
            if len(err) > 4:
                msg += f"; +{len(err) - 4} more"
        msg += (f" — {len(avail)} store(s) loaded as **samples**; showing `{names[0]}`, switch "
                "with the **Sample** selector on the View panel.")
        self.status.object = msg

    # ── multiple samples: open several stores, switch between them ─────────
    @staticmethod
    def _resolve_sample_paths(spec: str):
        """Turn the samples field into a de-duplicated list of .zarr store paths.

        Accepts a folder containing ``*.zarr`` stores, or several explicit paths
        (one per line / comma-separated). A token ending in ``.zarr`` is taken as a
        store; a plain folder is scanned one level deep for ``*.zarr`` stores.
        """
        from pathlib import Path as _P
        tokens = [t.strip() for line in spec.splitlines()
                  for t in line.split(",") if t.strip()]
        found = []
        for tok in tokens:
            p = _P(tok)
            if str(p).lower().endswith(".zarr"):
                if p.exists():
                    found.append(str(p))
            elif p.is_dir():
                found.extend(str(c) for c in sorted(p.iterdir())
                             if c.is_dir() and str(c).lower().endswith(".zarr"))
        seen, uniq = set(), []
        for x in found:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _register_samples(self, paths):
        """Register store paths as switchable samples; populate the selector.

        Populating is done with the watcher suspended so it does not auto-open a
        store -- the caller decides when to activate one. Returns the display names.
        """
        from pathlib import Path as _P
        samples: dict = {}
        for p in paths:
            base = _P(str(p)).name
            name, k = base, 2
            while name in samples:                 # disambiguate duplicate folder names
                name = f"{base} ({k})"
                k += 1
            samples[name] = str(p)
        self._samples = samples
        names = list(samples)
        self._suspend_sample_watch = True
        try:
            self.sample_sel.options = names
            if names:
                self.sample_sel.value = names[0]
        finally:
            self._suspend_sample_watch = False
        self.sample_sel.visible = len(names) > 1   # only show the switcher for 2+ samples
        return names

    async def _activate_sample(self, name):
        """Make the named sample the active store and re-render the View tabs."""
        path = self._samples.get(name)
        if not path:
            return
        await self._show_loading(f"⏳ Opening sample `{name}` and building the overview…")
        try:
            self.zarr_path.value = path
            sp = open_zarr(path)
            self._set_data(sp, f"Sample `{name}` — {len(sp)} spectra.")
        except Exception as e:
            self.status.object = f"❌ **Failed to open `{name}`:** {e}"
        finally:
            self.plot_panel.loading = False

    async def _on_sample_change(self, event):
        if self._suspend_sample_watch:
            return
        await self._activate_sample(event.new)

    # ── selection ──────────────────────────────────────────────────────────
    def _current_selection(self) -> PteroSpectra:
        sp = self.sp
        if self.ms_sel.value != "all":
            sp = sp.filter_ms_level(int(self.ms_sel.value))
        sp = sp.filter_rt(*self.rt.value)
        sp = sp.filter_mz_range(*self.mz.value)
        if sp.has_4d and not self.im.disabled:
            sp = sp.filter_mobility(*self.im.value)
        return sp

    # ── full-fidelity overview (one pass, all peaks) ──────────────────────
    def _full_overview(self, sp):
        """Read every scan in the selection and aggregate ALL peaks into the
        three density maps, computing the chromatogram, per-scan peak counts and
        QC exactly in the same pass. Returns a dict of results. Memory stays
        bounded to one batch; progress prints to the server terminal."""
        import datashader as dsh
        import colorcet as cc

        W, H = 820, 460
        sd = sp.spectra_data(["rtime", "ms_level"]).reset_index(drop=True)
        n = len(sd)
        rt_all = sd["rtime"].to_numpy(dtype="float64")
        lvl_all = sd["ms_level"].to_numpy()
        want_im = sp.has_4d
        cols = ("mz", "intensity", "ion_mobility") if want_im else ("mz", "intensity")

        rt_lo, rt_hi = float(self.rt.value[0]), float(self.rt.value[1])
        mz_lo, mz_hi = float(self.mz.value[0]), float(self.mz.value[1])
        if rt_hi <= rt_lo: rt_hi = rt_lo + 1e-6
        if mz_hi <= mz_lo: mz_hi = mz_lo + 1e-6
        cvs_fm = dsh.Canvas(plot_width=W, plot_height=H, x_range=(rt_lo, rt_hi), y_range=(mz_lo, mz_hi))
        if want_im:
            im_lo, im_hi = float(self.im.value[0]), float(self.im.value[1])
            if im_hi <= im_lo: im_hi = im_lo + 1e-6
            cvs_mob = dsh.Canvas(plot_width=W, plot_height=H, x_range=(mz_lo, mz_hi), y_range=(im_lo, im_hi))
            cvs_rtim = dsh.Canvas(plot_width=W, plot_height=H, x_range=(rt_lo, rt_hi), y_range=(im_lo, im_hi))

        acc_fm = acc_mob = acc_rtim = None
        tic = np.zeros(n); bpc = np.zeros(n); npks = np.zeros(n, dtype="int64")
        gmz_lo = gmz_hi = gim_lo = gim_hi = np.nan
        buf_rt, buf_mz, buf_int, buf_im = [], [], [], []
        buf_pts = 0; total_pts = 0
        BATCH_PTS = 4_000_000

        def flush():
            nonlocal acc_fm, acc_mob, acc_rtim, buf_rt, buf_mz, buf_int, buf_im, buf_pts
            if not buf_mz:
                return
            df = pd.DataFrame({"rt": np.concatenate(buf_rt), "mz": np.concatenate(buf_mz),
                               "intensity": np.concatenate(buf_int)})
            a = cvs_fm.points(df, "rt", "mz", dsh.sum("intensity"))
            acc_fm = a.fillna(0) if acc_fm is None else acc_fm + a.fillna(0)
            if want_im:
                df["ion_mobility"] = np.concatenate(buf_im)
                b = cvs_mob.points(df, "mz", "ion_mobility", dsh.sum("intensity"))
                acc_mob = b.fillna(0) if acc_mob is None else acc_mob + b.fillna(0)
                c = cvs_rtim.points(df, "rt", "ion_mobility", dsh.sum("intensity"))
                acc_rtim = c.fillna(0) if acc_rtim is None else acc_rtim + c.fillna(0)
            buf_rt, buf_mz, buf_int, buf_im = [], [], [], []
            buf_pts = 0

        seen = 0
        for i, P in sp.iter_peaks(columns=cols):
            seen += 1
            k = P.shape[0]; npks[i] = k
            if k == 0:
                continue
            inten = P[:, 1]; mz = P[:, 0]
            tic[i] = inten.sum(); bpc[i] = inten.max()
            lo, hi = mz.min(), mz.max()
            gmz_lo = lo if np.isnan(gmz_lo) else min(gmz_lo, lo)
            gmz_hi = hi if np.isnan(gmz_hi) else max(gmz_hi, hi)
            if want_im:
                ilo, ihi = P[:, 2].min(), P[:, 2].max()
                gim_lo = ilo if np.isnan(gim_lo) else min(gim_lo, ilo)
                gim_hi = ihi if np.isnan(gim_hi) else max(gim_hi, ihi)
            buf_rt.append(np.full(k, rt_all[i])); buf_mz.append(mz); buf_int.append(inten)
            if want_im:
                buf_im.append(P[:, 2])
            buf_pts += k; total_pts += k
            if buf_pts >= BATCH_PTS:
                flush()
                print(f"[overview] {seen}/{n} scans, {total_pts:,} peaks aggregated", flush=True)
        flush()
        print(f"[overview] done: {n} scans, {total_pts:,} peaks", flush=True)

        def _image(acc, cmap, title, xlabel, ylabel):
            acc = acc.where(acc > 0)
            return hv.Image(acc).opts(cmap=cmap, colorbar=True, cnorm="eq_hist",
                                      width=W, height=H, title=title, xlabel=xlabel, ylabel=ylabel,
                                      clabel="summed intensity",
                                      tools=["hover", "wheel_zoom", "pan", "box_zoom", "reset"],
                                      active_tools=["wheel_zoom"])
        fm_img = (_image(acc_fm, cc.fire, f"Feature map — {total_pts:,} peaks", "RT (s)", "m/z")
                  if acc_fm is not None else None)
        mob_img = (_image(acc_mob, cc.kgy, f"Mobilogram — {total_pts:,} peaks", "m/z", "1/K0")
                   if (want_im and acc_mob is not None) else None)
        rtim_img = (_image(acc_rtim, cc.bmy, f"RT × 1/K0 — {total_pts:,} peaks", "RT (s)", "1/K0")
                    if (want_im and acc_rtim is not None) else None)

        tic_df = pd.DataFrame({"rt": rt_all, "intensity": tic}).sort_values("rt").reset_index(drop=True)
        bpc_df = pd.DataFrame({"rt": rt_all, "intensity": bpc}).sort_values("rt").reset_index(drop=True)
        ms1_rt = np.sort(rt_all[lvl_all == 1])
        qc_df = _assemble_qc_row(n, int((lvl_all == 1).sum()), int((lvl_all == 2).sum()),
                                 ms1_rt, rt_all, npks, tic, gmz_lo, gmz_hi, gim_lo, gim_hi, want_im)
        return dict(tic=tic_df, bpc=bpc_df, fm=fm_img, mob=mob_img, rtim=rtim_img, qc=qc_df,
                    npks=npks, rt_all=rt_all, lvl_all=lvl_all, n=n, total=total_pts)

    # ── single spectrum at a clicked retention time ───────────────────────
    def _spectrum_at(self, sp, rt_all, x):
        # NOTE: the y dimension is deliberately named "intensity (summed)" — NOT
        # the chromatogram's "intensity" — so Bokeh does not LINK the two y-axes.
        # Sharing the name forced the spectrum onto the chromatogram's ~1e7 scale,
        # squashing every real peak (~1e3) into a flat line of dots.
        ydim = "intensity (summed)"
        empty = hv.Spikes([], "m/z", ydim)
        if x is None or len(rt_all) == 0:
            return empty.opts(width=820, height=300, title="Click the chromatogram to show a spectrum")
        i = int(np.argmin(np.abs(rt_all - x)))
        P = sp.peaks_data([i], ("mz", "intensity"))[0]
        if P.shape[0] == 0:
            return empty.opts(width=820, height=300,
                              title=f"Spectrum — scan {i}, RT {rt_all[i]:.1f} s (empty)")
        mz = P[:, 0]; inten = P[:, 1]
        # A timsTOF MS1 frame holds one peak per (m/z, ion-mobility) cell, i.e.
        # hundreds of thousands of points. Collapse the mobility dimension by
        # summing intensity into fine m/z bins so it reads as a normal mass
        # spectrum, then cap the number of drawn sticks so the browser stays fast.
        mzmin, mzmax = float(mz.min()), float(mz.max())
        width = 0.01                                   # display m/z bin (Da)
        nb = max(int(np.ceil((mzmax - mzmin) / width)) + 1, 1)
        idx = np.clip(((mz - mzmin) / width).astype(np.int64), 0, nb - 1)
        summed = np.zeros(nb)
        np.add.at(summed, idx, inten)
        nz = np.nonzero(summed)[0]
        centers = mzmin + width * (nz + 0.5)
        vals = summed[nz]
        MAXSTICKS = 6000
        capped = vals.size > MAXSTICKS
        if capped:                                     # keep the most intense m/z bins
            top = np.argpartition(vals, -MAXSTICKS)[-MAXSTICKS:]
            order = np.argsort(centers[top])
            centers, vals = centers[top][order], vals[top][order]
        note = f" · top {vals.size:,} peaks" if capped else f" · {vals.size:,} peaks"
        mob_clause = " (summed over 1/K0)" if sp.has_4d else ""
        title = f"Spectrum — scan {i}, RT {rt_all[i]:.1f} s{mob_clause}{note}"
        ymax = float(vals.max()) if vals.size else 1.0
        return hv.Spikes((centers, vals), "m/z", ydim).opts(
            width=820, height=300, title=title, color="#1b3b6f",
            xlim=(mzmin, mzmax), ylim=(0, ymax * 1.05),
            framewise=True, axiswise=True,            # own ranges, recomputed per click
            tools=["xwheel_zoom", "box_zoom", "pan", "hover", "reset"],
            active_tools=["xwheel_zoom"])             # scroll zooms m/z only; intensity scale stays

    # ── extracted-ion chromatogram + mobilogram (on demand) ───────────────
    @staticmethod
    def _parse_mz_list(text):
        """Parse 'a:500.25, b:712.4' or '500.25, 712.4' into [(label, mz), ...].

        A label may be given as ``label:mz`` or ``label=mz``; otherwise the m/z
        itself is the label. Blank / unparseable entries are skipped.
        """
        out = []
        for part in str(text or "").split(","):
            part = part.strip()
            if not part:
                continue
            label = None
            if ":" in part:
                label, val = part.split(":", 1)
            elif "=" in part:
                label, val = part.split("=", 1)
            else:
                val = part
            try:
                mz = float(val.strip())
            except ValueError:
                continue
            if mz > 0:
                out.append((label.strip() if label else f"{mz:g}", mz))
        return out

    def _render_xic(self, event=None):
        if self.sp is None:
            self.xic_panel[:] = [pn.pane.Markdown("Load data first.")]
            return
        targets = self._parse_mz_list(self.xic_mz.value)
        if not targets:
            self.xic_panel[:] = [pn.pane.Markdown("Enter one or more m/z (> 0), then Extract.")]
            return
        tol = self.xic_tol.value or 0.01
        try:
            import colorcet as cc
            sp = self._current_selection()
            sd = sp.spectra_data(["rtime"]).reset_index(drop=True)
            rt_all = sd["rtime"].to_numpy(dtype="float64"); nsc = len(sd)
            want_im = sp.has_4d
            cols = ("mz", "intensity", "ion_mobility") if want_im else ("mz", "intensity")
            K = len(targets)
            single = (K == 1)
            los = np.array([mz - tol for _, mz in targets])
            his = np.array([mz + tol for _, mz in targets])

            # ── single streaming pass: per-target XIC + mobility, plus the
            #    single-ion RT×1/K0 footprint (only meaningful for one m/z) ──
            xic = np.zeros((K, nsc))
            mob_v = [[] for _ in range(K)]; mob_i = [[] for _ in range(K)]
            foot_rt, foot_mob, foot_int = [], [], []
            for i, P in sp.iter_peaks(columns=cols):
                if P.shape[0] == 0:
                    continue
                mzcol = P[:, 0]; inten = P[:, 1]
                for k in range(K):
                    m = (mzcol >= los[k]) & (mzcol <= his[k])
                    if not m.any():
                        continue
                    xic[k, i] = inten[m].sum()
                    if want_im:
                        mob_v[k].append(P[m, 2]); mob_i[k].append(inten[m])
                        if single:
                            foot_rt.append(np.full(int(m.sum()), rt_all[i]))
                            foot_mob.append(P[m, 2]); foot_int.append(inten[m])

            palette = ["#1b3b6f", "#c1121f", "#2a9d8f", "#e76f51", "#6a4c93",
                       "#0077b6", "#bc6c25", "#588157"]
            order = np.argsort(rt_all); rts = rt_all[order]
            ydim_x = "intensity (XIC)"
            xmax = float(xic.max()) if xic.size and xic.max() > 0 else 1.0
            xic_curves = [
                hv.Curve((rts, xic[k][order]), "rt", ydim_x, label=lbl).opts(
                    color=palette[k % len(palette)])
                for k, (lbl, _) in enumerate(targets)
            ]
            xic_plot = hv.Overlay(xic_curves).opts(
                width=820, height=300, title="XIC" + (f" — m/z {targets[0][1]} ± {tol}" if single else f" — {K} ions, ± {tol} Da"),
                xlabel="RT (s)", ylim=(0, xmax * 1.05), framewise=True, axiswise=True,
                legend_position="top_right", show_legend=(K > 1),
                tools=["xwheel_zoom", "box_zoom", "pan", "hover", "reset"], active_tools=["xwheel_zoom"])
            panes = [pn.pane.HoloViews(xic_plot, sizing_mode="stretch_width")]

            # overlaid mobilograms (shared 1/K0 bins so curves are comparable)
            if want_im and any(mob_v[k] for k in range(K)):
                allmob = np.concatenate([np.concatenate(mob_v[k]) for k in range(K) if mob_v[k]])
                if allmob.size and allmob.max() > allmob.min():
                    bins = np.linspace(allmob.min(), allmob.max(), 160)
                    centers = (bins[:-1] + bins[1:]) / 2
                    ydim_m = "intensity (mobilogram)"
                    mmax = 0.0; mob_curves = []
                    for k, (lbl, _) in enumerate(targets):
                        summ = np.zeros(len(bins) - 1)
                        if mob_v[k]:
                            mv = np.concatenate(mob_v[k]); mi = np.concatenate(mob_i[k])
                            idx = np.clip(np.digitize(mv, bins) - 1, 0, len(bins) - 2)
                            np.add.at(summ, idx, mi)
                            mmax = max(mmax, float(summ.max()))
                        mob_curves.append(hv.Curve((centers, summ), "1/K0", ydim_m, label=lbl).opts(
                            color=palette[k % len(palette)]))
                    mob_plot = hv.Overlay(mob_curves).opts(
                        width=820, height=240, title="Mobilogram" + ("" if single else " (overlaid)"),
                        ylim=(0, (mmax or 1.0) * 1.05), framewise=True, axiswise=True,
                        legend_position="top_right", show_legend=(K > 1),
                        tools=["xwheel_zoom", "box_zoom", "pan", "hover", "reset"], active_tools=["xwheel_zoom"])
                    panes.append(pn.pane.HoloViews(mob_plot, sizing_mode="stretch_width"))

            # single-ion RT×1/K0 footprint (where the ion sits in BOTH dimensions)
            if single and want_im and foot_rt:
                import datashader as dsh
                fr = np.concatenate(foot_rt); fm = np.concatenate(foot_mob); fi = np.concatenate(foot_int)
                rt_lo, rt_hi = float(self.rt.value[0]), float(self.rt.value[1])
                im_lo, im_hi = float(self.im.value[0]), float(self.im.value[1])
                if rt_hi <= rt_lo: rt_hi = rt_lo + 1e-6
                if im_hi <= im_lo: im_hi = im_lo + 1e-6
                cvs = dsh.Canvas(plot_width=820, plot_height=300,
                                 x_range=(rt_lo, rt_hi), y_range=(im_lo, im_hi))
                agg = cvs.points(pd.DataFrame({"rt": fr, "im": fm, "intensity": fi}),
                                 "rt", "im", dsh.sum("intensity"))
                foot = hv.Image(agg.where(agg > 0)).opts(
                    width=820, height=300, cmap=cc.bmy, cnorm="eq_hist", colorbar=True,
                    clabel="summed intensity", title=f"RT × 1/K0 footprint — m/z {targets[0][1]} ± {tol}",
                    xlabel="RT (s)", ylabel="1/K0",
                    tools=["hover", "wheel_zoom", "box_zoom", "pan", "reset"], active_tools=["wheel_zoom"])
                panes.append(pn.pane.HoloViews(foot, sizing_mode="stretch_width"))
                panes.append(pn.pane.Markdown(
                    "_Footprint: where this ion sits in **both** RT and ion mobility — it pinpoints "
                    "which XIC peak goes with which mobility._"))
            elif not single:
                panes.append(pn.pane.Markdown(
                    "_The RT × 1/K0 footprint is shown when a **single** m/z is entered._"))

            # per-target apex summary
            lines = []
            for k, (lbl, mz) in enumerate(targets):
                xs = xic[k][order]
                if xs.max() > 0:
                    lines.append(f"**{lbl}** (m/z {mz}): apex RT ≈ {rts[int(np.argmax(xs))]:.1f} s · max {xs.max():,.0f}")
                else:
                    lines.append(f"**{lbl}** (m/z {mz}): no signal in this selection")
            panes.append(pn.pane.Markdown("  \n".join(lines)))
            self.xic_panel[:] = panes
        except Exception as e:
            self.xic_panel[:] = [pn.pane.Markdown(f"❌ XIC error: {e}")]

    # ── diagnostics + precursor builders ──────────────────────────────────
    def _diagnostics(self, res):
        rt_all, npks, lvl_all = res["rt_all"], res["npks"], res["lvl_all"]
        order = np.argsort(rt_all)
        pk = hv.Curve((rt_all[order], npks[order]), "RT (s)", "peaks/scan").opts(
            width=820, height=240, title="Peaks per scan", color="#2a9d8f",
            tools=["hover", "wheel_zoom", "pan", "reset"])
        ms1_rt = np.sort(rt_all[lvl_all == 1])
        items = [pn.pane.HoloViews(pk, sizing_mode="stretch_width")]
        if ms1_rt.size > 1:
            cyc = np.diff(ms1_rt)
            ct = hv.Curve((ms1_rt[:-1], cyc), "RT (s)", "cycle time (s)").opts(
                width=820, height=240, title="MS1 cycle time", color="#e76f51",
                tools=["hover", "wheel_zoom", "pan", "reset"])
            items.append(pn.pane.HoloViews(ct, sizing_mode="stretch_width"))
        items.append(pn.pane.Markdown(
            "_Peaks per scan tracks sample complexity / sensitivity across the run; "
            "MS1 cycle time is the gap between consecutive survey scans._"))
        return pn.Column(*items, sizing_mode="stretch_width")

    def _precursor(self):
        # Precursor data is inherently MS2 and comes from metadata only (fast even
        # for huge MS2 counts), so it ignores the MS-level selector and always
        # reads the run's MS2 scans within the current RT window.
        md = self.sp.spectra_data(["rtime", "ms_level", "precursor_mz"]).reset_index(drop=True)
        rt_lo, rt_hi = float(self.rt.value[0]), float(self.rt.value[1])
        md = md[(md["ms_level"] == 2) & (md["rtime"] >= rt_lo) & (md["rtime"] <= rt_hi)]
        prec = md["precursor_mz"].to_numpy(dtype="float64")
        mask = ~np.isnan(prec)
        if mask.sum() == 0:
            return pn.pane.Markdown(
                "_No MS2 / precursor data in this run (or in the current RT window). Convert with "
                "**MS level = MS2** or **MS1 + MS2 (both)** to populate this tab._")
        pm = prec[mask]; prt = md["rtime"].to_numpy(dtype="float64")[mask]
        hist = hv.Histogram(np.histogram(pm, bins=80)).opts(
            width=820, height=240, title="Precursor m/z distribution",
            xlabel="precursor m/z", ylabel="count", color="#264653", tools=["hover"])
        import datashader as dsh
        cvs = dsh.Canvas(plot_width=820, plot_height=300,
                         x_range=(float(prt.min()), float(prt.max()) + 1e-6),
                         y_range=(float(pm.min()), float(pm.max()) + 1e-6))
        agg = cvs.points(pd.DataFrame({"rt": prt, "precursor_mz": pm}), "rt", "precursor_mz", dsh.count())
        dens = hv.Image(agg.where(agg > 0)).opts(
            width=820, height=300, cmap="blues", cnorm="eq_hist", colorbar=True,
            title="Precursor m/z vs RT (density)", xlabel="RT (s)", ylabel="precursor m/z",
            clabel="count", tools=["hover", "wheel_zoom", "pan", "reset"])
        return pn.Column(
            pn.pane.Markdown(f"_{int(mask.sum()):,} MS2 precursor selections in this RT window "
                             f"(independent of the MS-level selector)._"),
            pn.pane.HoloViews(hist, sizing_mode="stretch_width"),
            pn.pane.HoloViews(dens, sizing_mode="stretch_width"),
            sizing_mode="stretch_width")

    # ── DIA: acquisition-mode detection + window-scheme view ───────────────
    def _is_dia(self) -> bool:
        """True if the loaded store is DIA.

        Prefers the ``acquisition_mode`` recorded at conversion time. Falls back
        to a structural heuristic, because the mere *presence* of isolation
        columns does NOT imply DIA -- DDA also records an isolation window (a
        narrow one) around each selected precursor. DIA is distinguished by a
        SMALL number of repeating isolation windows reused across the run, while
        DDA has very many distinct precursor m/z (≈ one per MS2 scan)."""
        if self.sp is None:
            return False
        mode = getattr(self.sp._backend, "acquisition_mode", "unknown")
        if mode in ("dia", "dda"):
            return mode == "dia"
        # heuristic fallback for stores with no recorded mode
        try:
            md = self.sp.spectra_data(["ms_level", "isolation_mz"])
            ms2 = md[md["ms_level"] == 2]
            n_ms2 = len(ms2)
            if n_ms2 == 0 or "isolation_mz" not in ms2.columns:
                return False
            n_distinct = ms2["isolation_mz"].round(1).nunique()
            # DIA reuses a fixed window set: few distinct centres relative to scans.
            # DDA targets a fresh precursor most scans: many distinct centres.
            return n_distinct <= max(80, n_ms2 * 0.05)
        except Exception:
            return False

    def _window_scheme(self):
        """diaPASEF isolation windows as boxes in m/z × 1/K0 — the DIA analogue of
        the DDA 'Precursor' tab. The scheme is a property of the acquisition
        method, so it is shown whole-run, independent of the view sliders."""
        try:
            wanted = ["ms_level", "window_group", "window_id", "isolation_mz",
                      "isolation_mz_lower", "isolation_mz_upper",
                      "isolation_im_lower", "isolation_im_upper"]
            # request only columns the store actually has (3D / DDA stores lack the
            # mobility-bound columns, and asking for a missing column errors)
            available = set(self.sp.spectra_data().columns)
            cols = [c for c in wanted if c in available]
            md = self.sp.spectra_data(cols)
            if "ms_level" in md.columns:
                md = md[md["ms_level"] == 2]
            if "window_id" in md.columns:
                md = md.drop_duplicates("window_id")
            if (len(md) == 0 or "isolation_mz_lower" not in md.columns
                    or md["isolation_mz_lower"].isna().all()):
                return pn.pane.Markdown("_No DIA window-scheme metadata in this store._")
            ngroup = int(md["window_group"].nunique()) if "window_group" in md.columns else 1
            have_im = ("isolation_im_lower" in md.columns
                       and bool(md["isolation_im_lower"].notna().any()))

            if have_im:
                y0 = md["isolation_im_lower"].to_numpy(dtype="float64")
                y1 = md["isolation_im_upper"].to_numpy(dtype="float64")
                ylabel = "1/K0 (mobility)"
            elif "window_group" in md.columns:
                # fallback when a store predates mobility bounds: stack by group so
                # the m/z tiling is still visible
                g = md["window_group"].to_numpy(dtype="float64")
                y0, y1 = g - 0.4, g + 0.4
                ylabel = "window group (no 1/K0 bounds stored)"
            else:
                y0 = np.zeros(len(md)); y1 = np.ones(len(md))
                ylabel = "(no group / mobility bounds stored)"

            rect = pd.DataFrame({
                "x0": md["isolation_mz_lower"].to_numpy("float64"),
                "y0": y0,
                "x1": md["isolation_mz_upper"].to_numpy("float64"),
                "y1": y1,
                "group": (md["window_group"].astype(int).astype(str).to_numpy()
                          if "window_group" in md.columns
                          else np.array(["1"] * len(md))),
                "mz": (md["isolation_mz"].to_numpy("float64")
                       if "isolation_mz" in md.columns
                       else np.full(len(md), np.nan)),
            })
            # colour by window group; hover carries the group + centre m/z. Legend is
            # left off deliberately (HoloViews Rectangles legends are unreliable) —
            # colour shows the cycling pattern and hover gives the specifics.
            boxes = hv.Rectangles(rect, kdims=["x0", "y0", "x1", "y1"],
                                  vdims=["group", "mz"]).opts(
                width=820, height=460, alpha=0.55, line_color="white", line_width=0.6,
                color="group", cmap="Category10", colorbar=False, show_legend=False,
                title=f"diaPASEF isolation windows — {len(md)} windows in {ngroup} groups",
                xlabel="m/z", ylabel=ylabel,
                tools=["hover", "wheel_zoom", "box_zoom", "pan", "reset"],
                active_tools=["wheel_zoom"])
            caption = (
                "_Each box is one isolation window: width = its **m/z** range, height = "
                "its **1/K0** range; colour = window group (one acquisition cycle). The "
                "staircase along the diagonal is the diaPASEF design — the quadrupole "
                "tracks the peptide m/z–mobility correlation. Hover a box for its group "
                "and centre m/z. This is the acquisition scheme, so it is the whole run "
                "regardless of the view sliders._"
                if have_im else
                "_Each box is one isolation window's **m/z** range; this store has no "
                "1/K0 bounds, so windows are stacked by group. Reconvert to add them._")
            return pn.Column(
                pn.pane.HoloViews(boxes, sizing_mode="stretch_width"),
                pn.pane.Markdown(caption), sizing_mode="stretch_width")
        except Exception as e:
            return pn.pane.Markdown(f"❌ **Window-scheme view error:** {e}")

    # ── render everything ──────────────────────────────────────────────────
    async def _render_comparison(self, event=None):
        """Compute mzQC ID-free metrics for every loaded sample, tabulate, flag outliers."""
        names = list(self._samples)
        if len(names) < 2:
            self.cmp_panel[:] = [pn.pane.Markdown(
                "_Load 2+ samples (open a folder of .zarr stores, or several paths) to compare._")]
            return
        self.cmp_panel.loading = True
        try:
            opened = {n: open_zarr(p) for n, p in self._samples.items()}
            tbl = compare_samples(opened)
            flags = flag_outliers(tbl, names)
            summary = flag_summary(flags, names)
            self.cmp_panel[:] = self._comparison_panels(tbl, flags, summary, names)
        except Exception as e:
            self.cmp_panel[:] = [pn.pane.Markdown(f"❌ **Comparison error:** {e}")]
        finally:
            self.cmp_panel.loading = False

    @staticmethod
    def _format_comparison(tbl, names):
        """Format the comparison matrix for display (metric as row label, %.4g cells)."""
        disp = tbl.copy()
        for c in list(names) + (["cv_pct"] if "cv_pct" in disp.columns else []):
            disp[c] = disp[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
        if "ms_level" in disp.columns:
            disp["ms_level"] = disp["ms_level"].map(lambda v: "" if pd.isna(v) else str(int(v)))
        return disp.set_index("metric")

    # outlier-cell colours: agreement of both methods is the strongest signal
    _FLAG_COLORS = {"both": "#f3a3a3", "z": "#ffd6a0", "iqr": "#fff2a8"}

    def _comparison_panels(self, tbl, flags, summary, names):
        """Assemble the comparison tab: highlighted matrix + legend + tally + detail."""
        disp = self._format_comparison(tbl, names)
        style = pd.DataFrame("", index=disp.index, columns=disp.columns)
        for _, f in flags.iterrows():
            key = "both" if (f["flag_z"] and f["flag_iqr"]) else ("z" if f["flag_z"] else "iqr")
            if f["metric"] in style.index and f["sample"] in style.columns:
                style.loc[f["metric"], f["sample"]] = (
                    f"background-color: {self._FLAG_COLORS[key]}; font-weight: 600")
        html = (disp.style.apply(lambda _: style, axis=None)
                    .set_table_styles([{"selector": "td, th",
                                        "props": [("padding", "2px 8px"),
                                                  ("font-size", "12px")]}])
                    .to_html())
        legend = pn.pane.HTML(
            "<b>Outlier flags</b> &mdash; "
            "<span style='background:#f3a3a3'>&nbsp;both methods&nbsp;</span> "
            "<span style='background:#ffd6a0'>&nbsp;z-score only&nbsp;</span> "
            "<span style='background:#fff2a8'>&nbsp;IQR only&nbsp;</span>. "
            "Modified z-score (median/MAD, |z|&gt;3.5) and Tukey fences (Q1/Q3 ± 1.5·IQR), "
            "per metric across samples. Flags require a \u22655% deviation from the median, "
            "so trivial differences are ignored. <code>cv_pct</code> is spread only (not "
            "flagged). Metrics with &lt;3 samples are not evaluated.")
        if flags.empty:
            detail = pn.pane.Markdown("_No samples flagged as outliers on any metric._")
        else:
            d = flags.copy()
            d["robust_z"] = d["robust_z"].map(lambda v: f"{v:.2f}")
            d["dev_pct"] = d["rel_dev"].map(lambda v: f"{v * 100:.1f}%")
            d["value"] = d["value"].map(lambda v: f"{v:.4g}")
            d["by"] = d.apply(lambda r: "both" if (r["flag_z"] and r["flag_iqr"])
                              else ("z" if r["flag_z"] else "IQR"), axis=1)
            detail = pn.pane.DataFrame(
                d[["metric", "sample", "value", "robust_z", "dev_pct", "direction", "by"]].reset_index(drop=True),
                width=720, index=False)
        plot_panes = [
            pn.pane.Matplotlib(plot_zscore_heatmap(tbl, names, flags), format="png",
                               dpi=96, sizing_mode="stretch_width"),
            pn.pane.Matplotlib(plot_flag_tally(summary), format="png", dpi=96),
        ]
        if not flags.empty:
            plot_panes.append(pn.pane.Matplotlib(
                plot_metric_strips(tbl, names, flags), format="png", dpi=96,
                sizing_mode="stretch_width"))
        plots_card = pn.Card(*plot_panes, title="Plots", collapsed=True,
                             sizing_mode="stretch_width")
        return [
            pn.pane.HTML(html, styles={"overflow-x": "auto", "max-width": "100%"}),
            legend,
            pn.pane.Markdown(_FLAG_GUIDE),
            pn.pane.Markdown("**Per-sample outlier tally** (most-flagged first):"),
            pn.pane.DataFrame(summary, width=720, index=False),
            pn.pane.Markdown("**Flagged metric–sample pairs:**"),
            detail,
            plots_card,
        ]

    def _render(self, event=None):
        if self.sp is None:
            return
        try:
            sp = self._current_selection()
            res = self._full_overview(sp)
            rt_all = res["rt_all"]

            # chromatogram + click-to-spectrum (Tap stream)
            chrom = (hv.Curve(res["tic"], "rt", "intensity", label="TIC").opts(color="#1b3b6f")
                     * hv.Curve(res["bpc"], "rt", "intensity", label="BPC").opts(color="#c1121f")).opts(
                width=820, height=300, title=f"Chromatogram — {res['n']} scans", xlabel="RT (s)",
                legend_position="top_right", tools=["tap", "hover", "wheel_zoom", "pan", "reset"])
            x0 = float(rt_all[len(rt_all) // 2]) if len(rt_all) else 0.0
            tap = hv.streams.Tap(source=chrom, x=x0)
            spec_dmap = hv.DynamicMap(lambda x, y=None: self._spectrum_at(sp, rt_all, x), streams=[tap])
            marker = hv.DynamicMap(
                lambda x, y=None: hv.VLine(x).opts(color="gray", line_dash="dashed", line_width=1),
                streams=[tap])
            # lock the overlay's frame size so the dynamic marker can't resize it on click
            chrom_view = (chrom * marker).opts(width=820, height=300)
            chrom_tab = pn.Column(
                pn.pane.HoloViews(chrom_view, width=860, height=340),
                pn.pane.Markdown("_Click anywhere on the chromatogram to show that scan's spectrum below "
                                 "(summed over ion mobility)._"),
                pn.pane.HoloViews(spec_dmap, width=860, height=340),
                sizing_mode="stretch_width")
            tabs = [("Chromatogram + spectrum", chrom_tab)]

            def _map_tab(img):
                return pn.Column(pn.pane.HoloViews(img, sizing_mode="stretch_width"),
                                 pn.pane.Markdown(_COLOR_HELP), sizing_mode="stretch_width")
            if res["fm"] is not None:
                tabs.append(("Feature map (RT × m/z)", _map_tab(res["fm"])))
            if res["mob"] is not None:
                tabs.append(("Mobility (m/z × 1/K0)", _map_tab(res["mob"])))
            if res["rtim"] is not None:
                tabs.append(("RT × 1/K0", _map_tab(res["rtim"])))

            xic_tab = pn.Column(
                pn.Row(self.xic_mz, self.xic_tol, pn.Column(pn.Spacer(height=18), self.xic_btn)),
                pn.pane.Markdown(
                    "_Enter one m/z for an XIC + mobilogram + RT×1/K0 footprint, or several "
                    "(comma-separated) to overlay their XICs and mobilograms. Streams the current "
                    "selection — narrow the RT slider first for speed on a large run._"),
                self.xic_panel, sizing_mode="stretch_width")
            tabs.append(("XIC", xic_tab))
            tabs.append(("Diagnostics", self._diagnostics(res)))
            # mode-aware single tab: DIA shows the window scheme, DDA the precursors
            if self._is_dia():
                tabs.append(("Window scheme", self._window_scheme()))
            else:
                tabs.append(("Precursor", self._precursor()))
            # QC: peak-derived metrics reflect the current view, but scan COUNTS
            # and the MS1 cycle time reflect the whole run (cheap metadata read) so
            # they don't read as "0 MS2" / NaN when you're looking at the MS1 (or
            # MS2) view — matching the Precursor tab.
            qc_df = res["qc"].copy()
            md_all = self.sp.spectra_data(["ms_level", "rtime"])
            lv_all = md_all["ms_level"].to_numpy()
            rt_full = md_all["rtime"].to_numpy(dtype="float64")
            n_tot = int(lv_all.size); n1 = int((lv_all == 1).sum()); n2 = int((lv_all == 2).sum())
            qc_df["n_spectra"] = n_tot
            qc_df["n_ms1"] = n1
            qc_df["n_ms2"] = n2
            qc_df["ms2_per_ms1"] = (n2 / n1) if n1 else np.nan
            ms1_rt_full = np.sort(rt_full[lv_all == 1])
            qc_df["median_ms1_cycle_time_s"] = (
                float(np.median(np.diff(ms1_rt_full))) if ms1_rt_full.size > 1 else np.nan)
            tabs.append(("QC", pn.Column(
                pn.pane.DataFrame(qc_df.T, width=460),
                pn.pane.Markdown(
                    "_Scan counts (n_spectra / n_ms1 / n_ms2 / ms2_per_ms1) and the MS1 cycle time are "
                    "for the **whole run**. The peak-derived metrics and the RT / m/z / 1/K0 ranges reflect "
                    "the **current MS-level + range selection**._"))))

            if len(self._samples) > 1:
                tabs.append(("Sample comparison", pn.Column(
                    pn.Row(self.cmp_btn, pn.pane.Markdown(
                        "_mzQC ID-free metrics for every loaded sample, computed on each sample's "
                        "full data. `cv_pct` is the spread across samples — a high value flags a "
                        "metric that differs between runs. Per-sample outlier flagging comes next._")),
                    self.cmp_panel, sizing_mode="stretch_width")))

            self.plot_panel[:] = [pn.Tabs(*tabs)]
        except Exception as e:
            self.plot_panel[:] = [pn.pane.Markdown(f"❌ **Render error:** {e}")]

    async def _on_update(self, event=None):
        if self.sp is None:
            return
        prev = self.status.object
        self.status.object = "⏳ Updating plots for the current selection — progress prints in the server terminal."
        self.plot_panel.loading = True
        await asyncio.sleep(0.05)
        try:
            self._render()
        finally:
            self.plot_panel.loading = False
            self.status.object = prev

    async def _on_xic(self, event=None):
        self.xic_panel.loading = True
        await asyncio.sleep(0.05)
        try:
            self._render_xic()
        finally:
            self.xic_panel.loading = False

    # ── layout ─────────────────────────────────────────────────────────────
    def view(self):
        from ._browse import browse_field
        open_help = ("One .zarr store, or a folder of .zarr stores to load several samples at once "
                     "(several paths also work — one per line or comma-separated). With 2+ samples, "
                     "switch between them with the Sample selector on the View panel.")
        convert_help = ("One raw file converts to a single store; a folder or several paths convert "
                        "in parallel, each to its own .zarr. The result(s) load as samples. PTERO "
                        "detects profile vs centroid: a .d is already centroid (instrument peak-picked), "
                        "and a single profile mzML is centroided on convert (needs pyopenms). Note: the "
                        "parallel path does not centroid profile mzML yet — convert those one at a time "
                        "for now.")
        open_section = pn.Column(
            pn.Row(pn.pane.Markdown("## 1 · Open .zarr"),
                   pn.widgets.TooltipIcon(value=open_help), align="center"),
            pn.Row(browse_field(self.zarr_path, patterns=["*.zarr"], width=620),
                   pn.Column(pn.Spacer(height=18), self.open_btn)),
            sizing_mode="stretch_width")
        convert_section = pn.Column(
            pn.Row(pn.pane.Markdown("## 2 · Convert to .zarr"),
                   pn.widgets.TooltipIcon(value=convert_help), align="center"),
            browse_field(self.conv_input, patterns=["*.d", "*.mzML"], width=620),
            pn.Row(self.conv_outdir, self.conv_ms, self.conv_frames, self.conv_batch),
            pn.Row(self.conv_workers, self.conv_overwrite,
                   pn.Column(pn.Spacer(height=18), self.conv_btn)),
            sizing_mode="stretch_width")
        source = pn.Column(
            open_section,
            pn.layout.Divider(),
            convert_section,
            self.status, sizing_mode="stretch_width")
        controls = pn.Column(
            pn.pane.Markdown("## 3 · View"),
            self.sample_sel,
            self.ms_sel, self.rt, self.mz, self.im, self.update_btn, width=340)
        return pn.Column(
            pn.pane.Markdown("# PTERO — Raw Data Dashboard"),
            source, pn.layout.Divider(),
            pn.Row(controls, self.plot_panel, sizing_mode="stretch_width"),
            sizing_mode="stretch_width")


def build_raw_page():
    """The raw-data / converter page (the original single-page dashboard)."""
    return Dashboard().view()


def build_app():
    """Build the multi-page PTERO dashboard and return a servable template.

    Pages share one :class:`~ptero.dashboard._state.AppState` so inputs loaded on
    one page are visible to others. The converter / raw-data dashboard is the
    main page; the ID-annotation page is page 1. More pages slot in here later.

    Returned object is a Panel template; call ``.servable()`` to serve it.
    """
    from ._state import AppState
    from .ids_view import IdsView

    state = AppState()

    raw_page = build_raw_page()
    ids_page = IdsView(state).view()

    template = pn.template.FastListTemplate(
        title="PTERO",
        sidebar=[pn.pane.Markdown("### Pages\nUse the tabs in the main area to switch pages.")],
        main=[pn.Tabs(
            ("Raw data", raw_page),
            ("Identifications", ids_page),
        )],
    )
    return template