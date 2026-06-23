"""
Convert a raw run (.d / mzML) into a PTERO Zarr store -- memory-safe.

Large timsTOF runs hold hundreds of millions of detector events, far more than
fits in RAM. So instead of reading the whole run into one array, this walks the
run in **batches** (frame batches for Bruker .d, spectrum batches for mzML) and
**streams** each batch to the Zarr store, releasing it before reading the next.
Peak memory is set by one batch, not the whole run.

The intended workflow for non-coders: convert once (creating a .zarr folder on
disk), then open that store in the dashboard -- it loads instantly and lazily.
"""
from __future__ import annotations

import os
from typing import Optional, Callable

import numpy as np

from ..spectra import from_mzml
from ..spectra import _dia
from ..spectra._bruker import (
    _open_timstof, _read_timstof_frames, _timstof_n_frames, from_timstof_dataframe,
)
from ..spectra.backends.memory import MsBackendMemory
from ..spectra.backends.zarr_backend import _ZarrStreamWriter


def _is_bruker_d(path: str) -> bool:
    low = path.lower().rstrip("\\/")
    return low.endswith(".d") or (
        os.path.isdir(path) and os.path.exists(os.path.join(path, "analysis.tdf"))
    )


def convert_to_zarr(input_path, output_path, ms_level: Optional[int] = 1,
                    frames: Optional[tuple] = None, frame_batch: int = 500,
                    spectrum_batch: int = 2000,
                    transform: Optional[Callable[[dict], dict]] = None,
                    progress: Optional[Callable[[str], None]] = None,
                    log_prefix: Optional[str] = None) -> str:
    """Read a raw run in batches and stream it to a Zarr store; return the path.

    Parameters
    ----------
    input_path : str | Path
        A Bruker ``.d`` folder, or an ``.mzML`` / ``.mzXML`` file.
    output_path : str | Path
        Destination ``.zarr`` directory (created/overwritten).
    ms_level : int or None, default 1
        For ``.d``: 1 = MS1, 2 = PASEF MS2, ``None`` = both levels in one store.
        For mzML: ``None`` = all levels.
    frames : (start, stop), optional
        Bruker only -- restrict to a frame range (stop exclusive). ``None`` reads
        the whole run.
    frame_batch : int, default 500
        Bruker only -- frames read per batch. Smaller uses less memory.
    spectrum_batch : int, default 2000
        mzML only -- spectra read per batch.
    transform : callable, optional
        A per-spectrum ``peaks_dict -> peaks_dict`` map applied during conversion
        (e.g. ``ptero.process.make_centroid_transform()`` to centroid profile data
        inline, so the store is written centroid with no profile intermediate).
        mzML only; raises for Bruker ``.d`` (already centroided).
    progress : callable, optional
        Called with short status strings as conversion proceeds.
    log_prefix : str, optional
        Tag inserted into the printed progress lines (``[convert:<prefix>] ...``)
        so that, when several conversions stream to one console at once (batch
        mode), each line says which file it belongs to.
    """
    p = str(input_path)
    out = str(output_path)
    tag = "convert" if not log_prefix else f"convert:{log_prefix}"

    def _log(msg: str) -> None:
        print(f"[{tag}] {msg}", flush=True)
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass

    if _is_bruker_d(p):
        if transform is not None:
            raise NotImplementedError(
                "transform/centroiding is for mzML profile data; a Bruker .d is "
                "already centroided and does not need it.")
        return _convert_bruker(p, out, ms_level, frames, frame_batch, _log)
    if p.lower().endswith((".mzml", ".mzxml")):
        return _convert_mzml(p, out, ms_level, spectrum_batch, _log, transform=transform)
    raise ValueError(
        f"Unsupported input {p!r}; expected a Bruker .d folder or a .mzML/.mzXML file."
    )


def _convert_bruker(path, out, ms_level, frames, frame_batch, _log) -> str:
    _log(f"opening {path} (indexing) …")
    data = _open_timstof(path)
    n_frames = _timstof_n_frames(data)

    # Decide DDA vs DIA from the analysis.tdf tables (robust; not a width heuristic).
    try:
        acq_mode = _dia.detect_acquisition_mode(path)
    except Exception:
        acq_mode = "unknown"
    if acq_mode == "dia":
        _log("diaPASEF detected (DIA window-scheme tables present).")
        return _convert_bruker_dia(data, path, out, ms_level, frames, frame_batch, n_frames, _log)
    if acq_mode == "ambiguous":
        _log("WARNING: both DIA and DDA tables look populated; converting as ddaPASEF. "
             "Inspect the run if results look wrong.")

    label = {1: "MS1", 2: "MS2"}.get(ms_level, "MS1+MS2")
    start = 0 if frames is None else int(frames[0])
    if frames is not None:
        stop = int(frames[1])
    else:
        stop = n_frames                      # may be None if the reader didn't expose it
    total = (stop - start) if stop is not None else None
    _log(f"reading {label} in batches of {frame_batch} frames"
         + (f" over frames {start}..{stop}" if stop is not None else " (walking to end of run)"))

    writer = _ZarrStreamWriter(out, has_4d=True,
                               acquisition_mode=(acq_mode if acq_mode == "dda" else "unknown"))
    f = start
    n_spec = n_peaks = 0
    empty_streak = 0
    SAFETY_CAP = 50_000_000                   # absolute guard if frame count is unknown
    while True:
        if stop is not None and f >= stop:
            break
        e = (f + frame_batch) if stop is None else min(f + frame_batch, stop)
        df = _read_timstof_frames(data, ms_level, (f, e))
        if len(df):
            empty_streak = 0
            mem = from_timstof_dataframe(df, ms_level=None).backend
            writer.append_backend(mem)
            n_spec += len(mem._meta)
            n_peaks += int(mem._offsets[-1])
            pct = f" [{min(e - start, total)}/{total} frames]" if total else ""
            _log(f"frames {f}-{e}: +{len(mem._meta)} spectra"
                 f" ({n_peaks:,} peaks total){pct}")
        else:
            empty_streak += 1
            if stop is None and empty_streak >= 2:    # two empty batches => past end of run
                break
        f = e
        if stop is None and f > SAFETY_CAP:
            _log("reached safety cap; stopping")
            break

    if n_spec == 0:
        raise ValueError(
            "No spectra were read. Check the path, the MS level, and (if set) the frame range."
        )
    _log(f"finalizing: writing metadata for {n_spec} spectra, {n_peaks:,} peaks …")
    writer.finalize()
    _log(f"done -> {out}")
    return out


def _convert_bruker_dia(data, path, out, ms_level, frames, frame_batch, n_frames, _log) -> str:
    """Stream a diaPASEF run to a Zarr store, batch by batch.

    Unlike the DDA path, MS2 is grouped per **isolation window** (not per
    precursor). Each batch is read at **both** levels and the MS1/MS2 split is
    taken from the frame->window-group map (a frame is MS2 iff it is in
    ``DiaFrameMsMsInfo``); the requested ``ms_level`` is then applied as a filter.

    Note on cost: a separate MS1- or MS2-only store still *reads* both levels per
    batch and discards the unwanted one. This is correct regardless of the
    vendor reader's DIA precursor-index semantics (which we have not verified for
    level-selective slicing). Reading only the wanted level would be faster but
    would require confirming those semantics on a real file first.
    """
    import warnings

    scheme = _dia.read_window_scheme(path)
    frame_map = _dia.read_frame_window_map(path)
    label = {1: "MS1", 2: "MS2"}.get(ms_level, "MS1+MS2")
    keep_level = ms_level if ms_level in (1, 2) else None
    _log(f"diaPASEF: {len(scheme)} windows in {int(scheme['window_group'].nunique())} groups; "
         f"converting {label} (reads both levels per batch, splits by frame)")

    start = 0 if frames is None else int(frames[0])
    stop = int(frames[1]) if frames is not None else n_frames
    total = (stop - start) if stop is not None else None

    writer = _ZarrStreamWriter(out, has_4d=True, acquisition_mode="dia")
    f = start
    n_spec = n_peaks = n_unassigned = 0
    enriched = False                          # add 1/K0 window bounds once
    empty_streak = 0
    SAFETY_CAP = 50_000_000
    while True:
        if stop is not None and f >= stop:
            break
        e = (f + frame_batch) if stop is None else min(f + frame_batch, stop)
        df = _read_timstof_frames(data, None, (f, e))     # both levels; level set by frame map
        if len(df):
            empty_streak = 0
            if not enriched:
                # the scan->1/K0 axis is run-constant; build it from this batch's
                # events (MS1/MS2 frames both cover the windowed scan range) and
                # attach per-window mobility bounds to the scheme, once.
                scheme = _dia.add_mobility_bounds(scheme, _dia.build_scan_mobility_lookup(df))
                enriched = True
            df = _dia.assign_windows_to_long_table(df, scheme, frame_map, method="scan")
            if keep_level is not None:
                df = df[df["ms_level"] == keep_level]
            if len(df):
                n_unassigned += int(((df["ms_level"] == 2) & (df["window_id"] == -1)).sum())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")       # per-batch drop notice; we log a total
                    mem = from_timstof_dataframe(df).backend
                writer.append_backend(mem)
                n_spec += len(mem._meta)
                n_peaks += int(mem._offsets[-1])
                pct = f" [{min(e - start, total)}/{total} frames]" if total else ""
                _log(f"frames {f}-{e}: +{len(mem._meta)} spectra ({n_peaks:,} peaks total){pct}")
        else:
            empty_streak += 1
            if stop is None and empty_streak >= 2:
                break
        f = e
        if stop is None and f > SAFETY_CAP:
            _log("reached safety cap; stopping")
            break

    if n_spec == 0:
        raise ValueError(
            "No spectra were read. Check the path, the MS level, and (if set) the frame range."
        )
    if n_unassigned:
        _log(f"note: {n_unassigned:,} MS2 events fell outside every isolation window and were dropped.")
    _log(f"finalizing: writing metadata for {n_spec} spectra, {n_peaks:,} peaks …")
    writer.finalize()
    _log(f"done -> {out}")
    return out


def _convert_mzml(path, out, ms_level, spectrum_batch, _log, transform=None) -> str:
    _log(f"opening {path} …")
    sp = from_mzml(path, ms_level=ms_level, lazy=True)
    n = len(sp)
    if n == 0:
        raise ValueError("The mzML file has no spectra at the requested MS level.")
    has_4d = sp.has_4d
    cols = ("mz", "intensity", "ion_mobility") if has_4d else ("mz", "intensity")
    meta_all = sp.spectra_data()              # metadata is tabular and small enough to hold
    _log(f"reading {n} spectra in batches of {spectrum_batch}"
         + (" (4D)" if has_4d else " (3D)")
         + (" with inline transform" if transform is not None else ""))

    writer = _ZarrStreamWriter(out, has_4d=has_4d)
    for s in range(0, n, spectrum_batch):
        ee = min(s + spectrum_batch, n)
        peaks = sp.peaks_data(indices=list(range(s, ee)), columns=cols)
        batch = []
        for pk in peaks:
            d = {"mz": pk[:, 0], "intensity": pk[:, 1]}
            if has_4d:
                d["ion_mobility"] = pk[:, 2]
            if transform is not None:
                d = transform(d)
            batch.append(d)
        mem = MsBackendMemory.from_spectra_list(
            meta_all.iloc[s:ee].reset_index(drop=True), batch
        )
        writer.append_backend(mem)
        _log(f"spectra {s}-{ee}/{n}")

    _log("finalizing …")
    writer.finalize()
    _log(f"done -> {out}")
    return out


# ── batch conversion (many raw files -> many Zarr stores, in parallel) ───────

def _is_raw_input(path) -> bool:
    """True if `path` looks like a convertible raw run (.d folder or mz(X)ML)."""
    from pathlib import Path
    p = Path(str(path))
    if _is_bruker_d(str(p)):
        return True
    return p.suffix.lower() in (".mzml", ".mzxml")


def resolve_batch_inputs(spec) -> list:
    """Turn a batch specification into a concrete list of raw-run paths.

    `spec` may be:
      * a list/tuple of paths,
      * a path to a *folder* to scan (non-recursively) for ``.d`` runs and
        ``.mzML``/``.mzXML`` files -- but if the folder is itself a ``.d`` run
        (contains ``analysis.tdf``) it is treated as a single input, and
      * a string of several paths separated by newlines or commas.
    """
    from pathlib import Path
    if isinstance(spec, (list, tuple)):
        return [str(s).strip() for s in spec if str(s).strip()]
    s = str(spec).strip()
    if not s:
        return []
    p = Path(s)
    if p.is_dir() and not (p / "analysis.tdf").exists():
        found = []
        for child in sorted(p.iterdir()):
            if _is_raw_input(child):
                found.append(str(child))
        return found
    # otherwise: one path, or several separated by newlines/commas
    parts = [x.strip() for x in s.replace(",", "\n").splitlines() if x.strip()]
    return parts


def _plan_outputs(inputs, output_dir) -> list:
    """Map each input to a unique ``<stem>.zarr`` path inside output_dir."""
    from pathlib import Path
    outdir = Path(output_dir)
    used = set()
    plan = []
    for inp in inputs:
        stem = Path(str(inp).rstrip("\\/")).stem or "run"
        cand = outdir / f"{stem}.zarr"
        k = 2
        while str(cand) in used:               # two inputs with the same stem
            cand = outdir / f"{stem}_{k}.zarr"
            k += 1
        used.add(str(cand))
        plan.append((str(inp), str(cand)))
    return plan


def convert_many(inputs, output_dir, ms_level: Optional[int] = 1,
                 frame_batch: int = 500, spectrum_batch: int = 2000,
                 max_workers: int = 2, overwrite: bool = False,
                 on_event: Optional[Callable[[dict], None]] = None) -> list:
    """Convert several raw runs to **separate** Zarr stores, in parallel.

    Each input becomes its own ``<stem>.zarr`` inside ``output_dir``. Work runs
    in independent OS processes (one ``python -m ptero.dashboard.convert`` per
    file), so timsTOF reading -- which is CPU- and I/O-heavy -- actually overlaps,
    crashes are isolated, and it behaves the same on Windows/macOS/Linux. At most
    ``max_workers`` run at once.

    The batch is **fault-tolerant**: if one file fails, it is recorded as an error
    and the rest continue. Existing stores are skipped unless ``overwrite=True``.

    Memory note: each worker holds about one ``frame_batch`` of peaks in RAM, so
    peak memory scales with ``max_workers`` -- raising workers speeds the batch but
    multiplies RAM and disk traffic. Returns a list of per-file result dicts.
    """
    import sys, time, subprocess, tempfile
    from pathlib import Path

    paths = resolve_batch_inputs(inputs)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plan = _plan_outputs(paths, outdir)
    n = len(plan)

    def emit(ev):
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:
                pass

    if n == 0:
        print("[batch] no raw inputs found.", flush=True)
        emit({"event": "batch_done", "ok": 0, "errors": 0, "skipped": 0, "results": []})
        return []

    workers = max(1, min(int(max_workers), n))
    print(f"[batch] converting {n} file(s) with {workers} worker(s) -> {outdir}", flush=True)
    emit({"event": "batch_start", "n_total": n, "workers": workers})

    results: list = []
    running: dict = {}                    # Popen -> meta dict
    queue = list(plan)
    done = 0

    def _launch(inp, outp):
        name = Path(inp).name
        stem = Path(str(inp).rstrip("\\/")).stem
        if Path(outp).exists() and not overwrite:
            return {"input": inp, "output": outp, "status": "skipped",
                    "reason": "output exists", "seconds": 0.0, "_skip": True}
        cmd = [sys.executable, "-m", "ptero.dashboard.convert", inp, outp,
               "--ms-level", ("none" if ms_level is None else str(ms_level)),
               "--frame-batch", str(int(frame_batch)),
               "--spectrum-batch", str(int(spectrum_batch)),
               "--label", stem]
        if overwrite:
            cmd.append("--overwrite")
        # stdout is inherited so progress streams live to the console (each line is
        # self-labelled). stderr goes to a temp file so a failure's traceback can be
        # captured without any pipe-buffer deadlock risk.
        err = tempfile.TemporaryFile(mode="w+")
        proc = subprocess.Popen(cmd, stderr=err)
        return {"input": inp, "output": outp, "proc": proc, "err": err,
                "t0": time.perf_counter(), "name": name}

    try:
        while queue or running:
            # fill the pool
            while queue and len(running) < workers:
                inp, outp = queue.pop(0)
                meta = _launch(inp, outp)
                if meta.get("_skip"):
                    done += 1
                    meta.pop("_skip", None)
                    meta.update(n_done=done, n_total=n)
                    results.append(meta)
                    print(f"[batch] – ({done}/{n}) {Path(inp).name} [skipped: output exists]", flush=True)
                    emit({"event": "file_done", **meta})
                    continue
                running[meta["proc"]] = meta
                emit({"event": "file_start", "input": inp, "output": outp,
                      "name": meta["name"], "n_total": n})
            # reap any finished workers
            for proc in list(running.keys()):
                rc = proc.poll()
                if rc is None:
                    continue
                meta = running.pop(proc)
                secs = time.perf_counter() - meta["t0"]
                meta["err"].seek(0)
                err_txt = meta["err"].read().strip()
                meta["err"].close()
                done += 1
                if rc == 0:
                    res = {"input": meta["input"], "output": meta["output"],
                           "status": "ok", "seconds": secs}
                    print(f"[batch] \u2713 ({done}/{n}) {meta['name']} [ok] {secs:.1f}s", flush=True)
                else:
                    tail = err_txt.splitlines()[-1] if err_txt else f"exit code {rc}"
                    res = {"input": meta["input"], "output": meta["output"],
                           "status": "error", "error": tail,
                           "stderr": err_txt, "seconds": secs}
                    print(f"[batch] \u2717 ({done}/{n}) {meta['name']} [error: {tail}] {secs:.1f}s", flush=True)
                res.update(n_done=done, n_total=n)
                results.append(res)
                emit({"event": "file_done", **res})
            time.sleep(0.1)
    finally:
        # never leave orphans if the parent is interrupted
        for proc, meta in running.items():
            proc.terminate()
            try:
                meta["err"].close()
            except Exception:
                pass

    nok = sum(r["status"] == "ok" for r in results)
    nerr = sum(r["status"] == "error" for r in results)
    nskip = sum(r["status"] == "skipped" for r in results)
    print(f"[batch] done: {nok} ok, {nerr} failed, {nskip} skipped", flush=True)
    emit({"event": "batch_done", "ok": nok, "errors": nerr, "skipped": nskip,
          "results": results})
    return results


def _main(argv=None) -> int:
    """CLI: convert ONE raw run. Used by convert_many as a worker subprocess, but
    also usable directly:  python -m ptero.dashboard.convert run.d run.zarr ...
    """
    import argparse, warnings, shutil
    from pathlib import Path
    warnings.filterwarnings("ignore")        # quiet alphatims/pandas chatter on stderr

    ap = argparse.ArgumentParser(description="Convert one raw MS run to a PTERO Zarr store.")
    ap.add_argument("input", help="Bruker .d folder or .mzML/.mzXML file")
    ap.add_argument("output", help="destination .zarr directory")
    ap.add_argument("--ms-level", default="1",
                    help="1, 2, or 'none' for both levels (default 1)")
    ap.add_argument("--frame-batch", type=int, default=500)
    ap.add_argument("--spectrum-batch", type=int, default=2000)
    ap.add_argument("--label", default=None, help="tag for progress lines")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the output store if it already exists")
    args = ap.parse_args(argv)

    ms = None if str(args.ms_level).lower() in ("none", "both", "") else int(args.ms_level)
    if args.overwrite and Path(args.output).exists():
        shutil.rmtree(args.output, ignore_errors=True)
    convert_to_zarr(args.input, args.output, ms_level=ms,
                    frame_batch=args.frame_batch, spectrum_batch=args.spectrum_batch,
                    log_prefix=args.label)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())