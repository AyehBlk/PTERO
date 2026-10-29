"""
ptero.process._store -- the processed-data Zarr container and the pass-through
driver that produces it.

Scope A, Step 1
---------------
This module owns PTERO's *processed-store* schema and the plumbing that turns a
raw PTERO Zarr store into a processed one. It deliberately contains NO
signal-processing algorithm yet: the default transform is the identity, so the
only thing exercised here is the container + the read/transform/write pipeline.
Centroiding and deisotoping (Steps 2-3) plug in later as a ``transform`` and do
not touch this file.

Design (decided in ROADMAP / NEXT_STEP_preprocessing)
-----------------------------------------------------
* **Separate sibling store, identical schema.** A processed store is structurally
  the same as a raw PTERO Zarr store (``peaks/mz``, ``peaks/intensity``, optional
  ``peaks/ion_mobility``, ``offsets``, ``spectra.parquet``), so it is read back by
  the unchanged ``ptero.spectra.open_zarr`` / :class:`MsBackendZarr` and registered
  into the raw<->ID link with no link changes. The raw store is never modified;
  the processed store is a derived artifact you can delete and rebuild.
* **Reuse the existing writer.** Writing streams through the already-tested
  ``_ZarrStreamWriter`` (the same path ``convert_to_zarr`` uses), so the processed
  store is byte-compatible with every other PTERO store. Provenance is recorded as
  extra root attributes after ``finalize``; :class:`MsBackendZarr` ignores unknown
  attrs, so this is backward/forward compatible (a raw store simply lacks them).
* **Memory-safe by construction.** Peaks are read from the source in large
  contiguous blocks (``iter_peaks_blocked``) and written in batches, so a 50 GB
  run is processed without ever holding it all in RAM -- the same guarantee the
  raw layer makes.
* **The 4D dimension is non-negotiable.** For a 4D source, the driver REQUIRES the
  transform to return a per-peak ``ion_mobility`` array of the right length for
  every spectrum, and raises loudly otherwise -- so processing can never silently
  collapse mobility. (The spectrum-level 1/K0 the ID linker disambiguates on lives
  in ``spectra.parquet`` and is copied through verbatim.)
"""
from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ._transforms import PeaksDict, transform_provenance

#: processed-store schema version (bump if the provenance/layout contract changes)
PROCESSED_SCHEMA_VERSION = 1

#: root-attribute keys, namespaced so they never collide with the raw store's bare
#: ``n_spectra`` / ``has_4d`` / ``acquisition_mode`` / ``peaks_variables`` attrs.
PROCESSED_FLAG_ATTR = "ptero_processed"
PROCESSING_PROVENANCE_ATTR = "ptero_processing"

#: a per-spectrum peak transform.
Transform = Callable[[PeaksDict], PeaksDict]


def identity_transform(peaks: PeaksDict) -> PeaksDict:
    """The Step-1 pass-through: return one spectrum's peaks unchanged.

    This is the seam Steps 2-3 replace with a real per-spectrum centroiding /
    deisotoping transform. A transform receives ``{"mz", "intensity"[,
    "ion_mobility"]}`` (1D numpy arrays of equal length, the peaks of ONE
    spectrum) and must return the same keys, also of equal length. For a 4D
    source it must keep ``ion_mobility``.
    """
    return peaks


def _ptero_version() -> str:
    try:
        from importlib.metadata import version
        return version("ptero")
    except Exception:
        return "unknown"


def _default_steps() -> list:
    return [{"name": "identity", "params": {}, "library": None, "library_version": None}]


def is_processed_store(path) -> bool:
    """True iff the Zarr store at *path* was written by PTERO's processing
    pipeline (i.e. carries the processed marker). A plain raw store returns
    False, and a non-store path returns False rather than raising."""
    import zarr
    try:
        root = zarr.open_group(str(path), mode="r")
    except Exception:
        return False
    return bool(dict(root.attrs).get(PROCESSED_FLAG_ATTR, False))


def read_processing_provenance(path) -> Optional[dict]:
    """Return the processing-provenance dict recorded in a processed store, or
    ``None`` if the store has none (e.g. a raw store)."""
    import zarr
    root = zarr.open_group(str(path), mode="r")
    prov = dict(root.attrs).get(PROCESSING_PROVENANCE_ATTR)
    return dict(prov) if prov is not None else None


def process_store(
    source_path,
    output_path,
    *,
    transform: Optional[Transform] = None,
    steps: Optional[list] = None,
    block_size: int = 4_000_000,
    spectra_per_batch: int = 20_000,
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Read a raw PTERO Zarr store, apply a per-spectrum *transform*, and write a
    processed sibling store; return the output path.

    With the default ``transform`` (identity) this is the Step-1 round-trip: the
    processed store is an exact copy that reopens with :func:`ptero.spectra.open_zarr`
    and runs through the raw<->ID link unchanged.

    Parameters
    ----------
    source_path : str | Path
        An existing PTERO raw Zarr store (3D or 4D).
    output_path : str | Path
        Destination ``.zarr`` directory for the processed store.
    transform : callable, optional
        ``peaks_dict -> peaks_dict`` applied to each spectrum's peaks. Defaults to
        :func:`identity_transform`. The dict has 1D numpy arrays ``mz``,
        ``intensity`` and, for 4D input, ``ion_mobility`` -- all the same length.
        For a 4D source the returned dict MUST keep ``ion_mobility`` (same length
        as ``mz``); the driver raises otherwise, so mobility cannot be lost.
    steps : list, optional
        Provenance descriptors recorded in the store (what was done, with which
        parameters / library). Defaults to a single ``identity`` step.
    block_size : int, default 4_000_000
        Peaks read per contiguous block from the source (the memory bound).
    spectra_per_batch : int, default 20_000
        Spectra accumulated before each streamed write batch.
    overwrite : bool, default False
        If the output already exists: replace it when True, else raise.
    progress : callable, optional
        Called with short status strings as processing proceeds.
    """
    # Imports are local so importing ptero.process is light and so this module
    # has no hard dependency on the raw extra until it is actually used.
    from ptero.spectra.backends.zarr_backend import MsBackendZarr, ZarrStreamWriter
    from ptero.spectra.validation import validate_store
    from ptero.spectra.backends.memory import MsBackendMemory

    transform = transform or identity_transform
    src_path = Path(str(source_path))
    out_path = Path(str(output_path))

    def _log(msg: str) -> None:
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass

    if not src_path.exists():
        raise FileNotFoundError(
            f"No Zarr store at {src_path}. Pass the path to a PTERO raw .zarr store."
        )
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_path} already exists. Pass overwrite=True to replace it."
            )
        shutil.rmtree(out_path, ignore_errors=True)

    src = MsBackendZarr(str(src_path))
    has_4d = bool(src.has_4d)
    peaks_vars = tuple(src.peaks_variables)
    acq_mode = str(src.acquisition_mode)
    n_total = int(src._n_total)

    # Faithful source metadata: read the exact parquet columns. ``scan_index`` and
    # ``data_storage`` are regenerated by the writer for the NEW store, so drop
    # them here (otherwise data_storage would still point at the source).
    meta = pd.read_parquet(src_path / "spectra.parquet")
    meta = meta.drop(columns=[c for c in ("scan_index", "data_storage") if c in meta.columns])
    if len(meta) != n_total:
        raise ValueError(
            f"metadata rows ({len(meta)}) != n_spectra ({n_total}) in {src_path}; "
            "the source store looks inconsistent."
        )

    _log(f"processing {n_total} spectra ({'4D' if has_4d else '3D'}, mode={acq_mode}) -> {out_path}")

    # Preserve declarations from the source; never infer mobility units.
    import zarr
    source_attrs = dict(zarr.open_group(str(src_path), mode="r").attrs)
    mobility_keys = (
        "peak_ion_mobility_type",
        "peak_ion_mobility_unit",
        "spectrum_ion_mobility_type",
        "spectrum_ion_mobility_unit",
    )
    mobility_kwargs = {
        key: source_attrs[key]
        for key in mobility_keys
        if key in source_attrs
    }
    writer = ZarrStreamWriter(
        str(out_path),
        has_4d=has_4d,
        acquisition_mode=acq_mode,
        **mobility_kwargs,
    )
    buf_peaks: list[dict] = []
    buf_idx: list[int] = []
    expected = 0
    written = 0

    def _flush() -> None:
        nonlocal written
        if not buf_peaks:
            return
        batch_meta = meta.iloc[buf_idx].reset_index(drop=True)
        mem = MsBackendMemory.from_spectra_list(batch_meta, buf_peaks)
        writer.append_backend(mem)
        written += len(buf_peaks)
        _log(f"  wrote {written}/{n_total} spectra")

    for pos, arr in src.iter_peaks_blocked(None, peaks_vars, block_size):
        # The round-trip aligns processed[i] <-> source[i], which requires the
        # source spectra to arrive in storage order. iter_peaks_blocked yields a
        # full-store read in storage order; assert it, so any future change to the
        # read path fails loudly here instead of silently misaligning the stores.
        if pos != expected:
            raise RuntimeError(
                f"iter_peaks_blocked yielded spectra out of storage order (got "
                f"{pos}, expected {expected}); processed-store alignment would "
                "break. This is a read-path bug, not a problem with your data."
            )
        expected += 1

        peaks: PeaksDict = {
            "mz": np.asarray(arr[:, 0], dtype=np.float64),
            "intensity": np.asarray(arr[:, 1], dtype=np.float64),
        }
        if has_4d:
            peaks["ion_mobility"] = np.asarray(arr[:, 2], dtype=np.float64)

        out = transform(peaks)

        # Enforce the transform contract (and the non-negotiable 4D constraint).
        n_mz = len(out["mz"])
        if len(out["intensity"]) != n_mz:
            raise ValueError(
                f"transform returned mismatched mz/intensity lengths "
                f"({n_mz} vs {len(out['intensity'])}) for spectrum {pos}."
            )
        if has_4d:
            if out.get("ion_mobility") is None:
                raise ValueError(
                    f"source is 4D but the transform dropped 'ion_mobility' for "
                    f"spectrum {pos}. The processed store must stay 4D -- preserve "
                    "the per-peak mobility (carry it through, or re-attach it)."
                )
            if len(out["ion_mobility"]) != n_mz:
                raise ValueError(
                    f"transform returned ion_mobility of length "
                    f"{len(out['ion_mobility'])} but {n_mz} m/z values for spectrum {pos}."
                )

        buf_peaks.append(out)
        buf_idx.append(pos)
        if len(buf_peaks) >= spectra_per_batch:
            _flush()
            buf_peaks.clear()
            buf_idx.clear()

    _flush()

    if expected != n_total:
        raise RuntimeError(
            f"read {expected} spectra but the source store reports {n_total}; aborting."
        )

    writer.finalize()
    # Validate the physical store before marking it as a completed processed artifact.
    validate_store(out_path)
    declared = transform_provenance(transform)
    effective_steps = steps if steps is not None else ([declared] if declared else _default_steps())
    _stamp_provenance(out_path, src_path, n_total, effective_steps)
    _log("done")
    return str(out_path)


def _stamp_provenance(out_path: Path, src_path: Path, n_total: int, steps: list) -> None:
    """Record the processed marker + provenance as root attributes on the store.

    Added AFTER the writer finalizes. :class:`MsBackendZarr` ignores unknown
    attrs, so a processed store still opens as an ordinary lazy store; these
    attrs are read only by :func:`is_processed_store` / :func:`read_processing_provenance`.
    """
    import zarr
    root = zarr.open_group(str(out_path), mode="r+")
    root.attrs[PROCESSED_FLAG_ATTR] = True
    root.attrs[PROCESSING_PROVENANCE_ATTR] = {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "source_store": str(src_path),
        "source_n_spectra": int(n_total),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "ptero_version": _ptero_version(),
        "steps": list(steps),
    }