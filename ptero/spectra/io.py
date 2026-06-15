"""
Importers that build a :class:`PteroSpectra` from raw MS sources.

* :func:`from_mzml` -- read 3D mzML/mzXML. Lazy by default: peaks stay on disk
  and are read on demand via :class:`MsBackendMzML`. Pass ``lazy=False`` to pull
  everything into RAM (an in-memory backend) up front.
* :func:`from_dataframe` -- build directly from peak/metadata arrays; the generic
  entry point used by tests and by other readers (e.g. a future Bruker ``.d``
  reader via alphatims/alpharaw producing 4D peaks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ._spectra import PteroSpectra
from .backends.memory import MsBackendMemory
from .backends.mzml_backend import MsBackendMzML
from .backends.zarr_backend import MsBackendZarr


def from_dataframe(meta: pd.DataFrame, peaks: list[dict]) -> PteroSpectra:
    """Build a PteroSpectra from spectrum metadata + per-spectrum peak dicts.

    ``peaks[i]`` is ``{'mz': array, 'intensity': array[, 'ion_mobility': array]}``.
    """
    meta = meta.reset_index(drop=True).copy()
    if "scan_index" not in meta.columns:
        meta["scan_index"] = range(len(meta))
    be = MsBackendMemory.from_spectra_list(meta, peaks)
    return PteroSpectra(be)


def from_mzml(path, ms_level: Optional[int] = None, lazy: bool = True) -> PteroSpectra:
    """Read an mzML (or mzXML) file into a PteroSpectra (3D: RT x m/z).

    Parameters
    ----------
    path : str | Path
    ms_level : int, optional
        If given, keep only spectra of this MS level.
    lazy : bool, default True
        If True, peaks are read on demand from the file (MsBackendMzML) and the
        whole file never has to fit in memory. If False, materialize everything
        into an in-memory backend immediately.
    """
    sp = PteroSpectra(MsBackendMzML(str(path)))
    if ms_level is not None:
        sp = sp.filter_ms_level(ms_level)
    if not lazy:
        sp = sp.set_backend("memory")
    return sp


def open_zarr(path) -> PteroSpectra:
    """Open a PTERO Zarr store as a lazy :class:`PteroSpectra`.

    This is the fast, reusable entry point for a run you have already saved:
    the store opens instantly and reads peaks on demand, so a large run never
    has to fit in RAM. It is the one-liner equivalent of
    ``PteroSpectra(MsBackendZarr(path))``.

    A Zarr store is produced either by ``spectra.set_backend('zarr', path=...)``
    or by :func:`ptero.dashboard.convert_to_zarr` (which converts a raw ``.d`` or
    ``mzML`` run in memory-safe batches).

    Parameters
    ----------
    path : str | Path
        Path to a ``.zarr`` store directory.

    Examples
    --------
    >>> import ptero.spectra as sp
    >>> run = sp.open_zarr("C:/data/hela_ms1.zarr")
    >>> run.qc()
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No Zarr store at {p}. Create one with "
            f"spectra.set_backend('zarr', path=...) or "
            f"ptero.dashboard.convert_to_zarr(raw_file, path)."
        )
    return PteroSpectra(MsBackendZarr(str(path)))
