"""Validation of the physical PTERO Zarr store contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ._schema import (
    ATTR_FORMAT, ATTR_FORMAT_VERSION, ATTR_HAS_4D, ATTR_N_SPECTRA,
    OFFSETS_ARRAY, PEAKS_GROUP, PEAK_INTENSITY, PEAK_ION_MOBILITY, PEAK_MZ,
    PTERO_FORMAT, SPECTRA_METADATA_FILE,
)


class StoreValidationError(ValueError):
    """Raised when a PTERO store violates one or more structural invariants."""


@dataclass(frozen=True)
class StoreValidationReport:
    path: str
    n_spectra: int
    n_peaks: int
    has_4d: bool
    format_version: str | None
    legacy_unversioned: bool


def validate_store(path, *, raise_on_error: bool = True) -> StoreValidationReport:
    """Validate a PTERO Zarr store without materializing its peak arrays.

    Checks the CSR/ragged layout, metadata row count, 3D/4D consistency, and
    version marker.  Unversioned legacy PTERO stores are accepted if their
    structure is valid; the report marks them as ``legacy_unversioned=True``.
    """
    try:
        import zarr
    except Exception as e:  # pragma: no cover
        raise ImportError("validate_store requires zarr; install ptero[zarr].") from e

    p = Path(path)
    errors: list[str] = []
    if not p.exists():
        raise FileNotFoundError(f"No PTERO store at {p}")

    try:
        root = zarr.open_group(str(p), mode="r")
    except Exception as e:
        raise StoreValidationError(f"Cannot open {p} as a Zarr group: {e}") from e

    attrs = dict(root.attrs)
    fmt = attrs.get(ATTR_FORMAT)
    version = attrs.get(ATTR_FORMAT_VERSION)
    legacy = fmt is None and version is None
    if fmt is not None and fmt != PTERO_FORMAT:
        errors.append(f"{ATTR_FORMAT}={fmt!r}, expected {PTERO_FORMAT!r}")

    if PEAKS_GROUP not in root:
        errors.append("missing 'peaks' group")
        peaks = None
    else:
        peaks = root[PEAKS_GROUP]

    if OFFSETS_ARRAY not in root:
        errors.append("missing 'offsets' array")
        offsets = np.array([], dtype=np.int64)
    else:
        offsets = np.asarray(root[OFFSETS_ARRAY][:], dtype=np.int64)

    n_spectra = int(attrs.get(ATTR_N_SPECTRA, max(0, len(offsets) - 1)))
    has_4d = bool(attrs.get(ATTR_HAS_4D, False))

    if len(offsets) != n_spectra + 1:
        errors.append(f"len(offsets)={len(offsets)} but n_spectra+1={n_spectra + 1}")
    if len(offsets) and offsets[0] != 0:
        errors.append(f"offsets[0]={int(offsets[0])}, expected 0")
    if len(offsets) > 1 and np.any(np.diff(offsets) < 0):
        errors.append("offsets must be monotonically non-decreasing")
    n_peaks = int(offsets[-1]) if len(offsets) else 0

    if peaks is not None:
        for name in (PEAK_MZ, PEAK_INTENSITY):
            if name not in peaks:
                errors.append(f"missing peaks/{name}")
            elif int(peaks[name].shape[0]) != n_peaks:
                errors.append(f"len(peaks/{name})={len(peaks[name])}, expected {n_peaks}")
        has_im_array = PEAK_ION_MOBILITY in peaks
        if has_4d and not has_im_array:
            errors.append("has_4d=True but peaks/ion_mobility is missing")
        if not has_4d and has_im_array:
            errors.append("has_4d=False but peaks/ion_mobility exists")
        if has_im_array and int(peaks[PEAK_ION_MOBILITY].shape[0]) != n_peaks:
            errors.append(
                f"len(peaks/ion_mobility)={len(peaks[PEAK_ION_MOBILITY])}, expected {n_peaks}"
            )

    meta_path = p / SPECTRA_METADATA_FILE
    if not meta_path.exists():
        errors.append(f"missing {SPECTRA_METADATA_FILE}")
    else:
        try:
            # Reading one narrow column avoids loading the full metadata table.
            meta = pd.read_parquet(meta_path, columns=["scan_index"])
            if len(meta) != n_spectra:
                errors.append(f"metadata rows={len(meta)}, expected {n_spectra}")
        except Exception as e:
            errors.append(f"cannot read {SPECTRA_METADATA_FILE}: {e}")

    if errors and raise_on_error:
        joined = "\n - ".join(errors)
        raise StoreValidationError(f"Invalid PTERO store {p}:\n - {joined}")

    return StoreValidationReport(
        path=str(p), n_spectra=n_spectra, n_peaks=n_peaks, has_4d=has_4d,
        format_version=str(version) if version is not None else None,
        legacy_unversioned=legacy,
    )
