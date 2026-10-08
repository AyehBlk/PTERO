"""Authoritative constants and invariants for the PTERO Zarr store format.

Version 1.0 formalizes the layout already used by PTERO.  It intentionally does
not change the physical peak representation, so stores written before this
version remain readable; legacy stores simply lack the version attributes.
"""
from __future__ import annotations

PTERO_FORMAT = "ptero-ms"
PTERO_FORMAT_VERSION = "1.0"

ATTR_FORMAT = "ptero_format"
ATTR_FORMAT_VERSION = "ptero_format_version"
ATTR_N_SPECTRA = "n_spectra"
ATTR_HAS_4D = "has_4d"
ATTR_ACQUISITION_MODE = "acquisition_mode"
ATTR_PEAKS_VARIABLES = "peaks_variables"

PEAKS_GROUP = "peaks"
PEAK_MZ = "mz"
PEAK_INTENSITY = "intensity"
PEAK_ION_MOBILITY = "ion_mobility"
OFFSETS_ARRAY = "offsets"
SPECTRA_METADATA_FILE = "spectra.parquet"

REQUIRED_PEAK_ARRAYS = (PEAK_MZ, PEAK_INTENSITY)
PEAK_ARRAYS_4D = (PEAK_MZ, PEAK_INTENSITY, PEAK_ION_MOBILITY)
VALID_ACQUISITION_MODES = frozenset({"dda", "dia", "unknown"})


def store_attrs(*, n_spectra: int, has_4d: bool, acquisition_mode: str) -> dict:
    """Return canonical root attributes for a PTERO Zarr store."""
    mode = str(acquisition_mode).lower()
    if mode not in VALID_ACQUISITION_MODES:
        mode = "unknown"
    return {
        ATTR_FORMAT: PTERO_FORMAT,
        ATTR_FORMAT_VERSION: PTERO_FORMAT_VERSION,
        ATTR_N_SPECTRA: int(n_spectra),
        ATTR_HAS_4D: bool(has_4d),
        ATTR_ACQUISITION_MODE: mode,
        ATTR_PEAKS_VARIABLES: list(PEAK_ARRAYS_4D if has_4d else REQUIRED_PEAK_ARRAYS),
    }
