"""Authoritative constants and invariants for the PTERO Zarr store format.

Version 1.0 formalizes the layout already used by PTERO.  It intentionally does
not change the physical peak representation, so stores written before this
version remain readable; legacy stores simply lack the version attributes.
"""
from __future__ import annotations

PTERO_FORMAT = "ptero-ms"
PTERO_FORMAT_VERSION = "1.0"

MZZARR_FORMAT = "mzZarr"
MZZARR_FORMAT_VERSION = "1.0"
ATTR_MZZARR_FORMAT = "mzzarr_format"
ATTR_MZZARR_FORMAT_VERSION = "mzzarr_format_version"

ATTR_FORMAT = "ptero_format"
ATTR_FORMAT_VERSION = "ptero_format_version"
ATTR_N_SPECTRA = "n_spectra"
ATTR_HAS_4D = "has_4d"
ATTR_ACQUISITION_MODE = "acquisition_mode"
ATTR_PEAKS_VARIABLES = "peaks_variables"
ATTR_RTIME_UNIT = "rtime_unit"

PEAKS_GROUP = "peaks"
PEAK_MZ = "mz"
PEAK_INTENSITY = "intensity"
PEAK_ION_MOBILITY = "ion_mobility"
OFFSETS_ARRAY = "offsets"
SPECTRA_METADATA_FILE = "spectra.parquet"

REQUIRED_PEAK_ARRAYS = (PEAK_MZ, PEAK_INTENSITY)
PEAK_ARRAYS_4D = (PEAK_MZ, PEAK_INTENSITY, PEAK_ION_MOBILITY)
VALID_ACQUISITION_MODES = frozenset({"dda", "dia", "unknown"})

# MZZARR_MOBILITY_SEMANTICS_V1
MZZARR_MOBILITY_PAIRS = {
    "inverse_reduced_ion_mobility": "V.s/cm2",
    "ion_mobility_drift_time": "millisecond",
}

def validate_mobility_semantics(attrs, *, has_peak_mobility,
                                has_spectrum_mobility,
                                strict=False):
    """Return semantic errors without inferring measurement quantities."""
    errors = []
    scopes = (
        ("peak", has_peak_mobility),
        ("spectrum", has_spectrum_mobility),
    )
    for scope, present in scopes:
        type_key = f"{scope}_ion_mobility_type"
        unit_key = f"{scope}_ion_mobility_unit"
        has_type = type_key in attrs
        has_unit = unit_key in attrs

        if has_type != has_unit:
            errors.append(
                f"{scope} ion mobility requires both "
                f"{type_key} and {unit_key}"
            )
            continue

        if not has_type:
            if strict and present:
                errors.append(
                    f"{scope} ion mobility requires both "
                    f"{type_key} and {unit_key}"
                )
            continue

        quantity = attrs[type_key]
        unit = attrs[unit_key]
        if not present:
            errors.append(
                f"{scope} ion mobility declaration has no corresponding data"
            )
        if (
            not isinstance(quantity, str)
            or quantity not in MZZARR_MOBILITY_PAIRS
        ):
            errors.append(f"invalid {type_key}")
        elif (
            not isinstance(unit, str)
            or unit != MZZARR_MOBILITY_PAIRS[quantity]
        ):
            errors.append(
                f"invalid {unit_key} for {quantity}"
            )
    return errors

def mobility_semantic_attrs(*, has_peak_mobility,
                            has_spectrum_mobility,
                            peak_ion_mobility_type=None,
                            peak_ion_mobility_unit=None,
                            spectrum_ion_mobility_type=None,
                            spectrum_ion_mobility_unit=None,
                            strict=False):
    """Validate and return only explicitly supplied mobility declarations."""
    attrs = {}
    values = {
        "peak_ion_mobility_type": peak_ion_mobility_type,
        "peak_ion_mobility_unit": peak_ion_mobility_unit,
        "spectrum_ion_mobility_type": spectrum_ion_mobility_type,
        "spectrum_ion_mobility_unit": spectrum_ion_mobility_unit,
    }
    attrs.update({
        key: value for key, value in values.items()
        if value is not None
    })
    errors = validate_mobility_semantics(
        attrs,
        has_peak_mobility=has_peak_mobility,
        has_spectrum_mobility=has_spectrum_mobility,
        strict=strict,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return attrs




def store_attrs(*, n_spectra: int, has_4d: bool, acquisition_mode: str) -> dict:
    """Return canonical root attributes for a PTERO Zarr store."""
    mode = str(acquisition_mode).lower()
    if mode not in VALID_ACQUISITION_MODES:
        mode = "unknown"
    return {
        ATTR_FORMAT: PTERO_FORMAT,
        ATTR_FORMAT_VERSION: PTERO_FORMAT_VERSION,
        ATTR_MZZARR_FORMAT: MZZARR_FORMAT,
        ATTR_MZZARR_FORMAT_VERSION: MZZARR_FORMAT_VERSION,
        ATTR_N_SPECTRA: int(n_spectra),
        ATTR_HAS_4D: bool(has_4d),
        ATTR_ACQUISITION_MODE: mode,
        ATTR_RTIME_UNIT: "second",
        ATTR_PEAKS_VARIABLES: list(PEAK_ARRAYS_4D if has_4d else REQUIRED_PEAK_ARRAYS),
    }
