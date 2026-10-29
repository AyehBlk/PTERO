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







    ATTR_MZZARR_FORMAT, ATTR_MZZARR_FORMAT_VERSION,







    MZZARR_FORMAT, MZZARR_FORMAT_VERSION,







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





    # MZZARR_IDENTITY_VALIDATION_V1



    # Complete identity pairs are required when either key is present.



    if (fmt is None) != (version is None):



        errors.append("incomplete PTERO format identity")







    if version is not None and version != "1.0":



        errors.append(f"unsupported PTERO version: {version!r}")





















    mzzarr_fmt = attrs.get(ATTR_MZZARR_FORMAT)







    mzzarr_version = attrs.get(ATTR_MZZARR_FORMAT_VERSION)















    if (mzzarr_fmt is None) != (mzzarr_version is None):







        errors.append("incomplete mzZarr format identity")















    if mzzarr_fmt is not None and mzzarr_fmt != MZZARR_FORMAT:







        errors.append(f"unsupported mzZarr format: {mzzarr_fmt!r}")















    if mzzarr_version is not None and mzzarr_version != MZZARR_FORMAT_VERSION:







        errors.append(f"unsupported mzZarr version: {mzzarr_version!r}")















    if mzzarr_fmt is not None:







        legacy = False















    if PEAKS_GROUP not in root:







        errors.append("missing 'peaks' group")







        peaks = None







    else:







        peaks = root[PEAKS_GROUP]













    # MZZARR_ARRAY_VALIDATION_V1





    expected_arrays = [





        (root, OFFSETS_ARRAY, np.dtype("int64"), "offsets"),





    ]











    if peaks is not None:





        expected_arrays.extend([





            (peaks, PEAK_MZ, np.dtype("float64"), "peaks/mz"),





            (peaks, PEAK_INTENSITY, np.dtype("float32"),





             "peaks/intensity"),





            (peaks, PEAK_ION_MOBILITY, np.dtype("float32"),





             "peaks/ion_mobility"),





        ])











    for group, name, dtype, label in expected_arrays:





        if name not in group:





            continue





        arr = group[name]





        if arr.ndim != 1:





            errors.append(f"{label} must be 1D, got {arr.ndim}D")





        if np.dtype(arr.dtype) != dtype:





            errors.append(





                f"{label} dtype={arr.dtype}, expected {dtype}"





            )



















    if OFFSETS_ARRAY not in root:







        errors.append("missing 'offsets' array")







        offsets = np.array([], dtype=np.int64)







    else:







        offsets = np.asarray(root[OFFSETS_ARRAY][:], dtype=np.int64).reshape(-1)













    # MZZARR_METADATA_CONFORMANCE_V2

    _default_n_spectra = max(0, len(offsets) - 1)

    _declared_n_spectra = attrs.get(ATTR_N_SPECTRA)



    if _declared_n_spectra is None:

        n_spectra = _default_n_spectra

    elif (

        isinstance(_declared_n_spectra, bool)

        or not isinstance(_declared_n_spectra, int)

        or _declared_n_spectra < 0

    ):

        errors.append("invalid n_spectra attribute")

        n_spectra = _default_n_spectra

    else:

        n_spectra = _declared_n_spectra



    # MZZARR_MOBILITY_VALIDATOR_V1
    from ._schema import validate_mobility_semantics
    import pyarrow.parquet as _mzzarr_mobility_pq
    _spectra_file = Path(path) / "spectra.parquet"
    _spectrum_mobility_present = False
    if _spectra_file.is_file():
        try:
            if (
                "ion_mobility"
                in _mzzarr_mobility_pq.read_schema(
                    _spectra_file
                ).names
            ):
                _mobility_column = _mzzarr_mobility_pq.read_table(
                    _spectra_file, columns=["ion_mobility"]
                ).column("ion_mobility")
                _spectrum_mobility_present = (
                    _mobility_column.null_count < len(_mobility_column)
                )
        except (OSError, ValueError, RuntimeError):
            # The existing Parquet validation reports unreadable metadata.
            pass
    errors.extend(validate_mobility_semantics(
        attrs,
        has_peak_mobility=(
            "peaks" in root
            and "ion_mobility" in root["peaks"]
        ),
        has_spectrum_mobility=_spectrum_mobility_present,
        strict=(
            attrs.get("mzzarr_format") == "mzZarr"
            and attrs.get("mzzarr_format_version") == "1.0"
        ),
    ))

    # MZZARR_RTIME_SEMANTICS_V1
    _rtime_unit = attrs.get("rtime_unit")
    if _rtime_unit is not None and _rtime_unit != "second":
        errors.append("invalid rtime_unit attribute: expected second")

    _declared_mode = attrs.get("acquisition_mode")

    if _declared_mode is not None:

        if (

            not isinstance(_declared_mode, str)

            or _declared_mode not in {"dda", "dia", "unknown"}

        ):

            errors.append("invalid acquisition_mode attribute")



    _declared_variables = attrs.get("peaks_variables")

    if _declared_variables is not None:

        if (

            not isinstance(_declared_variables, list)

            or not all(isinstance(x, str) for x in _declared_variables)

            or len(_declared_variables) != len(set(_declared_variables))

            or "mz" not in _declared_variables

            or "intensity" not in _declared_variables

        ):

            errors.append("invalid peaks_variables attribute")



    _declared_4d = attrs.get(ATTR_HAS_4D)

    if (

        _declared_4d is not None

        and not isinstance(_declared_4d, bool)

    ):

        errors.append("invalid has_4d attribute")







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







                errors.append(f"len(peaks/{name})={int(peaks[name].shape[0])}, expected {n_peaks}")







        has_im_array = PEAK_ION_MOBILITY in peaks







        if has_4d and not has_im_array:







            errors.append("has_4d=True but peaks/ion_mobility is missing")







        if not has_4d and has_im_array:







            errors.append("has_4d=False but peaks/ion_mobility exists")







        if has_im_array and int(peaks[PEAK_ION_MOBILITY].shape[0]) != n_peaks:







            errors.append(







                f"len(peaks/ion_mobility)={int(peaks[PEAK_ION_MOBILITY].shape[0])}, expected {n_peaks}"







            )















    meta_path = p / SPECTRA_METADATA_FILE







    if not meta_path.exists():







        errors.append(f"missing {SPECTRA_METADATA_FILE}")







    else:







        try:







            # Reading one narrow column avoids loading the full metadata table.







            meta = pd.read_parquet(meta_path, columns=["scan_index"])

            # MZZARR_MS_LEVEL_CONFORMANCE_V1
            if mzzarr_fmt == MZZARR_FORMAT and mzzarr_version == MZZARR_FORMAT_VERSION:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    schema = pq.read_schema(meta_path)
                    if "ms_level" not in schema.names:
                        errors.append("mzZarr v1.0 requires ms_level column")
                    elif not pa.types.is_int64(schema.field("ms_level").type):
                        errors.append("mzZarr v1.0 ms_level must have Parquet int64 type")
                    else:
                        levels = pd.read_parquet(
                            meta_path, columns=["ms_level"]
                        )["ms_level"]
                        if levels.isna().any():
                            errors.append("mzZarr v1.0 ms_level contains null values")
                        elif (levels <= 0).any():
                            errors.append(
                                "mzZarr v1.0 ms_level must contain positive integers"
                            )
                except Exception as exc:
                    errors.append(f"cannot validate mzZarr v1.0 ms_level: {exc}")







            if len(meta) != n_spectra:







                errors.append(f"metadata rows={len(meta)}, expected {n_spectra}")







        except Exception as e:







            errors.append(f"cannot read {SPECTRA_METADATA_FILE}: {e}")















    if errors and raise_on_error:







        joined = "\n - ".join(errors)







        raise StoreValidationError(f"Invalid PTERO store {p}:\n - {joined}")















    return StoreValidationReport(







        path=str(p), n_spectra=n_spectra, n_peaks=n_peaks, has_4d=has_4d,







        format_version=str(mzzarr_version if mzzarr_version is not None else version) if (mzzarr_version is not None or version is not None) else None,







        legacy_unversioned=legacy,







    )
