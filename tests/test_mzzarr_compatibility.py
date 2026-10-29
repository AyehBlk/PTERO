import numpy as np







import pandas as pd







import pytest















from ptero.spectra.backends.zarr_backend import MsBackendZarr







from ptero.spectra.validation import StoreValidationError, validate_store























@pytest.fixture







def legacy_store(tmp_path):







    path = tmp_path / "legacy.zarr"















    meta = pd.DataFrame({"scan_index": [0, 1], "ms_level": [1, 2]})







    mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)







    intensity = np.array([10.0, 20.0, 30.0], dtype=np.float32)







    offsets = np.array([0, 2, 3], dtype=np.int64)















    MsBackendZarr.write(path, meta, mz, intensity, offsets)







    return path























def test_current_ptero_store_validates(legacy_store):







    report = validate_store(legacy_store)















    assert report.n_spectra == 2







    assert report.n_peaks == 3







    assert report.format_version == "1.0"







    assert report.legacy_unversioned is False























def test_unversioned_legacy_store_validates(legacy_store):







    import zarr















    root = zarr.open_group(str(legacy_store), mode="a")







    del root.attrs["ptero_format"]







    del root.attrs["ptero_format_version"]







    del root.attrs["mzzarr_format"]







    del root.attrs["mzzarr_format_version"]















    report = validate_store(legacy_store)















    assert report.legacy_unversioned is True







    assert report.format_version is None























def test_unknown_ptero_format_rejected(legacy_store):







    import zarr















    root = zarr.open_group(str(legacy_store), mode="a")







    root.attrs["ptero_format"] = "unsupported-format"















    with pytest.raises(StoreValidationError):







        validate_store(legacy_store)























def test_existing_reader_preserves_peaks(legacy_store):







    reader = MsBackendZarr(str(legacy_store))







    peaks = reader.peaks_data(indices=[0, 1])















    np.testing.assert_allclose(







        peaks[0],







        [[100.0, 10.0], [200.0, 20.0]],







    )







    np.testing.assert_allclose(







        peaks[1],







        [[300.0, 30.0]],







    )















def test_new_store_has_mzzarr_identity(tmp_path):







    import zarr















    path = tmp_path / "new_mzzarr.zarr"















    meta = pd.DataFrame({"scan_index": [0], "ms_level": [1]})







    mz = np.array([100.0, 200.0], dtype=np.float64)







    intensity = np.array([10.0, 20.0], dtype=np.float32)







    offsets = np.array([0, 2], dtype=np.int64)















    MsBackendZarr.write(path, meta, mz, intensity, offsets)















    attrs = dict(zarr.open_group(str(path), mode="r").attrs)















    assert attrs["mzzarr_format"] == "mzZarr"







    assert attrs["mzzarr_format_version"] == "1.0"















    # Preserve legacy identifiers during the transition.







    assert attrs["ptero_format"] == "ptero-ms"







    assert attrs["ptero_format_version"] == "1.0"















    report = validate_store(path)







    assert report.n_spectra == 1















def test_incomplete_mzzarr_identity_rejected(legacy_store):







    import zarr















    root = zarr.open_group(str(legacy_store), mode="a")







    del root.attrs["mzzarr_format_version"]















    with pytest.raises(StoreValidationError):







        validate_store(legacy_store)























def test_unknown_mzzarr_version_rejected(legacy_store):







    import zarr















    root = zarr.open_group(str(legacy_store), mode="a")







    root.attrs["mzzarr_format_version"] = "999.0"















    with pytest.raises(StoreValidationError):







        validate_store(legacy_store)























def test_streaming_writer_mzzarr_roundtrip(tmp_path):







    from types import SimpleNamespace







    import zarr















    from ptero.spectra.backends.zarr_backend import ZarrStreamWriter















    path = tmp_path / "streamed_mzzarr.zarr"















    batch1 = SimpleNamespace(







        _meta=pd.DataFrame({"scan_index": [0, 1], "ms_level": [1, 2]}),







        _mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),







        _intensity=np.array([10.0, 20.0, 30.0], dtype=np.float32),







        _offsets=np.array([0, 2, 3], dtype=np.int64),







        _im=None,







    )















    batch2 = SimpleNamespace(







        _meta=pd.DataFrame({"scan_index": [0, 1], "ms_level": [1, 2]}),







        _mz=np.array([400.0, 500.0, 600.0], dtype=np.float64),







        _intensity=np.array([40.0, 50.0, 60.0], dtype=np.float32),







        _offsets=np.array([0, 1, 3], dtype=np.int64),







        _im=None,







    )















    writer = ZarrStreamWriter(







        path,







        has_4d=False,







        chunk_size=2,







        acquisition_mode="dda",







    )







    writer.append_backend(batch1)







    writer.append_backend(batch2)







    reader = writer.finalize()















    root = zarr.open_group(str(path), mode="r")







    attrs = dict(root.attrs)















    assert attrs["mzzarr_format"] == "mzZarr"







    assert attrs["mzzarr_format_version"] == "1.0"







    assert attrs["ptero_format"] == "ptero-ms"







    assert attrs["ptero_format_version"] == "1.0"







    assert attrs["n_spectra"] == 4







    assert attrs["acquisition_mode"] == "dda"















    np.testing.assert_array_equal(







        np.asarray(root["offsets"][:]),







        np.array([0, 2, 3, 4, 6], dtype=np.int64),







    )















    np.testing.assert_array_equal(







        np.asarray(root["peaks/mz"][:]),







        [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],







    )















    np.testing.assert_array_equal(







        np.asarray(root["peaks/intensity"][:]),







        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],







    )















    metadata = pd.read_parquet(path / "spectra.parquet")







    assert metadata["scan_index"].tolist() == [0, 1, 2, 3]















    peaks = reader.peaks_data(indices=[0, 1, 2, 3])







    expected = [







        [[100.0, 10.0], [200.0, 20.0]],







        [[300.0, 30.0]],







        [[400.0, 40.0]],







        [[500.0, 50.0], [600.0, 60.0]],







    ]















    for actual, wanted in zip(peaks, expected):







        np.testing.assert_array_equal(







            np.asarray(actual),







            np.asarray(wanted),







        )















    report = validate_store(path)







    assert report.n_spectra == 4







    assert report.n_peaks == 6







    assert report.legacy_unversioned is False























def test_streaming_writer_4d_ion_mobility(tmp_path):







    from types import SimpleNamespace







    import zarr















    from ptero.spectra.backends.zarr_backend import ZarrStreamWriter















    path = tmp_path / "streamed_4d_mzzarr.zarr"















    batch1 = SimpleNamespace(







        _meta=pd.DataFrame({"scan_index": [0], "ms_level": [1]}),







        _mz=np.array([100.0, 200.0], dtype=np.float64),







        _intensity=np.array([10.0, 20.0], dtype=np.float32),







        _im=np.array([1.1, 1.2], dtype=np.float32),







        _offsets=np.array([0, 2], dtype=np.int64),







    )















    batch2 = SimpleNamespace(







        _meta=pd.DataFrame({"scan_index": [0], "ms_level": [1]}),







        _mz=np.array([300.0, 400.0], dtype=np.float64),







        _intensity=np.array([30.0, 40.0], dtype=np.float32),







        _im=np.array([1.3, 1.4], dtype=np.float32),







        _offsets=np.array([0, 2], dtype=np.int64),







    )















    writer = ZarrStreamWriter(







        path,







        has_4d=True,







        chunk_size=2,







        acquisition_mode="dda",
        peak_ion_mobility_type="inverse_reduced_ion_mobility",
        peak_ion_mobility_unit="V.s/cm2",







    )















    writer.append_backend(batch1)







    writer.append_backend(batch2)







    writer.finalize()















    root = zarr.open_group(str(path), mode="r")







    attrs = dict(root.attrs)















    assert attrs["mzzarr_format"] == "mzZarr"







    assert attrs["mzzarr_format_version"] == "1.0"







    assert attrs["has_4d"] is True







    assert attrs["n_spectra"] == 2















    np.testing.assert_array_equal(







        np.asarray(root["offsets"][:]),







        [0, 2, 4],







    )







    np.testing.assert_array_equal(







        np.asarray(root["peaks/mz"][:]),







        [100.0, 200.0, 300.0, 400.0],







    )







    np.testing.assert_array_equal(







        np.asarray(root["peaks/intensity"][:]),







        [10.0, 20.0, 30.0, 40.0],







    )







    np.testing.assert_array_equal(







        np.asarray(root["peaks/ion_mobility"][:]),







        np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32),







    )















    metadata = pd.read_parquet(path / "spectra.parquet")







    assert metadata["scan_index"].tolist() == [0, 1]















    report = validate_store(path)







    assert report.n_spectra == 2







    assert report.n_peaks == 4







    assert report.has_4d is True























def test_legacy_ptero_only_store_validates(legacy_store):







    """A pre-mzZarr PTERO store remains readable and valid."""







    import zarr















    root = zarr.open_group(str(legacy_store), mode="a")







    del root.attrs["mzzarr_format"]







    del root.attrs["mzzarr_format_version"]















    report = validate_store(legacy_store)















    assert report.n_spectra == 2







    assert report.n_peaks == 3







    assert report.format_version == "1.0"







    assert report.legacy_unversioned is False















    reader = MsBackendZarr(str(legacy_store))







    peaks = reader.peaks_data(indices=[0, 1])















    np.testing.assert_array_equal(







        np.asarray(peaks[0]),







        [[100.0, 10.0], [200.0, 20.0]],







    )







    np.testing.assert_array_equal(







        np.asarray(peaks[1]),







        [[300.0, 30.0]],







    )























def test_unknown_mzzarr_format_rejected(legacy_store):







    """Reject stores claiming an unsupported mzZarr identity."""







    import zarr















    root = zarr.open_group(str(legacy_store), mode="a")







    root.attrs["mzzarr_format"] = "unsupported-format"















    with pytest.raises(StoreValidationError):







        validate_store(legacy_store)



























def test_partial_legacy_identity_rejected(legacy_store):



    import zarr



    import pytest



    from ptero.spectra.validation import (



        validate_store, StoreValidationError



    )







    root = zarr.open_group(str(legacy_store), mode="a")



    del root.attrs["ptero_format_version"]







    with pytest.raises(StoreValidationError):



        validate_store(legacy_store)


























def test_mzzarr_only_store_validates_and_reads(legacy_store):







    """A valid mzZarr dataset must not require PTERO identifiers."""







    import zarr















    root = zarr.open_group(str(legacy_store), mode="a")







    del root.attrs["ptero_format"]







    del root.attrs["ptero_format_version"]















    assert root.attrs["mzzarr_format"] == "mzZarr"







    assert root.attrs["mzzarr_format_version"] == "1.0"















    report = validate_store(legacy_store)















    assert report.n_spectra == 2







    assert report.n_peaks == 3







    assert report.legacy_unversioned is False















    reader = MsBackendZarr(str(legacy_store))







    peaks = reader.peaks_data(indices=[0, 1])















    np.testing.assert_array_equal(







        np.asarray(peaks[0]),







        [[100.0, 10.0], [200.0, 20.0]],







    )







    np.testing.assert_array_equal(







        np.asarray(peaks[1]),







        [[300.0, 30.0]],







    )























def test_mzzarr_only_unsupported_version_rejected(legacy_store):







    """mzZarr-only stores must still obey version validation."""







    import zarr















    root = zarr.open_group(str(legacy_store), mode="a")







    del root.attrs["ptero_format"]







    del root.attrs["ptero_format_version"]







    root.attrs["mzzarr_format_version"] = "999.0"















    with pytest.raises(StoreValidationError):







        validate_store(legacy_store)























def test_independent_mzzarr_reader_without_ptero(legacy_store):







    """Decode mzZarr using only standard Zarr, NumPy and Parquet APIs."""







    import zarr















    root = zarr.open_group(str(legacy_store), mode="r")







    attrs = dict(root.attrs)















    assert attrs["mzzarr_format"] == "mzZarr"







    assert attrs["mzzarr_format_version"] == "1.0"















    offsets = np.asarray(root["offsets"][:], dtype=np.int64)







    mz = np.asarray(root["peaks/mz"][:])







    intensity = np.asarray(root["peaks/intensity"][:])







    metadata = pd.read_parquet(legacy_store / "spectra.parquet")















    assert len(metadata) == attrs["n_spectra"]







    assert len(offsets) == len(metadata) + 1







    assert offsets[0] == 0







    assert offsets[-1] == len(mz) == len(intensity)















    spectra = []







    for i in range(len(metadata)):







        start = int(offsets[i])







        end = int(offsets[i + 1])







        spectra.append(np.column_stack((







            mz[start:end],







            intensity[start:end],







        )))















    np.testing.assert_array_equal(







        spectra[0],







        [[100.0, 10.0], [200.0, 20.0]],







    )







    np.testing.assert_array_equal(







        spectra[1],







        [[300.0, 30.0]],







    )























def test_independent_mzzarr_writer_ptero_reader(tmp_path):







    """Create mzZarr without PTERO's writer; read it with PTERO."""







    import zarr















    path = tmp_path / "independent_mzzarr.zarr"















    mz = np.array(







        [101.5, 202.5, 303.5, 404.5],







        dtype=np.float64,







    )







    intensity = np.array(







        [11.0, 22.0, 33.0, 44.0],







        dtype=np.float32,







    )







    offsets = np.array([0, 2, 2, 4], dtype=np.int64)















    metadata = pd.DataFrame({







        "scan_index": [0, 1, 2], "ms_level": [1, 2, 3],







    })















    # Write the physical format using Zarr directly.







    root = zarr.open_group(str(path), mode="w", zarr_format=3)















    root.create_array(







        "peaks/mz",







        data=mz,







    )







    root.create_array(







        "peaks/intensity",







        data=intensity,







    )







    root.create_array(







        "offsets",







        data=offsets,







    )















    # Write spectrum metadata using Parquet directly.







    metadata.to_parquet(path / "spectra.parquet", index=False)















    # The dataset identifies itself exclusively as mzZarr.







    root.attrs.update({







        "mzzarr_format": "mzZarr",







        "mzzarr_format_version": "1.0",







        "n_spectra": 3,







        "n_peaks": 4,







    })















    assert "ptero_format" not in root.attrs







    assert "ptero_format_version" not in root.attrs















    # Verify the independent dataset with PTERO.







    report = validate_store(path)















    assert report.n_spectra == 3







    assert report.n_peaks == 4







    assert report.legacy_unversioned is False















    reader = MsBackendZarr(str(path))







    peaks = reader.peaks_data(indices=[0, 1, 2])















    np.testing.assert_array_equal(







        np.asarray(peaks[0]),







        [[101.5, 11.0], [202.5, 22.0]],







    )







    assert np.asarray(peaks[1]).size == 0







    np.testing.assert_array_equal(







        np.asarray(peaks[2]),







        [[303.5, 33.0], [404.5, 44.0]],







    )





















# MZZARR_ARRAY_CONFORMANCE_TESTS_V1






# Expected failures document rules not yet enforced by validate_store.













import pytest as _mzzarr_pytest






import numpy as _mzzarr_np






import zarr as _mzzarr_zarr













from ptero.spectra.validation import (






    validate_store as _mzzarr_validate_store,






    StoreValidationError as _mzzarr_validation_error,






)




















def _mzzarr_replace_array(group, name, data):






    del group[name]






    group.create_array(






        name,






        data=_mzzarr_np.asarray(data),






    )




















@_mzzarr_pytest.mark.parametrize(






    "array_path,dtype",






    [






        ("peaks/mz", "f4"),






        ("peaks/intensity", "f8"),






        ("offsets", "i4"),






    ],






)






def test_mzzarr_rejects_wrong_array_dtype(






    legacy_store, array_path, dtype






):






    root = _mzzarr_zarr.open_group(str(legacy_store), mode="a")






    parts = array_path.split("/")






    group = root if len(parts) == 1 else root[parts[0]]






    name = parts[-1]













    data = _mzzarr_np.asarray(






        group[name][:], dtype=dtype






    )






    _mzzarr_replace_array(group, name, data)













    with _mzzarr_pytest.raises(_mzzarr_validation_error):






        _mzzarr_validate_store(legacy_store)




















@_mzzarr_pytest.mark.parametrize(






    "array_path",






    ["peaks/mz", "peaks/intensity", "offsets"],






)






def test_mzzarr_rejects_multidimensional_arrays(






    legacy_store, array_path






):






    root = _mzzarr_zarr.open_group(str(legacy_store), mode="a")






    parts = array_path.split("/")






    group = root if len(parts) == 1 else root[parts[0]]






    name = parts[-1]













    data = _mzzarr_np.asarray(group[name][:])






    _mzzarr_replace_array(group, name, data.reshape(-1, 1))













    with _mzzarr_pytest.raises(_mzzarr_validation_error):






        _mzzarr_validate_store(legacy_store)




















def test_mzzarr_rejects_wrong_ion_mobility_dtype(legacy_store):






    root = _mzzarr_zarr.open_group(str(legacy_store), mode="a")






    peaks = root["peaks"]













    mz = _mzzarr_np.asarray(peaks["mz"][:])






    peaks.create_array(






        "ion_mobility",






        data=_mzzarr_np.ones(len(mz), dtype="f8"),






    )






    root.attrs["has_4d"] = True













    with _mzzarr_pytest.raises(_mzzarr_validation_error):






        _mzzarr_validate_store(legacy_store)
















# MZZARR_IDENTITY_CONFORMANCE_TESTS_V1




# Proposed identity rules; implementation audit required before enforcement.









import pytest as _identity_pytest




import zarr as _identity_zarr









from ptero.spectra.validation import (




    validate_store as _identity_validate,




    StoreValidationError as _identity_error,




)














def test_mzzarr_rejects_unsupported_legacy_version(legacy_store):




    root = _identity_zarr.open_group(str(legacy_store), mode="a")




    root.attrs["ptero_format_version"] = "99.0"









    with _identity_pytest.raises(_identity_error):




        _identity_validate(legacy_store)














def test_mzzarr_rejects_legacy_version_without_format(legacy_store):




    root = _identity_zarr.open_group(str(legacy_store), mode="a")




    del root.attrs["ptero_format"]









    with _identity_pytest.raises(_identity_error):




        _identity_validate(legacy_store)














def test_mzzarr_rejects_legacy_format_without_version(legacy_store):




    root = _identity_zarr.open_group(str(legacy_store), mode="a")




    del root.attrs["ptero_format_version"]









    with _identity_pytest.raises(_identity_error):




        _identity_validate(legacy_store)














def test_mzzarr_only_reports_mzzarr_version(legacy_store):




    root = _identity_zarr.open_group(str(legacy_store), mode="a")









    del root.attrs["ptero_format"]




    del root.attrs["ptero_format_version"]









    report = _identity_validate(legacy_store)









    assert report.format_version == "1.0"




    assert report.legacy_unversioned is False








# MZZARR_STRUCTURAL_CONFORMANCE_TESTS_V2





from pathlib import Path as _struct_Path





import numpy as _struct_np


import pandas as _struct_pd


import pytest as _struct_pytest


import zarr as _struct_zarr





from ptero.spectra.validation import (


    validate_store as _struct_validate,


    StoreValidationError as _struct_error,


)








def _struct_replace(group, name, data):


    del group[name]


    group.create_array(name, data=_struct_np.asarray(data))








def test_mzzarr_rejects_nonzero_first_offset(legacy_store):


    root = _struct_zarr.open_group(str(legacy_store), mode="a")


    offsets = _struct_np.asarray(root["offsets"][:]).copy()


    offsets[0] = 1


    root["offsets"][:] = offsets





    with _struct_pytest.raises(_struct_error):


        _struct_validate(legacy_store)








def test_mzzarr_rejects_decreasing_offsets(legacy_store):


    root = _struct_zarr.open_group(str(legacy_store), mode="a")


    offsets = _struct_np.asarray(root["offsets"][:]).copy()





    assert offsets.size >= 3, "Fixture needs at least two spectra"





    offsets[1] = offsets[2] + 1


    root["offsets"][:] = offsets





    with _struct_pytest.raises(_struct_error):


        _struct_validate(legacy_store)








def test_mzzarr_rejects_final_offset_mismatch(legacy_store):


    root = _struct_zarr.open_group(str(legacy_store), mode="a")


    offsets = _struct_np.asarray(root["offsets"][:]).copy()


    offsets[-1] += 1


    root["offsets"][:] = offsets





    with _struct_pytest.raises(_struct_error):


        _struct_validate(legacy_store)








def test_mzzarr_rejects_short_intensity_array(legacy_store):


    root = _struct_zarr.open_group(str(legacy_store), mode="a")


    peaks = root["peaks"]





    intensity = _struct_np.asarray(peaks["intensity"][:])


    assert intensity.size > 0, "Fixture needs at least one peak"





    _struct_replace(peaks, "intensity", intensity[:-1])





    with _struct_pytest.raises(_struct_error):


        _struct_validate(legacy_store)








def test_mzzarr_rejects_wrong_metadata_row_count(legacy_store):


    path = _struct_Path(legacy_store) / "spectra.parquet"


    meta = _struct_pd.read_parquet(path)





    assert len(meta) > 0, "Fixture needs at least one spectrum"





    meta.iloc[:-1].to_parquet(path, index=False)





    with _struct_pytest.raises(_struct_error):


        _struct_validate(legacy_store)








def test_mzzarr_rejects_inconsistent_ion_mobility_flag(legacy_store):


    root = _struct_zarr.open_group(str(legacy_store), mode="a")


    root.attrs["has_4d"] = True





    with _struct_pytest.raises(_struct_error):


        _struct_validate(legacy_store)





# MZZARR_METADATA_TESTS_V2



import pytest as _meta_pytest

import zarr as _meta_zarr



from ptero.spectra.validation import (

    validate_store as _meta_validate,

    StoreValidationError as _meta_error,

)





@_meta_pytest.mark.parametrize(

    "key,value",

    [

        ("n_spectra", -1),

        ("n_spectra", "three"),

        ("acquisition_mode", "invalid-mode"),

        ("peaks_variables", "mz,intensity"),

        ("peaks_variables", ["mz", "mz", "intensity"]),

        ("has_4d", "false"),

    ],

)

def test_mzzarr_rejects_invalid_root_metadata(

    legacy_store, key, value

):

    group = _meta_zarr.open_group(

        str(legacy_store), mode="a"

    )

    group.attrs[key] = value



    with _meta_pytest.raises(_meta_error):

        _meta_validate(legacy_store)


# MZZARR_SHORT_ION_MOBILITY_REGRESSION_V1
def test_mzzarr_rejects_short_ion_mobility_with_validation_error(
    tmp_path,
):
    """A short mobility array must produce StoreValidationError."""
    import numpy as np
    import pandas as pd
    import pytest
    import zarr

    from ptero.spectra.validation import (
        StoreValidationError,
        validate_store,
    )

    store = tmp_path / "short_ion_mobility.zarr"

    group = zarr.open_group(
        str(store), mode="w", zarr_format=3
    )

    group.attrs.update({
        "mzzarr_format": "mzZarr",
        "mzzarr_format_version": "1.0",
        "n_spectra": 2,
        "has_4d": True,
        "peaks_variables": [
            "mz", "intensity", "ion_mobility"
        ],
    })

    peaks = group.create_group("peaks")

    peaks.create_array(
        "mz",
        data=np.array([100.0, 200.0, 300.0], dtype="f8"),
    )
    peaks.create_array(
        "intensity",
        data=np.array([10.0, 20.0, 30.0], dtype="f4"),
    )
    peaks.create_array(
        "ion_mobility",
        data=np.array([0.8, 0.9], dtype="f4"),
    )
    group.create_array(
        "offsets",
        data=np.array([0, 1, 3], dtype="i8"),
    )

    pd.DataFrame({
        "scan_index": [0, 1],
        "ms_level": [1, 2],
        "rt_seconds": [1.0, 2.0],
    }).to_parquet(
        store / "spectra.parquet",
        index=False,
        engine="pyarrow",
    )

    with pytest.raises(
        StoreValidationError,
        match=r"ion_mobility.*expected 3",
    ):
        validate_store(store)


# MZZARR_MS_LEVEL_REGRESSION_V1
@pytest.mark.parametrize(
    "levels,error",
    [
        ([1, 2, 3], None),
        (None, "requires ms_level column"),
        ([1, None, 3], "contains null values"),
        ([1, 0, 3], "positive integers"),
        ([1, -1, 3], "positive integers"),
        ([1.0, 2.0, 3.0], "Parquet int64 type"),
    ],
)
def test_mzzarr_v1_ms_level_conformance(tmp_path, levels, error):
    import numpy as np
    import pandas as pd
    import zarr
    from ptero.spectra.validation import validate_store, StoreValidationError

    path = tmp_path / "ms_level_conformance.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update({
        "mzzarr_format": "mzZarr",
        "mzzarr_format_version": "1.0",
    })

    group.create_array(
        "peaks/mz",
        data=np.array([100., 200., 300.], dtype=np.float64)
    )
    group.create_array(
        "peaks/intensity",
        data=np.array([10., 20., 30.], dtype=np.float32)
    )
    group.create_array(
        "offsets",
        data=np.array([0, 1, 2, 3], dtype=np.int64)
    )

    columns = {"scan_index": pd.Series([0, 1, 2], dtype="int64")}
    if levels is not None:
        if any(x is None for x in levels):
            columns["ms_level"] = pd.Series(levels, dtype="Int64")
        elif any(isinstance(x, float) for x in levels):
            columns["ms_level"] = pd.Series(levels, dtype="float64")
        else:
            columns["ms_level"] = pd.Series(levels, dtype="int64")

    pd.DataFrame(columns).to_parquet(path / "spectra.parquet", index=False)

    if error is None:
        report = validate_store(path)
        assert report.n_spectra == 3
    else:
        with pytest.raises(StoreValidationError, match=error):
            validate_store(path)


# MZZARR_RTIME_SEMANTICS_TEST_V1
def test_mzzarr_writer_declares_seconds_for_rtime(tmp_path):
    import numpy as np
    import pandas as pd
    import zarr
    from ptero.spectra.backends.zarr_backend import MsBackendZarr

    path = tmp_path / "rtime_seconds.zarr"
    meta = pd.DataFrame({
        "scan_index": pd.Series([0], dtype="int64"),
        "ms_level": pd.Series([1], dtype="int64"),
        "rtime": pd.Series([12.5], dtype="float64"),
    })
    MsBackendZarr.write(
        path, meta,
        np.asarray([100.0], dtype="float64"),
        np.asarray([10.0], dtype="float32"),
        np.asarray([0, 1], dtype="int64"),
    )
    assert zarr.open_group(str(path), mode="r").attrs["rtime_unit"] == "second"


def test_mzzarr_rejects_invalid_rtime_unit(tmp_path):
    import numpy as np
    import pandas as pd
    import pytest
    import zarr
    from ptero.spectra.backends.zarr_backend import MsBackendZarr
    from ptero.spectra.validation import validate_store, StoreValidationError

    path = tmp_path / "wrong_rtime_unit.zarr"
    meta = pd.DataFrame({
        "scan_index": pd.Series([0], dtype="int64"),
        "ms_level": pd.Series([1], dtype="int64"),
        "rtime": pd.Series([12.5], dtype="float64"),
    })
    MsBackendZarr.write(
        path, meta,
        np.asarray([100.0], dtype="float64"),
        np.asarray([10.0], dtype="float32"),
        np.asarray([0, 1], dtype="int64"),
    )
    zarr.open_group(str(path), mode="a").attrs["rtime_unit"] = "minute"
    with pytest.raises(StoreValidationError, match="rtime_unit"):
        validate_store(path)


# MZZARR_MOBILITY_TESTS_V1
def _mobility_test_data():
    import numpy as np
    import pandas as pd
    meta = pd.DataFrame({
        "scan_index": pd.Series([0], dtype="int64"),
        "ms_level": pd.Series([1], dtype="int64"),
        "ion_mobility": pd.Series([1.1], dtype="float64"),
    })
    return (
        meta,
        np.asarray([100.0], dtype="float64"),
        np.asarray([20.0], dtype="float32"),
        np.asarray([0, 1], dtype="int64"),
        np.asarray([1.1], dtype="float32"),
    )


def test_mzzarr_explicit_mobility_semantics_regular_writer(tmp_path):
    import zarr
    from ptero.spectra.backends.zarr_backend import MsBackendZarr
    from ptero.spectra.validation import validate_store

    meta, mz, inten, offsets, im = _mobility_test_data()
    path = tmp_path / "explicit_mobility.zarr"
    MsBackendZarr.write(
        path, meta, mz, inten, offsets, ion_mobility=im,
        peak_ion_mobility_type="inverse_reduced_ion_mobility",
        peak_ion_mobility_unit="V.s/cm2",
        spectrum_ion_mobility_type="inverse_reduced_ion_mobility",
        spectrum_ion_mobility_unit="V.s/cm2",
    )
    attrs = zarr.open_group(str(path), mode="r").attrs
    assert attrs["peak_ion_mobility_unit"] == "V.s/cm2"
    assert attrs["spectrum_ion_mobility_type"] == "inverse_reduced_ion_mobility"
    validate_store(path)


def test_mzzarr_explicit_mobility_semantics_stream_writer(tmp_path):
    import zarr
    from ptero.spectra.backends.memory import MsBackendMemory
    from ptero.spectra.backends.zarr_backend import ZarrStreamWriter
    from ptero.spectra.validation import validate_store

    meta, mz, inten, offsets, im = _mobility_test_data()
    mem = MsBackendMemory(meta, mz, inten, offsets, ion_mobility=im)
    path = tmp_path / "stream_mobility.zarr"
    writer = ZarrStreamWriter(
        path, has_4d=True,
        peak_ion_mobility_type="inverse_reduced_ion_mobility",
        peak_ion_mobility_unit="V.s/cm2",
        spectrum_ion_mobility_type="inverse_reduced_ion_mobility",
        spectrum_ion_mobility_unit="V.s/cm2",
    )
    writer.append_backend(mem)
    writer.finalize()
    attrs = zarr.open_group(str(path), mode="r").attrs
    assert attrs["peak_ion_mobility_unit"] == "V.s/cm2"
    validate_store(path)


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"peak_ion_mobility_type": "inverse_reduced_ion_mobility"},
         "requires both"),
        ({"peak_ion_mobility_type": "inverse_reduced_ion_mobility",
          "peak_ion_mobility_unit": "millisecond"},
         "invalid peak_ion_mobility_unit"),
        ({"peak_ion_mobility_type": "unsupported_quantity",
          "peak_ion_mobility_unit": "V.s/cm2"},
         "invalid peak_ion_mobility_type"),
        ({"spectrum_ion_mobility_type": "ion_mobility_drift_time"},
         "requires both"),
    ],
)
def test_mzzarr_validator_rejects_invalid_mobility(
    tmp_path, changes, expected
):
    import zarr
    from ptero.spectra.backends.zarr_backend import MsBackendZarr
    from ptero.spectra.validation import validate_store, StoreValidationError

    meta, mz, inten, offsets, im = _mobility_test_data()
    path = tmp_path / "invalid_mobility.zarr"
    MsBackendZarr.write(
        path, meta, mz, inten, offsets, ion_mobility=im,
        peak_ion_mobility_type="inverse_reduced_ion_mobility",
        peak_ion_mobility_unit="V.s/cm2",
        spectrum_ion_mobility_type="inverse_reduced_ion_mobility",
        spectrum_ion_mobility_unit="V.s/cm2",
    )
    attrs = zarr.open_group(str(path), mode="a").attrs
    attrs.update(changes)
    if "peak_ion_mobility_type" in changes and (
        "peak_ion_mobility_unit" not in changes
    ):
        del attrs["peak_ion_mobility_unit"]
    if "spectrum_ion_mobility_type" in changes and (
        "spectrum_ion_mobility_unit" not in changes
    ):
        del attrs["spectrum_ion_mobility_unit"]
    with pytest.raises(StoreValidationError, match=expected):
        validate_store(path)


def test_mzzarr_writer_does_not_guess_mobility_semantics(tmp_path):
    import zarr
    from ptero.spectra.backends.zarr_backend import MsBackendZarr

    meta, mz, inten, offsets, im = _mobility_test_data()
    path = tmp_path / "unspecified_mobility.zarr"
    with pytest.raises(ValueError, match="requires both"):
        MsBackendZarr.write(
            path, meta, mz, inten, offsets, ion_mobility=im
        )
    assert not path.exists()


def test_mzzarr_writer_rejects_wrong_mobility_unit(tmp_path):
    from ptero.spectra.backends.zarr_backend import MsBackendZarr

    meta, mz, inten, offsets, im = _mobility_test_data()
    with pytest.raises(ValueError, match="invalid peak_ion_mobility_unit"):
        MsBackendZarr.write(
            tmp_path / "bad_writer.zarr",
            meta, mz, inten, offsets, ion_mobility=im,
            peak_ion_mobility_type="inverse_reduced_ion_mobility",
            peak_ion_mobility_unit="millisecond",
        )
