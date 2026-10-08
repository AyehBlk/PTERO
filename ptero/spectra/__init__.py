"""PTERO raw MS data layer: a Spectra-style object with swappable backends."""
from ._spectra import PteroSpectra
from .io import from_mzml, from_dataframe, open_zarr
from ._bruker import from_bruker_d, from_timstof_dataframe
from ._centroid import detect_centroided, annotate_store_centroided
from .backends._base import MsBackend
from .backends.memory import MsBackendMemory
from .backends.mzml_backend import MsBackendMzML
from .backends.zarr_backend import MsBackendZarr, ZarrStreamWriter
from .backends.parquet_backend import MsBackendParquet
from ._schema import PTERO_FORMAT, PTERO_FORMAT_VERSION
from .validation import StoreValidationError, StoreValidationReport, validate_store

__all__ = [
    "PteroSpectra", "from_mzml", "from_dataframe", "open_zarr", "from_bruker_d",
    "from_timstof_dataframe", "detect_centroided", "annotate_store_centroided", "MsBackend",
    "MsBackendMemory", "MsBackendMzML", "MsBackendZarr", "MsBackendParquet",
    "ZarrStreamWriter", "PTERO_FORMAT", "PTERO_FORMAT_VERSION",
    "StoreValidationError", "StoreValidationReport", "validate_store",
]