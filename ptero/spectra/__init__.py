"""PTERO raw MS data layer: a Spectra-style object with swappable backends.

Public API
----------
PteroSpectra            user-facing raw-spectra object (3D and 4D)
from_mzml, from_dataframe   importers
MsBackendMemory / MsBackendMzML / MsBackendZarr / MsBackendParquet   storage backends
"""
from ._spectra import PteroSpectra
from .io import from_mzml, from_dataframe, open_zarr
from ._bruker import from_bruker_d, from_timstof_dataframe
from .backends._base import MsBackend
from .backends.memory import MsBackendMemory
from .backends.mzml_backend import MsBackendMzML
from .backends.zarr_backend import MsBackendZarr
from .backends.parquet_backend import MsBackendParquet

__all__ = [
    "PteroSpectra", "from_mzml", "from_dataframe", "open_zarr", "from_bruker_d",
    "from_timstof_dataframe", "MsBackend",
    "MsBackendMemory", "MsBackendMzML", "MsBackendZarr", "MsBackendParquet",
]
