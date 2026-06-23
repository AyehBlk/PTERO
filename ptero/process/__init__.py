"""PTERO raw-data processing: centroiding/deisotoping into a processed Zarr store.

Scope A is staged: Step 1 (this release) provides the processed-store *container*
and an identity-transform round-trip; the centroiding/deisotoping algorithms
(Steps 2-3, wrapping a proven deisotoper) plug into :func:`process_store` as a
``transform`` without changing the container or the readers.

The processed store is an ordinary PTERO Zarr store (same schema as a raw store),
so it reopens with :func:`ptero.spectra.open_zarr` and feeds the raw<->ID link
unchanged; it additionally carries a processed marker + processing provenance,
readable with :func:`is_processed_store` / :func:`read_processing_provenance`.
"""
from ._store import (
    process_store,
    identity_transform,
    is_processed_store,
    read_processing_provenance,
    PROCESSED_SCHEMA_VERSION,
    PROCESSED_FLAG_ATTR,
    PROCESSING_PROVENANCE_ATTR,
)
from ._centroiding import make_centroid_transform

__all__ = [
    "process_store",
    "identity_transform",
    "make_centroid_transform",
    "is_processed_store",
    "read_processing_provenance",
    "PROCESSED_SCHEMA_VERSION",
    "PROCESSED_FLAG_ATTR",
    "PROCESSING_PROVENANCE_ATTR",
]