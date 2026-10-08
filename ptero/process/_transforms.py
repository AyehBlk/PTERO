"""Typed processing-transform helpers with machine-readable provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

PeaksDict = dict[str, np.ndarray]


@runtime_checkable
class SpectrumTransform(Protocol):
    """Protocol for reproducible per-spectrum processing operations."""
    def __call__(self, peaks: PeaksDict) -> PeaksDict: ...
    def provenance(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FunctionTransform:
    """Attach stable provenance to an ordinary ``peaks -> peaks`` callable."""
    function: Callable[[PeaksDict], PeaksDict]
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    library: str | None = None
    library_version: str | None = None

    def __call__(self, peaks: PeaksDict) -> PeaksDict:
        return self.function(peaks)

    def provenance(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "library": self.library,
            "library_version": self.library_version,
        }


def transform_provenance(transform) -> dict | None:
    """Return provenance declared by a transform, if available."""
    method = getattr(transform, "provenance", None)
    if callable(method):
        return dict(method())
    return None
