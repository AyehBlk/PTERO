"""
Shared state for the multi-page PTERO dashboard.

A single :class:`AppState` instance is created once by :func:`build_app` and
handed to every page. Pages read and write the same object, so an input loaded
on one page (e.g. a Zarr store opened on the raw-data page, or PSMs + an mzML
store loaded on the ID page) is visible to any other page that needs it. This is
the mechanism that lets the pages interconnect as more are added.

Kept deliberately tiny and dependency-free: it holds objects and a list of
change-callbacks. Pages register a callback to refresh themselves when a value
they care about changes.
"""
from __future__ import annotations

from typing import Callable, Optional


class AppState:
    """Mutable state shared across dashboard pages.

    Attributes
    ----------
    raw_store
        A ``PteroSpectra`` opened on the raw-data / converter page (a raw
        ``.d``→Zarr store), or ``None``. Suitable for chromatograms, maps, QC.
    id_store
        A ``PteroSpectra`` opened on the ID page from a **vendor-peak-picked
        mzML** (the representation whose fragment m/z match a search engine's
        masses), or ``None``. This is the store the ID view annotates against.
    psms
        A psm_utils ``PSMList`` loaded on the ID page, or ``None``.
    psm_run
        The run id (``PSM.run``) registered for ``id_store``, or ``None``.
    quant
        A ``QuantData`` loaded on the differential-expression page, or ``None``.
    """

    def __init__(self) -> None:
        self.raw_store = None
        self.quant = None            # QuantData loaded on the differential-expression page, or None
        self.id_store = None
        self.psms = None
        self.psm_run: Optional[str] = None
        self._listeners: list[Callable[[str], None]] = []

    # ── change notification ──────────────────────────────────────────────
    def on_change(self, callback: Callable[[str], None]) -> None:
        """Register ``callback(key)`` to run whenever :meth:`set` is called."""
        self._listeners.append(callback)

    def set(self, key: str, value) -> None:
        """Set an attribute and notify listeners with the changed key."""
        setattr(self, key, value)
        for cb in list(self._listeners):
            try:
                cb(key)
            except Exception:
                # a misbehaving listener must not break state updates
                pass