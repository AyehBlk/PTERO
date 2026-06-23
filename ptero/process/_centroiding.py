"""
ptero.process._centroiding -- a per-spectrum centroiding transform for
``process_store``, wrapping pyOpenMS ``PeakPickerHiRes``.

This is the first real (non-identity) processing step. It is for **profile**
high-resolution data -- e.g. a Thermo Orbitrap ``.raw`` converted to a *profile*
mzML -- where each peak is a dense cluster of m/z points that a search engine
cannot use directly. ``PeakPickerHiRes`` reduces each profile peak to a single
centroid (m/z + intensity), which is what fragment annotation and search engines
expect.

Scope (deliberate, for now)
---------------------------
* **3D only (m/z + intensity).** Thermo Orbitrap data has no ion mobility, so the
  m/z-only picker is correct. A 4D (ion-mobility) source would need per-mobility
  -slice picking to preserve the per-peak 1/K0; that is **not** implemented here,
  and timsTOF ``.d`` data is already centroided so does not need it. Applying this
  transform to a 4D store raises, rather than silently dropping mobility.
* pyOpenMS is imported lazily and only when the transform is built, so importing
  ``ptero.process`` stays light; install it with the ``process`` extra
  (``pip install pyopenms``).

Use it with :func:`ptero.process.process_store`::

    from ptero.process import process_store, make_centroid_transform
    process_store(profile_zarr, centroided_zarr,
                  transform=make_centroid_transform(),
                  steps=[{"name": "centroid", "params": {}, "library": "pyopenms",
                          "library_version": "PeakPickerHiRes"}])

Validate the result the same way as any store: ``detect_centroided`` should flip
from profile to centroid, and the median peaks-per-spectrum should drop sharply.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def make_centroid_transform(signal_to_noise: float = 0.0) -> Callable[[dict], dict]:
    """Build a per-spectrum centroiding transform (pyOpenMS ``PeakPickerHiRes``).

    Parameters
    ----------
    signal_to_noise : float, default 0.0
        ``PeakPickerHiRes`` signal-to-noise threshold. ``0`` disables the SNR
        estimator (pick every peak); raise it (e.g. ``1.0``) to drop low-S/N
        peaks. Leave at 0 for clean high-res profile data.

    Returns
    -------
    callable
        ``peaks_dict -> peaks_dict`` suitable as the ``transform`` argument of
        :func:`ptero.process.process_store`. Raises :class:`NotImplementedError`
        if handed a 4D spectrum (peaks carrying ``ion_mobility``).
    """
    try:
        from pyopenms import MSSpectrum, PeakPickerHiRes
    except Exception as e:  # pragma: no cover - install-environment guard
        raise ImportError(
            "Centroiding needs pyOpenMS. Install it with:  pip install pyopenms"
        ) from e

    picker = PeakPickerHiRes()
    if signal_to_noise and signal_to_noise > 0:
        params = picker.getParameters()
        params.setValue("signal_to_noise", float(signal_to_noise))
        picker.setParameters(params)

    def centroid(peaks: dict) -> dict:
        if peaks.get("ion_mobility") is not None:
            raise NotImplementedError(
                "make_centroid_transform handles 3D (m/z-only) profile data; the "
                "source is 4D (per-peak ion mobility). Per-mobility-slice centroiding "
                "is not implemented, and timsTOF .d data is already centroided."
            )
        mz = np.asarray(peaks["mz"], dtype=np.float64)
        inten = np.asarray(peaks["intensity"], dtype=np.float64)
        if mz.size == 0:
            return {"mz": mz, "intensity": inten}
        s = MSSpectrum()
        s.set_peaks([mz, inten])
        out = MSSpectrum()
        picker.pick(s, out)
        omz, oint = out.get_peaks()
        return {"mz": np.asarray(omz, dtype=np.float64),
                "intensity": np.asarray(oint, dtype=np.float64)}

    return centroid