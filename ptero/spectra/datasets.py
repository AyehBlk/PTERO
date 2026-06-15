"""
Synthetic raw MS data generator for testing PTERO's spectra layer.

This produces *plausible* LC-MS data you can load, inspect, filter, and store
with the real PTERO code -- no instrument or vendor file required. It models:

* peptide-like **features**, each with an m/z, charge, a Gaussian **elution**
  profile over retention time, an intensity, and (optionally) an **ion
  mobility** value correlated with m/z (as on a real timsTOF);
* an **isotope envelope** per feature (monoisotopic + a few isotopes spaced by
  1/charge with decreasing intensity);
* **MS1 scans** sampled across the run, each summing the features eluting at
  that time, plus random **noise** peaks;
* optional **MS2 scans** that fragment the most intense precursors.

Use it two ways:

>>> from ptero.spectra.datasets import simulate_run, write_mzml
>>> sp = simulate_run(n_features=300, with_ion_mobility=True)   # -> PteroSpectra (4D)
>>> write_mzml("synthetic.mzML", simulate_run(with_ion_mobility=False))  # real mzML (3D)

Scale the parameters up (e.g. ``n_features=5000, n_ms1_scans=3000``) to create a
multi-million-peak run and stress-test the lazy Zarr/Parquet backends.
"""
from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .io import from_dataframe
from ._spectra import PteroSpectra


# ── feature simulation ──────────────────────────────────────────────────────

def simulate_features(
    n_features: int = 300,
    mz_range=(300.0, 1500.0),
    rt_range=(60.0, 1740.0),
    seed: int = 0,
    with_ion_mobility: bool = True,
) -> pd.DataFrame:
    """Create a table of peptide-like features (one row per feature)."""
    rng = np.random.RandomState(seed)
    mz = rng.uniform(*mz_range, n_features)
    charge = rng.choice([2, 3, 4], n_features, p=[0.6, 0.3, 0.1])
    rt_apex = rng.uniform(*rt_range, n_features)
    rt_sigma = rng.uniform(2.0, 8.0, n_features)          # elution width (s)
    intensity = 10 ** rng.uniform(4.0, 8.0, n_features)   # log-uniform abundance
    df = pd.DataFrame({
        "feature_id": np.arange(n_features),
        "mz": mz, "charge": charge,
        "rt_apex": rt_apex, "rt_sigma": rt_sigma,
        "intensity": intensity,
    })
    if with_ion_mobility:
        # 1/K0 rises roughly linearly with m/z, with biological scatter
        df["ion_mobility"] = 0.6 + (mz - mz_range[0]) / (mz_range[1] - mz_range[0]) * 0.7 \
            + rng.normal(0, 0.03, n_features)
    return df


def _isotope_pattern(mz: float, charge: int, base_int: float, n_iso: int = 4):
    """Monoisotopic + n_iso isotope peaks (spacing 1/charge, geometric decay)."""
    spacing = 1.0 / charge
    mzs = mz + spacing * np.arange(n_iso)
    # crude averagine-like decay
    weights = np.array([1.0, 0.55, 0.25, 0.10][:n_iso])
    return mzs, base_int * weights


# ── run simulation ───────────────────────────────────────────────────────────

def simulate_run(
    n_features: int = 300,
    n_ms1_scans: int = 120,
    rt_range=(60.0, 1740.0),
    mz_range=(300.0, 1500.0),
    with_ion_mobility: bool = True,
    with_ms2: bool = True,
    ms2_topn: int = 5,
    noise_peaks: int = 30,
    mz_ppm_jitter: float = 5.0,
    seed: int = 0,
) -> PteroSpectra:
    """Simulate a full LC-MS(/MS) run and return a :class:`PteroSpectra`.

    The object is 4D (carries per-peak ion mobility) when
    ``with_ion_mobility=True``, else 3D.
    """
    rng = np.random.RandomState(seed)
    feats = simulate_features(n_features, mz_range, rt_range, seed, with_ion_mobility)
    scan_times = np.linspace(rt_range[0], rt_range[1], n_ms1_scans)

    rows, peaks = [], []
    scan_index = 0

    for t in scan_times:
        # which features are eluting near time t (within ~3 sigma)
        active = feats[np.abs(feats["rt_apex"] - t) < 3 * feats["rt_sigma"]]
        mz_list, int_list, im_list = [], [], []
        for _, f in active.iterrows():
            scale = np.exp(-0.5 * ((t - f["rt_apex"]) / f["rt_sigma"]) ** 2)
            if scale < 1e-3:
                continue
            iso_mz, iso_int = _isotope_pattern(f["mz"], int(f["charge"]), f["intensity"] * scale)
            jitter = 1 + rng.normal(0, mz_ppm_jitter * 1e-6, len(iso_mz))
            mz_list.append(iso_mz * jitter)
            int_list.append(iso_int * rng.uniform(0.9, 1.1, len(iso_mz)))
            if with_ion_mobility:
                im_list.append(np.full(len(iso_mz), f["ion_mobility"]) + rng.normal(0, 0.005, len(iso_mz)))
        # background noise
        if noise_peaks:
            mz_list.append(rng.uniform(*mz_range, noise_peaks))
            int_list.append(10 ** rng.uniform(3.0, 5.0, noise_peaks))
            if with_ion_mobility:
                im_list.append(rng.uniform(0.6, 1.4, noise_peaks))

        if mz_list:
            mz = np.concatenate(mz_list)
            inten = np.concatenate(int_list).astype(np.float32)
            order = np.argsort(mz)
            mz, inten = mz[order], inten[order]
            im = np.concatenate(im_list)[order].astype(np.float32) if with_ion_mobility else None
        else:
            mz, inten, im = np.array([]), np.array([], np.float32), (np.array([], np.float32) if with_ion_mobility else None)

        rows.append({"ms_level": 1, "rtime": float(t), "scan_index": scan_index,
                     "precursor_mz": np.nan, "centroided": True})
        pk = {"mz": mz, "intensity": inten}
        if with_ion_mobility:
            pk["ion_mobility"] = im
        peaks.append(pk)
        scan_index += 1

        # interleave MS2 scans for the top-N precursors at this RT
        if with_ms2 and len(active):
            top = active.assign(
                cur=active["intensity"] * np.exp(-0.5 * ((t - active["rt_apex"]) / active["rt_sigma"]) ** 2)
            ).nlargest(ms2_topn, "cur")
            for _, f in top.iterrows():
                n_frag = rng.randint(6, 20)
                frag_mz = np.sort(rng.uniform(100, f["mz"], n_frag))
                frag_int = (10 ** rng.uniform(3.5, 6.5, n_frag)).astype(np.float32)
                rows.append({"ms_level": 2, "rtime": float(t + 0.1), "scan_index": scan_index,
                             "precursor_mz": float(f["mz"]), "precursor_charge": int(f["charge"]),
                             "centroided": True})
                pk2 = {"mz": frag_mz, "intensity": frag_int}
                if with_ion_mobility:
                    pk2["ion_mobility"] = np.full(n_frag, f["ion_mobility"], np.float32)
                peaks.append(pk2)
                scan_index += 1

    meta = pd.DataFrame(rows)
    meta["data_origin"] = "synthetic"
    sp = from_dataframe(meta, peaks)
    return sp


# ── convenience ───────────────────────────────────────────────────────────

def make_3d(**kw) -> PteroSpectra:
    kw.setdefault("with_ion_mobility", False)
    return simulate_run(**kw)


def make_4d(**kw) -> PteroSpectra:
    kw.setdefault("with_ion_mobility", True)
    return simulate_run(**kw)


# ── write a real mzML file (3D) ──────────────────────────────────────────────

def _encode(vals: np.ndarray) -> str:
    comp = zlib.compress(struct.pack("<%dd" % len(vals), *np.asarray(vals, dtype=np.float64)))
    return base64.b64encode(comp).decode()


def write_mzml(path, spectra: PteroSpectra) -> str:
    """Write a (3D) PteroSpectra to a spec-compliant, zlib+base64 mzML file.

    mzML's standard peak arrays are m/z + intensity, so ion mobility is dropped
    on write -- for 4D data, persist to Zarr/Parquet with ``set_backend`` instead.
    Files written here are readable again with :func:`ptero.spectra.from_mzml`
    (and by other mzML tools).
    """
    path = str(path)
    sd = spectra.spectra_data(["ms_level", "rtime", "precursor_mz"]).reset_index(drop=True)
    mzs = spectra.mz()
    ints = spectra.intensity()

    def bda(vals, kind):
        acc = "MS:1000514" if kind == "mz" else "MS:1000515"
        nm = "m/z array" if kind == "mz" else "intensity array"
        return (f'<binaryDataArray encodedLength="0">'
                f'<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
                f'<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression"/>'
                f'<cvParam cvRef="MS" accession="{acc}" name="{nm}"/>'
                f'<binary>{_encode(vals)}</binary></binaryDataArray>')

    specs = []
    for i in range(len(spectra)):
        lvl = int(sd.loc[i, "ms_level"])
        rt = float(sd.loc[i, "rtime"])
        pmz = sd.loc[i, "precursor_mz"]
        prec = ""
        if lvl >= 2 and pd.notna(pmz):
            prec = (f'<precursorList count="1"><precursor><selectedIonList count="1">'
                    f'<selectedIon><cvParam cvRef="MS" accession="MS:1000744" '
                    f'name="selected ion m/z" value="{float(pmz)}"/></selectedIon>'
                    f'</selectedIonList></precursor></precursorList>')
        specs.append(
            f'<spectrum index="{i}" id="scan={i+1}" defaultArrayLength="{len(mzs[i])}">'
            f'<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{lvl}"/>'
            f'<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum"/>'
            f'<scanList count="1"><scan><cvParam cvRef="MS" accession="MS:1000016" '
            f'name="scan start time" value="{rt}" unitName="second"/></scan></scanList>'
            f'{prec}<binaryDataArrayList count="2">{bda(mzs[i], "mz")}{bda(ints[i], "int")}'
            f'</binaryDataArrayList></spectrum>'
        )
    doc = ('<?xml version="1.0" encoding="utf-8"?>'
           '<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">'
           '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">'
           f'<run id="synthetic"><spectrumList count="{len(spectra)}">{"".join(specs)}'
           '</spectrumList></run></mzML></indexedmzML>')
    Path(path).write_text(doc, encoding="utf-8")
    return path
