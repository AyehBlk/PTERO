"""PteroExperiment — central data container for PTERO."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional
from .typing import ExperimentMetadata, LAYER_RAW
from .provenance import ProvenanceTracker
from .peptide_table import PteroPeptideTable

class PteroExperiment:
    def __init__(self, X, obs, var, layers=None, peptides=None, metadata=None, uns=None):
        X = np.asarray(X, dtype=np.float64)
        assert X.ndim == 2, f"X must be 2D, got {X.ndim}"
        n_prot, n_samp = X.shape
        assert len(obs) == n_samp, f"obs rows ({len(obs)}) != X cols ({n_samp})"
        assert len(var) == n_prot, f"var rows ({len(var)}) != X rows ({n_prot})"
        self._X = X; self._obs = obs.copy(); self._var = var.copy()
        if self._obs.index.name is None and "sample_id" in self._obs.columns:
            self._obs.index = self._obs["sample_id"]
        if self._var.index.name is None and "protein_id" in self._var.columns:
            self._var.index = self._var["protein_id"]
        self._layers = {}
        if layers:
            for k,v in layers.items(): self._set_layer(k,v)
        self._peptides = peptides
        self._metadata = metadata or ExperimentMetadata()
        self._uns = uns or {}
        self._history = ProvenanceTracker()
        self._created_at = datetime.now(timezone.utc).isoformat()

    @property
    def X(self): return self._X
    @X.setter
    def X(self, v):
        v = np.asarray(v, dtype=np.float64)
        assert v.shape == self._X.shape; self._X = v
    @property
    def obs(self): return self._obs
    @obs.setter
    def obs(self, v): assert len(v) == self.n_samples; self._obs = v.copy()
    @property
    def var(self): return self._var
    @var.setter
    def var(self, v): assert len(v) == self.n_proteins; self._var = v.copy()
    @property
    def layers(self): return self._layers
    @property
    def peptides(self): return self._peptides
    @peptides.setter
    def peptides(self, v): self._peptides = v
    @property
    def metadata(self): return self._metadata
    @property
    def uns(self): return self._uns
    @property
    def history(self): return self._history
    @property
    def n_proteins(self): return self._X.shape[0]
    @property
    def n_samples(self): return self._X.shape[1]
    @property
    def shape(self): return self._X.shape
    @property
    def protein_ids(self): return self._var.index.values
    @property
    def sample_ids(self): return self._obs.index.values
    @property
    def conditions(self):
        return self._obs["condition"].unique() if "condition" in self._obs.columns else None
    @property
    def missing_fraction(self): return float(np.isnan(self._X).sum() / self._X.size)

    def _set_layer(self, name, data):
        data = np.asarray(data, dtype=np.float64)
        assert data.shape == self._X.shape, f"Layer shape mismatch"
        self._layers[name] = data
    def add_layer(self, name, data): self._set_layer(name, data)
    def get_layer(self, name):
        if name == "X": return self._X
        if name not in self._layers: raise KeyError(f"Layer '{name}' not found")
        return self._layers[name]

    def subset_proteins(self, mask):
        idx = np.where(mask)[0] if mask.dtype == bool else mask
        new_layers = {k: v[idx] for k,v in self._layers.items()}
        new = PteroExperiment(self._X[idx], self._obs.copy(), self._var.iloc[idx].copy(),
                              layers=new_layers, metadata=self._metadata, uns=deepcopy(self._uns))
        new._history = deepcopy(self._history); return new

    def subset_samples(self, mask):
        idx = np.where(mask)[0] if mask.dtype == bool else mask
        new_layers = {k: v[:, idx] for k,v in self._layers.items()}
        new = PteroExperiment(self._X[:, idx], self._obs.iloc[idx].copy(), self._var.copy(),
                              layers=new_layers, metadata=self._metadata, uns=deepcopy(self._uns))
        new._history = deepcopy(self._history); return new

    def subset_condition(self, condition):
        assert "condition" in self._obs.columns
        return self.subset_samples((self._obs["condition"] == condition).values)

    def protein(self, pid):
        if pid in self._var.index:
            idx = self._var.index.get_loc(pid)
        elif "gene_name" in self._var.columns:
            m = self._var[self._var["gene_name"] == pid]
            if len(m) == 0: raise KeyError(f"'{pid}' not found")
            idx = self._var.index.get_loc(m.index[0])
        else: raise KeyError(f"'{pid}' not found")
        return pd.Series(self._X[idx], index=self._obs.index, name=pid)

    def to_dataframe(self, layer="X"):
        data = self.get_layer(layer) if layer != "X" else self._X
        return pd.DataFrame(data, index=self._var.index, columns=self._obs.index)

    def to_long_format(self, layer="X"):
        df = self.to_dataframe(layer)
        long = df.reset_index().melt(id_vars=[df.index.name or "index"], var_name="sample_id", value_name="intensity")
        long = long.rename(columns={df.index.name or "index": "protein_id"})
        obs_cols = [c for c in self._obs.columns if c != "sample_id"]
        if obs_cols:
            obs_merge = self._obs[obs_cols].copy().reset_index(drop=True)
            obs_merge["sample_id"] = self._obs.index.values
            long = long.merge(obs_merge, on="sample_id", how="left")
        return long

    def summary(self):
        lines = [f"{'='*55}", f"  PTERO: {self._metadata.name}", f"{'='*55}",
                 f"  Proteins: {self.n_proteins:>7,}  Samples: {self.n_samples:>4}",
                 f"  Missing:  {self.missing_fraction*100:>6.1f}%"]
        if self._peptides: lines.append(f"  Peptides: {self._peptides.n_peptides:>7,}")
        if self._layers: lines.append(f"  Layers:   {', '.join(self._layers.keys())}")
        if self.conditions is not None: lines.append(f"  Conditions: {', '.join(str(c) for c in self.conditions)}")
        lines.append(f"  Mode: {self._metadata.acquisition_mode.value} | Engine: {self._metadata.search_engine.value}")
        lines.append(f"  Steps: {len(self._history)}")
        lines.append(f"{'='*55}")
        return "\n".join(lines)

    def copy(self):
        new = PteroExperiment(self._X.copy(), self._obs.copy(), self._var.copy(),
            layers={k:v.copy() for k,v in self._layers.items()},
            peptides=self._peptides.copy() if self._peptides else None,
            metadata=deepcopy(self._metadata), uns=deepcopy(self._uns))
        new._history = deepcopy(self._history); return new

    def __repr__(self):
        return f"PteroExperiment: {self.n_proteins:,} proteins × {self.n_samples} samples | missing: {self.missing_fraction*100:.1f}%"
    def __str__(self): return self.summary()
