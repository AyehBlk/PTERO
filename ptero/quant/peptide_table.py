"""PTERO peptide-level data container."""
from __future__ import annotations
from typing import Optional
import numpy as np, pandas as pd

class PteroPeptideTable:
    def __init__(self, intensities: np.ndarray, metadata: pd.DataFrame, sample_names: list[str]):
        assert intensities.shape[0] == len(metadata), "Row mismatch"
        assert intensities.shape[1] == len(sample_names), "Col mismatch"
        self._intensities = intensities.astype(np.float64)
        self._metadata = metadata.reset_index(drop=True).copy()
        self._sample_names = list(sample_names)

    @property
    def intensities(self): return self._intensities
    @intensities.setter
    def intensities(self, v): self._intensities = v.astype(np.float64)
    @property
    def metadata(self): return self._metadata
    @property
    def sample_names(self): return self._sample_names
    @property
    def n_peptides(self): return self._intensities.shape[0]
    @property
    def n_samples(self): return self._intensities.shape[1]
    @property
    def sequences(self): return self._metadata.get("sequence", pd.Series(dtype=str)).values
    @property
    def is_unique(self):
        if "is_unique" in self._metadata.columns: return self._metadata["is_unique"].values
        if "protein_ids" in self._metadata.columns:
            return np.array([";" not in str(p) for p in self._metadata["protein_ids"]])
        return np.ones(self.n_peptides, dtype=bool)

    def filter_by_protein(self, pid):
        mask = self._metadata["protein_ids"].str.contains(pid, na=False)
        return self._subset_rows(mask)
    def filter_unique(self): return self._subset_rows(self.is_unique)
    def filter_by_qvalue(self, threshold=0.01):
        if "q_value" in self._metadata.columns:
            return self._subset_rows(self._metadata["q_value"] <= threshold)
        return self
    def _subset_rows(self, mask):
        return PteroPeptideTable(self._intensities[mask], self._metadata.loc[mask].reset_index(drop=True), self._sample_names)
    def _subset_columns(self, mask):
        new_names = [n for n,m in zip(self._sample_names, mask) if m]
        return PteroPeptideTable(self._intensities[:, mask], self._metadata.copy(), new_names)
    def missing_fraction(self): return float(np.isnan(self._intensities).sum() / self._intensities.size)
    def to_dataframe(self):
        return pd.concat([self._metadata, pd.DataFrame(self._intensities, columns=self._sample_names)], axis=1)
    def copy(self):
        return PteroPeptideTable(self._intensities.copy(), self._metadata.copy(), list(self._sample_names))
    def __repr__(self): return f"PteroPeptideTable: {self.n_peptides:,} peptides × {self.n_samples} samples ({self.missing_fraction()*100:.1f}% missing)"
    def __len__(self): return self.n_peptides
