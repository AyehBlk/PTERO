"""
Data layer for the protein-centric Identifications view (Stage 1).

Reads MaxQuant ``proteinGroups.txt`` and ``peptides.txt`` and joins them into:

* a **protein table** -- one row per protein group, using MaxQuant's own protein
  inference (sequence coverage, peptide counts, razor+unique counts, score), and
* a **peptide table** -- the peptides of a chosen protein group, each classified
  ``unique`` / ``razor`` / ``shared`` from MaxQuant's ``Unique (Groups)`` flag and
  ``Leading razor protein``.

These are deliberately thin readers over MaxQuant's authoritative output (we do
not re-derive protein inference). Column names are the real ones observed in the
user's files; missing optional columns degrade gracefully.

Gene / protein *names* are read from ``proteinGroups.txt`` where present and
filled from ``peptides.txt`` where blank (the protein-groups file can carry empty
name fields for some groups while the peptides file has them).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_tsv(path: str) -> pd.DataFrame:
    """Read a MaxQuant tab file as all-strings (BOM-safe), NaN -> ''."""
    return pd.read_csv(path, sep="\t", dtype=str,
                       encoding="utf-8-sig").fillna("")


def _to_int(s: str) -> int:
    try:
        return int(float(s)) if s != "" else 0
    except (ValueError, TypeError):
        return 0


def _first_int(s: str) -> int:
    """Parse a MaxQuant count cell that may be a single int OR a ';'-separated
    per-protein list (e.g. ``"21;19;3"`` for a multi-protein group). Returns the
    first value (the leading protein's count). ``""`` -> 0."""
    if s is None or s == "":
        return 0
    head = s.split(";", 1)[0].strip()
    return _to_int(head)


def _to_float(s: str) -> float:
    try:
        return float(s) if s != "" else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _group_in(cell: str, gid: str) -> bool:
    """True if protein-group id *gid* is in a ';'-separated Protein group IDs cell."""
    return gid in cell.split(";")


class IdentificationData:
    """Joined MaxQuant protein + peptide tables for the Identifications view."""

    def __init__(self, protein_groups_path: str, peptides_path: str):
        self.pg = _read_tsv(protein_groups_path)
        self.pep = _read_tsv(peptides_path)
        self._protein_table = self._build_protein_table()

    # ── protein table ────────────────────────────────────────────────────
    def _gene_protein_fallback(self, gid: str) -> tuple[str, str]:
        """Most common non-empty (gene, protein) among this group's peptides."""
        col_gene = "Gene names" if "Gene names" in self.pep.columns else None
        col_prot = "Protein names" if "Protein names" in self.pep.columns else None
        sub = self.pep[self.pep.get("Protein group IDs", "").apply(
            lambda c: _group_in(c, gid))]
        gene = next((x for x in (sub[col_gene] if col_gene else []) if x), "")
        prot = next((x for x in (sub[col_prot] if col_prot else []) if x), "")
        return gene, prot

    def _build_protein_table(self) -> pd.DataFrame:
        pg = self.pg
        rows = []
        for _, r in pg.iterrows():
            gid = r.get("id", "")
            gene = r.get("Gene names", "")
            prot = r.get("Protein names", "")
            if not gene or not prot:
                fg, fp = self._gene_protein_fallback(gid)
                gene = gene or fg
                prot = prot or fp
            rows.append({
                "gene": gene,
                "protein": prot,
                "n_peptides": _to_int(r.get("Peptides", ""))
                              or _first_int(r.get("Peptide counts (all)", "")),
                "n_unique": _to_int(r.get("Unique peptides", ""))
                            or _first_int(r.get("Peptide counts (unique)", "")),
                "razor_unique": _to_int(r.get("Razor + unique peptides", ""))
                                or _first_int(r.get("Peptide counts (razor+unique)", "")),
                "coverage_pct": _to_float(r.get("Sequence coverage [%]", "")),
                "n_psms": _to_int(r.get("MS/MS count", "")),
                "score": _to_float(r.get("Score", "")),
                "majority_ids": r.get("Majority protein IDs", ""),
                "_gid": gid,
            })
        df = pd.DataFrame(rows)
        # most-interesting first: highest score
        if "score" in df.columns and len(df):
            df = df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        return df

    @property
    def protein_table(self) -> pd.DataFrame:
        """Display columns for the protein table (keeps hidden ``_gid``)."""
        return self._protein_table

    # ── peptide table for a chosen group ─────────────────────────────────
    def peptides_for_group(self, gid: str) -> pd.DataFrame:
        pep = self.pep
        mask = pep.get("Protein group IDs", "").apply(lambda c: _group_in(c, gid))
        sub = pep[mask].copy()
        if not len(sub):
            return pd.DataFrame(columns=["peptide", "unique", "mods", "charge", "score", "PEP", "_seq"])

        def classify(r):
            if r.get("Unique (Groups)", "") == "yes":
                return "unique"
            leading = r.get("Leading razor protein", "")
            proteins = r.get("Proteins", "").split(";")
            return "razor" if leading in proteins else "shared"

        out = pd.DataFrame({
            "peptide": sub["Sequence"],
            "unique": sub.apply(classify, axis=1),
            "mods": sub.get("Modifications", "").replace("Unmodified", ""),
            "charge": sub.get("Charges", ""),
            "score": sub.get("Score", "").apply(_to_float),
            "PEP": sub.get("PEP", "").apply(_to_float),
            "_seq": sub["Sequence"],          # plain AA sequence, for PSM lookup
        })
        return out.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)