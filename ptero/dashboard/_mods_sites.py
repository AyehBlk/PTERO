"""
Modification-site layer for the Identifications view (Stage 3).

Reads MaxQuant ``<Mod>Sites.txt`` files (e.g. ``Oxidation (M)Sites.txt``,
``Phospho (STY)Sites.txt``) -- the authoritative per-site tables MaxQuant
produces for **variable, localisable** modifications. Each row is one
modification site with the protein position, the modified residue, and the
localisation probability ALREADY COMPUTED by MaxQuant -- so we do not re-derive
positions (which would be error-prone). Generic over modification type: any
``*Sites.txt`` present is read, so a phospho search's ``Phospho (STY)Sites.txt``
works exactly like oxidation here.

Sample-agnostic (Stage-2 discipline): a site's protein position does not depend
on how many samples were run. The per-experiment columns (``... dda1``) are
ignored for now; the global ``Localization prob`` / ``Positions within
proteins`` are used.

Terminal or fixed modifications (e.g. ``Acetyl (Protein N-term)``,
``Carbamidomethyl (C)``) have no ``Sites.txt`` -- correctly, since they are not
localisable internal sites. They appear in the peptide table's modification
column but not as sequence sites.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# "Oxidation (M)Sites.txt" -> "Oxidation (M)";  "Phospho (STY)Sites.txt" -> "Phospho (STY)"
_SITES_RE = re.compile(r"^(.*?)Sites\.txt$", re.IGNORECASE)


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig").fillna("")


def _to_float(s: str) -> float:
    try:
        return float(s) if s not in ("", "NaN") else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _first(s: str) -> str:
    """First element of a ';'-separated MaxQuant cell (leading protein/position)."""
    return s.split(";", 1)[0].strip() if s else ""


class ModificationSites:
    """All modification sites from the ``*Sites.txt`` files in a MaxQuant folder.

    Sites are indexed by protein-group id so the view can fetch the sites for a
    selected protein. Each site is a dict: ``mod`` (e.g. "Oxidation (M)"),
    ``position`` (1-based protein position), ``residue`` (amino acid),
    ``localization`` (float 0..1 or NaN), ``group_ids`` (list).
    """

    def __init__(self, txt_folder: str):
        self.by_group: dict[str, list[dict]] = {}
        self.mod_types: list[str] = []
        folder = Path(txt_folder)
        if not folder.is_dir():
            return
        for f in sorted(folder.glob("*Sites.txt")):
            m = _SITES_RE.match(f.name)
            mod_name = (m.group(1).strip() if m else f.stem)
            self._read_sites_file(f, mod_name)

    def _read_sites_file(self, path: Path, mod_name: str):
        df = _read_tsv(path)
        # tolerate column-name vari/casing differences
        cols = {c.lower(): c for c in df.columns}

        def col(*names):
            for n in names:
                if n.lower() in cols:
                    return cols[n.lower()]
            return None

        c_proteins = col("Proteins")
        c_pos = col("Positions within proteins", "Position")
        c_res = col("Amino acid")
        c_loc = col("Localization prob")
        c_grp = col("Protein group IDs")
        c_id = col("id")
        c_rev = col("Reverse")
        c_con = col("Potential contaminant")
        if c_pos is None or c_grp is None:
            return  # not a usable sites file

        self.mod_types.append(mod_name)
        for _, r in df.iterrows():
            if c_rev and r[c_rev] == "+":          # skip decoys
                continue
            if c_con and r[c_con] == "+":          # skip contaminants
                continue
            pos = _first(r[c_pos])
            if not pos:
                continue
            try:
                position = int(float(pos))
            except (ValueError, TypeError):
                continue
            protein_positions = {}
            if c_proteins:
                accessions = [v.strip() for v in r[c_proteins].split(";")]
                positions = [v.strip() for v in r[c_pos].split(";")]
                if len(accessions) == len(positions):
                    for accession, value in zip(accessions, positions):
                        if not accession or not value:
                            continue
                        try:
                            site_position = int(value)
                        except ValueError:
                            continue
                        if site_position > 0:
                            protein_positions[accession] = site_position

            site = {
                "mod": mod_name,
                "site_id": r[c_id].strip() if c_id else None,
                "position": position,                       # legacy leading position
                "residue": r[c_res] if c_res else "",
                "localization": _to_float(r[c_loc]) if c_loc else float("nan"),
                "protein_positions": protein_positions,
            }
            # a site row can belong to several groups (';'-separated)
            groups = (r[c_grp].split(";") if c_grp else [])
            for g in groups:
                g = g.strip()
                if g:
                    self.by_group.setdefault(g, []).append(site)

    def for_group(self, gid: str) -> list[dict]:
        """Modification sites for protein-group *gid* (may be empty)."""
        return self.by_group.get(gid, [])

    def for_group_accession(
        self,
        gid: str,
        accession: str,
        sequence: str,
        supported_site_ids: dict[str, set[str]] | None = None,
    ) -> list[dict]:
        """Sites explicitly assigned to an accession and verified against FASTA.

        Missing accession-position associations and residue mismatches are
        excluded rather than transferring coordinates from another protein.
        """
        result = []
        for site in self.for_group(gid):
            if supported_site_ids is not None:
                site_id = site.get("site_id")
                if not site_id or site_id not in supported_site_ids.get(
                    site["mod"], set()
                ):
                    continue
            position = site.get("protein_positions", {}).get(accession)
            if position is None or not 1 <= position <= len(sequence):
                continue
            if sequence[position - 1] != site["residue"]:
                continue
            result.append({
                **site,
                "position": position,
                "accession": accession,
            })
        return result

    def __len__(self):
        return sum(len(v) for v in self.by_group.values())