"""
FASTA / protein-sequence layer for the Identifications view (Stage 2).

Sample-agnostic by design: every function here takes a protein *sequence* and a
set of identified *peptides* and returns positions / coverage. None of it knows
about samples, runs, or mzML files -- so it is correct for a single run today and
unchanged when multi-sample support is added later (a peptide maps to the same
protein position regardless of how many samples it was seen in).

Provides:
* :class:`ProteinSequences` -- a parsed FASTA keyed by accession.
* :func:`locate_peptide` -- all positions a peptide occurs in a protein sequence.
* :func:`map_peptides` -- map a group's peptides onto its protein sequence,
  returning per-peptide spans (+ which could not be located), and the residue
  coverage. This is the data the coverage map draws.

The user must supply the *same* FASTA used in the search; a different version may
not contain a peptide's exact sequence, which :func:`map_peptides` reports as
"not located" rather than failing silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# accession from a UniProt-style header "sp|P06733|ENOA_HUMAN ..." -> "P06733";
# falls back to the first whitespace-delimited token for plain headers.
_ACC_RE = re.compile(r"^[a-zA-Z]+\|([^|]+)\|")


def _parse_accession(header: str) -> str:
    m = _ACC_RE.match(header)
    if m:
        return m.group(1)
    return header.split()[0] if header else header


class ProteinSequences:
    """A FASTA database keyed by accession (sequences upper-cased, no gaps)."""

    def __init__(self, path: str):
        self.by_acc: dict[str, str] = {}
        # use pyteomics if available; else a minimal parser (no hard dependency
        # on a particular pyteomics version / index behaviour)
        try:
            from pyteomics import fasta
            with open(path, encoding="utf-8") as fh:
                for header, seq in fasta.read(fh, use_index=False):
                    self.by_acc[_parse_accession(header)] = seq.upper().replace(" ", "")
        except Exception:
            self._parse_plain(path)

    def _parse_plain(self, path: str):
        acc, buf = None, []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if acc is not None:
                        self.by_acc[acc] = "".join(buf).upper()
                    acc = _parse_accession(line[1:])
                    buf = []
                else:
                    buf.append(line.strip())
        if acc is not None:
            self.by_acc[acc] = "".join(buf).upper()

    def get(self, accession: str) -> str | None:
        """Sequence for *accession*, trying the bare ID and common variants."""
        if accession in self.by_acc:
            return self.by_acc[accession]
        # MaxQuant IDs can carry isoform suffixes (e.g. P06733-2) or prefixes
        base = accession.split("-")[0]
        if base in self.by_acc:
            return self.by_acc[base]
        return None

    def __len__(self):
        return len(self.by_acc)


def locate_peptide(peptide: str, protein_seq: str) -> list[tuple[int, int]]:
    """All 0-based ``[start, end)`` spans where *peptide* occurs in *protein_seq*.

    Leucine/isoleucine are NOT treated as equivalent here (MaxQuant reports the
    exact identified sequence). Multiple spans mean the peptide repeats in the
    protein (positional ambiguity).
    """
    peptide = peptide.upper()
    spans = []
    start = 0
    while True:
        i = protein_seq.find(peptide, start)
        if i < 0:
            break
        spans.append((i, i + len(peptide)))
        start = i + 1
    return spans


@dataclass
class PeptideMapping:
    accession: str
    sequence: str                       # the protein sequence drawn
    located: list = field(default_factory=list)    # dicts: peptide, start, end, unique, ambiguous
    not_located: list = field(default_factory=list)  # peptide sequences not found
    coverage_pct: float = 0.0
    residue_class: list = field(default_factory=list)   # per-residue: ""/unique/razor/shared
    residue_peptides: list = field(default_factory=list)  # per-residue: list of covering peptides


# class priority for colouring a residue covered by several peptides:
# unique evidence beats razor beats shared.
_CLASS_RANK = {"unique": 3, "razor": 2, "shared": 1, "": 0}


def map_peptides(accession: str, protein_seq: str, peptides: list[dict]) -> PeptideMapping:
    """Map a protein group's *peptides* onto *protein_seq*.

    *peptides* is a list of dicts with at least ``peptide`` (plain sequence) and
    ``unique`` (``unique`` / ``razor`` / ``shared``). Returns a
    :class:`PeptideMapping` with per-occurrence spans, the peptides that could
    not be located, and the residue coverage.
    """
    covered = [False] * len(protein_seq)
    res_class = [""] * len(protein_seq)         # per-residue best class
    res_peps = [[] for _ in range(len(protein_seq))]   # per-residue covering peptides
    located, not_located = [], []
    for p in peptides:
        seq = p["peptide"]
        cls = p.get("unique", "")
        spans = locate_peptide(seq, protein_seq)
        if not spans:
            not_located.append(seq)
            continue
        ambiguous = len(spans) > 1
        for (s, e) in spans:
            located.append({
                "peptide": seq, "start": s, "end": e,
                "unique": cls, "ambiguous": ambiguous,
            })
            for k in range(s, e):
                covered[k] = True
                if _CLASS_RANK.get(cls, 0) > _CLASS_RANK.get(res_class[k], 0):
                    res_class[k] = cls          # unique > razor > shared
                if seq not in res_peps[k]:
                    res_peps[k].append(seq)
    cov = 100.0 * sum(covered) / len(protein_seq) if protein_seq else 0.0
    return PeptideMapping(accession=accession, sequence=protein_seq,
                          located=located, not_located=not_located, coverage_pct=cov,
                          residue_class=res_class, residue_peptides=res_peps)