"""
ptero.quant._contaminants -- annotate (not silently delete) likely contaminants.

A faithful, configurable port of the reference R ``annotate_contaminants``. Two
kinds of contaminant are tagged:

* **cRAP / database contaminants** -- MaxQuant's ``Potential contaminant`` flag
  (the cRAP FASTA), or a ``CON__`` / ``cRAP`` accession prefix. Tagged ``"cRAP"``.
* **tissue / handling contaminants** -- gene-symbol patterns for skeletal
  **muscle**, **keratin**, and **blood**. In a brain study these are real
  contamination (dissection muscle, skin/hair keratin, residual blood), which the
  cRAP list does *not* catch; flagging them keeps them out of biomarker calls
  while leaving them in the table (some keratins can be genuine in epithelium).

The philosophy matches the rest of PTERO: annotate and let the caller decide,
rather than drop. The default pattern lists reproduce the reference analysis; a
caller can override any of them (this is a public tool -- nothing brain-specific
is hardwired beyond the documented defaults).

cRAP is applied **last** so a contaminant-DB hit overrides a coincidental gene
pattern (matching the reference order: muscle -> keratin -> blood -> cRAP).
"""
from __future__ import annotations

import re

import pandas as pd

# default gene-symbol patterns (mouse/human symbols; matched case-insensitively,
# anchored exactly as the reference lists). Each entry is a standalone regex.
MUSCLE = (
    r"^Myh\d", r"^Myl\d", r"^Mybpc", r"^Tnnt", r"^Tnni", r"^Tnnc", r"^Ttn$",
    r"^Obscn$", r"^Actn[23]$", r"^Acta1$", r"^Casq[12]$", r"^Srl$", r"^Myoz",
    r"^Myom", r"^Sypl2$", r"^Neb$", r"^Ryr1$", r"^Atp2a1$", r"^Pgam2$", r"^Ckm$",
    r"^Eln$", r"^Des$", r"^Tcap$", r"^Myh1\d?$", r"^Myl11$", r"^Trdn$",
    r"^Smyd1$", r"^Ldb3$", r"^Csrp3$",
)
KERATIN = (r"^Krt\d", r"^Krtap", r"^Krt[0-9]")
BLOOD = (r"^Hba", r"^Hbb", r"^Hbq", r"^Alb$")

# accession substrings that mark a database contaminant when no flag is present
_CRAP_ACCESSION = re.compile(r"CON__|cRAP", re.IGNORECASE)


def _combined(patterns) -> str:
    return "(?:" + "|".join(patterns) + ")"


def annotate_contaminants(
    proteins: pd.DataFrame,
    *,
    flag_muscle: bool = True,
    flag_keratin: bool = True,
    flag_blood: bool = True,
    muscle=MUSCLE,
    keratin=KERATIN,
    blood=BLOOD,
    gene_col: str = "gene",
) -> pd.DataFrame:
    """Tag contaminants on a ``proteins`` table (e.g. ``QuantData.proteins``).

    Parameters
    ----------
    proteins : DataFrame
        Indexed by protein-group id. Uses ``gene_col`` for the gene-pattern
        tests and, when present, ``is_potential_contaminant`` /
        ``protein_ids`` / ``majority_protein_ids`` for the cRAP test.
    flag_muscle, flag_keratin, flag_blood : bool
        Whether to apply each tissue/handling pattern set. Default ``True``.
    muscle, keratin, blood : iterable[str]
        Regex pattern lists (override to customise). Defaults reproduce the
        reference analysis.
    gene_col : str
        Column holding gene symbols (default ``"gene"``).

    Returns
    -------
    DataFrame indexed like ``proteins`` with two columns:
        ``is_crap`` (bool) and ``contaminant_class`` (one of ``"ok"``,
        ``"muscle"``, ``"keratin"``, ``"blood"``, ``"cRAP"``). cRAP wins over a
        gene pattern when both match.
    """
    idx = proteins.index
    gene = proteins[gene_col] if gene_col in proteins.columns else pd.Series(index=idx, dtype=object)
    gene = gene.astype(object)

    def hit(patterns) -> pd.Series:
        return gene.str.contains(_combined(patterns), case=False, regex=True, na=False)

    cls = pd.Series("ok", index=idx, dtype=object)
    if flag_muscle:
        cls[hit(muscle)] = "muscle"
    if flag_keratin:
        cls[hit(keratin)] = "keratin"
    if flag_blood:
        cls[hit(blood)] = "blood"

    # cRAP / database contaminant: flag column first, then accession prefix
    is_crap = pd.Series(False, index=idx)
    if "is_potential_contaminant" in proteins.columns:
        is_crap = is_crap | proteins["is_potential_contaminant"].astype(bool)
    for col in ("protein_ids", "majority_protein_ids"):
        if col in proteins.columns:
            acc = proteins[col].astype(object)
            is_crap = is_crap | acc.str.contains(_CRAP_ACCESSION, na=False)
    cls[is_crap] = "cRAP"

    return pd.DataFrame({"is_crap": is_crap, "contaminant_class": cls}, index=idx)