"""
ptero.ids._mods -- rewrite MaxQuant-style modification labels in a ProForma
string to UNIMOD mass *offsets*, so the result annotates **offline**.

Why this exists
---------------
psm_utils reads MaxQuant ``Modified sequence`` values and emits ProForma whose
modifications keep MaxQuant's human-readable names, e.g.

    AAALATVNAWAEQTGM[Oxidation (M)]K/2
    [Acetyl (Protein N-term)]-AAAAAAAAAAGAAGGR/2

Those bracketed names are **not** resolvable mass deltas: spectrum_utils'
``annotate_proforma`` / ``get_theoretical_fragments`` must look the name up in
the UNIMOD database, which requires a network round-trip (and fails outright in
an offline environment). Rewriting each modification to a numeric mass offset --

    AAALATVNAWAEQTGM[+15.994915]K/2
    [+42.010565]-AAAAAAAAAAGAAGGR/2

makes the proteoform fully self-contained: spectrum_utils computes fragment
masses directly from the offset, with no lookup. (Verified: the N-terminal
offset shifts the entire b-ion ladder by the modification mass, exactly as a
real N-terminal acetylation does.)

Fixed modifications
-------------------
A search engine does **not** write *fixed* (static) modifications into the
sequence -- they are implied for every applicable residue and are already folded
into the fragment masses the engine reports. So a fixed Carbamidomethyl (C)
appears nowhere in the ProForma, yet every cysteine fragment mass includes
+57.021464. If your search used fixed mods, pass them in ``fixed_mods`` so they
are *added* to the rewritten string; otherwise C-containing fragments will be
mis-massed by the omitted delta.

IMPORTANT: ``fixed_mods`` must match the search that produced the PSMs. For the
HeLa run this module was first exercised on, the MaxQuant ``parameters.txt``
listed **no** fixed modifications (no cysteine alkylation), so the default here
is *no* fixed mods. A typical alkylated tryptic search would pass
``fixed_mods={"C": 57.021464}``.

Unknown modifications
--------------------
A modification label that is neither a known variable mod nor matched by the
caller's ``extra_var_mods`` raises :class:`UnknownModificationError`. Silently
passing an unrecognised name through would leave a string spectrum_utils can
only resolve online (defeating the purpose) or, worse, drop the mass entirely --
a wrong fragment mass is more dangerous than a loud failure.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional

# UNIMOD monoisotopic mass deltas (Da). Sources cross-checked against UNIMOD,
# Byonic, and MS-GF+ modification tables.
OXIDATION = 15.994915
ACETYL = 42.010565
CARBAMIDOMETHYL = 57.021464

# Map MaxQuant variable-modification *labels* (as psm_utils emits them, i.e. the
# text inside the ProForma brackets) to their mass delta. Extend via the
# ``extra_var_mods`` argument rather than editing this in place.
_DEFAULT_VAR_MODS: dict[str, float] = {
    "Oxidation (M)": OXIDATION,
    "Acetyl (Protein N-term)": ACETYL,
    # common extras seen in MaxQuant output (harmless to recognise even if a
    # given run has none of them):
    "Acetyl (N-term)": ACETYL,
    "Carbamidomethyl (C)": CARBAMIDOMETHYL,   # if it ever appears as *variable*
}

# Map a MaxQuant modification *name* (the part before the parenthesis) to its
# mass delta, for resolving entries in summary.txt's Fixed/Variable columns.
# The affected residue is taken from the parenthetical, e.g. "Carbamidomethyl
# (C)" -> name "Carbamidomethyl", residue "C". Extend as needed.
_MOD_NAME_MASS: dict[str, float] = {
    "Carbamidomethyl": CARBAMIDOMETHYL,
    "Oxidation": OXIDATION,
    "Acetyl": ACETYL,
}


class UnknownModificationError(ValueError):
    """A ProForma modification label has no known mass mapping."""


def _fmt_offset(mass: float) -> str:
    """Format a mass delta as a ProForma offset token, e.g. ``[+15.994915]``."""
    return f"[{mass:+.6f}]".replace("+-", "-")


def maxquant_to_mass_offset(
    proforma: str,
    *,
    fixed_mods: Optional[Mapping[str, float]] = None,
    extra_var_mods: Optional[Mapping[str, float]] = None,
) -> str:
    """Rewrite a MaxQuant-style ProForma to one using numeric mass offsets.

    Parameters
    ----------
    proforma : str
        A ProForma string as produced by psm_utils from MaxQuant output, e.g.
        ``"AAALATVNAWAEQTGM[Oxidation (M)]K/2"`` or
        ``"[Acetyl (Protein N-term)]-AAAAAAAAAAGAAGGR/2"``. A trailing
        ``/<charge>`` is preserved.
    fixed_mods : mapping {residue -> mass delta}, optional
        Fixed (static) modifications used by the search. Each occurrence of the
        residue gets the offset inserted after it. ``None`` (default) adds
        nothing -- correct for a search with no fixed modifications. Example for
        an alkylated search: ``{"C": 57.021464}``.
    extra_var_mods : mapping {label -> mass delta}, optional
        Additional variable-modification labels to recognise, merged over the
        built-in table (use this for mods beyond oxidation/acetyl rather than
        editing the module).

    Returns
    -------
    str
        The rewritten ProForma, annotatable offline.

    Raises
    ------
    UnknownModificationError
        If a bracketed modification label is not in the known/var-mod tables.
    """
    var_mods = dict(_DEFAULT_VAR_MODS)
    if extra_var_mods:
        var_mods.update(extra_var_mods)

    # split off a trailing charge (/2) so it is not disturbed
    charge = ""
    m = re.search(r"/(\d+)\s*$", proforma)
    if m:
        charge = proforma[m.start():]
        proforma = proforma[: m.start()]

    # 1) replace every [label] with its mass offset (skip tokens that are
    #    already numeric offsets, e.g. "+15.994915").
    def _sub(match: re.Match) -> str:
        label = match.group(1)
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", label):    # already an offset
            return match.group(0)
        if label not in var_mods:
            raise UnknownModificationError(
                f"unknown modification label [{label}]. Add it via "
                f"extra_var_mods={{'{label}': <mass>}} or fix the input. "
                f"Known: {sorted(var_mods)}"
            )
        return _fmt_offset(var_mods[label])

    out = re.sub(r"\[([^\]]*)\]", _sub, proforma)

    # 2) add fixed modifications: insert the offset after each occurrence of the
    #    residue. Done on the bare sequence portion; an N-terminal "[mod]-" prefix
    #    is left untouched because we only insert after residue letters.
    if fixed_mods:
        for residue, mass in fixed_mods.items():
            if len(residue) != 1 or not residue.isalpha():
                raise ValueError(f"fixed_mods residue must be a single letter, got {residue!r}")
            token = _fmt_offset(mass)
            # insert after each residue letter that is NOT already followed by '['
            out = re.sub(rf"({re.escape(residue)})(?!\[)", rf"\1{token}", out)

    return out + charge


def convert_psm_modifications(
    psms,
    *,
    fixed_mods: Optional[Mapping[str, float]] = None,
    extra_var_mods: Optional[Mapping[str, float]] = None,
    summary_path=None,
) -> int:
    """Rewrite every PSM's peptidoform in place to use mass-offset ProForma.

    Applies :func:`maxquant_to_mass_offset` to ``psm.peptidoform`` for each PSM
    in ``psms`` (a psm_utils ``PSMList`` or any iterable of PSMs), so the whole
    list becomes annotatable offline. Returns the number of PSMs whose ProForma
    string actually changed.

    Fixed modifications
    -------------------
    The search's fixed mods must be added because they are not written into the
    sequence (see module docstring). Supply them one of two ways:

    * ``summary_path`` -- path to the MaxQuant ``summary.txt``; the fixed mods are
      read from it automatically (recommended: it cannot drift from the search).
    * ``fixed_mods``   -- an explicit ``{residue: mass}`` mapping.

    If both are given, ``fixed_mods`` overrides the file. If neither is given, no
    fixed mods are added (correct only for a search that used none).

    The peptidoform is reconstructed via ``psm_utils.peptidoform.Peptidoform``;
    the precursor charge carried in the original proforma's ``/<charge>`` suffix
    is preserved by the rewrite.
    """
    from psm_utils.peptidoform import Peptidoform

    if fixed_mods is None and summary_path is not None:
        fixed_mods = read_search_modifications(summary_path)["fixed"]

    n = 0
    for psm in psms:
        if psm.peptidoform is None:
            continue
        old = psm.peptidoform.proforma
        new = maxquant_to_mass_offset(
            old, fixed_mods=fixed_mods, extra_var_mods=extra_var_mods)
        if new != old:
            psm.peptidoform = Peptidoform(new)
            n += 1
    return n


# ── reading the search's modifications from MaxQuant summary.txt ─────────────

def _parse_mod_label(label: str):
    """Split a MaxQuant modification label into ``(name, residue)``.

    ``"Carbamidomethyl (C)"`` -> ``("Carbamidomethyl", "C")``;
    ``"Acetyl (Protein N-term)"`` -> ``("Acetyl", "Protein N-term")``. Returns
    ``(label, None)`` if there is no parenthetical.
    """
    m = re.match(r"\s*(.+?)\s*\(([^)]*)\)\s*$", label)
    if not m:
        return label.strip(), None
    return m.group(1).strip(), m.group(2).strip()


def read_search_modifications(summary_path) -> dict:
    """Read the fixed/variable modifications a MaxQuant search used.

    Parses ``summary.txt`` (the file that records the actual search modifications
    -- note that ``parameters.txt`` does NOT list fixed mods). Returns::

        {
          "fixed":    {residue_letter: mass_delta, ...},   # for the converter
          "variable": [label, ...],                        # informational
          "fixed_labels":    [label, ...],
          "variable_labels": [label, ...],
        }

    Only fixed modifications whose target is a single residue letter are
    currently supported. Fixed modifications with terminal or other non-residue
    targets raise :class:`ValueError` rather than being silently omitted, because
    omitting a fixed modification would yield incorrect fragment masses.

    A fixed-modification *name* not present in the known mass table raises
    :class:`UnknownModificationError`.

    Parameters
    ----------
    summary_path : str | Path
        Path to MaxQuant ``summary.txt``.
    """
    import csv
    from pathlib import Path

    summary_path = Path(summary_path)
    with open(summary_path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        if "Fixed modifications" not in fields and "Variable modifications" not in fields:
            raise KeyError(
                f"{summary_path.name} has no 'Fixed modifications' / 'Variable "
                f"modifications' column. Is this a MaxQuant summary.txt? "
                f"Columns: {fields}"
            )
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]

    def _collect(colname: str) -> list[str]:
        labels: list[str] = []
        for r in rows:
            raw = (r.get(colname) or "").strip()
            if not raw:
                continue
            for part in raw.split(";"):
                part = part.strip()
                if part and part not in labels:
                    labels.append(part)
        return labels

    fixed_labels = _collect("Fixed modifications")
    variable_labels = _collect("Variable modifications")

    fixed: dict[str, float] = {}
    for label in fixed_labels:
        name, residue = _parse_mod_label(label)
        if name not in _MOD_NAME_MASS:
            raise UnknownModificationError(
                f"fixed modification {label!r} (name {name!r}) has no known mass. "
                f"Add it to _MOD_NAME_MASS. Known: {sorted(_MOD_NAME_MASS)}"
            )
        if residue and len(residue) == 1 and residue.isalpha():
            fixed[residue] = _MOD_NAME_MASS[name]
        else:
            raise ValueError(
                f"fixed modification {label!r} has unsupported target {residue!r}; "
                "only single-residue fixed modifications are currently supported"
            )
    return {
        "fixed": fixed,
        "variable": variable_labels,
        "fixed_labels": fixed_labels,
        "variable_labels": variable_labels,
    }