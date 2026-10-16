"""
ptero.quant._quant -- load MaxQuant proteinGroups.txt into a multi-sample
protein x sample quantification table (LFQ + raw intensity), ready for display
and (next arc) missing-value analysis.

Why a separate subpackage
-------------------------
`ptero.ids` links one PSM to its raw MS2 spectrum (sequence / spectrum grain).
Quantification lives at a different grain -- protein group x sample -- and grows
its own arc (quant display -> missingness -> later, statistics), so it gets its
own home rather than crowding `ptero.ids`. It reuses the same conventions:
BOM-safe reads (MaxQuant on Windows writes a UTF-8 BOM; we read 'utf-8-sig',
matching ``ptero.ids._mods.read_search_modifications``), pathlib, and loud
validation over silent guessing.

What the loader reads -- and what it deliberately does not
---------------------------------------------------------
Everything the text-side arc needs sits in this ONE ~19 MB file, per sample:
``LFQ intensity <sample>``, ``Intensity <sample>``, and (for the missingness arc)
``Identification type <sample>`` / ``MS/MS count <sample>``. The heavy
``evidence.txt`` (~450 MB) and ``msms.txt`` (~1 GB) are only for the per-sample
spectrum link (``ptero.ids``), so they are not touched here.

Built to be data-agnostic for a PUBLIC tool: it makes NO assumption about the
experiment's group naming. Samples come straight from the ``LFQ intensity
<sample>`` column headers; grouping is the caller's to supply -- either an
explicit ``{sample: group}`` mapping, or via :func:`suggest_groups_by_token`
(a convenience you inspect and confirm, never an automatic guess).

Key MaxQuant facts encoded here
------------------------------
* **0 means "not quantified", not a measured zero.** MaxQuant writes ``0`` (not
  blank) into an ``LFQ intensity`` / ``Intensity`` cell when the protein was not
  quantified in that sample. Both matrices map ``0 -> NaN`` so "missing" is a
  real ``NaN`` the missingness arc can reason about.
* **``Intensity`` > 0 while ``LFQ intensity`` == 0 is a real, distinct state**
  (detected but not LFQ-quantified -- did not meet the MaxLFQ ratio-count). It is
  kept separate, never collapsed into "present": that distinction is exactly what
  the three-tier missingness analysis (next arc) is built on.
* **No ``Gene names`` column in this output** -- gene symbols live inside
  ``Fasta headers`` as ``GN=...``; we parse the first one for a display label.
* **Index is MaxQuant's protein-group ``id``** (stable, and the join key to
  ``evidence`` / ``msms`` ``Protein group IDs`` when the spectrum link arrives).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

ENC = "utf-8-sig"   # decodes and drops a leading UTF-8 BOM (MaxQuant on Windows)
SEP = "\t"

_LFQ_PREFIX = "LFQ intensity "
_INT_PREFIX = "Intensity "
_IDTYPE_PREFIX = "Identification type "
_IDTYPE_VALUES = {"By MS/MS", "By matching"}   # everything else -> None (not identified)
_GENE_RE = re.compile(r"\bGN=(\S+)")

# proteinGroups columns carried as protein metadata, when present.
_META_COLS = (
    "Protein IDs", "Majority protein IDs", "Fasta headers", "Species",
    "Q-value", "Score", "Razor + unique peptides",
)
_FLAG_COLS = ("Reverse", "Potential contaminant", "Only identified by site")


@dataclass
class QuantData:
    """Multi-sample protein quantification read from one proteinGroups.txt.

    Matrices are indexed by MaxQuant protein-group ``id``; columns are samples.
    A missing value is ``NaN`` (MaxQuant's ``0`` is converted on load).

    Attributes
    ----------
    lfq, intensity : DataFrame
        Protein x sample. ``lfq`` is the active quantification matrix:
        MaxLFQ-normalised values when quantity='lfq', or raw summed
        intensities when quantity='intensity'. ``intensity`` contains raw
        summed intensities; with the Intensity fallback, both matrices
        contain the same values. ``NaN`` means not quantified in that sample.
    id_type : DataFrame
        protein x sample MaxQuant identification provenance: ``"By MS/MS"`` (a
        real MS/MS ID in that sample), ``"By matching"`` (match-between-runs
        transfer), or ``None`` (not identified there). The signal the missingness
        arc is built on; see :mod:`ptero.quant._missing`.
    proteins : DataFrame
        Per-protein metadata indexed by ``id``: ``protein_ids``,
        ``majority_protein_ids``, ``label`` (gene if known else leading
        accession), ``gene``, ``species``, ``q_value``, ``score``,
        ``razor_unique_peptides`` (those present in the file).
    samples : list[str]
        Sample names, in proteinGroups column order.
    groups : dict[str, str]
        ``sample -> group`` (empty if ungrouped).
    source : Path
        The proteinGroups.txt that was read.
    mq_version : str | None
        MaxQuant version, if a parameters.txt was supplied.
    n_protein_groups_raw : int
        Row count before filtering.
    n_filtered : dict[str, int]
        Per-reason flag counts plus ``rows_dropped`` (the union actually removed).
    """
    lfq: pd.DataFrame
    intensity: pd.DataFrame
    id_type: pd.DataFrame
    proteins: pd.DataFrame
    samples: list[str]
    groups: dict[str, str]
    source: Path
    mq_version: Optional[str] = None
    n_protein_groups_raw: int = 0
    n_filtered: dict[str, int] = field(default_factory=dict)

    @property
    def n_proteins(self) -> int:
        return int(self.lfq.shape[0])

    @property
    def grouped(self) -> bool:
        return bool(self.groups)

    def group_members(self) -> dict[str, list[str]]:
        """``group -> [samples]`` in sample order (empty if ungrouped)."""
        out: dict[str, list[str]] = {}
        for s in self.samples:
            g = self.groups.get(s)
            if g is not None:
                out.setdefault(g, []).append(s)
        return out

    def to_long(self) -> pd.DataFrame:
        """Tidy ``(id, sample, group, lfq, intensity, id_type, label, gene)``."""
        lfq_long = (self.lfq.reset_index()
                    .melt(id_vars="id", var_name="sample", value_name="lfq"))
        int_long = (self.intensity.reset_index()
                    .melt(id_vars="id", var_name="sample", value_name="intensity"))
        long = lfq_long.merge(int_long, on=["id", "sample"], how="outer")
        it_long = (self.id_type.reset_index()
                   .melt(id_vars="id", var_name="sample", value_name="id_type"))
        long = long.merge(it_long, on=["id", "sample"], how="left")
        long["group"] = long["sample"].map(self.groups) if self.groups else ""
        meta = [c for c in ("label", "gene") if c in self.proteins.columns]
        if meta:
            long = long.merge(self.proteins[meta], left_on="id",
                              right_index=True, how="left")
        return long


def _strip_prefix(col: str, prefix: str) -> str:
    return col[len(prefix):] if col.lower().startswith(prefix.lower()) else col


def _norm_idtype(x):
    """Normalise a raw 'Identification type' cell to {'By MS/MS','By matching',None}.

    Empty / blank / 'None' / anything unrecognised -> ``None`` (not identified in
    that sample). Robust to however MaxQuant renders the not-identified state.
    """
    if isinstance(x, str):
        x = x.strip()
        if x in _IDTYPE_VALUES:
            return x
    return None


def _parse_gene(fasta_headers) -> Optional[str]:
    if not isinstance(fasta_headers, str) or not fasta_headers:
        return None
    m = _GENE_RE.search(fasta_headers)
    return m.group(1) if m else None


def _first_accession(majority_ids) -> Optional[str]:
    if not isinstance(majority_ids, str) or not majority_ids:
        return None
    first = majority_ids.split(";")[0].strip()
    if "|" in first:                       # db|ACCESSION|NAME -> ACCESSION
        parts = first.split("|")
        return parts[1] if len(parts) >= 2 and parts[1] else first
    return first or None


def _label_for(gene, majority_ids, id_) -> str:
    """Display label: gene if known, else leading accession, else the id.

    Only real non-empty strings count -- NaN is truthy in Python, so a naive
    ``gene or accession`` would let a missing-gene NaN through as the label.
    """
    if isinstance(gene, str) and gene:
        return gene
    acc = _first_accession(majority_ids)
    return acc if acc else str(id_)


def _read_header(path: Path) -> list[str]:
    return pd.read_csv(path, sep=SEP, nrows=0, encoding=ENC).columns.tolist()


def read_sample_names(protein_groups_path, *, quantity: str = "lfq") -> list[str]:
    """Sample names from a proteinGroups.txt header alone (cheap; no full read).

    Reads only the column row and returns the ``<sample>`` part of every
    ``LFQ intensity <sample>`` column (``quantity="lfq"``, default) or every
    ``Intensity <sample>`` column (``quantity="intensity"`` -- use this to list
    samples for a search run without LFQ). Use it to build / confirm a group
    mapping before the full :func:`load_quant`.
    """
    header = _read_header(Path(protein_groups_path))

    if quantity == "lfq":
        prefix = _LFQ_PREFIX
    elif quantity == "intensity":
        prefix = _INT_PREFIX
    else:
        raise ValueError(
            f"quantity must be 'lfq' or 'intensity', got {quantity!r}."
        )

    cols = [
        c for c in header
        if c.lower().startswith(prefix.lower())
    ]
    return [_strip_prefix(c, prefix) for c in cols]


def suggest_groups_by_token(samples, *, sep: str = "_", index: int = 1) -> dict[str, str]:
    """Suggest ``{sample: group}`` by taking one delimiter-separated token.

    A GENERIC convenience for any naming scheme -- it is *not* applied
    automatically by :func:`load_quant`. Inspect the result, then pass it as
    ``groups=``. For ``'OE_PS_191'`` with ``sep='_', index=1`` you get ``'PS'``;
    negative indices count from the end (``index=-2`` -> ``'PS'`` too). Raises if
    any sample lacks a token at that position, so a bad scheme fails loudly
    rather than mislabelling.
    """
    out: dict[str, str] = {}
    for s in samples:
        parts = s.split(sep)
        n = len(parts)
        i = index if index >= 0 else n + index
        if i < 0 or i >= n:
            raise ValueError(
                f"sample {s!r} has no token at index {index} split on {sep!r} "
                f"(tokens: {parts}). Pass an explicit groups={{sample: group}} dict."
            )
        out[s] = parts[i]
    return out


def _read_mq_version(parameters_path) -> Optional[str]:
    p = Path(parameters_path)
    for line in p.read_text(encoding=ENC, errors="replace").splitlines():
        if line.lower().startswith("version"):
            parts = line.split(SEP)
            if len(parts) >= 2 and parts[1].strip():
                return parts[1].strip()
    return None


def load_quant(
    protein_groups_path,
    *,
    groups: Optional[Mapping[str, str]] = None,
    drop_decoys_contaminants: bool = True,
    quantity: str = "lfq",
    parameters_path=None,
) -> QuantData:
    """Load MaxQuant ``proteinGroups.txt`` into a :class:`QuantData`.

    Parameters
    ----------
    protein_groups_path : str | Path
        Path to ``proteinGroups.txt`` (read BOM-safe).
    groups : mapping {sample -> group}, optional
        Group label for every sample. ``None`` (default) leaves the data
        ungrouped. Must cover exactly the samples found in the file -- an
        unmapped sample, or a stray sample not in the file, raises, so a mismatch
        never silently corrupts downstream comparisons. Build one with
        :func:`suggest_groups_by_token` or by hand.
    drop_decoys_contaminants : bool
        Drop the standard MaxQuant junk rows -- those flagged ``Reverse`` (decoy),
        ``Potential contaminant``, or ``Only identified by site`` -- before
        building the table. Per-reason counts are recorded in ``n_filtered``
        regardless. Default ``True``.
    quantity : {"lfq", "intensity"}
        Which per-sample matrix to use as the quantification basis. ``"lfq"``
        (default) uses ``LFQ intensity`` columns (MaxLFQ-normalised). ``"intensity"``
        uses raw ``Intensity`` columns -- the fallback when a search was run
        **without LFQ** (then ``lfq`` holds raw intensity, and the
        LFQ-vs-Intensity cell-state distinction in :mod:`ptero.quant._missing`
        collapses, since there is only one matrix). For missingness classification
        either works; LFQ is preferable for cross-sample comparability when present.
    parameters_path : str | Path, optional
        A ``parameters.txt`` to record the MaxQuant version as provenance.

    Returns
    -------
    QuantData
    """
    path = Path(protein_groups_path)
    header = _read_header(path)

    lfq_cols = [
        c for c in header
        if c.lower().startswith(_LFQ_PREFIX.lower())
    ]
    int_cols = [
        c for c in header
        if c.lower().startswith(_INT_PREFIX.lower())
        and not c.lower().startswith(_LFQ_PREFIX.lower())
    ]
    idtype_cols = [
        c for c in header
        if c.lower().startswith(_IDTYPE_PREFIX.lower())
    ]

    if "id" not in header:
        raise ValueError(
            f"{path.name} has no 'id' column; cannot index protein groups."
        )

    lfq_samples = [_strip_prefix(c, _LFQ_PREFIX) for c in lfq_cols]
    int_samples = [_strip_prefix(c, _INT_PREFIX) for c in int_cols]

    if quantity == "lfq":
        if not lfq_cols:
            extra = (
                " Raw 'Intensity' columns are present -- pass "
                "quantity='intensity' to use them."
            ) if int_cols else ""
            raise ValueError(
                f"{path.name} has no 'LFQ intensity <sample>' columns -- was LFQ "
                f"enabled in this MaxQuant search?{extra} "
                f"Columns: {header[:20]}..."
            )
        main_cols, main_samples = lfq_cols, lfq_samples

    elif quantity == "intensity":
        if not int_cols:
            raise ValueError(
                f"{path.name} has no 'Intensity <sample>' columns. "
                f"Columns: {header[:20]}..."
            )
        main_cols, main_samples = int_cols, int_samples

    else:
        raise ValueError(
            f"quantity must be 'lfq' or 'intensity', got {quantity!r}."
        )

    samples = main_samples

    if (
        quantity == "lfq"
        and int_cols
        and set(int_samples) != set(lfq_samples)
    ):
        only_lfq = sorted(set(lfq_samples) - set(int_samples))
        only_int = sorted(set(int_samples) - set(lfq_samples))
        warnings.warn(
            "LFQ and raw-Intensity sample sets differ "
            f"(only in LFQ: {only_lfq}; only in Intensity: {only_int}). "
            "Using the LFQ sample set.",
            stacklevel=2,
        )

    meta_present = [c for c in _META_COLS if c in header]
    flag_present = [c for c in _FLAG_COLS if c in header]

    usecols = (
        ["id"]
        + meta_present
        + flag_present
        + lfq_cols
        + int_cols
        + idtype_cols
    )

    seen: set[str] = set()
    usecols = [
        c for c in usecols
        if not (c in seen or seen.add(c))
    ]

    df = pd.read_csv(
        path,
        sep=SEP,
        usecols=usecols,
        encoding=ENC,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    n_raw = int(df.shape[0])

    def _flag(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.strip() == "+"

    rev = _flag("Reverse")
    con = _flag("Potential contaminant")
    site = _flag("Only identified by site")

    n_filtered = {
        "reverse": int(rev.sum()),
        "potential_contaminant": int(con.sum()),
        "only_identified_by_site": int(site.sum()),
    }

    if drop_decoys_contaminants:
        keep = ~(rev | con | site)
        n_filtered["rows_dropped"] = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
    else:
        n_filtered["rows_dropped"] = 0

    # Drop rows whose 'id' is blank or non-numeric (for example, a truncated
    # final line), but never silently coerce a numeric non-integer identifier.
    numeric_ids = pd.to_numeric(df["id"], errors="coerce")
    id_ok = numeric_ids.notna()

    if not id_ok.all():
        warnings.warn(
            f"dropped {int((~id_ok).sum())} row(s) with a blank/non-integer "
            f"'id' (likely a truncated line).",
            stacklevel=2,
        )
        df = df[id_ok].reset_index(drop=True)
        numeric_ids = numeric_ids[id_ok].reset_index(drop=True)

    noninteger = numeric_ids != np.floor(numeric_ids)
    if noninteger.any():
        bad = df.loc[noninteger, "id"].tolist()
        raise ValueError(
            "non-integer protein-group id(s) found: "
            f"{bad[:10]}"
        )

    ids = numeric_ids.astype("int64")

    if ids.duplicated().any():
        raise ValueError(
            "duplicate protein-group ids found; "
            "protein-group ids must be unique."
        )

    idx = ids.to_numpy()

    def _matrix(cols: list[str], names: list[str]) -> pd.DataFrame:
        m = df[cols].apply(pd.to_numeric, errors="coerce")

        if (m < 0).any().any():
            raise ValueError(
                "quantification values must be non-negative; "
                "negative intensity value(s) found."
            )

        m.columns = names
        m.index = idx
        m.index.name = "id"
        return m.replace(0, np.nan)

    lfq = _matrix(main_cols, main_samples)

    if int_cols:
        intensity = _matrix(
            int_cols,
            int_samples,
        ).reindex(columns=samples)
    else:
        intensity = pd.DataFrame(index=idx)
        intensity.index.name = "id"

    if idtype_cols:
        it = df[idtype_cols].apply(
            lambda c: c.map(_norm_idtype)
        )
        it.columns = [
            _strip_prefix(c, _IDTYPE_PREFIX)
            for c in idtype_cols
        ]
        it.index = idx
        it.index.name = "id"
        it = it.reindex(columns=samples)
        id_type = it.astype(object).where(
            pd.notna(it),
            None,
        )
        id_type.index.name = "id"
    else:
        id_type = pd.DataFrame(
            index=idx,
            columns=samples,
            dtype=object,
        )
        id_type = id_type.where(
            pd.notna(id_type),
            None,
        )
        id_type.index.name = "id"

    proteins = pd.DataFrame(index=idx)
    proteins.index.name = "id"

    if "Protein IDs" in df:
        proteins["protein_ids"] = df["Protein IDs"].to_numpy()

    maj = (
        df["Majority protein IDs"].to_numpy()
        if "Majority protein IDs" in df
        else np.array([None] * len(df), dtype=object)
    )

    if "Majority protein IDs" in df:
        proteins["majority_protein_ids"] = maj

    genes = (
        [_parse_gene(x) for x in df["Fasta headers"]]
        if "Fasta headers" in df
        else [None] * len(df)
    )

    proteins["gene"] = pd.Series(
        genes,
        index=idx,
        dtype=object,
    )

    proteins["label"] = [
        _label_for(g, m, i)
        for g, m, i in zip(genes, maj, idx)
    ]

    if "Species" in df:
        proteins["species"] = df["Species"].to_numpy()

    if "Q-value" in df:
        proteins["q_value"] = pd.to_numeric(
            df["Q-value"],
            errors="coerce",
        ).to_numpy()

    if "Score" in df:
        proteins["score"] = pd.to_numeric(
            df["Score"],
            errors="coerce",
        ).to_numpy()

    if "Razor + unique peptides" in df:
        proteins["razor_unique_peptides"] = pd.to_numeric(
            df["Razor + unique peptides"],
            errors="coerce",
        ).to_numpy()

    proteins["is_reverse"] = _flag("Reverse").to_numpy()
    proteins["is_potential_contaminant"] = _flag(
        "Potential contaminant"
    ).to_numpy()
    proteins["is_only_by_site"] = _flag(
        "Only identified by site"
    ).to_numpy()

    if groups is not None:
        groups = dict(groups)

        unknown = [
            s for s in groups
            if s not in samples
        ]
        unmapped = [
            s for s in samples
            if s not in groups
        ]

        if unknown:
            raise ValueError(
                f"groups references samples not in the data: {unknown}. "
                f"Data samples: {samples}"
            )

        if unmapped:
            raise ValueError(
                f"these samples have no group assigned: {unmapped}. "
                f"Map every sample, or pass groups=None for ungrouped."
            )

        groups = {
            s: groups[s]
            for s in samples
        }
    else:
        groups = {}

    mq_version = (
        _read_mq_version(parameters_path)
        if parameters_path
        else None
    )

    return QuantData(
        lfq=lfq,
        intensity=intensity,
        id_type=id_type,
        proteins=proteins,
        samples=list(samples),
        groups=groups,
        source=path,
        mq_version=mq_version,
        n_protein_groups_raw=n_raw,
        n_filtered=n_filtered,
    )

    def _flag(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.strip() == "+"

    rev = _flag("Reverse")
    con = _flag("Potential contaminant")
    site = _flag("Only identified by site")

    n_filtered = {
        "reverse": int(rev.sum()),
        "potential_contaminant": int(con.sum()),
        "only_identified_by_site": int(site.sum()),
    }

    if drop_decoys_contaminants:
        keep = ~(rev | con | site)
        n_filtered["rows_dropped"] = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
    else:
        n_filtered["rows_dropped"] = 0

    # guard: drop rows whose 'id' isn't a valid integer (e.g. a truncated final
    # line in the export). Warn rather than fail the whole load or drop silently.
    id_ok = pd.to_numeric(df["id"], errors="coerce").notna()
    if not id_ok.all():
        warnings.warn(
            f"dropped {int((~id_ok).sum())} row(s) with a blank/non-integer "
            f"'id' (likely a truncated line).",
            stacklevel=2,
        )
        df = df[id_ok].reset_index(drop=True)

    ids = pd.to_numeric(df["id"], errors="raise").astype("int64")

    if ids.duplicated().any():
        raise ValueError(
            "duplicate protein-group ids found; "
            "protein-group ids must be unique."
        )

    idx = ids.to_numpy()

    def _matrix(cols: list[str], names: list[str]) -> pd.DataFrame:
        m = df[cols].apply(pd.to_numeric, errors="coerce")

        if (m < 0).any().any():
            raise ValueError(
                "quantification values must be non-negative; "
                "negative intensity value(s) found."
            )

        m.columns = names
        m.index = idx
        m.index.name = "id"
        return m.replace(0, np.nan)

    lfq = _matrix(main_cols, main_samples)

    if int_cols:
        intensity = _matrix(
            int_cols,
            int_samples,
        ).reindex(columns=samples)
    else:
        intensity = pd.DataFrame(index=idx)
        intensity.index.name = "id"

    if idtype_cols:
        it = df[idtype_cols].apply(
            lambda c: c.map(_norm_idtype)
        )
        it.columns = [
            _strip_prefix(c, _IDTYPE_PREFIX)
            for c in idtype_cols
        ]
        it.index = idx
        it.index.name = "id"
        it = it.reindex(columns=samples)
        id_type = it.astype(object).where(
            pd.notna(it),
            None,
        )
        id_type.index.name = "id"
    else:
        id_type = pd.DataFrame(
            index=idx,
            columns=samples,
            dtype=object,
        )
        id_type = id_type.where(
            pd.notna(id_type),
            None,
        )
        id_type.index.name = "id"

    proteins = pd.DataFrame(index=idx)
    proteins.index.name = "id"

    if "Protein IDs" in df:
        proteins["protein_ids"] = df["Protein IDs"].to_numpy()

    maj = (
        df["Majority protein IDs"].to_numpy()
        if "Majority protein IDs" in df
        else np.array([None] * len(df), dtype=object)
    )

    if "Majority protein IDs" in df:
        proteins["majority_protein_ids"] = maj

    genes = (
        [_parse_gene(x) for x in df["Fasta headers"]]
        if "Fasta headers" in df
        else [None] * len(df)
    )

    proteins["gene"] = pd.Series(
        genes,
        index=idx,
        dtype=object,
    )

    proteins["label"] = [
        _label_for(g, m, i)
        for g, m, i in zip(genes, maj, idx)
    ]

    if "Species" in df:
        proteins["species"] = df["Species"].to_numpy()

    if "Q-value" in df:
        proteins["q_value"] = pd.to_numeric(
            df["Q-value"],
            errors="coerce",
        ).to_numpy()

    if "Score" in df:
        proteins["score"] = pd.to_numeric(
            df["Score"],
            errors="coerce",
        ).to_numpy()

    if "Razor + unique peptides" in df:
        proteins["razor_unique_peptides"] = pd.to_numeric(
            df["Razor + unique peptides"],
            errors="coerce",
        ).to_numpy()

    # carry the MaxQuant flags (recomputed on the kept rows) so downstream can
    # annotate / retain contaminants (e.g. a transgene spiked into the cRAP db)
    # without re-reading the file. All False when drop_decoys_contaminants=True.
    proteins["is_reverse"] = _flag("Reverse").to_numpy()
    proteins["is_potential_contaminant"] = _flag(
        "Potential contaminant"
    ).to_numpy()
    proteins["is_only_by_site"] = _flag(
        "Only identified by site"
    ).to_numpy()

    # groups: explicit, validated, loud
    if groups is not None:
        groups = dict(groups)

        unknown = [
            s for s in groups
            if s not in samples
        ]
        unmapped = [
            s for s in samples
            if s not in groups
        ]

        if unknown:
            raise ValueError(
                f"groups references samples not in the data: {unknown}. "
                f"Data samples: {samples}"
            )

        if unmapped:
            raise ValueError(
                f"these samples have no group assigned: {unmapped}. "
                f"Map every sample, or pass groups=None for ungrouped."
            )

        groups = {
            s: groups[s]
            for s in samples
        }
    else:
        groups = {}

    mq_version = (
        _read_mq_version(parameters_path)
        if parameters_path
        else None
    )

    return QuantData(
        lfq=lfq,
        intensity=intensity,
        id_type=id_type,
        proteins=proteins,
        samples=list(samples),
        groups=groups,
        source=path,
        mq_version=mq_version,
        n_protein_groups_raw=n_raw,
        n_filtered=n_filtered,
    )

    def _flag(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.strip() == "+"

    rev = _flag("Reverse")
    con = _flag("Potential contaminant")
    site = _flag("Only identified by site")

    n_filtered = {
        "reverse": int(rev.sum()),
        "potential_contaminant": int(con.sum()),
        "only_identified_by_site": int(site.sum()),
    }

    if drop_decoys_contaminants:
        keep = ~(rev | con | site)
        n_filtered["rows_dropped"] = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
    else:
        n_filtered["rows_dropped"] = 0

    # guard: drop rows whose 'id' isn't a valid integer (e.g. a truncated final
    # line in the export). Warn rather than fail the whole load or drop silently.
    id_ok = pd.to_numeric(df["id"], errors="coerce").notna()
    if not id_ok.all():
        warnings.warn(
            f"dropped {int((~id_ok).sum())} row(s) with a blank/non-integer "
            f"'id' (likely a truncated line).",
            stacklevel=2,
        )
        df = df[id_ok].reset_index(drop=True)

    ids = pd.to_numeric(df["id"], errors="raise").astype("int64")

    if ids.duplicated().any():
        warnings.warn(
            "duplicate protein-group ids found; matrices will have a "
            "non-unique index.",
            stacklevel=2,
        )

    idx = ids.to_numpy()

    def _matrix(cols: list[str], names: list[str]) -> pd.DataFrame:
        m = df[cols].apply(pd.to_numeric, errors="coerce")

        if (m < 0).any().any():
            raise ValueError(
                "quantification values must be non-negative; "
                "negative intensity value(s) found."
            )

        m.columns = names
        m.index = idx
        m.index.name = "id"
        return m.replace(0, np.nan)

    lfq = _matrix(main_cols, main_samples)

    if int_cols:
        intensity = _matrix(
            int_cols,
            int_samples,
        ).reindex(columns=samples)
    else:
        intensity = pd.DataFrame(index=idx)
        intensity.index.name = "id"

    if idtype_cols:
        it = df[idtype_cols].apply(
            lambda c: c.map(_norm_idtype)
        )
        it.columns = [
            _strip_prefix(c, _IDTYPE_PREFIX)
            for c in idtype_cols
        ]
        it.index = idx
        it.index.name = "id"
        it = it.reindex(columns=samples)
        id_type = it.astype(object).where(
            pd.notna(it),
            None,
        )
        id_type.index.name = "id"
    else:
        id_type = pd.DataFrame(
            index=idx,
            columns=samples,
            dtype=object,
        )
        id_type = id_type.where(
            pd.notna(id_type),
            None,
        )
        id_type.index.name = "id"

    proteins = pd.DataFrame(index=idx)
    proteins.index.name = "id"

    if "Protein IDs" in df:
        proteins["protein_ids"] = df["Protein IDs"].to_numpy()

    maj = (
        df["Majority protein IDs"].to_numpy()
        if "Majority protein IDs" in df
        else np.array([None] * len(df), dtype=object)
    )

    if "Majority protein IDs" in df:
        proteins["majority_protein_ids"] = maj

    genes = (
        [_parse_gene(x) for x in df["Fasta headers"]]
        if "Fasta headers" in df
        else [None] * len(df)
    )

    proteins["gene"] = pd.Series(
        genes,
        index=idx,
        dtype=object,
    )

    proteins["label"] = [
        _label_for(g, m, i)
        for g, m, i in zip(genes, maj, idx)
    ]

    if "Species" in df:
        proteins["species"] = df["Species"].to_numpy()

    if "Q-value" in df:
        proteins["q_value"] = pd.to_numeric(
            df["Q-value"],
            errors="coerce",
        ).to_numpy()

    if "Score" in df:
        proteins["score"] = pd.to_numeric(
            df["Score"],
            errors="coerce",
        ).to_numpy()

    if "Razor + unique peptides" in df:
        proteins["razor_unique_peptides"] = pd.to_numeric(
            df["Razor + unique peptides"],
            errors="coerce",
        ).to_numpy()

    # carry the MaxQuant flags (recomputed on the kept rows) so downstream can
    # annotate / retain contaminants (e.g. a transgene spiked into the cRAP db)
    # without re-reading the file. All False when drop_decoys_contaminants=True.
    proteins["is_reverse"] = _flag("Reverse").to_numpy()
    proteins["is_potential_contaminant"] = _flag(
        "Potential contaminant"
    ).to_numpy()
    proteins["is_only_by_site"] = _flag(
        "Only identified by site"
    ).to_numpy()

    # groups: explicit, validated, loud
    if groups is not None:
        groups = dict(groups)

        unknown = [
            s for s in groups
            if s not in samples
        ]
        unmapped = [
            s for s in samples
            if s not in groups
        ]

        if unknown:
            raise ValueError(
                f"groups references samples not in the data: {unknown}. "
                f"Data samples: {samples}"
            )

        if unmapped:
            raise ValueError(
                f"these samples have no group assigned: {unmapped}. "
                f"Map every sample, or pass groups=None for ungrouped."
            )

        groups = {
            s: groups[s]
            for s in samples
        }
    else:
        groups = {}

    mq_version = (
        _read_mq_version(parameters_path)
        if parameters_path
        else None
    )

    return QuantData(
        lfq=lfq,
        intensity=intensity,
        id_type=id_type,
        proteins=proteins,
        samples=list(samples),
        groups=groups,
        source=path,
        mq_version=mq_version,
        n_protein_groups_raw=n_raw,
        n_filtered=n_filtered,
    )

    def _flag(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.strip() == "+"

    rev = _flag("Reverse")
    con = _flag("Potential contaminant")
    site = _flag("Only identified by site")
    n_filtered = {
        "reverse": int(rev.sum()),
        "potential_contaminant": int(con.sum()),
        "only_identified_by_site": int(site.sum()),
    }
    if drop_decoys_contaminants:
        keep = ~(rev | con | site)
        n_filtered["rows_dropped"] = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
    else:
        n_filtered["rows_dropped"] = 0

    # guard: drop rows whose 'id' isn't a valid integer (e.g. a truncated final
    # line in the export). Warn rather than fail the whole load or drop silently.
    id_ok = pd.to_numeric(df["id"], errors="coerce").notna()
    if not id_ok.all():
        warnings.warn(f"dropped {int((~id_ok).sum())} row(s) with a blank/non-integer "
                      f"'id' (likely a truncated line).", stacklevel=2)
        df = df[id_ok].reset_index(drop=True)

    ids = pd.to_numeric(df["id"], errors="raise").astype("int64")
    if ids.duplicated().any():
        warnings.warn("duplicate protein-group ids found; matrices will have a "
                      "non-unique index.", stacklevel=2)
    idx = ids.to_numpy()

    def _matrix(cols: list[str], names: list[str]) -> pd.DataFrame:
        m = df[cols].apply(pd.to_numeric, errors="coerce")
        m.columns = names
        m.index = idx
        m.index.name = "id"
        return m.replace(0, np.nan)

    lfq = _matrix(main_cols, main_samples)
    if int_cols:
        intensity = _matrix(int_cols, int_samples).reindex(columns=samples)
    else:
        intensity = pd.DataFrame(index=idx)
        intensity.index.name = "id"

    if idtype_cols:
        it = df[idtype_cols].apply(lambda c: c.map(_norm_idtype))
        it.columns = [_strip_prefix(c, _IDTYPE_PREFIX) for c in idtype_cols]
        it.index = idx
        it.index.name = "id"
        it = it.reindex(columns=samples)
        id_type = it.astype(object).where(pd.notna(it), None)   # not-identified -> None
        id_type.index.name = "id"
    else:
        id_type = pd.DataFrame(index=idx, columns=samples, dtype=object)
        id_type = id_type.where(pd.notna(id_type), None)
        id_type.index.name = "id"

    proteins = pd.DataFrame(index=idx)
    proteins.index.name = "id"
    if "Protein IDs" in df:
        proteins["protein_ids"] = df["Protein IDs"].to_numpy()
    maj = df["Majority protein IDs"].to_numpy() if "Majority protein IDs" in df \
        else np.array([None] * len(df), dtype=object)
    if "Majority protein IDs" in df:
        proteins["majority_protein_ids"] = maj
    genes = [_parse_gene(x) for x in df["Fasta headers"]] if "Fasta headers" in df \
        else [None] * len(df)
    proteins["gene"] = pd.Series(genes, index=idx, dtype=object)   # keep None (not NaN)
    proteins["label"] = [_label_for(g, m, i) for g, m, i in zip(genes, maj, idx)]
    if "Species" in df:
        proteins["species"] = df["Species"].to_numpy()
    if "Q-value" in df:
        proteins["q_value"] = pd.to_numeric(df["Q-value"], errors="coerce").to_numpy()
    if "Score" in df:
        proteins["score"] = pd.to_numeric(df["Score"], errors="coerce").to_numpy()
    if "Razor + unique peptides" in df:
        proteins["razor_unique_peptides"] = pd.to_numeric(
            df["Razor + unique peptides"], errors="coerce").to_numpy()
    # carry the MaxQuant flags (recomputed on the kept rows) so downstream can
    # annotate / retain contaminants (e.g. a transgene spiked into the cRAP db)
    # without re-reading the file. All False when drop_decoys_contaminants=True.
    proteins["is_reverse"] = _flag("Reverse").to_numpy()
    proteins["is_potential_contaminant"] = _flag("Potential contaminant").to_numpy()
    proteins["is_only_by_site"] = _flag("Only identified by site").to_numpy()

    # groups: explicit, validated, loud
    if groups is not None:
        groups = dict(groups)
        unknown = [s for s in groups if s not in samples]
        unmapped = [s for s in samples if s not in groups]
        if unknown:
            raise ValueError(f"groups references samples not in the data: {unknown}. "
                             f"Data samples: {samples}")
        if unmapped:
            raise ValueError(f"these samples have no group assigned: {unmapped}. "
                             f"Map every sample, or pass groups=None for ungrouped.")
        groups = {s: groups[s] for s in samples}
    else:
        groups = {}

    mq_version = _read_mq_version(parameters_path) if parameters_path else None

    return QuantData(
        lfq=lfq, intensity=intensity, id_type=id_type, proteins=proteins,
        samples=list(samples), groups=groups, source=path,
        mq_version=mq_version, n_protein_groups_raw=n_raw, n_filtered=n_filtered,
    )