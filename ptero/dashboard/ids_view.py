"""
ID-annotation page for the PTERO dashboard.

Workflow (spectra source: a PTERO ``.zarr`` store **or** an mzML):

1. Load a spectra store -- a PTERO ``.zarr`` (raw or processed) or a vendor
   peak-picked **mzML** -- plus a search-engine ``msms.txt`` (PSMs). Optionally
   ``accumulatedMsmsScans.txt`` (per-PSM 1/K0) and ``summary.txt`` (the search's
   fixed/variable mods).
2. PSMs are listed in a selectable table.
3. Selecting a PSM locates its MS2 spectrum in the store by physical coordinates
   (RT + precursor m/z + ion mobility), annotates the b/y ions, and shows:
   * the **annotated MS2 stick plot** -- matched b/y peaks coloured by series and
     labelled, the rest grey (only the genuine, one-to-one fragment matches are
     labelled, not the isotopolog/multi-charge over-count);
   * an **XIC context** -- the precursor m/z extracted over a narrow RT window
     around the PSM (the "raw data" view of where this identification elutes);
   * the PSM's coordinates and the fragment-coverage score.

Spectra source -- Zarr or mzML: the link is property-based (RT + precursor m/z +
1/K0) and the annotator matches b/y ions to the nearest peak within tolerance, so
it works directly on a PTERO ``.zarr`` store whose MS2 spectra are centroided
(verified on real timsTOF data: deisotoping is not required because the
monoisotopic fragment peak is present and is the nearest match). A vendor
peak-picked mzML also works. Using the ``.zarr`` keeps everything in PTERO's store
and removes the separate mzML (and the MSConvert step that produced it).
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv

from ._state import AppState

hv.extension("bokeh")

# b/y series colours for the annotated spectrum
_SERIES_COLOR = {"b": "#1f77b4", "y": "#d62728"}     # blue b-ions, red y-ions
_UNMATCHED_COLOR = "#b0b0b0"                          # grey

# peptide-class colours for the coverage map
_CLASS_COLOR = {"unique": "#1f77b4", "razor": "#ff7f0e", "shared": "#7f7f7f"}


class IdsView:
    """The ID-annotation page. Shares :class:`AppState` with the other pages."""

    XIC_RT_WINDOW_S = 30.0       # half-width of the XIC RT window around the PSM
    XIC_MZ_TOL = 0.02            # precursor XIC tolerance (Da)
    MAX_TABLE_ROWS = 2000        # cap the PSM table for responsiveness

    def __init__(self, state: AppState | None = None):
        self.state = state if state is not None else AppState()
        self._match = None       # last PsmMatch (for the plots)
        self._bruker_associations = None  # native Bruker PSM-to-MS2 associations
        self._bruker_selected_psm = None
        self._bruker_spectrum_indices = ()
        self._updating_bruker_selector = False
        self._xic_request = None

        # Shown only when a selected PSM has native Bruker associations.
        # Each option represents one distinct stored MS2 spectrum.
        self.bruker_spectrum_selector = pn.widgets.Select(
            name="Associated Bruker MS2 spectrum",
            options=[],
            value=None,
            width=460,
            visible=False,
        )
        self.bruker_association_note = pn.pane.Markdown(
            "",
            visible=False,
        )
        self._last_spectrum = None   # (annotated DataFrame, title) for redraws
        self._iddata = None      # IdentificationData (protein+peptide tables)
        self._psm_by_seq = {}    # (sequence, run) -> best PSM
        self._run_id_index = {}   # run -> direct PSM-supported groups/sequences
        self._native_scan_runs = set()
        self._updating_run_selector = False
        self.run_selector = pn.widgets.Select(
            name="Experimental run",
            options=[],
            value=None,
            width=460,
            visible=False,
        )
        self._pep_df = None      # current peptide table (for the selected protein)
        self.peptide_heading = pn.pane.Markdown(
            "### Peptides - direct PSM support in selected run"
        )
        self._fasta = None       # ProteinSequences (Stage 2), or None if not loaded
        self._coverage_mapping = None   # current PeptideMapping (for peptide highlight)
        self._mod_sites = None       # ModificationSites (Stage 3), or None
        self._run_mod_site_ids = {}  # raw run -> modification -> site IDs
        self._coverage_sites = []    # sites for the currently shown protein
        self._coverage_protein_row = None
        self._coverage_highlight_peptide = None
        self._updating_accession_selector = False
        self.accession_selector = pn.widgets.Select(
            name="FASTA protein accession",
            options=[],
            value=None,
            width=340,
            visible=False,
        )

        # ── inputs ──
        self.dataset_source = pn.widgets.RadioButtonGroup(
            name="PTERO Dataset source",
            options=["Use datasets from Raw Data",
                     "Open PTERO Dataset separately"],
            value="Use datasets from Raw Data",
        )
        self.store_path = pn.widgets.TextInput(
            name="PTERO Dataset (or folder of PTERO Datasets)",
            placeholder=r"C:\data\run.zarr   or   C:\data\datasets",
            width=460, visible=False)
        self.dataset_source.param.watch(
            self._on_dataset_source_change, "value")
        self.txt_dir = pn.widgets.TextInput(
            name="MaxQuant txt folder",
            placeholder=r"C:\data\combined\txt", width=460)
        self.fasta_path = pn.widgets.TextInput(
            name="FASTA (the database used in the search; for the coverage map)",
            placeholder=r"C:\data\human.fasta", width=460)
        self.load_btn = pn.widgets.Button(name="Load identifications", button_type="primary", width=170)

        self.status = pn.pane.Markdown(
            "**No identifications loaded.** Reuse PTERO Datasets from Raw Data "
            "or open them separately, then select the MaxQuant `txt` folder.")

        # ── protein table (top, selectable) ──
        self.protein_search = pn.widgets.TextInput(
            name="Search proteins",
            placeholder="Gene, protein name, or UniProt accession",
            width=460,
        )
        self._displayed_protein_groups = None
        self._updating_protein_table = False

        self.protein_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["gene", "protein", "n_peptides", "n_unique",
                                  "razor_unique", "coverage_pct", "n_psms", "score"]),
            selectable=1, disabled=True, height=260, width=820, show_index=True,
            pagination="local", page_size=15,
            widths={"protein": 240, "gene": 90})

        # ── peptide table (filtered to the selected protein, selectable) ──
        self.peptide_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["peptide", "unique", "mods", "charge", "score", "PEP"]),
            selectable=1, disabled=True, height=240, width=620, show_index=True,
            pagination="local", page_size=15,
            widths={"peptide": 230, "mods": 150})

        # ── output panels ──
        self.coverage_panel = pn.Column(sizing_mode="stretch_width")
        self.spectrum_panel = pn.Column(
            pn.pane.Markdown("_Select a protein, then a peptide, to see its annotated spectrum._"),
            sizing_mode="stretch_width")
        self.xic_panel = pn.Column(sizing_mode="stretch_width")
        self.xic_button = pn.widgets.Button(
            name="Generate precursor XIC",
            button_type="default", width=220, disabled=True)
        self.info_panel = pn.pane.Markdown("")

        # display toggle: hide unmatched (grey) peaks to see just the b/y ladder
        self.annotated_only = pn.widgets.Checkbox(
            name="Show annotated peaks only", value=False, width=220)

        # ── wiring ──
        self.load_btn.on_click(self._on_load)
        self.xic_button.on_click(self._on_generate_xic)
        self.protein_table.param.watch(self._on_select_protein, "selection")
        self.protein_search.param.watch(self._on_protein_search, "value")
        self.peptide_table.param.watch(self._on_select_peptide, "selection")
        self.run_selector.param.watch(self._on_select_run, "value")
        self.annotated_only.param.watch(self._on_redraw_spectrum, "value")
        self.bruker_spectrum_selector.param.watch(self._on_select_bruker_spectrum, "value")
        self.accession_selector.param.watch(
            self._on_select_coverage_accession, "value"
        )

    # ── loading ──────────────────────────────────────────────────────────
    def _on_dataset_source_change(self, event):
        self.store_path.visible = (
            event.new == "Open PTERO Dataset separately")

    async def _on_load(self, event=None):
        from pathlib import Path
        self._xic_request = None
        self.xic_button.disabled = True
        self.xic_panel[:] = []
        use_raw = self.dataset_source.value == "Use datasets from Raw Data"
        store_path = self.store_path.value.strip()
        shared_datasets = dict(self.state.raw_datasets) if use_raw else {}
        txt_dir = self.txt_dir.value.strip()
        if not txt_dir:
            self.status.object = "⚠️ Select the MaxQuant `txt` folder."
            return
        if use_raw and not shared_datasets:
            self.status.object = (
                "No PTERO Datasets are registered in Raw Data. "
                "Open or convert them there, or select the separate option.")
            return
        if not use_raw and not store_path:
            self.status.object = "Select a PTERO Dataset or dataset folder."
            return
        folder = Path(txt_dir)
        if not folder.is_dir():
            self.status.object = f"❌ **Not a folder:** `{txt_dir}`"
            return

        # resolve the MaxQuant files by their standard names (case-insensitive)
        def find(name):
            exact = folder / name
            if exact.is_file():
                return str(exact)
            for f in folder.iterdir():     # case-insensitive fallback
                if f.is_file() and f.name.lower() == name.lower():
                    return str(f)
            return None

        msms = find("msms.txt")
        proteins = find("proteinGroups.txt")
        peptides = find("peptides.txt")
        summary = find("summary.txt")
        accum = find("accumulatedMsmsScans.txt")

        missing_required = [n for n, p in
                            (("msms.txt", msms), ("proteinGroups.txt", proteins),
                             ("peptides.txt", peptides)) if p is None]
        if missing_required:
            self.status.object = (
                f"❌ **Missing required file(s) in the txt folder:** "
                f"{', '.join(missing_required)}. (Found: "
                f"{', '.join(n for n,p in (('msms.txt',msms),('proteinGroups.txt',proteins),('peptides.txt',peptides)) if p)}.)")
            return

        found = ["msms.txt", "proteinGroups.txt", "peptides.txt"]
        opt_note = []
        found.append("summary.txt") if summary else opt_note.append("summary.txt (mods)")
        found.append("accumulatedMsmsScans.txt") if accum else opt_note.append("accumulatedMsmsScans.txt (1/K0)")

        self.status.object = (
            f"⏳ Found {', '.join(found)}"
            + (f"; not found: {', '.join(opt_note)}" if opt_note else "")
            + ". Loading PTERO Dataset metadata, PSMs and protein/peptide tables…")
        self.spectrum_panel.loading = True
        await asyncio.sleep(0.05)
        try:
            from .. import ids
            from ._ids_data import IdentificationData, run_identification_index

            source_path = None if use_raw else Path(store_path)
            multi_run = use_raw or (
                source_path.is_dir()
                and source_path.suffix.lower() != ".zarr"
                and not (source_path / "offsets").exists()
            )
            if not multi_run and source_path.suffix.lower() != ".zarr":
                raise ValueError(
                    "Identification requires a PTERO .zarr store "
                    "or a folder containing .zarr stores."
                )
            store = None if multi_run else _open_spectra_store(store_path)

            psms = ids.read_psms(msms)
            n_mob = ids.attach_mobility_from_maxquant(psms, accum) if accum else 0
            n_mod = ids.convert_psm_modifications(psms, summary_path=summary)

            run_id = psms[0].run if len(psms) else "run"
            reg = ids.RunRegistry()
            if multi_run:
                expected_runs = {str(p.run) for p in psms}
                if not expected_runs:
                    raise ValueError("No MaxQuant PSM runs found.")

                if use_raw:
                    candidates = [Path(p) for p in shared_datasets.values()]
                else:
                    candidates = [
                        path for path in source_path.iterdir()
                        if path.is_dir() and path.suffix.lower() == ".zarr"
                    ]
                by_name = {}
                for path in candidates:
                    if path.suffix.lower() != ".zarr" or not path.is_dir():
                        raise ValueError(
                            f"Missing or invalid PTERO Dataset: {path}")
                    key = path.stem.casefold()
                    if key in by_name:
                        raise ValueError(
                            f"Ambiguous PTERO Dataset name: {path.stem}"
                        )
                    by_name[key] = path

                # A single registered dataset and a single MaxQuant run
                # form an unambiguous user-selected association, even when
                # the dataset has a validation-specific filename.
                # Multiple runs always require exact name matching.
                if use_raw and len(by_name) == 1 and len(expected_runs) == 1:
                    only_run = next(iter(expected_runs))
                    only_path = next(iter(by_name.values()))
                    by_name[only_run.casefold()] = only_path

                missing = sorted(
                    name for name in expected_runs
                    if name.casefold() not in by_name
                )
                if missing:
                    raise ValueError(
                        "Missing PTERO Datasets for MaxQuant runs: "
                        + ", ".join(missing)
                    )

                active_path = None
                if use_raw and self.state.raw_sample in shared_datasets:
                    active_path = Path(
                        shared_datasets[self.state.raw_sample]).resolve()
                for name in sorted(expected_runs):
                    matched = by_name[name.casefold()]
                    if (use_raw and self.state.raw_store is not None
                            and active_path == matched.resolve()):
                        reg.register(name, self.state.raw_store)
                    else:
                        reg.register(name, str(matched))

                run_id = sorted(expected_runs)[0]
                store = reg.get(run_id)
            else:
                expected_runs = {str(p.run) for p in psms}
                if len(expected_runs) > 1:
                    raise ValueError(
                        "MaxQuant contains multiple experimental runs. "
                        "Select the matching PTERO Dataset folder "
                        "or use datasets from Raw Data.")
                reg.register(run_id, store)

            if multi_run:
                matched_store_path = by_name[run_id.casefold()]
            else:
                matched_store_path = source_path

            # Native Bruker linking is opt-in by data capability.
            # mzML, Thermo, and legacy Zarr retain property-based matching.
            self._bruker_associations = None
            self._bruker_selected_psm = None
            self._bruker_spectrum_indices = ()
            self.bruker_spectrum_selector.visible = False
            self.bruker_association_note.visible = False
            if accum and len(reg.runs()) == 1 and matched_store_path.is_dir():
                import pyarrow.parquet as pq

                metadata_path = matched_store_path / "spectra.parquet"
                if metadata_path.is_file():
                    available = set(
                        pq.read_schema(metadata_path).names
                    )
                    required = {
                        "scan_index", "ms_level", "bruker_precursor_id"
                    }
                    if required.issubset(available):
                        native_metadata = pd.read_parquet(
                            metadata_path,
                            columns=sorted(required),
                        )
                        if (
                            native_metadata.loc[
                                native_metadata["ms_level"] == 2,
                                "bruker_precursor_id",
                            ].notna().all()
                        ):
                            self._bruker_associations = (
                                ids.BrukerPsmAssociations(
                                    accum, native_metadata
                                )
                            )
            self._registry = reg
            self._multi_run = multi_run

            by_seq = {}
            for p in psms:
                if p.peptidoform is None:
                    continue
                key = (p.peptidoform.sequence, str(p.run))
                sc = getattr(p, "score", None)
                sc = float(sc) if sc is not None else float("-inf")
                old = by_seq.get(key)
                old_score = getattr(old, "score", None) if old is not None else None
                old_score = (
                    float(old_score)
                    if old_score is not None else float("-inf")
                )
                if old is None or sc > old_score:
                    by_seq[key] = p
            self._psm_by_seq = by_seq

            # Exact native-scan matching is enabled only for stores whose
            # MS2 acquisition numbers are complete and unique.
            self._native_scan_runs = set()
            import pyarrow.parquet as pq

            for name in reg.runs():
                zarr_path = (
                    by_name[name.casefold()]
                    if multi_run else source_path
                )
                metadata_path = zarr_path / "spectra.parquet"
                if not metadata_path.is_file():
                    continue
                schema = set(pq.read_schema(metadata_path).names)
                required = {"acquisition_num", "ms_level"}
                if not required.issubset(schema):
                    continue
                metadata = pd.read_parquet(
                    metadata_path, columns=["acquisition_num", "ms_level"]
                )
                ms2 = metadata.loc[
                    metadata["ms_level"] == 2, "acquisition_num"
                ]
                if (
                    len(ms2) > 0
                    and ms2.notna().all()
                    and ms2.is_unique
                ):
                    self._native_scan_runs.add(name)

            self._updating_run_selector = True
            try:
                self.run_selector.options = reg.runs()
                self.run_selector.value = run_id
                self.run_selector.visible = len(reg.runs()) > 1
            finally:
                self._updating_run_selector = False

            self._iddata = IdentificationData(proteins, peptides)
            self._run_id_index = run_identification_index(msms)
            missing_runs = set(reg.runs()) - set(self._run_id_index)
            if missing_runs:
                raise ValueError(
                    "No MaxQuant PSM support for registered run(s): "
                    + ", ".join(sorted(missing_runs))
                )
            self._coverage_protein_row = None
            self._coverage_highlight_peptide = None
            self._coverage_mapping = None
            self._coverage_sites = []
            self.accession_selector.visible = False

            # modification sites for the sequence view (Stage 3): read all
            # *Sites.txt in the folder (oxidation, phospho, …). Authoritative
            # protein positions + localisation, generic over modification type.
            from ._mods_sites import ModificationSites
            self._mod_sites = ModificationSites(str(folder))
            self._run_mod_site_ids = self._index_run_mod_sites(
                msms, self._mod_sites.mod_types
            )

            # optional FASTA for the coverage map (Stage 2)
            fasta_path = self.fasta_path.value.strip()
            n_fasta = 0
            if fasta_path:
                from ._fasta import ProteinSequences
                self._fasta = ProteinSequences(fasta_path)
                n_fasta = len(self._fasta)
            else:
                self._fasta = None

            self.state.set("id_store", store)
            self.state.set("psms", psms)
            self.state.set("psm_run", run_id)

            self._updating_protein_table = True
            try:
                self.protein_search.value = ""
                self.protein_table.selection = []
                supported = self._run_id_index[run_id]["groups"]
                self._displayed_protein_groups = (
                    self._iddata.protein_table.loc[
                        lambda df: df["_gid"].isin(supported)
                    ].reset_index(drop=True).copy()
                )
                self.protein_table.value = (
                    self._displayed_protein_groups.drop(columns="_gid")
                    .reset_index(drop=True)
                )
            finally:
                self._updating_protein_table = False
            self.peptide_table.value = self.peptide_table.value.iloc[0:0]
            n_groups = len(self._iddata.protein_table)
            fasta_note = (f" FASTA: {n_fasta} sequences (coverage map on)."
                          if self._fasta else " No FASTA (coverage map off).")
            self.status.object = (
                f"✅ Loaded {len(psms)} PSMs, {n_groups} protein groups, "
                f"{len(reg.runs())} spectra run(s) "
                f"(first run: {len(store)} spectra, {'4D' if store.has_4d else '3D'}). "
                f"1/K0 on {n_mob}; modification-site protein-group entries: "
                f"{n_mod}.{fasta_note} "
                f"Select a protein, then a peptide."
                + (" Native Bruker precursor-ID associations available."
                   if self._bruker_associations is not None
                   else (
                       " Exact native-scan matching available."
                       if self._native_scan_runs
                       else " Property-based spectrum matching."
                   ))
                + (f"  _(optional not found: {', '.join(opt_note)})_" if opt_note else ""))
        except Exception as e:
            self.status.object = f"❌ **Load failed:** {e}"
        finally:
            self.spectrum_panel.loading = False

    # ── selection: protein -> peptides -> spectrum ────────────────────────
    def _on_protein_search(self, event=None):
        """Filter displayed groups without modifying original MaxQuant data."""
        if self._iddata is None or self._updating_protein_table:
            return

        supported = self._run_id_index.get(
            self.run_selector.value, {}
        ).get("groups", set())
        source = self._iddata.protein_table.loc[
            lambda df: df["_gid"].isin(supported)
        ]
        query = self.protein_search.value.strip()

        if query:
            mask = pd.Series(False, index=source.index)
            for field in ("gene", "protein", "majority_ids"):
                mask |= source[field].fillna("").astype(str).str.contains(
                    query, case=False, regex=False
                )
            displayed = source.loc[mask].reset_index(drop=True).copy()
        else:
            displayed = source.reset_index(drop=True).copy()

        self._updating_protein_table = True
        try:
            self.protein_table.selection = []
            self._displayed_protein_groups = displayed
            self.protein_table.value = (
                displayed.drop(columns="_gid").reset_index(drop=True)
            )
        finally:
            self._updating_protein_table = False

    async def _on_select_protein(self, event=None):
        sel = self.protein_table.selection
        if (
            self._updating_protein_table
            or not sel
            or self._iddata is None
            or self._displayed_protein_groups is None
        ):
            return
        row = int(sel[0])
        if not 0 <= row < len(self._displayed_protein_groups):
            return
        prow = self._displayed_protein_groups.iloc[row]
        gid = prow["_gid"]
        # show loading on the panels we're about to fill, and flush to the
        # browser before the (blocking) peptide filtering + FASTA mapping
        self.peptide_table.loading = True
        self.coverage_panel.loading = True
        self.coverage_panel[:] = [pn.pane.Markdown(
            f"⏳ Mapping peptides onto {prow.get('gene','') or prow.get('protein','')}…")]
        self.spectrum_panel[:] = [pn.pane.Markdown("_Select a peptide to see its annotated spectrum._")]
        self.xic_panel[:] = []
        self.info_panel.object = ""
        await asyncio.sleep(0.05)
        try:
            self._coverage_highlight_peptide = None
            all_peptides = self._iddata.peptides_for_group(gid)
            supported_sequences = self._run_id_index.get(
                self.run_selector.value, {}
            ).get("group_sequences", {}).get(str(gid), set())
            self._pep_df = all_peptides.loc[
                all_peptides["_seq"].isin(supported_sequences)
            ].reset_index(drop=True).copy()
            self.peptide_table.value = (
                self._pep_df.drop(columns="_seq").reset_index(drop=True)
            )
            gene = str(prow.get("gene", "") or "").strip()
            protein = str(prow.get("protein", "") or "").strip()
            label = " | ".join(part for part in (gene, protein) if part)
            self.peptide_heading.object = (
                f"### Peptides - {label} "
                f"({len(self._pep_df)} sequences with PSMs in selected run)"
            )
            self._render_coverage(prow)        # coverage map (Stage 2) if FASTA loaded
        finally:
            self.peptide_table.loading = False
            self.coverage_panel.loading = False

    @staticmethod
    def _index_run_mod_sites(msms_path, mod_types):
        """Index directly PSM-supported modification sites per raw run.

        MaxQuant site IDs are scoped to each modification type. This index
        does not imply run-specific localization probabilities.
        """
        import csv

        index = {}
        with open(msms_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=chr(9))
            columns = set(reader.fieldnames or [])
            if "Raw file" not in columns:
                raise ValueError("msms.txt lacks 'Raw file'")

            site_columns = {
                mod: f"{mod} site IDs"
                for mod in mod_types
                if f"{mod} site IDs" in columns
            }

            for row in reader:
                run = (row.get("Raw file") or "").strip()
                if not run:
                    continue

                run_index = index.setdefault(
                    run, {mod: set() for mod in mod_types}
                )
                for mod, column in site_columns.items():
                    for site_id in (row.get(column) or "").split(";"):
                        site_id = site_id.strip()
                        if site_id:
                            run_index[mod].add(site_id)

        return index

    def _render_coverage(self, protein_row):
        """Configure majority-protein accessions for sequence coverage."""
        self._coverage_protein_row = protein_row
        self._coverage_mapping = None
        self._coverage_sites = []

        majority = str(protein_row.get("majority_ids", "") or "")
        accessions = list(dict.fromkeys(
            accession.strip()
            for accession in majority.split(";")
            if accession.strip()
        ))

        self._updating_accession_selector = True
        try:
            self.accession_selector.options = accessions
            self.accession_selector.value = (
                accessions[0] if accessions else None
            )
            self.accession_selector.visible = (
                self._fasta is not None and len(accessions) > 1
            )
        finally:
            self._updating_accession_selector = False

        self._render_selected_coverage_accession()

    def _on_select_coverage_accession(self, event=None):
        """Recalculate the map when the selected accession changes."""
        if not self._updating_accession_selector:
            self._render_selected_coverage_accession()

    def _render_selected_coverage_accession(self):
        """Map peptides and verified sites onto the selected FASTA sequence."""
        self._coverage_mapping = None
        self._coverage_sites = []

        if self._fasta is None:
            self.coverage_panel[:] = [pn.pane.Markdown(
                "_Load a FASTA to see the sequence coverage map._"
            )]
            return

        protein_row = self._coverage_protein_row
        if protein_row is None or self._pep_df is None:
            return

        accession = self.accession_selector.value
        sequence = self._fasta.get(accession) if accession else None

        if not sequence:
            self.coverage_panel[:] = [pn.pane.Markdown(
                f"Protein accession `{accession or '?'}` is not present "
                "in the loaded FASTA. Coverage and modification markers "
                "cannot be displayed for this accession."
            )]
            return

        from ._fasta import map_peptides

        peptides = self._pep_df[["peptide", "unique"]].to_dict("records")
        mapping = map_peptides(accession, sequence, peptides)
        self._coverage_mapping = mapping

        gid = str(protein_row["_gid"])
        self._coverage_sites = (
            self._mod_sites.for_group_accession(
                gid,
                accession,
                sequence,
                supported_site_ids=self._run_mod_site_ids.get(
                    self.run_selector.value, {}
                ),
            )
            if self._mod_sites is not None else []
        )

        self._show_coverage(
            mapping,
            highlight_peptide=self._coverage_highlight_peptide,
        )


    def _show_coverage(self, mapping, highlight_peptide=None):
        """Render the coverage panel; optionally outline one peptide's residues.
        Modification sites (Stage 3) are marked from ``self._coverage_sites``."""
        from ._fasta import locate_peptide
        highlight = set()
        if highlight_peptide:
            for (s, e) in locate_peptide(highlight_peptide, mapping.sequence):
                highlight.update(range(s, e))
        sites = getattr(self, "_coverage_sites", []) or []
        note = (f"_Coverage **{mapping.coverage_pct:.1f}%** of {mapping.accession} "
                f"({len(mapping.sequence)} aa). Blue = unique, orange = razor, grey = shared; "
                f"hover a residue to see its peptide(s). "
                + (f"{len(mapping.not_located)} peptide(s) not located._"
                   if mapping.not_located else "All peptides located._"))
        if sites:
            mods = sorted({s["mod"] for s in sites})
            note += (f"  \n_Modification sites marked below the sequence "
                     f"({', '.join(mods)}): **solid** = localisation ≥ 0.75, "
                     f"**hollow** = below 0.75 (uncertain). "
                     f"Markers require direct PSM support in the selected run; "
                     f"localization probabilities are combined MaxQuant values. "
                     f"Hover a marker for details._")
        if highlight_peptide:
            note += f"  \n_Outlined in red: **{highlight_peptide}**._"
        self.coverage_panel[:] = [
            pn.pane.HTML(self._coverage_html(mapping, highlight=highlight, sites=sites),
                         sizing_mode="stretch_width"),
            pn.pane.Markdown(note),
        ]

    def _coverage_html(self, mapping, highlight=None, sites=None):
        """Protein sequence as monospace text with covered residues background-
        coloured (unique/razor/shared), position markers every 10 residues, a
        per-residue tooltip naming the covering peptide(s), an optional red
        outline (the selected peptide), and modification-site markers (Stage 3)
        beneath modified residues."""
        highlight = highlight or set()
        sites = sites or []
        seq = mapping.sequence
        rcls = mapping.residue_class
        rpep = mapping.residue_peptides
        bg = {"unique": "#1f77b4", "razor": "#ff7f0e", "shared": "#9e9e9e"}
        fg = {"unique": "#ffffff", "razor": "#000000", "shared": "#ffffff"}

        # one colour per modification type for the markers
        mod_palette = ["#2ca02c", "#9467bd", "#e377c2", "#17becf", "#bcbd22", "#8c564b"]
        mod_types = sorted({s["mod"] for s in sites})
        mod_color = {m: mod_palette[i % len(mod_palette)] for i, m in enumerate(mod_types)}

        # per-residue (0-based) modification markers: position -> (color, solid, title)
        LOC_THRESH = 0.75
        markers: dict[int, tuple] = {}
        for s in sites:
            i = s["position"] - 1                       # 1-based -> 0-based
            if not (0 <= i < len(seq)):
                continue
            loc = s["localization"]
            solid = (loc >= LOC_THRESH) if loc == loc else False   # NaN -> hollow
            loc_txt = f"{loc:.2f}" if loc == loc else "n/a"
            title = f"{s['residue']}{s['position']} — {s['mod']}, localization {loc_txt}"
            # if several mods on one residue, keep the higher-confidence one for the glyph
            if i not in markers or (solid and not markers[i][1]):
                markers[i] = (mod_color[s["mod"]], solid, title)

        def residue_cell(i, aa):
            cls = rcls[i]
            hl = ("outline:2px solid #d62728;outline-offset:-1px;" if i in highlight else "")
            if cls:
                peps = ", ".join(rpep[i][:6]) + ("…" if len(rpep[i]) > 6 else "")
                rtitle = f"{aa}{i+1} — {cls}: {peps}"
                letter = (f'<span title="{rtitle}" style="background:{bg[cls]};color:{fg[cls]};'
                          f'border-radius:2px;{hl}">{aa}</span>')
            else:
                letter = f'<span style="color:#888;{hl}">{aa}</span>'
            # marker slot beneath the letter
            if i in markers:
                color, solid, mtitle = markers[i]
                dot = (f'<span title="{mtitle}" style="display:block;width:8px;height:8px;'
                       f'margin:1px auto 0;border-radius:50%;border:1.5px solid {color};'
                       f'background:{color if solid else "transparent"};"></span>')
            else:
                dot = '<span style="display:block;height:9px;"></span>'
            return ('<span style="display:inline-block;text-align:center;">'
                    f'{letter}{dot}</span>')

        per_line = 60
        block = 10
        lines = []
        for start in range(0, len(seq), per_line):
            cells = [residue_cell(i, seq[i]) for i in range(start, min(start + per_line, len(seq)))]
            pieces = []
            for j in range(0, len(cells), block):
                pieces.append("".join(cells[j:j + block]))
            num = (f'<span style="display:inline-block;color:#666;vertical-align:top;'
                   f'padding-top:2px;">{start + 1:>5}</span>&nbsp;&nbsp;')
            lines.append('<div style="margin-bottom:4px;">' + num + "&nbsp;".join(pieces) + "</div>")

        # legend for modification colours
        legend = ""
        if mod_types:
            chips = "  ".join(
                f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                f'background:{mod_color[m]};border:1.5px solid {mod_color[m]};"></span> {m}'
                for m in mod_types)
            legend = f'<div style="margin:6px 0 2px;font-size:12px;color:#444;">{chips}</div>'

        body = "".join(lines)
        return (
            '<div style="font-family:Consolas,Menlo,monospace;font-size:14px;'
            'line-height:1.3;letter-spacing:1px;white-space:nowrap;overflow-x:auto;'
            'padding:8px;border:1px solid #e0e0e0;border-radius:4px;background:#fafafa;">'
            f'{legend}{body}</div>')

    def _on_select_run(self, event=None):
        if self._updating_run_selector:
            return
        run_id = self.run_selector.value
        if not run_id or not hasattr(self, "_registry"):
            return
        store = self._registry.get(run_id)
        self.state.set("id_store", store)
        self.state.set("psm_run", run_id)
        self._match = None
        self._last_spectrum = None
        self._xic_request = None
        self.xic_button.disabled = True
        self.xic_panel[:] = []
        self.info_panel.object = ""
        self.bruker_spectrum_selector.visible = False
        self.bruker_association_note.visible = False
        self.spectrum_panel[:] = [
            pn.pane.Markdown(
                "_Select a peptide to inspect its spectrum in the selected run._"
            )
        ]
        self.peptide_table.selection = []
        self._pep_df = None
        self.peptide_table.value = self.peptide_table.value.iloc[0:0]
        self.peptide_heading.object = (
            "### Peptides - direct PSM support in selected run"
        )
        self._coverage_mapping = None
        self._coverage_protein_row = None
        self._coverage_highlight_peptide = None
        self._coverage_sites = []
        self.accession_selector.visible = False
        self.coverage_panel[:] = [
            pn.pane.Markdown("_Select a protein to see its sequence coverage._")
        ]
        self._on_protein_search()

    async def _on_select_peptide(self, event=None):
        self._xic_request = None
        self.xic_button.disabled = True
        self.xic_panel[:] = []
        sel = self.peptide_table.selection
        if not sel or self._pep_df is None:
            return
        row = int(sel[0])
        seq = self._pep_df.iloc[row]["_seq"]
        # outline this peptide on the coverage sequence (if a map is shown)
        self._coverage_highlight_peptide = seq
        if getattr(self, "_coverage_mapping", None) is not None:
            self._show_coverage(self._coverage_mapping, highlight_peptide=seq)
        run_id = self.run_selector.value
        psm = self._psm_by_seq.get((seq, run_id))
        if psm is None:
            self.spectrum_panel[:] = [pn.pane.Markdown(
                f"⚠️ No PSM found in msms.txt for peptide `{seq}` "
                "(it may be identified only in another file).")]
            self.xic_panel[:] = []
            return
        self.spectrum_panel.loading = True
        await asyncio.sleep(0.05)
        try:
            from .. import ids
            if self._bruker_associations is not None:
                indices = self._bruker_associations.spectrum_indices_for_psm(psm)
                self._bruker_selected_psm = psm
                self._bruker_spectrum_indices = indices

                options = {
                    f"MS2 spectrum {index} ({position}/{len(indices)})": index
                    for position, index in enumerate(indices, 1)
                }
                self._updating_bruker_selector = True
                try:
                    self.bruker_spectrum_selector.options = options
                    self.bruker_spectrum_selector.value = (
                        indices[0] if indices else None
                    )
                    self.bruker_spectrum_selector.visible = bool(indices)
                finally:
                    self._updating_bruker_selector = False

                if len(indices) > 1:
                    self.bruker_association_note.object = (
                        f"**{len(indices)} native Bruker MS2 associations.** "
                        "Select an individual acquisition to inspect. "
                        "These associations do not establish that every "
                        "acquisition contributed to the accumulated "
                        "MaxQuant searched spectrum."
                    )
                elif indices:
                    self.bruker_association_note.object = (
                        "**1 native Bruker MS2 association.**"
                    )
                else:
                    self.bruker_association_note.object = (
                        "No native Bruker MS2 association for this PSM."
                    )
                self.bruker_association_note.visible = True

                self._render_selected_bruker_spectrum()
            else:
                self._bruker_selected_psm = None
                self._bruker_spectrum_indices = ()
                self.bruker_spectrum_selector.visible = False
                self.bruker_association_note.visible = False

                if psm.run in self._native_scan_runs:
                    match = ids.spectrum_for_native_scan(
                        psm, self._registry
                    )
                else:
                    match = ids.spectrum_for_psm(psm, self._registry)
                self._match = match
                self._render_spectrum(psm, match)
                self._prepare_xic(psm, match)
                self._render_info(psm, match)
        except Exception as e:
            self.spectrum_panel[:] = [pn.pane.Markdown(f"❌ **Annotation error:** {e}")]
        finally:
            self.spectrum_panel.loading = False

    def _on_select_bruker_spectrum(self, event=None):
        """Render an explicitly selected native Bruker MS2 association."""
        if self._updating_bruker_selector:
            return
        if self._bruker_associations is None:
            return
        if self._bruker_selected_psm is None:
            return
        if self.bruker_spectrum_selector.value not in self._bruker_spectrum_indices:
            return
        self._render_selected_bruker_spectrum()

    def _render_selected_bruker_spectrum(self):
        """Display one association without discarding the other candidates."""
        from .. import ids

        psm = self._bruker_selected_psm
        index = self.bruker_spectrum_selector.value
        indices = self._bruker_spectrum_indices

        if psm is None or index not in indices:
            self._match = None
            self._last_spectrum = None
            self.spectrum_panel[:] = [
                pn.pane.Markdown(
                    "No associated Bruker MS2 spectrum is selected."
                )
            ]
            self.xic_panel[:] = []
            return

        run = self._registry.get(psm.run)
        spectrum = run[int(index)]

        candidates = pd.DataFrame({
            "store_index": list(indices),
        })

        match = ids.PsmMatch(
            psm=psm,
            store_index=int(index),
            spectrum=spectrum,
            candidates=candidates,
            method="bruker-precursor-id-association",
        )

        self._match = match
        self._render_spectrum(psm, match)
        self._prepare_xic(psm, match)
        self._render_info(psm, match)

    def _render_spectrum(self, psm, match):
        if not match.resolved or match.spectrum is None:
            self._last_spectrum = None
            self.spectrum_panel[:] = [pn.pane.Markdown(
                f"⚠️ No unique spectrum found for this PSM (method: `{match.method}`, "
                f"{match.n_candidates} candidate(s)).")]
            return
        sp = match.spectrum
        mz = np.asarray(sp.mz([0])[0], float)
        inten = np.asarray(sp.intensity([0])[0], float)
        pep = psm.peptidoform
        ann = ids_peak_annotations(mz, inten, pep.proforma, pep.precursor_charge)

        if inten.size == 0:
            self._last_spectrum = None
            self.spectrum_panel[:] = [pn.pane.Markdown("⚠️ Matched spectrum has no peaks.")]
            return

        # remember for redraws (e.g. toggling "annotated only") without re-reading
        self._last_spectrum = (ann.assign(
            color=ann["series"].map(_SERIES_COLOR).fillna(_UNMATCHED_COLOR)), str(pep.proforma))
        self.spectrum_panel[:] = [
            pn.pane.HoloViews(self._spectrum_overlay(), sizing_mode="stretch_width"),
            pn.pane.Markdown(
                "_Blue = b-ions, red = y-ions, grey = unmatched peaks. Matched ions are drawn "
                "on top with a marker so short fragments stay visible. Only genuine one-to-one "
                "b/y matches (charge ≤ precursor) are labelled. Use the toggle to hide unmatched peaks._"),
        ]

    def _spectrum_overlay(self):
        """Build the annotated-spectrum overlay from the remembered peaks.

        Matched (b/y) peaks are drawn *after* the grey ones (so they sit on top,
        never hidden behind a taller grey stick at the same m/z), thicker, and
        with a marker at the tip so a short fragment is still obvious. The
        ``annotated_only`` toggle drops the grey layer entirely.
        """
        base, title = self._last_spectrum
        ymax = float(base["intensity"].max()) * 1.18 if len(base) else 1.0
        idim = hv.Dimension("intensity", label="intensity (MS2)")

        matched = base[base["matched"]].copy()
        unmatched = base[~base["matched"]].copy()

        layers = []
        if not self.annotated_only.value and len(unmatched):
            grey = hv.Spikes(unmatched, kdims="mz", vdims=[idim]).opts(
                color=_UNMATCHED_COLOR, line_width=1.0)
            layers.append(grey)
        if len(matched):
            msticks = hv.Spikes(matched, kdims="mz", vdims=[idim, "color"]).opts(
                color="color", line_width=2.2)
            mtips = hv.Scatter(matched, kdims="mz", vdims=["intensity", "color"]).opts(
                color="color", size=6, marker="triangle")
            layers += [msticks, mtips]

        overlay = layers[0]
        for ly in layers[1:]:
            overlay = overlay * ly

        # labels above the matched peaks
        if len(matched):
            labels = hv.Labels(matched, kdims=["mz", "intensity"], vdims="label").opts(
                text_color="color", text_font_size="8pt", yoffset=ymax * 0.03)
            overlay = overlay * labels

        return overlay.opts(
            height=400, responsive=True, ylim=(0, ymax),
            xlabel="m/z", ylabel="intensity",
            tools=["xwheel_zoom", "xpan", "hover"], default_tools=["save", "reset"],
            framewise=True, axiswise=True, title=title)

    def _on_redraw_spectrum(self, event=None):
        """Redraw the spectrum when the 'annotated only' toggle changes."""
        if getattr(self, "_last_spectrum", None) is None:
            return
        self.spectrum_panel[0] = pn.pane.HoloViews(
            self._spectrum_overlay(), sizing_mode="stretch_width")

    def _prepare_xic(self, psm, match):
        """Offer XIC extraction without blocking spectrum selection."""
        self._xic_request = (psm, match)
        self.xic_button.disabled = (
            self.state.id_store is None
            or psm.precursor_mz is None
            or psm.retention_time is None
        )
        self.xic_panel[:] = [pn.pane.Markdown(
            "_XIC not calculated. Click Generate precursor XIC "
            "to extract it for this peptide._"
        )]

    def _on_generate_xic(self, event=None):
        """Run the existing XIC algorithm only when requested."""
        if self._xic_request is None:
            return
        psm, match = self._xic_request
        self._render_xic(psm, match)

    def _render_xic(self, psm, match):
        """Windowed XIC: the PSM's precursor m/z over a narrow RT window — the
        'raw data' context of where this identification elutes."""
        store = self._registry.get(psm.run)
        if store is None or psm.precursor_mz is None or psm.retention_time is None:
            self.xic_panel[:] = []
            return
        rt_s = psm.retention_time * 60.0
        lo, hi = rt_s - self.XIC_RT_WINDOW_S, rt_s + self.XIC_RT_WINDOW_S
        try:
            sub = store.filter_rt(lo, hi)
            xic = sub.xic(psm.precursor_mz, mz_tol=self.XIC_MZ_TOL)   # rt, intensity
            if xic is None or len(xic) == 0:
                self.xic_panel[:] = [pn.pane.Markdown("_No XIC signal in the RT window._")]
                return
            df = pd.DataFrame({"rt": np.asarray(xic)[:, 0], "intensity (XIC)": np.asarray(xic)[:, 1]})
            curve = hv.Curve(df, "rt", "intensity (XIC)").opts(
                height=200, responsive=True, xlabel="RT (s)", ylabel="intensity",
                tools=["xwheel_zoom", "xpan", "hover"], default_tools=["save", "reset"],
                title=f"XIC  m/z {psm.precursor_mz:.4f} ± {self.XIC_MZ_TOL}")
            marker = hv.VLine(rt_s).opts(color="#d62728", line_dash="dashed", line_width=1)
            self.xic_panel[:] = [
                pn.pane.HoloViews(curve * marker, sizing_mode="stretch_width"),
                pn.pane.Markdown("_Precursor extracted-ion chromatogram around the PSM's "
                                 "retention time (dashed line = the PSM's RT)._"),
            ]
        except Exception as e:
            self.xic_panel[:] = [pn.pane.Markdown(f"_XIC unavailable: {e}_")]

    def _render_info(self, psm, match):
        pep = psm.peptidoform
        from .. import ids

        cov = None
        if match.resolved and match.spectrum is not None:
            sp = match.spectrum
            r = ids.match_fragments(
                np.asarray(sp.mz([0])[0], float),
                pep.proforma,
                pep.precursor_charge,
        )
            cov = f"{r['matched_ions']} ions, coverage {r['coverage']}/{r['max_coverage']}"

        rows = [
            f"**Peptide:** `{pep.proforma}`",
            f"**Charge:** {pep.precursor_charge}",
            f"**Precursor m/z:** {psm.precursor_mz:.4f}" if psm.precursor_mz else "",
            f"**RT:** {psm.retention_time:.3f} min" if psm.retention_time is not None else "",
            f"**1/K0:** {psm.ion_mobility:.4f}" if psm.ion_mobility is not None else "**1/K0:** —",
            f"**Match method:** `{match.method}` ({match.n_candidates} candidate(s))",
            f"**Fragments:** {cov}" if cov else "",
        ]
        self.info_panel.object = "  \n".join(r for r in rows if r)

    # ── layout ─────────────────────────────────────────────────────────────
    def view(self):
        from ._browse import browse_field
        inputs = pn.Column(
            pn.pane.Markdown("## Identifications"),
            self.dataset_source,
            browse_field(self.store_path, patterns=["*.zarr"], width=560),
            browse_field(self.txt_dir, pick_folder=True, width=560),
            browse_field(self.fasta_path, patterns=["*.fasta", "*.fa"], width=560),
            pn.Row(self.load_btn),
            self.status,
            sizing_mode="stretch_width")
        return pn.Column(
            inputs,
            pn.layout.Divider(),
            pn.pane.Markdown("### Explore identifications"),
            self.run_selector,
            pn.pane.Markdown(
                "### Proteins - direct PSM support in selected run"
            ),
            pn.pane.Markdown(
                "_Protein scores, counts and coverage percentages shown below "
                "are from the combined MaxQuant proteinGroups.txt results; "
                "they are not recalculated per run._"
            ),
            self.protein_search,
            self.protein_table,
            pn.pane.Markdown("### Sequence coverage"),
            self.accession_selector,
            self.coverage_panel,
            pn.layout.Divider(),
            pn.Row(
                pn.Column(self.peptide_heading,
                          self.peptide_table, self.info_panel,
                          width=660, min_width=660),
                pn.Column(pn.pane.Markdown("### Annotated spectrum"),
                          self.bruker_spectrum_selector,
                          self.bruker_association_note,
                          pn.Row(self.annotated_only, height=34, sizing_mode="stretch_width"),
                          self.spectrum_panel, self.xic_button,
                          self.xic_panel, sizing_mode="stretch_width"),
                sizing_mode="stretch_width"),
            sizing_mode="stretch_width")


# imported lazily by _render_spectrum to keep the heavy ids import off page import
def ids_peak_annotations(mz, intensity, proforma, charge):
    from .. import ids
    return ids.peak_annotations(mz, intensity, proforma, charge)


def _open_spectra_store(path: str):
    """Open the ID page's spectra source: a PTERO Zarr store when the path is a
    ``.zarr`` (or a directory holding a PTERO store), else an mzML/mzXML via
    ``from_mzml``. Both return a ``PteroSpectra``, so the link, annotation and XIC
    code downstream is identical regardless of source."""
    from pathlib import Path
    from ..spectra import open_zarr, from_mzml
    p = Path(path)
    if p.suffix.lower() == ".zarr" or (p.is_dir() and (p / "offsets").exists()):
        return open_zarr(str(p))
    return from_mzml(str(p))