"""
ID-annotation page for the PTERO dashboard.

Workflow (input is **mzML**, not Zarr — see note):

1. Load a vendor-peak-picked **mzML** as the spectra store, plus a search-engine
   ``msms.txt`` (PSMs). Optionally ``accumulatedMsmsScans.txt`` (per-PSM 1/K0)
   and ``summary.txt`` (the search's fixed/variable mods).
2. PSMs are listed in a selectable table.
3. Selecting a PSM locates its MS2 spectrum in the store by physical coordinates
   (RT + precursor m/z + ion mobility), annotates the b/y ions, and shows:
   * the **annotated MS2 stick plot** -- matched b/y peaks coloured by series and
     labelled, the rest grey (only the genuine, one-to-one fragment matches are
     labelled, not the isotopolog/multi-charge over-count);
   * an **XIC context** -- the precursor m/z extracted over a narrow RT window
     around the PSM (the "raw data" view of where this identification elutes);
   * the PSM's coordinates and the fragment-coverage score.

Why mzML and not the Zarr store: a raw ``.d``→Zarr store holds un-deisotoped
peaks, whose fragment m/z do not match a search engine's monoisotopic ``Masses``.
The vendor-peak-picked mzML does. (When PTERO grows its own MS/MS processing
step, the Zarr store could feed this too.)
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
        self._last_spectrum = None   # (annotated DataFrame, title) for redraws
        self._iddata = None      # IdentificationData (protein+peptide tables)
        self._psm_by_seq = {}    # plain peptide sequence -> best PSM (for lookup)
        self._pep_df = None      # current peptide table (for the selected protein)
        self._fasta = None       # ProteinSequences (Stage 2), or None if not loaded
        self._coverage_mapping = None   # current PeptideMapping (for peptide highlight)
        self._mod_sites = None       # ModificationSites (Stage 3), or None
        self._coverage_sites = []    # sites for the currently shown protein

        # ── inputs ──
        self.mzml_path = pn.widgets.TextInput(
            name="mzML store (vendor peak-picked)", placeholder=r"C:\data\run.mzML", width=460)
        self.txt_dir = pn.widgets.TextInput(
            name="MaxQuant txt folder",
            placeholder=r"C:\data\combined\txt", width=460)
        self.fasta_path = pn.widgets.TextInput(
            name="FASTA (the database used in the search; for the coverage map)",
            placeholder=r"C:\data\human.fasta", width=460)
        self.load_btn = pn.widgets.Button(name="Load identifications", button_type="primary", width=170)

        self.status = pn.pane.Markdown(
            "**No identifications loaded.** Enter the mzML store and the MaxQuant "
            "`txt` folder, then Load.")

        # ── protein table (top, selectable) ──
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
        self.info_panel = pn.pane.Markdown("")

        # display toggle: hide unmatched (grey) peaks to see just the b/y ladder
        self.annotated_only = pn.widgets.Checkbox(
            name="Show annotated peaks only", value=False, width=220)

        # ── wiring ──
        self.load_btn.on_click(self._on_load)
        self.protein_table.param.watch(self._on_select_protein, "selection")
        self.peptide_table.param.watch(self._on_select_peptide, "selection")
        self.annotated_only.param.watch(self._on_redraw_spectrum, "value")

    # ── loading ──────────────────────────────────────────────────────────
    async def _on_load(self, event=None):
        from pathlib import Path
        mzml = self.mzml_path.value.strip()
        txt_dir = self.txt_dir.value.strip()
        if not (mzml and txt_dir):
            self.status.object = "⚠️ Enter the mzML store and the MaxQuant `txt` folder."
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
            + ". Loading mzML metadata, PSMs and protein/peptide tables…")
        self.spectrum_panel.loading = True
        await asyncio.sleep(0.05)
        try:
            from ..spectra import from_mzml
            from .. import ids
            from ._ids_data import IdentificationData

            store = from_mzml(mzml)                       # lazy; metadata pass only
            psms = ids.read_psms(msms)
            n_mob = ids.attach_mobility_from_maxquant(psms, accum) if accum else 0
            n_mod = ids.convert_psm_modifications(psms, summary_path=summary)

            run_id = psms[0].run if len(psms) else "run"
            reg = ids.RunRegistry()
            reg.register(run_id, store)
            self._registry = reg

            by_seq = {}
            for p in psms:
                if p.peptidoform is None:
                    continue
                seq = p.peptidoform.sequence
                sc = getattr(p, "score", None) or float("-inf")
                if seq not in by_seq or sc > (getattr(by_seq[seq], "score", None) or float("-inf")):
                    by_seq[seq] = p
            self._psm_by_seq = by_seq

            self._iddata = IdentificationData(proteins, peptides)

            # modification sites for the sequence view (Stage 3): read all
            # *Sites.txt in the folder (oxidation, phospho, …). Authoritative
            # protein positions + localisation, generic over modification type.
            from ._mods_sites import ModificationSites
            self._mod_sites = ModificationSites(str(folder))

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

            self.protein_table.value = self._iddata.protein_table.drop(columns="_gid")
            self.peptide_table.value = self.peptide_table.value.iloc[0:0]
            n_groups = len(self._iddata.protein_table)
            fasta_note = (f" FASTA: {n_fasta} sequences (coverage map on)."
                          if self._fasta else " No FASTA (coverage map off).")
            self.status.object = (
                f"✅ Loaded {len(psms)} PSMs, {n_groups} protein groups, "
                f"mzML store ({len(store)} spectra). 1/K0 on {n_mob}; mods on {n_mod}.{fasta_note} "
                f"Select a protein, then a peptide."
                + (f"  _(optional not found: {', '.join(opt_note)})_" if opt_note else ""))
        except Exception as e:
            self.status.object = f"❌ **Load failed:** {e}"
        finally:
            self.spectrum_panel.loading = False

    # ── selection: protein -> peptides -> spectrum ────────────────────────
    async def _on_select_protein(self, event=None):
        sel = self.protein_table.selection
        if not sel or self._iddata is None:
            return
        row = int(sel[0])
        prow = self._iddata.protein_table.iloc[row]
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
            self._pep_df = self._iddata.peptides_for_group(gid)
            self.peptide_table.value = self._pep_df.drop(columns="_seq").reset_index(drop=True)
            self._render_coverage(prow)        # coverage map (Stage 2) if FASTA loaded
        finally:
            self.peptide_table.loading = False
            self.coverage_panel.loading = False

    def _render_coverage(self, protein_row):
        """Draw the sequence coverage map for the selected protein group."""
        self._coverage_mapping = None        # reset; set below only on success
        if self._fasta is None:
            self.coverage_panel[:] = [pn.pane.Markdown(
                "_Load a FASTA to see the sequence coverage map._")]
            return
        from ._fasta import map_peptides
        # leading majority protein id for this group
        majority = protein_row.get("majority_ids", "") or ""
        acc = majority.split(";")[0] if majority else ""
        seq = self._fasta.get(acc) if acc else None
        if not seq:
            self.coverage_panel[:] = [pn.pane.Markdown(
                f"⚠️ Protein `{acc or '?'}` not found in the FASTA — "
                "is it the same database used in the search?")]
            return
        peps = self._pep_df[["peptide", "unique"]].to_dict("records")
        mapping = map_peptides(acc, seq, peps)
        self._coverage_mapping = mapping        # keep for peptide-click highlight
        # modification sites for this group (Stage 3), positions valid for this protein
        gid = protein_row["_gid"]
        self._coverage_sites = (self._mod_sites.for_group(gid)
                                if self._mod_sites is not None else [])
        self._show_coverage(mapping, highlight_peptide=None)

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
                     f"**hollow** = below 0.75 (uncertain). Hover a marker for details._")
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

    async def _on_select_peptide(self, event=None):
        sel = self.peptide_table.selection
        if not sel or self._pep_df is None:
            return
        row = int(sel[0])
        seq = self._pep_df.iloc[row]["_seq"]
        # outline this peptide on the coverage sequence (if a map is shown)
        if getattr(self, "_coverage_mapping", None) is not None:
            self._show_coverage(self._coverage_mapping, highlight_peptide=seq)
        psm = self._psm_by_seq.get(seq)
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
            match = ids.spectrum_for_psm(psm, self._registry)
            self._match = match
            self._render_spectrum(psm, match)
            self._render_xic(psm, match)
            self._render_info(psm, match)
        except Exception as e:
            self.spectrum_panel[:] = [pn.pane.Markdown(f"❌ **Annotation error:** {e}")]
        finally:
            self.spectrum_panel.loading = False

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

    def _render_xic(self, psm, match):
        """Windowed XIC: the PSM's precursor m/z over a narrow RT window — the
        'raw data' context of where this identification elutes."""
        store = self.state.id_store
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
            r = ids._link._match_fragments(
                np.asarray(sp.mz([0])[0], float), pep.proforma, pep.precursor_charge)
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
            browse_field(self.mzml_path, patterns=["*.mzML"], width=560),
            browse_field(self.txt_dir, pick_folder=True, width=560),
            browse_field(self.fasta_path, patterns=["*.fasta", "*.fa"], width=560),
            pn.Row(self.load_btn),
            self.status,
            sizing_mode="stretch_width")
        return pn.Column(
            inputs,
            pn.layout.Divider(),
            pn.pane.Markdown("### Proteins"),
            self.protein_table,
            pn.pane.Markdown("### Sequence coverage"),
            self.coverage_panel,
            pn.layout.Divider(),
            pn.Row(
                pn.Column(pn.pane.Markdown("### Peptides"),
                          self.peptide_table, self.info_panel, width=580),
                pn.Column(pn.pane.Markdown("### Annotated spectrum"),
                          pn.Row(self.annotated_only, height=34, sizing_mode="stretch_width"),
                          self.spectrum_panel, self.xic_panel, sizing_mode="stretch_width"),
                sizing_mode="stretch_width"),
            sizing_mode="stretch_width")


# imported lazily by _render_spectrum to keep the heavy ids import off page import
def ids_peak_annotations(mz, intensity, proforma, charge):
    from .. import ids
    return ids.peak_annotations(mz, intensity, proforma, charge)