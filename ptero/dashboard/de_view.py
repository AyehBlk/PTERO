"""
Differential-expression page for the PTERO dashboard.

Shares :class:`AppState` with the other pages. Flow: load a MaxQuant
``proteinGroups.txt`` -> assign each sample to a group (auto-suggested from the
file names, fully editable) and pick the reference group -> Run. Groups, the
reference and the contrast are entirely user-supplied, so the page carries no
experiment-specific terms; it works for any two-group comparison.

Run computes the whole suite once (limma / DEqMS / Welch / Wilcoxon across the
complete-case and both imputed tracks) plus the on/off table and both imputed
matrices, so exploring is fast. Adjusting the Test / Track / thresholds and
pressing **Apply** re-renders. Selecting a protein (in either results table)
highlights it on the volcano and heatmap and draws its per-sample boxplot.

The **Imputation analysis** section lets the user judge imputation on their own
data: the observed-vs-imputed distribution, and a table of proteins that are only
testable *because* of imputation (they had missing values) with how many values
were imputed, in which group, and by which mechanism -- so they can decide whether
an imputed track is trustworthy or whether to report complete-case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import panel as pn

from ._state import AppState

pn.extension("tabulator")

_TRACKS = {"complete case": "complete_case",
           "imputed (mechanism)": "imputed_mechanism",
           "imputed (MinProb)": "imputed_minprob"}


def _short(s, maxlen: int = 22) -> str:
    """Gene symbol only (drop trailing accessions), length-capped -- so the tables
    match the plots."""
    s = str(s).split(";")[0].strip()
    return s if len(s) <= maxlen else s[: maxlen - 1] + "\u2026"


class DeView:
    """The differential-expression page."""

    def __init__(self, state: AppState | None = None):
        self.state = state if state is not None else AppState()
        self._qd = None
        self._diag = None
        self._pa = None
        self._results = None          # long table from differential_abundance
        self._primary = None          # current method x track result (indexed by id, has label)
        self._M = None                # matrix for the current track
        self._mcache: dict = {}       # track -> (matrix, was_missing | None)
        self._selected = None

        # ── load ──
        self.pg_path = pn.widgets.TextInput(
            name="MaxQuant proteinGroups.txt",
            placeholder=r"C:\data\combined\txt\proteinGroups.txt", width=460)
        self.quantity = pn.widgets.Select(
            name="Quantity", options={"LFQ": "lfq", "Intensity": "intensity"}, value="lfq", width=140)
        self.load_btn = pn.widgets.Button(name="Load proteins", button_type="primary", width=150)
        self.status = pn.pane.Markdown(
            "**No data loaded.** Enter a MaxQuant `proteinGroups.txt` and Load.")

        # ── groups & contrast ──
        self.group_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["sample", "group"]), show_index=False, height=280, width=380,
            editors={"group": {"type": "input"}}, layout="fit_columns")
        self.ref_group = pn.widgets.Select(name="Reference group (baseline)", options=[], width=220)
        self.on_min = pn.widgets.IntInput(name="on/off: min present in a group", value=4, start=1, width=210)
        self.run_btn = pn.widgets.Button(
            name="Run differential expression", button_type="success", width=250, disabled=True)
        self.group_panel = pn.Column(visible=False, sizing_mode="stretch_width")

        # ── analysis controls (Apply-driven) ──
        self.method = pn.widgets.Select(
            name="Test", options=["limma", "deqms", "welch", "wilcoxon"], value="limma", width=150)
        self.track = pn.widgets.Select(name="Track", options=_TRACKS, value="complete_case", width=190)
        self.alpha = pn.widgets.FloatInput(name="padj <", value=0.05, step=0.01, width=95)
        self.lfc = pn.widgets.FloatInput(name="|log2FC| \u2265", value=1.0, step=0.1, width=105)
        self.heat_top = pn.widgets.IntInput(name="heatmap top", value=30, start=5, end=100, step=5, width=120)
        self.apply_btn = pn.widgets.Button(name="Apply", button_type="primary", width=110)

        # ── outputs ──
        self.volcano_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, height=560)
        self.detmat_pane = pn.pane.Matplotlib(tight=True, height=520)
        self.heat_pane = pn.pane.Matplotlib(tight=True, height=620)
        self.box_pane = pn.pane.Matplotlib(tight=True, height=420)
        self.results_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["label", "log2FC", "p", "padj"]),
            selectable=1, disabled=True, height=360, width=520, pagination="local",
            page_size=15, show_index=False, hidden_columns=["id"], widths={"label": 150})
        self.summary = pn.pane.Markdown("")
        self.results_panel = pn.Column(visible=False, sizing_mode="stretch_width")

        # ── imputation-analysis section ──
        self.impute_summary = pn.pane.Markdown("")
        self.impute_dist_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, height=380)
        self.newhits_note = pn.pane.Markdown("")
        self.newhits_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["label", "log2FC", "p", "padj", "imputed", "group", "mechanism"]),
            selectable=1, disabled=True, height=360, width=620, pagination="local",
            page_size=15, show_index=False, hidden_columns=["id"], widths={"label": 130, "mechanism": 150})
        self.impute_box_pane = pn.pane.Matplotlib(tight=True, height=420)

        # ── wiring ──
        self.load_btn.on_click(self._on_load)
        self.run_btn.on_click(self._on_run)
        self.apply_btn.on_click(self._refresh)
        self.group_table.param.watch(self._refresh_ref_options, "value")
        self.results_table.param.watch(self._on_select_row, "selection")
        self.newhits_table.param.watch(self._on_select_newhit, "selection")

    # ── loading ──────────────────────────────────────────────────────────
    def _on_load(self, event=None):
        from pathlib import Path
        from ..quant import read_sample_names, suggest_groups_by_token

        p = self.pg_path.value.strip()
        if not p:
            self.status.object = "\u26a0\ufe0f Enter a `proteinGroups.txt` path."
            return
        if not Path(p).is_file():
            self.status.object = f"\u274c **Not a file:** `{p}`"
            return
        try:
            names = read_sample_names(p, quantity=self.quantity.value)
        except Exception as e:                                  # noqa: BLE001
            self.status.object = f"\u274c **Could not read sample columns:** {e}"
            return
        if not names:
            self.status.object = ("\u274c No sample columns found for that quantity. "
                                  "If LFQ wasn't enabled in MaxQuant, try Quantity = Intensity.")
            return
        sugg = suggest_groups_by_token(names)
        self.group_table.value = pd.DataFrame(
            {"sample": names, "group": [str(sugg.get(s, "A")) for s in names]})
        self._refresh_ref_options()
        self.group_panel.visible = True
        self.results_panel.visible = False
        self.run_btn.disabled = False
        self.status.object = (f"\u2705 Loaded **{len(names)}** samples. Edit the **group** column if "
                              "needed, pick the reference group, then Run.")

    def _refresh_ref_options(self, event=None):
        df = self.group_table.value
        groups = list(dict.fromkeys(df["group"].astype(str))) if len(df) else []
        self.ref_group.options = groups
        if groups and self.ref_group.value not in groups:
            self.ref_group.value = groups[0]

    # ── running ──────────────────────────────────────────────────────────
    def _on_run(self, event=None):
        from ..quant import (annotate_contaminants, classify_missingness, complete_case_matrix,
                             differential_abundance, impute_matrix, presence_absence)

        gdf = self.group_table.value
        groups = {str(s): str(g) for s, g in zip(gdf["sample"], gdf["group"])}
        uniq = list(dict.fromkeys(groups.values()))
        if len(uniq) != 2:
            self.status.object = (f"\u274c This test needs exactly **2** groups; you have {uniq}. "
                                  "Edit the group column so there are two.")
            return
        ref = self.ref_group.value
        if ref not in uniq:
            self.status.object = "\u274c Pick a reference group that is one of your two groups."
            return
        self.status.object = "\u23f3 Running \u2014 loading, classifying, testing, imputing\u2026 (progress prints in the terminal)."
        try:
            qd = self._load_qd(groups)
            cc = annotate_contaminants(qd.proteins)["contaminant_class"]
            diag = classify_missingness(qd, contaminant_class=cc, on_min=int(self.on_min.value))
            long = differential_abundance(
                qd, diag, ref_group=ref, methods=("limma", "deqms", "welch", "wilcoxon"))
            pa = presence_absence(qd, diag)
            mcache = {"complete_case": (complete_case_matrix(qd, diag), None)}
            imp_mech, miss = impute_matrix(qd, diag, method="mechanism")
            imp_mp, _ = impute_matrix(qd, diag, method="minprob")
            mcache["imputed_mechanism"] = (imp_mech, miss)
            mcache["imputed_minprob"] = (imp_mp, miss)      # same missing positions, different fills
        except Exception as e:                                  # noqa: BLE001
            self.status.object = f"\u274c **Analysis failed:** {e}"
            return

        self._qd, self._diag, self._pa, self._results, self._mcache = qd, diag, pa, long, mcache
        self.state.set("quant", qd)
        non_ref = next(g for g in uniq if g != ref)
        self.detmat_pane.object = self._make_detmat()
        self.impute_summary.object = self._impute_summary_md()
        self.impute_dist_pane.object = self._render_impute_dist()
        self.results_panel.visible = True
        self.status.object = (f"\u2705 Done. **{qd.lfq.shape[0]}** proteins \u00b7 groups {uniq} "
                              f"(reference **{ref}**, positive log2FC = higher in {non_ref}) \u00b7 "
                              f"on/off call-ready: **{int(pa['call_ready'].sum())}**.")
        self._refresh()

    def _load_qd(self, groups):
        from ..quant import QuantData, load_quant
        qd = load_quant(self.pg_path.value.strip(), groups=groups,
                        drop_decoys_contaminants=False, quantity=self.quantity.value)
        keep = ~qd.proteins["is_reverse"].to_numpy()
        return QuantData(
            lfq=qd.lfq[keep], intensity=qd.intensity[keep], id_type=qd.id_type[keep],
            proteins=qd.proteins[keep], samples=qd.samples, groups=qd.groups, source=qd.source,
            mq_version=getattr(qd, "mq_version", None),
            n_protein_groups_raw=getattr(qd, "n_protein_groups_raw", None),
            n_filtered=getattr(qd, "n_filtered", None))

    # ── current selection helpers ────────────────────────────────────────
    def _current(self):
        m, t = self.method.value, self.track.value
        sub = self._results[(self._results["method"] == m) & (self._results["track"] == t)].copy()
        sub = sub.set_index("id")
        sub.attrs["method"] = m
        sub.attrs["contrast"] = self._results.attrs.get("contrast", "")
        return sub, t

    def _make_detmat(self):
        from ..quant import detection_matrix
        return detection_matrix(self._qd, self._pa, which="call_ready",
                                ref_group=self.ref_group.value, title="On/off detection")

    def _render_volcano(self, highlight=None):
        from ..quant import volcano
        self.volcano_pane.object = volcano(
            self._primary, alpha=float(self.alpha.value), lfc_thresh=float(self.lfc.value),
            label_top=12, highlight=highlight,
            title=f"Volcano  ({self.method.value}, {self.track.value.replace('_', ' ')})")

    def _render_heatmap(self, highlight=None):
        from ..quant import heatmap_top
        self.heat_pane.object = heatmap_top(
            self._primary, self._M, self._qd.groups, top=int(self.heat_top.value), by="p",
            ref_group=self.ref_group.value, highlight=highlight,
            title=f"Top {int(self.heat_top.value)} proteins  ({self.method.value})")

    # ── main refresh (Apply) ─────────────────────────────────────────────
    def _refresh(self, event=None):
        if self._results is None:
            return
        sub, track = self._current()
        self._primary = sub
        self._M = self._mcache[track][0]
        self._selected = None
        self.box_pane.object = None

        self._render_volcano()
        self._render_heatmap()

        a, l = float(self.alpha.value), float(self.lfc.value)
        tab = sub.reset_index()[["id", "label", "log2FC", "p", "padj"]].sort_values("p").reset_index(drop=True)
        tab["label"] = tab["label"].map(_short)
        tab["log2FC"] = tab["log2FC"].round(3)
        tab["p"] = tab["p"].map(lambda v: float(f"{v:.3g}"))
        tab["padj"] = tab["padj"].round(4)
        self.results_table.value = tab
        nsig = int((sub["padj"] < a).sum())
        self.summary.object = (f"**{nsig}** significant at padj < {a:g} and |log2FC| \u2265 {l:g}"
                               f"  \u00b7  {len(sub)} proteins tested  \u00b7  contrast "
                               f"`{sub.attrs['contrast']}`")
        self._build_newhits(track)

    # ── selection -> boxplot + linked highlight ──────────────────────────
    def _select(self, pid, box_pane):
        from ..quant import plot_de_boxplot
        if pid is None or self._M is None or pid not in self._M.index:
            return
        self._selected = pid
        mask = self._mcache.get(self.track.value, (None, None))[1]
        imputed_mask = mask.loc[pid] if (mask is not None and pid in mask.index) else None
        box_pane.object = plot_de_boxplot(
            self._M, pid, self._qd.groups, de=self._primary, ref_group=self.ref_group.value,
            imputed_mask=imputed_mask)
        self._render_volcano(highlight=pid)
        self._render_heatmap(highlight=pid)

    def _on_select_row(self, event=None):
        sel = self.results_table.selection
        if not sel:
            return
        try:
            pid = self.results_table.value.iloc[sel[0]]["id"]
        except Exception:                                      # noqa: BLE001
            return
        self._select(pid, self.box_pane)

    def _on_select_newhit(self, event=None):
        sel = self.newhits_table.selection
        if not sel:
            return
        try:
            pid = self.newhits_table.value.iloc[sel[0]]["id"]
        except Exception:                                      # noqa: BLE001
            return
        self._select(pid, self.impute_box_pane)

    # ── imputation analysis ──────────────────────────────────────────────
    def _impute_summary_md(self):
        from ..quant import HIGH_CONFIDENCE, LOW_ABUNDANCE_MNAR, MAR_SCATTERED
        mech = self._diag["mechanism"]
        ok = self._diag["contaminant_class"] == "ok"
        n_hc = int(((mech == HIGH_CONFIDENCE) & ok).sum())
        n_mar = int(((mech == MAR_SCATTERED) & ok).sum())
        n_mnar = int(((mech == LOW_ABUNDANCE_MNAR) & ok).sum())
        _, miss = self._mcache["imputed_mechanism"]
        n_cells = int(miss.to_numpy().sum())
        return (f"**Imputable proteins:** {n_hc + n_mar + n_mnar}  "
                f"(high-confidence {n_hc} + MAR {n_mar} \u2192 within-group kNN,  "
                f"low-abundance MNAR {n_mnar} \u2192 QRILC)  \u00b7  **{n_cells}** values imputed. "
                f"On/off, too-sparse and undetected proteins are never imputed.")

    def _render_impute_dist(self):
        from ..quant import log2_median_normalize
        from matplotlib.figure import Figure
        from ptero.quant._deviz import _style_axes

        M = log2_median_normalize(self._qd.lfq)
        obs = M.to_numpy(dtype=float)
        obs = obs[~np.isnan(obs)]
        mechM, miss = self._mcache["imputed_mechanism"]
        mpM = self._mcache["imputed_minprob"][0]
        mask = miss.to_numpy()
        mech_v = mechM.to_numpy(dtype=float)[mask]
        mp_v = mpM.to_numpy(dtype=float)[mask]
        lo = float(np.nanmin([obs.min(), mech_v.min() if mech_v.size else obs.min(),
                              mp_v.min() if mp_v.size else obs.min()]))
        bins = np.linspace(lo, float(obs.max()), 60)
        fig = Figure(figsize=(8.6, 3.7), layout="constrained")
        ax = fig.add_subplot(111)
        ax.hist(obs, bins, density=True, color="#cccccc", alpha=0.7, label="observed")
        if mech_v.size:
            ax.hist(mech_v, bins, density=True, histtype="step", color="#1f4e79",
                    linewidth=1.9, label="imputed (mechanism)")
        if mp_v.size:
            ax.hist(mp_v, bins, density=True, histtype="step", color="#d97706",
                    linewidth=1.9, label="imputed (MinProb)")
        ax.set_xlabel("normalized log2 intensity")
        ax.set_ylabel("density")
        ax.set_title("What imputation fills in \u2014 observed vs imputed values")
        _style_axes(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        return fig

    def _build_newhits(self, track):
        cols = ["id", "label", "log2FC", "p", "padj", "imputed", "group", "mechanism"]
        if track == "complete_case":
            self.newhits_note.object = ("_You're on the **complete-case** track (no imputation). "
                                        "Switch **Track** to an imputed track and press **Apply** to "
                                        "see proteins recovered by imputation._")
            self.newhits_table.value = pd.DataFrame(columns=cols)
            self.impute_box_pane.object = None
            return
        cc_ids = set(self._mcache["complete_case"][0].index)
        sub = self._primary
        new = sub[~sub.index.isin(cc_ids)]
        _, mask = self._mcache[track]
        mech = self._diag["mechanism"]
        members = self._qd.group_members()
        a = float(self.alpha.value)
        rows = []
        for pid in new.index:
            mrow = mask.loc[pid] if pid in mask.index else None
            n_imp = int(mrow.sum()) if mrow is not None else 0
            gtxt = ""
            if mrow is not None:
                parts = []
                for g, scols in members.items():
                    c = int(sum(bool(mrow[s]) for s in scols if s in mrow.index))
                    if c:
                        parts.append(f"{g}:{c}")
                gtxt = "  ".join(parts)
            rows.append({"id": pid, "label": _short(sub.loc[pid, "label"]),
                         "log2FC": round(float(sub.loc[pid, "log2FC"]), 3),
                         "p": float(f"{sub.loc[pid, 'p']:.3g}"),
                         "padj": round(float(sub.loc[pid, "padj"]), 4),
                         "imputed": n_imp, "group": gtxt, "mechanism": str(mech.get(pid, ""))})
        df = pd.DataFrame(rows, columns=cols)
        if len(df):
            df = df.sort_values("p").reset_index(drop=True)
        self.newhits_table.value = df
        n_sig = int((df["padj"] < a).sum()) if len(df) else 0
        self.newhits_note.object = (
            f"**{len(df)}** proteins are testable only because of imputation (they had missing "
            f"values) on this track; **{n_sig}** reach padj < {a:g}. Ranked by p \u2014 click a row to "
            f"see its boxplot with **imputed values drawn as hollow rings**, so you can judge whether "
            f"the signal is real or an imputation artifact.")

    # ── layout ───────────────────────────────────────────────────────────
    def view(self):
        from ._browse import browse_field

        load_block = pn.Column(
            pn.pane.Markdown("## Differential expression"),
            browse_field(self.pg_path, patterns=["*.txt"], width=560),
            pn.Row(self.quantity, pn.Column(pn.Spacer(height=18), self.load_btn)),
            self.status,
            sizing_mode="stretch_width")

        self.group_panel[:] = [
            pn.layout.Divider(),
            pn.pane.Markdown("### Groups & contrast"),
            pn.pane.Markdown("_Groups are auto-suggested from the sample names \u2014 edit the "
                             "**group** column for your design, pick the reference (baseline) "
                             "group, then Run. Positive log2FC means higher in the other group._"),
            pn.Row(self.group_table,
                   pn.Column(self.ref_group, self.on_min, pn.Spacer(height=8), self.run_btn))]

        controls = pn.Row(self.method, self.track, self.alpha, self.lfc, self.heat_top,
                          pn.Column(pn.Spacer(height=18), self.apply_btn),
                          sizing_mode="stretch_width")

        impute_card = pn.Card(
            pn.pane.Markdown("Judge imputation on **your** data before trusting an imputed track. "
                             "The distribution shows what each strategy fills in; the table lists "
                             "proteins that are only testable because of imputation."),
            self.impute_summary,
            self.impute_dist_pane,
            pn.layout.Divider(),
            self.newhits_note,
            pn.Row(self.newhits_table,
                   pn.Column(pn.pane.Markdown("**Recovered-protein detail**"), self.impute_box_pane)),
            title="Imputation analysis  \u2014  is imputing a good idea for your data?",
            collapsed=True, sizing_mode="stretch_width")

        self.results_panel[:] = [
            pn.layout.Divider(),
            controls,
            self.summary,
            pn.Row(
                pn.Column(pn.pane.Markdown("### Volcano"), self.volcano_pane, sizing_mode="stretch_width"),
                pn.Column(pn.pane.Markdown("### On/off detection"), self.detmat_pane),
                sizing_mode="stretch_width"),
            pn.layout.Divider(),
            pn.Row(
                pn.Column(pn.pane.Markdown("### Top proteins"), self.heat_pane),
                pn.Column(pn.pane.Markdown("### Results \u2014 select a row for its boxplot"),
                          self.results_table,
                          pn.pane.Markdown("### Protein detail"), self.box_pane)),
            pn.layout.Divider(),
            impute_card,
        ]

        return pn.Column(load_block, self.group_panel, self.results_panel,
                         sizing_mode="stretch_width")