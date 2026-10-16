"""Construction + wiring tests for the dashboard Differential-Expression page.

Like the other dashboard tests these build the page and exercise its logic without
a live server. The DE statistics/plots themselves are covered by test_de_suite and
test_deviz; here we check the page constructs, lays out, auto-suggests groups, and
guards the two-group requirement.
"""
import pandas as pd
import panel as pn
import pytest

from ptero.dashboard._state import AppState
from ptero.dashboard.de_view import DeView


def test_constructs():
    v = DeView(AppState())
    assert v.method.value == "limma"
    assert list(v.track.options.values()) == ["complete_case", "imputed_mechanism", "imputed_minprob"]
    assert v.run_btn.disabled            # stays disabled until proteins are loaded


def test_view_builds():
    assert isinstance(DeView(AppState()).view(), pn.Column)


def test_group_suggest_populates_reference():
    v = DeView(AppState())
    v.group_table.value = pd.DataFrame(
        {"sample": ["a1", "a2", "b1", "b2"], "group": ["treated", "treated", "control", "control"]})
    v._refresh_ref_options()
    assert set(v.ref_group.options) == {"treated", "control"}
    assert v.ref_group.value in {"treated", "control"}


def test_run_requires_two_groups():
    v = DeView(AppState())
    v.group_table.value = pd.DataFrame(
        {"sample": ["a", "b", "c"], "group": ["x", "y", "z"]})   # three groups
    v.ref_group.value = "x"
    v._on_run()                                                  # must refuse, not crash
    assert "2" in v.status.object and "group" in v.status.object.lower()
    assert v.results_panel.visible is False


def test_appstate_has_quant_field():
    assert AppState().quant is None


def test_has_apply_and_imputation_widgets():
    v = DeView(AppState())
    assert v.apply_btn.label == "Apply" or v.apply_btn.name == "Apply"
    # the imputation-analysis widgets exist
    assert v.newhits_table is not None and v.impute_box_pane is not None
    assert v.impute_dist_pane is not None


def test_controls_do_not_autorefresh():
    # changing a control must NOT trigger a render on its own (Apply-driven);
    # with no results loaded, _refresh is a no-op and nothing errors
    v = DeView(AppState())
    v.method.value = "deqms"
    v.track.value = "imputed_mechanism"
    v.alpha.value = 0.1
    assert v.results_panel.visible is False   # still nothing rendered without Run/Apply

def test_summary_significance_requires_both_thresholds(monkeypatch):
    import numpy as np

    v = DeView(AppState())
    v.alpha.value = 0.05
    v.lfc.value = 1.0

    data = pd.DataFrame({
        "id": [1, 2, 3],
        "label": ["A", "B", "C"],
        "method": ["limma"] * 3,
        "track": ["complete_case"] * 3,
        "log2FC": [0.3, 1.5, 2.0],
        "p": [0.001, 0.002, 0.2],
        "padj": [0.01, 0.02, 0.3],
    })
    data.attrs["contrast"] = "treated_vs_control"
    v._results = data
    v._mcache = {
        "complete_case": (
            pd.DataFrame({"s1": [1.0, 2.0, 3.0]}, index=[1, 2, 3]),
            None,
        )
    }

    monkeypatch.setattr(v, "_render_volcano", lambda **kwargs: None)
    monkeypatch.setattr(v, "_render_heatmap", lambda **kwargs: None)
    monkeypatch.setattr(v, "_build_newhits", lambda track: None)

    v._refresh()

    assert "**1** significant" in v.summary.object
    assert "|log2FC|" in v.summary.object

    v.lfc.value = 0.0
    v._refresh()
    assert "**2** significant" in v.summary.object

def test_analysis_input_changes_invalidate_results():
    def seeded():
        state = AppState()
        v = DeView(state)
        v.group_table.value = pd.DataFrame({
            "sample": ["a1", "a2", "b1", "b2"],
            "group": ["A", "A", "B", "B"],
        })
        v._qd = object()
        v._results = pd.DataFrame({"p": [0.01]})
        v._mcache = {"complete_case": (pd.DataFrame(), None)}
        v.results_panel.visible = True
        state.set("quant", v._qd)
        return v, state

    changes = [
        lambda v: setattr(v.ref_group, "value", "B"),
        lambda v: setattr(v.on_min, "value", 3),
        lambda v: setattr(v.pg_path, "value", "another_proteinGroups.txt"),
        lambda v: setattr(v.quantity, "value", "intensity"),
        lambda v: setattr(v.group_table, "value", pd.DataFrame({
            "sample": ["a1", "a2", "b1", "b2"],
            "group": ["A", "B", "B", "B"],
        })),
    ]

    for change in changes:
        v, state = seeded()
        change(v)
        assert v._results is None
        assert v._qd is None
        assert v._mcache == {}
        assert state.quant is None
        assert v.results_panel.visible is False
        assert "Run differential abundance again" in v.status.object
        v._refresh()
        assert v._results is None


def test_display_controls_preserve_analysis():
    v = DeView(AppState())
    result = pd.DataFrame({"p": [0.01]})
    v._results = result
    v.method.value = "deqms"
    v.track.value = "imputed_mechanism"
    v.alpha.value = 0.1
    v.lfc.value = 2.0
    v.heat_top.value = 20
    assert v._results is result


def test_load_samples_without_group_tokens(tmp_path):
    path = tmp_path / "proteinGroups.txt"
    pd.DataFrame(columns=[
        "id", "LFQ intensity A", "LFQ intensity C"
    ]).to_csv(path, sep="\t", index=False)

    v = DeView(AppState())
    v.pg_path.value = str(path)
    v._on_load()

    assert v.group_panel.visible is True
    assert v.group_table.value["sample"].tolist() == ["A", "C"]
    assert v.group_table.value["group"].tolist() == ["", ""]
    assert "assign a group" in v.status.object.lower()


@pytest.mark.parametrize(
    "samples, labels, allowed",
    [
        (["A", "C"], ["A", "C"], False),
        (["A1", "C1", "C2", "C3"], ["A", "C", "C", "C"], False),
        (["A1", "A2", "C1", "C2"], ["A", "A", "C", "C"], True),
    ],
)
def test_minimum_group_sizes(monkeypatch, samples, labels, allowed):
    v = DeView(AppState())
    v.group_table.value = pd.DataFrame({
        "sample": samples,
        "group": labels,
    })

    called = []
    def fake_load(groups):
        called.append(groups)
        raise RuntimeError("Reached data loading")

    monkeypatch.setattr(v, "_load_qd", fake_load)
    v._on_run()

    if allowed:
        assert len(called) == 1
        assert "Reached data loading" in v.status.object
    else:
        assert called == []
        assert "at least 2 independent" in v.status.object


def test_blank_group_assignment_blocks_analysis(monkeypatch):
    v = DeView(AppState())
    v.group_table.value = pd.DataFrame({
        "sample": ["A1", "A2", "C1", "C2"],
        "group": ["A", "A", "", "C"],
    })

    called = []
    monkeypatch.setattr(v, "_load_qd", lambda groups: called.append(groups))
    v._on_run()

    assert called == []
    assert "non-empty group" in v.status.object


def test_imputed_track_summary_has_interpretation_caution(monkeypatch):
    import pandas as pd
    from ptero.dashboard.de_view import DeView
    from ptero.dashboard._state import AppState

    view = DeView(AppState())
    rows = pd.DataFrame({
        "id": [101],
        "label": ["Example"],
        "log2FC": [1.5],
        "p": [0.001],
        "padj": [0.01],
    }).set_index("id")
    rows.attrs["contrast"] = "PS_vs_WT"

    view._results = object()
    view._mcache = {
        "complete_case": (pd.DataFrame(index=[101]), None),
        "imputed_mechanism": (pd.DataFrame(index=[101]), None),
    }

    monkeypatch.setattr(view, "_current", lambda: (rows, "imputed_mechanism"))
    monkeypatch.setattr(view, "_render_volcano", lambda: None)
    monkeypatch.setattr(view, "_render_heatmap", lambda: None)
    monkeypatch.setattr(view, "_build_newhits", lambda track: None)

    view._refresh()
    assert "Interpretation caution" in view.summary.object
    assert "imputation uncertainty" in view.summary.object

    monkeypatch.setattr(view, "_current", lambda: (rows, "complete_case"))
    view._refresh()
    assert "Interpretation caution" not in view.summary.object


def test_recovered_protein_count_uses_padj_and_fold_change():
    import pandas as pd
    from ptero.dashboard.de_view import DeView
    from ptero.dashboard._state import AppState

    view = DeView(AppState())
    view.alpha.value = 0.05
    view.lfc.value = 1.0

    primary = pd.DataFrame({
        "label": ["Small effect", "Large effect", "Not significant"],
        "log2FC": [0.4, 1.5, 2.0],
        "p": [0.001, 0.002, 0.2],
        "padj": [0.01, 0.02, 0.3],
    }, index=pd.Index([201, 202, 203], name="id"))

    view._primary = primary
    view._diag = pd.DataFrame({
        "mechanism": ["MAR_scattered"] * 3,
    }, index=primary.index)
    view._mcache = {
        "complete_case": (pd.DataFrame(index=[]), None),
        "imputed_mechanism": (
            pd.DataFrame(index=primary.index),
            pd.DataFrame(
                False, index=primary.index, columns=["PS_1", "WT_1"]
            ),
        ),
    }

    class FakeQuant:
        def group_members(self):
            return {"PS": ["PS_1"], "WT": ["WT_1"]}

    view._qd = FakeQuant()
    view._build_newhits("imputed_mechanism")

    assert len(view.newhits_table.value) == 3
    assert "**1** meet padj < 0.05 and |log2FC| >= 1" in (
        view.newhits_note.object
    )
