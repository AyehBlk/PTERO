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