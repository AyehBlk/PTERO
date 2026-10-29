"""Regression tests for source-aware Bruker mobility declarations."""

import ast
import inspect

from ptero.dashboard import convert as converter_module


EXPECTED = {
    "peak_ion_mobility_type": "inverse_reduced_ion_mobility",
    "peak_ion_mobility_unit": "V.s/cm2",
    "spectrum_ion_mobility_type": "inverse_reduced_ion_mobility",
    "spectrum_ion_mobility_unit": "V.s/cm2",
}


def _writer_keywords(function):
    tree = ast.parse(inspect.getsource(function))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ZarrStreamWriter"
    ]
    assert len(calls) == 1
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[0].keywords
        if keyword.arg in EXPECTED
    }


def test_bruker_dda_declares_inverse_reduced_mobility():
    assert _writer_keywords(converter_module._convert_bruker) == EXPECTED


def test_bruker_dia_declares_inverse_reduced_mobility():
    assert _writer_keywords(converter_module._convert_bruker_dia) == EXPECTED


def test_mzml_does_not_guess_mobility_semantics():
    """mzML declarations must come from verified routing."""
    import inspect

    text = inspect.getsource(
        converter_module._convert_mzml
    )
    assert "_mzml_xml_peak_mobility_semantics" in text
    assert "_mzml_route_peak_semantics" in text
    assert "peak_ion_mobility_type" in text
    assert "peak_ion_mobility_unit" in text
    assert "spectrum_ion_mobility_type" not in text
    assert "spectrum_ion_mobility_unit" not in text
