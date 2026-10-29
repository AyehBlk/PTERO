"""mzML converter verified-routing contracts."""

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ptero/dashboard/convert.py"
)


def _function():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_convert_mzml"
    ]
    assert len(matches) == 1
    return matches[0]


def test_verified_xml_routing_is_used():
    text = ast.unparse(_function())
    assert "_mzml_xml_peak_mobility_semantics" in text
    assert "_mzml_route_peak_semantics" in text


def test_selected_native_ids_are_used():
    text = ast.unparse(_function())
    assert "native_spectrum_id" in text
    assert "selected_ids" in text


def test_backend_inspection_does_not_use_private_wrapper():
    text = ast.unparse(_function())
    assert "inspection_reader.get_by_id" in text
    assert "sp._backend" not in text


def test_writer_declares_verified_peak_semantics():
    node = _function()
    writers = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "ZarrStreamWriter"
    ]
    assert len(writers) == 1
    keywords = {
        kw.arg for kw in writers[0].keywords
    }
    assert "peak_ion_mobility_type" in keywords
    assert "peak_ion_mobility_unit" in keywords


def test_3d_mobility_loss_is_rejected():
    text = ast.unparse(_function())
    assert "any_selected_mobility" in text
    assert "not has_4d" in text


def test_spectrum_semantics_remain_undeclared():
    node = _function()
    writers = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "ZarrStreamWriter"
    ]
    keywords = {
        kw.arg for kw in writers[0].keywords
    }
    assert "spectrum_ion_mobility_type" not in keywords
    assert "spectrum_ion_mobility_unit" not in keywords
