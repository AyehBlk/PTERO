"""
Path fields for the dashboard. The in-browser file picker was removed (it was
non-functional in use); the user types or pastes the full path into the box.
Nothing is uploaded -- only the path string is used, so multi-GB files are fine.

``browse_field(text_input, ...)`` returns the text box wrapped for layout. The
keyword arguments (patterns, pick_folder, width, default_dir) are accepted and
ignored so existing call sites keep working unchanged.
"""
from __future__ import annotations

from typing import Sequence

import panel as pn


def browse_field(text_input: "pn.widgets.TextInput",
                 *, default_dir: str | None = None,
                 patterns: Sequence[str] = ("*",),
                 pick_folder: bool = False,
                 width: int = 720):
    """Return *text_input* wrapped for the layout (type or paste the full path)."""
    return pn.Column(text_input, sizing_mode="stretch_width")