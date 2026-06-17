"""
A navigable file picker for dashboard path fields: a folder box plus a list that
shows sub-folders (click to enter) and matching files (click to choose). Lets the
user walk the filesystem without typing paths, and passes only the chosen path
(so multi-GB files are fine -- nothing is uploaded).

A served Panel app runs in a browser and cannot open the OS file dialog; this is
a server-side browser of the machine running the dashboard (the user's own disk
for local use).

``browse_field(text_input, patterns=[...])`` returns a layout: the TextInput, a
"Browse…" button that reveals the picker. Choosing a file writes its full path
into the TextInput.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import panel as pn

_UP = "⬆ .. (up one level)"
_DIR_PREFIX = "📁 "
_FILE_PREFIX = "📄 "


def _entries(folder: str, patterns: Sequence[str]) -> tuple[list[str], dict]:
    """Build the navigable listing for *folder*.

    Returns ``(labels, mapping)`` where *labels* is what the dropdown shows
    (an up-entry, then sub-folders, then matching files) and *mapping* maps each
    label to ``("dir", path)`` or ``("file", path)``.
    """
    labels: list[str] = []
    mapping: dict[str, tuple[str, str]] = {}
    try:
        p = Path(folder).expanduser()
    except Exception:
        return labels, mapping
    if not p.is_dir():
        return labels, mapping

    # up one level (unless at a filesystem root)
    if p.parent != p:
        labels.append(_UP)
        mapping[_UP] = ("dir", str(p.parent))

    # sub-folders first (so the user can navigate down). Includes dir-like
    # stores (.d / .zarr) -- those are also selectable as files below.
    dirlike_pats = {pat for pat in patterns if pat.startswith("*.")
                    and pat.lower() in ("*.d", "*.zarr")}
    try:
        children = sorted(p.iterdir(), key=lambda c: c.name.lower())
    except (PermissionError, OSError):
        children = []
    for c in children:
        if c.is_dir():
            # a .d / .zarr directory that matches a requested pattern is a
            # selectable *file* (a store), not a folder to descend into
            is_store = any(c.match(pat) for pat in dirlike_pats)
            if is_store:
                lbl = _FILE_PREFIX + c.name
                mapping[lbl] = ("file", str(c))
            else:
                lbl = _DIR_PREFIX + c.name
                mapping[lbl] = ("dir", str(c))
            labels.append(lbl)

    # then files matching the patterns
    seen = set()
    for pat in patterns:
        for hit in sorted(p.glob(pat), key=lambda c: c.name.lower()):
            if hit.is_file() and hit.name not in seen:
                seen.add(hit.name)
                lbl = _FILE_PREFIX + hit.name
                mapping[lbl] = ("file", str(hit))
                labels.append(lbl)
    return labels, mapping


def browse_field(text_input: "pn.widgets.TextInput",
                 *, default_dir: str | None = None,
                 patterns: Sequence[str] = ("*",),
                 pick_folder: bool = False,
                 width: int = 720):
    """Wrap *text_input* with a navigable folder/file picker.

    If ``pick_folder`` is True the picker selects a *folder*: navigate into the
    folder you want and click "Use this folder" (the folder box's current path is
    written to the field). Files are still listed for context but choosing one
    does nothing in this mode.
    """
    start = default_dir
    if not start:
        cur = (text_input.value or "").strip()
        start = str(Path(cur).parent) if cur else str(Path.home())

    folder_box = pn.widgets.TextInput(name="Folder", value=start, width=width - 120)
    refresh_btn = pn.widgets.Button(name="↻", button_type="default", width=40)
    listing = pn.widgets.Select(name="Folders & files", options=[], size=12, width=width - 40)
    use_label = "Use this folder" if pick_folder else "Use selected file"
    use_btn = pn.widgets.Button(name=use_label, button_type="primary", width=150)
    cancel_btn = pn.widgets.Button(name="Cancel", button_type="default", width=90)
    browse_btn = pn.widgets.Button(name="Browse…", button_type="default", width=90)
    hint = pn.pane.Markdown("", width=width - 40)

    state = {"map": {}}     # current label -> (kind, path)

    card = pn.Card(
        pn.Row(folder_box, pn.Column(pn.Spacer(height=18), refresh_btn)),
        listing, hint,
        pn.Row(use_btn, cancel_btn),
        title=("Pick a folder" if pick_folder else "Pick a file"),
        collapsed=True, width=width)

    def _relist():
        labels, mapping = _entries(folder_box.value.strip(), patterns)
        state["map"] = mapping
        # clear the current selection BEFORE swapping options, so Panel never
        # tries to resolve a now-stale label against the new options list
        listing.value = None
        listing.options = labels
        if pick_folder:
            hint.object = ("_Click a 📁 folder to open it. When the box above shows the "
                           "folder you want, click **Use this folder**._")
        else:
            n_files = sum(1 for v in mapping.values() if v[0] == "file")
            hint.object = (f"_{n_files} matching file(s) here. Click a 📁 folder to open it, "
                           f"a 📄 file to select it, then **Use selected file**._")

    def _on_pick(event=None):
        """Entering a folder navigates; a file stays selected for the Use button."""
        lbl = listing.value
        if not lbl or lbl not in state["map"]:
            return
        kind, path = state["map"][lbl]
        if kind == "dir":
            # defer the navigation so we don't mutate this Select's options from
            # inside its own value-change callback (which races and errors)
            if pn.state.curdoc is not None:
                pn.state.curdoc.add_next_tick_callback(lambda: setattr(folder_box, "value", path))
            else:
                folder_box.value = path

    def _open(event=None):
        cur = (text_input.value or "").strip()
        if cur:
            # for folder-pick, the field already holds a folder; for file-pick, its parent
            p = Path(cur)
            folder_box.value = cur if (pick_folder and p.is_dir()) else str(p.parent)
        _relist()
        card.collapsed = False

    def _use(event=None):
        if pick_folder:
            text_input.value = folder_box.value.strip()
            card.collapsed = True
            return
        lbl = listing.value
        if lbl and lbl in state["map"]:
            kind, path = state["map"][lbl]
            if kind == "file":
                text_input.value = path
                card.collapsed = True
            else:
                folder_box.value = path      # hit Use on a folder -> enter it

    def _cancel(event=None):
        card.collapsed = True

    refresh_btn.on_click(lambda e: _relist())
    folder_box.param.watch(lambda e: _relist(), "value")
    listing.param.watch(_on_pick, "value")
    browse_btn.on_click(_open)
    use_btn.on_click(_use)
    cancel_btn.on_click(_cancel)

    return pn.Column(
        pn.Row(text_input, pn.Column(pn.Spacer(height=18), browse_btn)),
        card,
        sizing_mode="stretch_width")