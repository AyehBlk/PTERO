"""
Entry point for the PTERO dashboard.

Run from the repo root:

    panel serve dashboard.py --show

Serve *this* file, not ``ptero/dashboard/app.py`` directly. Serving this file
imports the app as part of the ``ptero.dashboard`` package (absolute import), so
the package's relative imports (``from ..spectra``, ``from .de_view``, ...) all
resolve. Serving ``app.py`` directly runs it with no parent package and fails with
"attempted relative import with no known parent package".
"""
from ptero.dashboard.app import build_app

build_app().servable()