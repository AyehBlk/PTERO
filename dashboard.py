"""Entry point for the PTERO dashboard. Run with:

    panel serve dashboard.py --show
"""
from ptero.dashboard import build_app

build_app().servable(title="PTERO — Raw Data Dashboard")
