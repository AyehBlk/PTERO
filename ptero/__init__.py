"""PTERO — Proteomics Tool for Exploratory Research & Optimization."""
import importlib as _il

__version__ = "0.2.0"

# Subpackages load on first access, so `import ptero` stays light and never
# fails when an optional dependency (panel, anndata, psm_utils, …) is absent.
_SUBPACKAGES = {"spectra", "process", "ids", "quant", "dashboard"}

def __getattr__(name):
    if name in _SUBPACKAGES:
        return _il.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'ptero' has no attribute {name!r}")

def __dir__():
    return sorted(list(globals()) + list(_SUBPACKAGES))