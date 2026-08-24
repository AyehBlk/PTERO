<p align="center">
  <img src="assets/pterosaur.avif" alt="PTERO" width="320">
</p>

# PTERO

**Proteomics Tool for Exploratory Research & Optimization** — a cross-platform
Python package for mass-spectrometry proteomics: loading and lazily storing raw MS
data (Zarr/Parquet backends), quality control, quantification, and
differential-expression analysis.

Early development; the API is not yet stable.

## Requirements

- Python 3.10 or newer
- Git

## Setup

Clone the repository:

```
git clone https://github.com/AyehBlk/PTERO
cd PTERO
```

Create and activate a virtual environment:

```
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

Install in development (editable) mode:

```
pip install -e ".[raw,dashboard,ids,dev]"
```

This pulls the raw-data readers, the Panel dashboard, the identifications layer,
and the test tooling. Additional optional extras are defined in `pyproject.toml`
— for example `bruker` for timsTOF `.d` reading, which requires Bruker's libraries
and runs on Windows/Linux only.

## Verify the install

```
python -c "from importlib.metadata import version; print('ptero', version('ptero'))"
python -m pytest -q
```

The first command prints the installed version; the test suite should pass.
