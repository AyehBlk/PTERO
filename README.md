# PTERO

A Python package (early development).

## Requirements

- Python 3.10 or newer
- Git

## Setup

Clone the repository:

```bash
git clone <REPO_URL>
cd PTERO
```

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

Install in development (editable) mode:

```bash
pip install -e ".[raw,dashboard,dev]
```

## Verify the install

```bash
python -c "import ptero; print(ptero.__version__)"
python -m pytest -q
```

The import should print a version, and the test suite should pass.