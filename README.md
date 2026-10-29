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

## mzZarr v1.0

PTERO provides tools for working with mzZarr, a proposed
vendor-independent format for mass-spectrometry data. The
mzZarr v1.0 format specification is maintained separately
from the PTERO software version.

See the [mzZarr v1.0 specification](docs/mzzarr/SPECIFICATION.md)
for the normative format requirements.

## License

PTERO software is distributed under the
[Apache License 2.0](LICENSE).

## Citation

If you use PTERO in research, please cite the software.
Machine-readable citation metadata is available in
[CITATION.cff](CITATION.cff).

The PTERO software version is 0.2.0; the mzZarr format
specification version is 1.0.
