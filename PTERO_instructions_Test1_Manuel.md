# PTERO — quickstart

## Install (once)

Needs Python << FILL IN: e.g. 3.10+ >> and Git.

```powershell
git clone https://github.com/AyehBlk/PTERO.git
cd PTERO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[raw,dashboard,ids]"
```

Prompt should now show `(.venv)`. Sanity check: `python -m pytest -q`.

> If PowerShell blocks the activate script:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

## Prep your data

- **MSConvert**: `.raw` → mzML, with **vendor peak picking on MS levels 1–2**.
- **MaxQuant**: search the **`.raw`** (not the mzML — it can crash on centroided mzML).
  FASTA: << FILL IN organism >>. Carbamidomethyl (C) fixed; Oxidation (M) + Acetyl (N-term) variable; 1% FDR.
  → you want the **`combined/txt`** folder it produces.

## Run

```powershell
.\.venv\Scripts\Activate.ps1   # if not already active
panel serve dashboard.py --show
```

Browser opens to the dashboard. **Paste file paths into the text boxes** — the Browse buttons aren't finished.

## Use it

**Raw data tab** → "convert a raw file": paste your `.mzML` path + an output `.zarr` path, MS level = **both**, Convert. Then explore the tabs (chromatogram, feature map, XIC, diagnostics, QC).

**Identifications tab** → paste the `.mzML` path, the **`txt`** folder path, and the FASTA path → Load. Click a protein → peptides + sequence coverage. Click a peptide → annotated spectrum + XIC.

Anything breaks: send me the error + which step.
