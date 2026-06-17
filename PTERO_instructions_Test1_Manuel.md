# Running PTERO — step-by-step guide

This guide takes you from nothing to viewing your own mass-spec data in the
PTERO dashboard, on **one sample**. Follow it top to bottom. After most steps
there is a **"You should see"** line — if you don't see that, stop and tell me
before continuing (don't push past a step that didn't work).

> Everything in `this font` is a command to **type (or paste) exactly** and press Enter.
> When the guide says "paste the path", copy the full folder/file path from
> Windows Explorer (click the address bar → it turns into text → copy it).

---

## What you'll do, in plain terms

1. Install the tools PTERO needs (Python, the PTERO code).
2. Convert your `.raw` file to mzML (with MSConvert).
3. Search your data with MaxQuant to identify proteins.
4. Convert the mzML to PTERO's fast format (Zarr) and explore it in the dashboard.
5. Load the MaxQuant results in the dashboard's Identifications tab.

Steps 2 and 3 (MSConvert + MaxQuant) are the trickiest. **Ask me to be on a
call with you for those two** — they're separate programs, not PTERO, and easy
to misconfigure.

---

## PART A — Install PTERO (one time only)

### A1. Install Python

PTERO needs **Python << FILL IN: e.g. 3.10 or newer — check pyproject.toml >>**.

1. Go to https://www.python.org/downloads/ and download Python
   << FILL IN exact version, e.g. 3.12 >> for Windows.
2. Run the installer. **Important:** on the first screen, tick the box
   **"Add Python to PATH"** before clicking Install.
3. Open **PowerShell** (Start menu → type "PowerShell" → Enter) and check it:

   `python --version`

   **You should see:** `Python << FILL IN >>` (the version you installed).
   If you see an error or a different version, tell me.

### A2. Install Git (if you don't already have it)

`git --version`

**You should see:** a version number (e.g. `git version 2.x`).
If you see an error, install Git from https://git-scm.com/download/win (accept
the defaults), close and reopen PowerShell, and try again.

### A3. Get the PTERO code

You have access to the private repository as a collaborator.

First, **go to wherever you want to keep the project** — pick any folder you
like (your Documents, a `Projects` folder, anywhere). For example, to use a
folder called `Projects` in your user directory:

`cd $HOME\Projects`

(If that folder doesn't exist yet, create it first: `mkdir $HOME\Projects` then
the `cd` above. Or just `cd` into any existing folder you prefer.)

Then clone the repository into it:

`git clone https://github.com/AyehBlk/PTERO.git`

**You should see:** Git downloading the project, ending without an error.
A new **`PTERO`** folder now exists inside whatever folder you were in.

If it asks you to log in to GitHub, use your GitHub account (the one I added
as a collaborator).

> **Remember where you put it.** From here on, the guide writes `<PTERO>` to mean
> *the PTERO folder you just created*. Wherever you see `<PTERO>`, use your real
> path to it (e.g. `C:\Users\you\Projects\PTERO`). Tip: in PowerShell you can
> always get back into it by typing `cd ` and then pasting your path.

### A4. Create a virtual environment (an isolated space for PTERO's tools)

Go into the PTERO folder you cloned:

`cd <PTERO>`

> (Type `cd ` then paste your real path to the PTERO folder.)

`python -m venv C:\Users\<< YOUR-WINDOWS-USERNAME >>\venvs\ptero`

> Replace `<< YOUR-WINDOWS-USERNAME >>` with your actual Windows username
> (the name in `C:\Users\...`). Or pick any folder you like for the venv.

Now **activate** it:

`& "C:\Users\<< YOUR-WINDOWS-USERNAME >>\venvs\ptero\Scripts\Activate.ps1"`

**You should see:** your PowerShell prompt now starts with **`(ptero)`**.
This `(ptero)` prefix must be there for every command from here on. If you open
a new PowerShell window later, run this activate command again first.

> If PowerShell refuses with a message about "running scripts is disabled",
> run this once, then try activating again:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### A5. Install PTERO

With `(ptero)` showing in your prompt, from inside the PTERO folder (`<PTERO>`):

`pip install -e ".[raw,dashboard,ids]"`

> << FILL IN: confirm these are the right extras. If you use different ones,
> change them here. >>

**You should see:** lots of packages installing, ending with something like
`Successfully installed ...` and no red `ERROR`. This step takes a few minutes.

### A6. Check it worked

`python -m pytest -q`

**You should see:** a row of dots / `passed` (some tests may be skipped — that's
fine). If you see failures, copy the output and send it to me before going on.

✅ **PTERO is now installed.** You only do PART A once. Next time, you skip
straight to activating the venv (step A4's activate line) and launching the
dashboard (PART D).

---

## PART B — Convert your `.raw` to mzML (with MSConvert)

> This uses **MSConvert** (from ProteoWizard), not PTERO. You already have it.
> **Do this on a call with me the first time.**

1. Open **MSConvert**.
2. Add your `.raw` file.
3. Output format: **mzML**.
4. Add these filters (this matters — PTERO needs centroided peaks):
   - **Peak Picking** (vendor), apply to **MS levels 1–2**.
   - << FILL IN any other MSConvert settings we fixed — e.g. the exact
     peak-picking settings used before. >>
5. Set an output folder you'll remember (e.g. `C:\PTERO_data`). **Do not** put it
   inside any MaxQuant working folder.
6. Click **Start**.

**You should see:** a new `.mzML` file appear in your output folder.

---

## PART C — Identify proteins with MaxQuant

> This uses **MaxQuant**, not PTERO. It's the most fiddly step.
> **Do this on a call with me.**

**Important rules:**
- Give MaxQuant your original **`.raw`** file (NOT the mzML you just made).
  Using the mzML here can make MaxQuant crash at the end — use the `.raw`.
- You need a **FASTA** protein database: << FILL IN: which organism / which
  UniProt file i wants you to use, and where to download it. >>

Settings (I will confirm each on the call):
- Load the `.raw` in the **Raw data** tab.
- Set an **Experiment** name for the file (any short name, e.g. `sample1`).
- **Fixed modifications:** Carbamidomethyl (C).
- **Variable modifications:** Oxidation (M), Acetyl (Protein N-term).
- **Enzyme:** Trypsin/P.
- **FASTA:** the file from above.
- **FDR:** 1% (default).
- Choose an output location you'll remember. **Not** inside the PTERO folder.
- Click **Start** and wait (can take a while).

**You should see:** when it finishes, a **`combined`** folder, and inside it a
**`txt`** folder containing files like `proteinGroups.txt`, `peptides.txt`,
`msms.txt`. **The `txt` folder is what PTERO needs.** If MaxQuant finishes but
there's no `txt` folder, tell me (it stopped early).

---

## PART D — Launch the PTERO dashboard

1. Open PowerShell.
2. Activate the venv (if `(ptero)` isn't already showing):

   `& "C:\Users\<< YOUR-WINDOWS-USERNAME >>\venvs\ptero\Scripts\Activate.ps1"`

3. Go to the project folder:

   `cd <PTERO>`

4. Launch:

   `panel serve dashboard.py --show`

**You should see:** some startup text, then your web browser opens automatically
to the PTERO dashboard (at an address like `http://localhost:5006/dashboard`).
If the browser doesn't open on its own, open it and go to that address manually.

> Leave this PowerShell window open while you use PTERO. To stop the dashboard
> later, click in the PowerShell window and press **Ctrl + C**.

---

## PART E — Using the dashboard

The dashboard has two pages (tabs near the top): **Raw data** and
**Identifications**.

> **About selecting files:** type or **paste the full path** into the text
> boxes. (There are "Browse" buttons but they're not finished yet — paste paths
> instead.)

### E1. Raw data page — convert your mzML to Zarr and explore it

1. Go to the **Raw data** page.
2. Find the **"convert a raw file"** section.
3. In **Raw file (.d folder or .mzML)**, paste the full path to your `.mzML`
   (the one from PART B), e.g.
   `C:\PTERO_data\<< your file >>.mzML`
4. In **Output .zarr path**, paste where you want the fast file saved, ending in
   `.zarr`, e.g.
   `C:\PTERO_data\<< your file >>.zarr`
   (Use a fresh name; **not** inside the MaxQuant `combined` folder.)
5. Set **MS level** to **MS1 + MS2 (both)**.
6. Click **Convert → Zarr & open**.

**You should see:** a green message like "Converted and opened ... — N spectra",
and the view fills with plots.

Now explore the tabs in the View area:
- **Chromatogram + spectrum** — the signal over time. Click a point on the top
  plot to see that scan's spectrum below.
- **Feature map** — a map of all ions (time × mass, colored by intensity).
- **XIC** — type one or more m/z values to track specific ions over time.
- **Diagnostics** — instrument-health plots.
- **Precursor** — (for this kind of data) which peptides were selected to fragment.
- **QC** — summary numbers. Switch the **MS level** selector (top left) between
  1, 2, and all to see how the numbers change.

### E2. Identifications page — load your MaxQuant results

1. Go to the **Identifications** page.
2. **mzML store** — paste the full path to your `.mzML` (same file as before).
3. **MaxQuant txt folder** — paste the full path to the **`txt`** folder from
   PART C, e.g.
   `C:\...\combined\txt`
4. **FASTA** (optional but recommended) — paste the path to the same FASTA file
   you used in MaxQuant. This turns on the sequence-coverage view.
5. Click **Load identifications**.

**You should see:** a green message saying it loaded N PSMs and N protein groups,
and a **Proteins** table fills in.

Then:
- **Click a protein** in the Proteins table → its **peptides** appear below, and
  (if you loaded a FASTA) the **sequence coverage** view shows the protein
  sequence with identified stretches colored.
- **Click a peptide** in the Peptides table → its **annotated spectrum** (the
  matched fragment ions) and an **XIC** appear on the right, and the peptide is
  outlined in red on the sequence.

---

## If something goes wrong

- Stop at the step that failed — don't continue.
- Copy the **exact error text** (from PowerShell or the dashboard) and send it to
  me, along with **which step number** you were on.
- A screenshot of the dashboard helps too.

That's everything for one sample. Thanks for testing PTERO!
