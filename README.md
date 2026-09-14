# Auto_OPDx

Tools for measuring polymer sample thickness from profilometer scans and photos.

This repository has two parts:

1. **The original tool** by [cjeong1021](https://github.com/cjeong1021/Auto_OPDx)
   — a GUI that reads `.OPDx` profilometer scans and fluorescence images and
   exports per-sample measurements. Documented in [§ Original tool](#original-tool)
   below, unchanged.
2. **This fork's work** (Qazim Pali, CCNY DSE capstone, fall 2026) — rebuilding
   the training labels for the *"predict height from a photo"* project, and what
   that revealed. Documented right here.

---

## The problem this fork works on

A materials lab prints **64 polymer samples in an 8×8 grid** on a silicon chip.
Each sample uses a different chemical formulation, and the lab needs to know how
**thick** each one came out.

Today that is measured with a **stylus profilometer** — a needle dragged across
the surface. It is accurate to a micron, but takes 20–30 minutes per chip and
**scratches the polymer off**, which matters because the samples have to survive
for a later experiment.

The idea: photograph the chip instead, and train a model to predict thickness
from the photo. To train it you need pairs of *(photo, true height)* — the true
heights come from the profilometer. Building those labels correctly is what this
fork does.

```
1. build labels     profilometer scan  →  height of each of the 64 cells     ← THIS FORK
2. train a model    photo              →  height                              (previous work)
3. deploy           photo replaces the needle                                 (the goal)
```

## What we found — the short version

**The previous labels were mostly measuring where a cell sat on the chip, not
how thick it was.**

The silicon chip sits slightly tilted in the profilometer. That tilt spans about
21 µm across a chip — **71% of the entire recorded height range** — and it had
never been subtracted from the labels. So a cell's "height" label correlated
with its **row number at r = +0.98**.

Consequence: on those labels, knowing only a cell's row and column predicts the
label with **R² = 0.95**. The previously reported model scored 0.83. A lookup
table beat the model, so that result was never evidence the photo predicts
thickness.

We rebuilt the labels (tilt removed, crops in real micrometres, both images
registered with a single grid fit), re-ran the model, and then stress-tested the
outcome:

| | old labels | rebuilt labels |
|---|---|---|
| R² from position alone (no photo) | 0.95 | 0.41 |
| photo model, ranking cells (Spearman ρ) | 0.93 | 0.42 → **0.20** once position is controlled |
| photo model, absolute height (R², 8 control cells) | 0.68 | **−0.05** |

So: the photo carries a **weak but real** thickness signal (ρ ≈ 0.20, and it is
structural — brightness alone finds nothing). It is **not enough for absolute
heights**. The real cell-to-cell variation is only ~1 µm, below what a single
flat-lit photo resolves. The "8 control points" the previous work needed were
not overhead to remove — they were propping up a task that was largely position
inference.

**What this means for the project:** stop improving features on a single photo;
give the camera a depth cue. Candidates, cheapest first: a deliberate focus
z-stack on the motorised stage already being built; one angled LED for shadow
length; known-height reference patches on the chip. Full reasoning in the
studies.

## Where to find what

```
Auto_OPDx/
│
├── src/Auto_OPDx/alignment/        THE LIBRARY — the reusable engine
│   ├── plate.py                      load .OPDx → µm · fit substrate plane · detrend · resample
│   ├── lattice.py                    detect the 64 features · fit ONE 8×8 grid model to all of them
│   ├── rgb.py                        find the same 64 features in the photo
│   └── dataset.py                    register photo ↔ scan · cut per-cell crops · compute labels
│
├── tests/test_lattice.py           19 tests of the grid solver against synthetic grids with known truth
│
├── scripts/                        THE PIPELINE — numbered, run in order   → scripts/README.md
│   ├── 01_diagnose_plate.py          one plate: geometry, tilt, feature count, figure
│   ├── 02_validate_alignment.py      all 69 plates: previous centroids vs ours
│   ├── 03_build_dataset.py           per-cell features + labels → data/processed/
│   ├── 04_dinov2_features.py         DINOv2 embeddings of the crops
│   └── 05_evaluate_labels.py         leave-one-plate-out, sweep over control-point count K
│
├── studies/                        THE FINDINGS — two write-ups, same shape   → studies/README.md
│   ├── 01-label-alignment/           "Were the labels built right?"        → no; here's the fix
│   └── 02-thickness-signal-audit/    "Is that negative result an artifact?" → no; here's why
│         each has: README.md (every number) · a runnable notebook · src/ · results/ · figures/
│
├── data/                           NOT IN GIT — 69 .OPDx scans, 69 photos, metadata, caches
│
└── src/Auto_OPDx/*.py, main.py …   the original tool, unchanged (see below)
```

**If you only read one thing:** `studies/01-label-alignment/README.md`, then
look at `studies/01-label-alignment/figures/summary.png`.

**If you want the figures interactively:** open either study's `.ipynb` in
VS Code or Jupyter and pick the `.venv` kernel (Python 3.12). Outputs are
already embedded, so they read without running.

## What we built, in more detail

**The grid fit.** A single feature is only 2–3 samples wide on the scan's coarse
axis, so its centroid cannot be located better than ~±15 µm. But the grid is
written by a DMD onto a piezostage — its pitch is deterministic. So instead of
64 independent centroids, we fit *one* 10-parameter model (translation, per-axis
scale, rotation/shear, the two quadrant gaps) to all 64 features at once, in
both the scan and the photo. Because both grids are indexed by `(row, col)`,
registering them needs no image matching: it is an affine fit on 64 known
correspondences. Median cross-modal error: **8 µm, 3% of a cell**.

**Detrending.** Fit a plane to the substrate (the lowest 45% of samples, so the
raised features don't pull it up) and subtract it. A plane is the right order —
a quadratic makes the position leakage *worse*, a cubic produces impossible
negative thicknesses.

**Physical units throughout.** The scan samples every 2.9 µm on one axis and
30 µm on the other (10.6× anisotropic); the photo's pixels are 1.09 × 1.44 µm
(aspect 1.32). Rotating or cropping in pixel space distorts both. Everything is
converted to micrometres first.

**Bugs found and fixed along the way** (each caught by inspecting a failing
plate, each now a regression test): wrong detector polarity on the photo;
illumination bias in thresholding; dust being forced into grid slots; and a
"phantom row" the model hid inside the quadrant gap — invisible to RMSE, visible
only in the crop montage, present on 20 of 69 plates. Details in
`studies/01-label-alignment/README.md § 4`.

## Set up and reproduce

Needs Python **3.12** — 3.14 has no wheels for numpy / OpenCV / PyQt5.

```bash
uv venv --python 3.12
uv pip install -e .
uv pip install pytest torch torchvision      # tests + DINOv2

# put the data in place (not in git):
#   data/opdx/*.OPDx        data/rgb/img/*.{jpg,bmp}        data/metadata/*.json

export MPLBACKEND=Agg
uv run python -m pytest tests -q
uv run python scripts/01_diagnose_plate.py
uv run python scripts/02_validate_alignment.py
uv run python scripts/03_build_dataset.py --save-crops
uv run python scripts/04_dinov2_features.py --model dinov2_vitb14 --out data/processed/dinov2_b.npz
uv run python scripts/05_evaluate_labels.py --dinov2 data/processed/dinov2_b.npz --alpha -1
```

The studies' own scripts are listed in `studies/README.md`.

Known quirk: `OPDx_read` calls `plt.imshow` on every load, which aborts headless
runs on macOS. `alignment/plate.py` suppresses it without touching the global
matplotlib backend. Also, NumPy 2 on Apple Accelerate raises spurious
`matmul` warnings — verified harmless, filtered narrowly in `05_evaluate_labels.py`.

## Status

- ✅ Labels rebuilt for all 69 plates; pipeline validated (reproduces the
  previous model's result on the *old* labels, so the comparison is sound)
- ✅ Negative result stress-tested and confirmed (study 02)
- ⬜ Present findings; decide between depth-from-focus, oblique
  illumination, and reference patches
- ⬜ Try a per-formulation target (between-chip variation is 2.3× the
  within-chip variation we've been chasing)

Any future evaluation on this data **must include the position control** from
study 02 — otherwise a model that has only learned the illumination gradient
looks identical to one that has learned thickness.

---

## Original tool

*Documentation for the GUI and notebooks as written by the original author,
unchanged.*

**Auto_OPDx** is a Python-based tool designed for the automated processing and
analysis of profilometry data (OPDx files). It provides both a graphical user
interface (GUI) for interactive data visualization and a Jupyter Notebook
interface for flexible, document-based analysis.

### Features

* **OPDx Parsing:** Native support for loading and extracting surface data from OPDx files.
* **Interactive Visualization:** Integrated Matplotlib plotting within a Qt-based GUI to inspect surface profiles instantly.
* **Automated Export:** Seamlessly transition from raw data to structured CSV files for downstream analysis.

### Installation

#### Prerequisites
* **Python 3.10+**
* It is highly recommended to use [uv](https://github.com/astral-sh/uv) for the fastest and most reliable dependency management.
* Or you can use pip.

#### Installing with pip
1.  **Clone the repository and navigate into it:**
    ```
    git clone https://github.com/cjeong1021/Auto_OPDx.git

    cd Auto_OPDx
    ```
2.  **Create and activate a virtual environment:**
    ```
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install:**
    ```
    pip install -e .
    ```

#### Installing with `uv`
[uv](https://github.com/astral-sh/uv) handles virtual environments and dependencies automatically based on the `pyproject.toml` file.

1.  **Clone the repository and navigate into it:**
    ```
    git clone https://github.com/cjeong1021/Auto_OPDx.git

    cd Auto_OPDx
    ```
2.  **Sync dependencies:**
    This command creates a virtual environment and installs all necessary packages in one step.
    ```
    uv sync
    ```

### Usage

#### Running the GUI
Run the application in your virtual environment:
```
auto-opdx
```

#### Running Jupyter Notebook
An interactive Jupyter Notebook is also provided for better visualization. You can run the Jupyter Notebook in the virtual environment:

1.  Launching Server
```bash
jupyter lab
```
This will open a web browser where you can open `profilometryheights_notebook.ipynb`. Ensure all OPDx files are in the same directory and complete the initial prompt asking for the file name and grid rows/cols.
