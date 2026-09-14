# Rebuilding the per-cell labels

Work on the "predict sample height from a photo" task. Every number below was
measured on the 69 plates in `data/`, and each is reproducible with the scripts
listed at the end.

## Summary

The per-cell labels the previous model trained on were dominated by substrate
tilt rather than polymer thickness. Correcting that changes what the benchmark
means.

| | previous | rebuilt |
|---|---|---|
| label definition | raw `z_max` | plane-relative `z_max` |
| R² from `(row, col)` position alone | **0.953** | **0.406** |
| gradient direction across plates | row-dominant on 69/69 | row 37 / col 32 |
| height-crop shapes | **534 distinct** | one fixed shape |
| crop geometry | square in pixels | square in micrometres |
| plates covered | 53 (height) / 60 (photo) | 69 / 69 |
| lattice residual, height map | 3.11% of pitch | **2.75%** |
| lattice residual, photo | 1.57% of pitch | **1.12%** |
| cross-modal registration | not measured | **8.15 µm** median, 3.1% of pitch |

## 1. The substrate tilt dominated the labels

A plane fit to the background (samples at or below the 45th percentile, so the
raised features do not drag it upward) spans a median of **20.99 µm** across a
plate, against a median raw z range of 26.93 µm.

```
tilt / raw z range:  median 71%,  above 50% on 58 of 69 plates
worst: 10-06-2025_P2, tilt 35.24 µm of a 39.22 µm range (90%)
median tilt gradient: 8.13 µm of height per mm of lateral travel
```

The stored `paired/opdx/*.npy` labels retain it in full. On plate
`10-01-2025_P1`, mean cell-max height by row:

```
r00   8.83 µm      r04  19.17 µm
r01  10.54         r05  21.13
r02  12.44         r06  24.39
r03  15.03         r07  26.57      correlation(row, label) = +0.985
```

Across all 45 plates in `paired/`: median correlation **+0.980**, |r| > 0.9 on
**41 of 45**.

### Why this is tilt and not real signal

Both would show up as a gradient, so direction is the discriminator. A substrate
tilt is set by how the slide sits in the holder and always runs along the stylus
scan axis; a formulation layout is chosen per experiment.

```
raw z labels        row-dominant on 69/69 plates, positive on 69/69
plane-relative      row-dominant on 37, column-dominant on 32; sign mixed
```

Plate `10-15-2025_P4` after detrending correlates with **column** at +0.892 —
which the y-axis substrate tilt cannot produce. `9-24-2025_P1` is flat in both
directions. The residual gradient after detrending is the experiment, not the
instrument.

A plane is the right model. Higher orders start absorbing the features:

| background surface | corr(row, label) | min label |
|---|---|---|
| plane | +0.643 | 5.36 µm |
| quadratic | +0.793 | 5.31 µm |
| cubic | −0.742 | **−1.10 µm** (impossible) |

## 2. What this does to the previous benchmark

The previous within-plate result was per-cell R² = 0.83. On the raw labels,
**row position alone reaches R² = 0.953** (median; above 0.83 on 64 of 69
plates). A trivial position lookup beats the model, so a high R² on those labels
is not evidence that anything learned thickness.

The mechanism is a position–position confound. The photos have uneven
illumination, so brightness is a smooth function of position; the raw label is
too. Measured directly, within plate:

```
brightness features -> row index      Spearman +0.973
brightness features -> raw label      Spearman +0.949
position            -> raw label      R²        0.953
```

Under leave-one-plate-out, brightness features against raw labels give Spearman
0.855. Against rebuilt labels: **0.021**. The apparent signal was the confound.

On the rebuilt labels the position-only baseline is **0.406**, and that residual
is real experimental design. A model that beats 0.406 is measuring something.

### Consequence for alignment error

With raw labels, misalignment converts directly into label error at the median
tilt gradient of 8.13 µm/mm. Genuine within-plate cell-to-cell variation is only
**1.02 µm SD** once detrended:

| crop misaligned by | label error | as % of real signal |
|---|---|---|
| 30 µm | 0.24 µm | 22% |
| 100 µm | 0.81 µm | 72% |
| 300 µm (one cell) | 2.44 µm | 216% |

Plane-relative labels remove the gradient, so the same misalignments inject
~0 µm. Detrending both cleans the labels and makes the pipeline robust to
whatever registration error remains.

## 3. Geometry

Measured per plate; nothing here should be hardcoded.

```
height-map shape     63x (901, 75),  4x (901, 77),  2x (901, 74)
scan span            x 2139-2369 µm,  y 2442-2664 µm     (varies per plate)
sampling             x 30.33 µm,      y 2.878 µm         (medians)
ANISOTROPY           10.61x  (range 9.52 - 11.23)
cell pitch           266.9 µm
feature size         ~98 µm square -> only 2-3 samples across the coarse axis
grid rotation        1.44 deg median, up to 4.21
photo                all 69 are 1536x2048
photo scale          1.09 µm/px on x,  1.44 µm/px on y
PHOTO PIXEL ASPECT   1.32  (not 1.0)
```

Two consequences:

**Rotation must happen in physical space.** Rotating a 10.6×-anisotropic array
while treating its cells as square necessarily turns squares into
parallelograms. No downstream transform repairs it.

**A square pixel crop is not a square physical region.** The photos are 4:3 but
cover a nearly square field, so pixel scale differs 1.32× between axes. The
previous per-cell RGB crops were ~105×105 px, i.e. ~114 µm × ~151 µm.

## 4. The lattice fit

Features span 2–3 samples on the coarse axis, so a single-feature centroid
cannot beat about ±15 µm. But the grid is written by a DMD through a 4×
objective onto a piezostage, so its pitch is deterministic. That justifies
fitting one rigid model to all 64 features at once — 10 parameters rather than
128 free coordinates.

Per output axis:

```
pos = a·r + b·c + d·[r ≥ rows/2] + e·[c ≥ cols/2] + f
```

Five parameters per axis, spanning translation, independent scale on each grid
axis, rotation and shear, and the quadrant gaps between rows/columns 3 and 4.
The gaps are real and substantial: on `9-29-2025_P3` the column spacings are
259, 263, 262, **346**, 270, 267, 283 µm.

Four defects had to be fixed to make this reliable, all found by inspecting
failing plates and outputs:

1. **Wrong polarity in the photo detector.** Features are dark on a bright
   substrate; the existing `calculate_brightness.tophat_response` uses a white
   top-hat, which detects bright-on-dark.
2. **Illumination bias.** A global percentile threshold detects mostly features
   from the brighter side of the plate. Dividing by a heavily blurred copy
   normalises contrast first. With (1), this took the photo residual from 108 px
   to ~3 px.
3. **No surplus candidates.** Truncating detections to exactly 64 forces dust
   into a slot, displacing two real features and shearing the whole model. This
   was the single largest source of failure — four plates sat above 20% of
   pitch. Retaining surplus candidates and pruning by residual fixed them:
   9-25-2025_P8 24.5% → 0.9%, 9-29-2025_P3 25.6% → 5.1%.

4. **Selecting on residual alone picks degenerate fits.** This one is subtle and
   was caught only by eyeballing the cell montage, where one row of crops came
   out blank. The model can place a predicted row *inside the quadrant gap* and
   discard a real edge row as an outlier. Seven of eight rows then fit tightly,
   so the residual looks excellent — 3 px — while the arrangement is wrong. On
   `10-01-2025_P1` the fitted middle gap came out **smaller** than the row pitch
   (123 px against 182) and the real bottom row at y = 1465 was dropped
   entirely.

   Because the parameter search *optimises* residual, it actively preferred
   these fits. **20 of 69 plates were affected.** The fix is to select on
   residual *plus* slot coverage — the fraction of predicted slots with a
   detection nearby — since a phantom row has nothing near it. Coverage is now
   1.000 on 66 of 69 plates (min 0.828), and cross-modal RMSE on
   `10-01-2025_P1` went from 56.05 µm to 7.65 µm.

Detection parameters are searched, keeping the best fit by that combined score.
Quality is strongly non-monotonic in the threshold — `9-25-2025_P9` fits to 1.3%
at the 85th percentile but fails at the 93rd, while `9-30-2025_P7` is the
reverse. Neither criterion consults labels or ground truth.

The lesson worth carrying: **a low residual is necessary but not sufficient.**
Any fit selected by minimising an error metric needs a second, independent check
that it actually explains the data.

### Honest assessment

The localisation gain is **modest** — 1.1× on height maps, 1.4× on photos at
the median. The previously stored centroids were already reasonable at ~3.1% of
pitch. What is new:

* **coverage**: 69/69 plates, against 53 and 60 previously
* **no catastrophic failures**: worst plate 7.78% of pitch, against 25.6% before
  the fixes above
* **a cross-modal registration that did not previously exist**: 8.15 µm median
  RMSE, measured between *detected* centroids on both sides rather than between
  the two smooth models, which would flatter the result

The large win in this work is the labels, not the localisation.

## 5. Does this reduce the control-point requirement? No — and that matters

The working hypothesis was that the "8 control points" figure was inflated by
label error, and would fall toward 2–3 once the labels were corrected.

**It does not. The opposite happens.** Leave-one-plate-out, DINOv2 features,
identical pipeline, only the label definition changed:

| K | previous labels R² | Spearman | rebuilt labels R² | Spearman | plates > 0 |
|---|---|---|---|---|---|
| 0 | 0.328 | 0.928 | −1.789 | 0.425 | 10/69 |
| 4 | 0.505 | 0.927 | −0.189 | 0.413 | 16/69 |
| **8** | **0.684** | **0.925** | **−0.045** | **0.407** | 30/69 |
| 16 | 0.772 | 0.923 | +0.012 | 0.421 | 37/69 |

*(DINOv2 ViT-B/14, 768-dim, alpha selected by grouped CV on training plates only.)*

**The pipeline is validated.** On the previous labels it reproduces the deck
closely — at K=8 it gets median R² 0.684 and 68/69 plates positive, against the
deck's reported 0.73 and 44/45. So the comparison is credible.

**On corrected labels, absolute-height prediction never becomes viable.** R²
first crosses zero only at K=16, and even there on just 37 of 69 plates.
Ranking survives but weakens sharply: Spearman 0.93 → 0.42.

This is robust to backbone size and to the phantom-row bug (§4). ViT-S/14
(384-dim) gives rebuilt-label Spearman 0.337; ViT-B/14 (768-dim) gives 0.425.
Fixing the phantom row, which had corrupted the crops on 20 of 69 plates, moved
R² at K=16 only from −0.019 to +0.012. Doubling model capacity and fixing a real
crop bug both leave the conclusion intact, so the deck's ~1,500 dims would not
rescue it either.

The brightness baseline collapses completely: Spearman 0.855 on previous labels,
**0.021** on rebuilt ones. The deck's Phase 1 "+0.47 Spearman after per-plate
normalisation" was position–position correlation, not depth.

### What this means

The 8 control points were not overhead to be optimised away. They were holding
up a task that was largely position inference. Once the substrate tilt is out of
the labels:

* **Ranking cells within a plate still works** — Spearman ~0.41 is well above
  chance, and above the 0.021 that brightness gives, so the photo does carry
  real thickness information.
* **Absolute heights do not** — the genuine within-plate signal is only ~1.0 µm
  SD, and current features cannot resolve it to better than that.

The honest read is that the deck's deployable claims ("Absolute heights, typical
batches: median R² ≈ 0.58", "Solved with K = 8") do not survive label
correction. The reference-patch plate redesign on the deck's call-to-action
slide becomes *more* important, not less: if µ and σ are read off known-height
corner patches, no per-batch profilometry is needed at all, and the remaining
job is the within-plate ranking that does still work.

## 6. Open

* **Bigger backbone / fine-tuning.** ViT-L/14 and an end-to-end fine-tune are
  untested. The trend suggests limited upside, but it is not ruled out.
* **Better labels than `h_max`.** A single-pixel maximum on a ~1 µm signal is
  noisy. `h_p95` and a top-decile mean are already computed; a feature-mask
  mean would be better still.
* **Two plates fit poorly** (`9-24-2025_P11` 7.78%, `6-18-2025_P6` 5.70%). Both
  yield only 57 detections — genuinely torn samples, not a solver failure.
* **25% of cells are dropped** for crop overhang, because the photos clip the
  top and bottom rows of the grid. Recoverable with edge-aware handling.
* **Is any of the tilt real?** The plane is fit to the substrate between
  features, so subtracting it is physically correct for thickness. But if the
  deposition process itself varies along the chamber, some genuine gradient is
  removed with it. Worth a conversation with the materials group.

## Reproducing

```bash
uv venv --python 3.12 && uv pip install -e .

uv run python scripts/01_diagnose_plate.py         # one plate, four questions
uv run python scripts/02_validate_alignment.py     # 69 plates, previous vs new
uv run python scripts/03_build_dataset.py --save-crops
uv run python scripts/04_dinov2_features.py
uv run python scripts/05_evaluate_labels.py        # brightness features
uv run python scripts/05_evaluate_labels.py --dinov2 data/processed/dinov2.npz
uv run python -m pytest tests/ -q

jupyter lab studies/01-label-alignment/alignment.ipynb   # visual walkthrough
uv run python studies/01-label-alignment/src/make_summary_figure.py   # figures/summary.png
```

`OPDx_read.reader` calls `plt.imshow` unconditionally inside `get_data_2D`,
which aborts a headless process under the default macOS backend and leaks a
figure per plate otherwise. `alignment.plate.load_plate` suppresses it without
touching the global matplotlib backend, so notebook plotting is unaffected.
