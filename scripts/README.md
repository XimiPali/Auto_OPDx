# Pipeline scripts

Numbered in the order they run. Each one reads what the previous one wrote.

| # | Script | Reads | Writes | Time |
|---|---|---|---|---|
| 01 | `01_diagnose_plate.py` | one `.OPDx` | 4-panel figure → `studies/01-label-alignment/figures/` | seconds |
| 02 | `02_validate_alignment.py` | all 69 plates + `data/metadata/*.json` | `studies/01-label-alignment/results/alignment.csv` (old centroids vs new, per plate) | ~4 min |
| 03 | `03_build_dataset.py` | all 69 plates | `data/processed/cells.csv`; with `--save-crops`, `crops.npz` | ~5 min |
| 04 | `04_dinov2_features.py` | `crops.npz` | `data/processed/dinov2*.npz` | ~2 min on MPS |
| 05 | `05_evaluate_labels.py` | `cells.csv` (+ `--dinov2 …npz`) | leave-one-plate-out table, printed | ~3–8 min |

`01` and `02` are diagnostics — run them to check a new plate or to reproduce
the alignment comparison. `03 → 04 → 05` is the training-label pipeline.

All scripts are run from the repo root and need `MPLBACKEND=Agg` when
headless. Study-specific tooling (figure and notebook generators) lives with
its study under `studies/<nn>-…/src/`, not here.
