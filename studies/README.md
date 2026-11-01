# Studies

Five investigations built on the `Auto_OPDx.alignment` package. Each is a
self-contained folder with the same shape:

```
studies/<nn>-<name>/
├── README.md        the write-up — every number, every figure, how to reproduce
├── <name>.ipynb     executed notebook, figures embedded, runnable top to bottom
├── src/             the study's own scripts (figures, notebook generator, …)
├── results/         small CSV tables that ARE the findings (tracked)
└── figures/         rendered PNGs (tracked)
```

Read them in order — the second exists because of what the first found.

| | Study | Question | Answer |
|---|---|---|---|
| 01 | [**Label alignment**](01-label-alignment/README.md) | Were the training labels for "height from a photo" built correctly? | No. 71% of the label was substrate tilt; a row/column lookup beat the published model. Rebuilt labels; absolute-height prediction does not survive. |
| 02 | [**Thickness-signal audit**](02-thickness-signal-audit/README.md) | Is that negative result an artifact of how the features were built? | No. Diluted features, crop size and label noise all ruled out. ~Half of the model's remaining skill was reading plate position; ρ≈0.20 of genuine structural signal survives. |
| 03 | [**Alignment IoU**](03-alignment-iou/README.md) | Is the new alignment measurably better than the previous crops, not just visibly? | Yes, once the crops centre on the detected pads: mean pad IoU 0.67 → 0.71, cells with no pad in the crop 59 → 0, on the same 2,855 cells. The first release scored 0.63 — below the previous crops — which this study caught and fixed. |
| 04 | [**Instrument checks**](04-instrument-checks/README.md) | Can the three open issues (4:3 calibration, coarse scan, new images) be fixed? | Calibration narrowed to one question for the lab, and it does not affect alignment or IoU. The coarse traces make pads that sit on a trace come out a third too narrow (corrected; fixed in study 05). Detection now handles larger, 16-bit and float images. |
| 05 | [**IoU closer to 1**](05-iou-closer-to-1/README.md) | What keeps the IoU below 1, and how much of it can honestly be fixed? | Mean IoU 0.709 → 0.767 (previous crops 0.669 → 0.728, same measurement), chiefly by rebuilding each scan pad from its traces. An independent check agrees the new alignment is better (median pairing error 8.3 → 5.7 µm). A per-cell "second pass" raises IoU to 0.816 but makes the real pairing worse, so it is not used. |

## How the pieces fit

```
src/Auto_OPDx/alignment/      the library: load → detrend → detect → lattice → register → crop
tests/test_lattice.py          19 synthetic-grid tests for the lattice solver
scripts/01–05_*.py             the pipeline, numbered in run order, over all 69 plates
studies/01-label-alignment/    what the pipeline found, and the figures that show it
studies/02-…-audit/            stress-test of that finding, with its own feature builder
```

Pipeline outputs land in `data/processed/` (gitignored caches) and in
`studies/01-label-alignment/{results,figures}/`.

## Reproduce everything

```bash
uv venv --python 3.12 && uv pip install -e . && uv pip install pytest torch torchvision

uv run python -m pytest tests -q
uv run python scripts/01_diagnose_plate.py
uv run python scripts/02_validate_alignment.py
uv run python scripts/03_build_dataset.py --save-crops
uv run python scripts/04_dinov2_features.py --model dinov2_vitb14 --out data/processed/dinov2_b.npz
uv run python scripts/05_evaluate_labels.py --dinov2 data/processed/dinov2_b.npz --alpha -1
uv run python studies/01-label-alignment/src/make_summary_figure.py

uv run python studies/02-thickness-signal-audit/src/build_features.py
uv run python studies/02-thickness-signal-audit/src/evaluate.py
uv run python studies/02-thickness-signal-audit/src/position_control.py
uv run python studies/02-thickness-signal-audit/src/make_figures.py

uv run python studies/03-alignment-iou/src/compute_iou.py
uv run python studies/03-alignment-iou/src/make_figures.py

uv run python studies/04-instrument-checks/src/check_calibration.py
uv run python studies/04-instrument-checks/src/sampling_ceiling.py
uv run python studies/04-instrument-checks/src/resolution_check.py
uv run python studies/04-instrument-checks/src/edge_check.py

uv run python studies/05-iou-closer-to-1/src/diagnose_worst.py
uv run python studies/05-iou-closer-to-1/src/run_study.py
uv run python studies/05-iou-closer-to-1/src/diagnostics.py
uv run python studies/05-iou-closer-to-1/src/make_figures.py
```

Set `MPLBACKEND=Agg` when running headless. Notebooks: regenerate with each
study's `src/make_notebook.py --write`, then
`python -m nbconvert --to notebook --execute --inplace <notebook>`.
