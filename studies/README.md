# Studies

Two investigations built on the `Auto_OPDx.alignment` package. Each is a
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
```

Set `MPLBACKEND=Agg` when running headless. Notebooks: regenerate with each
study's `src/make_notebook.py --write`, then
`python -m nbconvert --to notebook --execute --inplace <notebook>`.
