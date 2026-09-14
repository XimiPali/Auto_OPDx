"""Build the rebuilt per-cell dataset and cache it.

For every plate with both an ``.OPDx`` scan and a photo, this extracts each of
the 64 cells and stores, per cell:

* the seven brightness statistics the results deck used in its Phase 1
  (mean R, G, B; mean / max / 95th-percentile / std of grayscale)
* ``label_new`` -- plane-relative max height, um. The corrected label.
* ``label_old`` -- raw ``z_max`` in the same window, um. Reproduces the previous
  label so the two can be compared with everything else held constant.
* grid position, crop overhang, and per-plate registration diagnostics

Optionally also writes the RGB and height crops themselves, for feeding a
stronger feature extractor (e.g. DINOv2) later.

Usage::

    uv run python scripts/03_build_dataset.py
    uv run python scripts/03_build_dataset.py --save-crops --limit 5
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from Auto_OPDx.alignment.dataset import (  # noqa: E402
    build_cells,
    fit_cross_modal,
    solve_photo,
    solve_plate,
)

ROWS = COLS = 8
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def brightness_features(rgb: np.ndarray) -> dict[str, float]:
    """The seven intensity statistics from the deck's Phase 1 baseline."""
    a = np.asarray(rgb, float)
    gray = a.mean(axis=2)
    return {
        "mean_r": float(a[:, :, 0].mean()),
        "mean_g": float(a[:, :, 1].mean()),
        "mean_b": float(a[:, :, 2].mean()),
        "gray_mean": float(gray.mean()),
        "gray_max": float(gray.max()),
        "gray_p95": float(np.percentile(gray, 95)),
        "gray_std": float(gray.std()),
    }


def old_style_label(hm, row: int, col: int, half_um: float = 55.0) -> float:
    """Raw ``z_max`` in the cell window -- i.e. the previous label definition."""
    cx, cy = hm.fit.predict((row, col))[0]
    xi = np.where(np.abs(hm.plate.x - cx) <= half_um)[0]
    yi = np.where(np.abs(hm.plate.y - cy) <= half_um)[0]
    if xi.size == 0 or yi.size == 0:
        return float("nan")
    return float(hm.plate.z[np.ix_(yi, xi)].max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--label-um", type=float, default=110.0)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "processed")
    args = ap.parse_args()

    rgb_by_stem = {
        p.stem: p
        for p in (REPO / "data" / "rgb" / "img").iterdir()
        if p.suffix.lower() in IMG_EXT
    }
    plates = sorted((REPO / "data" / "opdx").glob("*.OPDx"))
    if args.limit:
        plates = plates[: args.limit]

    recs: list[dict] = []
    rgb_crops: list[np.ndarray] = []
    h_crops: list[np.ndarray] = []
    skipped: list[str] = []

    for i, path in enumerate(plates, 1):
        stem = path.stem
        img_path = rgb_by_stem.get(stem)
        if img_path is None:
            skipped.append(f"{stem}: no photo")
            continue
        try:
            hm = solve_plate(path, ROWS, COLS)
            ph = solve_photo(img_path, ROWS, COLS)
            cm = fit_cross_modal(hm, ph, COLS)
            cells = build_cells(hm, ph, cm, ROWS, COLS, label_um=args.label_um)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{stem}: {type(e).__name__}: {e}")
            continue

        for c in cells:
            if c.rgb is None or "h_max" not in c.labels:
                continue
            rec = {
                "plate": stem, "row": c.row, "col": c.col,
                "label_new": c.labels["h_max"],
                "label_new_p95": c.labels["h_p95"],
                "label_old": old_style_label(hm, c.row, c.col),
                "overhang": c.overhang,
                "hm_rmse_um": hm.fit.rmse,
                "cross_rmse_um": cm.rmse_um,
                "tilt_pp_um": hm.tilt["tilt_pp_um"],
            }
            rec.update(brightness_features(c.rgb))
            recs.append(rec)
            if args.save_crops:
                rgb_crops.append(c.rgb)
                h_crops.append(np.nan_to_num(c.height, nan=0.0).astype(np.float32))

        if i % 10 == 0 or i == len(plates):
            print(f"  {i}/{len(plates)} plates, {len(recs)} cells")

    if not recs:
        sys.exit("no cells produced; check data/ layout")

    args.out.mkdir(parents=True, exist_ok=True)
    keys = list(recs[0].keys())
    csv_path = args.out / "cells.csv"
    with csv_path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in recs:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    plates_done = sorted({r["plate"] for r in recs})
    print(f"\ncells: {len(recs)} across {len(plates_done)} plates")
    print(f"wrote {csv_path.relative_to(REPO)}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped[:10]:
            print(f"  {s}")

    if args.save_crops:
        npz = args.out / "crops.npz"
        np.savez_compressed(
            npz,
            rgb=np.stack(rgb_crops),
            height=np.stack(h_crops),
            names=np.array([f"{r['plate']}_r{r['row']:02d}_c{r['col']:02d}" for r in recs]),
            label_new=np.array([r["label_new"] for r in recs], np.float32),
            label_old=np.array([r["label_old"] for r in recs], np.float32),
        )
        print(f"wrote {npz.relative_to(REPO)} "
              f"({npz.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
