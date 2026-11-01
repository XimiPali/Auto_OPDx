"""Look at the cells that score worst, before changing anything.

Draws the N lowest-IoU cells of the current method (study 03, "new") with both
pad outlines and both pad centres, so the failure modes can be seen rather than
guessed. Writes figures/worst_cells_before.png.

Usage::

    MPLBACKEND=Agg uv run python studies/05-iou-closer-to-1/src/diagnose_worst.py [--n 24]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
S03 = REPO / "studies" / "03-alignment-iou"
sys.path.insert(0, str(S03 / "src"))

import iou_lib as L  # noqa: E402

NEW, INK, INK_2, INK_3, SURFACE, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb", "#e6e5e0"
BLUES = LinearSegmentedColormap.from_list("b", ["#f0efec", "#9ec5f4", "#3987e5", "#104281", "#0d366b"])

mpl.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 8,
                     "text.color": INK})


def worst_cells(n: int) -> list[tuple[str, int, int, float]]:
    with (S03 / "results" / "cells.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == "new" and r["status"] == "ok"]
    rows.sort(key=lambda r: float(r["iou"]))
    return [(r["plate"], int(r["row"]), int(r["col"]), float(r["iou"])) for r in rows[:n]]


def draw(ax_photo, ax_scan, res, window_um):
    px_um = window_um / (L.GRID - 1)
    ax_photo.imshow(res.photo_rgb)
    vmax = max(float(np.nanpercentile(res.hm_crop, 99.5)), 1.0)
    ax_scan.imshow(np.nan_to_num(res.hm_crop, nan=0.0), cmap=BLUES, vmin=0, vmax=vmax)
    for ax in (ax_photo, ax_scan):
        if res.photo_mask is not None:
            ax.contour(res.photo_mask.astype(float), levels=[0.5], colors=[NEW], linewidths=1.3)
        if res.hm_mask is not None:
            ax.contour(res.hm_mask.astype(float), levels=[0.5], colors=["#eb6834" if ax is ax_scan else INK],
                       linewidths=1.2, linestyles="dashed")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
    return px_um


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    picks = worst_cells(args.n)
    by_plate = defaultdict(list)
    for stem, r, c, v in picks:
        by_plate[stem].append((r, c, v))

    prof, img = L.load_old_metadata()
    results = {}
    for stem in by_plate:
        ev = L.evaluate_plate(stem, prof[stem], img[stem], keep_arrays=True)
        for x in ev.cells:
            if x.method == "new" and (x.row, x.col) in {(r, c) for r, c, _ in by_plate[stem]}:
                results[(stem, x.row, x.col)] = (x, ev.window_um)
        print(f"  {stem}: {len(by_plate[stem])} cells", flush=True)

    cols = 6
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.45, rows * 1.75),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.42})
    axes = np.atleast_2d(axes)
    for i, (stem, r, c, v) in enumerate(picks):
        rr, cc = divmod(i, cols)
        res, win = results[(stem, r, c)]
        draw(axes[rr, 2 * cc], axes[rr, 2 * cc + 1], res, win)
        axes[rr, 2 * cc].set_title(f"{stem} r{r}c{c}\nIoU {v:.2f} · offset {res.offset_um:.1f} µm",
                                   fontsize=7, loc="left", color=INK_2)
    for j in range(len(picks), rows * cols):
        rr, cc = divmod(j, cols)
        axes[rr, 2 * cc].axis("off")
        axes[rr, 2 * cc + 1].axis("off")
    fig.suptitle(f"The {len(picks)} lowest-IoU cells of the current method (photo | height map). "
                 "Blue = photo outline, dashed = height-map outline.", fontsize=10, y=0.995)
    out = STUDY / "figures" / "worst_cells_before.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
