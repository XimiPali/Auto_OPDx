"""Where do the photo outline and the height-map outline disagree, edge by edge?

Checks the claim that the scan's coarse trace spacing (~30 um across the pads,
x in the crop) is what keeps IoU below 1. If it were, the two outlines would
disagree much more on the left/right edges (coarse axis) than on the top/bottom
edges (fine axis, 2.9 um). They do not, which retracts that claim; see the
README, section 2.

For every scored cell, each outline's four edges are taken as the 2nd and 98th
percentiles of its pixel coordinates (robust to a stray pixel), and the scan
edge is compared with the photo edge. Uses the study-03 crops and masks
unchanged.

Writes results/edge_disagreement.csv.

Usage::

    MPLBACKEND=Agg uv run python studies/04-instrument-checks/src/edge_check.py
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
sys.path.insert(0, str(REPO / "studies" / "03-alignment-iou" / "src"))

import iou_lib as L  # noqa: E402

EDGES = ("left", "right", "top", "bottom")
AXIS = {"left": "coarse (≈30 µm traces)", "right": "coarse (≈30 µm traces)",
        "top": "fine (2.9 µm steps)", "bottom": "fine (2.9 µm steps)"}


def outline_edges(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return (np.percentile(xs, 2), np.percentile(xs, 98), np.percentile(ys, 2), np.percentile(ys, 98))


def main() -> None:
    prof, img = L.load_old_metadata()
    stems = L.comparable_plates()
    diffs = {m: {e: [] for e in EDGES} for m in ("old", "new")}
    for i, stem in enumerate(stems, 1):
        ev = L.evaluate_plate(stem, prof[stem], img[stem], keep_arrays=True)
        um = ev.window_um / (L.GRID - 1)
        for c in ev.cells:
            if c.method not in diffs or c.status != "ok":
                continue
            p = outline_edges(c.photo_mask)
            h = outline_edges(c.hm_mask)
            for e, pv, hv in zip(EDGES, p, h):
                diffs[c.method][e].append((hv - pv) * um)
        print(f"  [{i}/{len(stems)}] {stem}", flush=True)

    rows = []
    for m in ("old", "new"):
        for e in EDGES:
            v = np.array(diffs[m][e])
            rows.append({"method": m, "edge": e, "scan_axis": AXIS[e], "n_cells": len(v),
                         "mean_signed_um": float(v.mean()), "mean_abs_um": float(np.abs(v).mean()),
                         "median_abs_um": float(np.median(np.abs(v)))})
    out = STUDY / "results" / "edge_disagreement.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nscan outline minus photo outline, per edge (µm)")
    print(f"{'method':7} {'edge':7} {'scan axis':24} {'signed':>8} {'mean |d|':>9} {'median |d|':>11}")
    for r in rows:
        print(f"{r['method']:7} {r['edge']:7} {r['scan_axis']:24} {r['mean_signed_um']:+8.1f} "
              f"{r['mean_abs_um']:9.1f} {r['median_abs_um']:11.1f}")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
