"""Score photo/height-map alignment per cell with IoU, previous crops vs new.

Runs every plate the previous pipeline produced both a photo and a scan result
for (45 - the same 45 the earlier model trained on), evaluates each method with
the shared recipe in ``iou_lib`` and writes:

    results/cells.csv        one row per cell per method
    results/plates.csv       one row per plate per method
    results/summary.csv      pooled numbers, the table in the README
    results/unit_conversion.csv  how well the previous scan coordinates map to um

Usage::

    uv run python studies/03-alignment-iou/src/compute_iou.py
    uv run python studies/03-alignment-iou/src/compute_iou.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import iou_lib as L

RESULTS = L.STUDY / "results"


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _summarise(cells: list[dict], method: str) -> dict:
    """Pooled statistics for one method over the cells that have a pad."""
    mine = [c for c in cells if c["method"] == method]
    scored = [c for c in mine if c["status"] != "excluded_missing"]
    ious = np.array([c["iou"] for c in scored], float)
    ok = [c for c in scored if c["status"] == "ok"]
    ceil = np.array([c["ceiling_iou"] for c in ok], float)
    off = np.array([c["offset_um"] for c in ok], float)
    n = len(scored)
    return {
        "method": method,
        "label": L.METHOD_LABEL[method],
        "cells_with_pad": n,
        "cells_excluded_missing": len(mine) - n,
        "mean_iou": float(ious.mean()) if n else np.nan,
        "median_iou": float(np.median(ious)) if n else np.nan,
        "p10_iou": float(np.percentile(ious, 10)) if n else np.nan,
        "frac_iou_zero": float(np.mean(ious == 0)) if n else np.nan,
        "frac_iou_below_0p3": float(np.mean(ious < 0.3)) if n else np.nan,
        "n_not_found": sum(c["status"] == "not_found" for c in scored),
        "n_no_pad_in_crop": sum(c["status"] == "no_pad_in_crop" for c in scored),
        "n_failures": sum(c["status"] != "ok" for c in scored),
        "mean_ceiling_iou": float(ceil.mean()) if len(ceil) else np.nan,
        "mean_iou_over_ceiling": float(np.mean([c["iou"] / c["ceiling_iou"] for c in ok if c["ceiling_iou"] > 0])) if ok else np.nan,
        "median_offset_um": float(np.median(off)) if len(off) else np.nan,
        "p90_offset_um": float(np.percentile(off, 90)) if len(off) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    prof, img = L.load_old_metadata()
    stems = L.comparable_plates()
    if args.limit:
        stems = stems[: args.limit]
    print(f"{len(stems)} plates have results from both pipelines")

    cells: list[dict] = []
    plates: list[dict] = []
    conv_rows: list[dict] = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        t = time.time()
        try:
            ev = L.evaluate_plate(stem, prof[stem], img[stem])
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(stems)}] {stem:18} FAILED {type(e).__name__}: {e}")
            continue
        conv_rows.append({
            "plate": stem, "rmse_um": ev.conv.rmse_um, "n_used": ev.conv.n_used,
            "n_total": ev.conv.n_total, "um_per_unit_x": ev.conv.um_per_unit[0],
            "um_per_unit_y": ev.conv.um_per_unit[1], "rotation_deg": ev.conv.rotation_deg,
        })
        for c in ev.cells:
            cells.append({
                "plate": c.plate, "row": c.row, "col": c.col, "method": c.method,
                "status": c.status, "iou": round(c.iou, 4), "ceiling_iou": round(c.ceiling_iou, 4),
                "offset_um": round(c.offset_um, 2) if np.isfinite(c.offset_um) else "",
                "present_hm": bool(ev.present_hm[c.row, c.col]),
                "present_photo": bool(ev.present_photo[c.row, c.col]),
            })
        line = []
        for m in L.METHODS:
            mine = [c for c in ev.cells if c.method == m and c.status != "excluded_missing"]
            ious = np.array([c.iou for c in mine], float)
            plates.append({
                "plate": stem, "method": m, "n_scored": len(mine),
                "n_excluded": int((~(ev.present_hm & ev.present_photo)).sum()),
                "mean_iou": float(ious.mean()) if len(ious) else np.nan,
                "median_iou": float(np.median(ious)) if len(ious) else np.nan,
                "n_zero": int((ious == 0).sum()),
                "n_failures": sum(c.status != "ok" for c in mine),
                "pitch_um": ev.pitch_um, "window_um": ev.window_um,
            })
            line.append(f"{m} {ious.mean():.3f}" if len(ious) else f"{m} -")
        print(f"  [{i}/{len(stems)}] {stem:18} {' · '.join(line)} · "
              f"missing {int((~(ev.present_hm & ev.present_photo)).sum())} · {time.time()-t:4.1f}s")

    summary = [_summarise(cells, m) for m in L.METHODS]
    _write(RESULTS / "cells.csv", cells)
    _write(RESULTS / "plates.csv", plates)
    _write(RESULTS / "summary.csv", summary)
    _write(RESULTS / "unit_conversion.csv", conv_rows)

    print("\n" + "=" * 72)
    print(f"IoU between the photo pad and the height-map pad, {len(conv_rows)} plates, {time.time()-t0:.0f}s")
    print("=" * 72)
    print(f"{'method':34} {'mean':>6} {'median':>7} {'zero':>6} {'<0.3':>6} {'fail':>5} {'ceil':>6} {'off um':>7}")
    for s in summary:
        print(f"{s['label']:34} {s['mean_iou']:6.3f} {s['median_iou']:7.3f} "
              f"{s['frac_iou_zero']*100:5.1f}% {s['frac_iou_below_0p3']*100:5.1f}% {s['n_failures']:5d} "
              f"{s['mean_ceiling_iou']:6.3f} {s['median_offset_um']:7.1f}")
    s0 = summary[0]
    print(f"\ncells with a pad in both instruments: {s0['cells_with_pad']}  "
          f"excluded (pad missing): {s0['cells_excluded_missing']}")
    c = np.array([r["rmse_um"] for r in conv_rows])
    print(f"previous scan coordinates -> um: affine residual median {np.median(c):.1f} um, max {c.max():.1f} um")
    print(f"\nwrote {RESULTS.relative_to(L.REPO)}/{{cells,plates,summary,unit_conversion}}.csv")


if __name__ == "__main__":
    main()
