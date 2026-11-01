"""Study 05: run the final progression, the rejected ideas, and the independent alignment check.

Writes
    results/progression.csv        mean IoU per step, both methods (the headline table)
    results/rejected.csv           ideas that were tried and not kept, with their measured effect
    results/levels.csv             outline-threshold sensitivity, chosen by two-fold CV over plates
    results/pairing_error.csv      independent alignment check: photo vs scan centre, per method
    results/moved_centres.csv      recipes that move crop centres: IoU against real pairing error
    results/trace_offset.csv       scan width vs where the pad sits between traces, before/after
    results/cells_baseline.csv     per-cell results, study-03 recipe
    results/cells_final.csv        per-cell results, final recipe

Every step is applied to *both* methods unless marked "new only". The cell set
is study 03's (2,855 cells on 45 plates) throughout.

Usage::

    MPLBACKEND=Agg uv run python studies/05-iou-closer-to-1/src/run_study.py      # ~4 min
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib05 as L5  # noqa: E402

RESULTS = L5.STUDY / "results"

BASE = L5.Config("baseline")                                   # study 03, exactly
STEPS = [
    (BASE, "study-03 recipe"),
    (replace(BASE, name="finer_grid", grid=192), "finer grid (1.25 µm per pixel instead of 2.5)"),
    (replace(BASE, name="cleanup", grid=192, closing_um=2.5, opening_um=5.0),
     "outline clean-up (close gaps in the photo's dark ring; trim spurs after filling)"),
    (replace(BASE, name="scan_from_traces", grid=192, closing_um=2.5, opening_um=5.0, scan_model="traces",
             model_shape="min"),
     "scan pad rebuilt from its traces (plate width + edge softness, scan data only)"),
    (replace(BASE, name="centre_threshold", grid=192, closing_um=2.5, opening_um=5.0, scan_model="traces",
             model_shape="min", peak_region="centre", peak_median_um=2.5),
     "outline threshold read from the pad at the centre, after a small median filter (debris, specks)"),
    (replace(BASE, name="final", grid=192, closing_um=2.5, opening_um=5.0, scan_model="traces",
             model_shape="min", peak_region="centre", peak_median_um=2.5, photo_hull=True,
             hull_below_solidity=0.8),
     "broken photo rims joined: convex hull when the outline is under 80% solid"),
]
FINAL = STEPS[-1][0]

REJECTED = [
    (replace(FINAL, name="x_rotate", rotate=True), "rotate the photo crop into the scan's frame"),
    (replace(FINAL, name="x_local_plane", local_plane=True), "subtract a background plane inside each crop"),
    (replace(FINAL, name="x_smooth", photo_smooth_um=1.25), "smooth the photo crop before outlining"),
    (replace(FINAL, name="x_green", channel="green"), "outline the photo in the green channel"),
    (replace(FINAL, name="x_hull_all", hull_below_solidity=1.01),
     "convex hull for every photo outline, not just broken rims"),
    (replace(FINAL, name="x_check", consistency_um=25.0), "cross-check the two detections (new only)"),
    (replace(FINAL, name="x_model_at_detection", model_position="detection"),
     "place the rebuilt scan pad at the plate detection (builds the new crop centre into the measurement)"),
    (replace(FINAL, name="x_second_pass", second_pass="centroid"),
     "second pass: move the photo crop so the two outline centres meet (new only)"),
    (replace(FINAL, name="x_recentre_fit", scan_centre_from_model=True, model_position="fit"),
     "re-centre the height crop on the fitted pad centre (new only)"),
]

LEVELS = (0.3, 0.4, 0.5, 0.6, 0.7)


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cell_rows(outs, config):
    keep = []
    for o in outs:
        if o.config != config or o.status == "excluded_missing":
            continue
        keep.append({"plate": o.plate, "row": o.row, "col": o.col, "method": o.method, "status": o.status,
                     "iou": round(o.iou, 4), "ceiling": round(o.ceiling, 4),
                     "offset_um": round(o.offset_um, 2) if np.isfinite(o.offset_um) else "",
                     "photo_w_um": round(o.photo_edges_um[1] - o.photo_edges_um[0], 2) if o.status == "ok" else "",
                     "photo_h_um": round(o.photo_edges_um[3] - o.photo_edges_um[2], 2) if o.status == "ok" else "",
                     "scan_w_um": round(o.scan_edges_um[1] - o.scan_edges_um[0], 2) if o.status == "ok" else "",
                     "scan_h_um": round(o.scan_edges_um[3] - o.scan_edges_um[2], 2) if o.status == "ok" else "",
                     "scan_x_um": round(o.scan_centre_um[0], 2) if o.status == "ok" else ""})
    return keep


# ------------------------------------------------- independent alignment check ----

def _design(P, order=3):
    x, y = P[:, 0] / 1000.0, P[:, 1] / 1000.0
    cols = [np.ones_like(x), x, y, x * x, x * y, y * y]
    if order >= 3:
        cols += [x ** 3, x * x * y, x * y * y, y ** 3]
    return np.stack(cols, 1)


def _robust_loo(P, S):
    """Leave-one-out residual of S against a cubic map of P, with 3-MAD trimming of the fitting pads."""
    n = len(P)
    out = np.zeros_like(S)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        keep = m.copy()
        for _ in range(3):
            A = _design(P[keep])
            coef, *_ = np.linalg.lstsq(A, S[keep], rcond=None)
            r = np.linalg.norm(S - _design(P) @ coef, axis=1)
            med = np.median(r[keep])
            mad = 1.4826 * np.median(np.abs(r[keep] - med)) + 1e-9
            keep = m & (r < med + 3 * mad)
        out[i] = S[i] - _design(P[i:i + 1]) @ coef
    return out


def pairing_error(stems):
    """Where each method's photo centre and scan centre disagree, after a smooth plate-wide map.

    This does not use the outlines or the crops at all, so it cannot be raised by
    drawing outlines differently. It includes the profilometer's per-column stage
    shifts (shared by both methods).
    """
    rows = []
    for method in ("old", "new"):
        errs = []
        for st in stems:
            b = L5.load_bundle(st)
            pts = [(b.centres[method][s][:2], b.centres[method][s][2:]) for s in range(64)
                   if b.status[method][s] not in ("excluded_missing", "not_found")
                   and np.isfinite(b.centres[method][s]).all()]
            if len(pts) < 20:
                continue
            P = np.array([p for p, _ in pts])
            S = np.array([q for _, q in pts])
            errs.append(np.linalg.norm(_robust_loo(P, S), axis=1))
        e = np.concatenate(errs)
        rows.append({"method": method, "pairs": len(e), "median_um": float(np.median(e)),
                     "p90_um": float(np.percentile(e, 90)), "frac_over_20um": float(np.mean(e > 20)),
                     "frac_over_50um": float(np.mean(e > 50))})
    return rows


def trace_offset_table(outs, configs):
    xs = {st: (L5.load_bundle(st).x[0], float(np.median(np.diff(L5.load_bundle(st).x))))
          for st in {o.plate for o in outs}}
    rows = []
    for cfg in configs:
        ok = [o for o in outs if o.config == cfg and o.method == "new" and o.status == "ok"]
        ratio = np.array([(o.scan_edges_um[1] - o.scan_edges_um[0]) / (o.photo_edges_um[1] - o.photo_edges_um[0])
                          for o in ok])
        iou = np.array([o.iou for o in ok])
        ph = []
        for o in ok:
            x0, dx = xs[o.plate]
            f = ((o.scan_centre_um[0] - x0) / dx) % 1.0
            ph.append(min(f, 1 - f))
        ph = np.array(ph)
        for lo in (0.0, 0.1, 0.2, 0.3, 0.4):
            m = (ph >= lo) & (ph < lo + 0.1)
            rows.append({"config": cfg, "offset_from_trace_lo": lo, "offset_from_trace_hi": lo + 0.1,
                         "cells": int(m.sum()), "scan_over_photo_width": float(np.median(ratio[m])),
                         "mean_iou": float(iou[m].mean())})
    return rows


def main() -> None:
    t0 = time.time()
    stems = L5.L.comparable_plates()
    L5.ensure_cache(stems)

    configs = [c for c, _ in STEPS] + [c for c, _ in REJECTED]
    outs = L5.run(stems, configs)
    print(f"evaluated {len(configs)} recipes in {time.time() - t0:.0f}s")

    prog = []
    for cfg, label in STEPS:
        so, sn = L5.summarise(outs, cfg.name, "old"), L5.summarise(outs, cfg.name, "new")
        prog.append({"step": cfg.name, "what": label, "previous_mean_iou": so["mean_iou"],
                     "new_mean_iou": sn["mean_iou"], "new_median_iou": sn["median_iou"],
                     "new_ceiling": sn["mean_ceiling"], "previous_cells_zero": so["n_zero"],
                     "new_cells_zero": sn["n_zero"], "new_frac_below_0p5": sn["frac_below_0p5"],
                     "previous_frac_below_0p5": so["frac_below_0p5"]})
    write(RESULTS / "progression.csv", prog)

    fs = {"old": L5.summarise(outs, FINAL.name, "old")["mean_iou"], "new": L5.summarise(outs, FINAL.name, "new")["mean_iou"]}
    rej = []
    for cfg, label in REJECTED:
        so, sn = L5.summarise(outs, cfg.name, "old"), L5.summarise(outs, cfg.name, "new")
        rej.append({"idea": cfg.name, "what": label, "previous_mean_iou": so["mean_iou"], "new_mean_iou": sn["mean_iou"],
                    "change_previous": so["mean_iou"] - fs["old"], "change_new": sn["mean_iou"] - fs["new"],
                    "new_ceiling": sn["mean_ceiling"]})
    write(RESULTS / "rejected.csv", rej)

    write(RESULTS / "cells_baseline.csv", cell_rows(outs, BASE.name))
    write(RESULTS / "cells_final.csv", cell_rows(outs, FINAL.name))
    write(RESULTS / "trace_offset.csv", trace_offset_table(outs, ["finer_grid", FINAL.name]))

    # outline-threshold sensitivity (fast background: only the masks change)
    lv_cfgs = [replace(FINAL, name=f"p{p}_s{s}", photo_level=p, scan_level=s, fast_background=True)
               for p in LEVELS for s in LEVELS]
    lv = L5.run(stems, lv_cfgs)
    fold = {st: i % 2 for i, st in enumerate(stems)}

    def mean(cfg, methods=("old", "new"), f=None):
        return float(np.mean([o.iou for o in lv if o.config == cfg and o.method in methods
                              and o.status != "excluded_missing" and (f is None or fold[o.plate] == f)]))
    lrows = []
    for c in lv_cfgs:
        lrows.append({"photo_level": c.photo_level, "scan_level": c.scan_level,
                      "previous_mean_iou": mean(c.name, ("old",)), "new_mean_iou": mean(c.name, ("new",))})
    for f in (0, 1):
        best = max((c.name for c in lv_cfgs), key=lambda n: mean(n, f=1 - f))
        lrows.append({"photo_level": f"cv fold {f}: chosen on the other half", "scan_level": best,
                      "previous_mean_iou": mean(best, ("old",), f), "new_mean_iou": mean(best, ("new",), f)})
    write(RESULTS / "levels.csv", lrows)

    pe = pairing_error(stems)
    write(RESULTS / "pairing_error.csv", pe)

    # recipes that move the new method's crop centres: their real pairing error, same check
    moved = []
    for name in (FINAL.name, "x_second_pass", "x_recentre_fit"):
        errs = []
        for st in stems:
            mine = [o for o in outs if o.config == name and o.method == "new" and o.plate == st and o.status == "ok"]
            if len(mine) >= 20:
                P = np.array([o.photo_crop_px for o in mine])
                S = np.array([o.scan_crop_um for o in mine])
                errs.append(np.linalg.norm(_robust_loo(P, S), axis=1))
        e = np.concatenate(errs)
        moved.append({"recipe": name, "new_mean_iou": L5.summarise(outs, name, "new")["mean_iou"],
                      "pairing_median_um": float(np.median(e)), "pairing_p90_um": float(np.percentile(e, 90))})
    write(RESULTS / "moved_centres.csv", moved)

    print("\nprogression (mean IoU over 2,855 cells; previous | new | new ceiling)")
    for r in prog:
        print(f"  {r['step']:18} {r['previous_mean_iou']:.4f} | {r['new_mean_iou']:.4f} | {r['new_ceiling']:.4f}   {r['what']}")
    print("\nrejected (change in mean IoU vs final; previous | new)")
    for r in rej:
        print(f"  {r['idea']:16} {r['change_previous']:+.4f} | {r['change_new']:+.4f}   {r['what']}")
    print("\nlevel sensitivity: best by CV ->", [r for r in lrows if str(r["photo_level"]).startswith("cv")])
    print("\nindependent pairing error (photo centre vs scan centre after a smooth map):")
    for r in pe:
        print(f"  {r['method']}: median {r['median_um']:.2f} µm, 90th pct {r['p90_um']:.2f} µm, "
              f"> 20 µm {100 * r['frac_over_20um']:.1f}%, > 50 µm {100 * r['frac_over_50um']:.1f}%  (n={r['pairs']})")
    print("\nrecipes that move crop centres: IoU vs real pairing error")
    for r in moved:
        print(f"  {r['recipe']:16} IoU {r['new_mean_iou']:.4f}  pairing median {r['pairing_median_um']:.2f} µm, "
              f"90th {r['pairing_p90_um']:.2f} µm")
    print(f"\ndone in {time.time() - t0:.0f}s; wrote {RESULTS.relative_to(L5.REPO)}/")


if __name__ == "__main__":
    main()
