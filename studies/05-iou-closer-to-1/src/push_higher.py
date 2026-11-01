"""Can the IoU reach 0.8-0.9? What each remaining limit costs, and what would remove it.

Three kinds of experiment, labelled for what they can and cannot show:

* ORACLE - not methods. Replace one thing with the answer from the photo and see
  how high IoU goes: if the two instruments agreed on each pad's size, or on
  its centre, or both. This apportions the remaining gap between the causes.
* CALIBRATION - a method, but one that uses the photo as the size reference:
  each chip's systematic size difference between the instruments is measured
  on the *other* pads of that chip (leave-one-out) and removed.
* SIMULATION - what a finer scan would buy. Each photo outline is taken as the
  true pad, sampled the way the profilometer would at a given line spacing,
  rebuilt with this study's trace model, and compared with the original.

Writes results/push_higher.csv and results/finer_scan_simulation.csv.

Usage::

    MPLBACKEND=Agg uv run python studies/05-iou-closer-to-1/src/push_higher.py      # ~3 min
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib05 as L5  # noqa: E402
import run_study as RS  # noqa: E402

RESULTS = L5.STUDY / "results"
SPACINGS = (30.0, 20.0, 15.0, 10.0, 5.0)


def extents(m, upg):
    ys, xs = np.nonzero(m)
    return (np.percentile(xs, 98) - np.percentile(xs, 2)) * upg, (np.percentile(ys, 98) - np.percentile(ys, 2)) * upg


def rescale(m, fx, fy):
    """Stretch a binary mask about its own centroid by (fx, fy)."""
    cy, cx = ndi.center_of_mass(m)
    M = np.diag([1.0 / fy, 1.0 / fx])
    offset = np.array([cy, cx]) - M @ np.array([cy, cx])
    return ndi.affine_transform(m.astype(float), M, offset=offset, order=1) > 0.5


def centre_on(a, b):
    """Shift ``b`` so its centroid lands on ``a``'s (sub-pixel)."""
    ca, cb = np.array(ndi.center_of_mass(a)), np.array(ndi.center_of_mass(b))
    return ndi.shift(b.astype(float), ca - cb, order=1) > 0.5


def simulate_scan(truth, upg, spacing, sigma, rng):
    """Sample a true footprint the way the profilometer would and rebuild it with the trace model.

    Lines run along y (fine), spaced ``spacing`` across x at a random offset;
    the edge is blurred by ``sigma`` across lines, as the plate fits measured.
    The rebuilt pad uses the plate's true width as its prior (the best case).
    """
    g = truth.shape[0]
    u = (np.arange(g) - (g - 1) / 2) * upg
    blurred = ndi.gaussian_filter1d(truth.astype(float), sigma / upg, axis=1)
    phase = rng.uniform(0, spacing)
    xs = np.arange(u[0] + phase, u[-1], spacing)
    cols = np.clip(np.round((xs - u[0]) / upg).astype(int), 0, g - 1)
    traces = blurred[:, cols]                           # (rows, lines)
    tallest = traces.max(axis=0)
    if tallest.max() <= 0:
        return None
    inside = tallest > 0.5 * tallest.max()
    g_y = traces[:, inside].mean(axis=1)
    g_y = g_y / max(g_y.max(), 1e-9)
    half = u[g_y > 0.5]
    if half.size == 0:
        return None
    yc, length = 0.5 * (half.min() + half.max()), half.max() - half.min()
    mid = np.abs(u - yc) < max(0.25 * length, 6.0)
    uval = np.median(traces[mid], axis=0) / max(np.median(traces[mid][:, inside]), 1e-9)
    w_true = extents(truth, upg)[0]
    sel = uval > 0.3
    x0 = float(np.sum(xs[sel] * uval[sel]) / np.sum(uval[sel])) if sel.any() else 0.0
    w = L5._fit_width(xs, uval, x0, sigma, w_true, 6.0)
    across = L5._box(u[None, :] - x0, 0.0, w, sigma)
    rebuilt = np.minimum(across, g_y[:, None])
    return rebuilt >= 0.5


def main():
    from Auto_OPDx.alignment.rgb import load_rgb
    stems = L5.L.comparable_plates()
    cfg = replace(RS.FINAL, fast_background=True)
    rng = np.random.default_rng(0)
    rows = {k: [] for k in ("final", "size_oracle", "centre_oracle", "both_oracle", "calibrated")}
    rows_old = {"final": [], "calibrated": []}
    edge_abs = []
    sim = {s: [] for s in SPACINGS}
    for i, st in enumerate(stems, 1):
        b = L5.load_bundle(st)
        img = load_rgb(b.photo)
        field = L5.photo_field(img, "gray", True)
        wp = L5.plate_trace_model(b)
        for method in ("new", "old"):
            got = []
            for slot in range(64):
                o = L5.evaluate_cell(b, field, slot, method, cfg, img, wp)
                if o.status == "excluded_missing":
                    continue
                if o.status != "ok":
                    got.append((o, None, None, None))
                    continue
                pm, sm, upg = o.arrays["photo_mask"], o.arrays["scan_mask"], o.arrays["upg"]
                got.append((o, pm, sm, upg))
            # per-chip size ratios (photo / scan), each cell calibrated on the OTHER cells of its chip
            ratios = np.array([[e[0] / f[0], e[1] / f[1]] if pm is not None else [np.nan, np.nan]
                               for (o, pm, sm, upg) in got
                               for e, f in [((extents(pm, upg)), (extents(sm, upg))) if pm is not None else ((1, 1), (1, 1))]])
            for j, (o, pm, sm, upg) in enumerate(got):
                bucket = rows if method == "new" else rows_old
                if pm is None:
                    for k in bucket:
                        bucket[k].append(0.0)
                    continue
                others = np.delete(ratios, j, axis=0)
                fx, fy = np.nanmedian(others[:, 0]), np.nanmedian(others[:, 1])
                bucket["final"].append(o.iou)
                bucket["calibrated"].append(L5.iou(pm, rescale(sm, fx, fy)))
                if method == "new":
                    (pw, ph), (sw, sh) = extents(pm, upg), extents(sm, upg)
                    sized = rescale(sm, pw / sw, ph / sh)
                    rows["size_oracle"].append(L5.iou(pm, sized))
                    rows["centre_oracle"].append(L5.iou(pm, centre_on(pm, sm)))
                    rows["both_oracle"].append(L5.iou(pm, centre_on(pm, sized)))
                    pe, se = L5.edges_um(pm, upg), L5.edges_um(sm, upg)
                    edge_abs.append(np.abs(se - pe))
                    for s in SPACINGS:
                        rec = simulate_scan(pm, upg, s, wp[1], rng)
                        if rec is not None:
                            sim[s].append(L5.iou(pm, rec))
        print(f"  [{i}/{len(stems)}] {st}", flush=True)

    ea = np.array(edge_abs)
    out = [{"experiment": "final recipe", "kind": "measured", "new_mean_iou": np.mean(rows["final"]),
            "previous_mean_iou": np.mean(rows_old["final"])},
           {"experiment": "per-chip size calibration (leave-one-out)", "kind": "calibration",
            "new_mean_iou": np.mean(rows["calibrated"]), "previous_mean_iou": np.mean(rows_old["calibrated"])},
           {"experiment": "if both instruments agreed on each pad's size", "kind": "oracle",
            "new_mean_iou": np.mean(rows["size_oracle"])},
           {"experiment": "if both instruments agreed on each pad's centre", "kind": "oracle",
            "new_mean_iou": np.mean(rows["centre_oracle"])},
           {"experiment": "if they agreed on size and centre (only outline shape differs)", "kind": "oracle",
            "new_mean_iou": np.mean(rows["both_oracle"])}]
    for r in out:
        v = np.array(rows["final"] if r["kind"] == "measured" else rows["calibrated"] if r["kind"] == "calibration"
                     else rows[{"if both instruments agreed on each pad's size": "size_oracle",
                                "if both instruments agreed on each pad's centre": "centre_oracle"}.get(r["experiment"], "both_oracle")])
        r["new_frac_at_least_0p8"] = float(np.mean(v >= 0.8))
        r["new_frac_at_least_0p9"] = float(np.mean(v >= 0.9))
    with (RESULTS / "push_higher.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "kind", "new_mean_iou", "previous_mean_iou",
                                          "new_frac_at_least_0p8", "new_frac_at_least_0p9"], restval="")
        w.writeheader()
        w.writerows(out)
    srows = [{"line_spacing_um": s, "mean_iou_vs_true_outline": float(np.mean(v)), "cells": len(v)}
             for s, v in sim.items()]
    with (RESULTS / "finer_scan_simulation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)

    print(f"\nper-edge disagreement now (mean |scan - photo|, µm): left {ea[:,0].mean():.2f}, right {ea[:,1].mean():.2f}, "
          f"top {ea[:,2].mean():.2f}, bottom {ea[:,3].mean():.2f}")
    print(f"\n{'experiment':66} {'new':>6} {'prev':>6} {'>=0.8':>6} {'>=0.9':>6}")
    for r in out:
        print(f"{r['experiment']:66} {r['new_mean_iou']:6.3f} {r.get('previous_mean_iou', float('nan')) if r.get('previous_mean_iou') != '' else float('nan'):6.3f} "
              f"{100 * r['new_frac_at_least_0p8']:5.0f}% {100 * r['new_frac_at_least_0p9']:5.0f}%")
    print("\nsimulated: a perfect-photo pad, sampled at a given line spacing and rebuilt by the trace model")
    for r in srows:
        print(f"  {r['line_spacing_um']:4.0f} µm lines -> mean IoU {r['mean_iou_vs_true_outline']:.3f}  (n={r['cells']})")


if __name__ == "__main__":
    main()
