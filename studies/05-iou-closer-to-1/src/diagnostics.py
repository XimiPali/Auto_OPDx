"""The checks behind study 05's decisions, none of which can be raised by redrawing outlines.

    results/plate_trace_model.csv   per plate: pad width and edge softness across traces (scan data only)
    results/centre_estimators.csv   which pad-centre estimate is most accurate, judged against each
                                    instrument's own rigid grid (lower residual = more accurate)
    results/stage_shifts.csv        how much of the photo-vs-scan disagreement is shared by a whole
                                    grid column (the profilometer's stage) vs. per pad
    results/pinned_pads.csv         on pads whose two edges both fall on a trace: is the scan
                                    detection or the photo-predicted position closer to the truth?
    results/loss_decomposition.csv  where the gap from IoU = 1 goes: alignment, size, outline shape

Usage::

    MPLBACKEND=Agg uv run python studies/05-iou-closer-to-1/src/diagnostics.py      # ~3 min
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib05 as L5  # noqa: E402
import run_study as RS  # noqa: E402

RESULTS = L5.STUDY / "results"
rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731


def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def grid_resid(pts, rc):
    """Residual from the same 10-parameter rigid grid model the lattice solver uses."""
    A = np.c_[rc[:, 0], rc[:, 1], rc[:, 0] >= 4, rc[:, 1] >= 4, np.ones(len(rc))].astype(float)
    out = np.zeros_like(pts, dtype=float)
    for k in range(2):
        coef, *_ = np.linalg.lstsq(A, pts[:, k], rcond=None)
        out[:, k] = pts[:, k] - A @ coef
    return out


def minus_group_mean(v, g):
    return v - np.array([v[g == k].mean() for k in g])


def group_share(v, g):
    m = np.array([v[g == k].mean() for k in g])
    return float(np.var(m) / np.var(v))


def detected(b, slot):
    c = b.centres["new"][slot]
    return not (np.allclose(c[:2], b.slot_px[slot]) or np.allclose(c[2:], b.slot_um[slot]))


def main():
    from Auto_OPDx.alignment.rgb import load_rgb
    stems = L5.L.comparable_plates()
    cfg = replace(RS.FINAL, fast_background=True)

    trace_rows, est, xm = [], {k: [] for k in ("p_det", "p_out", "p_mid", "s_det", "s_out", "s_mid", "s_fit")}, []
    chan = {ch: [] for ch in ("gray", "red", "green", "blue")}
    pinned = {"detection": [], "photo_prediction": []}
    for i, st in enumerate(stems, 1):
        b = L5.load_bundle(st)
        img = load_rgb(b.photo)
        field = L5.photo_field(img, "gray", True)
        w_plate, sigma = L5.plate_trace_model(b)
        trace_rows.append({"plate": st, "pad_width_across_traces_um": w_plate, "edge_softness_sigma_um": sigma})
        upg = b.window_um / (cfg.grid - 1)
        J = np.diag(1.0 / b.um_per_px)
        rows = []
        for slot in range(64):
            if b.status["new"][slot] == "excluded_missing" or not detected(b, slot):
                continue
            c = b.centres["new"][slot]
            _, pm = L5.photo_outline(L5.crop_photo(field, b, c[:2], cfg), cfg, upg)
            _, sm = L5.scan_outline(L5.crop_scan(b, c[2:], cfg), cfg, upg)
            smp = L5._trace_samples(b, c[2], c[3])
            if pm is None or sm is None or smp is None:
                continue
            xs, u = smp[0], smp[1]
            x0, _ = L5.fit_across_traces(xs, u, c[2], sigma, w_plate, cfg.width_prior_um)
            pe = L5.edges_um(pm, upg) - b.window_um / 2
            se = L5.edges_um(sm, upg) - b.window_um / 2
            pin = bool(((xs < x0) & (u > 0.12) & (u < 0.88)).any() and ((xs > x0) & (u > 0.12) & (u < 0.88)).any())
            rows.append({"rc": divmod(slot, 8), "p_det": c[:2], "p_out": c[:2] + J @ L5.centroid_um(pm, upg),
                         "p_mid": c[:2] + J @ np.array([(pe[0] + pe[1]) / 2, (pe[2] + pe[3]) / 2]),
                         "s_det": c[2:], "s_out": c[2:] + L5.centroid_um(sm, upg),
                         "s_mid": c[2:] + np.array([(se[0] + se[1]) / 2, (se[2] + se[3]) / 2]),
                         "s_fit": np.array([x0, smp[4]]), "pinned": pin, "x0": x0})
        if len(rows) < 20:
            continue
        rc = np.array([r["rc"] for r in rows])
        scale = float(np.sqrt(np.prod(b.um_per_px)))
        for k in est:
            pts = np.array([r[k] for r in rows], float) * (scale if k.startswith("p_") else 1.0)
            g = grid_resid(pts, rc)
            if k.startswith("s_"):            # remove the profilometer's per-column / per-row stage shifts
                g = np.c_[minus_group_mean(g[:, 0], rc[:, 1]), minus_group_mean(g[:, 1], rc[:, 0])]
            est[k].append(g)
        # the same height-weighted centroid computed in each colour channel, around each detection
        from scipy import ndimage as _ndi
        fields = {ch: (field if ch == "gray" else L5.photo_field(img, ch, True)) for ch in chan}
        for ch, f in fields.items():
            pts, rcs = [], []
            for row in rows:
                x, y = (int(round(v)) for v in row["p_det"])
                h = 70
                if x - h < 0 or y - h < 0 or x + h >= f.shape[1] or y + h >= f.shape[0]:
                    continue
                win = f[y - h:y + h, x - h:x + h]
                bg, pk = np.median(win), np.percentile(win, 99.5)
                m = win > bg + 0.3 * (pk - bg)
                lab, n = _ndi.label(m)
                if n == 0:
                    continue
                k = lab[h, h] if lab[h, h] > 0 else 1 + int(np.argmax(_ndi.sum(m, lab, range(1, n + 1))))
                wts = (win - bg) * (lab == k)
                yy, xx = np.mgrid[-h:h, -h:h]
                pts.append([x + (xx * wts).sum() / wts.sum(), y + (yy * wts).sum() / wts.sum()])
                rcs.append(row["rc"])
            if len(pts) >= 20:
                chan[ch].append(grid_resid(np.array(pts) * scale, np.array(rcs)))
        # photo vs scan after a smooth map (leave-one-out), and its per-column / per-row parts
        P = np.array([r["p_det"] for r in rows])
        S = np.array([r["s_det"] for r in rows])
        r = RS._robust_loo(P, S)
        xm.append((r, rc, grid_resid(S, rc)[:, 0], grid_resid(P * scale, rc)[:, 0]))
        # pinned pads: scan detection vs photo + plate geometry (map + that column's stage shift)
        pred = S[:, 0] - r[:, 0]
        for j, row in enumerate(rows):
            if row["pinned"]:
                same = (rc[:, 1] == rc[j, 1]) & (np.arange(len(rows)) != j)
                shift = float(np.median(r[same, 0])) if same.sum() >= 3 else 0.0
                pinned["detection"].append(abs(S[j, 0] - row["x0"]))
                pinned["photo_prediction"].append(abs(pred[j] + shift - row["x0"]))
        print(f"  [{i}/{len(stems)}] {st}", flush=True)

    write(RESULTS / "plate_trace_model.csv", trace_rows)

    labels = {"p_det": ("photo", "plate detection (intensity-weighted centroid)"),
              "p_out": ("photo", "outline centroid"), "p_mid": ("photo", "outline edge midpoint"),
              "s_det": ("scan", "plate detection (intensity-weighted centroid)"),
              "s_out": ("scan", "outline centroid"), "s_mid": ("scan", "outline edge midpoint"),
              "s_fit": ("scan", "fit across traces (x) / half-height midpoint (y)")}
    erows = []
    for k, (inst, what) in labels.items():
        a = np.concatenate(est[k])
        erows.append({"instrument": inst, "estimator": what, "grid_residual_x_um": rms(a[:, 0]),
                      "grid_residual_y_um": rms(a[:, 1]), "pads": len(a)})
    for ch, parts in chan.items():
        a = np.concatenate(parts)
        erows.append({"instrument": "photo", "estimator": f"height-weighted centroid, {ch} channel",
                      "grid_residual_x_um": rms(a[:, 0]), "grid_residual_y_um": rms(a[:, 1]), "pads": len(a)})
    write(RESULTS / "centre_estimators.csv", erows)

    rx = np.concatenate([m[0][:, 0] for m in xm])
    ry = np.concatenate([m[0][:, 1] for m in xm])
    rx_own = np.concatenate([minus_group_mean(m[0][:, 0], m[1][:, 1]) for m in xm])
    ry_own = np.concatenate([minus_group_mean(m[0][:, 1], m[1][:, 0]) for m in xm])
    sx = np.concatenate([m[2] for m in xm])
    px = np.concatenate([m[3] for m in xm])
    s_share = float(np.mean([group_share(m[2], m[1][:, 1]) for m in xm]))
    p_share = float(np.mean([group_share(m[3], m[1][:, 1]) for m in xm]))
    write(RESULTS / "stage_shifts.csv", [
        {"quantity": "photo vs scan centre (x), all", "rms_um": rms(rx)},
        {"quantity": "photo vs scan centre (x), per-column part removed", "rms_um": rms(rx_own)},
        {"quantity": "photo vs scan centre (y), all", "rms_um": rms(ry)},
        {"quantity": "photo vs scan centre (y), per-row part removed", "rms_um": rms(ry_own)},
        {"quantity": "scan rigid-grid residual (x)", "rms_um": rms(sx), "share_by_column": s_share},
        {"quantity": "photo rigid-grid residual (x)", "rms_um": rms(px), "share_by_column": p_share},
    ])
    d, p = np.array(pinned["detection"]), np.array(pinned["photo_prediction"])
    write(RESULTS / "pinned_pads.csv", [
        {"scan_x_estimate": "plate detection", "pads": len(d), "median_err_um": float(np.median(d)), "rms_err_um": rms(d)},
        {"scan_x_estimate": "photo + plate geometry", "pads": len(p), "median_err_um": float(np.median(p)), "rms_err_um": rms(p)},
    ])

    # where the gap from 1 goes, from the per-cell results of run_study.py
    drows = []
    for name in ("cells_baseline.csv", "cells_final.csv"):
        with (RESULTS / name).open() as f:
            ok = [r for r in csv.DictReader(f) if r["method"] == "new" and r["status"] == "ok"]
        iou = np.array([float(r["iou"]) for r in ok])
        ceil = np.array([float(r["ceiling"]) for r in ok])
        pw, ph = (np.array([float(r[k]) for r in ok]) for k in ("photo_w_um", "photo_h_um"))
        sw, sh = (np.array([float(r[k]) for r in ok]) for k in ("scan_w_um", "scan_h_um"))
        iw, ih = np.minimum(pw, sw), np.minimum(ph, sh)
        rect = iw * ih / (pw * ph + sw * sh - iw * ih)
        plates = np.array([r["plate"] for r in ok])
        within = np.median([np.std(pw[plates == q]) for q in np.unique(plates)])
        drows.append({"recipe": name.replace("cells_", "").replace(".csv", ""), "mean_iou": float(iou.mean()),
                      "loss_alignment_in_crop": float(ceil.mean() - iou.mean()),
                      "loss_size": float(1 - rect.mean()), "loss_outline_shape": float(rect.mean() - ceil.mean()),
                      "photo_vs_scan_width_r": float(np.corrcoef(pw, sw)[0, 1]),
                      "photo_vs_scan_height_r": float(np.corrcoef(ph, sh)[0, 1]),
                      "photo_width_sd_within_plate_um": float(within)})
    write(RESULTS / "loss_decomposition.csv", drows)

    w = np.array([r["pad_width_across_traces_um"] for r in trace_rows])
    sg = np.array([r["edge_softness_sigma_um"] for r in trace_rows])
    print(f"\nplate trace model: width {np.median(w):.1f} µm (IQR {np.percentile(w, 25):.1f}-{np.percentile(w, 75):.1f}), "
          f"edge softness {np.median(sg):.2f} µm")
    for r in erows:
        print(f"  {r['instrument']:5} {r['estimator']:52} x {r['grid_residual_x_um']:.2f}  y {r['grid_residual_y_um']:.2f}")
    print(f"photo vs scan x {rms(rx):.2f} -> {rms(rx_own):.2f} µm without per-column part; y {rms(ry):.2f} -> {rms(ry_own):.2f}")
    print(f"scan grid x residual {rms(sx):.2f} µm ({100 * s_share:.0f}% by column); photo {rms(px):.2f} µm ({100 * p_share:.0f}%)")
    print(f"pinned pads ({len(d)}): detection median {np.median(d):.2f} µm, photo-predicted {np.median(p):.2f} µm")
    for r in drows:
        print(f"  {r['recipe']:9} mean {r['mean_iou']:.3f}: alignment {r['loss_alignment_in_crop']:.3f}, "
              f"size {r['loss_size']:.3f}, shape {r['loss_outline_shape']:.3f}; width r {r['photo_vs_scan_width_r']:.2f}, "
              f"height r {r['photo_vs_scan_height_r']:.2f}, photo width SD within plate {r['photo_width_sd_within_plate_um']:.1f} µm")
    for r in erows[-4:]:
        print(f"  {r['estimator']:44} x {r['grid_residual_x_um']:.2f}  y {r['grid_residual_y_um']:.2f}")


if __name__ == "__main__":
    main()
