"""Measure the photo-vs-scan axis disagreement on every plate.

The registration recovers a per-axis photo scale of about 1.07 x 1.46 um/px,
while the camera's optics predict square 1.25 um pixels. Equivalently, the grid
is ~1.3x wider than tall in the photo but nearly square in the scan. One of the
two instruments has an axis scale wrong. This script measures the disagreement
on all 69 plates and writes what each explanation implies, so the lab's answer
can be checked against numbers rather than impressions.

    H1  the camera is right (square 1.25 um pixels); the profilometer's axis
        steps are off.
    H2  the profilometer is right; the photo was resampled on export, so its
        pixels are not square.

Writes results/calibration.csv (one row per plate) and prints the summary that
the README quotes.

Usage::

    MPLBACKEND=Agg uv run python studies/04-instrument-checks/src/check_calibration.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
sys.path.insert(0, str(REPO / "src"))

from Auto_OPDx.alignment.dataset import (  # noqa: E402
    _slot_observations,
    fit_cross_modal,
    solve_photo,
    solve_plate,
)

OPTICS_UM_PER_PX = 1.25      # 2.5 um sensor pixels behind a 4x objective + 0.5x demag
PHOTO_EXT = (".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff")


def _pad_size(field, xs, ys, cx, cy, half):
    """Width/height of the pad nearest (cx, cy) from second moments, in axis units.

    For a uniform rectangle sigma = side / sqrt(12), so side = sqrt(12) * sigma.
    Moments are used rather than a bounding box because the scan has only 2-3
    samples across a pad on its coarse axis, where a bounding box quantises to
    whole 30 um steps.
    """
    xi = np.where((xs >= cx - half) & (xs <= cx + half))[0]
    yi = np.where((ys >= cy - half) & (ys <= cy + half))[0]
    if xi.size < 2 or yi.size < 2:
        return np.nan, np.nan
    patch = np.nan_to_num(field[np.ix_(yi, xi)], nan=0.0)
    bg, peak = float(np.median(patch)), float(np.percentile(patch, 99.5))
    if peak <= bg:
        return np.nan, np.nan
    binary = patch > bg + 0.5 * (peak - bg)
    n, lab, _, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.nan, np.nan
    centre = np.array([xi.size / 2, yi.size / 2])
    k = 1 + int(np.argmin(np.linalg.norm(cents[1:] - centre, axis=1)))
    m = binary_fill_holes(lab == k)
    yy, xx = np.nonzero(m)
    if len(xx) < 3:
        return np.nan, np.nan
    gx, gy = xs[xi][xx], ys[yi][yy]
    return float(np.sqrt(12) * gx.std()), float(np.sqrt(12) * gy.std())


def measure(stem: str, photo_path: Path) -> dict:
    hm = solve_plate(REPO / "data" / "opdx" / f"{stem}.OPDx")
    ph = solve_photo(photo_path)
    cross = fit_cross_modal(hm, ph)

    p_col, p_row = ph.fit.pitch          # px
    s_col, s_row = hm.fit.pitch          # um
    sx, sy = cross.um_per_px

    # pad sizes: photo in native pixels, scan in micrometres
    h_px, w_px = ph.response.shape
    pxs, pys = np.arange(w_px, dtype=float), np.arange(h_px, dtype=float)
    photo_sizes = [_pad_size(ph.response, pxs, pys, *xy, 0.45 * min(p_col, p_row))
                   for xy in _slot_observations(ph.fit, 8).values()]
    scan_sizes = [_pad_size(hm.resid, hm.plate.x, hm.plate.y, *xy, 0.45 * min(s_col, s_row))
                  for xy in _slot_observations(hm.fit, 8).values()]
    pw, ph_ = np.nanmedian(np.array(photo_sizes), axis=0)
    sw, sh = np.nanmedian(np.array(scan_sizes), axis=0)

    return {
        "plate": stem,
        "photo_pitch_col_px": p_col, "photo_pitch_row_px": p_row,
        "scan_pitch_col_um": s_col, "scan_pitch_row_um": s_row,
        "photo_grid_aspect": p_col / p_row, "scan_grid_aspect": s_col / s_row,
        "aspect_disagreement": (p_col / p_row) / (s_col / s_row),
        "fitted_um_per_px_x": sx, "fitted_um_per_px_y": sy,
        "fitted_um_per_px_geomean": float(np.sqrt(sx * sy)),
        "photo_pad_w_px": pw, "photo_pad_h_px": ph_,
        "scan_pad_w_um": sw, "scan_pad_h_um": sh,
        # H1: camera right -> physical pitch from optics; the scan's axis factor is the ratio
        "h1_pitch_col_um": p_col * OPTICS_UM_PER_PX, "h1_pitch_row_um": p_row * OPTICS_UM_PER_PX,
        "h1_scan_x_factor": s_col / (p_col * OPTICS_UM_PER_PX),
        "h1_scan_y_factor": s_row / (p_row * OPTICS_UM_PER_PX),
        # H2: scan right -> the photo's pixels are sx x sy; physical pad shape follows
        "h2_photo_pad_w_um": pw * sx, "h2_photo_pad_h_um": ph_ * sy,
    }


def main() -> None:
    photos = {p.stem: p for p in (REPO / "data" / "rgb" / "img").iterdir() if p.suffix.lower() in PHOTO_EXT}
    stems = sorted(p.stem for p in (REPO / "data" / "opdx").glob("*.OPDx") if p.stem in photos)
    rows: list[dict] = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        try:
            rows.append(measure(stem, photos[stem]))
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(stems)}] {stem} FAILED {type(e).__name__}: {e}")
            continue
        r = rows[-1]
        print(f"  [{i}/{len(stems)}] {stem:18} grid aspect photo {r['photo_grid_aspect']:.3f} "
              f"scan {r['scan_grid_aspect']:.3f}  um/px {r['fitted_um_per_px_x']:.3f} x {r['fitted_um_per_px_y']:.3f}")

    out = STUDY / "results" / "calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def med(k):
        return float(np.nanmedian([r[k] for r in rows]))

    def spread(k):
        v = np.array([r[k] for r in rows], float)
        return float(np.nanpercentile(v, 10)), float(np.nanpercentile(v, 90))

    print("\n" + "=" * 72)
    print(f"{len(rows)} plates, {time.time() - t0:.0f}s   (median, with 10th-90th percentile)")
    print("=" * 72)
    for k, label in [
        ("photo_grid_aspect", "grid aspect (col/row pitch), photo"),
        ("scan_grid_aspect", "grid aspect (col/row pitch), scan"),
        ("aspect_disagreement", "disagreement photo / scan"),
        ("fitted_um_per_px_x", "registration: photo um/px along x"),
        ("fitted_um_per_px_y", "registration: photo um/px along y"),
        ("fitted_um_per_px_geomean", "geometric mean (optics predict 1.25)"),
        ("photo_pad_w_px", "pad width in photo, px"),
        ("photo_pad_h_px", "pad height in photo, px"),
        ("scan_pad_w_um", "pad width in scan, um"),
        ("scan_pad_h_um", "pad height in scan, um"),
        ("h1_pitch_col_um", "H1 physical pitch, columns, um"),
        ("h1_pitch_row_um", "H1 physical pitch, rows, um"),
        ("h1_scan_x_factor", "H1 scan x-axis reads too long by"),
        ("h1_scan_y_factor", "H1 scan y-axis reads too long by"),
        ("h2_photo_pad_w_um", "H2 physical pad width, um"),
        ("h2_photo_pad_h_um", "H2 physical pad height, um"),
    ]:
        lo, hi = spread(k)
        print(f"  {label:40} {med(k):8.3f}   ({lo:.3f} – {hi:.3f})")
    print(f"\nscan pitch: columns {med('scan_pitch_col_um'):.1f} um, rows {med('scan_pitch_row_um'):.1f} um")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
