"""One pass over every plate, emitting the features and labels the audit needs.

Three things are varied so the audit can attribute blame:

**Feature window.** Brightness statistics are computed over a centred window of
W micrometres, for W from the full cell pitch down to roughly the pad size.
This is the direct test of the suspected flaw: the pad is only ~8% of a
full-pitch cell (61 x 98 um inside 267 x 267 um), so statistics over the whole
cell are ~92% bare silicon. If absorption is the real cue, it was diluted
twelvefold.

**Pad mask.** A data-driven alternative to the fixed windows: threshold the
crop's own local-contrast response and keep the component nearest the centre.
Derived from the photo only, never from the height map, so it stays legal at
deployment.

**Label definition.** ``h_max`` is a single pixel of a ~1 um quantity, so it
carries real noise. Percentile and mask-mean alternatives are emitted alongside
it.

Feature statistics are read straight off the raw pixels -- no resampling -- so
nothing here is smoothed before being measured.

Usage::

    uv run python studies/02-thickness-signal-audit/src/build_features.py
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
REPO = AUDIT.parent.parent
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

import cv2  # noqa: E402

from Auto_OPDx.alignment.dataset import (  # noqa: E402
    fit_cross_modal,
    solve_photo,
    solve_plate,
)

ROWS = COLS = 8
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# Centred feature windows, micrometres. 267 is the full cell pitch (what the
# previous evaluation used); ~80 is about the pad's own footprint.
WINDOWS_UM = (267.0, 200.0, 150.0, 110.0, 80.0, 60.0)

# Window the scalar labels are read from, micrometres. Kept fixed so the label
# is not confounded with the feature window being swept.
LABEL_UM = 110.0


def stats_from(patch: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """The deck's seven Phase-1 intensity statistics over a region."""
    a = np.asarray(patch, float)
    if mask is not None:
        if mask.sum() < 6:
            return {}
        rgb = a[mask]                       # (n, 3)
        gray = rgb.mean(axis=1)
    else:
        rgb = a.reshape(-1, 3)
        gray = rgb.mean(axis=1)
    if gray.size < 6:
        return {}
    return {
        "mean_r": float(rgb[:, 0].mean()),
        "mean_g": float(rgb[:, 1].mean()),
        "mean_b": float(rgb[:, 2].mean()),
        "gray_mean": float(gray.mean()),
        "gray_max": float(gray.max()),
        "gray_p95": float(np.percentile(gray, 95)),
        "gray_std": float(gray.std()),
    }


def sharpness_from(patch: np.ndarray) -> dict[str, float]:
    """Focus / edge-sharpness measures over a patch.

    Motivation: brightness carries no thickness signal on corrected labels, yet
    DINOv2 reaches Spearman ~0.42. So whatever the photo encodes is structural,
    not intensity. One physically plausible candidate is **focus**: the lens has
    a very shallow depth of field, so a taller pad sits measurably closer to the
    focal plane and should render slightly sharper.

    If that is the cue, it is depth-from-focus happening by accident in a single
    frame -- which would make a deliberate z-stack (on the motorised stage
    already being built) the obvious way to amplify it.
    """
    gray = np.asarray(patch, float).mean(axis=2)
    if min(gray.shape) < 8:
        return {}
    g = gray / max(gray.mean(), 1e-6)          # normalise out exposure
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=3)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # Ratio of high- to low-frequency energy, via a difference of Gaussians.
    lo = cv2.GaussianBlur(g, (0, 0), sigmaX=3.0)
    hi = g - lo
    return {
        "lap_var": float(lap.var()),
        "lap_absmean": float(np.abs(lap).mean()),
        "grad_mean": float(mag.mean()),
        "grad_p95": float(np.percentile(mag, 95)),
        "grad_max": float(mag.max()),
        "hf_energy": float((hi ** 2).mean()),
        "hf_ratio": float(
            (hi ** 2).mean() / max(float(((lo - lo.mean()) ** 2).mean()), 1e-9)
        ),
    }


def pad_mask_from_photo(patch: np.ndarray) -> np.ndarray:
    """Locate the pad inside an RGB patch, using the photo alone.

    Local contrast against a blurred copy of the same patch, thresholded, then
    the connected component closest to the patch centre. Uses no height data,
    so it is available at deployment.
    """
    gray = np.asarray(patch, float).mean(axis=2)
    h, w = gray.shape
    if min(h, w) < 12:
        return np.zeros(gray.shape, bool)
    sigma = max(3.0, min(h, w) / 4.0)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dark = np.clip(bg - gray, 0, None)          # pads are darker than local bg
    if dark.max() <= 1e-6:
        return np.zeros(gray.shape, bool)

    thr = 0.45 * float(dark.max())
    binary = (dark >= thr).astype(np.uint8)
    n, lab = cv2.connectedComponents(binary, connectivity=4)
    if n <= 1:
        return binary.astype(bool)

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    best, best_d = 0, np.inf
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < 8:
            continue
        d = (ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2
        if d < best_d:
            best, best_d = k, d
    return (lab == best) if best else binary.astype(bool)


def height_labels(hm, cx: float, cy: float, size_um: float) -> dict[str, float]:
    """Label variants from the native, un-interpolated detrended grid."""
    half = size_um / 2.0
    xi = np.where(np.abs(hm.plate.x - cx) <= half)[0]
    yi = np.where(np.abs(hm.plate.y - cy) <= half)[0]
    if xi.size == 0 or yi.size == 0:
        return {}
    patch = hm.resid[np.ix_(yi, xi)]
    patch = patch[np.isfinite(patch)]
    if patch.size < 6:
        return {}

    # Mask the pad within the height patch by its own contrast, then average.
    # Unlike h_max this pools many samples, so it is far less noisy.
    thr = 0.5 * float(patch.max())
    pad = patch >= thr
    out = {
        "h_max": float(patch.max()),
        "h_p95": float(np.percentile(patch, 95)),
        "h_p99": float(np.percentile(patch, 99)),
        "h_maskmean": float(patch[pad].mean()) if pad.sum() >= 3 else float(patch.max()),
        "h_maskn": int(pad.sum()),
    }
    # The previous label definition, for reference: raw z, tilt included.
    raw = hm.plate.z[np.ix_(yi, xi)]
    out["h_max_raw"] = float(raw[np.isfinite(raw)].max())
    return out


def main() -> None:
    rgb_by_stem = {
        p.stem: p
        for p in (REPO / "data" / "rgb" / "img").iterdir()
        if p.suffix.lower() in IMG_EXT
    }
    plates = sorted((REPO / "data" / "opdx").glob("*.OPDx"))

    recs: list[dict] = []
    pad_fracs: list[float] = []
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
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{stem}: {type(e).__name__}: {e}")
            continue

        M, t = cm.affine[:2], cm.affine[2]
        try:
            M_inv = np.linalg.inv(M.T)
        except np.linalg.LinAlgError:
            skipped.append(f"{stem}: singular affine")
            continue
        sx, sy = cm.um_per_px
        img = ph.img
        H, W = img.shape[:2]

        for r in range(ROWS):
            for c in range(COLS):
                hx, hy = hm.fit.predict((r, c))[0]
                lab = height_labels(hm, hx, hy, LABEL_UM)
                if not lab:
                    continue
                px = M_inv @ (np.array([hx, hy]) - t)

                rec = {"plate": stem, "row": r, "col": c, **lab}

                # --- fixed centred windows ------------------------------
                overhang = 0.0
                for wum in WINDOWS_UM:
                    hw = (wum / 2.0) / sx
                    hh = (wum / 2.0) / sy
                    x0, x1 = int(round(px[0] - hw)), int(round(px[0] + hw))
                    y0, y1 = int(round(px[1] - hh)), int(round(px[1] + hh))
                    want = max(1, (x1 - x0) * (y1 - y0))
                    xa, xb = max(0, x0), min(W, x1)
                    ya, yb = max(0, y0), min(H, y1)
                    if xb - xa < 4 or yb - ya < 4:
                        continue
                    got = (xb - xa) * (yb - ya)
                    if wum == WINDOWS_UM[0]:
                        overhang = 1.0 - got / want
                    s = stats_from(img[ya:yb, xa:xb])
                    tag = f"w{int(wum)}"
                    for k, v in s.items():
                        rec[f"{tag}_{k}"] = v

                # --- data-driven pad mask, on the largest window --------
                hw = (WINDOWS_UM[0] / 2.0) / sx
                hh = (WINDOWS_UM[0] / 2.0) / sy
                xa = max(0, int(round(px[0] - hw)))
                xb = min(W, int(round(px[0] + hw)))
                ya = max(0, int(round(px[1] - hh)))
                yb = min(H, int(round(px[1] + hh)))
                if xb - xa >= 8 and yb - ya >= 8:
                    patch = img[ya:yb, xa:xb]
                    mask = pad_mask_from_photo(patch)
                    frac = float(mask.mean())
                    pad_fracs.append(frac)
                    rec["pad_frac"] = frac
                    s = stats_from(patch, mask)
                    for k, v in s.items():
                        rec[f"pad_{k}"] = v
                    # Ring around the pad: substrate only. If this predicts as
                    # well as the pad itself, the "signal" is not absorption.
                    ring = (~mask) & (cv2.dilate(
                        mask.astype(np.uint8), np.ones((9, 9), np.uint8)
                    ).astype(bool))
                    s = stats_from(patch, ring)
                    for k, v in s.items():
                        rec[f"ring_{k}"] = v

                # --- sharpness / focus, over the pad and its immediate edge --
                # 110 um keeps the pad plus its boundary, which is where any
                # focus difference shows up most strongly.
                hw = (110.0 / 2.0) / sx
                hh = (110.0 / 2.0) / sy
                xa = max(0, int(round(px[0] - hw)))
                xb = min(W, int(round(px[0] + hw)))
                ya = max(0, int(round(px[1] - hh)))
                yb = min(H, int(round(px[1] + hh)))
                if xb - xa >= 8 and yb - ya >= 8:
                    for k, v in sharpness_from(img[ya:yb, xa:xb]).items():
                        rec[f"sharp_{k}"] = v

                rec["overhang"] = overhang
                recs.append(rec)

        if i % 10 == 0 or i == len(plates):
            print(f"  {i}/{len(plates)} plates, {len(recs)} cells")

    if not recs:
        sys.exit("no cells produced")

    keys: list[str] = []
    for rec in recs:
        for k in rec:
            if k not in keys:
                keys.append(k)

    out = AUDIT / "results" / "cells_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(",".join(keys) + "\n")
        for rec in recs:
            f.write(",".join(str(rec.get(k, "")) for k in keys) + "\n")

    print(f"\ncells {len(recs)} across {len({r['plate'] for r in recs})} plates")
    if pad_fracs:
        pf = np.array(pad_fracs)
        print(f"pad occupies {pf.mean()*100:.1f}% of a full-pitch cell "
              f"(median {np.median(pf)*100:.1f}%) "
              f"-> statistics over the whole cell are ~{(1-pf.mean())*100:.0f}% substrate")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:4]}")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
