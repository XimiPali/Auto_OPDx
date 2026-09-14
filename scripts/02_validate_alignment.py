"""Compare the previous feature localisation against the joint-lattice fit.

The comparison metric is residual RMSE expressed as a **fraction of the cell
pitch**. Raw RMSE is not comparable between the two, because the stored
centroids live in a resampled pixel space of unknown scale while ours are in
micrometres (height map) and native pixels (photo). Normalising by the pitch
each fit recovers makes the numbers scale-free.

What "residual" means here: fit the 10-parameter 8x8 lattice to a set of
centroids and measure how far each centroid sits from the model. A set of
centroids that genuinely describes a rigid DMD-written grid will sit close to
one; scattered or mislocated centroids will not.

Usage::

    uv run python scripts/02_validate_alignment.py
    uv run python scripts/02_validate_alignment.py --limit 8
    uv run python scripts/02_validate_alignment.py --csv out.csv
"""

from __future__ import annotations

import argparse
import json
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
    fit_cross_modal,
    solve_photo,
    solve_plate,
)
from Auto_OPDx.alignment.lattice import solve_lattice  # noqa: E402

ROWS = COLS = 8


def _old_centroids(entry: dict) -> np.ndarray:
    """Extract centroids from a metadata entry, in (row, col) key order."""
    pts = []
    for r in range(ROWS):
        for c in range(COLS):
            v = entry.get(f"{r},{c}") or entry.get(f"({r}, {c})")
            if v and v.get("found") and v.get("centroid"):
                pts.append(v["centroid"])
    return np.asarray(pts, float)


def _normalised_rmse(centres: np.ndarray) -> tuple[float, float, int]:
    """Fit a lattice and return (rmse/pitch, rmse, n_inliers)."""
    if len(centres) < 8:
        return float("nan"), float("nan"), 0
    fit = solve_lattice(centres, ROWS, COLS)
    pitch = float(np.mean(fit.pitch))
    if not np.isfinite(pitch) or pitch <= 0:
        return float("nan"), fit.rmse, int(fit.inliers.sum())
    return fit.rmse / pitch, fit.rmse, int(fit.inliers.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--csv", type=Path, default=REPO / "studies/01-label-alignment" / "results" / "alignment.csv")
    args = ap.parse_args()

    meta_dir = REPO / "data" / "metadata"
    prof_old = json.loads((meta_dir / "prof_crop_details_v2.json").read_text())
    img_old = json.loads((meta_dir / "img_crop_details.json").read_text())
    prof_old = {k.rsplit(".", 1)[0]: v for k, v in prof_old.items()}
    img_old = {k.rsplit(".", 1)[0]: v for k, v in img_old.items()}

    rgb_by_stem = {
        p.stem: p
        for p in (REPO / "data" / "rgb" / "img").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    }

    plates = sorted((REPO / "data" / "opdx").glob("*.OPDx"))
    if args.limit:
        plates = plates[: args.limit]

    rows: list[dict] = []
    for path in plates:
        stem = path.stem
        rec: dict = {"plate": stem}
        try:
            hm = solve_plate(path, ROWS, COLS)
            rec.update(
                hm_n=len(hm.centres),
                hm_in=int(hm.fit.inliers.sum()),
                hm_rmse_um=hm.fit.rmse,
                hm_pitch_um=float(np.mean(hm.fit.pitch)),
                hm_norm=hm.fit.rmse / float(np.mean(hm.fit.pitch)),
                hm_rot=hm.fit.rotation_deg,
                tilt_pp_um=hm.tilt["tilt_pp_um"],
                tilt_frac=hm.tilt["tilt_fraction"],
            )
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"hm: {type(e).__name__}: {e}"
            rows.append(rec)
            continue

        img_path = rgb_by_stem.get(stem)
        if img_path is not None:
            try:
                ph = solve_photo(img_path, ROWS, COLS)
                rec.update(
                    rgb_n=len(ph.centres),
                    rgb_in=int(ph.fit.inliers.sum()),
                    rgb_rmse_px=ph.fit.rmse,
                    rgb_pitch_px=float(np.mean(ph.fit.pitch)),
                    rgb_norm=ph.fit.rmse / float(np.mean(ph.fit.pitch)),
                    rgb_rot=ph.fit.rotation_deg,
                )
                cm = fit_cross_modal(hm, ph, COLS)
                rec.update(
                    cross_rmse_um=cm.rmse_um,
                    cross_max_um=cm.max_um,
                    cross_slots=len(cm.slots),
                    um_per_px_x=cm.um_per_px[0],
                    um_per_px_y=cm.um_per_px[1],
                    px_aspect=cm.um_per_px[1] / cm.um_per_px[0],
                    rel_rot=cm.rotation_deg,
                )
            except Exception as e:  # noqa: BLE001
                rec["rgb_error"] = f"{type(e).__name__}: {e}"

        if stem in prof_old:
            n, r, i = _normalised_rmse(_old_centroids(prof_old[stem]))
            rec.update(old_hm_norm=n, old_hm_rmse=r, old_hm_in=i)
        if stem in img_old:
            n, r, i = _normalised_rmse(_old_centroids(img_old[stem]))
            rec.update(old_rgb_norm=n, old_rgb_rmse=r, old_rgb_in=i)

        rows.append(rec)

    ok = [r for r in rows if "error" not in r]
    print(f"plates processed: {len(ok)}/{len(rows)}")
    for r in rows:
        if "error" in r:
            print(f"  FAILED {r['plate']}: {r['error']}")

    def col(key: str) -> np.ndarray:
        v = np.array([r.get(key, np.nan) for r in ok], float)
        return v[np.isfinite(v)]

    def line(label: str, v: np.ndarray, pct: bool = False) -> None:
        if not len(v):
            print(f"  {label:34} (no data)")
            return
        f = 100.0 if pct else 1.0
        u = "%" if pct else ""
        print(f"  {label:34} median {np.median(v)*f:7.2f}{u}  "
              f"mean {v.mean()*f:7.2f}{u}  max {v.max()*f:7.2f}{u}  n={len(v)}")

    print("\n" + "=" * 78)
    print("LATTICE RESIDUAL AS A FRACTION OF CELL PITCH  (lower is better)")
    print("=" * 78)
    print("  height map")
    line("    previous centroids", col("old_hm_norm"), pct=True)
    line("    joint lattice fit (this work)", col("hm_norm"), pct=True)
    print("  photo")
    line("    previous centroids", col("old_rgb_norm"), pct=True)
    line("    joint lattice fit (this work)", col("rgb_norm"), pct=True)

    for tag, old, new in (("height map", "old_hm_norm", "hm_norm"),
                          ("photo", "old_rgb_norm", "rgb_norm")):
        o, n = col(old), col(new)
        if len(o) and len(n):
            print(f"\n  >> {tag}: {np.median(o)/np.median(n):.1f}x tighter at the median")

    print("\n" + "=" * 78)
    print("ABSOLUTE NUMBERS")
    print("=" * 78)
    line("height-map rmse (um)", col("hm_rmse_um"))
    line("photo rmse (px)", col("rgb_rmse_px"))
    line("CROSS-MODAL rmse (um)", col("cross_rmse_um"))
    line("cross-modal worst cell (um)", col("cross_max_um"))
    line("cell pitch (um)", col("hm_pitch_um"))
    line("features found, height map", col("hm_n"))
    line("features found, photo", col("rgb_n"))

    print("\n" + "=" * 78)
    print("GEOMETRY")
    print("=" * 78)
    line("substrate tilt (um p-p)", col("tilt_pp_um"))
    line("tilt as fraction of raw z range", col("tilt_frac"), pct=True)
    line("photo um/px, x axis", col("um_per_px_x"))
    line("photo um/px, y axis", col("um_per_px_y"))
    line("photo pixel aspect (y/x)", col("px_aspect"))
    line("height-map grid rotation (deg)", col("hm_rot"))
    line("relative rotation, photo->hm (deg)", col("rel_rot"))

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"\nwrote {args.csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
