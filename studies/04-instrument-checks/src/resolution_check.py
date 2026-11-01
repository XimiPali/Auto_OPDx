"""Does photo detection hold up on a larger image of the same plate?

The lab's planned camera (1-inch sensor, 10x objective, 0.5x demag) images the
same plate at roughly 2.7x today's pixel count per side. No such images exist
yet, so this enlarges two of today's photos by 2.67x and runs the unchanged
detection on them. It checks scale independence and speed, not focus stacking.

Writes results/resolution_check.csv. Enlarged images go to $TMPDIR.

Usage::

    MPLBACKEND=Agg uv run python studies/04-instrument-checks/src/resolution_check.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
sys.path.insert(0, str(REPO / "src"))

from Auto_OPDx.alignment.dataset import solve_photo  # noqa: E402

PLATES = ("10-01-2025_P1", "9-24-2025_P6")
SCALE = 2.67          # 5468 x 4101: a 1-inch sensor covering today's field


def main() -> None:
    tmp = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "auto_opdx_resolution_check"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in PLATES:
        src = next(p for p in (REPO / "data" / "rgb" / "img").glob(f"{stem}.*"))
        img = cv2.imread(str(src))
        for scale in (1.0, SCALE):
            path = src
            if scale != 1.0:
                path = tmp / f"{stem}_x{scale}.png"
                if not path.exists():
                    cv2.imwrite(str(path), cv2.resize(img, None, fx=scale, fy=scale,
                                                      interpolation=cv2.INTER_CUBIC))
            t = time.time()
            ph = solve_photo(path)
            dt = time.time() - t
            pitch = float(np.mean(ph.fit.pitch))
            rows.append({
                "plate": stem, "scale": scale, "width_px": ph.img.shape[1], "height_px": ph.img.shape[0],
                "inliers": int(ph.fit.inliers.sum()), "rmse_over_pitch": ph.fit.rmse / pitch,
                "coverage": ph.fit.coverage(ph.centres), "pitch_col_px": ph.fit.pitch[0],
                "seconds": dt,
            })
            r = rows[-1]
            print(f"  {stem} x{scale}: {r['width_px']}x{r['height_px']}  {r['inliers']}/64 inliers  "
                  f"residual {r['rmse_over_pitch'] * 100:.2f}% of pitch  coverage {r['coverage']:.3f}  "
                  f"{r['seconds']:.1f}s", flush=True)
    out = STUDY / "results" / "resolution_check.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
