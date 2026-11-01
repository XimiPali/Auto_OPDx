"""How much IoU does the scan's coarse trace spacing cost, and what would a finer scan buy?

CORRECTED 2026-10-31. This simulation draws the height-map outline by linear
interpolation across the traces and treats the pad edge as a sharp step, so it
over-states the cost: it predicted that trace spacing alone caps IoU at 0.71.
Study 05 shows the cost is concentrated in pads that sit on a trace, and that
rebuilding each pad from its traces removes most of it in software. Kept for
the record; the README's section 2 has the corrected account.

The stylus records a trace every ~30 um across the pads (and every ~2.9 um
along each trace). A ~60 um pad is therefore only two or three traces wide, so
its outline in the height map is blurred on that axis no matter how well the
crops are aligned.

This isolates that one effect. Take each pad's outline from the photo -- sharp,
1.25 um per pixel -- sample it the way the profilometer would (one value every
``s`` um across, at a random offset), rebuild the outline from those samples
with the same linear interpolation the crops use, and measure the IoU of the
rebuilt outline against the original. Nothing else differs, so the result is
the ceiling that trace spacing alone imposes.

Writes results/sampling_ceiling.csv and figures/sampling_ceiling.png.

Usage::

    MPLBACKEND=Agg uv run python studies/04-instrument-checks/src/sampling_ceiling.py
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
sys.path.insert(0, str(REPO / "studies" / "03-alignment-iou" / "src"))

import iou_lib as L  # noqa: E402

SPACINGS_UM = (2.9, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)
PHASES = 8
N_PLATES = 15          # every third of the 45 comparable plates; ample for a mean

NEW, INK, INK_2, INK_3, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0", "#fcfcfb"


def resample_mask(mask: np.ndarray, window_um: float, spacing_um: float, phase: float) -> np.ndarray:
    """Sample ``mask`` every ``spacing_um`` along x, rebuild by linear interpolation, threshold at 0.5."""
    n = mask.shape[1]
    u = np.linspace(-window_um / 2, window_um / 2, n)
    xs = np.arange(-window_um / 2 + phase * spacing_um, window_um / 2, spacing_um)
    if xs.size < 2:
        return np.zeros_like(mask)
    m = mask.astype(float)
    samples = np.array([np.interp(xs, u, row) for row in m])          # profilometer traces
    rebuilt = np.array([np.interp(u, xs, row) for row in samples])     # what the crop shows
    return rebuilt >= 0.5


def main() -> None:
    prof, img = L.load_old_metadata()
    stems = L.comparable_plates()[::3][:N_PLATES]
    masks: list[tuple[np.ndarray, float]] = []
    for stem in stems:
        ev = L.evaluate_plate(stem, prof[stem], img[stem], keep_arrays=True)
        masks += [(c.photo_mask, ev.window_um) for c in ev.cells
                  if c.method == "new" and c.status == "ok" and c.photo_mask is not None]
        print(f"  {stem:18} {len(masks):5d} pad outlines so far", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for s in SPACINGS_UM:
        vals = []
        for m, win in masks:
            for ph in rng.random(PHASES):
                vals.append(L.iou(m, resample_mask(m, win, s, ph)))
        v = np.array(vals)
        rows.append({"trace_spacing_um": s, "mean_iou": float(v.mean()), "p10_iou": float(np.percentile(v, 10)),
                     "n": len(v)})
        print(f"  trace spacing {s:5.1f} um  ->  IoU ceiling from sampling alone {v.mean():.3f}")

    out = STUDY / "results" / "sampling_ceiling.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # figure
    mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 11, "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_3, "ytick.color": INK_3, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = [r["trace_spacing_um"] for r in rows]
    y = [r["mean_iou"] for r in rows]
    ax.plot(x, y, color=NEW, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE)
    for xv, yv in zip(x, y):
        if xv in (10.0, 30.0):
            ax.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 10), ha="center",
                        fontsize=11, color=NEW, fontweight="bold")
    ax.axvspan(28, 32, color=GRID, alpha=0.8, zorder=0)
    ax.text(30, 0.02 + min(y) - 0.04, "today\n(≈30 µm)", ha="center", va="top", fontsize=9.5, color=INK_2)
    ax.set_xlabel("profilometer trace spacing, µm")
    ax.set_ylabel("IoU ceiling from sampling alone")
    ax.set_ylim(min(y) - 0.08, 1.0)
    ax.set_xlim(0, 37)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_title("Simulated: interpolated outline of a sharp-edged pad (an over-estimate; see study 05)",
                 loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    (STUDY / "figures").mkdir(exist_ok=True)
    fig.savefig(STUDY / "figures" / "sampling_ceiling.png", dpi=180)
    plt.close(fig)
    print(f"\n{len(masks)} pad outlines from {len(stems)} plates, {PHASES} random trace offsets each")
    print(f"wrote {out.relative_to(REPO)} and figures/sampling_ceiling.png")


if __name__ == "__main__":
    main()
