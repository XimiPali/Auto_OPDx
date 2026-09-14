"""Render the audit figures from the result CSVs.

Four panels, one per question the audit set out to answer:

  1. did concentrating the features on the pad recover any signal?  (no)
  2. can the features tell where a cell sits on the plate?          (yes, easily)
  3. what survives once that positional information is removed?     (a little)
  4. what does a cell actually contain?                             (96% substrate)

Palette: categorical slots 1-3 of the reference palette, validated all-pairs in
light mode. Slot 3 (aqua) carries a contrast WARN against the surface, so every
mark that uses it is directly labelled.

Usage::

    uv run python studies/02-thickness-signal-audit/src/make_figures.py
"""

from __future__ import annotations

import csv
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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

BRIGHT = "#2a78d6"    # slot 1
DINO = "#eb6834"      # slot 2
CTRL = "#1baf7a"      # slot 3 - contrast WARN, always direct-labelled
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e0"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "font.family": "sans-serif",
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_3, "ytick.color": INK_3,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
})

NOISE_FLOOR = 0.10   # what brightness features reach; treat below this as nil


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def tidy(ax, ylabel: str | None = None) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)


def main() -> None:
    ev = read(AUDIT / "results" / "evaluation.csv")
    pc = read(AUDIT / "results" / "position_control.csv")

    fig = plt.figure(figsize=(15.0, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.24,
                  left=0.065, right=0.975, top=0.855, bottom=0.075)

    # ---- 1. feature region sweep ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    reg = {r["config"]: float(r["spearman"])
           for r in ev if r["experiment"] == "feature_region"}
    wins = [267, 200, 150, 110, 80, 60]
    ys = [reg.get(f"window {w} um", np.nan) for w in wins]
    ax.plot(wins, ys, color=BRIGHT, lw=2.0, marker="o", ms=6.5,
            mec=SURFACE, mew=1.4, zorder=3, label="centred window")
    for nm, key, ty in (("pad mask only", "pad mask (data-driven)", 0.30),
                        ("substrate ring (control)", "ring around pad (control)", 0.39)):
        v = reg.get(key)
        if v is None:
            continue
        ax.scatter([72], [v], color=CTRL, s=75, zorder=4, marker="D",
                   edgecolors=SURFACE, linewidths=1.2)
        ax.annotate(f"{nm}   {v:+.2f}", xy=(72, v), xytext=(215, ty),
                    fontsize=8.2, color=INK_2, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.8,
                                    shrinkA=2, shrinkB=4,
                                    connectionstyle="arc3,rad=-0.15"))
    ax.axhspan(-NOISE_FLOOR, NOISE_FLOOR, color=INK_3, alpha=0.09, lw=0, zorder=0)
    ax.text(60, NOISE_FLOOR + 0.012, "noise floor", fontsize=7.8,
            color=INK_3, ha="left", va="bottom")
    ax.axhline(0, color=INK_3, lw=0.9, zorder=1)
    # Same y-range as panel 3, so the two are directly comparable: every value
    # here sits inside the noise band, which is the finding.
    ax.set_ylim(-0.18, 0.47)
    ax.invert_xaxis()
    ax.set_xlabel("feature window (µm)  ·  ← wider          tighter →", fontsize=8.5)
    ax.set_title("1 · Concentrating on the pad recovers nothing",
                 fontsize=10.5, pad=17, fontweight="medium")
    ax.text(0.5, 1.018, "brightness statistics · label h_max · K = 8",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, color=INK_2)
    tidy(ax, "Spearman ρ vs true height")
    ax.legend(fontsize=8, frameon=False, loc="lower left", handlelength=1.6,
              labelcolor=INK_2)

    # ---- 2. can the features read position? -------------------------------
    ax = fig.add_subplot(gs[0, 1])
    pos = {r["features"]: (float(r["row_rho"]), float(r["col_rho"]))
           for r in read(AUDIT / "results" / "position_recovery.csv")}
    bkey = next(k for k in pos if k.startswith("brightness"))
    dkey = next(k for k in pos if k.startswith("DINOv2"))
    x = np.arange(2)
    w = 0.30
    gap = 0.012
    b1 = ax.bar(x - w / 2 - gap, list(pos[bkey]), w, color=BRIGHT,
                label="brightness (7 features)", zorder=3)
    b2 = ax.bar(x + w / 2 + gap, list(pos[dkey]), w, color=DINO,
                label="DINOv2 (768 features)", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.022,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8.6,
                    color=INK, fontweight="medium", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["recover the ROW index", "recover the COLUMN index"],
                       fontsize=8.6, color=INK_2)
    ax.set_ylim(0, 1.13)
    ax.set_title("2 · The photo betrays where a cell sits",
                 fontsize=10.5, pad=17, fontweight="medium")
    ax.text(0.5, 1.018, "held-out plates · illumination varies smoothly across the plate",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, color=INK_2)
    tidy(ax, "Spearman ρ vs grid index")
    ax.legend(fontsize=8, frameon=False, loc="lower left", handlelength=1.4,
              labelcolor=INK_2)

    # ---- 3. what survives after removing position -------------------------
    ax = fig.add_subplot(gs[1, 0])
    def gv(feat: str, lab: str) -> float:
        for r in pc:
            if r["features"].startswith(feat) and r["label"] == lab:
                return float(r["spearman"])
        return np.nan
    before = [gv("brightness", "corrected"), gv("DINOv2", "corrected")]
    after = [gv("brightness", "corrected, position removed"),
             gv("DINOv2", "corrected, position removed")]
    x = np.arange(2)
    b1 = ax.bar(x - w / 2 - gap, before, w, color=BRIGHT,
                label="as measured", zorder=3)
    b2 = ax.bar(x + w / 2 + gap, after, w, color=DINO,
                label="with the plate's row/col trend removed", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8.6,
                    color=INK, fontweight="medium", zorder=4)
    ax.axhspan(-NOISE_FLOOR, NOISE_FLOOR, color=INK_3, alpha=0.09, lw=0, zorder=0)
    ax.text(-0.44, NOISE_FLOOR + 0.008, "noise floor", fontsize=7.8,
            color=INK_3, ha="left", va="bottom")
    ax.axhline(0, color=INK_3, lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["brightness (7)", "DINOv2 (768)"],
                       fontsize=8.6, color=INK_2)
    ax.set_ylim(0, 0.47)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("3 · Half of DINOv2's skill was the layout",
                 fontsize=10.5, pad=17, fontweight="medium")
    ax.text(0.5, 1.018, "what is left is real, position-independent thickness signal",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, color=INK_2)
    tidy(ax, "Spearman ρ vs true height")
    ax.legend(fontsize=8, frameon=False, loc="upper left", handlelength=1.4,
              labelcolor=INK_2)

    # ---- 4. what a cell actually contains ---------------------------------
    ax = fig.add_subplot(gs[1, 1])
    from Auto_OPDx.alignment.dataset import (
        fit_cross_modal, solve_photo, solve_plate,
    )
    from build_features import pad_mask_from_photo

    stem = "10-01-2025_P1"
    hm = solve_plate(REPO / "data" / "opdx" / f"{stem}.OPDx")
    ph = solve_photo(next(q for q in (REPO / "data" / "rgb" / "img").iterdir()
                          if q.stem == stem))
    cm = fit_cross_modal(hm, ph)
    M, t = cm.affine[:2], cm.affine[2]
    px = np.linalg.inv(M.T) @ (hm.fit.predict((4, 4))[0] - t)
    sx, sy = cm.um_per_px
    hw, hh = (267 / 2) / sx, (267 / 2) / sy
    xa, xb = int(px[0] - hw), int(px[0] + hw)
    ya, yb = int(px[1] - hh), int(px[1] + hh)
    patch = ph.img[ya:yb, xa:xb]
    mask = pad_mask_from_photo(patch)

    ax.imshow(patch)
    ax.contour(mask.astype(float), levels=[0.5], colors=[CTRL], linewidths=2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    frac = mask.mean()
    ax.add_patch(Rectangle((0.015, 0.015), 0.40, 0.115, transform=ax.transAxes,
                           facecolor=SURFACE, alpha=0.92, lw=0, zorder=4))
    ax.text(0.03, 0.072, f"pad = {frac*100:.1f}% of the cell",
            transform=ax.transAxes, fontsize=9.5, color=INK,
            fontweight="medium", va="center", zorder=5)
    ax.set_title("4 · A full-pitch cell is ~96% bare silicon",
                 fontsize=10.5, pad=17, fontweight="medium")
    ax.text(0.5, 1.018,
            f"{stem} cell (4,4) · outline = pad found from the photo alone",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, color=INK_2)

    fig.suptitle("Tier-1 audit: is the negative result an artifact?  —  no",
                 fontsize=13.5, color=INK, fontweight="semibold", y=0.965)
    fig.text(0.5, 0.917,
             "diluted features, crop tightness and label noise were all ruled "
             "out; what remains is a weak but real structural signal",
             ha="center", fontsize=10, color=INK_2)

    out = AUDIT / "figures" / "audit_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
