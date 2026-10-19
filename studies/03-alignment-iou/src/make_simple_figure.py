"""One slide-ready figure for an audience that has not met IoU before.

    figures/iou_comparison_simple.png

Layout: a three-step picture of what IoU means, then the previous crops on
the left and the updated tool on the right, each with its headline number,
three supporting tiles, and the same two real cells so the number has a face.

Usage::

    uv run python studies/03-alignment-iou/src/make_simple_figure.py
"""

from __future__ import annotations

import csv
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

import iou_lib as L

RESULTS = L.STUDY / "results"
FIGS = L.STUDY / "figures"

OLD, NEW = "#eb6834", "#2a78d6"
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
TILE = "#f3f2ef"
HEIGHT_OUTLINE = "#1f1f1e"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
})

# the same two cells for both sides: one the previous pipeline got wrong, one typical
EXAMPLES = [("9-26-2025_P2", 3, 7, "a cell the previous crop missed"),
            ("9-29-2025_P2", 5, 3, "a typical cell")]


def read_summary() -> dict:
    with (RESULTS / "summary.csv").open() as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def plate_wins() -> tuple[int, int]:
    """(plates where the updated tool scores higher, plates compared)."""
    by_plate: dict[str, dict[str, float]] = {}
    with (RESULTS / "plates.csv").open() as f:
        for r in csv.DictReader(f):
            by_plate.setdefault(r["plate"], {})[r["method"]] = float(r["mean_iou"])
    wins = sum(v["new"] > v["old"] for v in by_plate.values())
    return wins, len(by_plate)


def explainer(ax):
    """Three pairs of boxes: no overlap, half overlap, full overlap."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    cases = [(0.55, 0.0, "IoU = 0"), (3.95, 0.5, "IoU ≈ 0.5"), (7.35, 1.0, "IoU = 1")]
    w, h = 1.3, 1.0
    for x0, frac, label in cases:
        shift = (1 - frac) * w + (0.18 if frac == 0 else 0)   # visible gap when disjoint
        a = Rectangle((x0, 0.9), w, h, facecolor=OLD, alpha=0.55, edgecolor="none")
        b = Rectangle((x0 + shift, 0.9), w, h, facecolor=NEW, alpha=0.55, edgecolor="none")
        ax.add_patch(a)
        ax.add_patch(b)
        ax.text(x0 + (w + shift) / 2, 0.45, label, ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(5, 2.25, "IoU = overlap ÷ total area of the two outlines", ha="center", va="center",
            fontsize=11.5, color=INK_2)
    ax.text(1.2, 0.05, "no overlap — the crop is in the wrong place", ha="center", fontsize=8.5, color=INK_3)
    ax.text(4.9, 0.05, "partly overlapping — a few µm off", ha="center", fontsize=8.5, color=INK_3)
    ax.text(8.0, 0.05, "same place — perfectly aligned", ha="center", fontsize=8.5, color=INK_3)


def tile(ax, value, label, color):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor=TILE, edgecolor="none", transform=ax.transAxes))
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=20, fontweight="bold", color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=9, color=INK_2, transform=ax.transAxes, wrap=True)


def hero(ax, title, mean_iou, color, blurb):
    ax.axis("off")
    ax.text(0.5, 0.93, title, ha="center", va="top", fontsize=16, fontweight="bold", color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.52, f"{mean_iou:.2f}", ha="center", va="center", fontsize=56, fontweight="bold",
            color=color, transform=ax.transAxes)
    ax.text(0.5, 0.12, blurb, ha="center", va="center", fontsize=10, color=INK_2, transform=ax.transAxes)


def example(ax, res, color, caption):
    if res.photo_rgb is not None:
        ax.imshow(res.photo_rgb)
    if res.photo_mask is not None:
        ax.contour(res.photo_mask.astype(float), levels=[0.5], colors=[color], linewidths=2.2)
    if res.hm_mask is not None:
        ax.contour(res.hm_mask.astype(float), levels=[0.5], colors=[HEIGHT_OUTLINE], linewidths=2.0,
                   linestyles="dashed")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    label = f"IoU {res.iou:.2f}" if res.status == "ok" else "IoU 0 — not found"
    ax.set_title(caption, fontsize=9, color=INK_2, pad=4)
    ax.text(0.5, -0.07, label, transform=ax.transAxes, ha="center", va="top", fontsize=12,
            fontweight="bold", color=color)


def main() -> None:
    s = read_summary()
    old, new = s["old"], s["new"]
    n_pad = int(old["cells_with_pad"])
    wins, n_plates = plate_wins()

    prof, img = L.load_old_metadata()
    evals = {stem: L.evaluate_plate(stem, prof[stem], img[stem], keep_arrays=True)
             for stem in {e[0] for e in EXAMPLES}}

    fig = plt.figure(figsize=(13, 9.6))
    gs = fig.add_gridspec(
        5, 7,
        height_ratios=[0.95, 1.15, 0.72, 1.55, 0.05],
        width_ratios=[1, 1, 1, 0.22, 1, 1, 1],
        left=0.04, right=0.96, top=0.86, bottom=0.07, hspace=0.45, wspace=0.18,
    )

    fig.text(0.5, 0.965, "IoU metric comparison: previous crops vs. updated alignment",
             ha="center", va="top", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.5, 0.925,
             "For every cell we outline the pad in the photo and in the height map. "
             "If the two crops are aligned, the outlines land on top of each other.",
             ha="center", va="top", fontsize=11.5, color=INK_2)

    explainer(fig.add_subplot(gs[0, :]))

    # hero numbers
    hero(fig.add_subplot(gs[1, 0:3]), "PREVIOUS crops", float(old["mean_iou"]), OLD,
         f"average overlap across {n_pad:,} cells on {n_plates} plates")
    hero(fig.add_subplot(gs[1, 4:7]), "UPDATED  align-plates", float(new["mean_iou"]), NEW,
         f"average overlap across the same {n_pad:,} cells")

    # arrow between the heroes
    mid = fig.add_subplot(gs[1, 3])
    mid.axis("off")
    mid.text(0.5, 0.52, "→", ha="center", va="center", fontsize=40, color=INK_3, transform=mid.transAxes)

    # tiles
    tiles_old = [
        (f"{float(old['median_iou']):.2f}", "median IoU"),
        (f"{int(old['n_failures'])}", "cells where the crop\ncontained no pad"),
        (f"{float(old['median_offset_um']):.1f} µm", "typical distance between\nthe two outlines"),
    ]
    tiles_new = [
        (f"{float(new['median_iou']):.2f}", "median IoU"),
        (f"{int(new['n_failures'])}", "cells where the crop\ncontained no pad"),
        (f"{float(new['median_offset_um']):.1f} µm", "typical distance between\nthe two outlines"),
    ]
    for j, (v, lab) in enumerate(tiles_old):
        tile(fig.add_subplot(gs[2, j]), v, lab, OLD)
    for j, (v, lab) in enumerate(tiles_new):
        tile(fig.add_subplot(gs[2, 4 + j]), v, lab, NEW)

    # examples: the same two cells under each method
    for i, (stem, r, c, caption) in enumerate(EXAMPLES):
        ev = evals[stem]
        res = {x.method: x for x in ev.cells if x.row == r and x.col == c}
        example(fig.add_subplot(gs[3, i]), res["old"], OLD, caption)
        example(fig.add_subplot(gs[3, 4 + i]), res["new"], NEW, caption)
    key = fig.add_subplot(gs[3, 2])
    key.axis("off")
    key.text(0.05, 0.80, "How to read\nthe cells", fontsize=10, fontweight="bold", color=INK, va="top",
             transform=key.transAxes)
    key.plot([0.08, 0.30], [0.50, 0.50], color=INK_3, linewidth=2.2, transform=key.transAxes)
    key.text(0.36, 0.50, "pad as the camera\nsees it (orange / blue)", fontsize=8.5, color=INK_2, va="center",
             transform=key.transAxes)
    key.plot([0.08, 0.30], [0.28, 0.28], color=HEIGHT_OUTLINE, linewidth=2.0, linestyle="dashed",
             transform=key.transAxes)
    key.text(0.36, 0.28, "pad as the profilometer\nsees it", fontsize=8.5, color=INK_2, va="center",
             transform=key.transAxes)
    key.text(0.05, 0.04, "Same two cells\non both sides.", fontsize=8.5, color=INK_3, va="bottom",
             transform=key.transAxes)
    spacer = fig.add_subplot(gs[3, 6])
    spacer.axis("off")
    spacer.text(0.5, 0.5, f"wins on\n{wins} of {n_plates}\nplates", ha="center", va="center", fontsize=13,
                fontweight="bold", color=NEW, transform=spacer.transAxes)

    fig.text(0.5, 0.012,
             f"Scored on the {n_plates} plates both pipelines processed; cells with no pad in the scan or the photo "
             f"are excluded for both.\nThe best any method can reach on this data is ≈{float(new['mean_ceiling_iou']):.2f}, "
             "because the profilometer samples the pad only every 30 µm along one axis.",
             ha="center", va="bottom", fontsize=8.5, color=INK_3, linespacing=1.5)

    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "iou_comparison_simple.png", dpi=170)
    plt.close(fig)
    print("wrote figures/iou_comparison_simple.png")


if __name__ == "__main__":
    main()
