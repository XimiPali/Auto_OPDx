"""Render the figures for the alignment IoU study.

    figures/iou_summary.png      distribution, per-plate comparison, failure
                                 counts, and what an IoU value means in um
    figures/iou_examples.png     the same cells under both methods, masks drawn
    figures/iou_plate_grid.png   per-cell IoU on two plates as an 8x8 grid

Palette: categorical slots 1-3 of the reference palette (blue = new tool,
orange = previous crops, aqua = new tool with detected centres). Aqua carries
a contrast WARN on the light surface, so every aqua mark is direct-labelled.
Heatmaps use the blue sequential ramp.

Usage::

    uv run python studies/03-alignment-iou/src/make_figures.py
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import iou_lib as L

RESULTS = L.STUDY / "results"
FIGS = L.STUDY / "figures"

NEW = "#2a78d6"       # slot 1
OLD = "#eb6834"       # slot 2
DET = "#1baf7a"       # slot 3 - contrast WARN, always direct-labelled
SURFACE = "#fcfcfb"
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("ref_blue", RAMP)
COLOR = {"old": OLD, "new": NEW, "new_lattice": DET}
SHORT = {"old": "previous crops", "new": "align-plates", "new_lattice": "align-plates, lattice slots only"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "font.family": "sans-serif", "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_3, "ytick.color": INK_3,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.titlecolor": INK,
})


def read(name: str) -> list[dict]:
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- figure 1 ----

def fig_summary(cells, plates, summary):
    scored = [c for c in cells if c["status"] != "excluded_missing"]
    by_m = defaultdict(list)
    for c in scored:
        by_m[c["method"]].append(float(c["iou"]))
    stats = {s["method"]: s for s in summary}

    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28, left=0.07, right=0.98, top=0.90, bottom=0.07)

    # A. distribution -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 21)
    for m in ("old", "new", "new_lattice"):
        v = np.array(by_m[m])
        h, _ = np.histogram(v, bins)
        h = h / len(v) * 100
        ax.stairs(h, bins, color=COLOR[m], linewidth=2 if m != "new_lattice" else 1.4,
                  linestyle="-" if m != "new_lattice" else "--", baseline=None)
        if m != "new_lattice":
            ax.stairs(h, bins, color=COLOR[m], fill=True, alpha=0.10, baseline=0)
    for m, dy in (("old", 0), ("new", 1), ("new_lattice", 2)):
        s = stats[m]
        ax.text(0.015, 0.97 - 0.085 * dy,
                f"{SHORT[m]}  mean {float(s['mean_iou']):.2f} · {float(s['frac_iou_zero'])*100:.0f}% zero",
                transform=ax.transAxes, color=COLOR[m], fontsize=8.5, va="top")
    ax.set_xlabel("IoU between the photo pad and the height-map pad")
    ax.set_ylabel("% of cells")
    ax.set_xlim(0, 1)
    ax.set_title("A  Per-cell IoU, all plates", loc="left", fontsize=10, fontweight="bold")
    tidy(ax)

    # B. per-plate scatter ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    pm = defaultdict(dict)
    for p in plates:
        pm[p["plate"]][p["method"]] = float(p["mean_iou"])
    o = np.array([pm[s]["old"] for s in pm])
    n = np.array([pm[s]["new"] for s in pm])
    d = np.array([pm[s]["new_lattice"] for s in pm])
    lo = min(o.min(), n.min(), d.min()) - 0.03
    ax.plot([lo, 0.85], [lo, 0.85], color=INK_3, linewidth=0.8, linestyle=":")
    ax.scatter(o, d, s=26, facecolor="none", edgecolor=DET, linewidth=1.1, zorder=3)
    ax.scatter(o, n, s=30, color=NEW, edgecolor=SURFACE, linewidth=0.8, zorder=4)
    better_n = int((n > o).sum())
    better_d = int((d > o).sum())
    ax.text(0.03, 0.97, f"align-plates above the line on {better_n} of {len(o)} plates",
            transform=ax.transAxes, color=NEW, fontsize=8.5, va="top")
    ax.text(0.03, 0.90, f"lattice slots only (open circles): {better_d} of {len(o)}",
            transform=ax.transAxes, color=DET, fontsize=8.5, va="top")
    # name the plates where the previous crops collapsed
    for stem in pm:
        if pm[stem]["old"] < 0.45:
            ax.annotate(stem, (pm[stem]["old"], pm[stem]["new"]), textcoords="offset points",
                        xytext=(6, -3), fontsize=7, color=INK_2)
    ax.set_xlabel("previous crops — plate mean IoU")
    ax.set_ylabel("new tool — plate mean IoU")
    ax.set_xlim(lo, 0.85)
    ax.set_ylim(lo, 0.85)
    ax.set_aspect("equal")
    ax.set_title("B  Plate by plate (45 plates)", loc="left", fontsize=10, fontweight="bold")
    tidy(ax, grid_axis="both")

    # C. failure accounting ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    cats = [("not_found", "not found by the method"), ("no_pad_in_crop", "crop contains no pad"),
            ("low", "IoU below 0.3 (wrong place)")]
    methods = ("old", "new_lattice", "new")
    x = np.arange(len(cats))
    w = 0.26
    for i, m in enumerate(methods):
        mine = [c for c in scored if c["method"] == m]
        counts = [
            sum(c["status"] == "not_found" for c in mine),
            sum(c["status"] == "no_pad_in_crop" for c in mine),
            sum(c["status"] == "ok" and float(c["iou"]) < 0.3 for c in mine),
        ]
        bars = ax.bar(x + (i - 1) * w, counts, width=w - 0.03, color=COLOR[m])
        for b, v in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", va="bottom",
                    fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in cats], fontsize=8.5)
    ax.set_ylabel(f"cells (of {len(scored)//3} with a pad)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)      # headroom so the legend clears the bars
    ax.set_title("C  Where the crops go wrong", loc="left", fontsize=10, fontweight="bold")
    ax.legend(handles=[Line2D([], [], color=COLOR[m], linewidth=6, label=SHORT[m]) for m in methods],
              frameon=False, fontsize=8, loc="upper left")
    tidy(ax)

    # D. what an IoU value means ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ok = [c for c in scored if c["status"] == "ok" and c["offset_um"] != ""]
    for m in ("old", "new"):
        pts = [(float(c["offset_um"]), float(c["iou"])) for c in ok if c["method"] == m]
        if pts:
            a = np.array(pts)
            ax.scatter(a[:, 0], a[:, 1], s=7, color=COLOR[m], alpha=0.25, linewidths=0, zorder=2)
    offs = np.linspace(0, 80, 200)
    ceiling = float(stats["new"]["mean_ceiling_iou"])
    ref = L.rectangle_iou_vs_offset(61, 98, offs)
    ax.plot(offs, ref, color=INK_2, linewidth=1.6, zorder=3)
    ax.plot(offs, ref * ceiling, color=INK_2, linewidth=1.2, linestyle="--", zorder=3)
    box = dict(facecolor=SURFACE, edgecolor="none", pad=1.5)
    ax.text(37, L.rectangle_iou_vs_offset(61, 98, np.array([37.0]))[0] + 0.075,
            "two identical 61×98 µm pads, shifted", fontsize=7.5, color=INK_2, bbox=box)
    ax.text(37, L.rectangle_iou_vs_offset(61, 98, np.array([37.0]))[0] * ceiling - 0.075,
            f"× {ceiling:.2f} shape ceiling (scan samples x every 30 µm)", fontsize=7.5,
            color=INK_2, va="top", bbox=box)
    ax.axhline(ceiling, color=GRID, linewidth=0.8)
    ax.set_xlabel("distance between the two pad centroids, µm")
    ax.set_ylabel("IoU")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1)
    ax.set_title("D  Reading the number: IoU vs offset", loc="left", fontsize=10, fontweight="bold")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=COLOR[m], markersize=5, label=SHORT[m])
                       for m in ("old", "new")], frameon=False, fontsize=8, loc="upper right")
    tidy(ax, grid_axis="both")

    s_old, s_new, s_lat = stats["old"], stats["new"], stats["new_lattice"]
    n_pad = int(s_old["cells_with_pad"])
    fig.suptitle("Do the photo crop and the height-map crop show the same pad in the same place?",
                 x=0.07, ha="left", fontsize=12.5, fontweight="bold", y=0.975)
    fig.text(0.07, 0.935,
             f"Mean IoU over {n_pad} cells on 45 plates:  previous crops {float(s_old['mean_iou']):.2f}  →  "
             f"align-plates {float(s_new['mean_iou']):.2f}  (lattice slots only, before this study: "
             f"{float(s_lat['mean_iou']):.2f}).   Cells with zero overlap: "
             f"{int(float(s_old['frac_iou_zero'])*n_pad+0.5)} → {int(float(s_new['frac_iou_zero'])*n_pad+0.5)}.",
             fontsize=9, color=INK_2)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "iou_summary.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2 ----

def _outline(ax, mask, color):
    if mask is not None:
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.4)


def _panel_photo(ax, res, color):
    if res.photo_rgb is not None:
        ax.imshow(res.photo_rgb)
    else:
        ax.text(0.5, 0.5, "no crop\n(not found)", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=INK_3)
    _outline(ax, res.photo_mask, color)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID)


def _panel_height(ax, res, color, vmax):
    if res.hm_crop is not None:
        ax.imshow(np.nan_to_num(res.hm_crop, nan=0.0), cmap=BLUES, vmin=0, vmax=vmax)
    else:
        ax.text(0.5, 0.5, "no crop\n(not found)", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=INK_3)
    _outline(ax, res.hm_mask, color)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID)


def _panel_overlay(ax, res):
    canvas = np.ones((L.GRID, L.GRID, 3)) * 0.985
    if res.photo_mask is not None and res.hm_mask is not None:
        both = res.photo_mask & res.hm_mask
        only_p = res.photo_mask & ~res.hm_mask
        only_h = res.hm_mask & ~res.photo_mask
        canvas[only_p] = mpl.colors.to_rgb("#f4b89a")      # photo only, tinted
        canvas[only_h] = mpl.colors.to_rgb("#a9c9f2")      # height only, tinted
        canvas[both] = mpl.colors.to_rgb("#4a4a48")        # overlap, neutral dark
    ax.imshow(canvas)
    label = {"ok": f"IoU {res.iou:.2f}", "not_found": "IoU 0 — not found",
             "no_pad_in_crop": "IoU 0 — no pad in crop"}[res.status]
    ax.text(0.5, -0.08, label, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color=INK, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID)


def fig_examples(cells):
    """Pick cells that tell the story and redraw them under both methods."""
    key = lambda c: (c["plate"], int(c["row"]), int(c["col"]))  # noqa: E731
    old = {key(c): c for c in cells if c["method"] == "old" and c["status"] != "excluded_missing"}
    new = {key(c): c for c in cells if c["method"] == "new" and c["status"] != "excluded_missing"}

    def pick(pred, sort_key, interior=False):
        cands = [k for k in old if k in new and pred(old[k], new[k])
                 and (not interior or (1 <= k[1] <= 6 and 1 <= k[2] <= 6))]
        cands.sort(key=sort_key)
        return cands[0] if cands else None

    picks = [
        ("previous pipeline found nothing", pick(lambda o, n: o["status"] == "not_found" and n["status"] == "ok",
                                                  lambda k: -float(new[k]["iou"]))),
        ("previous crop misses the pad", pick(lambda o, n: o["status"] == "ok" and float(o["iou"]) < 0.05 and n["status"] == "ok",
                                              lambda k: -float(new[k]["iou"]))),
        ("typical cell", pick(lambda o, n: o["status"] == "ok" and n["status"] == "ok",
                              lambda k: abs(float(old[k]["iou"]) - 0.67) + abs(float(new[k]["iou"]) - 0.71),
                              interior=True)),
        ("both good", pick(lambda o, n: o["status"] == "ok" and n["status"] == "ok",
                           lambda k: -(float(old[k]["iou"]) + float(new[k]["iou"])), interior=True)),
    ]
    picks = [(t, k) for t, k in picks if k is not None]

    # Recompute only the plates involved, keeping the arrays.
    prof, img = L.load_old_metadata()
    evals = {}
    for _, (stem, _, _) in picks:
        if stem not in evals:
            evals[stem] = L.evaluate_plate(stem, prof[stem], img[stem], keep_arrays=True)

    nrows = len(picks)
    fig, axes = plt.subplots(nrows, 7, figsize=(13.2, 2.05 * nrows + 1.1),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.18, 1, 1, 1], "wspace": 0.06, "hspace": 0.5})
    axes = np.atleast_2d(axes)
    for i, (title, (stem, r, c)) in enumerate(picks):
        ev = evals[stem]
        res = {x.method: x for x in ev.cells if x.row == r and x.col == c}
        vmax = max(np.nanpercentile(res["new"].hm_crop, 99.5) if res["new"].hm_crop is not None else 1, 1)
        for j, m in ((0, "old"), (4, "new")):
            _panel_photo(axes[i, j], res[m], COLOR[m])
            _panel_height(axes[i, j + 1], res[m], COLOR[m], vmax)
            _panel_overlay(axes[i, j + 2], res[m])
        axes[i, 3].axis("off")
        axes[i, 0].set_ylabel(f"{title}\n{stem} r{r} c{c}", fontsize=8, color=INK_2, rotation=0,
                              ha="right", va="center", labelpad=8)
    for j, t in ((0, "photo"), (1, "height map"), (2, "masks overlaid"), (4, "photo"), (5, "height map"), (6, "masks overlaid")):
        axes[0, j].set_title(t, fontsize=8.5, color=INK_2, pad=4)
    fig.text(0.30, 0.985, "previous crops", ha="center", va="top", fontsize=11, fontweight="bold", color=OLD)
    fig.text(0.735, 0.985, "align-plates", ha="center", va="top", fontsize=11, fontweight="bold", color=NEW)
    fig.text(0.5, 0.012, f"Each crop is the same {ev.window_um:.0f} µm window in both instruments. Outlines are the pad masks; "
             "in the overlay, dark = both, orange tint = photo only, blue tint = height map only.",
             ha="center", fontsize=8, color=INK_2)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.90, bottom=0.07)
    fig.savefig(FIGS / "iou_examples.png", dpi=170)
    plt.close(fig)
    return [(t, k) for t, k in picks]


# ---------------------------------------------------------------- figure 3 ----

def fig_plate_grid(cells, plates):
    pm = defaultdict(dict)
    for p in plates:
        pm[p["plate"]][p["method"]] = p
    worst_old = min(pm, key=lambda s: float(pm[s]["old"]["mean_iou"]))
    best_old = max(pm, key=lambda s: float(pm[s]["old"]["mean_iou"]))
    stems = [worst_old, best_old]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), gridspec_kw={"wspace": 0.12, "hspace": 0.32})
    for i, stem in enumerate(stems):
        for j, m in enumerate(("old", "new_lattice", "new")):
            ax = axes[i, j]
            grid = np.full((8, 8), np.nan)
            excluded = np.zeros((8, 8), bool)
            for c in cells:
                if c["plate"] == stem and c["method"] == m:
                    r, k = int(c["row"]), int(c["col"])
                    if c["status"] == "excluded_missing":
                        excluded[r, k] = True
                    else:
                        grid[r, k] = float(c["iou"])
            ax.imshow(np.nan_to_num(grid, nan=0), cmap=BLUES, vmin=0, vmax=0.8)
            for r in range(8):
                for k in range(8):
                    if excluded[r, k]:
                        ax.text(k, r, "×", ha="center", va="center", fontsize=11, color=INK_3)
                    else:
                        v = grid[r, k]
                        ax.text(k, r, f"{v:.2f}"[1:] if v > 0 else "0", ha="center", va="center", fontsize=7,
                                color=SURFACE if v > 0.45 else INK)
            # cell spacers
            for t in np.arange(-0.5, 8, 1):
                ax.axhline(t, color=SURFACE, linewidth=2)
                ax.axvline(t, color=SURFACE, linewidth=2)
            ax.set_xticks(range(8)); ax.set_yticks(range(8))
            ax.set_xticklabels([f"c{k}" for k in range(8)], fontsize=7)
            ax.set_yticklabels([f"r{k}" for k in range(8)], fontsize=7)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            p = pm[stem][m]
            ax.set_title(f"{SHORT[m]}\nmean {float(p['mean_iou']):.2f} · {p['n_zero']} zero",
                         fontsize=9, color=COLOR[m], pad=6)
        axes[i, 0].text(-0.28, 0.5, stem, transform=axes[i, 0].transAxes, rotation=90, ha="center",
                        va="center", fontsize=10, fontweight="bold", color=INK)
    fig.suptitle("Per-cell IoU on the plate where the previous crops did worst (top) and best (bottom)",
                 x=0.5, fontsize=11.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.015, "× = no pad in the scan or the photo at that slot (excluded for every method). "
             "Numbers are IoU; darker is better.", ha="center", fontsize=8, color=INK_2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.06)
    fig.savefig(FIGS / "iou_plate_grid.png", dpi=170)
    plt.close(fig)
    return stems


def main() -> None:
    cells = read("cells.csv")
    plates = read("plates.csv")
    summary = read("summary.csv")
    FIGS.mkdir(exist_ok=True)
    fig_summary(cells, plates, summary)
    print("wrote figures/iou_summary.png")
    stems = fig_plate_grid(cells, plates)
    print(f"wrote figures/iou_plate_grid.png  ({', '.join(stems)})")
    picks = fig_examples(cells)
    print("wrote figures/iou_examples.png  " + "; ".join(f"{t}: {k[0]} r{k[1]}c{k[2]}" for t, k in picks))


if __name__ == "__main__":
    main()
