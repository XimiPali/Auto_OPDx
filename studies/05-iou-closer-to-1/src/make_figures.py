"""Figures for study 05 (run run_study.py first).

    figures/progression.png        mean IoU step by step, and the ideas that raise IoU but not alignment
    figures/trace_offset.png       the coarse traces: one pad on a trace, and width vs trace offset
    figures/stylus_dragged.png     cells where the pad is missing from the scan at the photo's position
    figures/worst_cells_after.png  the 24 lowest cells under the final recipe
    figures/pairing_error.png      the independent alignment check, both methods
    figures/how_high.png           can it reach 0.8-0.9? oracle, calibration and finer-scan tests

Palette: orange = previous method, blue = new method (as in studies 03/04).

Usage::

    MPLBACKEND=Agg uv run python studies/05-iou-closer-to-1/src/make_figures.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib05 as L5  # noqa: E402
import run_study as RS  # noqa: E402

FIGS = L5.STUDY / "figures"
RESULTS = L5.STUDY / "results"
OLD, NEW, NEW_DARK = "#eb6834", "#2a78d6", "#1c5cab"
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
BLUES = LinearSegmentedColormap.from_list("b", ["#f0efec", "#9ec5f4", "#3987e5", "#104281", "#0d366b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_3, "ytick.color": INK_3, "axes.edgecolor": GRID,
})


def read(name):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ----------------------------------------------------------- progression ----

def fig_progression():
    prog = read("progression.csv")
    labels = ["study 03\nrecipe", "finer\ngrid", "outline\nclean-up", "scan from\nits traces",
              "centre\nthreshold", "broken rims\njoined"]
    old = [float(r["previous_mean_iou"]) for r in prog]
    new = [float(r["new_mean_iou"]) for r in prog]
    ceil = [float(r["new_ceiling"]) for r in prog]
    x = np.arange(len(prog))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.28})
    ax.plot(x, ceil, color=INK_3, linestyle="--", linewidth=1.5, marker="o", markersize=5)
    ax.plot(x, new, color=NEW, linewidth=2.4, marker="o", markersize=8, markeredgecolor=SURFACE)
    ax.plot(x, old, color=OLD, linewidth=2.4, marker="o", markersize=8, markeredgecolor=SURFACE)
    ax.text(x[-1] + 0.12, new[-1], f"new {new[-1]:.3f}", color=NEW, va="center", fontweight="bold")
    ax.text(x[-1] + 0.12, old[-1], f"previous {old[-1]:.3f}", color=OLD, va="center", fontweight="bold")
    ax.text(x[-1] + 0.12, ceil[-1], f"best possible {ceil[-1]:.3f}", color=INK_2, va="center")
    ax.text(x[0] - 0.05, new[0] + 0.012, f"{new[0]:.3f}", color=NEW, ha="center", fontsize=10)
    ax.text(x[0] - 0.05, old[0] - 0.02, f"{old[0]:.3f}", color=OLD, ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.4, len(x) + 0.9)
    ax.set_ylim(0.64, 0.84)
    ax.set_ylabel("mean IoU over 2,855 cells")
    ax.set_title("Each change applied to both methods, one at a time", loc="left", fontsize=12.5, fontweight="bold")
    tidy(ax)

    # IoU up, alignment worse
    mc = {r["recipe"]: r for r in read("moved_centres.csv")}
    pts = [("final recipe (kept)", "final", NEW),
           ("second pass: move photo crop\nso outline centres meet", "x_second_pass", INK_2),
           ("re-centre height crop\non the fitted pad", "x_recentre_fit", INK_2)]
    pts = [(lab, float(mc[k]["new_mean_iou"]), float(mc[k]["pairing_median_um"]), col) for lab, k, col in pts]
    for label, iou, err, color in pts:
        bx.scatter(err, iou, s=110, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        bx.annotate(label, (err, iou), textcoords="offset points", xytext=(10, -4 if "kept" in label else 4),
                    fontsize=9.5, color=color, fontweight="bold" if "kept" in label else "normal")
    bx.annotate("", xy=(pts[1][2], pts[1][1]), xytext=(pts[0][2], pts[0][1]),
                arrowprops=dict(arrowstyle="->", color=INK_3, lw=1.2, linestyle="--"))
    bx.set_xlim(5.3, 7.7)
    bx.set_ylim(0.75, 0.83)
    bx.set_xlabel("real pairing error, median µm (lower = better)")
    bx.set_ylabel("mean IoU")
    bx.set_title("A higher IoU is not always better alignment", loc="left", fontsize=12.5, fontweight="bold")
    tidy(bx, "both")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.17)
    fig.savefig(FIGS / "progression.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------- trace offset ----

def fig_trace_offset():
    rows = read("trace_offset.csv")
    before = [r for r in rows if r["config"] == "finer_grid"]
    after = [r for r in rows if r["config"] == "final"]
    mids = [0.05, 0.15, 0.25, 0.35, 0.45]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.3})

    # one pad centred on a trace, across its traces
    ax = axes[0]
    b = L5.load_bundle("9-29-2025_P4")
    slot = 2 * 8 + 0
    cx, cy = b.centres["new"][slot][2:]
    xs, u, *_ = L5._trace_samples(b, cx, cy)
    w_plate, sigma = L5.plate_trace_model(b)
    xr = np.linspace(cx - 80, cx + 80, 800)
    ax.axvspan(-52.8 / 2, 52.8 / 2, color="#cde2fb", alpha=0.7, zorder=0)
    ax.text(0, 1.16, "pad width in the photo", ha="center", fontsize=9, color=NEW_DARK)
    lin = np.interp(xr, xs, u)
    ax.plot(xr - cx, lin, color=OLD, linewidth=2, linestyle="--", label="linear interpolation (before)")
    x0 = float(np.sum(xs[u > 0.3] * u[u > 0.3]) / np.sum(u[u > 0.3]))
    wfit = L5._fit_width(xs, u, x0, sigma, w_plate, 6.0)
    ax.plot(xr - cx, L5._box(xr, x0, wfit, sigma), color=NEW, linewidth=2.4, label="rebuilt from the traces (after)")
    ax.scatter(xs - cx, u, s=70, color=INK, zorder=5, label="the scan's traces")
    ax.axhline(0.5, color=INK_3, linewidth=0.8, linestyle=":")
    ax.text(78, 0.53, "half height", ha="right", fontsize=9, color=INK_3)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-0.08, 1.25)
    ax.set_xlabel("µm across the traces, from the pad centre")
    ax.set_ylabel("height, fraction of the pad's top")
    ax.set_title("A pad sitting on a trace", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    tidy(ax, "both")

    for ax, key, title, ylab in ((axes[1], "scan_over_photo_width", "Pad width, scan ÷ photo", "median ratio"),
                                 (axes[2], "mean_iou", "Mean IoU", "mean IoU")):
        bv = [float(r[key]) for r in before]
        av = [float(r[key]) for r in after]
        ax.plot(mids, bv, color=OLD, linewidth=2.2, marker="o", markersize=7, markeredgecolor=SURFACE)
        ax.plot(mids, av, color=NEW, linewidth=2.4, marker="o", markersize=7, markeredgecolor=SURFACE)
        ax.text(mids[0], bv[0], "before  ", ha="right", va="center", color=OLD, fontsize=9.5, fontweight="bold")
        ax.text(mids[0], av[0], "after  ", ha="right", va="center", color=NEW, fontsize=9.5, fontweight="bold")
        if key.startswith("scan"):
            ax.axhline(1.0, color=INK_3, linewidth=0.8, linestyle=":")
        ax.set_xticks(mids)
        ax.set_xticklabels(["on a\ntrace", "0.1–0.2", "0.2–0.3", "0.3–0.4", "halfway"], fontsize=9)
        ax.set_xlim(-0.07, 0.5)
        ax.set_xlabel("where the pad centre sits between traces")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        tidy(ax)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.18)
    fig.savefig(FIGS / "trace_offset.png", dpi=180)
    plt.close(fig)


# -------------------------------------------------------- stylus dragged ----

def fig_stylus_dragged():
    from Auto_OPDx.alignment.rgb import load_rgb
    cells = [("10-22-2025_P8", 0, 2), ("10-13-2025_P10", 0, 4), ("10-27-2025_P2", 0, 1)]
    cfg = RS.FINAL
    fig, axes = plt.subplots(2, len(cells), figsize=(4.2 * len(cells), 7.6), gridspec_kw={"hspace": 0.28, "wspace": 0.08})
    for j, (stem, r, c) in enumerate(cells):
        b = L5.load_bundle(stem)
        img = load_rgb(b.photo)
        field = L5.photo_field(img, "gray", False)
        o = L5.evaluate_cell(b, field, r * 8 + c, "new", cfg, img, L5.plate_trace_model(b))
        raw = L5.crop_scan(b, b.centres["new"][r * 8 + c][2:], cfg)
        A = o.arrays
        axes[0, j].imshow(A["rgb"])
        if A["photo_mask"] is not None:
            axes[0, j].contour(A["photo_mask"].astype(float), [0.5], colors=[NEW], linewidths=1.8)
        axes[0, j].set_title(f"{stem} r{r} c{c}\nphoto: the pad is here", fontsize=10.5, color=INK_2)
        axes[1, j].imshow(np.nan_to_num(raw), cmap=BLUES, vmin=0, vmax=max(float(np.nanmax(raw)), 1))
        if A["photo_mask"] is not None:
            axes[1, j].contour(A["photo_mask"].astype(float), [0.5], colors=[NEW], linewidths=1.5, linestyles="dashed")
        axes[1, j].set_title("height map, same place: no pad;\nthe material sits further along the trace",
                             fontsize=10.5, color=INK_2)
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
    fig.suptitle("Pads the stylus appears to have dragged: present in the photo, displaced in the scan (all in row 0, "
                 "where each trace starts)", fontsize=11.5, fontweight="bold", y=0.98)
    fig.savefig(FIGS / "stylus_dragged.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------- worst after ----

def fig_worst_after(n=24):
    rows = [r for r in read("cells_final.csv") if r["method"] == "new"]
    rows.sort(key=lambda r: float(r["iou"]))
    picks = [(r["plate"], int(r["row"]), int(r["col"]), float(r["iou"])) for r in rows[:n]]
    keep: dict = {}
    for stem, r, c, _ in picks:
        keep.setdefault(stem, set()).add((r * 8 + c, "new", RS.FINAL.name))
    outs = L5.run(sorted(keep), [RS.FINAL], keep=keep)
    got = {(o.plate, o.row, o.col): o for o in outs if o.arrays}
    cols = 6
    nrows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(nrows, cols * 2, figsize=(cols * 2 * 1.45, nrows * 1.8),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.45})
    for i, (stem, r, c, v) in enumerate(picks):
        rr, cc = divmod(i, cols)
        o = got[(stem, r, c)]
        A = o.arrays
        ap, asn = axes[rr, 2 * cc], axes[rr, 2 * cc + 1]
        ap.imshow(A["rgb"])
        sc = A["scan_crop"]
        asn.imshow(np.nan_to_num(sc if sc is not None else np.zeros((10, 10))), cmap=BLUES)
        for ax in (ap, asn):
            if A["photo_mask"] is not None:
                ax.contour(A["photo_mask"].astype(float), [0.5], colors=[NEW], linewidths=1.2)
            if A["scan_mask"] is not None:
                ax.contour(A["scan_mask"].astype(float), [0.5], colors=[INK if ax is ap else OLD], linewidths=1.1,
                           linestyles="dashed")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
        ap.set_title(f"{stem} r{r}c{c}\nIoU {v:.2f}", fontsize=7, loc="left", color=INK_2)
    for j in range(len(picks), nrows * cols):
        rr, cc = divmod(j, cols)
        axes[rr, 2 * cc].axis("off")
        axes[rr, 2 * cc + 1].axis("off")
    fig.suptitle(f"The {len(picks)} lowest-IoU cells after the fixes (photo | rebuilt height map). "
                 "Blue = photo outline, dashed = height-map outline.", fontsize=10, y=0.995)
    fig.savefig(FIGS / "worst_cells_after.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------- pairing error ----

def fig_pairing_error():
    stems = L5.L.comparable_plates()
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    rows = {r["method"]: r for r in read("pairing_error.csv")}
    for method, color, label in (("old", OLD, "previous"), ("new", NEW, "new")):
        errs = []
        for st in stems:
            b = L5.load_bundle(st)
            pts = [(b.centres[method][s][:2], b.centres[method][s][2:]) for s in range(64)
                   if b.status[method][s] not in ("excluded_missing", "not_found")
                   and np.isfinite(b.centres[method][s]).all()]
            if len(pts) < 20:
                continue
            P = np.array([p for p, _ in pts])
            S = np.array([q for _, q in pts])
            errs.append(np.linalg.norm(RS._robust_loo(P, S), axis=1))
        e = np.sort(np.concatenate(errs))
        ax.plot(e, np.arange(1, len(e) + 1) / len(e) * 100, color=color, linewidth=2.4)
        med = float(rows[method]["median_um"])
        ax.text(med + 1, 50 - (6 if method == "old" else -6), f"{label}: median {med:.1f} µm", color=color,
                fontsize=10.5, fontweight="bold")
    ax.axhline(50, color=GRID, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 200)
    ax.set_ylim(0, 100)
    ax.set_xlabel("photo pad centre vs scan pad centre, after a smooth plate-wide map (µm)")
    ax.set_ylabel("% of pairs at or below")
    ax.set_title("Independent alignment check (does not use the outlines)", loc="left", fontsize=12, fontweight="bold")
    tidy(ax, "both")
    fig.tight_layout()
    fig.savefig(FIGS / "pairing_error.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- how high ----

def fig_how_high():
    rows = read("push_higher.csv")
    sim = read("finer_scan_simulation.csv")
    labels = ["today", "+ per-chip size\ncalibration", "if both agreed\non size", "if both agreed\non centre",
              "if both agreed on\nsize and centre"]
    vals = [float(r["new_mean_iou"]) for r in rows]
    kinds = [r["kind"] for r in rows]
    colors = [NEW if k == "measured" else NEW_DARK if k == "calibration" else "#b7d3f6" for k in kinds]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.25})
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    for bar, v, k in zip(bars, vals, kinds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, f"{v:.2f}", ha="center", fontsize=11,
                fontweight="bold", color=INK)
    for yv, lab in ((0.8, "0.8"), (0.9, "0.9")):
        ax.axhline(yv, color=INK_3, linewidth=0.8, linestyle=":")
        ax.text(len(vals) - 0.45, yv + 0.004, lab, color=INK_3, fontsize=9, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0.6, 0.95)
    ax.set_ylabel("mean IoU, new method")
    ax.set_title("What each remaining limit costs (light bars: what-if tests, not methods)", loc="left",
                 fontsize=11.5, fontweight="bold")
    tidy(ax)
    sp = [float(r["line_spacing_um"]) for r in sim]
    sv = [float(r["mean_iou_vs_true_outline"]) for r in sim]
    bx.plot(sp, sv, color=NEW, linewidth=2.4, marker="o", markersize=8, markeredgecolor=SURFACE)
    for xv, yv in zip(sp, sv):
        bx.text(xv, yv + 0.006, f"{yv:.2f}", ha="center", fontsize=10, color=NEW_DARK)
    bx.axvspan(28, 32, color=GRID, alpha=0.9, zorder=0)
    bx.text(30, min(sv) - 0.03, "today", ha="center", fontsize=9.5, color=INK_2)
    bx.set_xlim(33, 2)
    bx.set_ylim(0.75, 0.95)
    bx.set_xlabel("distance between the needle's lines, µm")
    bx.set_ylabel("mean IoU")
    bx.set_title("Simulated: a perfect photo pad, scanned and rebuilt", loc="left", fontsize=11.5, fontweight="bold")
    tidy(bx, "both")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.17)
    fig.savefig(FIGS / "how_high.png", dpi=180)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    for name, fn in (("progression", fig_progression), ("trace_offset", fig_trace_offset),
                     ("stylus_dragged", fig_stylus_dragged), ("pairing_error", fig_pairing_error),
                     ("worst_cells_after", fig_worst_after), ("how_high", fig_how_high)):
        if not only or name in only:
            fn()
            print(f"wrote figures/{name}.png")


if __name__ == "__main__":
    main()
