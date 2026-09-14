"""Render the single summary figure for the label-correction result.

Three things in one image:
  top    - where the signal went: raw z, the fitted substrate plane, and the
           remainder, all on ONE shared colour scale so the tilt's share is
           visible rather than asserted
  bottom left  - what the label actually tracks: standardised label against row
           index, previous vs rebuilt, pooled over all plates
  bottom right - the answer: R^2 of a position-only baseline against the photo
           model, under each label definition

Usage::

    uv run python studies/01-label-alignment/src/make_summary_figure.py
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parent.parent
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from Auto_OPDx.alignment.dataset import solve_plate
from Auto_OPDx.alignment.plate import fit_background_plane, plane_surface

# --- design tokens -----------------------------------------------------------
# Categorical slots 1 and 2 of the reference palette; validated all-pairs in
# light mode (CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface).
PREV = "#2a78d6"     # previous labels (raw z_max)
NEW = "#eb6834"      # rebuilt labels (plane-relative)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e0"
# Sequential = one hue, light -> dark. Not a rainbow: the three top panels share
# a scale, so a perceptually monotonic single hue is what makes them comparable.
SEQ = "Blues"

PLATE = "10-01-2025_P1"

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_3,
    "ytick.color": INK_3,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
})


def row_profiles() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Per-row mean of each label, standardised within plate, pooled over plates.

    Standardising per plate puts both label definitions on one axis, so no dual
    axis is needed: the slope *is* the correlation.
    """
    old_rows: list[np.ndarray] = []
    new_rows: list[np.ndarray] = []
    r_old: list[float] = []
    r_new: list[float] = []

    for path in sorted((REPO / "data" / "opdx").glob("*.OPDx")):
        try:
            s = solve_plate(path)
        except Exception:
            continue
        o = np.full((8, 8), np.nan)
        n = np.full((8, 8), np.nan)
        for r in range(8):
            for c in range(8):
                cx, cy = s.fit.predict((r, c))[0]
                xi = np.where(np.abs(s.plate.x - cx) <= 55)[0]
                yi = np.where(np.abs(s.plate.y - cy) <= 55)[0]
                if xi.size and yi.size:
                    sl = np.ix_(yi, xi)
                    o[r, c] = s.plate.z[sl].max()
                    n[r, c] = s.resid[sl].max()
        for M, store, rs in ((o, old_rows, r_old), (n, new_rows, r_new)):
            v = M.ravel()
            g = np.isfinite(v)
            if g.sum() < 20 or v[g].std() == 0:
                continue
            z = (M - np.nanmean(M)) / np.nanstd(M)
            store.append(np.nanmean(z, axis=1))
            rr = np.repeat(np.arange(8), 8)
            rs.append(np.corrcoef(rr[g], v[g])[0, 1])

    return (np.array(old_rows), np.array(new_rows),
            float(np.median(r_old)), float(np.median(r_new)))


def main() -> None:
    print("solving plate for the spatial panels ...")
    s = solve_plate(REPO / "data" / "opdx" / f"{PLATE}.OPDx")
    coef, icept = fit_background_plane(s.plate)
    plane = plane_surface(s.plate, coef, icept)

    print("pooling row profiles over all plates ...")
    old_rows, new_rows, med_r_old, med_r_new = row_profiles()
    print(f"  median corr(row, label): previous {med_r_old:+.3f}, "
          f"rebuilt {med_r_new:+.3f}  (n={len(old_rows)} plates)")

    fig = plt.figure(figsize=(16.0, 9.4))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.95],
                  hspace=0.40, wspace=0.30,
                  left=0.055, right=0.955, top=0.855, bottom=0.075)

    # ---------------- top: where the signal went ----------------------------
    # Panels 1-3 share one scale, which is what makes the tilt's share visible
    # rather than merely asserted. Panel 4 repeats the remainder on its own
    # scale, because on the shared scale it is almost invisible -- true, but it
    # would undersell that the polymer is cleanly resolvable once detrended.
    vmin = float(min(s.plate.z.min(), plane.min(), s.resid.min()))
    vmax = float(max(s.plate.z.max(), plane.max(), s.resid.max()))
    frac = np.ptp(plane) / np.ptp(s.plate.z)
    panels = [
        (s.plate.z, "What the profilometer records",
         f"raw z  ·  spans {np.ptp(s.plate.z):.1f} µm", True),
        (plane, "The slide is sitting crooked",
         f"fitted substrate plane  ·  {np.ptp(plane):.1f} µm "
         f"= {frac*100:.0f}% of it", True),
        (s.resid, "What is actually polymer",
         f"raw − plane  ·  spans {np.ptp(s.resid):.1f} µm", True),
        (s.resid, "…the same, on its own scale",
         "all 64 samples resolve cleanly", False),
    ]
    shared_axes = []
    for i, (img, title, sub, shared) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        kw = dict(vmin=vmin, vmax=vmax) if shared else {}
        im = ax.imshow(img, cmap=SEQ, origin="lower",
                       extent=s.plate.extent, aspect="equal", **kw)
        ax.set_title(title, fontsize=10.5, color=INK, pad=17,
                     fontweight="medium")
        ax.text(0.5, 1.018, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.2, color=INK_2)
        ax.set_xticks([0, 1000, 2000])
        ax.set_yticks([0, 1000, 2000])
        ax.tick_params(length=2, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if i == 0:
            ax.set_ylabel("y (µm)", fontsize=8.5)
        ax.set_xlabel("x (µm)", fontsize=8.5)
        if shared:
            shared_axes.append(ax)
            im_shared = im
        else:
            cb2 = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.03)
            cb2.ax.tick_params(labelsize=7, length=2)
            cb2.outline.set_visible(False)

    cb = fig.colorbar(im_shared, ax=shared_axes, fraction=0.016, pad=0.015)
    cb.set_label("height above the lowest point (µm)  ·  panels 1–3 share this scale",
                 fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.outline.set_visible(False)

    fig.text(0.5, 0.935,
             f"{frac*100:.0f}% of the recorded height range is the slide being "
             f"crooked, not the samples",
             ha="center", fontsize=10.5, color=INK_2)

    # ---------------- bottom left: what the label tracks --------------------
    ax = fig.add_subplot(gs[1, 0:2])
    rows = np.arange(8)
    for data, colour, name, r in (
        (old_rows, PREV, "previous labels", med_r_old),
        (new_rows, NEW, "rebuilt labels", med_r_new),
    ):
        m = np.nanmean(data, axis=0)
        sd = np.nanstd(data, axis=0)
        ax.fill_between(rows, m - sd, m + sd, color=colour, alpha=0.13, lw=0)
        ax.plot(rows, m, color=colour, lw=2.0, marker="o", ms=5.5,
                mec=SURFACE, mew=1.4, label=f"{name}  (r = {r:+.2f})",
                clip_on=False, zorder=3)
    ax.axhline(0, color=GRID, lw=1.0, zorder=0)
    ax.set_xlabel("row index on the plate", fontsize=8.5)
    ax.set_ylabel("label, standardised (SD)", fontsize=8.5)
    ax.set_title("What the label actually tracks", fontsize=10.5, color=INK,
                 pad=17, fontweight="medium")
    ax.text(0.5, 1.018,
            f"row means pooled over {len(old_rows)} plates · band = ±1 SD",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    leg = ax.legend(fontsize=8, frameon=False, loc="upper left",
                    handlelength=1.6, labelcolor=INK_2)
    leg.set_zorder(5)
    ax.set_xticks(rows)
    ax.tick_params(length=2, labelsize=7.5)

    # ---------------- bottom right: the answer ------------------------------
    ax = fig.add_subplot(gs[1, 2:4])
    groups = ["Row/column position only\n(no photo at all)",
              "Photo model, DINOv2\n(8 measured cells per plate)"]
    prev_vals = [0.953, 0.684]
    new_vals = [0.406, -0.045]
    x = np.array([0.0, 0.9])
    w = 0.26
    gap = 0.012          # 2px-equivalent surface gap between adjacent bars

    b1 = ax.bar(x - w / 2 - gap, prev_vals, w, color=PREV,
                label="previous labels (raw z)", zorder=3)
    b2 = ax.bar(x + w / 2 + gap, new_vals, w, color=NEW,
                label="rebuilt labels (plane-relative)", zorder=3)

    for bars, vals in ((b1, prev_vals), (b2, new_vals)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.035 if v >= 0 else -0.055),
                    f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=9, color=INK, fontweight="medium", zorder=4)

    ax.axhline(0, color=INK_3, lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8.6, color=INK_2)
    ax.set_ylabel("R²  (1 = perfect, 0 = no better than the average)",
                  fontsize=8.5)
    ax.set_xlim(-0.42, 1.32)
    ax.set_ylim(-0.22, 1.20)
    ax.set_title("Can a photo predict absolute height?", fontsize=10.5,
                 color=INK, pad=17, fontweight="medium")
    ax.text(0.5, 1.018, "leave-one-plate-out over 69 plates",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              handlelength=1.4, labelcolor=INK_2)
    ax.tick_params(length=2, labelsize=7.5)

    ax.annotate("position beats the model:\nit never learned thickness",
                xy=(x[0] - w / 2 - gap, 0.953), xytext=(0.16, 1.17),
                fontsize=8.4, color=INK_2, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.9,
                                shrinkA=0, shrinkB=3))

    fig.suptitle(
        "The labels were measuring plate position, not polymer thickness",
        fontsize=13.5, color=INK, fontweight="semibold", y=0.975)

    out = STUDY / "figures" / "summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
