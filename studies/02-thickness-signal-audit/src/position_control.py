"""The decisive test: is DINOv2 reading thickness, or reading position?

Hand-crafted brightness and sharpness features reach Spearman ~0.05 against
corrected labels -- noise. DINOv2 reaches ~0.43. Either DINOv2 finds a real
structural cue that fourteen numbers miss, or it is exploiting something that
correlates with thickness without being thickness.

The prime suspect is **position**. Even after detrending, the corrected label is
still ~0.41 R^2 predictable from (row, col), because the formulation layout is
systematic. And illumination varies smoothly across the plate, so a
high-capacity feature extractor can infer where a cell sits. Predicting the
layout is not the same as measuring thickness -- it would not generalise to a
plate with a different layout, and it cannot rank two cells of the same
formulation.

The test: remove each plate's own (row, col) linear trend from the label, then
re-run. What survives is thickness signal that is *independent of where the cell
sits*. That is the quantity the lab actually needs.

Usage::

    uv run python studies/02-thickness-signal-audit/src/position_control.py
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
REPO = AUDIT.parent.parent
sys.path.insert(0, str(HERE))

from evaluate import STATS, WINDOWS, col, load, lopo  # noqa: E402

K_MAIN = 8
MAX_OVERHANG = 0.10


def deposition(y: np.ndarray, plate: np.ndarray, row: np.ndarray,
               col_: np.ndarray) -> np.ndarray:
    """Remove each plate's own (row, col) linear trend from the label."""
    out = y.astype(float).copy()
    for pl in np.unique(plate):
        m = plate == pl
        if m.sum() < 6:
            continue
        A = np.c_[row[m], col_[m], np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        out[m] = y[m] - A @ coef
    return out


def predict_position(X: np.ndarray, plate: np.ndarray,
                     target: np.ndarray) -> float:
    """How well do the features recover a cell's grid index, out of sample?"""
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr

    plates = np.unique(plate)
    scores = []
    for held in plates:
        te = plate == held
        tr = ~te
        if te.sum() < 8 or tr.sum() < 100:
            continue
        Xtr = X[tr] - X[tr].mean(0)
        sc = Xtr.std(0)
        sc[sc < 1e-12] = 1.0
        m = Ridge(alpha=10.0).fit(Xtr / sc, target[tr] - target[tr].mean())
        pred = m.predict((X[te] - X[te].mean(0)) / sc)
        rho = spearmanr(target[te], pred).statistic
        if np.isfinite(rho):
            scores.append(float(rho))
    return float(np.median(scores)) if scores else float("nan")


def main() -> None:
    rows = load(AUDIT / "results" / "cells_audit.csv")
    plate = np.array([r["plate"] for r in rows])
    row_i = col(rows, "row")
    col_i = col(rows, "col")
    over = col(rows, "overhang")
    y_raw = col(rows, "h_max")

    names = np.array([
        f"{r['plate']}_r{int(float(r['row'])):02d}_c{int(float(r['col'])):02d}"
        for r in rows
    ])

    # --- DINOv2 features, joined to this table by cell name -----------------
    npz = REPO / "data" / "processed" / "dinov2_b.npz"
    if not npz.exists():
        sys.exit(f"{npz} not found -- run scripts/04_dinov2_features.py first")
    d = np.load(npz, allow_pickle=False)
    dmap = {str(n): i for i, n in enumerate(d["names"])}
    idx = np.array([dmap.get(n, -1) for n in names])
    has_dino = idx >= 0
    X_dino = np.full((len(rows), d["X"].shape[1]), np.nan)
    X_dino[has_dino] = d["X"][idx[has_dino]]

    bright = [f"w{WINDOWS[0]}_{s}" for s in STATS]
    X_bright = np.column_stack([col(rows, c) for c in bright])

    keep = (np.isfinite(over) & (over <= MAX_OVERHANG) & np.isfinite(y_raw)
            & np.isfinite(X_bright).all(1) & np.isfinite(X_dino).all(1))
    print(f"cells usable in both feature sets: {keep.sum()} "
          f"across {len(set(plate[keep]))} plates\n")

    P, R, C = plate[keep], row_i[keep], col_i[keep]
    Xb, Xd = X_bright[keep], X_dino[keep]
    y = y_raw[keep]
    y_res = deposition(y, P, R, C)

    var_removed = 1.0 - y_res.var() / y.var()
    print(f"the (row, col) trend accounts for {var_removed*100:.0f}% of the "
          f"corrected label's variance\n")

    # --- can the features tell where a cell is? ---------------------------
    print("=" * 78)
    print("STEP 1 — can the features recover the cell's POSITION? (out of sample)")
    print("=" * 78)
    print(f"  {'features':26} {'-> row index':>14} {'-> col index':>14}")
    pos_rows = []
    for nm, X in (("brightness (7)", Xb), ("DINOv2 (768)", Xd)):
        rr = predict_position(X, P, R)
        cc = predict_position(X, P, C)
        print(f"  {nm:26} {rr:14.3f} {cc:14.3f}")
        pos_rows.append({"features": nm, "row_rho": rr, "col_rho": cc})

    pos_out = AUDIT / "results" / "position_recovery.csv"
    with pos_out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["features", "row_rho", "col_rho"])
        wr.writeheader()
        wr.writerows(pos_rows)

    # --- does the signal survive removing position? -----------------------
    print("\n" + "=" * 78)
    print(f"STEP 2 — prediction before and after removing position (K={K_MAIN})")
    print("=" * 78)
    print(f"  {'features':26} {'label':24} {'median R2':>10} {'Spearman':>9}")
    out_rows = []
    for nm, X in (("brightness (7)", Xb), ("DINOv2 (768)", Xd)):
        for lab_nm, yv in (("corrected", y),
                           ("corrected, position removed", y_res)):
            res = lopo(X, yv, P, (K_MAIN,))[K_MAIN]
            print(f"  {nm:26} {lab_nm:24} {res['median_r2']:10.3f} "
                  f"{res['spearman']:9.3f}")
            out_rows.append({"features": nm, "label": lab_nm,
                             "K": K_MAIN, **res})

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    dino_before = next(r["spearman"] for r in out_rows
                       if r["features"].startswith("DINOv2")
                       and r["label"] == "corrected")
    dino_after = next(r["spearman"] for r in out_rows
                      if r["features"].startswith("DINOv2")
                      and r["label"].endswith("removed"))
    print(f"  DINOv2 Spearman: {dino_before:.3f} -> {dino_after:.3f} once the")
    print("  plate's own row/column trend is taken out of the label.")
    if dino_after < 0.15:
        print("\n  >> Most of DINOv2's apparent skill was the LAYOUT, not thickness.")
        print("     It was predicting where a cell sits, which the formulation")
        print("     layout makes correlate with height. That does not generalise")
        print("     to a new layout and cannot rank two cells of the same")
        print("     formulation -- which is what the lab actually needs.")
    else:
        print("\n  >> A substantial part survives: there IS position-independent")
        print("     thickness signal in the photo, and it is structural rather")
        print("     than intensity-based (brightness alone does not find it).")

    # csv.writer, not manual joins: one of the label strings contains a comma
    # ("corrected, position removed"), which silently shifted every column.
    keys = list(out_rows[0].keys())
    out = AUDIT / "results" / "position_control.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
