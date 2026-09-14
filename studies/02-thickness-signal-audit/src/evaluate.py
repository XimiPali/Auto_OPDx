"""Leave-one-plate-out evaluation across feature windows and label definitions.

Answers the three Tier-1 questions, holding everything else fixed:

1. **Were the features diluted?** Sweep the feature window from the full cell
   pitch down to the pad's own footprint, plus a data-driven pad mask. If the
   negative result was an artifact of averaging over ~92% bare silicon, the
   smaller windows will win clearly.

2. **Does tighter help or hurt on clean labels?** The results deck found
   tightening crops *hurt* (per-cell R^2 0.77 -> 0.59) and concluded context is
   real signal. But that was measured on tilt-contaminated labels, where
   surrounding context also encodes plate position. On corrected labels the
   conclusion may reverse. The same sweep answers this.

3. **Are the labels too noisy?** ``h_max`` is one pixel of a ~1 um quantity.
   Compare it against percentile and pad-mean variants.

A **ring** feature set is included as a control: statistics from the substrate
immediately around each pad, excluding the pad itself. If the ring predicts as
well as the pad, whatever is being picked up is not the pad's absorption.

Protocol matches the main evaluation: features centred per plate (no labels
needed, so legal at deployment), the model predicts a standardised deviation,
and K measured cells supply the mean and spread. Evaluation excludes those K.

Usage::

    uv run python studies/02-thickness-signal-audit/src/evaluate.py
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
REPO = AUDIT.parent.parent

STATS = ["mean_r", "mean_g", "mean_b",
         "gray_mean", "gray_max", "gray_p95", "gray_std"]
WINDOWS = (267, 200, 150, 110, 80, 60)
SHARP = ["lap_var", "lap_absmean", "grad_mean", "grad_p95", "grad_max",
         "hf_energy", "hf_ratio"]
FEATURE_SETS: dict[str, list[str]] = {
    **{f"window {w} um": [f"w{w}_{s}" for s in STATS] for w in WINDOWS},
    "pad mask (data-driven)": [f"pad_{s}" for s in STATS],
    "ring around pad (control)": [f"ring_{s}" for s in STATS],
    "sharpness / focus": [f"sharp_{s}" for s in SHARP],
    "brightness + sharpness": ([f"w{WINDOWS[0]}_{s}" for s in STATS]
                               + [f"sharp_{s}" for s in SHARP]),
}
LABELS = {
    "h_max (previous choice)": "h_max",
    "h_p95": "h_p95",
    "h_p99": "h_p99",
    "h_maskmean": "h_maskmean",
}
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
K_MAIN = 8
K_SWEEP = (0, 2, 4, 8, 16)
N_REPEATS = 20
MAX_OVERHANG = 0.10


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        sys.exit(f"{path} is empty -- run build_features.py first")
    return rows


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([
        float(r[key]) if r.get(key) not in (None, "", "nan") else np.nan
        for r in rows
    ])


def r2(y: np.ndarray, p: np.ndarray) -> float:
    d = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - p) ** 2).sum() / d) if d > 0 else float("nan")


def lopo(
    X: np.ndarray,
    y: np.ndarray,
    plate: np.ndarray,
    ks: tuple[int, ...],
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    rng = np.random.default_rng(seed)
    plates = np.unique(plate)
    out: dict[int, dict[str, float]] = {}

    for K in ks:
        r2s: list[float] = []
        rhos: list[float] = []
        for held in plates:
            te = plate == held
            if te.sum() < K + 4:
                continue

            Xtr, ytr, gtr = [], [], []
            for pl in plates[plates != held]:
                m = plate == pl
                if m.sum() < 4:
                    continue
                yp = y[m]
                s = yp.std()
                if s == 0:
                    continue
                Xtr.append(X[m] - X[m].mean(0))
                ytr.append((yp - yp.mean()) / s)
                gtr.append(np.full(int(m.sum()), pl))
            if len(Xtr) < 5:
                continue

            Xf = np.ascontiguousarray(np.vstack(Xtr), float)
            yf = np.ascontiguousarray(np.concatenate(ytr), float)
            gf = np.concatenate(gtr)
            sc = Xf.std(0)
            sc[sc < 1e-12] = 1.0
            Xf /= sc

            # alpha chosen on a grouped split of the TRAINING plates only
            uq = np.unique(gf)
            inner = np.isin(gf, uq[::3])
            a_best, s_best = ALPHAS[0], -np.inf
            if inner.any() and (~inner).any():
                for a in ALPHAS:
                    mdl = Ridge(alpha=a).fit(Xf[~inner], yf[~inner])
                    sco = r2(yf[inner], mdl.predict(Xf[inner]))
                    if np.isfinite(sco) and sco > s_best:
                        a_best, s_best = a, sco
            model = Ridge(alpha=a_best).fit(Xf, yf)

            mu_p = float(np.median([y[plate == p].mean() for p in plates[plates != held]]))
            sd_p = float(np.median([y[plate == p].std() for p in plates[plates != held]]))

            Xte = (X[te] - X[te].mean(0)) / sc
            yte = y[te]
            zh = model.predict(Xte)
            idx = np.arange(te.sum())

            reps = 1 if K == 0 else N_REPEATS
            rr, rs = [], []
            for _ in range(reps):
                if K == 0:
                    cal, mu, sd = np.array([], int), mu_p, sd_p
                else:
                    cal = rng.choice(idx, size=K, replace=False)
                    mu = float(yte[cal].mean())
                    sd = float(yte[cal].std(ddof=1)) if K >= 3 else sd_p
                    if not np.isfinite(sd) or sd <= 0:
                        sd = sd_p
                ev = np.setdiff1d(idx, cal)
                if len(ev) < 4:
                    continue
                pred = mu + sd * zh[ev]
                rr.append(r2(yte[ev], pred))
                rho = spearmanr(yte[ev], pred).statistic
                if np.isfinite(rho):
                    rs.append(float(rho))
            if rr:
                r2s.append(float(np.mean(rr)))
            if rs:
                rhos.append(float(np.mean(rs)))

        a, b = np.array(r2s), np.array(rhos)
        out[K] = {
            "median_r2": float(np.median(a)) if len(a) else float("nan"),
            "mean_r2": float(a.mean()) if len(a) else float("nan"),
            "positive": int((a > 0).sum()),
            "n": len(a),
            "spearman": float(np.median(b)) if len(b) else float("nan"),
        }
    return out


def label_noise(rows: list[dict], key: str) -> float:
    """Rough label-noise proxy: median |cell - mean of its 4 grid neighbours|.

    Neighbouring cells are different formulations, so this is not pure noise --
    but it is measured identically for every label variant, so a *lower* value
    still means a smoother, less spiky label.
    """
    by_plate: dict[str, dict[tuple[int, int], float]] = {}
    for r in rows:
        v = r.get(key)
        if v in (None, "", "nan"):
            continue
        by_plate.setdefault(r["plate"], {})[(int(float(r["row"])), int(float(r["col"])))] = float(v)
    diffs = []
    for grid in by_plate.values():
        for (rr, cc), v in grid.items():
            nb = [grid[(rr + dr, cc + dc)]
                  for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if (rr + dr, cc + dc) in grid]
            if len(nb) >= 3:
                diffs.append(abs(v - float(np.mean(nb))))
    return float(np.median(diffs)) if diffs else float("nan")


def main() -> None:
    rows = load(AUDIT / "results" / "cells_audit.csv")
    over = col(rows, "overhang")
    keep_base = np.isfinite(over) & (over <= MAX_OVERHANG)
    plate_all = np.array([r["plate"] for r in rows])
    print(f"cells {len(rows)}, plates {len(set(plate_all))}, "
          f"kept after overhang filter: {keep_base.sum()}")

    pf = col(rows, "pad_frac")
    pf = pf[np.isfinite(pf)]
    if len(pf):
        print(f"pad occupies {pf.mean()*100:.1f}% of a full-pitch cell "
              f"-> whole-cell statistics are ~{(1-pf.mean())*100:.0f}% substrate\n")

    out_rows: list[dict] = []

    # ---- Experiments 1 + 2: feature window sweep -------------------------
    print("=" * 84)
    print("EXPERIMENTS 1 & 2 — feature region  (label = h_max, K = 8)")
    print("=" * 84)
    print(f"  {'feature region':28} {'median R2':>10} {'mean R2':>9} "
          f"{'plates>0':>9} {'Spearman':>9}")
    y = col(rows, "h_max")
    best_name, best_rho = None, -np.inf
    for name, cols in FEATURE_SETS.items():
        if not all(c in rows[0] for c in cols):
            continue
        X = np.column_stack([col(rows, c) for c in cols])
        keep = keep_base & np.isfinite(y) & np.isfinite(X).all(1)
        if keep.sum() < 400:
            print(f"  {name:28} (only {keep.sum()} cells)")
            continue
        res = lopo(X[keep], y[keep], plate_all[keep], (K_MAIN,))[K_MAIN]
        print(f"  {name:28} {res['median_r2']:10.3f} {res['mean_r2']:9.3f} "
              f"{res['positive']:4d}/{res['n']:<4d} {res['spearman']:9.3f}")
        out_rows.append({"experiment": "feature_region", "config": name,
                         "label": "h_max", "K": K_MAIN, **res})
        if name != "ring around pad (control)" and res["spearman"] > best_rho:
            best_name, best_rho = name, res["spearman"]

    print(f"\n  best region: {best_name}  (Spearman {best_rho:.3f})")

    # ---- Experiment 3: label definition ----------------------------------
    print("\n" + "=" * 84)
    print(f"EXPERIMENT 3 — label definition  (features = {best_name}, K = 8)")
    print("=" * 84)
    print(f"  {'label':28} {'median R2':>10} {'mean R2':>9} "
          f"{'plates>0':>9} {'Spearman':>9} {'noise':>8}")
    cols = FEATURE_SETS[best_name]
    X_best = np.column_stack([col(rows, c) for c in cols])
    for name, key in LABELS.items():
        yv = col(rows, key)
        keep = keep_base & np.isfinite(yv) & np.isfinite(X_best).all(1)
        if keep.sum() < 400:
            continue
        res = lopo(X_best[keep], yv[keep], plate_all[keep], (K_MAIN,))[K_MAIN]
        print(f"  {name:28} {res['median_r2']:10.3f} {res['mean_r2']:9.3f} "
              f"{res['positive']:4d}/{res['n']:<4d} {res['spearman']:9.3f} "
              f"{label_noise(rows, key):8.3f}")
        out_rows.append({"experiment": "label", "config": name,
                         "label": key, "K": K_MAIN, **res})

    # ---- Reference: the previous configuration ---------------------------
    print("\n" + "=" * 84)
    print("REFERENCE — the previous configuration, for comparison")
    print("=" * 84)
    ref = [
        ("whole cell + raw z label (previous)", FEATURE_SETS["window 267 um"], "h_max_raw"),
        ("whole cell + corrected label", FEATURE_SETS["window 267 um"], "h_max"),
    ]
    for name, cols_, key in ref:
        yv = col(rows, key)
        X = np.column_stack([col(rows, c) for c in cols_])
        keep = keep_base & np.isfinite(yv) & np.isfinite(X).all(1)
        res = lopo(X[keep], yv[keep], plate_all[keep], (K_MAIN,))[K_MAIN]
        print(f"  {name:38} R2 {res['median_r2']:+.3f}   Spearman {res['spearman']:.3f}")
        out_rows.append({"experiment": "reference", "config": name,
                         "label": key, "K": K_MAIN, **res})

    # ---- K sweep on the winning configuration ----------------------------
    best_label = max(
        LABELS.items(),
        key=lambda kv: next(
            (r["spearman"] for r in out_rows
             if r["experiment"] == "label" and r["label"] == kv[1]),
            -np.inf,
        ),
    )
    print("\n" + "=" * 84)
    print(f"K SWEEP — {best_name} + {best_label[0]}")
    print("=" * 84)
    yv = col(rows, best_label[1])
    keep = keep_base & np.isfinite(yv) & np.isfinite(X_best).all(1)
    sweep = lopo(X_best[keep], yv[keep], plate_all[keep], K_SWEEP)
    print(f"  {'K':>3} {'median R2':>10} {'mean R2':>9} {'plates>0':>9} {'Spearman':>9}")
    for K, res in sweep.items():
        print(f"  {K:>3} {res['median_r2']:10.3f} {res['mean_r2']:9.3f} "
              f"{res['positive']:4d}/{res['n']:<4d} {res['spearman']:9.3f}")
        out_rows.append({"experiment": "k_sweep", "config": best_name,
                         "label": best_label[1], "K": K, **res})

    keys = list(out_rows[0].keys())
    out = AUDIT / "results" / "evaluation.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
