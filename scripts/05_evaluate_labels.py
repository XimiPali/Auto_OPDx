"""Does fixing the labels reduce the number of control points needed?

That is the question the whole task turns on. The previous work found it needed
~8 measured cells per plate to get usable absolute heights -- 12.5% of every
plate still has to go under the stylus, which defeats the purpose.

The hypothesis is that the 8 was inflated by label error, not by an intrinsic
limit: raw ``z_max`` labels carry the substrate tilt, so they encode position as
much as thickness.

Protocol (mirrors the results deck)
-----------------------------------
Leave-one-plate-out. Train on all plates but one, test on the held-out plate --
the deployment case, where every new batch is unseen.

Features are centred per plate, which needs no labels and so is available at
deployment; this is the deck's Phase 1 fix for the Simpson's-paradox confound.
The model predicts a standardised deviation, and ``K`` measured cells from the
held-out plate supply the mean and spread needed to convert back to absolute
micrometres. Evaluation excludes the K calibration cells.

Everything is held constant between the two label definitions except the labels
themselves, so any difference is attributable to label quality.

Features: by default the deck's *Phase 1* brightness baseline (7 intensity
statistics). Pass ``--dinov2`` to use DINOv2 features instead, which is the
deck's Phase 2. Either way the extractor is held fixed across the two label
definitions, so the comparison is valid; absolute R^2 is only comparable to the
deck's own figures when the same backbone and dimensionality are used.

Usage::

    uv run python scripts/03_build_dataset.py --save-crops
    uv run python scripts/05_evaluate_labels.py
    uv run python scripts/04_dinov2_features.py
    uv run python scripts/05_evaluate_labels.py --dinov2 data/processed/dinov2.npz --alpha -1
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

# NumPy 2.x on Apple Accelerate raises spurious divide/overflow/invalid warnings
# from matmul: SIMD padding lanes set the FP status flags even though the result
# is exact. Verified on this data -- the Gram matrix is fully finite and matches
# an einsum reference to within floating-point equality. Suppressed narrowly so
# genuine numerical problems elsewhere still surface.
warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)

REPO = Path(__file__).resolve().parent.parent

# Alphas for the inner selection. Fitted on training plates only, so the
# held-out plate never influences the choice.
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

FEATURES = ["mean_r", "mean_g", "mean_b",
            "gray_mean", "gray_max", "gray_p95", "gray_std"]
K_VALUES = (0, 1, 2, 4, 8, 16)
N_REPEATS = 25
RNG = np.random.default_rng(0)


def load_cells(path: Path, max_overhang: float) -> dict[str, np.ndarray]:
    import csv

    rows = list(csv.DictReader(path.open()))
    if not rows:
        sys.exit(f"{path} is empty")

    def num(key: str) -> np.ndarray:
        return np.array([float(r[key]) if r[key] not in ("", "nan") else np.nan
                         for r in rows])

    keep = np.ones(len(rows), bool)
    keep &= num("overhang") <= max_overhang
    for k in ("label_new", "label_old", *FEATURES):
        keep &= np.isfinite(num(k))

    out = {
        "plate": np.array([r["plate"] for r in rows])[keep],
        "row": num("row")[keep].astype(int),
        "col": num("col")[keep].astype(int),
        "label_new": num("label_new")[keep],
        "label_old": num("label_old")[keep],
        "X": np.column_stack([num(f)[keep] for f in FEATURES]),
    }
    print(f"loaded {keep.sum()}/{len(rows)} cells "
          f"({len(rows) - keep.sum()} dropped: overhang > {max_overhang} or non-finite)")
    return out


def load_dinov2(npz: Path, cells_csv: Path, max_overhang: float) -> dict[str, np.ndarray]:
    """Load DINOv2 features, joining overhang from the cell table by name."""
    import csv

    d = np.load(npz, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    over = {
        f"{r['plate']}_r{int(float(r['row'])):02d}_c{int(float(r['col'])):02d}":
            float(r["overhang"])
        for r in csv.DictReader(cells_csv.open())
    }

    plate, row, col, keep = [], [], [], []
    for n in names:
        stem, r_tok, c_tok = n.rsplit("_", 2)
        plate.append(stem)
        row.append(int(r_tok[1:]))
        col.append(int(c_tok[1:]))
        keep.append(over.get(n, 1.0) <= max_overhang)

    keep = np.array(keep, bool)
    keep &= np.isfinite(d["label_new"]) & np.isfinite(d["label_old"])
    keep &= np.isfinite(d["X"]).all(axis=1)

    out = {
        "plate": np.array(plate)[keep],
        "row": np.array(row)[keep],
        "col": np.array(col)[keep],
        "label_new": d["label_new"][keep].astype(float),
        "label_old": d["label_old"][keep].astype(float),
        "X": d["X"][keep].astype(float),
    }
    print(f"loaded {keep.sum()}/{len(names)} cells with "
          f"{out['X'].shape[1]} DINOv2 features")
    return out


def r2(y: np.ndarray, p: np.ndarray) -> float:
    denom = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - p) ** 2).sum() / denom) if denom > 0 else float("nan")


def position_baseline(d: dict, label: str) -> tuple[float, float]:
    """R^2 obtainable from (row, col) alone, per plate. No image content."""
    scores = []
    for pl in np.unique(d["plate"]):
        m = d["plate"] == pl
        y = d[label][m]
        if len(y) < 6 or y.std() == 0:
            continue
        A = np.c_[d["row"][m], d["col"][m], np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        scores.append(r2(y, A @ coef))
    s = np.array(scores)
    return float(np.median(s)), float(s.mean())


def lopo(d: dict, label: str, alpha: float = 1.0) -> dict[int, dict[str, float]]:
    """Leave-one-plate-out sweep over calibration budget K."""
    plates = np.unique(d["plate"])
    y_all = d[label]
    out: dict[int, dict[str, float]] = {}

    for K in K_VALUES:
        r2s: list[float] = []
        rhos: list[float] = []
        for held in plates:
            te = d["plate"] == held
            tr = ~te
            if te.sum() < K + 4 or tr.sum() < 20:
                continue

            # --- train: centre features and standardise labels per plate ---
            Xtr, ytr, gtr = [], [], []
            for pl in plates[plates != held]:
                m = d["plate"] == pl
                if m.sum() < 4:
                    continue
                yp = y_all[m]
                s = yp.std()
                if s == 0:
                    continue
                Xtr.append(d["X"][m] - d["X"][m].mean(0))
                ytr.append((yp - yp.mean()) / s)
                gtr.append(np.full(int(m.sum()), pl))
            if not Xtr:
                continue
            Xf = np.ascontiguousarray(np.vstack(Xtr), dtype=np.float64)
            yf = np.ascontiguousarray(np.concatenate(ytr), dtype=np.float64)
            gf = np.concatenate(gtr)
            # Standardise per dimension using TRAINING statistics only.
            # Required for high-dimensional features: unscaled 384-dim DINOv2
            # columns leave the ridge system badly conditioned.
            scale = Xf.std(axis=0)
            scale[scale < 1e-12] = 1.0
            Xf = Xf / scale

            if alpha < 0:
                # Grouped split over TRAINING plates only, so the held-out
                # plate never influences the choice of alpha. Done manually
                # with plain Ridge: RidgeCV's GCV path trips a spurious
                # matmul RuntimeWarning in Accelerate BLAS.
                a_best, s_best = ALPHAS[0], -np.inf
                uq = np.unique(gf)
                inner_te = np.isin(gf, uq[::3])
                if inner_te.any() and (~inner_te).any():
                    for a in ALPHAS:
                        inner = Ridge(alpha=a).fit(Xf[~inner_te], yf[~inner_te])
                        sc = r2(yf[inner_te], inner.predict(Xf[inner_te]))
                        if np.isfinite(sc) and sc > s_best:
                            a_best, s_best = a, sc
                model = Ridge(alpha=a_best).fit(Xf, yf)
            else:
                model = Ridge(alpha=alpha).fit(Xf, yf)
            sigma_prior = float(np.median([
                y_all[d["plate"] == pl].std() for pl in plates[plates != held]
            ]))
            mu_prior = float(np.median([
                y_all[d["plate"] == pl].mean() for pl in plates[plates != held]
            ]))

            # --- test: centre by the held-out plate's own feature mean ---
            # No labels are used here, so this is available at deployment.
            Xte = (d["X"][te] - d["X"][te].mean(0)) / scale
            yte = y_all[te]
            z_hat = model.predict(Xte)
            idx = np.arange(te.sum())

            reps = 1 if K == 0 else N_REPEATS
            rep_r2, rep_rho = [], []
            for _ in range(reps):
                if K == 0:
                    cal = np.array([], int)
                    mu, sig = mu_prior, sigma_prior
                else:
                    cal = RNG.choice(idx, size=K, replace=False)
                    mu = float(yte[cal].mean())
                    # A 1-cell std is undefined and a 2-cell std is very noisy,
                    # so fall back to the training prior at small K.
                    sig = float(yte[cal].std(ddof=1)) if K >= 3 else sigma_prior
                    if not np.isfinite(sig) or sig <= 0:
                        sig = sigma_prior
                ev = np.setdiff1d(idx, cal)
                if len(ev) < 4:
                    continue
                pred = mu + sig * z_hat[ev]
                rep_r2.append(r2(yte[ev], pred))
                rho = spearmanr(yte[ev], pred).statistic
                if np.isfinite(rho):
                    rep_rho.append(float(rho))
            if rep_r2:
                r2s.append(float(np.mean(rep_r2)))
            if rep_rho:
                rhos.append(float(np.mean(rep_rho)))

        a = np.array(r2s)
        b = np.array(rhos)
        out[K] = {
            "median_r2": float(np.median(a)) if len(a) else float("nan"),
            "mean_r2": float(a.mean()) if len(a) else float("nan"),
            "positive": int((a > 0).sum()),
            "n_plates": len(a),
            "spearman": float(np.median(b)) if len(b) else float("nan"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path,
                    default=REPO / "data" / "processed" / "cells.csv")
    ap.add_argument("--max-overhang", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="negative selects alpha by CV on training plates")
    ap.add_argument("--dinov2", type=Path, default=None,
                    help="path to dinov2.npz; uses those features instead "
                         "of the brightness statistics")
    args = ap.parse_args()

    if not args.cells.exists():
        sys.exit(f"{args.cells} not found -- run scripts/03_build_dataset.py first")

    if args.dinov2:
        if not args.dinov2.exists():
            sys.exit(f"{args.dinov2} not found -- run "
                     "scripts/04_dinov2_features.py first")
        d = load_dinov2(args.dinov2, args.cells, args.max_overhang)
        feat_name = f"DINOv2 ({d['X'].shape[1]}-dim)"
    else:
        d = load_cells(args.cells, args.max_overhang)
        feat_name = "brightness statistics (deck Phase 1)"
    n_plates = len(np.unique(d["plate"]))
    print(f"plates: {n_plates}   features: {feat_name}\n")

    print("=" * 78)
    print("SANITY CHECK: R^2 FROM (row, col) POSITION ALONE, NO IMAGE CONTENT")
    print("=" * 78)
    for label, name in (("label_old", "raw z_max (previous)"),
                        ("label_new", "plane-relative (rebuilt)")):
        med, mean = position_baseline(d, label)
        print(f"  {name:28} median {med:6.3f}   mean {mean:6.3f}")
    print("\n  A model must beat these to be measuring anything real.")

    results = {}
    for label, name in (("label_old", "PREVIOUS LABELS (raw z_max)"),
                        ("label_new", "REBUILT LABELS (plane-relative)")):
        results[label] = lopo(d, label, alpha=args.alpha)
        print("\n" + "=" * 78)
        print(f"{name} -- leave-one-plate-out")
        print("=" * 78)
        print(f"  {'K':>3} {'median R2':>10} {'mean R2':>10} "
              f"{'plates>0':>9} {'Spearman':>9}")
        for K, v in results[label].items():
            print(f"  {K:>3} {v['median_r2']:10.3f} {v['mean_r2']:10.3f} "
                  f"{v['positive']:4d}/{v['n_plates']:<4d} {v['spearman']:9.3f}")

    print("\n" + "=" * 78)
    print("CONTROL POINTS NEEDED  (K to reach a given median R^2)")
    print("=" * 78)
    for target in (0.5, 0.6, 0.7):
        line = f"  R2 >= {target:.1f}:  "
        for label, short in (("label_old", "previous"), ("label_new", "rebuilt")):
            hit = [K for K, v in results[label].items()
                   if np.isfinite(v["median_r2"]) and v["median_r2"] >= target]
            line += f"{short} K={hit[0] if hit else '>16':>3}   "
        print(line)

    print(f"\nFeatures: {feat_name}. The label-vs-label comparison holds the")
    print("feature extractor fixed, so any difference is attributable to the")
    print("labels. Absolute R^2 is not directly comparable to the deck's own")
    print("figures unless the same backbone and dimensionality are used.")


if __name__ == "__main__":
    main()
