"""Joint 8x8 lattice fitting, shared by both modalities.

Why fit a lattice instead of trusting per-feature centroids
-----------------------------------------------------------
Height map features are ~98 um square but the coarse axis samples every ~30 um,
so a feature spans only 2-3 samples there. A centroid from 3 samples cannot beat
roughly +/-15 um, which is ~5% of the 312 um cell pitch.

The grid is not organic: it is written by a DMD through a 4x objective onto a
piezostage, so its pitch is deterministic by construction. That justifies
fitting one rigid model to all 64 features simultaneously -- 10 parameters
instead of 128 free coordinates -- which averages the per-feature quantisation
noise down by ~sqrt(64).

The model, per output axis:

    pos = a*r + b*c + d*[r >= rows/2] + e*[c >= cols/2] + f

Five parameters per axis, ten in total. This spans translation, independent
scale on each grid axis, rotation and shear (via the cross terms), and the
quadrant gaps visible between rows/columns 3 and 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

N_PARAMS = 5


def _design(rc: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Design matrix for grid indices ``rc`` of shape (n, 2) as (row, col)."""
    rc = np.asarray(rc, float)
    r, c = rc[:, 0], rc[:, 1]
    return np.column_stack([
        r,
        c,
        (r >= rows / 2.0).astype(float),   # quadrant gap in the row direction
        (c >= cols / 2.0).astype(float),   # quadrant gap in the column direction
        np.ones(len(rc)),
    ])


@dataclass
class LatticeFit:
    """A fitted lattice plus the diagnostics that make it auditable."""

    coef: np.ndarray          # (5, 2) -> columns are x and y
    rows: int
    cols: int
    rc: np.ndarray            # (n, 2) grid indices used in the fit
    observed: np.ndarray      # (n, 2) detected positions
    residuals: np.ndarray     # (n,) per-point residual magnitude
    inliers: np.ndarray       # (n,) bool mask of points kept
    n_detected: int
    n_expected: int

    def predict(self, rc: np.ndarray) -> np.ndarray:
        """Predicted (x, y) positions for grid indices ``rc``."""
        return _design(np.atleast_2d(rc), self.rows, self.cols) @ self.coef

    def all_centres(self) -> np.ndarray:
        """Predicted centres for every slot, shape ``(rows*cols, 2)``.

        Ordered row-major: index ``r * cols + c``.
        """
        rc = np.array([(r, c) for r in range(self.rows) for c in range(self.cols)])
        return self.predict(rc)

    @property
    def rmse(self) -> float:
        """Root-mean-square residual over inliers, in input units."""
        m = self.inliers
        return float(np.sqrt(np.mean(self.residuals[m] ** 2))) if m.any() else float("nan")

    @property
    def max_residual(self) -> float:
        m = self.inliers
        return float(self.residuals[m].max()) if m.any() else float("nan")

    def coverage(self, centres: np.ndarray, tol_frac: float = 0.25) -> float:
        """Fraction of predicted slots that have a detection close to them.

        Residual RMSE alone is *not* sufficient to validate a fit. This model can
        place a phantom row inside the quadrant gap and drop a real row at the
        edge: seven of eight rows then fit beautifully, RMSE stays low, and the
        leftover real row is discarded as an outlier. Selecting on RMSE actively
        prefers such a fit.

        Coverage catches it, because the phantom row has nothing near it.
        """
        centres = np.atleast_2d(np.asarray(centres, float))
        if len(centres) == 0:
            return 0.0
        pitch = float(np.mean(self.pitch))
        if not np.isfinite(pitch) or pitch <= 0:
            return 0.0
        pred = self.all_centres()
        d = np.linalg.norm(pred[:, None, :] - centres[None, :, :], axis=2).min(axis=1)
        return float((d <= tol_frac * pitch).mean())

    def quality(self, centres: np.ndarray, miss_weight: float = 1.0) -> float:
        """Combined selection score; lower is better.

        Residual as a fraction of pitch, plus a penalty for uncovered slots.
        The penalty dominates a degenerate fit (one phantom row costs ~0.125)
        while a good fit's residual is ~0.01-0.03, so the correct arrangement
        wins even though its raw RMSE may be marginally higher.
        """
        pitch = float(np.mean(self.pitch))
        if not np.isfinite(pitch) or pitch <= 0 or not np.isfinite(self.rmse):
            return float("inf")
        return self.rmse / pitch + miss_weight * (1.0 - self.coverage(centres))

    @property
    def gap_offsets(self) -> tuple[float, float]:
        """Magnitude of the fitted quadrant offsets, in input units.

        Physically these are *gaps*, so a well-formed fit has them positive
        along the corresponding grid direction. A negative value means the model
        compressed the middle of the grid instead, which is the phantom-row
        failure described in :meth:`coverage`.
        """
        return float(np.hypot(*self.coef[2])), float(np.hypot(*self.coef[3]))

    @property
    def pitch(self) -> tuple[float, float]:
        """(column pitch, row pitch) magnitudes, in input units."""
        d_col = np.hypot(*self.coef[1])   # step per unit column
        d_row = np.hypot(*self.coef[0])   # step per unit row
        return float(d_col), float(d_row)

    @property
    def rotation_deg(self) -> float:
        """Rotation of the grid's column axis away from horizontal."""
        return float(np.degrees(np.arctan2(self.coef[1, 1], self.coef[1, 0])))

    def summary(self) -> str:
        pc, pr = self.pitch
        return (
            f"lattice {self.rows}x{self.cols}: "
            f"{self.inliers.sum()}/{self.n_detected} inliers "
            f"(expected {self.n_expected}), rmse {self.rmse:.2f}, "
            f"max {self.max_residual:.2f}, pitch {pc:.1f}/{pr:.1f}, "
            f"rot {self.rotation_deg:+.2f} deg"
        )


def detect_features(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    min_area_px: int = 8,
    max_features: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate raised features and return intensity-weighted centroids.

    Weighted rather than binary centroids matter here: with only 2-3 samples
    across a feature on the coarse axis, a binary centroid quantises hard to
    sample boundaries, whereas weighting by height recovers some sub-sample
    precision.

    Parameters
    ----------
    field
        2-D array, typically the detrended height map. Shape ``(len(y), len(x))``.
    x, y
        1-D physical axes, um.
    threshold
        Values at or above this count as feature.
    min_area_px
        Reject blobs smaller than this.
    max_features
        Keep only the this-many largest blobs, if given.

    Returns
    -------
    centres : ndarray, shape (n, 2)
        Weighted centroids as (x, y) in physical units.
    areas : ndarray, shape (n,)
        Blob areas in pixels.
    """
    field = np.asarray(field, float)
    if field.shape != (y.size, x.size):
        raise ValueError(f"field {field.shape} != {(y.size, x.size)}")

    mask = np.nan_to_num(field, nan=-np.inf) >= threshold
    n_lab, labels = cv2.connectedComponents(
        mask.astype(np.uint8) * 255, connectivity=4
    )
    if n_lab <= 1:
        return np.empty((0, 2)), np.empty((0,), int)

    weight = np.clip(np.nan_to_num(field, nan=0.0) - threshold, 0.0, None)
    xm, ym = np.meshgrid(x, y)

    # Accumulate every blob's moments in a single pass with bincount. Masking
    # per label (`labels == lab`) instead costs one full-array scan per blob,
    # which on a 2048x1536 photo with a few hundred blobs dominated runtime --
    # ~90 s per image versus well under a second here.
    flat = labels.ravel()
    w_flat = weight.ravel()
    area = np.bincount(flat, minlength=n_lab)
    wsum = np.bincount(flat, weights=w_flat, minlength=n_lab)
    wx = np.bincount(flat, weights=(xm.ravel() * w_flat), minlength=n_lab)
    wy = np.bincount(flat, weights=(ym.ravel() * w_flat), minlength=n_lab)
    # Unweighted fallback for blobs whose weights sum to zero.
    ux = np.bincount(flat, weights=xm.ravel(), minlength=n_lab)
    uy = np.bincount(flat, weights=ym.ravel(), minlength=n_lab)

    ids = np.arange(1, n_lab)
    ids = ids[area[ids] >= min_area_px]
    if len(ids) == 0:
        return np.empty((0, 2)), np.empty((0,), int)

    good = wsum[ids] > 0
    cx = np.where(good, wx[ids] / np.where(good, wsum[ids], 1.0), ux[ids] / area[ids])
    cy = np.where(good, wy[ids] / np.where(good, wsum[ids], 1.0), uy[ids] / area[ids])

    centres = np.column_stack([cx, cy]).astype(float)
    areas = area[ids].astype(int)
    if max_features is not None and len(areas) > max_features:
        keep = np.argsort(areas)[::-1][:max_features]
        centres, areas = centres[keep], areas[keep]
    return centres, areas


def _slots_from_bbox(centres: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Evenly spaced slots spanning the detected bounding box.

    Cheap, but fragile: one spurious blob outside the grid stretches the box and
    shifts every slot, which can push the assignment into a wrong permutation.
    """
    if len(centres) == 0:
        gx = np.linspace(0.0, 1.0, cols)
        gy = np.linspace(0.0, 1.0, rows)
    else:
        gx = np.linspace(centres[:, 0].min(), centres[:, 0].max(), cols)
        gy = np.linspace(centres[:, 1].min(), centres[:, 1].max(), rows)
    return np.array([(gx[c], gy[r]) for r in range(rows) for c in range(cols)])


def _cluster_1d(values: np.ndarray, k: int) -> np.ndarray:
    """Sorted centres of ``k`` clusters in a 1-D set of coordinates.

    The grid sits within a few degrees of axis aligned, so projecting onto each
    axis yields ``k`` tight groups. Clustering those recovers the true line
    positions -- including the quadrant gaps -- whereas an evenly spaced guess
    ignores them. Robust to missing features, which merely thin a cluster.
    """
    v = np.sort(np.asarray(values, float))
    if len(v) <= k:
        return np.pad(v, (0, k - len(v)), mode="edge") if len(v) else np.zeros(k)

    # Seed on quantiles, then a few Lloyd iterations. Avoids a sklearn
    # dependency here and is trivially fast at this size.
    centres = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(25):
        idx = np.argmin(np.abs(v[:, None] - centres[None, :]), axis=1)
        new = centres.copy()
        for j in range(k):
            sel = v[idx == j]
            if len(sel):
                new[j] = sel.mean()
        if np.allclose(new, centres):
            break
        centres = new
    return np.sort(centres)


def _slots_from_clusters(centres: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Slots from independently clustered x and y coordinates."""
    if len(centres) == 0:
        return _slots_from_bbox(centres, rows, cols)
    gx = _cluster_1d(centres[:, 0], cols)
    gy = _cluster_1d(centres[:, 1], rows)
    return np.array([(gx[c], gy[r]) for r in range(rows) for c in range(cols)])


def assign_to_grid(
    centres: np.ndarray, slots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one matching of detections to slots.

    Uses the Hungarian algorithm, so this is globally optimal rather than greedy
    nearest-neighbour. Handles a rectangular cost matrix, i.e. missing features
    (torn samples) or spurious blobs.

    Returns
    -------
    det_idx, slot_idx : ndarray
        Paired indices into ``centres`` and ``slots``.
    """
    if len(centres) == 0 or len(slots) == 0:
        return np.empty(0, int), np.empty(0, int)
    cost = np.linalg.norm(centres[:, None, :] - slots[None, :, :], axis=2)
    return linear_sum_assignment(cost)


def fit_lattice(
    rc: np.ndarray,
    observed: np.ndarray,
    rows: int,
    cols: int,
    robust_sigma: float = 3.5,
    n_detected: int | None = None,
) -> LatticeFit:
    """Least-squares lattice fit with one robust re-weighting pass.

    Outliers are identified by MAD-scaled residual and dropped, then the model
    is refit on the survivors.
    """
    rc = np.atleast_2d(np.asarray(rc, float))
    observed = np.atleast_2d(np.asarray(observed, float))
    if len(rc) != len(observed):
        raise ValueError("rc and observed must have equal length")
    if len(rc) < N_PARAMS:
        raise ValueError(f"need >= {N_PARAMS} points to fit, got {len(rc)}")

    A = _design(rc, rows, cols)
    coef, *_ = np.linalg.lstsq(A, observed, rcond=None)
    resid = np.linalg.norm(observed - A @ coef, axis=1)

    # Robust pass: MAD is insensitive to the handful of torn/displaced samples.
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = 1.4826 * mad
    if scale > 0:
        inliers = resid <= med + robust_sigma * scale
    else:
        inliers = np.ones(len(resid), bool)
    if inliers.sum() >= N_PARAMS and not inliers.all():
        coef, *_ = np.linalg.lstsq(A[inliers], observed[inliers], rcond=None)
        resid = np.linalg.norm(observed - A @ coef, axis=1)

    return LatticeFit(
        coef=coef, rows=rows, cols=cols, rc=rc, observed=observed,
        residuals=resid, inliers=inliers,
        n_detected=len(rc) if n_detected is None else n_detected,
        n_expected=rows * cols,
    )


def _refine_from(
    centres: np.ndarray,
    slots: np.ndarray,
    rows: int,
    cols: int,
    n_iter: int,
    robust_sigma: float,
) -> LatticeFit:
    """Alternate assignment and fitting from a given slot initialisation."""
    fit = None
    prev_pairs = None
    for _ in range(n_iter):
        det_idx, slot_idx = assign_to_grid(centres, slots)
        rc = np.column_stack([slot_idx // cols, slot_idx % cols])
        fit = fit_lattice(
            rc, centres[det_idx], rows, cols,
            robust_sigma=robust_sigma, n_detected=len(centres),
        )
        pairs = tuple(zip(det_idx.tolist(), slot_idx.tolist()))
        if pairs == prev_pairs:
            break
        prev_pairs = pairs
        slots = fit.all_centres()
    assert fit is not None
    return fit


def _prune_and_refit(
    centres: np.ndarray,
    rows: int,
    cols: int,
    n_iter: int,
    robust_sigma: float,
    init,
    prune_frac: float = 0.35,
    max_prune: int = 8,
) -> LatticeFit:
    """Fit, then drop detections that clearly are not grid features, and refit.

    The robust pass inside :func:`fit_lattice` down-weights outliers in the
    *fit*, but they still occupy a slot in the *assignment*. When the number of
    detections happens to equal the number of slots, the Hungarian match is
    forced to consume every detection -- so two specks of debris push two real
    features into wrong slots and the whole model shears.

    Dropping the worst detection and refitting breaks that. Once removed, the
    slot it had taken is simply predicted by the model instead, which is exactly
    the desired behaviour for a torn or invisible sample.
    """
    keep = np.ones(len(centres), bool)
    fit = _refine_from(
        centres, init(centres, rows, cols), rows, cols, n_iter, robust_sigma
    )
    for _ in range(max_prune):
        pitch = float(np.mean(fit.pitch))
        if not np.isfinite(pitch) or pitch <= 0:
            break
        if fit.max_residual <= prune_frac * pitch:
            break
        if keep.sum() <= max(N_PARAMS + 2, rows * cols // 2):
            break
        # Identify the offending detection by position, since fit.observed is
        # ordered by assignment rather than by input index.
        worst = fit.observed[np.argmax(fit.residuals)]
        hit = np.argmin(np.linalg.norm(centres - worst, axis=1) + (~keep) * 1e12)
        keep[hit] = False
        try:
            fit = _refine_from(
                centres[keep], init(centres[keep], rows, cols),
                rows, cols, n_iter, robust_sigma,
            )
        except (ValueError, np.linalg.LinAlgError):
            break
    return fit


def solve_lattice(
    centres: np.ndarray,
    rows: int = 8,
    cols: int = 8,
    n_iter: int = 6,
    robust_sigma: float = 3.5,
) -> LatticeFit:
    """Fit the lattice, choosing the best of several initialisations.

    Assignment and fitting are alternated: the fitted lattice predicts slot
    positions far better than any initial guess, so the labelling converges in
    a few passes. But the *starting* assignment matters -- a poor one can settle
    into a wrong permutation (typically a whole-row shift) that fits badly and
    never recovers, since each iteration reinforces it.

    Two initialisations are therefore tried and the lower-residual result kept:
    evenly spaced slots across the bounding box, and slots from independently
    clustering the x and y coordinates. The latter is much more robust because
    it recovers the real line positions including the quadrant gaps, but the
    former occasionally wins on plates with many missing features.
    """
    centres = np.atleast_2d(np.asarray(centres, float))
    if len(centres) < N_PARAMS:
        raise ValueError(
            f"only {len(centres)} features detected; need >= {N_PARAMS}"
        )

    best: LatticeFit | None = None
    best_q = float("inf")
    for init in (_slots_from_clusters, _slots_from_bbox):
        try:
            fit = _prune_and_refit(
                centres, rows, cols, n_iter, robust_sigma, init
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        # Select on coverage-penalised quality, not raw residual: see
        # LatticeFit.coverage for why RMSE alone picks degenerate fits.
        q = fit.quality(centres)
        if best is None or q < best_q:
            best, best_q = fit, q
    if best is None:
        raise ValueError("no lattice initialisation produced a usable fit")
    return best
