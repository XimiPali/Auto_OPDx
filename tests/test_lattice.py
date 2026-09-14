"""Tests for the lattice solver, against synthetic grids with known truth.

These need no data files: a grid is generated with a chosen pitch, rotation and
quadrant gap, optionally corrupted, and the solver must recover it. That makes
the failure modes we actually hit on real plates -- missing features, dust,
quantisation -- directly testable.
"""

from __future__ import annotations

import numpy as np
import pytest

from Auto_OPDx.alignment.lattice import (
    assign_to_grid,
    detect_features,
    fit_lattice,
    solve_lattice,
)

ROWS = COLS = 8


def make_grid(
    pitch_x: float = 250.0,
    pitch_y: float = 180.0,
    gap_x: float = 0.0,
    gap_y: float = 60.0,
    rotation_deg: float = 0.0,
    origin: tuple[float, float] = (100.0, 90.0),
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Ideal 8x8 grid centres, ordered row-major, with optional jitter."""
    rng = np.random.default_rng(seed)
    th = np.radians(rotation_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = []
    for r in range(ROWS):
        for c in range(COLS):
            x = c * pitch_x + (gap_x if c >= COLS / 2 else 0.0)
            y = r * pitch_y + (gap_y if r >= ROWS / 2 else 0.0)
            pts.append(R @ np.array([x, y]) + np.asarray(origin))
    pts = np.asarray(pts)
    if noise:
        pts = pts + rng.normal(0.0, noise, pts.shape)
    return pts


def test_recovers_exact_grid():
    truth = make_grid()
    fit = solve_lattice(truth, ROWS, COLS)
    assert fit.rmse < 1e-6
    assert fit.inliers.sum() == ROWS * COLS
    np.testing.assert_allclose(fit.all_centres(), truth, atol=1e-6)


@pytest.mark.parametrize("rotation", [0.0, 1.5, -2.35, 5.0, -8.0])
def test_recovers_rotation(rotation):
    """Rotation must be recovered; the real plates run 0-4.4 degrees."""
    truth = make_grid(rotation_deg=rotation)
    fit = solve_lattice(truth, ROWS, COLS)
    assert fit.rmse < 1e-6
    assert fit.rotation_deg == pytest.approx(rotation, abs=1e-3)


def test_recovers_quadrant_gap():
    """The gap between rows/cols 3 and 4 is part of the model, not an outlier."""
    truth = make_grid(gap_x=90.0, gap_y=70.0)
    fit = solve_lattice(truth, ROWS, COLS)
    assert fit.rmse < 1e-6


def test_pitch_and_anisotropy():
    fit = solve_lattice(make_grid(pitch_x=250.0, pitch_y=180.0), ROWS, COLS)
    pc, pr = fit.pitch
    assert pc == pytest.approx(250.0, rel=1e-6)
    assert pr == pytest.approx(180.0, rel=1e-6)


def test_averaging_beats_single_feature_noise():
    """The core claim: a joint fit averages down per-feature localisation error.

    With independent noise on 64 features and 10 parameters, the fitted centres
    should be materially closer to truth than the noisy observations are.
    """
    truth = make_grid()
    noisy = make_grid(noise=15.0, seed=3)          # ~the +/-15 um quantisation limit
    fit = solve_lattice(noisy, ROWS, COLS)
    err_raw = np.linalg.norm(noisy - truth, axis=1).mean()
    err_fit = np.linalg.norm(fit.all_centres() - truth, axis=1).mean()
    assert err_fit < err_raw / 3.0, f"raw {err_raw:.2f} -> fit {err_fit:.2f}"


def test_tolerates_missing_features():
    """Torn samples leave slots empty; the fit must still be correct."""
    truth = make_grid()
    keep = np.ones(len(truth), bool)
    keep[[0, 9, 27, 40, 63]] = False               # 5 missing
    fit = solve_lattice(truth[keep], ROWS, COLS)
    assert fit.rmse < 1e-6
    np.testing.assert_allclose(fit.all_centres(), truth, atol=1e-6)


def test_rejects_dust_when_given_surplus_candidates():
    """Debris must not displace the assignment.

    This is the bug that made four plates fail: truncating detections to exactly
    64 forced dust into a slot. Given surplus candidates, the assignment should
    ignore points that do not lie on the grid.
    """
    truth = make_grid()
    dust = np.array([[20.0, 20.0], [1900.0, 40.0], [60.0, 1400.0], [980.0, 700.0]])
    fit = solve_lattice(np.vstack([truth, dust]), ROWS, COLS)
    assert fit.rmse < 1.0, f"dust displaced the fit: rmse {fit.rmse}"
    np.testing.assert_allclose(fit.all_centres(), truth, atol=1.0)


def test_robust_pass_downweights_one_displaced_feature():
    """A single badly displaced feature should not drag the whole model."""
    truth = make_grid()
    corrupt = truth.copy()
    corrupt[30] += np.array([160.0, 120.0])        # torn and shifted
    fit = solve_lattice(corrupt, ROWS, COLS)
    assert not fit.inliers[np.argmax(fit.residuals)]
    good = np.arange(len(truth)) != 30
    np.testing.assert_allclose(fit.all_centres()[good], truth[good], atol=5.0)


def test_no_phantom_row_in_the_quadrant_gap():
    """Regression: the fit must not hide a row inside the quadrant gap.

    Selecting on residual alone lets the model place one predicted row in the
    empty gap and discard a real edge row as an outlier. Seven of eight rows
    then fit tightly, so RMSE looks excellent while the arrangement is wrong --
    which is exactly what happened on the photos: the fitted gap came out
    *smaller* than the row pitch (123 px against 182) and the real bottom row
    at y=1465 was dropped.
    """
    truth = make_grid(pitch_x=250.0, pitch_y=182.0, gap_y=120.0, gap_x=0.0)
    fit = solve_lattice(truth, ROWS, COLS)

    assert fit.coverage(truth) == pytest.approx(1.0), "a real row was dropped"

    ys = np.array([fit.predict((r, 0))[0][1] for r in range(ROWS)])
    gaps = np.diff(ys)
    mid = gaps[ROWS // 2 - 1]
    others = np.delete(gaps, ROWS // 2 - 1)
    assert mid > others.max(), (
        f"middle gap {mid:.1f} should exceed the row pitch {others.max():.1f}"
    )
    np.testing.assert_allclose(fit.all_centres(), truth, atol=1e-6)


def test_coverage_falls_when_model_is_misregistered():
    """Coverage must drop when predictions no longer sit on real features."""
    truth = make_grid()
    fit = solve_lattice(truth, ROWS, COLS)
    assert fit.coverage(truth) == pytest.approx(1.0)

    # Translate the model well clear of the data; nothing should be covered.
    import dataclasses
    coef = fit.coef.copy()
    coef[4] += np.array([500.0, 500.0])          # row 4 is the intercept
    moved = dataclasses.replace(fit, coef=coef)
    assert moved.coverage(truth) < 0.5


def test_quality_prefers_covering_fit_over_lower_residual():
    """The selection score must not be won by a tighter but incomplete fit."""
    truth = make_grid(gap_y=120.0)
    good = solve_lattice(truth, ROWS, COLS)
    # Drop a real row and re-fit on 7 rows' worth of points: residual can be
    # tiny, but coverage against the full point set must be worse.
    partial_pts = truth[: ROWS * COLS - COLS]
    partial = solve_lattice(partial_pts, ROWS, COLS)
    assert good.quality(truth) <= partial.quality(truth)


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="need >="):
        solve_lattice(make_grid()[:3], ROWS, COLS)


def test_assign_handles_rectangular_cost():
    centres = make_grid()[:60]
    slots = make_grid()
    det, slot = assign_to_grid(centres, slots)
    assert len(det) == len(slot) == 60
    assert len(set(slot.tolist())) == 60          # one-to-one


def test_fit_lattice_rejects_mismatched_lengths():
    rc = np.array([[0, 0], [0, 1]])
    with pytest.raises(ValueError, match="equal length"):
        fit_lattice(rc, np.zeros((3, 2)), ROWS, COLS)


def test_weighted_centroid_beats_binary_on_coarse_axis():
    """Intensity weighting should recover sub-sample precision.

    Mirrors the real geometry: ~2.9 um sampling on one axis and ~30 um on the
    other, so a feature spans only a few samples across.
    """
    x = np.arange(0.0, 2400.0, 30.0)
    y = np.arange(0.0, 2400.0, 2.9)
    xm, ym = np.meshgrid(x, y)
    cx, cy = 611.0, 707.0                          # deliberately off-sample
    field = 10.0 * np.exp(-(((xm - cx) / 50.0) ** 2 + ((ym - cy) / 50.0) ** 2))

    centres, areas = detect_features(field, x, y, threshold=2.0, min_area_px=3)
    assert len(centres) == 1
    err = np.linalg.norm(centres[0] - np.array([cx, cy]))
    # Half the coarse sample spacing would be 15 um; weighting should beat it.
    assert err < 15.0, f"weighted centroid off by {err:.2f} um"
