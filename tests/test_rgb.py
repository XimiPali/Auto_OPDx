"""Tests for photo loading and the illumination-flattened response.

These cover what changes when the lab moves from today's 4x photos to larger,
all-in-focus composites: higher resolution of the same field, and 16-bit or
floating-point files from focus-stacking software.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from Auto_OPDx.alignment import rgb
from Auto_OPDx.alignment.lattice import solve_lattice


def synthetic_plate(w=512, h=384, rows=8, cols=8, seed=0):
    """Bright substrate with a vignetting gradient and an 8x8 grid of dark pads."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    light = 200 - 60 * (((xx - 0.3 * w) / w) ** 2 + ((yy - 0.4 * h) / h) ** 2)
    img = light.copy()
    pitch_x, pitch_y = 0.105 * w, 0.105 * w * 0.73
    gap = 0.03 * w
    side = 0.3 * pitch_x
    x0, y0 = 0.08 * w, 0.1 * h
    for r in range(rows):
        for c in range(cols):
            cx = x0 + c * pitch_x + (gap if c >= cols // 2 else 0)
            cy = y0 + r * pitch_y + (gap if r >= rows // 2 else 0)
            m = (np.abs(xx - cx) < side / 2) & (np.abs(yy - cy) < side / 2)
            img[m] *= 0.55
    img += rng.normal(0, 2.0, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([gray] * 3, -1)


# ---------------------------------------------------------------- loading ----

def test_8bit_files_take_the_legacy_path(tmp_path):
    img = (np.random.default_rng(1).random((40, 60, 3)) * 255).astype(np.uint8)
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), img)
    expected = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(rgb.load_rgb(p), expected)


def test_12bit_data_in_a_16bit_file_keeps_its_contrast(tmp_path):
    # values 0..4095, as a 12-bit camera writes into a 16-bit container
    ramp = np.tile(np.linspace(0, 4095, 200), (50, 1)).astype(np.uint16)
    p = tmp_path / "stack.png"
    cv2.imwrite(str(p), ramp)
    out = rgb.load_rgb(p)
    assert out.dtype == np.uint8 and out.shape == (50, 200, 3)
    assert out.max() >= 250, "dividing by 65535 would have left this below 16"
    assert np.corrcoef(out[..., 0].ravel(), ramp.ravel())[0, 1] > 0.999


def test_float_tiff_is_accepted(tmp_path):
    img = np.random.default_rng(2).random((30, 30)).astype(np.float32)
    p = tmp_path / "composite.tif"
    cv2.imwrite(str(p), img)
    out = rgb.load_rgb(p)
    assert out.dtype == np.uint8 and out.shape == (30, 30, 3)


def test_unreadable_file_raises(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="could not read"):
        rgb.load_rgb(p)


# ------------------------------------------------------------- response ----

def test_reference_size_images_are_unchanged(monkeypatch):
    monkeypatch.setattr(rgb, "_REFERENCE_PX", 512)
    img = synthetic_plate()
    dark, bg = rgb.dark_feature_response(img, sigma=30)
    gray = img.astype(float).mean(axis=2)
    direct = cv2.GaussianBlur(gray, (0, 0), sigmaX=30, sigmaY=30)
    np.testing.assert_array_equal(bg, direct)


def test_larger_image_of_the_same_field_gives_the_same_response(monkeypatch):
    monkeypatch.setattr(rgb, "_REFERENCE_PX", 512)
    small = synthetic_plate()
    big = cv2.resize(small, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    d_small, _ = rgb.dark_feature_response(small, sigma=30)
    d_big, _ = rgb.dark_feature_response(big, sigma=30)
    back = cv2.resize(d_big, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.corrcoef(back.ravel(), d_small.ravel())[0, 1] > 0.97


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_grid_is_found_at_any_resolution(monkeypatch, scale):
    monkeypatch.setattr(rgb, "_REFERENCE_PX", 512)
    img = synthetic_plate()
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    centres, _, _ = rgb.detect_rgb_features(img, sigma=30, percentile=90.0)
    fit = solve_lattice(centres, 8, 8)
    assert int(fit.inliers.sum()) == 64
    assert fit.coverage(centres) == 1.0
    # pitch scales with the image; residual stays a small fraction of it
    assert fit.pitch[0] == pytest.approx(0.105 * 512 * scale, rel=0.02)
    assert fit.rmse / np.mean(fit.pitch) < 0.02
