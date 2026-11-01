"""Feature detection in full-plate RGB photos.

Two properties of these images drive the implementation:

1. **Features are dark on a bright substrate.** A white top-hat, as used by
   ``calculate_brightness.tophat_response``, detects bright-on-dark and so has
   the wrong polarity here.

2. **Illumination varies strongly across the plate** -- the results deck flags
   this as a confound, and it is severe enough that a global percentile
   threshold detects mostly features from the brighter side of the plate,
   biasing any grid fit. Dividing by a heavily blurred copy of the image
   normalises contrast before thresholding.

The existing ``fit_robust_grid`` brute-forces a grid over pixel ranges
hardcoded to one camera at one magnification (``s_range = range(118, 129)``
and friends) and silently ignores its ``n_lines`` argument, always emitting
exactly 8 positions. Here, detected blobs are handed to the same generic
lattice solver used for height maps, so one code path serves both modalities
and neither is pinned to a magnification.

Note on pixel aspect: the stored images are 2048x1536 but the physical field of
view is very close to square, so the pixel scale differs by ~1.36x between axes.
Cropping a square *pixel* box therefore yields a non-square *physical* region.
Use :func:`Auto_OPDx.alignment.dataset.pixel_scale_um` to work in micrometres.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from Auto_OPDx.alignment.lattice import detect_features

# Percentiles tried in order until at least rows*cols blobs survive.
_THRESHOLD_LADDER = (93.0, 95.0, 90.0, 97.0, 87.0, 83.0)

# Long side, in pixels, of the photos the defaults were tuned on (2048x1536 at
# 4x). Larger images -- e.g. a 1-inch sensor imaging the same plate -- are
# treated as the same field at higher resolution.
_REFERENCE_PX = 2048


def load_rgb(path: str | Path) -> np.ndarray:
    """Load a plate photo as an RGB uint8 array.

    Ordinary 8-bit files (.jpg, .bmp, 8-bit .png/.tif) take the same path as
    always. 16-bit and floating-point images -- the usual output of focus-
    stacking software for an all-in-focus composite -- are scaled to 8 bits by
    their own 99.9th percentile rather than by the type's maximum, so a 12-bit
    image stored in 16 bits keeps its contrast instead of collapsing into the
    bottom 16 grey levels. Greyscale is replicated to three channels and alpha
    is dropped.
    """
    path = Path(path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"could not read image: {path}")
    if raw.dtype == np.uint8:
        # unchanged legacy path (also applies any EXIF orientation)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    img = raw.astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    top = float(np.percentile(img, 99.9))
    low = float(min(img.min(), 0.0))
    if not np.isfinite(top) or top <= low:
        raise ValueError(f"image has no usable dynamic range: {path}")
    img8 = np.clip((img - low) / (top - low) * 255.0, 0, 255).astype(np.uint8)
    if img8.ndim == 2:
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)


def dark_feature_response(
    img_rgb: np.ndarray, sigma: float = 120.0
) -> tuple[np.ndarray, np.ndarray]:
    """Illumination-flattened response in which dark features read bright.

    ``sigma`` must comfortably exceed the feature size so the blur estimates
    illumination rather than the features themselves. At ~250 px cell pitch,
    120 px is a good default.

    ``sigma`` is in units of a :data:`_REFERENCE_PX`-wide image. A larger image
    of the same field gets a proportionally larger blur, so the background
    estimate means the same thing at any resolution. The blur is then computed
    on a copy shrunk back to the reference size and enlarged again: the
    illumination is smooth, so nothing is lost, and a 22-megapixel photo costs
    the same as a 3-megapixel one (13 s -> 0.1 s for the blur on a 5468x4101
    image). Images no larger than the reference take the direct path exactly
    as before.

    Returns
    -------
    dark : ndarray
        Non-negative response, larger where the image is darker than its local
        background. Unitless (a fraction of local brightness).
    background : ndarray
        The illumination estimate, useful for plotting.
    """
    gray = np.asarray(img_rgb, float)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    h, w = gray.shape
    scale = max(h, w) / _REFERENCE_PX
    if scale <= 1.0:
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        small = cv2.resize(gray, (max(1, round(w / scale)), max(1, round(h / scale))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
        background = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    flat = gray / np.maximum(background, 1e-6)
    return np.clip(1.0 - flat, 0.0, None), background


def detect_at(
    dark: np.ndarray,
    rows: int = 8,
    cols: int = 8,
    percentile: float = 90.0,
    min_area_frac: float = 0.008,
    surplus: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-shot detection on a precomputed response, at fixed parameters.

    Parameters
    ----------
    min_area_frac
        Minimum blob area as a fraction of one cell's area, so the floor adapts
        to image resolution rather than being hardcoded.
    surplus
        Cap on retained candidates, as a multiple of ``rows * cols``.
        Deliberately greater than 1: truncating to exactly ``rows * cols`` by
        area forces dust and edge artefacts into a slot, which displaces the
        whole assignment and wrecks the lattice fit. Surplus candidates let the
        Hungarian match reject the points that do not sit on the grid.
    """
    h, w = dark.shape
    xs = np.arange(w, dtype=float)
    ys = np.arange(h, dtype=float)
    cell_area = (w / cols) * (h / rows)
    min_area_px = max(12, int(cell_area * min_area_frac))

    thr = float(np.percentile(dark, percentile))
    centres, areas = detect_features(
        dark, xs, ys, threshold=thr, min_area_px=min_area_px
    )
    cap = int(rows * cols * surplus)
    if len(centres) > cap:
        keep = np.argsort(areas)[::-1][:cap]
        centres, areas = centres[keep], areas[keep]
    return centres, areas


def detect_rgb_features(
    img_rgb: np.ndarray,
    rows: int = 8,
    cols: int = 8,
    sigma: float = 120.0,
    percentile: float = 90.0,
    min_area_frac: float = 0.008,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect grid features and return weighted centroids in pixel coordinates.

    Single-parameter convenience wrapper. Detection quality on these plates is
    strongly and *non-monotonically* sensitive to the threshold: 9-25-2025_P9
    fits to 1.3% of pitch at the 85th percentile but fails badly at the 93rd,
    while 9-30-2025_P7 is the other way round. Prefer
    :func:`Auto_OPDx.alignment.dataset.solve_photo`, which searches these
    parameters and keeps whichever forms the tightest grid.

    Returns
    -------
    centres : ndarray, shape (n, 2)
        Weighted centroids as (x, y) in pixels.
    areas : ndarray, shape (n,)
    dark : ndarray
        The response image, for plotting.
    """
    dark, _ = dark_feature_response(img_rgb, sigma=sigma)
    centres, areas = detect_at(
        dark, rows, cols, percentile=percentile, min_area_frac=min_area_frac
    )
    return centres, areas, dark
