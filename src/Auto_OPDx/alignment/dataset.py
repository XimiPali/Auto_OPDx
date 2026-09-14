"""End-to-end construction of clean per-cell labels and crops.

Pipeline per plate::

    .OPDx  -> load (um)  -> fit background plane -> detrend
                         -> detect features      -> fit 8x8 lattice   (um)
    .jpg   -> load       -> flatten illumination -> detect features
                         -> fit 8x8 lattice                            (px)
             -> affine px->um from the 64 matched slots
             -> per-cell crops + labels

Design decisions worth knowing
------------------------------
*Labels come from native, un-interpolated data.* Interpolation smooths peaks,
and the label is a maximum, so resampling before labelling would bias it. The
cell *images* are resampled (they have to be, to have a fixed shape), but the
scalar labels are read straight off the native grid.

*Labels are plane-relative.* This is the fix for the position confound: raw
``z_max`` labels correlate with row index at r = +0.98, so they largely encode
where a cell sits. Subtracting the fitted substrate plane removes that.

*Crops are square in micrometres, not in pixels.* The photos are 2048x1536 but
cover a nearly square field, so pixel scale differs ~1.36x between axes.
A square pixel box is a non-square physical region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from Auto_OPDx.alignment.lattice import (
    LatticeFit,
    assign_to_grid,
    detect_features,
    solve_lattice,
)
from Auto_OPDx.alignment.plate import Plate, detrend, load_plate
from Auto_OPDx.alignment.rgb import detect_rgb_features, load_rgb

# Fraction of the 99th-percentile residual used as the feature threshold, with
# a floor so near-flat plates do not detect noise.
_FEATURE_FRAC = 0.30
_FEATURE_FLOOR_UM = 2.0


@dataclass
class PlateSolution:
    """Everything derived from one plate's height map."""

    plate: Plate
    resid: np.ndarray
    tilt: dict[str, float]
    fit: LatticeFit
    centres: np.ndarray
    threshold_um: float


@dataclass
class PhotoSolution:
    """Everything derived from one plate's RGB photo."""

    img: np.ndarray
    response: np.ndarray
    fit: LatticeFit
    centres: np.ndarray


@dataclass
class CrossModal:
    """Affine map from photo pixels to height-map micrometres."""

    affine: np.ndarray           # (3, 2): [px_x, px_y, 1] @ affine -> (um_x, um_y)
    residuals_um: np.ndarray     # per-slot residual magnitude, um
    slots: np.ndarray            # (n,) slot ids contributing
    um_per_px: tuple[float, float]
    rotation_deg: float

    @property
    def rmse_um(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_um ** 2))) if len(self.residuals_um) else float("nan")

    @property
    def max_um(self) -> float:
        return float(self.residuals_um.max()) if len(self.residuals_um) else float("nan")

    def to_um(self, px: np.ndarray) -> np.ndarray:
        px = np.atleast_2d(np.asarray(px, float))
        return np.c_[px, np.ones(len(px))] @ self.affine


@dataclass
class Cell:
    """One grid cell: matched RGB and height-map crops plus scalar labels."""

    plate: str
    row: int
    col: int
    slot: int
    rgb: np.ndarray | None
    height: np.ndarray | None
    labels: dict[str, float] = field(default_factory=dict)
    complete: bool = True
    overhang: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.plate}_r{self.row:02d}_c{self.col:02d}"


# Detection quality is strongly non-monotonic in the threshold, so both
# modalities search a small parameter grid and keep the best fit. Selection is on
# residual PLUS slot coverage, never residual alone -- see LatticeFit.coverage:
# residual alone prefers a degenerate fit that hides a phantom row in the
# quadrant gap. No labels or ground truth are consulted either way.
_HM_SEARCH = tuple(
    (frac, area)
    for frac in (0.30, 0.20, 0.45, 0.12)
    for area in (6, 3, 15)
)
_RGB_SEARCH = tuple(
    (pct, frac)
    for pct in (90.0, 93.0, 85.0, 80.0, 96.0)
    for frac in (0.008, 0.003, 0.02)
)


def _score(fit: LatticeFit, centres: np.ndarray) -> float:
    """Coverage-penalised lattice quality; lower is better.

    Residual alone is not a safe selection criterion: the model can insert a
    phantom row inside the quadrant gap and drop a real row at the edge, which
    fits seven of eight rows tightly and so *wins* on RMSE while being wrong.
    See ``LatticeFit.coverage``.
    """
    return fit.quality(centres)


def solve_plate(path: str | Path, rows: int = 8, cols: int = 8) -> PlateSolution:
    """Load a height map, detrend it, and fit the lattice via parameter search."""
    plate = load_plate(path)
    resid, tilt = detrend(plate)
    p99 = float(np.percentile(resid, 99.0))

    best: tuple[float, LatticeFit, np.ndarray, float] | None = None
    for frac, min_area in _HM_SEARCH:
        thr = max(_FEATURE_FLOOR_UM, p99 * frac)
        centres, _ = detect_features(
            resid, plate.x, plate.y, threshold=thr, min_area_px=min_area
        )
        if len(centres) < rows * cols:
            continue
        try:
            fit = solve_lattice(centres, rows, cols)
        except (ValueError, np.linalg.LinAlgError):
            continue
        s = _score(fit, centres)
        if best is None or s < best[0]:
            best = (s, fit, centres, thr)

    if best is None:
        # Nothing reached a full grid; fall back to the nominal settings so the
        # caller still gets a (possibly poor) answer rather than an exception.
        thr = max(_FEATURE_FLOOR_UM, p99 * _FEATURE_FRAC)
        centres, _ = detect_features(
            resid, plate.x, plate.y, threshold=thr, min_area_px=6
        )
        fit = solve_lattice(centres, rows, cols)
        best = (_score(fit, centres), fit, centres, thr)

    _, fit, centres, thr = best
    return PlateSolution(plate, resid, tilt, fit, centres, thr)


def solve_photo(path: str | Path, rows: int = 8, cols: int = 8) -> PhotoSolution:
    """Load a photo, flatten illumination, and fit the lattice via search."""
    from Auto_OPDx.alignment.rgb import dark_feature_response, detect_at

    img = load_rgb(path)
    dark, _ = dark_feature_response(img)

    best: tuple[float, LatticeFit, np.ndarray] | None = None
    for pct, frac in _RGB_SEARCH:
        centres, _ = detect_at(
            dark, rows, cols, percentile=pct, min_area_frac=frac
        )
        if len(centres) < rows * cols:
            continue
        try:
            fit = solve_lattice(centres, rows, cols)
        except (ValueError, np.linalg.LinAlgError):
            continue
        s = _score(fit, centres)
        if best is None or s < best[0]:
            best = (s, fit, centres)

    if best is None:
        centres, _, dark = detect_rgb_features(img, rows=rows, cols=cols)
        fit = solve_lattice(centres, rows, cols)
        best = (_score(fit, centres), fit, centres)

    _, fit, centres = best
    return PhotoSolution(img, dark, fit, centres)


def _slot_observations(fit: LatticeFit, cols: int) -> dict[int, np.ndarray]:
    """Map slot id -> the detected position assigned to it (inliers only)."""
    out: dict[int, np.ndarray] = {}
    for (r, c), obs, keep in zip(fit.rc, fit.observed, fit.inliers):
        if keep:
            out[int(r) * cols + int(c)] = obs
    return out


def fit_cross_modal(
    hm: PlateSolution, photo: PhotoSolution, cols: int = 8
) -> CrossModal:
    """Least-squares affine from photo pixels to height-map micrometres.

    Because both lattices are indexed by the same (row, col), correspondence is
    free -- no image feature matching is needed. Six parameters are fit from up
    to 64 point pairs.

    The reported residual uses the *detected* centroids on both sides, not the
    lattice predictions. Comparing predictions to predictions would flatter the
    result, since both are smooth models of the same grid; comparing detections
    includes the real detection noise of both modalities.
    """
    h_obs = _slot_observations(hm.fit, cols)
    r_obs = _slot_observations(photo.fit, cols)
    slots = np.array(sorted(set(h_obs) & set(r_obs)))
    if len(slots) < 3:
        raise ValueError(f"only {len(slots)} slots common to both modalities")

    src = np.array([r_obs[s] for s in slots])   # px
    dst = np.array([h_obs[s] for s in slots])   # um
    A = np.c_[src, np.ones(len(src))]
    affine, *_ = np.linalg.lstsq(A, dst, rcond=None)
    resid = np.linalg.norm(dst - A @ affine, axis=1)

    # Column 0 of `affine` maps px -> um_x; rows are the px_x / px_y inputs.
    sx = float(np.hypot(affine[0, 0], affine[0, 1]))   # um per px along image x
    sy = float(np.hypot(affine[1, 0], affine[1, 1]))   # um per px along image y
    rot = float(np.degrees(np.arctan2(affine[0, 1], affine[0, 0])))
    return CrossModal(affine, resid, slots, (sx, sy), rot)


def _crop_um(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    size_um: float,
    out_px: int,
) -> tuple[np.ndarray | None, bool, float]:
    """Crop a physically square window and resample to ``out_px`` square.

    Sample coordinates are clamped to the data bounds, so a window that
    overhangs the edge replicates the border rather than producing NaN or
    black. Border cells are still reported via the ``complete`` flag and the
    overhang fraction, so downstream code can drop them if it wants to.

    Returns
    -------
    crop, complete, overhang
        ``overhang`` is the fraction of the requested window that fell outside
        the data.
    """
    from scipy.interpolate import RegularGridInterpolator

    xs, ys, vals = x, y, values
    if xs[0] > xs[-1]:
        xs, vals = xs[::-1], vals[:, ::-1]
    if ys[0] > ys[-1]:
        ys, vals = ys[::-1], vals[::-1, :]

    half = size_um / 2.0
    gx = np.linspace(cx - half, cx + half, out_px)
    gy = np.linspace(cy - half, cy + half, out_px)
    outside = (
        (gx < xs[0]).sum() + (gx > xs[-1]).sum()
    ) / out_px + (
        (gy < ys[0]).sum() + (gy > ys[-1]).sum()
    ) / out_px
    overhang = float(min(outside, 1.0))
    complete = overhang == 0.0

    gx = np.clip(gx, xs[0], xs[-1])
    gy = np.clip(gy, ys[0], ys[-1])
    interp = RegularGridInterpolator(
        (ys, xs), vals, method="linear", bounds_error=False, fill_value=None
    )
    mesh_y, mesh_x = np.meshgrid(gy, gx, indexing="ij")
    crop = interp(np.stack([mesh_y.ravel(), mesh_x.ravel()], -1)).reshape(out_px, out_px)
    if not np.isfinite(crop).any():
        return None, False, overhang
    return crop, complete, overhang


def _native_labels(
    hm: PlateSolution, cx: float, cy: float, size_um: float
) -> dict[str, float]:
    """Scalar labels from the native, un-interpolated detrended grid.

    A plane, not a higher-order surface, is subtracted. Fitting a quadratic
    raises the label's correlation with row index (0.64 -> 0.79 on
    10-01-2025_P1) and a cubic overcorrects into the features themselves,
    producing physically impossible negative thicknesses. The background is
    genuinely planar; higher orders just start absorbing signal.
    """
    half = size_um / 2.0
    xi = np.where((hm.plate.x >= cx - half) & (hm.plate.x <= cx + half))[0]
    yi = np.where((hm.plate.y >= cy - half) & (hm.plate.y <= cy + half))[0]
    if xi.size == 0 or yi.size == 0:
        return {}
    patch = hm.resid[np.ix_(yi, xi)]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return {}
    return {
        # h_max matches the deck's y_max definition, but plane-relative.
        "h_max": float(patch.max()),
        # p95 and the top-decile mean are robust alternatives: h_max is set by a
        # single pixel and so is sensitive to spikes.
        "h_p95": float(np.percentile(patch, 95)),
        "h_top10_mean": float(patch[patch >= np.percentile(patch, 90)].mean()),
        "h_mean": float(patch.mean()),
        "n_px": int(patch.size),
    }


def build_cells(
    hm: PlateSolution,
    photo: PhotoSolution | None,
    cross: CrossModal | None,
    rows: int = 8,
    cols: int = 8,
    cell_um: float | None = None,
    label_um: float | None = None,
    height_px: int = 32,
    rgb_px: int = 96,
) -> list[Cell]:
    """Extract all ``rows*cols`` cells for one plate.

    Parameters
    ----------
    cell_um
        Physical size of the image crops. Defaults to the fitted lattice pitch
        so crops tile the plate. Keeping the full pitch preserves surrounding
        context, which the results deck found to be genuine signal (tightening
        crops 40% dropped per-cell R^2 from 0.77 to 0.59).
    label_um
        Physical size of the window the scalar labels are read from. Defaults to
        ``cell_um``. ``h_max`` is insensitive to this (the peak sample is inside
        any window at least feature-sized), but ``h_p95`` and ``h_mean`` are
        not: a feature covers only ~12% of a full-pitch window, so percentile
        labels over the whole cell are mostly background. Set this near the
        feature size (~110 um) if you intend to use those.
    """
    if cell_um is None:
        cell_um = float(np.mean(hm.fit.pitch))
    if label_um is None:
        label_um = cell_um

    # Pre-compute the pixel-space transform once rather than per cell.
    px_of_um = None
    if photo is not None and cross is not None:
        M, t = cross.affine[:2], cross.affine[2]
        try:
            M_inv = np.linalg.inv(M.T)
        except np.linalg.LinAlgError:
            M_inv = None
        if M_inv is not None:
            px_of_um = lambda um: M_inv @ (np.asarray(um, float) - t)  # noqa: E731

    cells: list[Cell] = []
    for r in range(rows):
        for c in range(cols):
            hx, hy = hm.fit.predict((r, c))[0]

            hcrop, hok, hover = _crop_um(
                hm.resid, hm.plate.x, hm.plate.y, hx, hy, cell_um, height_px
            )
            labels = _native_labels(hm, hx, hy, label_um)

            rcrop, rok, rover = None, True, 0.0
            if px_of_um is not None:
                px = px_of_um((hx, hy))
                sx, sy = cross.um_per_px
                img = np.asarray(photo.img, float)
                # Scale pixel indices into micrometres so the crop is square in
                # physical space; the photo's pixel scale differs ~1.36x by axis.
                ix = np.arange(img.shape[1], dtype=float) * sx
                iy = np.arange(img.shape[0], dtype=float) * sy
                chans = []
                for ch in range(img.shape[2]):
                    cc, ok, ov = _crop_um(
                        img[:, :, ch], ix, iy,
                        px[0] * sx, px[1] * sy, cell_um, rgb_px,
                    )
                    rok = rok and ok
                    rover = max(rover, ov)
                    if cc is None:
                        chans = []
                        break
                    chans.append(cc)
                if chans:
                    stacked = np.nan_to_num(np.stack(chans, -1), nan=0.0)
                    rcrop = np.clip(stacked, 0, 255).astype(np.uint8)

            cells.append(Cell(
                plate=hm.plate.name, row=r, col=c, slot=r * cols + c,
                rgb=rcrop, height=hcrop, labels=labels,
                complete=bool(hok and rok and hcrop is not None),
                overhang=float(max(hover, rover)),
            ))
    return cells
