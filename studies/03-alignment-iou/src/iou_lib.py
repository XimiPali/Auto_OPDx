"""Shared machinery for the alignment IoU study.

The question this study answers is narrow: *for each cell, does the photo crop
and the height-map crop a method hands out contain the same pad in the same
place?* Intersection-over-union of the two pad masks is the metric.

To compare the previous crops with the new ones fairly, everything except the
cell centres is held fixed:

* **window** - the same physical size (a fraction of the fitted cell pitch) in
  both modalities, for both methods, resampled to the same grid;
* **mask** - the same rule in both modalities: threshold at half the crop's
  peak above its background, keep the connected component nearest the crop
  centre, fill holes;
* **frame** - photo pixels are converted to micrometres with the per-plate
  scale the registration recovers, so both masks live in the scan's frame.

A method therefore contributes exactly two numbers per cell: where it says the
cell is in the photo (px) and where it says it is in the height map (um). For
the previous crops those are the stored centroids that produced the training
set (confirmed by template-matching the stored crops back into the photos:
correlation 0.997-0.999). For the new tool they are what ``align-plates``
writes to disk: the detected pad centre when the solver assigned one to the
slot, the lattice prediction otherwise. The first release used the lattice
prediction for every cell; it is scored too, as ``new_lattice``, because this
study is what showed the difference.

Rotation of the photo relative to the scan (median 1.7 deg) is ignored when
converting a crop: over a 240 um window it moves a pad edge by ~2 um, far below
the 30 um sampling of the scan's coarse axis.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, map_coordinates

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
sys.path.insert(0, str(REPO / "src"))

from Auto_OPDx.alignment.dataset import (  # noqa: E402
    CrossModal,
    PhotoSolution,
    PlateSolution,
    _crop_um,
    _slot_observations,
    fit_cross_modal,
    solve_photo,
    solve_plate,
)

ROWS = COLS = 8
WINDOW_FRAC = 0.9        # window side as a fraction of the fitted pitch
GRID = 96                # resampled crop side, px  (~2.5 um/px at 240 um)
MIN_AREA_FRAC = 0.012    # smallest component accepted as "a pad": ~110 px^2 of 96^2
HM_PRESENCE_UM = 2.5     # peak above background needed to call a pad present
HM_FLOOR_UM = 1.5
PHOTO_PRESENCE = 0.08    # same, in dark-response units (fraction of local brightness)
PHOTO_FLOOR = 0.04
SEARCH_FRAC = 1.2        # window used only to decide whether a pad physically exists

METHODS = ("old", "new_lattice", "new")
METHOD_LABEL = {
    "old": "previous crops",
    "new_lattice": "align-plates, lattice slots only (as shipped 2026-10-08)",
    "new": "align-plates (detected centres, current default)",
}


# ----------------------------------------------------------------- inputs ----

def photo_path(stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".bmp", ".png"):
        p = REPO / "data" / "rgb" / "img" / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_old_metadata() -> tuple[dict, dict]:
    """Stored centroids of the previous pipeline, keyed by plate stem."""
    meta = REPO / "data" / "metadata"
    prof = json.loads((meta / "prof_crop_details_v2.json").read_text())
    img = json.loads((meta / "img_crop_details.json").read_text())
    strip = lambda d: {k.rsplit(".", 1)[0]: v for k, v in d.items()}  # noqa: E731
    return strip(prof), strip(img)


def old_centroid(entry: dict, r: int, c: int) -> np.ndarray | None:
    """One stored centroid, or None if the previous pipeline reported it missing."""
    v = entry.get(f"{r},{c}") or entry.get(f"({r}, {c})")
    if v and v.get("found") and v.get("centroid"):
        return np.asarray(v["centroid"], float)
    return None


def comparable_plates() -> list[str]:
    """Plates the previous pipeline produced both a photo and a scan result for."""
    prof, img = load_old_metadata()
    stems = sorted(set(prof) & set(img))
    return [s for s in stems if (REPO / "data" / "opdx" / f"{s}.OPDx").exists()
            and photo_path(s) is not None]


# ------------------------------------------------- old units -> micrometres ----

@dataclass
class UnitConversion:
    """Affine from the previous pipeline's resampled scan space to micrometres.

    The previous pipeline rotated the scan upright and resampled it before
    finding centroids, so its coordinates are an unknown affine image of the
    native axes. Recovering that affine needs point pairs; the (row, col) index
    gives them for free. The fit is robust (3-MAD rejection, 3 rounds) so the
    cells the previous pipeline got wrong - typically off by a whole pitch - do
    not bend the conversion for the ones it got right.
    """

    M: np.ndarray                 # (3, 2): [u, v, 1] @ M -> (x_um, y_um)
    rmse_um: float
    n_used: int
    n_total: int
    um_per_unit: tuple[float, float]
    rotation_deg: float

    def to_um(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        return (np.c_[uv, np.ones(len(uv))] @ self.M)[0]


def fit_unit_conversion(entry: dict, hm: PlateSolution) -> UnitConversion:
    obs = _slot_observations(hm.fit, COLS)
    U, Y = [], []
    for r in range(ROWS):
        for c in range(COLS):
            o = old_centroid(entry, r, c)
            if o is not None and (r * COLS + c) in obs:
                U.append(o)
                Y.append(obs[r * COLS + c])
    U, Y = np.asarray(U), np.asarray(Y)
    if len(U) < 4:
        raise ValueError("too few stored centroids to recover the unit conversion")
    A = np.c_[U, np.ones(len(U))]
    keep = np.ones(len(U), bool)
    for _ in range(3):
        M, *_ = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
        res = np.linalg.norm(Y - A @ M, axis=1)
        med = np.median(res[keep])
        mad = 1.4826 * np.median(np.abs(res[keep] - med)) + 1e-9
        keep = res < med + 3 * mad
    M, *_ = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
    res = np.linalg.norm(Y - A @ M, axis=1)
    return UnitConversion(
        M=M,
        rmse_um=float(np.sqrt(np.mean(res[keep] ** 2))),
        n_used=int(keep.sum()),
        n_total=int(len(U)),
        um_per_unit=(float(np.hypot(M[0, 0], M[0, 1])), float(np.hypot(M[1, 0], M[1, 1]))),
        rotation_deg=float(np.degrees(np.arctan2(M[0, 1], M[0, 0]))),
    )


# ------------------------------------------------------------- cropping ----

def crop_height(hm: PlateSolution, cx: float, cy: float, size_um: float) -> np.ndarray | None:
    """Window of the detrended height map; outside the scan is NaN.

    ``_crop_um`` (what the tool uses) replicates the border instead, which is
    right for a training crop but wrong here: a replicated pad edge becomes a
    streak that the mask step would count as pad. Edge cells on both methods
    would be penalised for where the scan stops, not for alignment.
    """
    from scipy.interpolate import RegularGridInterpolator

    xs, ys, vals = hm.plate.x, hm.plate.y, hm.resid
    if xs[0] > xs[-1]:
        xs, vals = xs[::-1], vals[:, ::-1]
    if ys[0] > ys[-1]:
        ys, vals = ys[::-1], vals[::-1, :]
    half = size_um / 2.0
    gx = np.linspace(cx - half, cx + half, GRID)
    gy = np.linspace(cy - half, cy + half, GRID)
    interp = RegularGridInterpolator((ys, xs), vals, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    my, mx = np.meshgrid(gy, gx, indexing="ij")
    crop = interp(np.stack([my.ravel(), mx.ravel()], -1)).reshape(GRID, GRID)
    return crop if np.isfinite(crop).any() else None


def crop_photo(
    arr: np.ndarray, cx: float, cy: float, size_um: float, um_per_px: tuple[float, float]
) -> np.ndarray:
    """Resample a ``size_um``-square window of a photo-space array to GRID.

    ``arr`` may be the dark response (float, 2-D) or the RGB image (3-D).
    Outside the image is NaN (response) or 0 (RGB), for the reason given on
    :func:`crop_height`.
    """
    sx, sy = um_per_px
    half = size_um / 2.0
    u = np.linspace(-half, half, GRID)
    gx = cx + u / sx
    gy = cy + u / sy
    my, mx = np.meshgrid(gy, gx, indexing="ij")
    coords = np.stack([my.ravel(), mx.ravel()])
    if arr.ndim == 2:
        return map_coordinates(arr.astype(float), coords, order=1,
                               mode="constant", cval=np.nan).reshape(GRID, GRID)
    out = [map_coordinates(arr[..., k].astype(float), coords, order=1, mode="constant", cval=0.0).reshape(GRID, GRID)
           for k in range(arr.shape[2])]
    return np.stack(out, -1)


# --------------------------------------------------------------- masking ----

def pad_mask(crop: np.ndarray, floor: float, presence: float) -> np.ndarray | None:
    """Binary mask of the pad nearest the crop centre, or None if there is none.

    Threshold is half the crop's peak (99.5th percentile, so one hot pixel does
    not set it) above its background (median). Among components larger than
    ``MIN_AREA_FRAC`` of the crop, the one whose centroid is nearest the centre
    is kept; holes are filled so a ring-like photo response counts as a solid
    pad, matching what the height map sees.
    """
    if crop is None or not np.isfinite(crop).any():
        return None
    bg = float(np.nanmedian(crop))
    peak = float(np.nanpercentile(crop, 99.5))
    if peak - bg < presence:
        return None
    thr = max(bg + 0.5 * (peak - bg), floor)
    binary = np.nan_to_num(crop, nan=bg) > thr
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    min_area = MIN_AREA_FRAC * GRID * GRID
    centre = np.array([GRID / 2, GRID / 2])
    best, best_d = None, np.inf
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        d = np.linalg.norm(cents[k] - centre)
        if d < best_d:
            best, best_d = k, d
    if best is None:
        return None
    return binary_fill_holes(labels == best)


def iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_centroid_um(m: np.ndarray | None, size_um: float) -> np.ndarray | None:
    if m is None or not m.any():
        return None
    yy, xx = np.nonzero(m)
    return (np.array([xx.mean(), yy.mean()]) - (GRID - 1) / 2) * (size_um / (GRID - 1))


def centred_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """IoU after sliding ``b`` so the two centroids coincide.

    This is the ceiling any alignment could reach given that the two
    instruments draw the pad with different shapes (the scan samples the pad
    at 30 um along x; the photo at 1.25 um). The gap between this and the
    measured IoU is the part attributable to *position*.
    """
    if a is None or b is None:
        return 0.0
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    dy = int(round(ya.mean() - yb.mean()))
    dx = int(round(xa.mean() - xb.mean()))
    shifted = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
    # roll wraps; blank the wrapped band so nothing re-enters from the far side
    if dy > 0:
        shifted[:dy] = False
    elif dy < 0:
        shifted[dy:] = False
    if dx > 0:
        shifted[:, :dx] = False
    elif dx < 0:
        shifted[:, dx:] = False
    return iou(a, shifted)


# ------------------------------------------------------------ evaluation ----

@dataclass
class CellResult:
    plate: str
    row: int
    col: int
    method: str
    status: str                    # ok | not_found | no_pad_in_crop | excluded_missing
    iou: float
    ceiling_iou: float
    offset_um: float               # distance between the two mask centroids
    photo_centre_px: tuple[float, float] | None
    hm_centre_um: tuple[float, float] | None
    # kept for figures; dropped before writing CSV
    photo_crop: np.ndarray | None = field(default=None, repr=False)
    photo_rgb: np.ndarray | None = field(default=None, repr=False)
    hm_crop: np.ndarray | None = field(default=None, repr=False)
    photo_mask: np.ndarray | None = field(default=None, repr=False)
    hm_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class PlateEval:
    stem: str
    hm: PlateSolution
    photo: PhotoSolution
    cross: CrossModal
    conv: UnitConversion
    pitch_um: float
    window_um: float
    present_hm: np.ndarray         # (8, 8) bool - a pad exists in the scan
    present_photo: np.ndarray      # (8, 8) bool - a pad exists in the photo
    cells: list[CellResult]


def _affine_inverse(cross: CrossModal):
    M, t = cross.affine[:2], cross.affine[2]
    M_inv = np.linalg.inv(M.T)
    return lambda um: M_inv @ (np.asarray(um, float) - t)


def evaluate_plate(stem: str, prof_entry: dict, img_entry: dict, keep_arrays: bool = False) -> PlateEval:
    hm = solve_plate(REPO / "data" / "opdx" / f"{stem}.OPDx", ROWS, COLS)
    photo = solve_photo(photo_path(stem), ROWS, COLS)
    cross = fit_cross_modal(hm, photo, COLS)
    conv = fit_unit_conversion(prof_entry, hm)
    um_to_px = _affine_inverse(cross)
    pitch = float(np.mean(hm.fit.pitch))
    window = WINDOW_FRAC * pitch
    hm_obs = _slot_observations(hm.fit, COLS)
    ph_obs = _slot_observations(photo.fit, COLS)
    img_f = np.asarray(photo.img, float)

    present_hm = np.zeros((ROWS, COLS), bool)
    present_photo = np.zeros((ROWS, COLS), bool)
    cells: list[CellResult] = []

    for r in range(ROWS):
        for c in range(COLS):
            slot = r * COLS + c
            slot_um = hm.fit.predict((r, c))[0]
            slot_px = um_to_px(slot_um)

            # Does a pad physically exist here? Decided once, from the geometry,
            # with a generous window so a registration error cannot hide one.
            search = SEARCH_FRAC * pitch
            present_hm[r, c] = pad_mask(crop_height(hm, *slot_um, search), HM_FLOOR_UM, HM_PRESENCE_UM) is not None
            present_photo[r, c] = pad_mask(
                crop_photo(photo.response, *slot_px, search, cross.um_per_px), PHOTO_FLOOR, PHOTO_PRESENCE
            ) is not None
            physically_present = present_hm[r, c] and present_photo[r, c]

            centres = {
                "old": (
                    old_centroid(img_entry, r, c),
                    None if (o := old_centroid(prof_entry, r, c)) is None else conv.to_um(o),
                ),
                # the first align-plates release cut every crop at the lattice
                # prediction; the current one centres on the detected pad when
                # the solver assigned one to the slot (same rule as build_cells)
                "new_lattice": (slot_px, slot_um),
                "new": (
                    ph_obs.get(slot, slot_px),
                    hm_obs.get(slot, slot_um),
                ),
            }

            for method, (p_px, h_um) in centres.items():
                res = CellResult(stem, r, c, method, "ok", 0.0, 0.0, float("nan"),
                                 None if p_px is None else (float(p_px[0]), float(p_px[1])),
                                 None if h_um is None else (float(h_um[0]), float(h_um[1])))
                if not physically_present:
                    res.status = "excluded_missing"
                    cells.append(res)
                    continue
                if p_px is None or h_um is None:
                    res.status = "not_found"
                    cells.append(res)
                    continue
                hcrop = crop_height(hm, *h_um, window)
                pcrop = crop_photo(photo.response, *p_px, window, cross.um_per_px)
                hmask = pad_mask(hcrop, HM_FLOOR_UM, HM_PRESENCE_UM)
                pmask = pad_mask(pcrop, PHOTO_FLOOR, PHOTO_PRESENCE)
                if hmask is None or pmask is None:
                    res.status = "no_pad_in_crop"
                else:
                    res.iou = iou(pmask, hmask)
                    res.ceiling_iou = centred_iou(pmask, hmask)
                    a, b = mask_centroid_um(pmask, window), mask_centroid_um(hmask, window)
                    res.offset_um = float(np.linalg.norm(a - b))
                if keep_arrays:
                    res.photo_crop, res.hm_crop = pcrop, hcrop
                    res.photo_mask, res.hm_mask = pmask, hmask
                    res.photo_rgb = crop_photo(img_f, *p_px, window, cross.um_per_px).clip(0, 255).astype(np.uint8)
                cells.append(res)

    return PlateEval(stem, hm, photo, cross, conv, pitch, window, present_hm, present_photo, cells)


def rectangle_iou_vs_offset(w_um: float, h_um: float, offsets_um: np.ndarray) -> np.ndarray:
    """IoU of two identical w x h rectangles displaced diagonally by ``offset``.

    Reference curve for reading the metric: it shows what a given IoU means in
    micrometres for a pad-sized box.
    """
    d = offsets_um / np.sqrt(2)
    inter = np.clip(w_um - d, 0, None) * np.clip(h_um - d, 0, None)
    return inter / (2 * w_um * h_um - inter)
