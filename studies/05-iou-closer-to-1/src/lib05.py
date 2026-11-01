"""Machinery for study 05: pushing the photo / height-map IoU toward 1.

Builds on the study-03 recipe (``iou_lib``) and makes every step of it
configurable, so each change can be switched on alone and measured:

* **measurement** changes alter how a pad outline is drawn or how finely it is
  rasterised. They are applied to *both* methods, so the previous-vs-new
  comparison stays fair: grid resolution, crop rotation, local background
  plane, outline clean-up, threshold level, photo colour channel.
* **alignment** changes alter where the new method cuts its crops: a
  cross-modal consistency check on the detections, and a second, per-cell pass.
  The previous method is Wayne's stored centroids and is not re-aligned.

The slow part of every run -- solving the two lattices and the registration for
each plate -- is cached under ``data/processed/iou05_cache/`` (git-ignored).
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
S03 = REPO / "studies" / "03-alignment-iou"
sys.path.insert(0, str(S03 / "src"))
sys.path.insert(0, str(REPO / "src"))

import iou_lib as L  # noqa: E402

CACHE = REPO / "data" / "processed" / "iou05_cache"
ROWS = COLS = 8
CHANNELS = ("gray", "red", "green", "blue")


# ------------------------------------------------------------- the config ----

@dataclass(frozen=True)
class Config:
    """One complete recipe. Defaults reproduce study 03 exactly."""

    name: str = "baseline"
    grid: int = L.GRID                 # crop side in pixels (96 -> 2.5 um/px over 240 um)
    rotate: bool = False               # resample the photo with the full affine (rotation + shear)
    local_plane: bool = False          # subtract a background plane fitted inside each crop
    closing_um: float = 0.0            # binary closing radius before filling (photo ring gaps)
    opening_um: float = 0.0            # binary opening radius (debris spurs, thin tails)
    photo_level: float = 0.5           # outline threshold as a fraction of peak above background
    scan_level: float = 0.5
    channel: str = "gray"              # which colour the photo dark response is computed from
    fast_background: bool = False      # illumination blur on a 4x-reduced copy (study speed-up)
    consistency_um: float = 0.0        # >0: cross-check the two detections (new method only)
    second_pass: str = "none"          # "none" | "centroid": per-cell re-centring (new only)
    second_pass_iters: int = 2
    scan_model: str = "none"           # "none" | "traces": fit each pad across the coarse traces
    edge_sigma_um: float = 0.0         # edge softness across traces; 0 = estimate per plate
    scan_centre_from_model: bool = False   # new method: centre the height crop on the fitted pad
    model_position: str = "centroid"   # where the rebuilt pad sits: "centroid" (weighted, like the
                                       # detection; the more accurate estimator) or "fit" (fitted x0)
    width_prior_um: float = 6.0        # strength of the plate-level width prior (smaller = stronger)
    photo_smooth_um: float = 0.0       # Gaussian smoothing of the photo crop before outlining
    model_shape: str = "product"       # rebuilt pad: "product" of across x along, or "min" (both at once)
    photo_hull: bool = False           # photo: join rim fragments around the pad and fill their convex hull
    hull_below_solidity: float = 1.01  # ...only when the filled outline fills less than this of its hull
    peak_median_um: float = 0.0        # >0: median-filter the crop before reading the peak (specks)
    peak_region: str = "crop"          # where the outline threshold's peak is read: the whole crop,
                                       # or "centre" (a disk of radius 0.3 x window around the crop centre)

    @property
    def um_per_grid(self) -> float:
        return 1.0  # placeholder; real value depends on the plate's window, see Bundle


# ------------------------------------------------------------- the bundle ----

@dataclass
class Bundle:
    """Everything a run needs from one plate, independent of the recipe."""

    stem: str
    resid: np.ndarray            # detrended height map, um, shape (len(y), len(x))
    x: np.ndarray                # scan axes, um
    y: np.ndarray
    affine: np.ndarray           # (3, 2): [px_x, px_y, 1] @ affine -> (um_x, um_y)
    um_per_px: np.ndarray        # (2,)
    pitch_um: float
    window_um: float
    slot_um: np.ndarray          # (64, 2) lattice prediction in the scan
    slot_px: np.ndarray          # (64, 2) that prediction mapped into the photo
    photo_pred_px: np.ndarray    # (64, 2) photo lattice prediction
    centres: dict[str, np.ndarray]   # method -> (64, 4): photo px (x, y), scan um (x, y); NaN = not found
    status: dict[str, list[str]]     # method -> study-03 status per slot
    photo: Path

    @property
    def M_inv(self) -> np.ndarray:
        """Maps a micrometre offset in the scan frame to a pixel offset in the photo."""
        return np.linalg.inv(self.affine[:2].T)

    def px_to_um(self, px: np.ndarray) -> np.ndarray:
        px = np.atleast_2d(px)
        return np.c_[px, np.ones(len(px))] @ self.affine

    def um_to_px(self, um: np.ndarray) -> np.ndarray:
        um = np.atleast_2d(um)
        return (um - self.affine[2]) @ self.M_inv.T


def _status_table() -> dict[tuple[str, str, int, int], str]:
    with (S03 / "results" / "cells.csv").open() as f:
        return {(r["plate"], r["method"], int(r["row"]), int(r["col"])): r["status"] for r in csv.DictReader(f)}


def build_bundle(stem: str, prof_entry: dict, img_entry: dict, status_table: dict) -> Bundle:
    from Auto_OPDx.alignment.dataset import _slot_observations, fit_cross_modal, solve_photo, solve_plate

    hm = solve_plate(REPO / "data" / "opdx" / f"{stem}.OPDx", ROWS, COLS)
    ph = solve_photo(L.photo_path(stem), ROWS, COLS)
    cross = fit_cross_modal(hm, ph, COLS)
    conv = L.fit_unit_conversion(prof_entry, hm)
    pitch = float(np.mean(hm.fit.pitch))
    hm_obs = _slot_observations(hm.fit, COLS)
    ph_obs = _slot_observations(ph.fit, COLS)
    M_inv = np.linalg.inv(cross.affine[:2].T)
    t = cross.affine[2]

    slot_um = np.array([hm.fit.predict((r, c))[0] for r in range(ROWS) for c in range(COLS)])
    slot_px = (slot_um - t) @ M_inv.T
    photo_pred = np.array([ph.fit.predict((r, c))[0] for r in range(ROWS) for c in range(COLS)])
    old = np.full((64, 4), np.nan)
    new = np.full((64, 4), np.nan)
    for r in range(ROWS):
        for c in range(COLS):
            s = r * COLS + c
            o_px = L.old_centroid(img_entry, r, c)
            o_hm = L.old_centroid(prof_entry, r, c)
            if o_px is not None and o_hm is not None:
                old[s] = [*o_px, *conv.to_um(o_hm)]
            new[s] = [*ph_obs.get(s, slot_px[s]), *hm_obs.get(s, slot_um[s])]
    status = {m: [status_table.get((stem, m, r, c), "excluded_missing") for r in range(ROWS) for c in range(COLS)]
              for m in ("old", "new")}
    return Bundle(stem, hm.resid.astype(np.float32), hm.plate.x, hm.plate.y, cross.affine,
                  np.array(cross.um_per_px), pitch, L.WINDOW_FRAC * pitch, slot_um, slot_px, photo_pred,
                  {"old": old, "new": new}, status, L.photo_path(stem))


def save_bundle(b: Bundle) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / f"{b.stem}.npz", resid=b.resid, x=b.x, y=b.y, affine=b.affine, um_per_px=b.um_per_px,
        pitch_um=b.pitch_um, window_um=b.window_um, slot_um=b.slot_um, slot_px=b.slot_px,
        photo_pred_px=b.photo_pred_px, old=b.centres["old"], new=b.centres["new"],
        status_old=np.array(b.status["old"]), status_new=np.array(b.status["new"]), photo=str(b.photo))


def load_bundle(stem: str) -> Bundle:
    z = np.load(CACHE / f"{stem}.npz", allow_pickle=False)
    return Bundle(stem, z["resid"], z["x"], z["y"], z["affine"], z["um_per_px"], float(z["pitch_um"]),
                  float(z["window_um"]), z["slot_um"], z["slot_px"], z["photo_pred_px"],
                  {"old": z["old"], "new": z["new"]},
                  {"old": list(z["status_old"]), "new": list(z["status_new"])}, Path(str(z["photo"])))


def ensure_cache(stems: list[str], workers: int = 8) -> None:
    """Solve and cache every plate that is not cached yet (parallel over plates)."""
    todo = [s for s in stems if not (CACHE / f"{s}.npz").exists()]
    if todo:
        _fan_out("cache", todo, None, workers)


def _cache_one(stem: str) -> str:
    prof, img = L.load_old_metadata()
    save_bundle(build_bundle(stem, prof[stem], img[stem], _status_table()))
    return stem


def _fan_out(task: str, stems: list[str], payload, workers: int) -> list:
    """Run ``task`` over ``stems`` in separate Python processes.

    Plain subprocesses rather than a process pool: the sandbox this was developed
    in forbids the OS semaphores multiprocessing needs, and separate processes
    behave the same everywhere.
    """
    import pickle
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="iou05_"))
    job = tmp / "job.pkl"
    job.write_bytes(pickle.dumps({"task": task, "stems": stems, "payload": payload}))
    n = max(1, min(workers, len(stems)))
    env = dict(os.environ, MPLBACKEND="Agg")
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(job), str(k), str(n),
                               str(tmp / f"out_{k}.pkl")], env=env) for k in range(n)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"{sum(c != 0 for c in codes)} worker(s) failed for task {task!r}")
    results = []
    for k in range(n):
        results.extend(pickle.loads((tmp / f"out_{k}.pkl").read_bytes()))
    return results


def _worker(job_path: str, k: int, n: int, out_path: str) -> None:
    import pickle
    job = pickle.loads(Path(job_path).read_bytes())
    mine = job["stems"][k::n]
    res: list = []
    for stem in mine:
        if job["task"] == "cache":
            res.append(_cache_one(stem))
            print(f"  cached {stem}", flush=True)
        else:
            configs, keep = job["payload"]
            res.extend(evaluate_plate(stem, configs, keep.get(stem)))
    Path(out_path).write_bytes(pickle.dumps(res))


# ------------------------------------------------------------ photo fields ----

def photo_field(img_rgb: np.ndarray, channel: str = "gray", fast: bool = False) -> np.ndarray:
    """Illumination-flattened dark response from one colour channel (or the grey mean).

    ``fast`` computes the illumination on a 4x-reduced copy; the background is
    smooth (sigma 120 px), so the response is unchanged to within the noise, and
    it is ~20x quicker. The baseline is checked both ways in run_study.py.
    """
    from Auto_OPDx.alignment.rgb import dark_feature_response
    if channel == "gray":
        src = img_rgb
    else:
        src = img_rgb[..., {"red": 0, "green": 1, "blue": 2}[channel]]
    if not fast:
        return dark_feature_response(src)[0]
    g = np.asarray(src, float)
    if g.ndim == 3:
        g = g.mean(axis=2)
    h, w = g.shape
    small = cv2.resize(g, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 30.0), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(1.0 - g / np.maximum(bg, 1e-6), 0.0, None)


# --------------------------------------------------------------- cropping ----

def crop_scan(b: Bundle, centre_um: np.ndarray, cfg: Config) -> np.ndarray | None:
    """Height-map window in the scan's micrometre frame; NaN outside the scan."""
    xs, ys, vals = b.x, b.y, b.resid
    if xs[0] > xs[-1]:
        xs, vals = xs[::-1], vals[:, ::-1]
    if ys[0] > ys[-1]:
        ys, vals = ys[::-1], vals[::-1, :]
    half = b.window_um / 2
    gx = np.linspace(centre_um[0] - half, centre_um[0] + half, cfg.grid)
    gy = np.linspace(centre_um[1] - half, centre_um[1] + half, cfg.grid)
    interp = RegularGridInterpolator((ys, xs), vals, method="linear", bounds_error=False, fill_value=np.nan)
    my, mx = np.meshgrid(gy, gx, indexing="ij")
    out = interp(np.stack([my.ravel(), mx.ravel()], -1)).reshape(cfg.grid, cfg.grid)
    return out if np.isfinite(out).any() else None


def _photo_coords(b: Bundle, centre_px: np.ndarray, cfg: Config) -> np.ndarray:
    half = b.window_um / 2
    u = np.linspace(-half, half, cfg.grid)
    du, dv = np.meshgrid(u, u, indexing="xy")          # du along scan x (columns), dv along scan y (rows)
    if cfg.rotate:
        J = b.M_inv
        px = centre_px[0] + J[0, 0] * du + J[0, 1] * dv
        py = centre_px[1] + J[1, 0] * du + J[1, 1] * dv
    else:                                              # study 03: per-axis scale only
        px = centre_px[0] + du / b.um_per_px[0]
        py = centre_px[1] + dv / b.um_per_px[1]
    return np.stack([py.ravel(), px.ravel()])


def crop_photo(field: np.ndarray, b: Bundle, centre_px: np.ndarray, cfg: Config) -> np.ndarray:
    coords = _photo_coords(b, centre_px, cfg)
    return map_coordinates(field.astype(float), coords, order=1, mode="constant",
                           cval=np.nan).reshape(cfg.grid, cfg.grid)


def crop_rgb(img: np.ndarray, b: Bundle, centre_px: np.ndarray, cfg: Config) -> np.ndarray:
    coords = _photo_coords(b, centre_px, cfg)
    ch = [map_coordinates(img[..., k].astype(float), coords, order=1, mode="constant", cval=0.0)
          .reshape(cfg.grid, cfg.grid) for k in range(3)]
    return np.clip(np.stack(ch, -1), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- outlines ----

def _disk(r_px: int) -> np.ndarray:
    yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
    return (xx ** 2 + yy ** 2) <= r_px ** 2


def remove_local_plane(crop: np.ndarray, guess: np.ndarray | None) -> np.ndarray:
    """Subtract a plane fitted to the crop's background (everything well away from the pad)."""
    n = crop.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    if guess is not None and guess.any():
        bgmask = ~ndi.binary_dilation(guess, _disk(max(2, n // 24)))
    else:
        bgmask = (xx - n / 2) ** 2 + (yy - n / 2) ** 2 > (0.3 * n) ** 2
    bgmask &= np.isfinite(crop)
    if bgmask.sum() < 30:
        return crop
    A = np.c_[xx[bgmask], yy[bgmask], np.ones(bgmask.sum())]
    coef, *_ = np.linalg.lstsq(A, crop[bgmask], rcond=None)
    return crop - (coef[0] * xx + coef[1] * yy + coef[2])


def outline(crop: np.ndarray | None, level: float, floor: float, presence: float, cfg: Config,
            um_per_grid: float, hull: bool = False) -> np.ndarray | None:
    """Pad mask: threshold at ``level`` of (peak - background), clean up, keep the blob nearest the centre."""
    if crop is None or not np.isfinite(crop).any():
        return None
    bg = float(np.nanmedian(crop))
    pk_src = crop
    if cfg.peak_median_um > 0:
        r = max(1, round(cfg.peak_median_um / um_per_grid))
        pk_src = ndi.median_filter(np.nan_to_num(crop, nan=bg), size=2 * r + 1)
    if cfg.peak_region == "centre":
        # the pad being outlined, not the tallest thing in the window (debris, a neighbour's edge)
        g = crop.shape[0]
        yy, xx = np.mgrid[0:g, 0:g]
        central = (xx - g / 2) ** 2 + (yy - g / 2) ** 2 <= (0.3 * g) ** 2
        vals = pk_src[central & np.isfinite(crop)]
        # same meaning as the whole-crop 99.5th percentile: the value with as many pixels above it
        # (0.5% of the crop) -- only now they must lie near the centre
        k = max(1, int(round(0.005 * np.isfinite(crop).sum())))
        peak = float(np.sort(vals)[-k]) if vals.size >= k else bg
    else:
        peak = float(np.nanpercentile(pk_src, 99.5))
    if peak - bg < presence:
        return None
    thr = max(bg + level * (peak - bg), floor)
    binary = np.nan_to_num(crop, nan=bg) > thr
    if cfg.closing_um > 0:
        binary = ndi.binary_closing(binary, _disk(max(1, round(cfg.closing_um / um_per_grid))))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    g = crop.shape[0]
    min_area = L.MIN_AREA_FRAC * g * g
    centre = np.array([g / 2, g / 2])
    best, best_d = None, np.inf
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        d = np.linalg.norm(cents[k] - centre)
        if d < best_d:
            best, best_d = k, d
    if best is None:
        return None
    if hull and cfg.hull_below_solidity <= 1.0:
        filled = ndi.binary_fill_holes(lab == best)
        pts0 = np.column_stack(np.nonzero(filled))[:, ::-1].astype(np.int32)
        hull_area = cv2.contourArea(cv2.convexHull(pts0)) if len(pts0) >= 3 else 0.0
        if hull_area <= 0 or filled.sum() / hull_area >= cfg.hull_below_solidity:
            hull = False                       # a solid outline: leave it as drawn
    if hull:
        # a pad is a convex rectangle; a dark rim with a gap is several fragments around it.
        # join every fragment whose centre lies within a pad's half-size of the chosen one, then
        # fill their convex hull
        reach = 0.2 * g
        near = [k for k in range(1, n)
                if np.linalg.norm(cents[k] - cents[best]) < reach and stats[k, cv2.CC_STAT_AREA] >= 4]
        pts = np.column_stack(np.nonzero(np.isin(lab, near)))[:, ::-1].astype(np.int32)
        m = np.zeros_like(binary, dtype=np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(m, [cv2.convexHull(pts)], 1)
        m = ndi.binary_fill_holes(m.astype(bool) | (lab == best))
    else:
        m = ndi.binary_fill_holes(lab == best)
    if cfg.opening_um > 0:
        # after filling: a photo pad is a thin dark ring until its hole is filled
        opened = ndi.binary_opening(m, _disk(max(1, round(cfg.opening_um / um_per_grid))))
        if opened.sum() >= min_area:
            n2, lab2, _, cents2 = cv2.connectedComponentsWithStats(opened.astype(np.uint8), connectivity=8)
            k2 = 1 + int(np.argmin(np.linalg.norm(cents2[1:] - centre, axis=1)))
            m = lab2 == k2
    return m


def scan_outline(crop, cfg, um_per_grid):
    if crop is not None and cfg.local_plane:
        first = outline(crop, cfg.scan_level, L.HM_FLOOR_UM, L.HM_PRESENCE_UM, replace(cfg, closing_um=0,
                        opening_um=0), um_per_grid)
        crop = remove_local_plane(crop, first)
    return crop, outline(crop, cfg.scan_level, L.HM_FLOOR_UM, L.HM_PRESENCE_UM, cfg, um_per_grid)


def photo_outline(crop, cfg, um_per_grid):
    if crop is not None and cfg.local_plane:
        first = outline(crop, cfg.photo_level, L.PHOTO_FLOOR, L.PHOTO_PRESENCE, replace(cfg, closing_um=0,
                        opening_um=0), um_per_grid)
        crop = remove_local_plane(crop, first)
    return crop, outline(crop, cfg.photo_level, L.PHOTO_FLOOR, L.PHOTO_PRESENCE, cfg, um_per_grid, cfg.photo_hull)


# ------------------------------------------- the scan across its traces ----
#
# Along a trace the stylus samples every 2.9 um; across traces only every ~30 um.
# A ~53 um pad is therefore covered by one or two traces, and where it is only
# one, linear interpolation turns its flat top into a triangle whose half-height
# width is ~30 um. The pad's real width cannot be read from that pad alone -- the
# edge is somewhere in a 30 um gap -- but every pad on a plate is printed from
# the same pattern and the traces fall at a different offset on each one, so the
# pads whose edge happens to sit on a trace pin the plate's width down. Each pad
# is then fitted across its traces with that plate width as a weak prior. Only
# scan data is used; the photo never enters.

from scipy.special import erf as _erf
from scipy.optimize import least_squares as _lsq


def _box(x, x0, w, sigma):
    """Box of width ``w`` centred at ``x0``, blurred by a Gaussian of ``sigma``; 1 inside, 0 far outside."""
    k = np.sqrt(2.0) * max(sigma, 1e-3)
    return 0.5 * (_erf((x - x0 + w / 2) / k) - _erf((x - x0 - w / 2) / k))


def _trace_samples(b: Bundle, cx: float, cy: float):
    """Per-trace normalised height through the middle of the pad near (cx, cy), plus the along-trace profile."""
    xi = np.where(np.abs(b.x - cx) < 80.0)[0]
    yi = np.where(np.abs(b.y - cy) < 90.0)[0]
    if xi.size < 3 or yi.size < 10:
        return None
    patch = b.resid[np.ix_(yi, xi)].astype(float)
    xs, ys = b.x[xi], b.y[yi]
    far = (np.abs(xs[None, :] - cx) > 55) | (np.abs(ys[:, None] - cy) > 60)
    base = float(np.median(patch[far])) if far.any() else 0.0
    patch = patch - base
    tallest = patch.max(axis=0)
    H = float(np.percentile(tallest, 100 * (1 - 1.0 / len(tallest))))
    if H <= 0:
        return None
    inside = tallest > 0.5 * H
    if not inside.any():
        return None
    g = patch[:, inside].mean(axis=1)                 # along-trace profile through the pad
    gmax = float(g.max())
    if gmax <= 0:
        return None
    half = ys[g > 0.5 * gmax]
    yc = float(0.5 * (half.min() + half.max()))
    length = float(half.max() - half.min())
    mid = np.abs(ys - yc) < max(0.25 * length, 6.0)
    u = np.median(patch[mid], axis=0) / max(np.median(patch[mid][:, inside]), 1e-6)
    return xs, u, ys, g / gmax, yc


def fit_across_traces(xs, u, x_guess, sigma, w_prior=None, tau=6.0):
    """Least-squares (x0, w) of a blurred box to the per-trace values, optionally pulled toward ``w_prior``."""
    def resid(p):
        r = _box(xs, p[0], p[1], sigma) - u
        if w_prior is not None:
            r = np.r_[r, (p[1] - w_prior) / tau]
        return r
    w0 = w_prior if w_prior is not None else 50.0
    sol = _lsq(resid, x0=[x_guess, w0], bounds=([x_guess - 40, 5.0], [x_guess + 40, 150.0]))
    return float(sol.x[0]), float(sol.x[1])


def _fit_width(xs, u, x0, sigma, w_prior, tau):
    """Width of a blurred box at a fixed centre, pulled toward the plate width."""
    def resid(p):
        return np.r_[_box(xs, x0, p[0], sigma) - u, (p[0] - w_prior) / tau]
    return float(_lsq(resid, x0=[w_prior], bounds=([5.0], [150.0])).x[0])


def plate_trace_model(b: Bundle) -> tuple[float, float]:
    """Plate-level pad width and edge softness across traces, from all pads at once.

    Every pad on a plate is printed from the same pattern, so one width and one
    edge softness are shared; each pad keeps its own centre. The traces fall at a
    different offset on each pad, so together they sample the edge profile
    finely even though each pad alone does not. Scan data only.
    """
    pads = []
    for slot in range(64):
        if b.status["new"][slot] == "excluded_missing":
            continue
        cx, cy = b.centres["new"][slot][2:]
        smp = _trace_samples(b, cx, cy)
        if smp is not None and (smp[1] > 0.85).sum() >= 1:
            pads.append((smp[0], smp[1], cx))
    if len(pads) < 8:
        return 53.0, 2.0

    def resid(p):
        sigma, w = p[0], p[1]
        return np.concatenate([_box(xs, p[2 + i], w, sigma) - u for i, (xs, u, _) in enumerate(pads)])

    x0 = [2.0, 53.0] + [cx for _, _, cx in pads]
    lo = [0.3, 20.0] + [cx - 40 for _, _, cx in pads]
    hi = [15.0, 120.0] + [cx + 40 for _, _, cx in pads]
    sol = _lsq(resid, x0=x0, bounds=(lo, hi))
    return float(sol.x[1]), float(sol.x[0])


def scan_model_crop(b: Bundle, centre_um: np.ndarray, smask: np.ndarray, cfg: Config, upg: float,
                    plate: tuple[float, float]):
    """Rebuild the pad found in ``smask`` from its native traces, rasterised onto the crop grid.

    Returns (field, fitted centre in um) or None.
    """
    w_plate, sigma = plate
    if cfg.edge_sigma_um > 0:              # >0 overrides the plate estimate (sensitivity runs)
        sigma = cfg.edge_sigma_um
    cx, cy = centre_um + centroid_um(smask, upg)
    smp = _trace_samples(b, cx, cy)
    if smp is None:
        return None
    xs, u, ys, gnorm, yc = smp
    if cfg.model_position == "detection":
        # the plate detection nearest the pad found in this crop: the centre estimator that beat
        # every alternative against the scan's own grid (study 05, section 4)
        det = b.centres["new"][:, 2:]
        d = np.linalg.norm(det - np.array([cx, cy]), axis=1)
        k = int(np.argmin(d))
        x0 = float(det[k, 0]) if d[k] < 30.0 else float(cx)
        w = _fit_width(xs, u, x0, sigma, w_plate, cfg.width_prior_um)
    elif cfg.model_position == "centroid":
        # position from a height-weighted centroid across the traces -- the same kind of estimator
        # as the plate detection, which beat every alternative against the grid -- width from the fit
        inside = u > 0.3
        x0 = float(np.sum(xs[inside] * u[inside]) / np.sum(u[inside])) if inside.any() else cx
        w = _fit_width(xs, u, x0, sigma, w_plate, cfg.width_prior_um)
    else:
        x0, w = fit_across_traces(xs, u, cx, sigma, w_plate, cfg.width_prior_um)
    half = b.window_um / 2
    gx = np.linspace(centre_um[0] - half, centre_um[0] + half, cfg.grid)
    gy = np.linspace(centre_um[1] - half, centre_um[1] + half, cfg.grid)
    prof_y = np.interp(gy, ys, gnorm, left=0.0, right=0.0)
    # pad edges follow the grid's row direction, which is tilted by the lattice rotation
    tilt = _row_tilt(b)
    X = gx[None, :] - (x0 + tilt * (gy[:, None] - yc))
    across = _box(X, 0.0, w, sigma)
    if cfg.model_shape == "min":
        # the pad's across-trace edges are sharp and its along-trace ends are smeared by the stylus;
        # a point is inside where it is inside on both axes, so the width holds along the whole length
        return 5.0 * np.minimum(across, prof_y[:, None]), np.array([x0, yc])
    return 5.0 * across * prof_y[:, None], np.array([x0, yc])   # nominal 5 um height for outline()


def _row_tilt(b: Bundle) -> float:
    """dx/dy of the lattice's row direction in the scan (how pad edges lean)."""
    s = b.slot_um.reshape(ROWS, COLS, 2)
    d = np.concatenate([s[1:4] - s[0:3], s[5:8] - s[4:7]], axis=0).reshape(-1, 2)
    return float(np.median(d[:, 0]) / np.median(d[:, 1]))


# ----------------------------------------------------------------- metrics ----

def centroid_um(m: np.ndarray, um_per_grid: float) -> np.ndarray:
    yy, xx = np.nonzero(m)
    g = m.shape[0]
    return (np.array([xx.mean(), yy.mean()]) - (g - 1) / 2) * um_per_grid


def iou(a, b) -> float:
    return L.iou(a, b)


def centred_iou(a, b) -> float:
    return L.centred_iou(a, b)


def edges_um(m: np.ndarray, um_per_grid: float) -> np.ndarray:
    ys, xs = np.nonzero(m)
    return np.array([np.percentile(xs, 2), np.percentile(xs, 98),
                     np.percentile(ys, 2), np.percentile(ys, 98)]) * um_per_grid


# ---------------------------------------------------------------- one cell ----

@dataclass
class CellOut:
    plate: str
    row: int
    col: int
    method: str
    config: str
    status: str
    iou: float = 0.0
    ceiling: float = 0.0
    offset_um: float = float("nan")
    photo_area_um2: float = float("nan")
    scan_area_um2: float = float("nan")
    edge_diff_um: tuple = (np.nan, np.nan, np.nan, np.nan)    # scan - photo: left, right, top, bottom
    photo_edges_um: tuple = (np.nan,) * 4                     # left, right, top, bottom in the crop
    scan_edges_um: tuple = (np.nan,) * 4
    scan_centre_um: tuple = (np.nan, np.nan)                  # absolute, scan frame
    photo_crop_px: tuple = (np.nan, np.nan)                   # where the photo crop was finally cut
    scan_crop_um: tuple = (np.nan, np.nan)                    # where the height crop was finally cut
    moved_by_check: bool = False
    second_pass_um: float = 0.0
    arrays: dict = field(default_factory=dict, repr=False)


def consistency_fix(b: Bundle, slot: int, p_px: np.ndarray, h_um: np.ndarray, tol_um: float):
    """If the two detections disagree with the plate registration, replace the one that is off its own grid.

    The registration maps every photo detection into the scan with ~8 um RMSE,
    so a pair more than ``tol_um`` apart means one detection latched onto
    something else (dust, a fragment, a neighbouring blob). Whichever detection
    sits further from its own lattice prediction is replaced by the other one
    mapped through the registration.
    """
    mapped = b.px_to_um(p_px)[0]
    if np.linalg.norm(mapped - h_um) <= tol_um:
        return p_px, h_um, False
    r_scan = np.linalg.norm(h_um - b.slot_um[slot])
    r_photo = np.linalg.norm(p_px - b.photo_pred_px[slot]) * float(np.sqrt(np.prod(b.um_per_px)))
    if r_scan >= r_photo:
        return p_px, mapped, True
    return b.um_to_px(h_um)[0], h_um, True


def _smooth(crop: np.ndarray, cfg: Config, upg: float) -> np.ndarray:
    if cfg.photo_smooth_um <= 0 or crop is None:
        return crop
    filled = np.where(np.isfinite(crop), crop, np.nanmedian(crop))
    out = ndi.gaussian_filter(filled, cfg.photo_smooth_um / upg)
    return np.where(np.isfinite(crop), out, np.nan)


def evaluate_cell(b: Bundle, field: np.ndarray, slot: int, method: str, cfg: Config,
                  img: np.ndarray | None = None, w_prior: float | None = None) -> CellOut:
    r, c = divmod(slot, COLS)
    status = b.status[method][slot]
    out = CellOut(b.stem, r, c, method, cfg.name, status)
    if status == "excluded_missing":
        return out
    cen = b.centres[method][slot]
    if not np.isfinite(cen).all():
        out.status = "not_found"
        return out
    p_px, h_um = cen[:2].copy(), cen[2:].copy()
    if method == "new" and cfg.consistency_um > 0:
        p_px, h_um, out.moved_by_check = consistency_fix(b, slot, p_px, h_um, cfg.consistency_um)
    upg = b.window_um / (cfg.grid - 1)

    scrop, smask = scan_outline(crop_scan(b, h_um, cfg), cfg, upg)
    if cfg.scan_model == "traces" and smask is not None:
        plate = w_prior if w_prior is not None else (53.0, 2.0)
        fit = scan_model_crop(b, h_um, smask, cfg, upg, plate)
        if fit is not None and method == "new" and cfg.scan_centre_from_model:
            # re-centre the height crop on the fitted pad and rebuild it there
            h_um = fit[1]
            scrop, smask = scan_outline(crop_scan(b, h_um, cfg), cfg, upg)
            fit = scan_model_crop(b, h_um, smask, cfg, upg, plate) if smask is not None else None
        if fit is not None:
            m2 = outline(fit[0], cfg.scan_level, L.HM_FLOOR_UM, L.HM_PRESENCE_UM, cfg, upg)
            if m2 is not None:
                scrop, smask = fit[0], m2
    pcrop, pmask = photo_outline(_smooth(crop_photo(field, b, p_px, cfg), cfg, upg), cfg, upg)

    if method == "new" and cfg.second_pass == "centroid" and smask is not None and pmask is not None:
        J = b.M_inv if cfg.rotate else np.diag(1.0 / b.um_per_px)
        moved = np.zeros(2)
        for _ in range(cfg.second_pass_iters):
            d = centroid_um(pmask, upg) - centroid_um(smask, upg)     # photo pad minus scan pad, um
            if np.linalg.norm(d) < 0.25 * upg:
                break
            step = J @ d
            p_px = p_px + step
            moved += d
            pcrop2, pmask2 = photo_outline(_smooth(crop_photo(field, b, p_px, cfg), cfg, upg), cfg, upg)
            if pmask2 is None:
                p_px = p_px - step
                break
            pcrop, pmask = pcrop2, pmask2
        out.second_pass_um = float(np.linalg.norm(moved))

    out.photo_crop_px, out.scan_crop_um = tuple(p_px), tuple(h_um)
    if smask is None or pmask is None:
        out.status = "no_pad_in_crop"
    else:
        out.status = "ok"
        out.iou = iou(pmask, smask)
        out.ceiling = centred_iou(pmask, smask)
        out.offset_um = float(np.linalg.norm(centroid_um(pmask, upg) - centroid_um(smask, upg)))
        out.photo_area_um2 = float(pmask.sum() * upg ** 2)
        out.scan_area_um2 = float(smask.sum() * upg ** 2)
        pe, se = edges_um(pmask, upg), edges_um(smask, upg)
        out.edge_diff_um = tuple(se - pe)
        out.photo_edges_um, out.scan_edges_um = tuple(pe), tuple(se)
        out.scan_centre_um = tuple(h_um + centroid_um(smask, upg))
    if img is not None:
        out.arrays = {"photo_crop": pcrop, "scan_crop": scrop, "photo_mask": pmask, "scan_mask": smask,
                      "rgb": crop_rgb(img, b, p_px, cfg), "upg": upg}
    return out


def evaluate_plate(stem: str, configs: list[Config], keep: set | None = None) -> list[CellOut]:
    """All cells x both methods x every config for one plate. ``keep`` = {(slot, method, config)} to keep arrays."""
    from Auto_OPDx.alignment.rgb import load_rgb
    b = load_bundle(stem)
    img = load_rgb(b.photo)
    fields: dict[tuple[str, bool], np.ndarray] = {}
    outs: list[CellOut] = []
    priors: dict = {}
    for cfg in configs:
        key = (cfg.channel, cfg.fast_background)
        if key not in fields:
            fields[key] = photo_field(img, cfg.channel, cfg.fast_background)
        wp = None
        if cfg.scan_model == "traces":
            if "plate" not in priors:
                priors["plate"] = plate_trace_model(b)
            wp = priors["plate"]
        for slot in range(64):
            for method in ("old", "new"):
                want = keep is not None and (slot, method, cfg.name) in keep
                outs.append(evaluate_cell(b, fields[key], slot, method, cfg, img if want else None, wp))
    return outs


def run(stems: list[str], configs: list[Config], workers: int = 8, keep: dict | None = None) -> list[CellOut]:
    return _fan_out("evaluate", stems, (configs, keep or {}), workers)


def summarise(outs: list[CellOut], config: str, method: str) -> dict:
    mine = [o for o in outs if o.config == config and o.method == method and o.status != "excluded_missing"]
    v = np.array([o.iou for o in mine])
    ok = [o for o in mine if o.status == "ok"]
    return {
        "config": config, "method": method, "cells": len(mine),
        "mean_iou": float(v.mean()), "median_iou": float(np.median(v)),
        "frac_below_0p5": float(np.mean(v < 0.5)), "n_zero": int(np.sum(v == 0)),
        "n_failures": sum(o.status != "ok" for o in mine),
        "mean_ceiling": float(np.mean([o.ceiling for o in ok])) if ok else np.nan,
        "median_offset_um": float(np.median([o.offset_um for o in ok])) if ok else np.nan,
        "moved_by_check": sum(o.moved_by_check for o in mine),
    }


if __name__ == "__main__" and len(sys.argv) == 6 and sys.argv[1] == "--worker":
    # import under the real module name so pickled results unpickle in the parent
    sys.path.insert(0, str(HERE))
    import lib05
    lib05._worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
