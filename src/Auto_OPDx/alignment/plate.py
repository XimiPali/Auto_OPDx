"""Loading, unit normalisation, detrending and resampling of OPDx height maps."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# DektakLoad returns x/y already in micrometres but z in metres, which the
# original GUI papered over with scattered 1e6 factors. Normalise once, here.
_METRE_SPAN_CUTOFF = 1e-1


@dataclass
class Plate:
    """One OPDx scan, normalised to micrometres throughout.

    Attributes
    ----------
    name : str
        File stem, e.g. ``10-01-2025_P1``.
    x, y : ndarray
        1-D axes in um. ``len(y), len(x) == z.shape``.
    z : ndarray
        Height in um, shape ``(len(y), len(x))``.
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def dx(self) -> float:
        """Median sample spacing on the x axis, um."""
        return float(abs(np.median(np.diff(self.x)))) if self.x.size > 1 else float("nan")

    @property
    def dy(self) -> float:
        """Median sample spacing on the y axis, um."""
        return float(abs(np.median(np.diff(self.y)))) if self.y.size > 1 else float("nan")

    @property
    def anisotropy(self) -> float:
        lo, hi = sorted((self.dx, self.dy))
        return float(hi / lo) if lo > 0 else float("nan")

    @property
    def extent(self) -> list[float]:
        """matplotlib ``extent`` for imshow with ``origin='lower'``."""
        return [float(self.x.min()), float(self.x.max()),
                float(self.y.min()), float(self.y.max())]

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        xm, ym = np.meshgrid(self.x, self.y)
        return xm, ym


def _to_um(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` in micrometres, converting from metres when needed."""
    arr = np.asarray(arr, float)
    span = float(np.ptp(arr))
    if 0.0 < span < _METRE_SPAN_CUTOFF:
        return arr * 1e6
    return arr


class _NullPlt:
    """Absorbs any attribute access and returns a no-op callable."""

    def __getattr__(self, name):  # noqa: D105
        return lambda *args, **kwargs: None


@contextmanager
def _mute_reader_plot():
    """Stop ``OPDx_read`` from drawing a figure on every load.

    ``OPDx_read/reader.py`` calls ``plt.imshow`` unconditionally inside
    ``get_data_2D``. Under the default macOS backend that aborts the process in
    a headless run, and even under Agg it leaks one figure per plate. Swapping
    the module's ``plt`` reference is surgical: it avoids touching the global
    matplotlib backend, so a notebook's inline plotting is left alone.
    """
    from OPDx_read import reader as _reader

    real = getattr(_reader, "plt", None)
    _reader.plt = _NullPlt()
    try:
        yield
    finally:
        if real is not None:
            _reader.plt = real


def load_plate(path: str | Path) -> Plate:
    """Read an ``.OPDx`` file and return a :class:`Plate` in micrometres."""
    from OPDx_read.reader import DektakLoad

    path = Path(path)
    with _mute_reader_plot():
        x, y, z = DektakLoad(str(path)).get_data_2D()
    x = _to_um(x)
    y = _to_um(y)
    z = _to_um(z)

    # Axes occasionally arrive as full meshes rather than 1-D vectors.
    if x.ndim == 2:
        x = x[0, :]
    if y.ndim == 2:
        y = y[:, 0]

    if z.shape != (y.size, x.size):
        if z.shape == (x.size, y.size):
            z = z.T
        else:
            raise ValueError(
                f"{path.name}: z shape {z.shape} matches neither "
                f"{(y.size, x.size)} nor its transpose"
            )
    return Plate(name=path.stem, x=x, y=y, z=z)


def fit_background_plane(
    plate: Plate, percentile: float = 45.0
) -> tuple[np.ndarray, float]:
    """Least-squares plane through the background of ``plate``.

    Points at or below ``percentile`` of the height distribution are treated as
    substrate, so the raised features do not drag the plane upward.

    Returns
    -------
    coef : ndarray, shape (2,)
        Gradient ``(dz/dx, dz/dy)`` in um per um.
    intercept : float
        Height in um at the coordinate origin.
    """
    xm, ym = plate.mesh()
    bg = plate.z <= np.percentile(plate.z, percentile)
    n = int(bg.sum())
    if n < 3:
        raise ValueError(f"{plate.name}: only {n} background points; need >= 3")
    A = np.c_[xm[bg], ym[bg], np.ones(n)]
    coeffs, *_ = np.linalg.lstsq(A, plate.z[bg], rcond=None)
    return coeffs[:2], float(coeffs[2])


def plane_surface(plate: Plate, coef: np.ndarray, intercept: float) -> np.ndarray:
    """Evaluate a fitted plane over the plate's full grid."""
    xm, ym = plate.mesh()
    return intercept + coef[0] * xm + coef[1] * ym


def detrend(
    plate: Plate, percentile: float = 45.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Subtract the fitted background plane.

    This is the step that removes the position confound: the raw label
    correlates with row index at r = +0.98, the detrended one should not.

    Returns
    -------
    resid : ndarray
        Plane-relative height in um, same shape as ``plate.z``.
    info : dict
        ``tilt_pp_um`` (plane peak-to-peak), ``gradient_um_per_mm`` (magnitude
        of the tilt slope), ``raw_pp_um``, ``resid_pp_um`` and
        ``tilt_fraction`` (tilt as a fraction of the raw range).
    """
    coef, intercept = fit_background_plane(plate, percentile)
    plane = plane_surface(plate, coef, intercept)
    resid = plate.z - plane

    raw_pp = float(np.ptp(plate.z))
    tilt_pp = float(np.ptp(plane))
    info = {
        "tilt_pp_um": tilt_pp,
        "gradient_um_per_mm": float(np.hypot(*coef) * 1000.0),
        "raw_pp_um": raw_pp,
        "resid_pp_um": float(np.ptp(resid)),
        "tilt_fraction": tilt_pp / raw_pp if raw_pp else float("nan"),
    }
    return resid, info


def resample_isotropic(
    plate: Plate,
    values: np.ndarray | None = None,
    pitch_um: float = 2.878,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample onto a square grid of ``pitch_um`` per pixel.

    The native sampling is ~10.6x anisotropic (median 30.3 um on x, 2.88 um on
    y). Rotating an array that anisotropic while treating its cells as square
    necessarily turns squares into parallelograms -- which is why the earlier
    attempt at rotation failed. Resampling into physical space first makes
    rotation well defined.

    Interpolation cannot create information that was never sampled: the coarse
    axis stays genuinely coarse. What this buys is a coordinate system in which
    geometry is correct.

    Parameters
    ----------
    values
        Array to resample; defaults to ``plate.z``. Pass the detrended residual
        to resample plane-relative heights.
    pitch_um
        Output pixel size. The default matches the native fine axis so no real
        resolution is discarded.

    Returns
    -------
    x_new, y_new : ndarray
        1-D axes of the resampled grid, um.
    out : ndarray
        Resampled values, shape ``(y_new.size, x_new.size)``.
    """
    if values is None:
        values = plate.z
    values = np.asarray(values, float)
    if values.shape != plate.z.shape:
        raise ValueError(
            f"values shape {values.shape} != plate shape {plate.z.shape}"
        )
    if pitch_um <= 0:
        raise ValueError("pitch_um must be positive")

    # RegularGridInterpolator needs strictly increasing axes.
    xs, ys = plate.x, plate.y
    flip_x, flip_y = xs[0] > xs[-1], ys[0] > ys[-1]
    if flip_x:
        xs, values = xs[::-1], values[:, ::-1]
    if flip_y:
        ys, values = ys[::-1], values[::-1, :]

    interp = RegularGridInterpolator(
        (ys, xs), values, method=method, bounds_error=False, fill_value=np.nan
    )
    x_new = np.arange(xs[0], xs[-1] + pitch_um * 0.5, pitch_um)
    y_new = np.arange(ys[0], ys[-1] + pitch_um * 0.5, pitch_um)
    gy, gx = np.meshgrid(y_new, x_new, indexing="ij")
    out = interp(np.stack([gy.ravel(), gx.ravel()], axis=-1)).reshape(gy.shape)
    return x_new, y_new, out
