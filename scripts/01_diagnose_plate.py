"""Diagnostic pass over a single OPDx plate.

Answers four questions before we build anything on top of this data:

  Q1  What is the real array shape, and the physical sample spacing on each axis?
  Q2  How large is the substrate tilt, in um, next to the 4-39 um signal range?
      (If tilt >> signal, labels built from raw z are position-contaminated.)
  Q3  How many of the 64 features does the existing pipeline find, and how many
      pixels across is a feature on each axis?
  Q4  What does it look like? Writes a 4-panel figure to studies/01-label-alignment/figures/.

Usage:
    uv run python scripts/01_diagnose_plate.py               # first file in data/opdx
    uv run python scripts/01_diagnose_plate.py data/opdx/X.OPDx
"""

import os

# Must precede any matplotlib import: the macosx backend aborts when there is
# no GUI session, and the abort discards buffered stdout along with it.
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Reference numbers from John's slides, for comparison against what we measure.
SLIDE_SHAPE = (901, 77)
SLIDE_SPAN_UM = 2500.0  # 2.5 mm x 2.5 mm scan area
SLIDE_GRID = 8
SIGNAL_RANGE_UM = (4.0, 39.0)  # true cell height range, per results deck


def find_plate(argv):
    if len(argv) > 1:
        p = Path(argv[1])
        if not p.exists():
            sys.exit(f"No such file: {p}")
        return p
    candidates = sorted(
        q for q in (REPO / "data" / "opdx").glob("*") if q.suffix.lower() == ".opdx"
    )
    if not candidates:
        sys.exit(
            "No .opdx files found in data/opdx/.\n"
            "Download the Drive `data` folder and unzip it into "
            f"{REPO / 'data'}"
        )
    print(f"Found {len(candidates)} plate(s); using the first.\n")
    return candidates[0]


def to_um(arr, name):
    """OPDx values arrive in metres. Detect and normalise to micrometres."""
    span = float(np.ptp(arr))
    if span == 0:
        return np.asarray(arr, float), "unknown (zero span)"
    # A 2.5 mm plate is 2.5e-3 in metres, 2.5e3 in micrometres.
    if span < 1e-1:
        return np.asarray(arr, float) * 1e6, "metres -> um"
    return np.asarray(arr, float), "already um"


def fit_plane(x_mesh, y_mesh, z, mask=None):
    """Least-squares plane through (optionally masked) points. Returns (a, b, c)."""
    if mask is None:
        mask = np.ones(z.shape, bool)
    A = np.c_[x_mesh[mask], y_mesh[mask], np.ones(int(mask.sum()))]
    coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
    return coeffs


def main():
    path = find_plate(sys.argv)
    print("=" * 72)
    print(f"PLATE: {path.name}  ({path.stat().st_size / 1024:.0f} KiB on disk)")
    print("=" * 72)

    from OPDx_read.reader import DektakLoad

    x, y, z = DektakLoad(str(path)).get_data_2D()
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z, float)

    # ---- Q1: geometry -----------------------------------------------------
    print("\n--- Q1  GEOMETRY ---")
    x_um, xnote = to_um(x, "x")
    y_um, ynote = to_um(y, "y")
    z_um, znote = to_um(z, "z")

    print(f"  z.shape            {z.shape}      (slides say {SLIDE_SHAPE})")
    print(f"  z.dtype            {z.dtype}")
    print(f"  unit conversion    x: {xnote} | y: {ynote} | z: {znote}")

    # x and y may come back as 1-D axes or as full meshes.
    x_ax = x_um if x_um.ndim == 1 else x_um[0, :]
    y_ax = y_um if y_um.ndim == 1 else y_um[:, 0]

    dx = float(np.median(np.diff(x_ax))) if x_ax.size > 1 else float("nan")
    dy = float(np.median(np.diff(y_ax))) if y_ax.size > 1 else float("nan")
    span_x, span_y = float(np.ptp(x_ax)), float(np.ptp(y_ax))

    print(f"  x span             {span_x:9.1f} um over {x_ax.size:4d} samples"
          f"  -> {dx:7.3f} um/sample")
    print(f"  y span             {span_y:9.1f} um over {y_ax.size:4d} samples"
          f"  -> {dy:7.3f} um/sample")

    if np.isfinite(dx) and np.isfinite(dy) and min(abs(dx), abs(dy)) > 0:
        aniso = max(abs(dx), abs(dy)) / min(abs(dx), abs(dy))
        print(f"  ANISOTROPY         {aniso:.2f}x   <-- why resampling is required")

    pitch_x = span_x / SLIDE_GRID
    pitch_y = span_y / SLIDE_GRID
    print(f"  cell pitch         {pitch_x:.1f} um (x) x {pitch_y:.1f} um (y)"
          f"   [{SLIDE_GRID}x{SLIDE_GRID} grid]")
    if np.isfinite(dx) and np.isfinite(dy):
        print(f"  samples per cell   {pitch_x / abs(dx):.1f} (x) x "
              f"{pitch_y / abs(dy):.1f} (y)")

    # ---- Q2: tilt vs signal ----------------------------------------------
    print("\n--- Q2  SUBSTRATE TILT vs SIGNAL ---")
    x_mesh, y_mesh = np.meshgrid(x_ax, y_ax)
    if x_mesh.shape != z_um.shape:
        print(f"  ! mesh {x_mesh.shape} != z {z_um.shape}; transposing mesh")
        x_mesh, y_mesh = x_mesh.T, y_mesh.T

    # Fit to the lower half of the height distribution, i.e. background only,
    # so the features themselves do not drag the plane upward.
    bg = z_um <= np.percentile(z_um, 45)
    a, b, c = fit_plane(x_mesh, y_mesh, z_um, bg)
    plane = a * x_mesh + b * y_mesh + c

    tilt_pp = float(np.ptp(plane))
    z_pp = float(np.ptp(z_um))
    resid = z_um - plane

    print(f"  raw z range        {z_um.min():8.2f} .. {z_um.max():8.2f} um"
          f"   (peak-to-peak {z_pp:.2f})")
    print(f"  fitted plane       z = {a:+.4g}*x {b:+.4g}*y {c:+.4g}")
    print(f"  TILT across plate  {tilt_pp:8.2f} um peak-to-peak")
    print(f"  signal range       {SIGNAL_RANGE_UM[0]:.0f} .. "
          f"{SIGNAL_RANGE_UM[1]:.0f} um  (per results deck)")
    ratio = tilt_pp / (SIGNAL_RANGE_UM[1] - SIGNAL_RANGE_UM[0])
    print(f"  tilt / signal span {ratio:8.2f}x")
    if ratio > 0.25:
        print("  >> VERDICT: tilt is a material fraction of the signal.")
        print("     Labels built from RAW z carry a position-dependent offset.")
        print("     Detrend before constructing labels.  [hypothesis H1 SUPPORTED]")
    else:
        print("  >> VERDICT: tilt is small next to the signal. H1 not the "
              "main issue here.")
    print(f"  detrended residual {resid.min():8.2f} .. {resid.max():8.2f} um")

    # ---- Q3: feature detection -------------------------------------------
    print("\n--- Q3  FEATURE DETECTION (existing pipeline) ---")
    import cv2

    from Auto_OPDx.adaptive_thresholds import compute_adaptive_thresholds
    from Auto_OPDx.filter import filter_components

    # compute_adaptive_thresholds expects metres, matching DektakLoad output.
    z_m, xm_m, ym_m = z_um * 1e-6, x_mesh * 1e-6, y_mesh * 1e-6
    try:
        sug = compute_adaptive_thresholds(
            z_m, xm_m, ym_m, rows=SLIDE_GRID, cols=SLIDE_GRID
        )
        print(f"  auto thresholds    bg={sug['background_percentile']:.1f}%  "
              f"dist={sug['feature_distance_um']:.2f}um  "
              f"min_area={sug['min_area_px']}px")
        feat_dist, min_area = sug["feature_distance_um"], sug["min_area_px"]
    except Exception as e:
        print(f"  ! adaptive thresholds failed ({e}); using defaults")
        feat_dist, min_area = 2.0, 15

    feature_mask = np.abs(resid) >= feat_dist
    num, labels, stats, cents = cv2.connectedComponentsWithStats(
        feature_mask.astype(np.uint8) * 255, 4, cv2.CV_32S
    )
    n2, _, fstats, _ = filter_components(
        num, labels, stats, cents, min_area_threshold=min_area
    )
    found = n2 - 1
    expected = SLIDE_GRID * SLIDE_GRID
    print(f"  raw components     {num - 1}")
    print(f"  after area filter  {found}   (expected {expected})")
    if found:
        w = fstats[1:, cv2.CC_STAT_WIDTH].astype(float)
        h = fstats[1:, cv2.CC_STAT_HEIGHT].astype(float)
        area = fstats[1:, cv2.CC_STAT_AREA].astype(float)
        print(f"  feature width      median {np.median(w):5.1f} px "
              f"= {np.median(w) * abs(dx):7.1f} um")
        print(f"  feature height     median {np.median(h):5.1f} px "
              f"= {np.median(h) * abs(dy):7.1f} um")
        print(f"  feature area       median {np.median(area):.0f} px")
        narrow = min(np.median(w), np.median(h))
        print(f"\n  >> Coarse axis gives ~{narrow:.0f} px across a feature.")
        print(f"     Single-feature centroid precision is therefore about "
              f"+/-{max(abs(dx), abs(dy)) / 2:.0f} um.")
        print("     Fit ONE lattice to all 64 features instead: error falls "
              f"by ~sqrt(64) = 8x -> ~"
              f"{max(abs(dx), abs(dy)) / 2 / 8:.1f} um.  [motivates H2 fix]")

    # ---- Q4: figure -------------------------------------------------------
    print("\n--- Q4  FIGURE ---")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    outdir = REPO / "studies/01-label-alignment" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    extent = [x_ax.min(), x_ax.max(), y_ax.min(), y_ax.max()]
    # Equal physical aspect so the anisotropy is visible rather than hidden.
    kw = dict(origin="lower", extent=extent, aspect="equal")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    for ax, (img, title) in zip(axes, [
        (z_um, f"1. Raw z\n{z_um.min():.1f}..{z_um.max():.1f} um"),
        (plane, f"2. Fitted plane\ntilt {tilt_pp:.1f} um p-p"),
        (resid, f"3. Detrended\n{resid.min():.1f}..{resid.max():.1f} um"),
        (feature_mask.astype(float), f"4. Feature mask\n{found} found / {expected}"),
    ]):
        im = ax.imshow(img, cmap="viridis", **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (um)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("y (um)", fontsize=8)
    fig.suptitle(f"{path.name}  —  shape {z.shape}, "
                 f"{abs(dx):.2f} x {abs(dy):.2f} um/sample", fontsize=11)
    fig.tight_layout()
    out = outdir / f"diag_{path.stem}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  wrote {out.relative_to(REPO)}")
    print("\ndone.")


if __name__ == "__main__":
    main()
