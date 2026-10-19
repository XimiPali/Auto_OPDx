"""``align-plates`` — point it at two folders, get aligned per-cell data out.

This is the tool form of the alignment pipeline: no notebook, no editing paths,
nothing to run interactively. Give it a folder of photos and a folder of
profilometer scans that share a naming convention, and it writes one ``.npz``
per plate containing the matched photo crop, height crop and true height for
every cell, plus a summary table and a QC image per plate so a lab user can
check the result without reading any code.

Usage
-----
::

    align-plates --images data/rgb/img --labels data/opdx --out aligned/

    # dry run: show what would be paired, process nothing
    align-plates --images ... --labels ... --out ... --dry-run

    # resume a partial run, skipping plates already written
    align-plates --images ... --labels ... --out ... --skip-existing

Output layout
-------------
::

    out/
      plates/<stem>.npz     per-plate arrays (see `_save_plate` for the keys)
      qc/<stem>.png         8x8 montage of the photo crops, labelled
      summary.csv           one row per plate: cell count, errors, geometry
      cells.csv             one row per cell: height, grid position, overhang
      manifest.json         the exact settings this run used

Exit status is 0 if at least one plate succeeded, 1 if none did.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Must precede any matplotlib import: the QC renderer is headless, and the
# default macOS backend aborts without a window server.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

PHOTO_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SCAN_EXT = (".opdx",)

# Fraction of the tighter grid pitch used for the photo window. The photo's two
# pitches differ (~250 px by column, ~182 px by row), so sizing off the smaller
# one keeps a square window from reaching into the neighbouring cell.
_WINDOW_FRACTION = 0.95


@contextlib.contextmanager
def _watchdog(seconds: int):
    """Abort the current plate if it takes longer than ``seconds``.

    The OPDx reader can loop forever on a malformed file instead of raising.
    The size guard in ``load_plate`` catches the common empty/truncated case,
    but a file damaged in the middle could still stall, and an unattended batch
    must not hang on one bad input. ``SIGALRM`` is POSIX-only and only works on
    the main thread; where it is unavailable this degrades to no timeout, which
    is the current behaviour anyway.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(_signum, _frame):
        raise TimeoutError(
            f"exceeded --timeout of {seconds}s (likely a malformed scan file)"
        )

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class PlateResult:
    stem: str
    ok: bool
    n_cells: int = 0
    n_complete: int = 0
    scan_rmse_um: float = float("nan")
    photo_rmse_px: float = float("nan")
    cross_rmse_um: float = float("nan")
    cross_max_um: float = float("nan")
    pitch_um: float = float("nan")
    tilt_pp_um: float = float("nan")
    tilt_fraction: float = float("nan")
    um_per_px_x: float = float("nan")
    um_per_px_y: float = float("nan")
    photo_window_px: int = 0
    seconds: float = 0.0
    error: str = ""


def find_pairs(
    images: Path, labels: Path, pattern: str = "{stem}"
) -> tuple[list[tuple[str, Path, Path]], list[str], list[str]]:
    """Match photos to scans by filename stem.

    ``pattern`` lets a photo stem be derived from a scan stem when the two
    folders disagree -- e.g. ``--pattern '{stem}_photo'`` pairs ``P1.OPDx``
    with ``P1_photo.jpg``. The default assumes identical stems.

    Returns ``(pairs, scans_without_photo, photos_without_scan)``.
    """
    photos = {
        p.stem: p for p in sorted(images.iterdir())
        if p.is_file() and p.suffix.lower() in PHOTO_EXT
    }
    scans = {
        p.stem: p for p in sorted(labels.iterdir())
        if p.is_file() and p.suffix.lower() in SCAN_EXT
    }

    pairs, orphan_scans = [], []
    matched_photos = set()
    for stem, scan in scans.items():
        want = pattern.format(stem=stem)
        photo = photos.get(want)
        if photo is None:
            orphan_scans.append(stem)
        else:
            pairs.append((stem, photo, scan))
            matched_photos.add(want)
    orphan_photos = sorted(set(photos) - matched_photos)
    return pairs, orphan_scans, orphan_photos


def _photo_window(photo_fit, override: int | None) -> int:
    """Square photo window in pixels, sized off the tighter grid pitch."""
    if override:
        return int(override)
    pitch = min(photo_fit.pitch)
    if not np.isfinite(pitch) or pitch <= 0:
        return 160
    return max(16, int(round(pitch * _WINDOW_FRACTION)))


def _save_plate(path: Path, stem: str, cells, hm, photo, cross, window_px: int,
                centre: str = "detected") -> int:
    """Write one plate's arrays. Returns the number of cells stored."""
    keep = [c for c in cells if c.rgb is not None and "h_max" in c.labels]
    if not keep:
        return 0
    np.savez_compressed(
        path,
        # --- where each crop was centred -----------------------------------
        # 1 = on the pad the detector found, 0 = on the lattice prediction
        # (no detection was assigned to that slot). See build_cells(centre=).
        hm_centre_detected=np.array(
            [c.labels.get("hm_centre_detected", 0.0) for c in keep], np.int8),
        photo_centre_detected=np.array(
            [c.labels.get("photo_centre_detected", 0.0) for c in keep], np.int8),
        centre=np.array(centre),
        # --- the data a model consumes -------------------------------------
        photo_crops=np.stack([c.rgb for c in keep]),
        height_crops=np.stack([
            np.nan_to_num(c.height, nan=0.0).astype(np.float32) for c in keep
        ]),
        height_um=np.array([c.labels["h_max"] for c in keep], np.float32),
        height_p95_um=np.array([c.labels["h_p95"] for c in keep], np.float32),
        row_col=np.array([[c.row, c.col] for c in keep], np.int16),
        cell_names=np.array([c.name for c in keep]),
        # --- provenance, so a crop can always be traced back ---------------
        overhang=np.array([c.overhang for c in keep], np.float32),
        plate=np.array(stem),
        photo_window_px=np.array(window_px, np.int32),
        scan_rmse_um=np.array(hm.fit.rmse, np.float32),
        photo_rmse_px=np.array(photo.fit.rmse, np.float32),
        cross_rmse_um=np.array(cross.rmse_um, np.float32),
        pitch_um=np.array(float(np.mean(hm.fit.pitch)), np.float32),
        tilt_pp_um=np.array(hm.tilt["tilt_pp_um"], np.float32),
        # Both scales are recorded because the photo and the scan disagree
        # about the grid aspect ratio by ~1.32; whichever axis turns out to be
        # miscalibrated, the number needed to correct it is here.
        um_per_px=np.array(cross.um_per_px, np.float32),
    )
    return len(keep)


def _save_qc(path: Path, stem: str, cells, cross, rows: int, cols: int) -> None:
    """8x8 montage of the photo crops, each labelled with its height."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.5))
    axes = np.atleast_2d(axes)
    for c in cells:
        ax = axes[rows - 1 - c.row][c.col]        # row 0 at the bottom
        if c.rgb is not None:
            ax.imshow(c.rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        h = c.labels.get("h_max")
        if h is not None:
            ax.set_title(
                f"{h:.1f}", fontsize=7,
                color="black" if c.overhang == 0 else "crimson",
            )
    found = sum(c.rgb is not None for c in cells)
    fig.suptitle(
        f"{stem} — {found}/{rows * cols} cells · "
        f"registration {cross.rmse_um:.1f} µm · "
        f"number = height (µm) · red = crop reaches the photo edge",
        y=0.995, fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def process_plate(
    stem: str, photo_path: Path, scan_path: Path, out: Path, args
) -> tuple[PlateResult, list[dict]]:
    """Align one plate and write its outputs."""
    from Auto_OPDx.alignment.dataset import (
        build_cells, fit_cross_modal, solve_photo, solve_plate,
    )

    t0 = time.time()
    res = PlateResult(stem=stem, ok=False)
    try:
        with _watchdog(args.timeout):
            hm = solve_plate(scan_path, args.rows, args.cols)
            photo = solve_photo(photo_path, args.rows, args.cols)
            cross = fit_cross_modal(hm, photo, args.cols)
            window = _photo_window(photo.fit, args.photo_window_px)
            cells = build_cells(
                hm, photo, cross, args.rows, args.cols,
                label_um=args.label_um,
                height_px=args.height_px,
                rgb_px=args.photo_px,
                photo_square_px=window,
                centre=args.centre,
            )

        (out / "plates").mkdir(parents=True, exist_ok=True)
        n = _save_plate(
            out / "plates" / f"{stem}.npz", stem, cells, hm, photo, cross, window,
            args.centre,
        )
        if n == 0:
            raise ValueError("no usable cells produced")

        if not args.no_qc:
            (out / "qc").mkdir(parents=True, exist_ok=True)
            _save_qc(out / "qc" / f"{stem}.png", stem, cells, cross,
                     args.rows, args.cols)

        res = PlateResult(
            stem=stem, ok=True, n_cells=n,
            n_complete=sum(c.complete for c in cells),
            scan_rmse_um=hm.fit.rmse,
            photo_rmse_px=photo.fit.rmse,
            cross_rmse_um=cross.rmse_um,
            cross_max_um=cross.max_um,
            pitch_um=float(np.mean(hm.fit.pitch)),
            tilt_pp_um=hm.tilt["tilt_pp_um"],
            tilt_fraction=hm.tilt["tilt_fraction"],
            um_per_px_x=cross.um_per_px[0],
            um_per_px_y=cross.um_per_px[1],
            photo_window_px=window,
            seconds=time.time() - t0,
        )
        cell_rows = [
            {
                "plate": stem, "row": c.row, "col": c.col,
                "height_um": round(c.labels["h_max"], 4),
                "height_p95_um": round(c.labels["h_p95"], 4),
                "overhang": round(c.overhang, 4),
            }
            for c in cells if c.rgb is not None and "h_max" in c.labels
        ]
        return res, cell_rows

    except Exception as e:  # noqa: BLE001 — one bad plate must not stop the run
        res.error = f"{type(e).__name__}: {e}"
        res.seconds = time.time() - t0
        if args.traceback:
            traceback.print_exc()
        return res, []


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="align-plates",
        description="Align profilometer scans to photos and emit per-cell arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
               "  align-plates --images data/rgb/img --labels data/opdx --out aligned/",
    )
    p.add_argument("--images", type=Path, required=True,
                   help="folder of plate photos (.jpg/.bmp/.png/.tif)")
    p.add_argument("--labels", type=Path, required=True,
                   help="folder of profilometer scans (.OPDx)")
    p.add_argument("--out", type=Path, required=True,
                   help="output folder (created if absent)")
    p.add_argument("--pattern", default="{stem}",
                   help="how a photo stem derives from a scan stem "
                        "(default '{stem}', i.e. identical names)")
    p.add_argument("--rows", type=int, default=8, help="grid rows (default 8)")
    p.add_argument("--cols", type=int, default=8, help="grid columns (default 8)")
    p.add_argument("--label-um", type=float, default=110.0,
                   help="window the height label is read from, µm (default 110)")
    p.add_argument("--photo-px", type=int, default=96,
                   help="output size of each photo crop, px (default 96)")
    p.add_argument("--height-px", type=int, default=32,
                   help="output size of each height crop, px (default 32)")
    p.add_argument("--photo-window-px", type=int, default=None,
                   help="square photo window in source pixels; "
                        "default is 95%% of the tighter fitted grid pitch")
    p.add_argument("--centre", choices=("detected", "lattice"), default="detected",
                   help="centre each crop on the pad the detector found in that "
                        "slot (default), or on the fitted grid's prediction for "
                        "every cell. 'detected' falls back to the prediction "
                        "where no pad was assigned. Study 03 measured mean pad "
                        "IoU 0.71 vs 0.63 in favour of 'detected'")
    p.add_argument("--timeout", type=int, default=180,
                   help="give up on a plate after this many seconds "
                        "(default 180; 0 disables). Guards against malformed "
                        "scan files, which can hang the reader indefinitely")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most this many plates")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave plates that already have an .npz alone")
    p.add_argument("--no-qc", action="store_true",
                   help="skip the per-plate QC montages (faster)")
    p.add_argument("--dry-run", action="store_true",
                   help="report the pairing and exit without processing")
    p.add_argument("--traceback", action="store_true",
                   help="print full tracebacks for failing plates")
    p.add_argument("--quiet", action="store_true", help="only print the summary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for d, name in ((args.images, "--images"), (args.labels, "--labels")):
        if not d.is_dir():
            print(f"error: {name} is not a folder: {d}", file=sys.stderr)
            return 1

    pairs, orphan_scans, orphan_photos = find_pairs(
        args.images, args.labels, args.pattern
    )
    print(f"found {len(pairs)} photo/scan pairs")
    if orphan_scans:
        print(f"  {len(orphan_scans)} scan(s) with no matching photo: "
              f"{', '.join(orphan_scans[:5])}"
              f"{' …' if len(orphan_scans) > 5 else ''}")
    if orphan_photos:
        print(f"  {len(orphan_photos)} photo(s) with no matching scan: "
              f"{', '.join(orphan_photos[:5])}"
              f"{' …' if len(orphan_photos) > 5 else ''}")
    if not pairs:
        print("nothing to do — check --pattern if the two folders name files "
              "differently", file=sys.stderr)
        return 1

    if args.limit:
        pairs = pairs[: args.limit]
    skipped = 0
    if args.skip_existing:
        before = len(pairs)
        pairs = [p for p in pairs
                 if not (args.out / "plates" / f"{p[0]}.npz").exists()]
        skipped = before - len(pairs)
        if skipped:
            print(f"  skipping {skipped} already written")
    if not pairs:
        print("nothing left to process — every plate is already in "
              f"{args.out / 'plates'}")
        return 0

    if args.dry_run:
        print("\ndry run — would process:")
        for stem, photo, scan in pairs:
            print(f"  {stem:24} {photo.name:28} + {scan.name}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[PlateResult] = []
    all_cells: list[dict] = []
    t0 = time.time()

    for i, (stem, photo, scan) in enumerate(pairs, 1):
        res, cells = process_plate(stem, photo, scan, args.out, args)
        results.append(res)
        all_cells.extend(cells)
        if not args.quiet:
            if res.ok:
                print(f"  [{i}/{len(pairs)}] {stem:24} "
                      f"{res.n_cells:3d} cells · "
                      f"registration {res.cross_rmse_um:5.1f} µm · "
                      f"{res.seconds:4.1f}s")
            else:
                print(f"  [{i}/{len(pairs)}] {stem:24} FAILED  {res.error}")

    if skipped:
        # summary.csv, cells.csv and manifest.json describe the whole folder;
        # writing them from a partial run would replace them with a subset
        print(f"\nprocessed {len(results)} plate(s). summary.csv, cells.csv and "
              "manifest.json were NOT rewritten because this run skipped "
              f"{skipped}. Rerun without --skip-existing to regenerate them.")
        return 0 if all(r.ok for r in results) else 1

    ok = [r for r in results if r.ok]
    _write_csv(args.out / "summary.csv", [vars(r) for r in results])
    _write_csv(args.out / "cells.csv", all_cells)
    (args.out / "manifest.json").write_text(json.dumps({
        "images": str(args.images.resolve()),
        "labels": str(args.labels.resolve()),
        "pattern": args.pattern,
        "grid": [args.rows, args.cols],
        "label_um": args.label_um,
        "photo_px": args.photo_px,
        "height_px": args.height_px,
        "photo_window_px": args.photo_window_px or "auto (95% of tighter pitch)",
        "photo_crop_geometry": "square in sensor pixels",
        "centre": args.centre,
        "plates_ok": len(ok),
        "plates_failed": len(results) - len(ok),
        "cells": len(all_cells),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2) + "\n")

    def med(key: str) -> float:
        v = np.array([getattr(r, key) for r in ok], float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")

    print("\n" + "=" * 68)
    print(f"aligned {len(ok)}/{len(results)} plates · {len(all_cells)} cells · "
          f"{time.time() - t0:.0f}s")
    print("=" * 68)
    if ok:
        print(f"  registration error   median {med('cross_rmse_um'):6.2f} µm "
              f"({med('cross_rmse_um') / med('pitch_um') * 100:.1f}% of a cell)")
        print(f"  worst cell on a plate median {med('cross_max_um'):6.2f} µm")
        print(f"  cells per plate      median {med('n_cells'):6.0f} "
              f"of {args.rows * args.cols}")
        print(f"  substrate tilt removed median {med('tilt_pp_um'):6.2f} µm "
              f"({med('tilt_fraction') * 100:.0f}% of the raw height range)")
        print(f"  photo window         {ok[0].photo_window_px} px square "
              f"(pads undistorted)")
    for r in results:
        if not r.ok:
            print(f"  FAILED {r.stem}: {r.error}")
    print(f"\nwrote {args.out}/")
    print(f"  plates/*.npz   per-plate arrays")
    if not args.no_qc:
        print(f"  qc/*.png       montage per plate — open these to eyeball the result")
    print(f"  summary.csv    one row per plate")
    print(f"  cells.csv      one row per cell")
    print(f"  manifest.json  the settings this run used")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
