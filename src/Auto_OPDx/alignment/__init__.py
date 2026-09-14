"""Cross-modal alignment between OPDx height maps and RGB plate photos.

The problem this package solves
------------------------------
Per-cell labels for this project were previously built by cropping raw height
map values around hand-placed centres. Two defects came with that:

1. The substrate tilt was never removed. A plane fit to the background spans a
   median of ~21 um across a plate, which is ~71% of the raw z range. The
   resulting per-cell label correlates with the cell's ROW INDEX at r = +0.98,
   so it largely encodes *where* a cell sits rather than *how tall* it is.

2. Height map cells were stretched to square. A cell is only ~9 samples wide on
   the coarse axis but the stored crops are ~70-108 columns, so most of that
   width is interpolated fiction.

The fix implemented here
------------------------
* Detrend against a fitted background plane *before* labelling, which removes
  the position confound at zero measurement cost.
* Work in physical micrometres, and resample once onto an isotropic grid rather
  than stretching pixels.
* Register the two modalities by fitting a single 8x8 lattice model to all 64
  features at once. Individual features span only 2-3 samples on the coarse
  axis, so per-feature centroids are quantisation limited to ~+/-15 um; a joint
  fit over 64 features reduces that by ~sqrt(64).

The (row, col) index is the correspondence between modalities, so no explicit
cross-modal warp is required: evaluate each modality's own lattice at (r, c).
"""

from Auto_OPDx.alignment.lattice import (
    LatticeFit,
    assign_to_grid,
    detect_features,
    fit_lattice,
    solve_lattice,
)
from Auto_OPDx.alignment.plate import (
    Plate,
    detrend,
    fit_background_plane,
    load_plate,
    resample_isotropic,
)
from Auto_OPDx.alignment.rgb import (
    dark_feature_response,
    detect_rgb_features,
    load_rgb,
)

__all__ = [
    "LatticeFit",
    "Plate",
    "assign_to_grid",
    "dark_feature_response",
    "detect_features",
    "detect_rgb_features",
    "detrend",
    "fit_background_plane",
    "fit_lattice",
    "load_plate",
    "load_rgb",
    "resample_isotropic",
    "solve_lattice",
]
