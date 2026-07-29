import numpy as np

def estimate_grid_spacing(final_centroids, final_stats, rows, cols):
    """
    Estimates the median center-to-center grid spacing in X and Y
    from the reordered component centroids.

    Parameters
    ----------
    final_centroids : ndarray, shape (num_samples+1, 2)
        Reordered centroids array (index 0 is background).
    final_stats : ndarray, shape (num_samples+1, 5)
        Reordered stats array. Slots with all-zero stats are missing.
    rows, cols : int
        Grid dimensions.

    Returns
    -------
    dict with keys:
        'spacing_x': float — median column spacing (px)
        'spacing_y': float — median row spacing (px)
        'box_w_px': int — recommended bounding box width
        'box_h_px': int — recommended bounding box height
    """
    # Reconstruct grid slot mapping matching reorder_components quadrant order
    half_rows, half_cols = rows // 2, cols // 2
    quadrant_ranges = [
        ((0, half_cols), (0, half_rows)),       # BL
        ((0, half_cols), (half_rows, rows)),     # TL
        ((half_cols, cols), (0, half_rows)),     # BR
        ((half_cols, cols), (half_rows, rows)),  # TR
    ]
    
    grid = {}
    slot = 1
    for col_range, row_range in quadrant_ranges:
        for r in range(row_range[0], row_range[1]):
            for c in range(col_range[0], col_range[1]):
                grid[(r, c)] = slot
                slot += 1

    x_spacings = []
    y_spacings = []

    # Check horizontal adjacent slots (same row r, adjacent columns c and c+1)
    for r in range(rows):
        for c in range(cols - 1):
            s1 = grid[(r, c)]
            s2 = grid[(r, c + 1)]
            
            valid_s1 = not np.all(final_stats[s1] == 0)
            valid_s2 = not np.all(final_stats[s2] == 0)
            
            if valid_s1 and valid_s2:
                dx = abs(final_centroids[s2][0] - final_centroids[s1][0])
                x_spacings.append(dx)

    # Check vertical adjacent slots (same column c, adjacent rows r and r+1)
    for c in range(cols):
        for r in range(rows - 1):
            s1 = grid[(r, c)]
            s2 = grid[(r + 1, c)]
            
            valid_s1 = not np.all(final_stats[s1] == 0)
            valid_s2 = not np.all(final_stats[s2] == 0)
            
            if valid_s1 and valid_s2:
                dy = abs(final_centroids[s2][1] - final_centroids[s1][1])
                y_spacings.append(dy)

    # Determine median spacing with fallbacks
    if len(x_spacings) >= 2:
        spacing_x = float(np.median(x_spacings))
        box_w_px = max(10, int(round(spacing_x)))
    else:
        spacing_x = 15.0
        box_w_px = 15

    if len(y_spacings) >= 2:
        spacing_y = float(np.median(y_spacings))
        box_h_px = max(10, int(round(spacing_y)))
    else:
        spacing_y = 80.0
        box_h_px = 80

    return {
        'spacing_x': spacing_x,
        'spacing_y': spacing_y,
        'box_w_px': box_w_px,
        'box_h_px': box_h_px,
    }
