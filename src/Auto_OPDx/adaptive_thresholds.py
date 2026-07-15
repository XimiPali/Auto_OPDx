import numpy as np
import cv2

def _otsu_threshold(data, n_bins=256):
    """
    Computes Otsu's threshold on 1D numpy array.
    Returns the threshold value, or None if computation is not possible.
    """
    if len(data) == 0:
        return None
    
    min_val, max_val = np.min(data), np.max(data)
    if min_val == max_val:
        return min_val

    counts, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize counts to get probabilities
    p = counts / float(len(data))

    # Cumulative sum of class probabilities
    omega = np.cumsum(p)
    # Cumulative sum of class means
    mu = np.cumsum(p * bin_centers)

    # Total mean
    mu_t = mu[-1]

    # Find the threshold that maximizes inter-class variance
    # Avoid division by zero by filtering indices where 0 < omega < 1
    idx = (omega > 0.0) & (omega < 1.0)
    if not np.any(idx):
        return None

    sigma_b_squared = (mu_t * omega[idx] - mu[idx]) ** 2 / (omega[idx] * (1.0 - omega[idx]))
    max_idx = np.argmax(sigma_b_squared)
    
    return bin_centers[idx][max_idx]

def compute_adaptive_thresholds(z, x_mesh, y_mesh, group_size=10, rows=8, cols=8):
    """
    Computes adaptive thresholds based on height data z:
    1. background_percentile (float, e.g. 45.0)
    2. feature_distance_um (float, e.g. 2.0)
    3. min_area_px (int, e.g. 15)
    """
    num_rows = z.shape[0]
    num_groups = int(np.ceil(num_rows / group_size))

    # 1. Compute Otsu percentile for each group, then take the median
    percentiles = []
    for i in range(num_groups):
        start_row = i * group_size
        end_row = min((i + 1) * group_size, num_rows)
        current_group_data = z[start_row:end_row, :]
        combined_heights = current_group_data.flatten()

        otsu_val = _otsu_threshold(combined_heights)
        if otsu_val is not None:
            percentile_rank = np.mean(combined_heights <= otsu_val) * 100.0
            percentiles.append(percentile_rank)
        else:
            percentiles.append(50.0)

    suggested_percentile = np.median(percentiles) if percentiles else 45.0
    # Keep percentile in a reasonable bounds
    suggested_percentile = float(np.clip(suggested_percentile, 30.0, 70.0))

    # 2. Fit a quick global plane to estimate background noise scale (MAD-based)
    group_background_thresholds = []
    for i in range(num_groups):
        start_row = i * group_size
        end_row = min((i + 1) * group_size, num_rows)
        current_group_data = z[start_row:end_row, :]
        combined_heights = current_group_data.flatten()
        percentile_value = np.percentile(combined_heights, suggested_percentile)
        group_background_thresholds.append(percentile_value)

    group_background_masks_rows = []
    for row_idx in range(num_rows):
        group_index = row_idx // group_size
        threshold = group_background_thresholds[group_index]
        row_data = z[row_idx, :]
        row_mask = row_data <= threshold
        group_background_masks_rows.append(row_mask)

    group_background_mask = np.array(group_background_masks_rows)
    x_background = x_mesh[group_background_mask]
    y_background = y_mesh[group_background_mask]
    z_actual_background = z[group_background_mask]

    A_background = np.c_[x_background, y_background, np.ones(x_background.shape[0])]
    try:
        coeffs_background, _, _, _ = np.linalg.lstsq(A_background, z_actual_background, rcond=None)
        reg_coef_background = coeffs_background[:2]
        reg_intercept_background = coeffs_background[2]
    except Exception:
        # Fallback if lstsq fails
        reg_coef_background = np.array([0.0, 0.0])
        reg_intercept_background = np.mean(z_actual_background)

    z_predicted_background = reg_intercept_background + reg_coef_background[0] * x_background + reg_coef_background[1] * y_background
    residuals = z_actual_background - z_predicted_background

    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    sigma = 1.4826 * mad
    suggested_distance_um = float(max(3.0 * sigma * 1e6, 2.0))

    # 3. Predict full plane, create preliminary mask, and compute component areas
    z_predicted_full = reg_intercept_background + reg_coef_background[0] * x_mesh + reg_coef_background[1] * y_mesh
    preliminary_bg_mask = np.abs(z - z_predicted_full) < (suggested_distance_um * 1e-6)

    # Run connected components on features (negation of background mask)
    output_mask_cv = (~preliminary_bg_mask).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_mask_cv, 4, cv2.CV_32S)

    total_pixels = z.shape[0] * z.shape[1]
    num_expected_samples = rows * cols

    if num_labels > 1:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    else:
        areas = []

    if len(areas) >= 3:
        median_area = np.median(areas)
        suggested_min_area = int(max(int(median_area * 0.25), 10))
    else:
        # Fallback to resolution-aware estimate
        suggested_min_area = int(max(int(total_pixels / (num_expected_samples * 100)), 10))

    return {
        "background_percentile": suggested_percentile,
        "feature_distance_um": suggested_distance_um,
        "min_area_px": suggested_min_area
    }
