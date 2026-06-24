import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def get_top_n_peaks_nms(projection, n, min_dist=30):
    """
    Greedily selects top N peaks from a 1D projection array using prominence and 
    Non-Maximum Suppression (NMS), then refines coordinates using boundary-walk midpoints.
    """
    # Find all peaks and their prominences
    peaks, properties = find_peaks(projection, prominence=np.max(projection) * 0.015)
    
    if len(peaks) == 0:
        # Fallback to simple local maxes if no peaks found
        peaks, _ = find_peaks(projection)
        if len(peaks) == 0:
            # If still nothing, return evenly spaced points
            return np.linspace(0, len(projection) - 1, n, dtype=int)
        prominences = projection[peaks]
    else:
        prominences = properties["prominences"]
    
    # Greedily select top N peaks that are at least min_dist apart (Non-Maximum Suppression)
    sort_indices = np.argsort(prominences)[::-1]
    selected_peaks = []
    for idx in sort_indices:
        p = peaks[idx]
        too_close = False
        for sp in selected_peaks:
            if abs(p - sp) < min_dist:
                too_close = True
                break
        if not too_close:
            selected_peaks.append(p)
            if len(selected_peaks) == n:
                break
                
    # If we have fewer than N, pad or just sort what we have
    selected_peaks = np.sort(selected_peaks)
    
    # Calculate boundaries and midpoints
    centers = []
    threshold_ratio = 0.15
    for idx, p in enumerate(selected_peaks):
        val = projection[p]
        thresh = val * threshold_ratio
        
        min_left = 0
        if idx > 0:
            min_left = (selected_peaks[idx - 1] + p) // 2
            
        max_right = len(projection) - 1
        if idx < len(selected_peaks) - 1:
            max_right = (p + selected_peaks[idx + 1]) // 2
            
        left = p
        while left > min_left and projection[left] > thresh:
            left -= 1
            
        right = p
        while right < max_right and projection[right] > thresh:
            right += 1
            
        center = int((left + right) / 2)
        centers.append(center)
        
    # If we couldn't find exactly N peaks, let's pad/interpolate so we always return exactly N centers
    if len(centers) < n:
        if len(centers) == 0:
            return np.linspace(0, len(projection) - 1, n, dtype=int)
        elif len(centers) == 1:
            return np.array([centers[0]] * n)
        else:
            diffs = np.diff(centers)
            avg_diff = np.mean(diffs)
            padded = list(centers)
            while len(padded) < n:
                padded.append(int(padded[-1] + avg_diff))
            return np.array(padded[:n])
            
    return np.array(centers[:n])

def _get_local_mean_intensity(img, cx, cy, box_w=20):
    """
    Computes the mean pixel intensity of a image region of size box_w x box_w centered at (cx, cy).
    """
    hw = box_w // 2
    x_min = max(0, cx - hw)
    x_max = min(img.shape[1], cx + hw)
    y_min = max(0, cy - hw)
    y_max = min(img.shape[0], cy + hw)
    region = img[y_min:y_max, x_min:x_max]
    if region.size == 0:
        return 0.0
    return float(np.mean(region))

def _compute_alignment_confidence(mean_val, dist_to_grid, dist_to_corner=0.0, aspect_ratio=1.0):
    """
    Calculates alignment confidence score incorporating corner alignment and aspect ratio.
    """
    penalty_corner = dist_to_corner * 2.0
    penalty_ar = (aspect_ratio - 1.0) * 15.0
    
    penalty_grid = 0.0
    if dist_to_grid > 15:
        penalty_grid = (dist_to_grid - 15) * 2.0
    if dist_to_grid > 30:
        penalty_grid += 100.0
        
    return mean_val - penalty_corner - penalty_ar - penalty_grid

def refine_centroid_locally(gray_tophat, cx, cy, window_size=80, refinement_method='contour'):
    """
    Refines a centroid coordinate (cx, cy) by using morphological closing to fill
    in the square features, and then calculating the centers using either the contour
    moments centroid, the local area horizontal and vertical projections, or a dynamic
    'best' confidence scoring selection.
    """
    half_w = window_size // 2
    
    # Local search area bounds
    x_min = max(0, cx - half_w)
    x_max = min(gray_tophat.shape[1], cx + half_w)
    y_min = max(0, cy - half_w)
    y_max = min(gray_tophat.shape[0], cy + half_w)
    
    local_region = gray_tophat[y_min:y_max, x_min:x_max]
    
    # If the local region is empty, has low contrast, or has low maximum intensity (background noise), return original guess
    if local_region.size == 0 or np.max(local_region) < 10 or np.max(local_region) - np.min(local_region) < 5:
        return cx, cy
        
    # Ensure it's uint8 for OpenCV image operations
    local_region_u8 = local_region.astype(np.uint8)
    
    # Pad the local region with zeros to prevent edge truncation during large morphological operations
    ksize = max(5, int(window_size // 2) | 1)
    padded = cv2.copyMakeBorder(local_region_u8, ksize, ksize, ksize, ksize, cv2.BORDER_CONSTANT, value=0)
    
    # Binarize the padded region to isolate bright features
    _, thresh = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use morphological closing with a large kernel to completely fill in the square features
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed_padded = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Crop back to the original size
    closed = closed_padded[ksize:-ksize, ksize:-ksize]
    thresh_cropped = thresh[ksize:-ksize, ksize:-ksize]
    
    # Define the peak boundary midpoint method helper
    def get_peak_center(proj, fallback_val):
        p = np.argmax(proj)
        val = proj[p]
        if val == 0:
            return fallback_val
        thresh_val = val * 0.15
        left = p
        while left > 0 and proj[left] > thresh_val:
            left -= 1
        right = p
        while right < len(proj) - 1 and proj[right] > thresh_val:
            right += 1
        return (left + right) / 2.0

    ccx, ccy = None, None
    if refinement_method in ('contour', 'best'):
        # Shape-based square center finding
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        min_dist_to_center = float('inf')
        center_local_x = (x_max - x_min) / 2.0
        center_local_y = (y_max - y_min) / 2.0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                bcx = M["m10"] / M["m00"]
                bcy = M["m01"] / M["m00"]
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                bcx = bx + bw / 2.0
                bcy = by + bh / 2.0
                
            dist = (bcx - center_local_x)**2 + (bcy - center_local_y)**2
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                best_contour = cnt
                
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx_local = M["m10"] / M["m00"]
                cy_local = M["m01"] / M["m00"]
            else:
                bx, by, bw, bh = cv2.boundingRect(best_contour)
                cx_local = bx + bw / 2.0
                cy_local = by + bh / 2.0
                
            ccx = x_min + int(round(cx_local))
            ccy = y_min + int(round(cy_local))
            
            if refinement_method == 'contour':
                return ccx, ccy
            
    # Compute horizontal and vertical projections of the morphology results (fallback or direct projection)
    proj_y = np.sum(closed, axis=1)
    proj_x = np.sum(closed, axis=0)
    cy_local = get_peak_center(proj_y, cy - y_min)
    cx_local = get_peak_center(proj_x, cx - x_min)
    
    pcx = x_min + int(round(cx_local))
    pcy = y_min + int(round(cy_local))
    
    if refinement_method == 'projection':
        return pcx, pcy
        
    if refinement_method == 'best':
        # 1. Contour on Binary (ccx, ccy from closed)
        # 2. Projection on Binary (pcx, pcy from closed)
        # 3. Contour on Grayscale (ccx_g, ccy_g from thresh_cropped)
        # 4. Projection on Grayscale (pcx_g, pcy_g from local_region)
        candidates = []
        if ccx is not None:
            candidates.append((ccx, ccy, "Contour_Binary"))
        candidates.append((pcx, pcy, "Projection_Binary"))
        
        # Contour on Grayscale (thresh_cropped)
        contours_g, _ = cv2.findContours(thresh_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour_g = None
        min_dist_to_center_g = float('inf')
        center_local_x = (x_max - x_min) / 2.0
        center_local_y = (y_max - y_min) / 2.0
        
        for cnt in contours_g:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                bcx = M["m10"] / M["m00"]
                bcy = M["m01"] / M["m00"]
            else:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                bcx = bx + bw / 2.0
                bcy = by + bh / 2.0
            dist = (bcx - center_local_x)**2 + (bcy - center_local_y)**2
            if dist < min_dist_to_center_g:
                min_dist_to_center_g = dist
                best_contour_g = cnt
                
        if best_contour_g is not None:
            M = cv2.moments(best_contour_g)
            if M["m00"] != 0:
                cx_l = M["m10"] / M["m00"]
                cy_l = M["m01"] / M["m00"]
            else:
                bx, by, bw, bh = cv2.boundingRect(best_contour_g)
                cx_l, cy_l = bx + bw / 2.0, by + bh / 2.0
            ccx_g = x_min + int(round(cx_l))
            ccy_g = y_min + int(round(cy_l))
            candidates.append((ccx_g, ccy_g, "Contour_Gray"))
            
        # Projection on Grayscale (local_region)
        proj_y_g = np.sum(local_region, axis=1)
        proj_x_g = np.sum(local_region, axis=0)
        cy_l_p_g = get_peak_center(proj_y_g, cy - y_min)
        cx_l_p_g = get_peak_center(proj_x_g, cx - x_min)
        pcx_g = x_min + int(round(cx_l_p_g))
        pcy_g = y_min + int(round(cy_l_p_g))
        candidates.append((pcx_g, pcy_g, "Projection_Gray"))
        
        # Fit minimum area rectangle to locate corner center and aspect ratio from best morphological contour
        # Fall back to unclosed grayscale contour if closed contour is distorted (e.g., merged with noise)
        rect_b = cv2.minAreaRect(best_contour) if best_contour is not None else None
        rect_g = cv2.minAreaRect(best_contour_g) if best_contour_g is not None else None
        
        ar_b = max(rect_b[1]) / (min(rect_b[1]) + 1e-5) if rect_b is not None else 999.0
        ar_g = max(rect_g[1]) / (min(rect_g[1]) + 1e-5) if rect_g is not None else 999.0
        
        use_g = False
        if rect_g is not None and ar_g < 1.8:
            if rect_b is None or ar_b >= 1.8 or abs(ar_g - 1.0) < abs(ar_b - 1.0):
                use_g = True
                
        chosen_rect = rect_g if use_g else rect_b
        if chosen_rect is not None:
            corner_cx = x_min + chosen_rect[0][0]
            corner_cy = y_min + chosen_rect[0][1]
            w, h = chosen_rect[1]
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        else:
            corner_cx, corner_cy, aspect_ratio = None, None, 1.0
            
        best_cand = (pcx, pcy)
        best_score = -float('inf')
        
        for cand_x, cand_y, name in candidates:
            dist_to_grid = np.sqrt((cand_x - cx)**2 + (cand_y - cy)**2)
            
            if corner_cx is not None:
                dist_to_corner = np.sqrt((cand_x - corner_cx)**2 + (cand_y - corner_cy)**2)
            else:
                dist_to_corner = 0.0
                
            conf = _compute_alignment_confidence(
                _get_local_mean_intensity(gray_tophat, cand_x, cand_y, 20),
                dist_to_grid,
                dist_to_corner,
                aspect_ratio if corner_cx is not None else 1.0
            )
            
            if conf > best_score:
                best_score = conf
                best_cand = (cand_x, cand_y)
                
        return best_cand[0], best_cand[1]
        
    return pcx, pcy

def fit_robust_grid(projection, n_lines=8, is_horizontal=True):
    """
    Fits a robust grid model to a 1D projection array using the physical spacing
    and gap constraints of the chip. Optimized with precomputed window sums
    and narrowed search bounds for 140x speedup.
    """
    if is_horizontal:
        s_range = range(118, 129)
        g_range = range(180, 220)
        offset_range = range(10, 70)
    else:
        s_range = range(160, 175)
        g_range = range(195, 215)
        offset_range = range(10, 60)
        
    best_score = -1
    best_params = None
    smoothed = np.convolve(projection, np.ones(5)/5.0, mode='same')
    
    n_pts = len(projection)
    line_scores = np.zeros(n_pts)
    for p in range(n_pts):
        w_min = max(0, p - 5)
        w_max = min(n_pts, p + 6)
        line_scores[p] = np.sum(smoothed[w_min:w_max])
        
    for s in s_range:
        for g in g_range:
            for offset in offset_range:
                p0 = offset
                p1 = offset + s
                p2 = offset + 2 * s
                p3 = offset + 3 * s
                p4 = offset + 3 * s + g
                p5 = p4 + s
                p6 = p4 + 2 * s
                p7 = p4 + 3 * s
                
                if p7 >= n_pts:
                    continue
                    
                score = (line_scores[p0] + line_scores[p1] + line_scores[p2] + line_scores[p3] +
                         line_scores[p4] + line_scores[p5] + line_scores[p6] + line_scores[p7])
                
                if score > best_score:
                    best_score = score
                    best_params = (p0, p1, p2, p3, p4, p5, p6, p7)
                    
    if best_params is None:
        return np.linspace(0, len(projection) - 1, n_lines, dtype=int)
        
    return np.array(best_params)

def refine_grid_lines_locally(projection, coarse_lines, search_radius=10):
    """
    Refines coarse grid line positions locally by finding the peak in vertical/horizontal
    projection in a small window, accounting for minor lens distortion or chip tilt.
    """
    refined_lines = []
    smoothed = np.convolve(projection, np.ones(5)/5.0, mode='same')
    for line in coarse_lines:
        w_min = max(0, line - search_radius)
        w_max = min(len(projection), line + search_radius + 1)
        sub_proj = smoothed[w_min:w_max]
        local_peak_idx = np.argmax(sub_proj)
        refined_val = w_min + local_peak_idx
        refined_lines.append(refined_val)
    return np.array(refined_lines)

def process_fluorescence_image(img_path, rows=8, cols=8, box_size=50, use_circle_mask=False, ordering='reversed', refinement_method='contour'):
    """
    Runs the complete fluorescence grid finding, centroid refinement, and spot stats extraction.
    """
    # 1. Load image and convert to RGB
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image at: {img_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Preprocess to extract grid
    img_median = cv2.medianBlur(img_rgb, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    img_tophat = cv2.morphologyEx(img_median, cv2.MORPH_TOPHAT, kernel)
    gray_tophat = cv2.cvtColor(img_tophat, cv2.COLOR_RGB2GRAY)
    
    # 3. Global projections & Hybrid Robust Grid Fitting
    horizontal_projection = np.sum(gray_tophat, axis=1)
    vertical_projection = np.sum(gray_tophat, axis=0)
    
    h_coarse = fit_robust_grid(horizontal_projection, rows, is_horizontal=True)
    v_coarse = fit_robust_grid(vertical_projection, cols, is_horizontal=False)
    
    h_centers = refine_grid_lines_locally(horizontal_projection, h_coarse, search_radius=10)
    v_centers = refine_grid_lines_locally(vertical_projection, v_coarse, search_radius=10)
    
    # 4. Grid intersections & Local Refinement
    feature_centers = []
    global_centers = []
    for r_idx, row_idx in enumerate(h_centers):
        row_centers = []
        global_row = []
        for c_idx, col_idx in enumerate(v_centers):
            cx, cy = col_idx, row_idx
            global_row.append((cx, cy))
            
            # Local 80x80 refinement
            refined_cx, refined_cy = refine_centroid_locally(gray_tophat, cx, cy, window_size=80, refinement_method=refinement_method)
            row_centers.append((refined_cx, refined_cy))
            
        feature_centers.append(row_centers)
        global_centers.append(global_row)
        
    # 5. Spot quadrant-based ordering
    quadrant_ranges = [
        ((0, 4), (0, 4)), # Quadrant 1
        ((0, 4), (4, 8)), # Quadrant 2
        ((4, 8), (0, 4)), # Quadrant 3
        ((4, 8), (4, 8))  # Quadrant 4
    ]
    
    if ordering == 'reversed':
        rev_r, rev_c = True, True
    elif ordering == 'standard':
        rev_r, rev_c = False, False
    elif ordering == 'p8':
        rev_r, rev_c = False, True
    else:
        rev_r, rev_c = True, True
    
    ordered_centers = []
    for col_range, row_range in quadrant_ranges:
        for r in sorted(range(row_range[0], row_range[1]), reverse=rev_r):
            for c in sorted(range(col_range[0], col_range[1]), reverse=rev_c):
                ordered_centers.append(feature_centers[r][c])
                
    # 6. Extract region stats on (r+g+b)/3 uint8 average grayscale image
    results = []
    
    # Construct circular mask if use_circle_mask is enabled
    if use_circle_mask:
        w = box_size
        y_grid, x_grid = np.ogrid[:w, :w]
        if w == 51:
            cx_mask, cy_mask = 25.5, 25.5
            radius = 30.0
        else:
            cx_mask = (w - 1) / 2.0
            cy_mask = (w - 1) / 2.0
            radius = 30.0 * (w / 51.0)
        mask = (x_grid - cx_mask)**2 + (y_grid - cy_mask)**2 <= radius**2
        area_val = int(np.sum(mask))
    else:
        area_val = box_size * box_size
    
    for i, (cx, cy) in enumerate(ordered_centers):
        x_min = max(0, cx - box_size // 2)
        x_max = min(img_rgb.shape[1], cx - box_size // 2 + box_size)
        y_min = max(0, cy - box_size // 2)
        y_max = min(img_rgb.shape[0], cy - box_size // 2 + box_size)
        
        region_rgb = img_rgb[y_min:y_max, x_min:x_max].astype(float)
        if region_rgb.size == 0:
            mean_val, std_val, min_val, max_val = 0.0, 0.0, 0.0, 0.0
            actual_area = 0
        else:
            region = np.mean(region_rgb, axis=2).astype(np.uint8)
            if use_circle_mask:
                if region.shape == (box_size, box_size):
                    pixels = region[mask]
                    actual_area = area_val
                else:
                    # Handle boundaries by cropping mask
                    mask_y_min = max(0, - (cy - box_size // 2))
                    mask_y_max = mask_y_min + region.shape[0]
                    mask_x_min = max(0, - (cx - box_size // 2))
                    mask_x_max = mask_x_min + region.shape[1]
                    cropped_mask = mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
                    pixels = region[cropped_mask]
                    actual_area = int(np.sum(cropped_mask))
            else:
                pixels = region
                actual_area = region.size
                
            if pixels.size == 0:
                mean_val, std_val, min_val, max_val = 0.0, 0.0, 0.0, 0.0
            else:
                mean_val = np.mean(pixels)
                std_val = np.std(pixels)
                min_val = np.min(pixels)
                max_val = np.max(pixels)
            
        results.append({
            'component_id': i + 1,
            'centroid_x': cx,
            'centroid_y': cy,
            'mean_brightness': mean_val,
            'std_deviation': std_val,
            'min_value': min_val,
            'max_value': max_val,
            'area': actual_area
        })
        
    df_fluorescence = pd.DataFrame(results)
    
    return {
        'df': df_fluorescence,
        'global_peaks_h': h_centers,
        'global_peaks_v': v_centers,
        'global_centers': global_centers,
        'refined_centers': feature_centers,
        'img_rgb': img_rgb
    }

def compile_fluorescence_results(template_path, processed_data, output_csv_path, box_size=50):
    """
    Compiles all processed fluorescence results into a clean, simple CSV
    containing only the raw data (Num, Area, Mean, StdDev, Min, Max) without post-processing.
    """
    try:
        area_val = box_size * box_size
        
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.writer(f)
            
            for file_idx, (img_name, df_flu) in enumerate(processed_data.items()):
                writer.writerow([img_name])
                writer.writerow(["Num", "Area", "Mean", "StdDev", "Min", "Max"])
                
                for i, row in df_flu.iterrows():
                    row_area = int(row['area']) if 'area' in row else area_val
                    writer.writerow([
                        int(row['component_id']),
                        row_area,
                        f"{row['mean_brightness']:.6f}" if isinstance(row['mean_brightness'], float) else row['mean_brightness'],
                        f"{row['std_deviation']:.6f}" if isinstance(row['std_deviation'], float) else row['std_deviation'],
                        f"{row['min_value']:.6f}" if isinstance(row['min_value'], float) else row['min_value'],
                        f"{row['max_value']:.6f}" if isinstance(row['max_value'], float) else row['max_value']
                    ])
                    # Separate every 16 spots with a blank line
                    if (i + 1) % 16 == 0 and (i + 1) != len(df_flu):
                        writer.writerow([])
                        
                if file_idx < len(processed_data) - 1:
                    writer.writerow([])
                    
        return True, "Successfully compiled results to CSV."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Failed to save results: {str(e)}"
