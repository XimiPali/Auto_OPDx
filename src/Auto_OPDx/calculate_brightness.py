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

def refine_centroid_locally(gray_tophat, cx, cy, window_size=60):
    """
    Refines a centroid coordinate (cx, cy) using 1D projections in a local window.
    """
    half_w = window_size // 2
    
    # Local search area bounds
    x_min = max(0, cx - half_w)
    x_max = min(gray_tophat.shape[1], cx + half_w)
    y_min = max(0, cy - half_w)
    y_max = min(gray_tophat.shape[0], cy + half_w)
    
    local_region = gray_tophat[y_min:y_max, x_min:x_max]
    
    # Refine Y (Row Index) using local horizontal projection (summing across rows, i.e., axis=1)
    local_horizontal = np.sum(local_region, axis=1)
    if len(local_horizontal) > 0:
        p_y = np.argmax(local_horizontal)
        val_y = local_horizontal[p_y]
        thresh_y = val_y * 0.15
        
        left_y = p_y
        while left_y > 0 and local_horizontal[left_y] > thresh_y:
            left_y -= 1
        right_y = p_y
        while right_y < len(local_horizontal) - 1 and local_horizontal[right_y] > thresh_y:
            right_y += 1
        refined_cy = y_min + (left_y + right_y) // 2
    else:
        refined_cy = cy
        
    # Refine X (Column Index) using local vertical projection (summing across columns, i.e., axis=0)
    local_vertical = np.sum(local_region, axis=0)
    if len(local_vertical) > 0:
        p_x = np.argmax(local_vertical)
        val_x = local_vertical[p_x]
        thresh_x = val_x * 0.15
        
        left_x = p_x
        while left_x > 0 and local_vertical[left_x] > thresh_x:
            left_x -= 1
        right_x = p_x
        while right_x < len(local_vertical) - 1 and local_vertical[right_x] > thresh_x:
            right_x += 1
        refined_cx = x_min + (left_x + right_x) // 2
    else:
        refined_cx = cx
        
    return refined_cx, refined_cy

def process_fluorescence_image(img_path, rows=8, cols=8, box_size=50):
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
    
    # 3. Global projections & peaks (NMS)
    horizontal_projection = np.sum(gray_tophat, axis=1)
    vertical_projection = np.sum(gray_tophat, axis=0)
    
    h_centers = get_top_n_peaks_nms(horizontal_projection, rows, min_dist=30)
    v_centers = get_top_n_peaks_nms(vertical_projection, cols, min_dist=30)
    
    # 4. Grid intersections & Local Refinement
    feature_centers = []
    global_centers = []
    for r_idx, row_idx in enumerate(h_centers):
        row_centers = []
        global_row = []
        for c_idx, col_idx in enumerate(v_centers):
            cx, cy = col_idx, row_idx
            global_row.append((cx, cy))
            
            # Local 60x60 projection refinement
            refined_cx, refined_cy = refine_centroid_locally(gray_tophat, cx, cy, window_size=60)
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
    
    ordered_centers = []
    for col_range, row_range in quadrant_ranges:
        for r in sorted(range(row_range[0], row_range[1]), reverse=True):
            for c in sorted(range(col_range[0], col_range[1]), reverse=True):
                ordered_centers.append(feature_centers[r][c])
                
    # 6. Extract region stats on (r+g+b)/3 uint8 average grayscale image
    results = []
    hw = box_size // 2
    hh = box_size // 2
    
    for i, (cx, cy) in enumerate(ordered_centers):
        x_min = max(0, cx - hw)
        x_max = min(img_rgb.shape[1], cx + hw)
        y_min = max(0, cy - hh)
        y_max = min(img_rgb.shape[0], cy + hh)
        
        region_rgb = img_rgb[y_min:y_max, x_min:x_max].astype(float)
        if region_rgb.size == 0:
            mean_val, std_val, min_val, max_val = 0.0, 0.0, 0.0, 0.0
        else:
            region = np.mean(region_rgb, axis=2).astype(np.uint8)
            mean_val = np.mean(region)
            std_val = np.std(region)
            min_val = np.min(region)
            max_val = np.max(region)
            
        results.append({
            'component_id': i + 1,
            'centroid_x': cx,
            'centroid_y': cy,
            'mean_brightness': mean_val,
            'std_deviation': std_val,
            'min_value': min_val,
            'max_value': max_val
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
                    writer.writerow([
                        int(row['component_id']),
                        area_val,
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
