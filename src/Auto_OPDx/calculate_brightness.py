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
    Compiles all processed fluorescence results using October2024Flu.xlsx as a template,
    replicating the exact sheet structure, normalization formulas, and formatting.
    """
    import openpyxl
    
    try:
        if not os.path.exists(template_path):
            # Fallback to simple concatenation if template is missing
            dfs = []
            for name, res_df in processed_data.items():
                df_copy = res_df.copy()
                df_copy.insert(0, 'image_name', name)
                dfs.append(df_copy)
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                combined.to_csv(output_csv_path, index=False)
                return False, "Template file not found. Saved simple combined CSV."
            return False, "No data to compile."
            
        wb = openpyxl.load_workbook(template_path, data_only=True)
        sheet = wb['Sheet1']
        
        # 1. Read SNA data from Excel template itself
        sna_images = [
            '25 uM SNA 2 mg/ML 13 hours A', '25 uM SNA 2 mg/ML 13 hours B',
            '25 uM SNA 2 mg/ML 13 hours C', '25 uM SNA 2 mg/ML 13 hours D'
        ]
        
        sna_raw = {}
        current_sna_img = None
        for r in range(1, sheet.max_row + 1):
            val_a = sheet.cell(row=r, column=1).value
            if val_a in sna_images:
                current_sna_img = val_a
                sna_raw[current_sna_img] = []
            elif current_sna_img is not None and isinstance(val_a, int) and val_a <= 64:
                row_vals = [sheet.cell(row=r, column=c).value for c in range(1, 7)]
                sna_raw[current_sna_img].append({
                    'component_id': row_vals[0],
                    'mean_brightness': row_vals[2],
                    'std_deviation': row_vals[3],
                    'min_value': row_vals[4],
                    'max_value': row_vals[5]
                })
                
        # Default NFs from original sheet
        nfs = {
            'P2': 6.7207560539245605,
            'P6': 5.0425318876902265,
            'P7': 4.211039066314697,
            'P8': 7.406955401102702,
            'P5': 1.0
        }
        
        # Dynamic calculation of NFs if images are processed
        for group, prefix in [('P2', '25 uM SCR073  200uM P2'), 
                              ('P6', '25 uM SCR074  200uM P6'), 
                              ('P7', '25 uM SCR079  200uM P7'), 
                              ('P8', '25 uM SCR080  200uM P8')]:
            img_a = f"{prefix} A"
            img_b = f"{prefix} B"
            if img_a in processed_data and img_b in processed_data:
                a_means = processed_data[img_a]['mean_brightness'].iloc[:3].tolist()
                b_means = processed_data[img_b]['mean_brightness'].iloc[:3].tolist()
                nfs[group] = np.mean(a_means + b_means)
                
        p5_a_name = '25 uM FL 200uM P5 A'
        if p5_a_name in processed_data:
            nfs['P5'] = processed_data[p5_a_name]['mean_brightness'].iloc[2]
            
        csv_rows = []
        
        def fmt(val):
            if val is None or pd.isna(val):
                return ""
            if isinstance(val, float):
                return f"{val:.6f}"
            return str(val)
            
        # Recreate the sheet row-by-row
        for r in range(1, sheet.max_row + 1):
            row_data = ["" for _ in range(29)] # 29 columns A to AC
            
            # Read original values across all 29 columns in the template
            for c in range(1, 30):
                val = sheet.cell(row=r, column=c).value
                if val is not None:
                    row_data[c-1] = val
                    
            orig_a = sheet.cell(row=r, column=1).value
            orig_i = sheet.cell(row=r, column=9).value
            
            # Left table check
            if isinstance(orig_a, int) and orig_a <= 64:
                # Walk up to find the active block name
                block_name = None
                for walk_r in range(r, 0, -1):
                    cand = sheet.cell(row=walk_r, column=1).value
                    if cand in processed_data or cand in sna_images:
                        block_name = cand
                        break
                        
                if block_name in processed_data:
                    df_csv = processed_data[block_name]
                    spot_row = df_csv[df_csv['component_id'] == orig_a].iloc[0]
                    row_data[0] = orig_a
                    row_data[1] = box_size * box_size # area of the bounding box
                    row_data[2] = spot_row['mean_brightness']
                    row_data[3] = spot_row['std_deviation']
                    row_data[4] = spot_row['min_value']
                    row_data[5] = spot_row['max_value']
                    
                    if r >= 1415:
                        group = 'P5' if 'P5' in block_name else ('P2' if 'P2' in block_name else ('P6' if 'P6' in block_name else ('P7' if 'P7' in block_name else 'P8')))
                        row_data[2] = spot_row['mean_brightness'] / nfs[group]
                        row_data[4] = spot_row['min_value'] / nfs[group]
                        row_data[5] = spot_row['max_value'] / nfs[group]
                        
                elif block_name in sna_images:
                    spot_row = [spot for spot in sna_raw[block_name] if spot['component_id'] == orig_a][0]
                    row_data[0] = orig_a
                    row_data[1] = box_size * box_size
                    row_data[2] = spot_row['mean_brightness']
                    row_data[3] = spot_row['std_deviation']
                    row_data[4] = spot_row['min_value']
                    row_data[5] = spot_row['max_value']
                    
            # Right table check
            if isinstance(orig_i, int) and orig_i <= 16:
                block_name = None
                for walk_r in range(r, 0, -1):
                    cand = sheet.cell(row=walk_r, column=9).value
                    if cand in processed_data or cand in sna_images:
                        block_name = cand
                        break
                        
                if block_name is not None:
                    base_img = block_name[:-2]
                    
                    spot_means = []
                    for letter in ['A', 'B', 'C', 'D']:
                        img_name = f"{base_img} {letter}"
                        if img_name in processed_data:
                            df_csv = processed_data[img_name]
                            q1_mean = df_csv['mean_brightness'].iloc[orig_i - 1]
                            q2_mean = df_csv['mean_brightness'].iloc[orig_i - 1 + 16]
                            q3_mean = df_csv['mean_brightness'].iloc[orig_i - 1 + 32]
                            q4_mean = df_csv['mean_brightness'].iloc[orig_i - 1 + 48]
                            spot_means.extend([q1_mean, q2_mean, q3_mean, q4_mean])
                        elif img_name in sna_images:
                            sna_df = sna_raw[img_name]
                            q1_mean = sna_df[orig_i - 1]['mean_brightness']
                            q2_mean = sna_df[orig_i - 1 + 16]['mean_brightness']
                            q3_mean = sna_df[orig_i - 1 + 32]['mean_brightness']
                            q4_mean = sna_df[orig_i - 1 + 48]['mean_brightness']
                            spot_means.extend([q1_mean, q2_mean, q3_mean, q4_mean])
                            
                    row_data[8] = orig_i
                    for c_idx, val in enumerate(spot_means):
                        row_data[9 + c_idx] = val
                        
                    row_data[25] = np.mean(spot_means)
                    
                    group = 'P2' if 'P2' in block_name else ('P6' if 'P6' in block_name else ('P7' if 'P7' in block_name else ('P8' if 'P8' in block_name else 'SNA')))
                    if group == 'SNA':
                        row_data[26] = np.std(spot_means, ddof=1)
                    else:
                        row_data[26] = row_data[25] / nfs[group]
                        
            # Bottom helper table right side
            if r >= 1415 and orig_i is not None:
                if isinstance(orig_i, (int, float)):
                    # Map rows to spots 1,2,3 of A and B
                    spot_map = {
                        1416: ('25 uM FL 200uM P5 A', 0), 1417: ('25 uM FL 200uM P5 A', 1), 1418: ('25 uM FL 200uM P5 A', 2),
                        1419: ('25 uM FL 200uM P5 B', 0), 1420: ('25 uM FL 200uM P5 B', 1), 1421: ('25 uM FL 200uM P5 B', 2),
                        
                        1425: ('25 uM SCR073  200uM P2 A', 0), 1426: ('25 uM SCR073  200uM P2 A', 1), 1427: ('25 uM SCR073  200uM P2 A', 2),
                        1428: ('25 uM SCR073  200uM P2 B', 0), 1429: ('25 uM SCR073  200uM P2 B', 1), 1430: ('25 uM SCR073  200uM P2 B', 2),
                        
                        1434: ('25 uM SCR074  200uM P6 A', 0), 1435: ('25 uM SCR074  200uM P6 A', 1), 1436: ('25 uM SCR074  200uM P6 A', 2),
                        1437: ('25 uM SCR074  200uM P6 B', 0), 1438: ('25 uM SCR074  200uM P6 B', 1), 1439: ('25 uM SCR074  200uM P6 B', 2),
                        
                        1443: ('25 uM SCR079  200uM P7 A', 0), 1444: ('25 uM SCR079  200uM P7 A', 1), 1445: ('25 uM SCR079  200uM P7 A', 2),
                        1446: ('25 uM SCR079  200uM P7 B', 0), 1447: ('25 uM SCR079  200uM P7 B', 1), 1448: ('25 uM SCR079  200uM P7 B', 2),
                        
                        1452: ('25 uM SCR080  200uM P8 A', 0), 1453: ('25 uM SCR080  200uM P8 A', 1), 1454: ('25 uM SCR080  200uM P8 A', 2),
                        1455: ('25 uM SCR080  200uM P8 B', 0), 1456: ('25 uM SCR080  200uM P8 B', 1), 1457: ('25 uM SCR080  200uM P8 B', 2),
                    }
                    if r in spot_map:
                        img_name, spot_idx = spot_map[r]
                        group = 'P5' if 'P5' in img_name else ('P2' if 'P2' in img_name else ('P6' if 'P6' in img_name else ('P7' if 'P7' in img_name else 'P8')))
                        if img_name in processed_data:
                            val_raw = processed_data[img_name]['mean_brightness'].iloc[spot_idx]
                            row_data[8] = val_raw / nfs[group]
                            
                    avg_map = {
                        1416: 'P5', 1426: 'P2', 1434: 'P6', 1444: 'P7', 1452: 'P8'
                    }
                    if r in avg_map:
                        grp = avg_map[r]
                        row_data[9] = nfs[grp]
                        
            # Replicate Col Y/Z Background row for the top blocks
            bg_rows = {
                37: 'P2', 58: 'P6', 79: 'P7', 100: 'P8'
            }
            if r in bg_rows:
                grp = bg_rows[r]
                row_data[24] = "Background"
                row_data[25] = nfs[grp]
                
            csv_rows.append(row_data)
            
        # Write out compiled CSV
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            for row in csv_rows:
                f.write(",".join(map(fmt, row)) + "\n")
                
        return True, "Successfully compiled results using template sheet."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Failed to compile using template: {str(e)}"
