import sys
import os
import numpy as np
import pandas as pd
import cv2
import traceback
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QFileDialog, 
                             QSpinBox, QTextEdit, QMessageBox, QListWidget, QListWidgetItem,
                             QSizePolicy, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon

from OPDx_read.reader import DektakLoad
from Auto_OPDx.global_plane import generate_global_plane
from Auto_OPDx.mask import refine_background_mask
from Auto_OPDx.filter import filter_components
from Auto_OPDx.reorder import reorder_components
from Auto_OPDx.calculate_heights import calculate_heights
from Auto_OPDx.calculate_brightness import process_fluorescence_image, compile_fluorescence_results

class ProfilometryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Auto-OPDx & Fluorescence Data Processor')
        self.resize(1100, 850)
        
        main_layout = QHBoxLayout()
        layout = QVBoxLayout()
        
        # 0. Mode Selection
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Operation Mode:")
        self.mode_label.setStyleSheet("font-weight: bold;")
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(["Profilometry Mode (OPDx)", "Fluorescence Mode (JPG/PNG)"])
        self.mode_dropdown.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_dropdown)
        layout.addLayout(mode_layout)
        
        # 1. File Selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("OPDx Files:")
        self.file_input = QLineEdit()
        self.file_btn = QPushButton("Browse")
        self.file_btn.clicked.connect(self.browse_files)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.file_btn)
        layout.addLayout(file_layout)
        
        # 2. Grid Layout Specification
        grid_layout = QHBoxLayout()
        self.rows_label = QLabel("Grid Rows:")
        self.rows_input = QSpinBox()
        self.rows_input.setMinimum(1)
        self.rows_input.setValue(8)
        
        self.cols_label = QLabel("Grid Columns:")
        self.cols_input = QSpinBox()
        self.cols_input.setMinimum(1)
        self.cols_input.setValue(8)
        
        grid_layout.addWidget(self.rows_label)
        grid_layout.addWidget(self.rows_input)
        grid_layout.addWidget(self.cols_label)
        grid_layout.addWidget(self.cols_input)
        layout.addLayout(grid_layout)
        
        # Bounding Box Size Control (only visible in Fluorescence Mode)
        self.box_size_layout = QHBoxLayout()
        self.box_size_label = QLabel("Bounding Box Size (px):")
        self.box_size_input = QSpinBox()
        self.box_size_input.setRange(5, 200)
        self.box_size_input.setValue(50)
        self.box_size_input.valueChanged.connect(self.on_box_size_changed)
        self.box_size_layout.addWidget(self.box_size_label)
        self.box_size_layout.addWidget(self.box_size_input)
        layout.addLayout(self.box_size_layout)
        
        # Circular Mask Control (only visible in Fluorescence Mode)
        self.mask_layout = QHBoxLayout()
        self.circle_mask_checkbox = QCheckBox("Use Circular Mask")
        self.circle_mask_checkbox.setChecked(False)
        self.circle_mask_checkbox.stateChanged.connect(self.on_mask_changed)
        self.mask_layout.addWidget(self.circle_mask_checkbox)
        layout.addLayout(self.mask_layout)
        
        # Spot Ordering Control (only visible in Fluorescence Mode)
        self.ordering_layout = QHBoxLayout()
        self.ordering_label = QLabel("Spot Ordering:")
        self.ordering_dropdown = QComboBox()
        self.ordering_dropdown.addItems(["reversed", "standard", "p8"])
        self.ordering_dropdown.currentIndexChanged.connect(self.on_ordering_changed)
        self.ordering_layout.addWidget(self.ordering_label)
        self.ordering_layout.addWidget(self.ordering_dropdown)
        layout.addLayout(self.ordering_layout)
        
        # Hide box size and advanced controls by default (Profilometry Mode)
        self.box_size_label.hide()
        self.box_size_input.hide()
        self.circle_mask_checkbox.hide()
        self.ordering_label.hide()
        self.ordering_dropdown.hide()
        
        # 3. Output CSV Selection
        csv_layout = QHBoxLayout()
        self.csv_label = QLabel("Output CSV:")
        self.csv_input = QLineEdit()
        self.csv_input.setText("sample_heights.csv")
        self.csv_btn = QPushButton("Browse")
        self.csv_btn.clicked.connect(self.browse_csv)
        csv_layout.addWidget(self.csv_label)
        csv_layout.addWidget(self.csv_input)
        csv_layout.addWidget(self.csv_btn)
        layout.addLayout(csv_layout)
        
        # 4. Run Button
        self.run_btn = QPushButton("Process Data")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b78e4; color: white;")
        self.run_btn.clicked.connect(self.process_data)
        layout.addWidget(self.run_btn)
        
        # 5. Log Output Area
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        
        main_layout.addLayout(layout, stretch=1)

        # Right side visualization
        viz_main_layout = QVBoxLayout()
        
        self.preview_list = QListWidget()
        self.preview_list.setViewMode(QListWidget.IconMode)
        self.preview_list.setIconSize(QSize(100, 100))
        self.preview_list.setResizeMode(QListWidget.Adjust)
        self.preview_list.setFixedHeight(150)
        self.preview_list.itemClicked.connect(self.display_selected_visualization)
        viz_main_layout.addWidget(self.preview_list)
        
        self.viz_container = QWidget()
        self.viz_container_layout = QVBoxLayout(self.viz_container)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.viz_container_layout.addWidget(self.canvas)
        viz_main_layout.addWidget(self.viz_container, stretch=1)
        
        main_layout.addLayout(viz_main_layout, stretch=2)
        
        self.setLayout(main_layout)
        
        # Cache dictionaries
        self.cached_scans = {}
        self.cached_fluorescence = {}

    def log(self, message):
        """Helper to print messages to the GUI text box."""
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
        QApplication.processEvents()
        
    def on_mode_changed(self, index):
        """Called when operations mode is switched."""
        self.file_input.clear()
        self.preview_list.clear()
        self.figure.clear()
        self.canvas.draw()
        self.cached_scans.clear()
        self.cached_fluorescence.clear()
        
        if index == 0:
            # Profilometry Mode
            self.file_label.setText("OPDx Files:")
            self.box_size_label.hide()
            self.box_size_input.hide()
            self.circle_mask_checkbox.hide()
            self.ordering_label.hide()
            self.ordering_dropdown.hide()
            self.csv_label.setText("Output CSV:")
            self.csv_input.setText("sample_heights.csv")
            self.log("Switched to Profilometry Mode (OPDx).")
        else:
            # Fluorescence Mode
            self.file_label.setText("Fluorescence Images:")
            self.box_size_label.show()
            self.box_size_input.show()
            self.circle_mask_checkbox.show()
            self.ordering_label.show()
            self.ordering_dropdown.show()
            self.csv_label.setText("Output CSV:")
            self.csv_input.setText("compiled_fluorescence_results.csv")
            self.log("Switched to Fluorescence Mode (JPG/PNG).")
            
    def reload_previews(self):
        files_text = self.file_input.text()
        if files_text:
            files = [f.strip() for f in files_text.split(";") if f.strip()]
            self.update_visualizations(files)

    def on_box_size_changed(self):
        """Called when bounding box size is modified."""
        self.reload_previews()

    def on_mask_changed(self, state):
        """Called when circular mask option is toggled."""
        self.reload_previews()

    def on_ordering_changed(self, index):
        """Called when ordering convention is changed."""
        self.reload_previews()

    def browse_files(self):
        is_fluorescence = self.mode_dropdown.currentIndex() == 1
        if not is_fluorescence:
            filenames, _ = QFileDialog.getOpenFileNames(self, "Select OPDx Files", "", "OPDx Files (*.OPDx *.opdx);;All Files (*)")
        else:
            filenames, _ = QFileDialog.getOpenFileNames(self, "Select Fluorescence Images", "", "Image Files (*.jpg *.jpeg *.png);;All Files (*)")
            
        if filenames:
            self.file_input.setText(";".join(filenames))
            self.update_visualizations(filenames)
            
    def browse_csv(self):
        is_fluorescence = self.mode_dropdown.currentIndex() == 1
        default_name = "compiled_fluorescence_results.csv" if is_fluorescence else "sample_heights.csv"
        filename, _ = QFileDialog.getSaveFileName(self, "Save Output CSV", default_name, "CSV Files (*.csv);;All Files (*)")
        if filename:
            self.csv_input.setText(filename)

    def update_visualizations(self, filenames):
        self.preview_list.clear()
        self.figure.clear()
        self.canvas.draw()
        self.cached_scans.clear()
        self.cached_fluorescence.clear()
        QApplication.processEvents()
        
        is_fluorescence = self.mode_dropdown.currentIndex() == 1
        
        for idx, filepath in enumerate(filenames):
            try:
                if not is_fluorescence:
                    loader = DektakLoad(filepath)
                    x, y, z = loader.get_data_2D()
                    self.cached_scans[filepath] = (x, y, z)
                    
                    import matplotlib.cm as cm
                    # Normalize and colorize for thumbnail using matplotlib viridis colormap
                    z_norm_plt = (z - z.min()) / (z.max() - z.min() + 1e-8)
                    z_color_rgba = cm.viridis(z_norm_plt)
                    z_color_rgb = (z_color_rgba[:, :, :3] * 255).astype(np.uint8)
                    z_color_rgb = np.ascontiguousarray(np.flipud(z_color_rgb))
                    
                    h, w, ch = z_color_rgb.shape
                    bytes_per_line = ch * w
                    qimg = QImage(z_color_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                else:
                    self.log(f"Preprocessing preview for {os.path.basename(filepath)}...")
                    rows = self.rows_input.value()
                    cols = self.cols_input.value()
                    box_size = self.box_size_input.value()
                    use_circle_mask = self.circle_mask_checkbox.isChecked()
                    ordering = self.ordering_dropdown.currentText()
                    
                    res = process_fluorescence_image(filepath, rows=rows, cols=cols, box_size=box_size,
                                                     use_circle_mask=use_circle_mask, ordering=ordering)
                    self.cached_fluorescence[filepath] = res
                    
                    # Create thumbnail from the raw color image
                    img_rgb = res['img_rgb']
                    img_rgb_contig = np.ascontiguousarray(img_rgb)
                    h, w, ch = img_rgb_contig.shape
                    bytes_per_line = ch * w
                    qimg = QImage(img_rgb_contig.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                
                icon_pixmap = pixmap.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                icon = QIcon(icon_pixmap)
                
                item = QListWidgetItem(icon, os.path.basename(filepath))
                item.setData(Qt.UserRole, filepath)
                self.preview_list.addItem(item)
                
                if idx == 0:
                    self.preview_list.setCurrentItem(item)
                    self.display_selected_visualization(item)
                    
            except Exception as e:
                self.log(f"Error previewing {filepath}: {e}")
                traceback.print_exc()
                
        if not filenames:
            self.figure.clear()
            self.canvas.draw()

    def display_selected_visualization(self, item=None):
        if item is None or not isinstance(item, QListWidgetItem):
            item = self.preview_list.currentItem()
        if not item:
            return
            
        filepath = item.data(Qt.UserRole)
        is_fluorescence = self.mode_dropdown.currentIndex() == 1
        
        self.figure.clear()
        
        if not is_fluorescence:
            if filepath in self.cached_scans:
                x, y, z = self.cached_scans[filepath]
                ax = self.figure.add_subplot(111)
                im = ax.imshow(z, cmap='viridis', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect='equal')
                self.figure.colorbar(im, ax=ax, label='Height')
                ax.set_title(os.path.basename(filepath))
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Y (μm)')
                self.figure.tight_layout()
                self.canvas.draw()
        else:
            if filepath in self.cached_fluorescence:
                res = self.cached_fluorescence[filepath]
                img_rgb = res['img_rgb']
                global_centers = res['global_centers']
                refined_centers = res['refined_centers']
                
                ax = self.figure.add_subplot(111)
                ax.imshow(img_rgb)
                
                box_size = self.box_size_input.value()
                hw = box_size / 2
                hh = box_size / 2
                
                first_global = True
                first_refined = True
                first_box = True
                
                for r in range(len(global_centers)):
                    for c in range(len(global_centers[r])):
                        g_cx, g_cy = global_centers[r][c]
                        r_cx, r_cy = refined_centers[r][c]
                        
                        # Plot initial global crosses in cyan
                        label_g = "Global Grid" if first_global else ""
                        ax.scatter(g_cx, g_cy, color='cyan', marker='+', s=35, linewidths=1.0, label=label_g)
                        first_global = False
                        
                        # Plot refined centers in red
                        label_r = "Refined Centroid" if first_refined else ""
                        ax.scatter(r_cx, r_cy, color='red', marker='o', s=10, label=label_r)
                        first_refined = False
                        
                        # Plot dynamic green bounding boxes or circular masks
                        x_min = max(0, r_cx - hw)
                        y_min = max(0, r_cy - hh)
                        
                        if self.circle_mask_checkbox.isChecked():
                            if box_size == 51:
                                radius = 30.0
                                c_cx = r_cx + 0.5
                                c_cy = r_cy + 0.5
                            else:
                                radius = 30.0 * (box_size / 51.0)
                                c_cx = r_cx
                                c_cy = r_cy
                            label_b = f"Circular Mask (r={radius:.1f})" if first_box else ""
                            circ = patches.Circle((c_cx, c_cy), radius, linewidth=1.5, edgecolor='lime', facecolor='none', label=label_b)
                            ax.add_patch(circ)
                        else:
                            label_b = f"Bounding Box ({box_size}x{box_size})" if first_box else ""
                            rect = patches.Rectangle((x_min, y_min), box_size, box_size, linewidth=1.5, edgecolor='lime', facecolor='none', label=label_b)
                            ax.add_patch(rect)
                        first_box = False
                
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
                ax.set_title(os.path.basename(filepath))
                ax.axis('on')
                self.figure.tight_layout()
                self.canvas.draw()
            
    def process_data(self):
        files_text = self.file_input.text()
        csv_file = self.csv_input.text()
        rows = self.rows_input.value()
        cols = self.cols_input.value()

        if not files_text or not csv_file:
            QMessageBox.warning(self, "Input Error", "Please specify both the input files and output CSV file paths.")
            return

        files = [f.strip() for f in files_text.split(";") if f.strip()]
        is_fluorescence = self.mode_dropdown.currentIndex() == 1

        if not is_fluorescence:
            all_results = []
            for idx, opdx_file in enumerate(files):
                self.log(f"Loading data from {opdx_file}...")

                try:
                    loader = DektakLoad(opdx_file)
                    x, y, z = loader.get_data_2D()

                    self.log(f"Data loaded successfully. Matrix shape: {z.shape}")
                    self.log(f"Applying Grid Layout: {rows} rows by {cols} cols...")

                    num_samples = rows * cols
                    x_mesh, y_mesh = np.meshgrid(x, y)

                    global_plane, intercept, coeff  = generate_global_plane(z, x_mesh, y_mesh, group_size=10)
                    background_mask = refine_background_mask(z, x_mesh, y_mesh, intercept, coeff)

                    connectivity = 4
                    output_mask_cv = (~background_mask).astype(np.uint8) * 255

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_mask_cv, connectivity, cv2.CV_32S)
                    self.log(f"OpenCV found {num_labels} raw components.")

                    new_num_labels, filtered_labels, filtered_stats, filtered_centroids = filter_components(num_labels, labels, stats, centroids)
                    self.log(f"After filtering, {new_num_labels} components remain.")

                    final_labels, final_stats, final_centroids = reorder_components(num_labels, stats, centroids, rows, cols)
                    self.log(f"Final count for height calculation: {len(final_stats)}")

                    height_results = calculate_heights(z, x_mesh, y_mesh, final_stats, final_centroids, background_mask, intercept, coeff, num_samples)

                    filename = os.path.basename(opdx_file)
                    all_results.append((filename, height_results))

                except Exception as e:
                    traceback.print_exc()
                    self.log(f"Error processing {opdx_file} at: {traceback.format_exc().splitlines()[-2]}")
                    self.log(f"Message: {str(e)}")
                    continue

            if all_results:
                import csv
                try:
                    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if all_results and all_results[0][1]:
                            columns = list(all_results[0][1][0].keys())
                        else:
                            columns = ['component_id', 'centroid_x', 'centroid_y', 'top', 'bottom', 'difference']

                        for file_idx, (filename, file_results) in enumerate(all_results):
                            writer.writerow([f"{filename}"])
                            writer.writerow(columns)

                            for i, res in enumerate(file_results):
                                row = [res.get(col, '') for col in columns]
                                writer.writerow(row)
                                if (i + 1) % 16 == 0 and (i + 1) != len(file_results):
                                    writer.writerow([])
                                    
                            if file_idx < len(all_results) - 1:
                                writer.writerow([])

                    self.log(f"Calculation complete. Results successfully saved to {csv_file}")
                    QMessageBox.information(self, "Success", f"Results successfully saved to:\n{csv_file}")
                except Exception as e:
                    traceback.print_exc()
                    self.log(f"Error saving CSV to {csv_file}: {e}")
                    QMessageBox.critical(self, "Save Error", f"Failed to save CSV to:\n{csv_file}\n\nError: {str(e)}")
            else:
                self.log("No results were generated. Check for errors.")
        else:
            self.log("Starting Fluorescence Brightness Extraction...")
            box_size = self.box_size_input.value()
            use_circle_mask = self.circle_mask_checkbox.isChecked()
            ordering = self.ordering_dropdown.currentText()
            processed_data = {}

            for filepath in files:
                self.log(f"Processing image: {filepath}...")
                try:
                    res = process_fluorescence_image(filepath, rows=rows, cols=cols, box_size=box_size,
                                                     use_circle_mask=use_circle_mask, ordering=ordering)
                    df_flu = res['df']
                    img_base_name = os.path.splitext(os.path.basename(filepath))[0]
                    processed_data[img_base_name] = df_flu
                    
                    # Also save individual CSV in the same directory
                    base_dir = os.path.dirname(filepath)
                    csv_filename = os.path.join(base_dir, f"fluorescence_data_{img_base_name}.csv")
                    
                    area_val = box_size * box_size
                    with open(csv_filename, 'w', encoding='utf-8', newline='') as f:
                        import csv
                        writer = csv.writer(f)
                        writer.writerow([img_base_name])
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
                            if (i + 1) % 16 == 0 and (i + 1) != len(df_flu):
                                writer.writerow([])
                    self.log(f"Saved individual results to: {csv_filename}")
                    
                except Exception as e:
                    traceback.print_exc()
                    self.log(f"Error processing image {filepath}: {e}")
                    continue
            
            if processed_data:
                # Compile into the single Excel-matching CSV
                first_dir = os.path.dirname(files[0])
                template_path = os.path.join(first_dir, "October2024Flu.xlsx")
                if not os.path.exists(template_path):
                    template_path = "/home/chris/Auto_OPDx_Fl/Fluorescence Data - OCTOBER 2024 SCR-FL/POST WASH - OCTOBER 2024 SCR-FL/October2024Flu.xlsx"
                
                self.log(f"Compiling results into {csv_file}...")
                self.log(f"Using template file: {template_path if os.path.exists(template_path) else '(none - simple CSV fallback)'}")
                
                try:
                    success, msg = compile_fluorescence_results(template_path, processed_data, csv_file, box_size=box_size)
                    self.log(msg)
                    if success:
                        self.log(f"Fluorescence data processing complete! Saved to {csv_file}")
                        QMessageBox.information(self, "Success", f"Fluorescence data successfully saved to:\n{csv_file}")
                    else:
                        self.log(f"Fluorescence processing completed with warnings: {msg}")
                        QMessageBox.warning(self, "Save Warning", f"Fluorescence processing completed with warning:\n{msg}")
                except Exception as e:
                    traceback.print_exc()
                    self.log(f"Error compiling fluorescence results: {e}")
                    QMessageBox.critical(self, "Save Error", f"Failed to save fluorescence results to:\n{csv_file}\n\nError: {str(e)}")
            else:
                self.log("No images were successfully processed.")

def main():
    app = QApplication(sys.argv)
    window = ProfilometryApp()
    window.show()
    sys.exit(app.exec_())
