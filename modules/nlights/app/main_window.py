"""
Main application window for the Photometric Stereo application
This implements the main GUI previously defined in the NLights MATLAB class
"""

import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                            QLabel, QPushButton, QFileDialog, QMessageBox,QLineEdit,
                            QTabWidget, QScrollArea, QFrame, QGridLayout,
                            QGroupBox, QComboBox, QCheckBox, QSpinBox,
                            QDoubleSpinBox, QProgressBar, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QDir
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage

from core.photometric_stereo import PhotometricStereo
from core.normal_maps import compute_normal_maps_new
from core.depth_maps import compute_depth_maps
#from ui.widgets.interactive_table import LightDirectionTable
from utils.image_processing import specularize_x

import configparser



class ProcessingThread(QThread):
    """Worker thread to handle image processing operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    

    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        try:
            # This would be called with all the processing steps

            #QUI INVOCA IL PROCESSING -> normal_maps, depth_maps, ecc
            if(self.config['output_directory'] == ''):
                self.progress.emit(0)
                raise ValueError("Output directory not set. Please select an output directory.")

            self.progress.emit(0)
            results = compute_normal_maps_new(self.config, self.progress)
            #result = {}
            
            
            # Signal progress as processing continues
            #for i in range(101):
            #    self.progress.emit(i)
            #    self.msleep(20)  # Just for demonstration
            
            #self.finished.emit(results) 
        except Exception as e:
            self.progress.emit(0)
            print(f"Error during processing: {e.__traceback__}")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window for Photometric Stereo"""
    
    def __init__(self):
        super().__init__()

        # Crea l'oggetto ConfigParser
        self.configIni = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        self.configIni.read(config_path)

        light_strenght_file_path = self.configIni['Assets']['light_strenght_file_path']

        self.lights_file_path_def_v = self.configIni['Assets']['lights_file_path_def_v']
        self.black_error_v = self.configIni['Assets']['black_error_v']
        self.black_error_2_v = self.configIni['Assets']['black_error_2_v']

        self.lights_file_path_def_h = self.configIni['Assets']['lights_file_path_def_h']
        self.black_error_h = self.configIni['Assets']['black_error_h']
        self.black_error_2_h = self.configIni['Assets']['black_error_2_h']

        lights_file_path_cust_v = self.configIni['Assets']['lights_file_path_cust_v']
        lights_file_path_cust_h = self.configIni['Assets']['lights_file_path_cust_h']
        
        self.config = {
            'light_directions': None,
            "lights_file_path": self.lights_file_path_def_h, #"./assets/Default_Horizontal.txt",
            #"custom_lights_file_path": "",
            "black_error": self.black_error_h, #"./assets/Black_Error_Horizontal.tif",
            "black_error_2": self.black_error_2_h, #"./assets/Black_Error_Horizontal_2.tif",
            #"black_error_v": "",
            #"black_error_v2": "",
            "light_strenght_file_path": light_strenght_file_path, #"./assets/lightStrenght.txt",
            "loadOptions":None,
            #'input_images': [],
            #'working_directory': '',
            'output_directory': '',
            'all_lights_on_image': '',
            'DDLightsOrder':'45N...15W',
            'StrelSize': 5,
            'light_direction_images': [],
            'height_map_format': 'PLY',
            'decimation_method': 'Accurate',
            'decimation_surface': '1',
            'poly_correction': True,
            'image_downsample': True,
            'normal_map': None,
            'depth_map': None,
            'albedo': None
        }
        
        #QtGui.QImageReader.setAllocationLimit(0)

        self.processing_thread = None
        
        self.init_ui()
        self.setWindowTitle("Photometric Stereo")
        self.resize(1200, 800)
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget for different sections
        tab_widget = QTabWidget()
        
        # Create tabs
        input_tab = self.create_input_tab()
        #processing_tab = self.create_processing_tab()
        results_tab = self.create_results_tab()
        #export_tab = self.create_export_tab()
        
        # Add tabs to the tab widget
        tab_widget.addTab(input_tab, "Setup, Input, Process")
        #tab_widget.addTab(processing_tab, "Processing")
        #tab_widget.addTab(results_tab, "Results")
        #tab_widget.addTab(export_tab, "Export")
        
        # Add tab widget to the main layout
        main_layout.addWidget(tab_widget)
        
        # Add status bar for messages
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Set the central widget
        self.setCentralWidget(central_widget)
    
    def create_input_tab(self):
        """Create the input images tab"""
        tab = QWidget()
        #layout = QVBoxLayout(tab)
        layout = QGridLayout(tab)
        
        # Create setup dropdown
        setup_group = QGroupBox("Setup Section")
        setup_layout = QGridLayout(setup_group)
        self.setup_dropdown = QComboBox()
        self.setup_dropdown.addItems(["Horizontal","Vertical"])
        self.setup_dropdown.currentIndexChanged.connect(self.setup_changed)
        setup_layout.addWidget(QLabel("Setup"),0,0)
        setup_layout.addWidget(self.setup_dropdown,0,1)

        #Create button for Light Position Customization
        customize_light_button = QPushButton("Customize Light Positions")
        customize_light_button.clicked.connect(self.customize_light_positions)
        setup_layout.addWidget(QLabel("Light Position"),1,0)
        setup_layout.addWidget(customize_light_button,1,1)


        self.output_dir_label = QLabel("No directory selected")
        select_output_dir = QPushButton("Select Output Directory...")
        select_output_dir.clicked.connect(self.select_output_directory)
        
        setup_layout.addWidget(self.output_dir_label,2,0)
        setup_layout.addWidget(select_output_dir,2,1)






        # Create group for selecting input directory
        input_group = QGroupBox("Input Images")
        input_layout = QVBoxLayout(input_group)
        
        # Add buttons for selecting input directory and files
        select_dir_button = QPushButton("Select Image with all lights...")
        select_dir_button.clicked.connect(self.select_input_allLights)
        input_layout.addWidget(select_dir_button)
        
        # Add container for image previews
        self.image_preview_layout = QGridLayout()
        input_layout.addLayout(self.image_preview_layout)
        

        # Add group for light direction settings
        light_group = QGroupBox("Light Directions")
        light_layout = QVBoxLayout(light_group)
        
        # light direction selection
        self.select_ldir_button = QPushButton("Select Light Direction Images...")
        self.select_ldir_button.clicked.connect(self.select_light_direction_images)
        light_layout.addWidget(self.select_ldir_button)

        self.ldir_preview_layout = QGridLayout()
        light_layout.addLayout(self.ldir_preview_layout)


        #OTHER INPUTS GROUPBOX
        other_inputs_group = QGroupBox("Configuration Inputs")
        other_inputs_layout = QGridLayout(other_inputs_group)

        self.light_order_dropdown = QComboBox()
        self.light_order_dropdown.addItems(["45N ...15W", "45W... 15S"])
        self.light_order_dropdown.currentIndexChanged.connect(self.light_order_changed)
        other_inputs_layout.addWidget(QLabel("Light Order"),0,0)
        other_inputs_layout.addWidget(self.light_order_dropdown,0,1)

        self.height_map_dropdown = QComboBox()
        self.height_map_dropdown.addItems(["PLY", "STL"])
        self.height_map_dropdown.currentIndexChanged.connect(self.height_map_changed)
        other_inputs_layout.addWidget(QLabel("Height Map Format"),1,0)
        other_inputs_layout.addWidget(self.height_map_dropdown,1,1)

        self.decimation_method_dropdown = QComboBox()
        self.decimation_method_dropdown.addItems(["Accurate", "Coarse"])
        self.decimation_method_dropdown.currentIndexChanged.connect(self.decimation_method_changed)
        other_inputs_layout.addWidget(QLabel("Decimation Method"),2,0)
        other_inputs_layout.addWidget(self.decimation_method_dropdown,2,1)
        

        self.dec_surface_text = QLineEdit()
        self.dec_surface_text.setText("1")
        self.dec_surface_text.editingFinished.connect(self.dec_surface)
        other_inputs_layout.addWidget(QLabel("Decimation Surface"),3,0)
        other_inputs_layout.addWidget(self.dec_surface_text, 3, 1)

 
        self.close_figures_button = QPushButton("Close all")
        self.close_figures_button.clicked.connect(self.close_all)
        other_inputs_layout.addWidget(QLabel("Figures"),4,0)
        other_inputs_layout.addWidget(self.close_figures_button, 4, 1)

        


        self.poly = QCheckBox("Poly correction")
        self.poly.setChecked(True)
        self.poly.stateChanged.connect(self.poly_correction_changed)
        other_inputs_layout.addWidget(self.poly, 5, 1)

        self.downsample = QCheckBox("Image downsample")
        self.downsample.setChecked(True)
        self.downsample.stateChanged.connect(self.image_downsample_changed)
        other_inputs_layout.addWidget(self.downsample, 5, 2)



        # Progress Group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")

        # Process button
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.process_button)
        
        




        
        # Add groups to main layout
        layout.addWidget(setup_group)
        layout.addWidget(input_group)
        layout.addWidget(light_group)
        layout.addWidget(other_inputs_group)
        layout.addWidget(progress_group)
        
        return tab
    
    
    
    def create_processing_tab(self):
        """Create the processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout(options_group)
        
        # Add various processing options
        options_layout.addWidget(QLabel("Normal Map Generation:"), 0, 0)
        normal_method = QComboBox()
        normal_method.addItems(["Standard Photometric Stereo", "Advanced (NLLS)"])
        options_layout.addWidget(normal_method, 0, 1)
        
        options_layout.addWidget(QLabel("Depth Map Generation:"), 1, 0)
        depth_method = QComboBox()
        depth_method.addItems(["Gradient Integration", "Poisson"])
        options_layout.addWidget(depth_method, 1, 1)
        
        options_layout.addWidget(QLabel("Polynomial Correction:"), 2, 0)
        poly_correction = QCheckBox("Apply polynomial correction")
        options_layout.addWidget(poly_correction, 2, 1)
        
        # Progress indicator
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        
        # Process button
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        
        # Add all widgets to layout
        layout.addWidget(options_group)
        layout.addWidget(progress_group)
        layout.addWidget(self.process_button)
        
        return tab
    
    def create_results_tab(self):
        """Create the results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create splitter for main sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Results selection panel (left side)
        selection_widget = QWidget()
        selection_layout = QVBoxLayout(selection_widget)
        
        selection_layout.addWidget(QLabel("<b>Available Results:</b>"))
        
        # Buttons for different result types
        self.normal_map_button = QPushButton("Normal Map")
        self.normal_map_button.clicked.connect(lambda: self.show_result("normal_map"))
        
        self.depth_map_button = QPushButton("Depth Map")
        self.depth_map_button.clicked.connect(lambda: self.show_result("depth_map"))
        
        self.albedo_button = QPushButton("Albedo")
        self.albedo_button.clicked.connect(lambda: self.show_result("albedo"))
        
        selection_layout.addWidget(self.normal_map_button)
        selection_layout.addWidget(self.depth_map_button)
        selection_layout.addWidget(self.albedo_button)
        selection_layout.addStretch()
        
        # Results display panel (right side)
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        
        self.result_label = QLabel("No results to display yet")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        display_layout.addWidget(self.result_label)
        
        # Add widgets to splitter
        splitter.addWidget(selection_widget)
        splitter.addWidget(display_widget)
        splitter.setSizes([200, 600])  # Initial sizes
        
        layout.addWidget(splitter)
        
        return tab
    
    def create_export_tab(self):
        """Create the export tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Output directory selection
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout(dir_group)
        
        self.output_dir_label = QLabel("No directory selected")
        select_output_dir = QPushButton("Select Output Directory...")
        select_output_dir.clicked.connect(self.select_output_directory)
        
        dir_layout.addWidget(self.output_dir_label)
        dir_layout.addWidget(select_output_dir)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QGridLayout(export_group)
        
        # Normal map export
        export_layout.addWidget(QLabel("Normal Map:"), 0, 0)
        self.export_normal_checkbox = QCheckBox("Export Normal Map")
        self.export_normal_checkbox.setChecked(True)
        export_layout.addWidget(self.export_normal_checkbox, 0, 1)
        
        # Depth map export
        export_layout.addWidget(QLabel("Depth Map:"), 1, 0)
        self.export_depth_checkbox = QCheckBox("Export Depth Map")
        self.export_depth_checkbox.setChecked(True)
        export_layout.addWidget(self.export_depth_checkbox, 1, 1)
        
        # 3D model export
        export_layout.addWidget(QLabel("3D Model:"), 2, 0)
        self.export_3d_checkbox = QCheckBox("Export 3D Model")
        self.export_3d_checkbox.setChecked(True)
        export_layout.addWidget(self.export_3d_checkbox, 2, 1)
        
        # 3D format selection
        export_layout.addWidget(QLabel("3D Format:"), 3, 0)
        self.export_3d_format = QComboBox()
        self.export_3d_format.addItems(["OBJ", "STL", "PLY"])
        export_layout.addWidget(self.export_3d_format, 3, 1)
        
        # Albedo export
        export_layout.addWidget(QLabel("Albedo:"), 4, 0)
        self.export_albedo_checkbox = QCheckBox("Export Albedo")
        self.export_albedo_checkbox.setChecked(True)
        export_layout.addWidget(self.export_albedo_checkbox, 4, 1)
        
        # ICC profile
        export_layout.addWidget(QLabel("Color Profile:"), 5, 0)
        self.icc_profile_combo = QComboBox()
        self.icc_profile_combo.addItems(["None", "sRGB", "Display P3"])
        export_layout.addWidget(self.icc_profile_combo, 5, 1)
        
        # Export button
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.export_results)
        
        # Add all widgets to layout
        layout.addWidget(dir_group)
        layout.addWidget(export_group)
        layout.addWidget(export_button)
        layout.addStretch()
        
        return tab
    
    
    
    # TODO: AGGIUNGI ANCHE I CUSTOM!!!
    def setup_changed(self):
        """Handle changes in the setup dropdown"""
        setup = self.setup_dropdown.currentText()
        if setup == "Vertical":
            self.config["lights_file_path"] = self.lights_file_path_def_v #"./assets/Default_Vertical.txt"
            self.config["black_error"] = self.black_error_v #"./assets/Black_Error_Vertical.tif"
            self.config["black_error_2"] = self.black_error_2_v #"./assets/Black_Error_Vertical_2.tif"
        else:
            self.config["lights_file_path"] = self.lights_file_path_def_h #"./assets/Default_Horizontal.txt"
            self.config["black_error"] = self.black_error_h #"./assets/Black_Error_Horizontal.tif"
            self.config["black_error_2"] = self.black_error_2_h #"./assets/Black_Error_Horizontal_2.tif"
        #self.console_output.setText(f"Setup changed to: {setup}")
        #VALUTARE DI CREARE create_console
    
    #CAMBIARE INPUT! -> prevedere immissione file
    def customize_light_positions(self):
        """Handle changes in the setup dropdown
        setup = self.setup_dropdown.currentText()
        if setup == "Vertical":
            self.config["lights_file_path"] = "./assets/Custom_Vertical.txt"
        else:
            self.config["lights_file_path"] = "./assets/Custom_Horizontal.txt"
        """

        light_pos_path, _  = QFileDialog.getOpenFileName(
            self, "Select custom file for light positions", "", "Files (*.txt)")
        #QFileDialog.getExistingDirectory(self, "Select Input Image Directory")
        if light_pos_path:
            self.config["lights_file_path"] = light_pos_path
        #else:
        #    raise ValueError("Light positions file not selected. Please select a custom file.")
        #self.console_output.setText(f"Setup changed to Custom: {setup}")
    
    def light_order_changed(self):
        """Handle changes in the setup dropdown"""
        self.config['DDLightsOrder'] = self.light_order_dropdown.currentText()
        #self.console_output.setText(f"Setup changed to: {setup}")

    def height_map_changed(self):
        """Handle changes in the setup dropdown"""
        self.config['height_map_format'] = self.height_map_dropdown.currentText()
        #self.console_output.setText(f"Setup changed to: {setup}")

    def decimation_method_changed(self):
        """Handle changes in the setup dropdown"""
        self.config['decimation_method'] = self.decimation_method_dropdown.currentText()
        #self.console_output.setText(f"Setup changed to: {setup}")

    def close_all(self):
        self.clear_layout(self.image_preview_layout)
        self.clear_layout(self.ldir_preview_layout)

    def poly_correction_changed(self):
        self.config['poly_correction'] = self.poly.isChecked()

    def image_downsample_changed(self):
        self.config['image_downsample'] = self.downsample.isChecked()

    def dec_surface(self):
        self.config['decimation_surface'] = self.dec_surface_text.text()

    def select_input_allLights(self):
        """Open a dialog to select the input directory containing images"""
        image_path, _  = QFileDialog.getOpenFileName(
            self, "Select All Lights Image", "", "Image Files (*.jpg *.jpeg *.png *.tif *.tiff)")
        #QFileDialog.getExistingDirectory(self, "Select Input Image Directory")
        if image_path:
            self.config["all_lights_on_image"] = image_path
            #self.config['working_directory'] = directory
            print(image_path)
            image_files = [image_path]
            self.display_preview(image_files=image_files)

    def display_preview(self, image_files):#directory):
        """Load images from the selected directory"""
        # Clear current images
        #self.clear_layout(self.image_preview_layout)
        '''self.config['input_images'] = []
        
        # Find image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(QDir(directory).entryList([ext], QDir.Filter.Files))'''
        
        # Display preview of found images
        if image_files:
            # Store image paths
            #self.config['input_images'] = [os.path.join(directory, f) for f in image_files]
            
            # Create image previews (up to 16)
            max_preview = min(16, len(image_files))
            rows = (max_preview + 3) // 4  # Ceil division for rows

            if(max_preview == 1):
                self.clear_layout(self.image_preview_layout)
            else:
                self.clear_layout(self.ldir_preview_layout)
            
            for i in range(max_preview):
                image_path = image_files[i] #os.path.join(directory, image_files[i])
                preview = QLabel()
                pixmap = QPixmap(image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
                preview.setPixmap(pixmap)
                preview.setToolTip(image_files[i])
                
                row = i // 4
                col = i % 4
                if(max_preview == 1):
                    self.image_preview_layout.addWidget(preview, row, col)
                else:
                    self.ldir_preview_layout.addWidget(preview, row, col)
            # Update status
            self.status_bar.showMessage(f"Loaded {len(image_files)} images ")
        else:
            QMessageBox.warning(self, "No Images Found", 
                               f"No image files found in the selected directory")
    
    def select_light_direction_images(self):
        """Open dialog to select a chrome sphere image for light direction calculation"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Light Direction Images", "", "Image Files (*.jpg *.jpeg *.png *.tif *.tiff)")
        
        if file_paths:
            self.config['light_direction_images'] = file_paths

            self.display_preview(image_files=file_paths)

            # Here you would call the chrome sphere processing function
            # and populate the light direction table with the results
            #self.status_bar.showMessage(f"Chrome sphere image selected: {os.path.basename(file_paths)}") 
    
    def select_output_directory(self):
        """Open a dialog to select the output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.config['output_directory'] = directory
            self.output_dir_label.setText(directory)
    
    def start_processing(self):
        """Begin photometric stereo processing"""
        # Check if we have all necessary inputs
        if not self.config['all_lights_on_image']:
            QMessageBox.warning(self, "Missing Input", "No input images selected")
            return
        
        if not self.config['light_direction_images'] :
            QMessageBox.warning(self, "Missing Light Directions", 
                               "Please load light directions")
            return
        

        self.config['loadOptions'] ={
            'ImageChannel': 1,
            'NormalizePercentile': 99, #IN MATLAB è STATICO!!!
            'resample': self.config['image_downsample'] #PRENDI DAL CHECKBOX!!
        }

        # Update UI
        self.progress_bar.setValue(0)
        self.process_button.setEnabled(False)
        self.progress_label.setText("Processing...")
        
        # Create worker thread
        self.processing_thread = ProcessingThread(self.config)
        self.status_bar.showMessage("Processing...")
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        
        # Start processing
        self.processing_thread.start()



    @pyqtSlot(int)
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    @pyqtSlot(dict)
    def processing_finished(self, results):
        """Handle completed processing"""
        # Update UI
        self.process_button.setEnabled(True)
        self.progress_label.setText("Processing completed")
        
        # Store results
        if 'normal_map' in results:
            self.config['normal_map'] = results['normal_map']
        
        if 'depth_map' in results:
            self.config['depth_map'] = results['depth_map']
        
        if 'albedo' in results:
            self.config['albedo'] = results['albedo']
        
        # Enable result buttons
        #self.normal_map_button.setEnabled('normal_map' in results)
        #self.depth_map_button.setEnabled('depth_map' in results)
        #self.albedo_button.setEnabled('albedo' in results)
        
        # Show message
        self.status_bar.showMessage("Processing completed successfully")
    
    @pyqtSlot(str)
    def processing_error(self, error_msg):
        """Handle processing errors"""
        self.process_button.setEnabled(True)
        self.progress_label.setText("Error during processing")
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.status_bar.showMessage("Error during processing")
    
    def show_result(self, result_type):
        """Display the selected result type"""
        if result_type not in self.config or self.config[result_type] is None:
            self.result_label.setText(f"No {result_type.replace('_', ' ')} available")
            return
        
        # Convert the numpy array to a QImage and display it
        # This is a placeholder for the actual display logic
        self.result_label.setText(f"Displaying {result_type.replace('_', ' ')}")
    
    def export_results(self):
        """Export results to the selected output directory"""
        if not self.config['output_directory']:
            QMessageBox.warning(self, "No Output Directory", 
                               "Please select an output directory first")
            return
        
        # Check if we have results to export
        has_results = any(self.config.get(key) is not None for key in 
                          ['normal_map', 'depth_map', 'albedo'])
        
        if not has_results:
            QMessageBox.warning(self, "No Results", 
                               "No results available to export")
            return
        
        # Perform export (placeholder)
        self.status_bar.showMessage("Exporting results...")
        
        # In a real app, you would call your export functions here
        
        QMessageBox.information(self, "Export Complete", 
                               "Results exported successfully")
        self.status_bar.showMessage("Export completed")
    
    def clear_layout(self, layout):
        """Clear all widgets from a layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())