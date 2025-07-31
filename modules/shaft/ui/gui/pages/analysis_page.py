# from PyQt5.QtCore import pyqtSignal
# from PyQt5.QtGui import QIcon
# from PyQt5.QtWidgets import (
#     QFileDialog,
#     QFrame,
#     QGridLayout,
#     QHBoxLayout,
#     QMessageBox,
#     QSizePolicy,
#     QSpacerItem,
#     QVBoxLayout,
#     QWidget,
# )

from config.config import SKIP_PARAMS, AnalysisSteps, Mode
from core.core import Core
from core.store.store import appStore
from ui.gui.common.manual_correction_wizard.manual_correction_wizard import (
    ManualCorrectionWizard,
)
from ui.gui.common.phase_manager import PhaseManager
from ui.gui.configs.main_window_config import COLOR_SPACES, FILE_FORMATS
from ui.gui.contexts.pages_context import PagesContext
from ui.gui.styles.main_window_style import *
from ui.gui.utils.gui_utils import (
    create_qt_button,
    create_qt_checkbox_widget,
    create_qt_dropdown,
    create_qt_label,
    get_color_space_name_by_id,
    get_format_name_by_id,
)
from ui.process_threads.worker_thread_qt import WorkerThreadQT


class AnalysisPage(QWidget):
    update_progress = pyqtSignal(int)
    worker_started = pyqtSignal()
    worker_finished = pyqtSignal(bool)

    def __init__(self, parent, context: PagesContext):
        super().__init__()

        self.parent = parent
        self.context = context
        self.context.set_color_space_index(1) #Dovrebbe leggerlo dai default in realtà

        self.analysis_output_path = None
        self.worker_thread: WorkerThreadQT = None
        self.core: Core = None
        self.input_path = None
        self.white_field_path = None
        self.fitting_degree = 2
        self.sharpen = None
        self.light_balance = None

        # Skip parameters
        self.skip_params = {
            SKIP_PARAMS.CCM: False,
            SKIP_PARAMS.POLYNOMIAL_FITTING: False,
            SKIP_PARAMS.SHAFT: True,
            SKIP_PARAMS.WHITE_BALANCE: False,
            SKIP_PARAMS.DENOISING: False,
            SKIP_PARAMS.FINLAYSON_CCM: False,
            SKIP_PARAMS.RIDGE_CCM: False,
            SKIP_PARAMS.WLS: False,
            SKIP_PARAMS.EXPOSURE: False,
            SKIP_PARAMS.FLAT_FIELDING: False,
            SKIP_PARAMS.RAWTHERAPEE: False,
        }

        page1_layout = QHBoxLayout()
        self.setLayout(page1_layout)

        content_frame = QFrame()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(20, 20, 0, 0)  # Align with the shaft icon
        content_layout.setSpacing(20)
        content_frame.setLayout(content_layout)

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(20)

        # First Row: Title
        title = self.create_title()
        grid_layout.addWidget(title, 0, 0, 1, 2)

        # Second Row: Colorchecker
        colorchecker_label, colorchecker_button = (
            self.create_colorchecker_path_selection()
        )
        # 3rd row
        white_label, white_button = (
            self.create_white_path_selection()
        )
        grid_layout.addWidget(colorchecker_label, 1, 0)
        grid_layout.addWidget(colorchecker_button, 1, 1)

        grid_layout.addWidget(white_label, 2, 0)
        grid_layout.addWidget(white_button, 2, 1)

        # Fourth Row: Output path
        output_label, output_button = self.create_output_path_selection()
        grid_layout.addWidget(output_label, 3, 0)
        grid_layout.addWidget(output_button, 3, 1)

        # Fifth Row: CCM
        ccm_label, ccm_widget = self.create_ccm_checkbox()
        grid_layout.addWidget(ccm_label, 4, 0)
        grid_layout.addWidget(ccm_widget, 4, 1)

        # Sixth Row: Fitting
        fitting_label, polyfit_widget = self.create_fitting_checkbox()
        grid_layout.addWidget(fitting_label, 5, 0)
        grid_layout.addWidget(polyfit_widget, 5, 1)

        fitting_degree_text, fitting_degree_dropdown = self.create_fitting_dropdown()
        grid_layout.addWidget(fitting_degree_text, 5, 2)
        grid_layout.addWidget(fitting_degree_dropdown, 5, 3)

        # Seventh Row: Shaft
        shaft_label, shaft_widget = self.create_shaft_checkbox()
        grid_layout.addWidget(shaft_label, 6, 0)
        grid_layout.addWidget(shaft_widget, 6, 1)

        # Eight Row: Rawtherapee / Postprocessing
        sharpen_label, sharpen_widget = self.create_sharpen_checkbox()
        grid_layout.addWidget(sharpen_label, 7, 0)
        grid_layout.addWidget(sharpen_widget, 7, 1)

        light_balance_label, light_balabce_widget = self.create_light_balance_checkbox()
        grid_layout.addWidget(light_balabce_widget, 7, 2)


        # Nineth Row: Output color space and Output file format
        output_color_space_label, output_color_space_dropdown = (
            self.create_output_color_space_dropdown()
        )
        grid_layout.addWidget(output_color_space_label, 8, 0)
        grid_layout.addWidget(output_color_space_dropdown, 8, 1)

        output_file_format_label, output_file_format_dropdown = (
            self.create_output_file_format_dropdown()
        )
        grid_layout.addWidget(output_file_format_label, 8, 2)
        grid_layout.addWidget(output_file_format_dropdown, 8, 3)

        # Add the grid layout to the content layout
        content_layout.addLayout(grid_layout)

        # Add the phase manager
        self.phase_manager = PhaseManager("Analyze Color Checker", self)
        self.phase_manager.start_phase.connect(self.start_analysis)
        self.phase_manager.stop_worker.connect(self.stop_worker)
        content_layout.addLayout(self.phase_manager)

        page1_layout.addWidget(content_frame)

        # Add a spacer to take up the remaining space on the right
        page1_layout.addSpacerItem(
            QSpacerItem(999 // 2, 478, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

    def on_page_changed(self):
        pass

    def create_title(self):
        text = "Checkerboard detection and analysis"

        title = create_qt_label(text, common_title_label_style)

        return title

    def create_colorchecker_path_selection(self):
        label_text = "Colorchecker path"
        button_text = "Browse"

        label = create_qt_label(label_text, common_label_style)
        button = create_qt_button(button_text, lambda: self.select_input_file(button))

        return label, button

    def create_white_path_selection(self):
        label_text = "White path"
        button_text = "Browse"

        label = create_qt_label(label_text, common_label_style)
        button = create_qt_button(button_text, lambda: self.select_white_file(button))

        return label, button

    def create_output_path_selection(self):
        label_text = "Output path"
        button_text = "Browse"

        label = create_qt_label(label_text, common_label_style)
        button = create_qt_button(button_text, lambda: self.select_output_file(button))

        return label, button

    def create_ccm_checkbox(self):
        label_text = "CCM"
        checkbox_text = "Use Imatest Color Correction Matrix"
        checkbox_type = SKIP_PARAMS.CCM

        label = create_qt_label(label_text, common_label_style)
        widget = create_qt_checkbox_widget(
            checkbox_type,
            lambda state: self.update_skip_params_checkbox(state, checkbox_type),
            checkbox_text,
            True,
        )

        return label, widget

    def create_fitting_checkbox(self):
        label_text = "Fitting"
        checkbox_text = "Use Polyfit"
        checkbox_type = SKIP_PARAMS.POLYNOMIAL_FITTING

        label = create_qt_label(label_text, common_label_style)
        widget = create_qt_checkbox_widget(
            checkbox_type,
            lambda state: self.update_skip_params_checkbox(state, checkbox_type),
            checkbox_text,
            True,
        )

        return label, widget

    def create_fitting_dropdown(self):
        text = "Fitting degree"
        items = ["1", "2", "3"]

        label, dropdown = create_qt_dropdown(text, items, 1)
        dropdown.currentTextChanged.connect(self.update_fitting_degree)

        return label, dropdown

    def update_fitting_degree(self, value):
        self.fitting_degree = value

    def create_shaft_checkbox(self):
        label_text = "Shaft"
        checkbox_text = "Use Shaft"
        checkbox_type = SKIP_PARAMS.SHAFT

        label = create_qt_label(label_text, common_label_style)
        widget = create_qt_checkbox_widget(
            checkbox_type,
            lambda state: self.update_skip_params_checkbox(state, checkbox_type),
            checkbox_text,
        )

        return label, widget

    def create_sharpen_checkbox(self):
        label_text = "Postprocessing"
        checkbox_text = "Sharpen"
        checkbox_type = self.sharpen

        label = create_qt_label(label_text, common_label_style)
        widget = create_qt_checkbox_widget(
            checkbox_type,
            lambda state: self.update_sharpen_checkbox(state),
            checkbox_text,
        )
        # widget.stateChanged.connect(self.sharpen)
        return label, widget

    def create_light_balance_checkbox(self):
        label_text = ""
        checkbox_text = "Light balance"
        checkbox_type = self.light_balance

        label = create_qt_label(label_text, common_label_style)
        widget = create_qt_checkbox_widget(
            checkbox_type,
            lambda state: self.update_light_balance_checkbox(state),
            checkbox_text,
        )
        #widget.stateChanged.connect(self.light_balance)
        return label, widget

    def create_output_color_space_dropdown(self):
        text = "Output color space"
        items = [color_space["name"] for color_space in COLOR_SPACES]

        label, dropdown = create_qt_dropdown(text, items, 1)

        dropdown.currentIndexChanged.connect(self.context.set_color_space_index)
        return label, dropdown

    def create_output_file_format_dropdown(self):
        text = "Output file format"
        items = [file_format["name"] for file_format in FILE_FORMATS]

        label, dropdown = create_qt_dropdown(text, items)
        dropdown.currentIndexChanged.connect(self.context.set_file_format_index)

        return label, dropdown

    def handle_start_n_stop(self, is_start):
        self.phase_manager.handle_start_n_stop(is_start)

    def select_input_file(self, button):
        options = QFileDialog.Options()
        self.input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select the file containing the target",
            "",
            "All Files (*);;Text Files (*.txt)",
            options=options,
        )
        if self.input_path:
            button.setText(self.input_path)
            button.setToolTip(self.input_path)
            self.check_enable_start()

    def select_white_file(self, button):
        options = QFileDialog.Options()
        self.white_field_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File containing the white field",
            "",
            "All Files (*);;Text Files (*.txt)",
            options=options,
        )
        if self.white_field_path:
            button.setText(self.white_field_path)
            button.setToolTip(self.white_field_path)
            self.check_enable_start()

    def select_output_file(self, button):
        self.analysis_output_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", "", QFileDialog.ShowDirsOnly
        )
        if self.analysis_output_path:
            self.context.set_analysis_output_path(self.analysis_output_path)
            button.setText(self.analysis_output_path)
            button.setToolTip(self.analysis_output_path)
            self.check_enable_start()

    def check_enable_start(self):
        color_space_index = self.context.get_color_space_index()
        file_format_index = self.context.get_file_format_index()

        is_enabled = bool(
            self.input_path and color_space_index >= 0 and file_format_index >= 0 #Rimuovo il white obbligatorio: #and self.white_field_path
        )
        self.phase_manager.set_main_button_enabled(is_enabled)

    def start_worker(self, mode, skip_appstore_setting=False):
        self.worker_thread = WorkerThreadQT(self.core, mode, skip_appstore_setting)
        self.worker_thread.update_signal.connect(self.update_progress.emit)
        self.worker_thread.finished_signal.connect(self.handle_worker_finished)
        self.worker_thread.manual_correction_signal.connect(self.launch_wizard)
        self.worker_thread.start()
        self.worker_started.emit()
        self.handle_start_n_stop(True)

    def stop_worker(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
            self.handle_start_n_stop(False)
            self.phase_manager.pause_resume_button.setIcon(
                QIcon(self.phase_manager.pause_icon)
            )

    def launch_wizard(self, step):
        appStore.set(AnalysisSteps.KEY, step)

        wizard = ManualCorrectionWizard(self.core.equipment, self)
        result = wizard.exec_()
        if result == wizard.Accepted:
            try:
                calibration = wizard.get_results()
                self.core.calibration = calibration

            except Exception as e:
                self.core.skip_geometric_correction = True
                self.core.skip_vignetting_correction = True
                QMessageBox.critical(self, "Error", str(e))
        else:
            # User canceled the wizard
            self.core.skip_geometric_correction = True
            self.core.skip_vignetting_correction = True

        self.start_analysis(True)

    def start_analysis(self, skip_appstore_setting):
        if not skip_appstore_setting:
            color_space_index = self.context.get_color_space_index()
            file_format_index = self.context.get_file_format_index()

            output_format = get_format_name_by_id(file_format_index)
            output_color_space = get_color_space_name_by_id(color_space_index)

            self.core = Core(
                self.input_path,
                output_format,
                self.white_field_path,
                self.analysis_output_path,
                self.skip_params,
                output_color_space,
                self.fitting_degree,
                self.sharpen,
                self.light_balance,
            )
            self.core.parent_ref = self

        self.start_worker(Mode.ANALYSIS, skip_appstore_setting)

    def update_parameter_file(self, path):
        self.context.set_parameter_file(path)

    def update_analysis_output_file(self, path):
        self.context.set_analysis_output_path(path)

    def update_skip_params_checkbox(self, state, type):
        """Update the skip_params dictionary when a checkbox state changes."""
        self.skip_params[type] = True if state == 0 else False

    def update_sharpen_checkbox(self, state):
        """Update the skip_params dictionary when a checkbox state changes."""
        self.sharpen = False if state == 0 else True

    def update_light_balance_checkbox(self, state):
        """Update the skip_params dictionary when a checkbox state changes."""
        self.light_balance = False if state == 0 else True

    def handle_worker_finished(self, is_stopped=False):
        self.handle_start_n_stop(False)
        self.worker_finished.emit(is_stopped)
