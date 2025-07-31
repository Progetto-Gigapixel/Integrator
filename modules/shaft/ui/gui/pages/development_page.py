# from PyQt5.QtCore import pyqtSignal
# from PyQt5.QtGui import QIcon
# from PyQt5.QtWidgets import (
#     QFileDialog,
#     QFrame,
#     QGridLayout,
#     QHBoxLayout,
#     QSizePolicy,
#     QSpacerItem,
#     QVBoxLayout,
#     QWidget,
# )

from config.config import DEVELOPMENT_PARAMS, Mode, SKIP_PARAMS
from core.core import Core
from ui.gui.common.phase_manager import PhaseManager
from ui.gui.configs.main_window_config import FILE_FORMATS
from ui.gui.contexts.pages_context import PagesContext
from ui.gui.styles.main_window_style import *
from ui.gui.utils.gui_utils import (
    create_qt_button,
    create_qt_checkbox_widget,
    create_qt_dropdown,
    create_qt_input,
    create_qt_label,
    get_color_space_name_by_id,
    get_format_name_by_id,
)
from ui.process_threads.worker_thread_qt import WorkerThreadQT


class DevelopmentPage(QWidget):
    update_progress = pyqtSignal(int)
    worker_started = pyqtSignal()
    worker_finished = pyqtSignal(bool)

    def __init__(self, parent, context: PagesContext):
        super().__init__()

        self.context = context
        self.phase_manager = PhaseManager("Development", self)
        self.phase_manager.start_phase.connect(self.start_development)
        self.phase_manager.stop_worker.connect(self.stop_worker)

        self.worker_thread: WorkerThreadQT = None
        self.core: Core = None
        self.input_path = None
        self.development_output_path = None
        self.development_params = {
            DEVELOPMENT_PARAMS.PROCESS_SUBFOLDER: False,
            DEVELOPMENT_PARAMS.DO_NOT_OVERWRITE_FILES: False,
            DEVELOPMENT_PARAMS.EXTENSION_2_PROCESS: None,
            DEVELOPMENT_PARAMS.PARAMETER_PATH: None,
        }

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

        self.on_page_changed_operations: list = []

        self.parent = parent

        page_layout = QHBoxLayout()
        self.setLayout(page_layout)

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

        # Second Row: Folder 2 develop
        folder2develop_label, folder2develop_button = (
            self.create_folder2develop_selection()
        )
        grid_layout.addWidget(folder2develop_label, 1, 0)
        grid_layout.addWidget(folder2develop_button, 1, 1)

        # Third Row: Subfolder & Overwrite checkboxes
        subfolder_label, subfolder_widget = self.create_subfolder_checkbox()
        grid_layout.addWidget(subfolder_label, 2, 0)
        grid_layout.addWidget(subfolder_widget, 2, 1)

        do_not_overwrite_widget = self.create_overwrite_checkbox()
        grid_layout.addWidget(do_not_overwrite_widget, 2, 2)

        # Fourth Row: Extension 2 process
        extension2process_label, extension2process_input = (
            self.create_extension2process_input()
        )
        grid_layout.addWidget(extension2process_label, 3, 0)
        grid_layout.addWidget(extension2process_input, 3, 1)

        # Fifth Row: Parameter file with button
        parameter_file_label, parameter_file_button = (
            self.create_parameter_file_label_button()
        )
        grid_layout.addWidget(parameter_file_label, 4, 0)
        grid_layout.addWidget(parameter_file_button, 4, 1)

        # Sixth Row: development Output path selection
        output_path_dropdown, output_path_button = self.create_output_path_selection()
        grid_layout.addWidget(output_path_dropdown, 5, 0)
        grid_layout.addWidget(output_path_button, 5, 1)

        # Seventh Row: Output file format with dropdown
        output_file_format_label, output_file_format_dropdown = (
            self.create_output_file_format_dropdown()
        )
        grid_layout.addWidget(output_file_format_label, 6, 0)
        grid_layout.addWidget(output_file_format_dropdown, 6, 1)

        # Add the grid layout to the content layout
        content_layout.addLayout(grid_layout)

        # Button at the bottom

        content_layout.addLayout(self.phase_manager)

        page_layout.addWidget(content_frame)

        # Add a spacer to take up the remaining space on the right
        page_layout.addSpacerItem(
            QSpacerItem(999 // 2, 478, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

    def on_page_changed(self):
        for operation in self.on_page_changed_operations:
            operation()

        self.check_enable_start()

    def create_title(self):
        text = "Development"
        title = create_qt_label(text, common_title_label_style)
        return title

    def create_folder2develop_selection(self):
        label_text = "Folder 2 develop"
        button_text = "Browse"

        label = create_qt_label(label_text, common_label_style)
        button = create_qt_button(button_text, lambda: self.select_folder_2_develop(button))

        return label, button

    def create_subfolder_checkbox(self):
        label_text = "Subfolder"
        checkbox_text = "Process subfolders"
        checkbox_type = DEVELOPMENT_PARAMS.PROCESS_SUBFOLDER

        label = create_qt_label(label_text, common_label_style)
        widget = create_qt_checkbox_widget(
            checkbox_type,
            lambda state: self.update_development_params_checkbox(state, checkbox_type),
            checkbox_text,
        )

        return label, widget

    def create_overwrite_checkbox(self):
        checkbox_text = "Do not overwrite existing files"
        checkbox_type = DEVELOPMENT_PARAMS.DO_NOT_OVERWRITE_FILES

        widget = create_qt_checkbox_widget(
            checkbox_type,
            lambda state: self.update_development_params_checkbox(state, checkbox_type),
            checkbox_text,
        )

        return widget

    def create_extension2process_input(self):
        label_text = "Extension 2 process"
        input_text = "Type"

        label = create_qt_label(label_text, common_label_style)
        input = create_qt_input(
            input_text,
            lambda: self.update_development_params(
                DEVELOPMENT_PARAMS.EXTENSION_2_PROCESS, input.text()
            ),
        )
        input.setText("NEF")
        self.check_enable_start()
        return label, input

    def create_parameter_file_label_button(self):
        label_text = "Parameter file"
        button_text = "Browse"

        label = create_qt_label(label_text, common_label_style)
        button = create_qt_button(button_text, lambda: self.select_file(button, "json"))

        self.on_page_changed_operations.append(
            lambda: self.update_parameter_file(button)
        )
        return label, button

    def create_output_path_selection(self):
        label_text = "Development Output path"
        button_text = "Browse"

        label = create_qt_label(label_text, common_label_style)
        button = create_qt_button(button_text, lambda: self.select_output_path(button))

        # self.on_page_changed_operations.append(lambda: self.update_output_path(button))
        return label, button

    def update_parameter_file(self, button):
        parameter_path = self.context.get_parameter_file()

        if parameter_path:
            button.setText(parameter_path)
            self.development_params[DEVELOPMENT_PARAMS.PARAMETER_PATH] = parameter_path


    def select_output_path(self, button):
        self.development_output_path = QFileDialog.getExistingDirectory(
            self, "Select the folder to store your developments", "", QFileDialog.ShowDirsOnly
        )
        if self.development_output_path:
            button.setText(self.development_output_path)
            button.setToolTip(self.development_output_path)
            self.check_enable_start()

        self.development_params[DEVELOPMENT_PARAMS.OUTPUT_PATH] = self.development_output_path

    # def update_output_path(self, button):
    #     #output_path = self.context.get_output_path()
    #
    #     if development_output_path:
    #         button.setText(development_output_path)
    #         button.setToolTip(development_output_path)
    #         self.development_params[DEVELOPMENT_PARAMS.OUTPUT_PATH] = self.development_output_path

    def create_output_file_format_dropdown(self):
        text = "Output file format"
        items = [file_format["name"] for file_format in FILE_FORMATS]

        label, dropdown = create_qt_dropdown(text, items)
        dropdown.currentIndexChanged.connect(self.context.set_file_format_index)

        return label, dropdown

    def select_folder_2_develop(self, button):
        self.input_path = QFileDialog.getExistingDirectory(
            self, "Select Folder 2 develop", "", QFileDialog.ShowDirsOnly
        )
        if self.input_path:
            button.setText(self.input_path)
            button.setToolTip(self.input_path)
            self.check_enable_start()

    def check_enable_start(self):
        parameter_file = self.development_params #self.context.get_parameter_file()
        extension_2_process = self.development_params[
            DEVELOPMENT_PARAMS.EXTENSION_2_PROCESS
        ]
        is_enabled = bool(self.input_path and parameter_file and extension_2_process)

        self.phase_manager.set_main_button_enabled(is_enabled)

    def update_development_params(self, key, value):
        self.development_params[key] = value
        self.check_enable_start()

    def start_development(self, skip_appstore_setting):
        if not skip_appstore_setting:
            if self.development_output_path is None:
                self.development_output_path = self.context.get_development_output_path()
            color_space_index = self.context.get_color_space_index()
            file_format_index = self.context.get_file_format_index()

            output_format = get_format_name_by_id(file_format_index)
            output_color_space = get_color_space_name_by_id(color_space_index)

            self.core = Core(
                self.input_path,
                output_format,
                None, # TODO!!!! WHITE PATH. Scrivere nel file dei risultati dove si trova
                self.development_output_path,
                self.skip_params,
                output_color_space,
                None,
                None,
                None,
                self.development_params,
            )

        self.start_worker(Mode.DEVELOPMENT, skip_appstore_setting)

    def handle_start_n_stop(self, is_start):
        self.phase_manager.handle_start_n_stop(is_start)

    def handle_worker_finished(self, is_stopped=False):
        self.handle_start_n_stop(False)
        self.worker_finished.emit(is_stopped)

    def stop_worker(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
            self.handle_start_n_stop(False)
            self.phase_manager.pause_resume_button.setIcon(
                QIcon(self.phase_manager.pause_icon)
            )

    def start_worker(self, mode, skip_appstore_setting):
        self.worker_thread = WorkerThreadQT(self.core, mode, skip_appstore_setting)
        self.worker_thread.update_signal.connect(self.update_progress.emit)
        self.worker_thread.finished_signal.connect(self.handle_worker_finished)
        self.worker_thread.start()
        self.worker_started.emit()
        self.handle_start_n_stop(True)

    def select_file(self, button, file_type=None):
        options = QFileDialog.Options()

        # Default filter if no file_type is provided
        file_filter = "All Files (*);;Text Files (*.txt)"

        # Override the filter if file_type is specified
        if file_type:
            file_filter = f"{file_type} Files (*.{file_type})"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            file_filter,
            options=options,
        )

        if file_path:
            button.setText(file_path)
            button.setToolTip(file_path)
            self.development_params[DEVELOPMENT_PARAMS.PARAMETER_PATH] = file_path

    def update_development_params_checkbox(self, state, type):
        """Update the development_params dictionary when a checkbox state changes."""
        self.development_params[type] = False if state == 0 else True
