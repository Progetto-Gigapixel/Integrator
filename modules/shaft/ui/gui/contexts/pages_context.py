from PyQt5.QtCore import QObject


class PagesContext(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color_space_index = 0
        self._file_format_index = 0
        self._analysis_output_path = None
        self._development_output_path = None
        self._parameter_file = None

    def get_color_space_index(self):
        return self._color_space_index

    def set_color_space_index(self, value):
        self._color_space_index = value

    def get_file_format_index(self):
        return self._file_format_index

    def set_file_format_index(self, value):
        self._file_format_index = value

    def get_analysis_output_path(self):
        return self._analysis_output_path

    def set_analysis_output_path(self, value):
        self._analysis_output_path = value

    def get_development_output_path(self):
        return self._development_output_path

    def set_development_output_path(self, value):
        self._development_output_path = value

    def get_parameter_file(self):
        return self._parameter_file

    def set_parameter_file(self, value):
        self._parameter_file = value
