import webview  # Import pywebviewf
import os

class Api:
    def __init__(self):
        self._window = None

    def set_window(self, window):
        self._window = window

    def quit(self):
        self._window.destroy()
    def open_file_dialog(self):
        #file_types = ('Image Files (*.bmp;*.jpg;*.gif)', 'All files (*.*)')
        file_types = ( 'All files (*.*)')

        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=True
        )
        return result
    def save_file_dialog(self):
        # file_types = 'Json (*.bmp;*.jpg;*.gif)'
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG
        , save_filename='project.json'
        )
        return result
    def open_folder_dialog(self):
        # file_types = 'Json (*.bmp;*.jpg;*.gif)'
        # result = self._window.create_file_dialog(
        #     webview.FOLDER_DIALOG, directory=os.getcwd()
        # )
        result = self._window.create_file_dialog(
            webview.FOLDER_DIALOG
        )
        return result
    def destroy_window(self):
        self._window.destroy()
        #close app
        os._exit(0)

