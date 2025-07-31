import os

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QSpacerItem

from ui.gui.styles.main_window_style import *


class PhaseManager(QHBoxLayout):
    start_phase = pyqtSignal(bool)
    stop_worker = pyqtSignal()

    def __init__(self, start_phase_label, parent=None):
        super().__init__()

        self.main_window = parent
        self.pause_icon = os.path.join(
            os.path.dirname(__file__), "../assets/images/pause.png"
        )
        self.start_icon = os.path.join(
            os.path.dirname(__file__), "../assets/images/start.png"
        )
        stop_icon = os.path.join(os.path.dirname(__file__), "../assets/images/stop.png")

        # Start button
        self.main_button = self.create_start_phase_button(
            start_phase_label,
            analyze_button_style,
            lambda: self.start_phase.emit(False),
        )

        # Pause/Resume button
        self.pause_resume_button = self.create_flow_button(
            self.pause_icon, stop_resume_button_style, self.toggle_pause_resume
        )

        # Stop button
        self.stop_button = self.create_flow_button(
            stop_icon, stop_resume_button_style, self.handle_stop_worker
        )

        # Create a layout to hold the analyze, pause/resume, and stop buttons
        self.addWidget(self.main_button)
        self.addWidget(self.pause_resume_button)
        self.addWidget(self.stop_button)
        self.addSpacerItem(
            QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

    def create_start_phase_button(self, text, style, event):
        button = QPushButton(text)
        button.setEnabled(False)
        button.setFixedSize(350, 40)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(style)
        button.clicked.connect(event)

        return button

    def create_flow_button(self, icon, style, event):
        button = QPushButton()
        button.setFixedSize(40, 40)
        button.setIcon(QIcon(icon))
        button.setIconSize(QSize(40, 49))
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(style)
        button.clicked.connect(event)
        button.setVisible(False)

        return button

    def set_main_button_enabled(self, is_enabled):
        self.main_button.setEnabled(is_enabled)

    def handle_start_n_stop(self, is_start):
        self.pause_resume_button.setVisible(is_start)
        self.stop_button.setVisible(is_start)
        self.main_button.setEnabled(not is_start)

    def handle_stop_worker(self):
        self.pause_resume_button.setIcon(QIcon(self.start_icon))
        self.stop_worker.emit()

    def toggle_pause_resume(self):
        if self.main_window.worker_thread._paused:
            self.main_window.worker_thread.resume()
            self.pause_resume_button.setIcon(QIcon(self.pause_icon))
        else:
            self.main_window.worker_thread.pause()
            self.pause_resume_button.setIcon(QIcon(self.start_icon))
