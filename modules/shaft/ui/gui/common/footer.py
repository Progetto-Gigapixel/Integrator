from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
)

from ui.gui.common.bar_progress import BarProgress
from ui.gui.styles.main_window_style import *
from ui.gui.utils.gui_utils import get_qt_shadow_effect


class Footer(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        # Footer frame configuration
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("footer")
        self.setFixedSize(999, 73)
        self.setStyleSheet(footer_style)
        self.setGraphicsEffect(get_qt_shadow_effect())

        # Main layout for the footer
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(20, 0, 20, 0)
        footer_layout.setSpacing(0)

        # BarProgress widget (custom progress bar with percentage and text)
        self.bar_progress = BarProgress()
        footer_layout.addWidget(self.bar_progress, alignment=Qt.AlignLeft)

        # Next Phase button
        self.next_phase_button = QPushButton("Next phase")
        self.next_phase_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.next_phase_button.setStyleSheet(next_phase_button_style)
        self.next_phase_button.setCursor(Qt.PointingHandCursor)
        self.next_phase_button.setVisible(False)
        self.next_phase_button.clicked.connect(self.handle_next_phase)
        footer_layout.addWidget(self.next_phase_button, alignment=Qt.AlignRight)

        # Set the layout for the footer
        self.setLayout(footer_layout)

    def set_progress(self, value, label=None):
        """Updates the BarProgress widget with progress and optional text."""
        self.bar_progress.set_progress(value, label)

        isInProgress = value != 100 and value != 0
        self.bar_progress.set_style(isInProgress)

    def handle_next_phase(self):
        """Handles the Next Phase button logic."""
        self.main_window.show_page(1)
        self.set_progress(0, "Waiting to launch process")
        self.show_next_phase_button(False)

    def show_next_phase_button(self, show):
        """Shows or hides the Next Phase button."""
        self.next_phase_button.setVisible(show)
