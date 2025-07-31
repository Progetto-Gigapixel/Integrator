import os

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton

from ui.gui.styles.main_window_style import (
    app_title_style,
    close_button_style,
    generate_phase_button_style,
    header_style,
)
from ui.gui.utils.gui_utils import get_qt_shadow_effect


class Header(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        # Get the path of the current script
        self.script_dir = os.path.dirname(os.path.realpath(__file__))

        # Frame header configuration
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("header")
        self.setStyleSheet(header_style)
        self.setGraphicsEffect(get_qt_shadow_effect())

        # Main layout for the header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 7, 20, 7)
        header_layout.setSpacing(20)

        # Create the left section layout (app icon and title)
        left_frame = self.create_left_section()
        header_layout.addWidget(left_frame)

        # Spacer to manage space between the left and right sections
        header_layout.addStretch()

        # Create the right section layout (close button)
        right_frame = self.create_right_section()
        header_layout.addWidget(right_frame)

        # Set the layout of the header
        self.setLayout(header_layout)

    def create_left_section(self):
        """Creates the left section of the header with the app icon and title."""

        left_frame = QFrame()
        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Internal frame for the app icon and title
        internal_frame = QFrame()
        internal_layout = QHBoxLayout()
        internal_layout.setContentsMargins(0, 0, 0, 0)

        # App icon
        app_icon = QLabel()
        icon_path = os.path.join(self.script_dir, "../assets/images/shaft.png")
        pixmap = QPixmap(icon_path).scaled(
            30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        app_icon.setPixmap(pixmap)
        internal_layout.addWidget(app_icon)

        # App title
        app_title = QLabel("SHAFT")
        app_title.setStyleSheet(app_title_style)
        internal_layout.addWidget(app_title)

        internal_frame.setLayout(internal_layout)
        left_layout.addWidget(internal_frame)

        # Phase buttons
        button_list = self.create_phase_buttons()
        left_layout.addWidget(button_list)

        left_frame.setLayout(left_layout)
        return left_frame

    def create_right_section(self):
        """Creates the right section of the header with the close button."""

        right_frame = QFrame()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Close button
        close_button = QToolButton()
        close_button_icon_path = os.path.join(
            self.script_dir, "../assets/images/close.png"
        )
        close_button.setIcon(QIcon(close_button_icon_path))
        close_button.setIconSize(QSize(30, 30))
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.setStyleSheet(close_button_style)
        close_button.clicked.connect(self.close_application)

        right_layout.addWidget(close_button)
        right_frame.setLayout(right_layout)

        return right_frame

    def create_phase_buttons(self):
        """Creates the Phase 1 and Phase 2 buttons."""

        button_list = QFrame()
        button_list_layout = QHBoxLayout()
        button_list_layout.setContentsMargins(0, 0, 0, 0)
        button_list_layout.setSpacing(10)

        # Phase 1 button
        self.phase1_button = QPushButton("Target Analysis Phase")
        self.phase1_button.setFixedSize(150, 24)
        self.phase1_button.setStyleSheet(generate_phase_button_style(True))
        self.phase1_button.clicked.connect(lambda: self.main_window.show_page(0))
        button_list_layout.addWidget(self.phase1_button)

        # Phase 2 button
        self.phase2_button = QPushButton("Development Phase")
        self.phase2_button.setFixedSize(150, 24)
        self.phase2_button.setStyleSheet(generate_phase_button_style(False))
        self.phase2_button.clicked.connect(lambda: self.main_window.show_page(1))
        button_list_layout.addWidget(self.phase2_button)

        button_list.setLayout(button_list_layout)
        return button_list

    def close_application(self):
        """Closes the main window."""

        self.main_window.close()

    def update_button_styles(self, index):
        """Updates the styles of the phase buttons based on the selected index."""

        self.phase1_button.setStyleSheet(
            generate_phase_button_style(is_active=(index == 0))
        )
        self.phase2_button.setStyleSheet(
            generate_phase_button_style(is_active=(index != 0))
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_position = (
                event.globalPos() - self.main_window.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() & Qt.LeftButton:
            self.main_window.move(event.globalPos() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._is_dragging = False
        event.accept()
