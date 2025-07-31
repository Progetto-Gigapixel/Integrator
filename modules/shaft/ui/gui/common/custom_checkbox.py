from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
from PyQt5.QtWidgets import QCheckBox

from config.config import SKIP_PARAMS


class CustomCheckBox(QCheckBox):
    state_changed = pyqtSignal(int, str)

    def __init__(self, type: SKIP_PARAMS = None, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.type = type
        self.stateChanged.connect(self.emit_state_changed)

    def emit_state_changed(self, state):
        """Emit both the checkbox state and the type."""
        self.state_changed.emit(state, self.type)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = QRect(0, 0, self.width(), self.height())

        # Draw the border
        pen = QPen(QColor("#800A00"), 2)
        painter.setPen(pen)
        painter.setBrush(
            QBrush(QColor("#800A00") if self.isChecked() else Qt.transparent)
        )

        corner_radius = 4
        painter.drawRoundedRect(rect, corner_radius, corner_radius)

        # Draw the checkmark if checked
        if self.isChecked():
            pen = QPen(QColor("#FFFFFF"), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(4, 10, 8, 14)
            painter.drawLine(8, 14, 16, 6)

        painter.end()
