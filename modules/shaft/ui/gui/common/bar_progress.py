from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ui.gui.styles.main_window_style import (
    generate_progress_bar_style,
    percentage_label_style,
    waiting_label_style,
)


class BarProgress(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(328, 38)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top frame (text and percentage)
        frame7 = QFrame()
        frame7.setFixedHeight(18)
        frame7_layout = QHBoxLayout()
        frame7_layout.setContentsMargins(0, 0, 0, 0)
        frame7_layout.setSpacing(10)  # Smaller spacing for flexibility

        # Waiting text
        self.waiting_label = QLabel("Waiting to launch process")
        self.waiting_label.setStyleSheet(waiting_label_style)
        frame7_layout.addWidget(self.waiting_label)

        # Spacer to manage space between the text and percentage label
        frame7_layout.addStretch()

        # Percentage label
        self.percentage_label = QLabel("0%")
        self.percentage_label.setStyleSheet(percentage_label_style)
        frame7_layout.addWidget(self.percentage_label)

        frame7.setLayout(frame7_layout)
        layout.addWidget(frame7)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(generate_progress_bar_style())
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    # Update progress bar value and percentage label
    def set_progress(self, value, label=None):
        self.progress_bar.setValue(value)
        self.percentage_label.setText(f"{value}%")
        if label:
            self.waiting_label.setText(label)

    def set_style(self, isInProgress=False):
        self.progress_bar.setStyleSheet(generate_progress_bar_style(isInProgress))
