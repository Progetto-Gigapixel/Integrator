from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWizardPage


class ConfirmPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Missing Lensfun Data")

        label = QLabel("Do you want to proceed with manual correction?")
        label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)

    def validatePage(self):
        # Always allow to proceed; actual confirmation handled by wizard's cancel button
        return True
