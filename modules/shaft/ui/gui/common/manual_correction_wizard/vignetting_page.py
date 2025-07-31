from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWizardPage,
)

from log.logger import logger
from ui.gui.models.models import CalibrationVignetting
from ui.gui.utils.gui_utils import create_input_field, get_positive_float_validator


class VignettingPage(QWizardPage):
    def __init__(self, equipment, parent=None):
        super().__init__(parent)
        self.setTitle("Vignetting Calibration")
        self.setSubTitle("Please enter the vignetting calibration coefficients.")

        layout = QVBoxLayout()

        # Form for vignetting parameters
        form_layout = QFormLayout()
        self.k1_input = create_input_field("e.g., 0.01", get_positive_float_validator())
        self.k2_input = create_input_field("e.g., 0.01", get_positive_float_validator())
        self.k3_input = create_input_field("e.g., 0.01", get_positive_float_validator())
        self.focal_input = create_input_field(
            "e.g., 35.0", get_positive_float_validator()
        )
        self.focal_input.setText(str(equipment.focal_length))
        self.focal_input.setReadOnly(True)
        self.aperture_input = create_input_field(
            "e.g., 2.8", get_positive_float_validator()
        )
        self.aperture_input.setText(str(equipment.aperture))
        self.aperture_input.setReadOnly(True)
        self.distance_input = create_input_field(
            "e.g., 1.0", get_positive_float_validator()
        )

        form_layout.addRow("Focal Length:", self.focal_input)
        form_layout.addRow("Aperture:", self.aperture_input)
        form_layout.addRow("Distance:", self.distance_input)
        form_layout.addRow("Coefficient k1:", self.k1_input)
        form_layout.addRow("Coefficient k2:", self.k2_input)
        form_layout.addRow("Coefficient k3:", self.k3_input)

        layout.addLayout(form_layout)
        self.setLayout(layout)
        logger.debug("VignettingPage initialized")

    def validatePage(self):
        logger.debug("Validating VignettingPage")
        try:
            focal = float(self.focal_input.text())
            aperture = float(self.aperture_input.text())
            distance = float(self.distance_input.text())
            k1 = float(self.k1_input.text())
            k2 = float(self.k2_input.text())
            k3 = float(self.k3_input.text())

            self.vignetting = CalibrationVignetting(
                model="pa",
                focal=focal,
                aperture=aperture,
                distance=distance,
                k1=k1,
                k2=k2,
                k3=k3,
            )

            return True
        except ValueError:
            QMessageBox.warning(
                self, "Input Error", "Please enter valid numeric values for all fields."
            )
            return False

    def get_vignetting(self):
        return self.vignetting
