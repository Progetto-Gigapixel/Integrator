from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWizardPage,
)

from log.logger import logger
from ui.gui.models.models import CalibrationDistortion
from ui.gui.utils.gui_utils import create_input_field, get_positive_float_validator


class DistortionPage(QWizardPage):
    def __init__(self, equipment, parent=None):
        super().__init__(parent)
        self.setTitle("Distortion Calibration")
        self.setSubTitle(
            "Select the distortion model and enter the required coefficients."
        )
        self.focal_length = equipment.focal_length
        layout = QVBoxLayout()
        model_layout = QFormLayout()

        # Focal Length
        self.focal_input = create_input_field(
            "e.g., 35.0", get_positive_float_validator()
        )
        # Set the focal length to the equipment's focal length (float)
        self.focal_input.setText(str(self.focal_length))
        self.focal_input.setReadOnly(True)

        model_layout.addRow("Focal Length:", self.focal_input)

        # Distortion Model Selection
        self.model_label = QLabel("Distortion Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["poly3", "poly5", "ptlens"])
        self.model_combo.currentTextChanged.connect(self.on_model_change)

        model_layout.addRow(self.model_label, self.model_combo)
        layout.addLayout(model_layout)

        # Coefficients Input
        self.coeff_layout = QFormLayout()
        self.coeff_inputs = {}

        self.update_coefficients("poly3")
        layout.addLayout(self.coeff_layout)

        self.setLayout(layout)
        logger.debug("DistortionPage initialized")

    def on_model_change(self, model):
        # Clear existing inputs
        while self.coeff_layout.count():
            item = self.coeff_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.coeff_inputs = {}
        self.update_coefficients(model)

    def update_coefficients(self, model):
        if model == "poly3":
            self.add_coefficient("k1", "Coefficient k1:")
        elif model == "poly5":
            self.add_coefficient("k1", "Coefficient k1:")
            self.add_coefficient("k2", "Coefficient k2:")
        elif model == "ptlens":
            self.add_coefficient("a", "Coefficient a:")
            self.add_coefficient("b", "Coefficient b:")
            self.add_coefficient("c", "Coefficient c:")

    def add_coefficient(self, key, label_text):
        label = QLabel(label_text)
        input_field = create_input_field("e.g., 0.01", get_positive_float_validator())
        self.coeff_layout.addRow(label, input_field)
        self.coeff_inputs[key] = input_field

    def validatePage(self):
        logger.debug("Validating DistortionPage")
        try:
            model = self.model_combo.currentText()
            focal_length = float(self.focal_input.text())

            if model == "poly3":
                k1 = float(self.coeff_inputs["k1"].text())
                distortion = CalibrationDistortion(
                    model=model, focal=focal_length, k1=k1
                )
            elif model == "poly5":
                k1 = float(self.coeff_inputs["k1"].text())
                k2 = float(self.coeff_inputs["k2"].text())
                distortion = CalibrationDistortion(
                    model=model, focal=focal_length, k1=k1, k2=k2
                )
            elif model == "ptlens":
                a = float(self.coeff_inputs["a"].text())
                b = float(self.coeff_inputs["b"].text())
                c = float(self.coeff_inputs["c"].text())
                distortion = CalibrationDistortion(
                    model=model, focal=focal_length, a=a, b=b, c=c
                )
            else:
                QMessageBox.warning(
                    self, "Input Error", "Unknown distortion model selected."
                )
                return False
            logger.info(f"Distortion calibration: {distortion}")
            self.distortion = distortion
            return True
        except ValueError:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please enter valid numeric values for all coefficients.",
            )
            return False

    def get_distortion(self):
        return self.distortion
