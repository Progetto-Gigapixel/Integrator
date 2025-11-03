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

from ui.gui.models.models import CalibrationTca
from ui.gui.utils.gui_utils import create_input_field, get_positive_float_validator


class TCAPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("TCA Calibration")
        self.setSubTitle("Select the TCA model and enter the required coefficients.")

        layout = QVBoxLayout()

        # TCA Model Selection
        self.model_label = QLabel("TCA Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["linear", "poly3"])
        self.model_combo.currentTextChanged.connect(self.on_model_change)

        model_layout = QFormLayout()
        model_layout.addRow(self.model_label, self.model_combo)
        layout.addLayout(model_layout)

        # Coefficients Input
        self.coeff_layout = QFormLayout()
        self.coeff_inputs = {}

        self.update_coefficients("linear")
        layout.addLayout(self.coeff_layout)

        self.setLayout(layout)

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
        if model == "linear":
            self.add_coefficient("kr", "Coefficient kr:")
            self.add_coefficient("kb", "Coefficient kb:")
            self.add_coefficient("vr", "Coefficient vr:")
            self.add_coefficient("vb", "Coefficient vb:")
            self.add_coefficient("cr", "Coefficient cr:")
            self.add_coefficient("cb", "Coefficient cb:")
            self.add_coefficient("br", "Coefficient br:")
            self.add_coefficient("bb", "Coefficient bb:")
        elif model == "poly3":
            self.add_coefficient("kr", "Coefficient kr:")
            self.add_coefficient("kb", "Coefficient kb:")
            self.add_coefficient("vr", "Coefficient vr:")
            self.add_coefficient("vb", "Coefficient vb:")

    def add_coefficient(self, key, label_text):
        label = QLabel(label_text)
        input_field = create_input_field("e.g., 0.01", get_positive_float_validator())
        self.coeff_layout.addRow(label, input_field)
        self.coeff_inputs[key] = input_field

    def validatePage(self):
        try:
            model = self.model_combo.currentText()
            if model == "linear":
                kr = float(self.coeff_inputs["kr"].text())
                kb = float(self.coeff_inputs["kb"].text())
                vr = float(self.coeff_inputs["vr"].text())
                vb = float(self.coeff_inputs["vb"].text())
                cr = float(self.coeff_inputs["cr"].text())
                cb = float(self.coeff_inputs["cb"].text())
                br = float(self.coeff_inputs["br"].text())
                bb = float(self.coeff_inputs["bb"].text())
                tca = CalibrationTca(
                    model=model,
                    focal=0.0,
                    kr=kr,
                    kb=kb,
                    vr=vr,
                    vb=vb,
                    cr=cr,
                    cb=cb,
                    br=br,
                    bb=bb,
                )
            elif model == "poly3":
                kr = float(self.coeff_inputs["kr"].text())
                kb = float(self.coeff_inputs["kb"].text())
                vr = float(self.coeff_inputs["vr"].text())
                vb = float(self.coeff_inputs["vb"].text())
                tca = CalibrationTca(model=model, focal=0.0, kr=kr, kb=kb, vr=vr, vb=vb)
            else:
                QMessageBox.warning(self, "Input Error", "Unknown TCA model selected.")
                return False
            self.tca = tca
            return True
        except ValueError:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please enter valid numeric values for all coefficients.",
            )
            return False

    def get_tca(self):
        return self.tca
