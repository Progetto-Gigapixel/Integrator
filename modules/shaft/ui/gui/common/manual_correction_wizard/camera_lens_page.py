# wizard_pages/camera_lens_page.py

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWizardPage,
)

from ui.gui.models.models import Camera, Lens
from ui.gui.utils.gui_utils import create_input_field, get_positive_float_validator


class CameraLensPage(QWizardPage):
    def __init__(self, equipment, parent=None):
        super().__init__(parent)
        self.setTitle("Camera and Lens Information")
        self.setSubTitle("Please enter the details of your camera and lens.")

        layout = QVBoxLayout()

        form_layout = QFormLayout()

        # Camera Information
        self.camera_maker_input = create_input_field("e.g., Canon", None)
        if hasattr(equipment, "camera_make"):
            self.camera_maker_input.setText(equipment.camera_make)
            self.camera_maker_input.setReadOnly(True)
            self.camera_model_input = create_input_field("e.g., EOS 5D Mark IV", None)
        if hasattr(equipment, "camera_model"):
            self.camera_model_input.setText(equipment.camera_model)
            self.camera_model_input.setReadOnly(True)

        form_layout.addRow("Camera Maker:", self.camera_maker_input)
        form_layout.addRow("Camera Model:", self.camera_model_input)

        # Separator
        separator = QLabel("<hr>")
        form_layout.addRow(separator)

        # Lens Information
        self.lens_maker_input = create_input_field("e.g., Canon", None)
        self.lens_maker_input.setText(equipment.lens_make)
        self.lens_maker_input.setReadOnly(True)
        self.lens_model_input = create_input_field("e.g., EF 24-70mm f/2.8L II", None)
        self.lens_model_input.setText(equipment.lens_model)
        self.lens_model_input.setReadOnly(True)

        form_layout.addRow("Lens Maker:", self.lens_maker_input)
        form_layout.addRow("Lens Model:", self.lens_model_input)

        # Separator
        separator = QLabel("<hr>")
        form_layout.addRow(separator)

        self.mount_input = create_input_field("e.g., EF", None)
        self.crop_factor_input = create_input_field(
            "e.g., 1.5", get_positive_float_validator()
        )

        form_layout.addRow("Mount:", self.mount_input)
        form_layout.addRow("Crop Factor:", self.crop_factor_input)

        layout.addLayout(form_layout)
        self.setLayout(layout)

    def validatePage(self):
        # Validate Camera Information
        camera_maker = self.camera_maker_input.text().strip()
        camera_model = self.camera_model_input.text().strip()

        if not camera_maker:
            QMessageBox.warning(self, "Input Error", "Camera Maker cannot be empty.")
            return False
        if not camera_model:
            QMessageBox.warning(self, "Input Error", "Camera Model cannot be empty.")
            return False

        # Validate Lens Information
        lens_maker = self.lens_maker_input.text().strip()
        lens_model = self.lens_model_input.text().strip()

        if not lens_maker:
            QMessageBox.warning(self, "Input Error", "Lens Maker cannot be empty.")
            return False
        if not lens_model:
            QMessageBox.warning(self, "Input Error", "Lens Model cannot be empty.")
            return False

        # Validate common information
        mount = self.mount_input.text().strip()
        crop_factor_text = self.crop_factor_input.text().strip()

        try:
            crop_factor = float(crop_factor_text)
            if crop_factor <= 0:
                QMessageBox.warning(
                    self, "Input Error", "Crop Factor must be a positive number."
                )
                return False
        except ValueError:
            QMessageBox.warning(
                self, "Input Error", "Crop Factor must be a valid number."
            )
            return False

        # Store the data
        self.camera = Camera(
            maker=camera_maker, model=camera_model, mount=mount, cropfactor=crop_factor
        )

        self.lens = Lens(
            maker=lens_maker, model=lens_model, mount=mount, cropfactor=crop_factor
        )

        return True

    def get_camera_lens_info(self):
        return self.camera, self.lens

    def get_camera(self):
        return self.camera

    def get_lens(self):
        return self.lens
