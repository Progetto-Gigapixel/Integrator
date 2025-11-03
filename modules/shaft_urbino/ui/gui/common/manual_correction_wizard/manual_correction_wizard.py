from PyQt5.QtWidgets import QWizard

from ui.gui.models.models import Calibration

from .camera_lens_page import CameraLensPage
from .confirm_page import ConfirmPage
from .distortion_page import DistortionPage
from .summary_page import SummaryPage
from .tca_page import TCAPage
from .vignetting_page import VignettingPage


class ManualCorrectionWizard(QWizard):
    def __init__(self, equipment, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Correction")
        self.setWizardStyle(QWizard.AeroStyle)

        # Initialize data attributes
        self.calibration = Calibration()

        # Add pages
        self.confirm_page = ConfirmPage()
        self.addPage(self.confirm_page)

        self.camera_lens_page = CameraLensPage(equipment)
        self.addPage(self.camera_lens_page)

        self.distortion_page = DistortionPage(equipment)
        self.addPage(self.distortion_page)

        self.vignetting_page = VignettingPage(equipment)
        self.addPage(self.vignetting_page)

        self.summary_page = SummaryPage()
        self.addPage(self.summary_page)

    def accept(self):
        try:
            # Gather data from CameraLensPage
            camera, lens = self.camera_lens_page.get_camera_lens_info()
            self.calibration.camera = camera
            self.calibration.lens = lens

            # Gather distortion calibration
            distortion = self.distortion_page.get_distortion()
            self.calibration.distortion = distortion

            # Gather Vignetting calibration
            vignetting = self.vignetting_page.get_vignetting()
            self.calibration.vignetting = vignetting

            super().accept()
        except Exception as e:
            raise Exception(f"Error during manual geometric correction: {str(e)}")

    def get_results(self):
        return self.calibration

    def get_camera(self):
        return self.camera_lens_page.get_camera()

    def get_lens(self):
        return self.camera_lens_page.get_lens()

    def get_distorsion(self):
        return self.distortion_page.get_distortion()

    def get_vignetting(self):
        return self.vignetting_page.get_vignetting()
