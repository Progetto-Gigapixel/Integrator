from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWizardPage

from ui.gui.models.models import (
    CalibrationDistortion,
    CalibrationVignetting,
    Camera,
    Lens,
)


class SummaryPage(QWizardPage):
    def __init__(self, parent=None):
        super(SummaryPage, self).__init__(parent)

        self.setTitle("Summary")
        self.setSubTitle("The following changes will be made to the dataset:")

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.summary_label)
        self.setLayout(layout)

    def initializePage(self):
        wizard = self.wizard()
        self.camera = wizard.get_camera()
        self.lens = wizard.get_lens()
        self.distorsion = wizard.get_distorsion()
        self.vignetting = wizard.get_vignetting()

        summary_text = "<b>Camera Information:</b><br>"
        summary_text += f"Make: {self.camera.maker}<br>"
        summary_text += f"Model: {self.camera.model}<br>"
        summary_text += f"Mount: {self.camera.mount}<br>"
        summary_text += f"Cropfactor: {self.camera.cropfactor}<br>"

        summary_text += "<br><b>Lens Information:</b><br>"
        summary_text += f"Make: {self.lens.maker}<br>"
        summary_text += f"Model: {self.lens.model}<br>"
        summary_text += f"Mount: {self.lens.mount}<br>"
        summary_text += f"Cropfactor: {self.lens.cropfactor}<br>"

        summary_text += "<br><b>Distorsion Information:</b><br>"
        summary_text += f"Model: {self.distorsion.model}<br>"
        summary_text += f"Focal: {self.distorsion.focal}<br>"
        if self.distorsion.model == "poly3":
            summary_text += f"k1: {self.distorsion.k1}<br>"
        elif self.distorsion.model == "poly5":
            summary_text += f"k1: {self.distorsion.k1}<br>"
            summary_text += f"k2: {self.distorsion.k2}<br>"
        elif self.distorsion.model == "ptlens":
            summary_text += f"a: {self.distorsion.a}<br>"
            summary_text += f"b: {self.distorsion.b}<br>"
            summary_text += f"c: {self.distorsion.c}<br>"

        summary_text += "<br><b>Vignetting Information:</b><br>"
        summary_text += f"Model: {self.vignetting.model}<br>"
        summary_text += f"Focal: {self.vignetting.focal}<br>"
        summary_text += f"Aperture: {self.vignetting.aperture}<br>"
        summary_text += f"Distance: {self.vignetting.distance}<br>"
        summary_text += f"k1: {self.vignetting.k1}<br>"
        summary_text += f"k2: {self.vignetting.k2}<br>"
        summary_text += f"k3: {self.vignetting.k3}<br>"

        self.summary_label.setText(summary_text)
