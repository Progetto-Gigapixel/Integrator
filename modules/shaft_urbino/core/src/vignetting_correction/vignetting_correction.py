import numpy as np

from core.common.exceptions.vignetting_correction_exceptions import (
    VignettingCorrectionException,
)
from core.common.lensfun_corrector.lensfun_corrector import LensfunCorrector
from locales.localization import _
from log.logger import logger


class VignettingCorrector(LensfunCorrector):
    def run(self):
        """
        Runs the vignetting correction.

        :return: The corrected image as a numpy array
        """
        try:
            return self._run()
        except Exception as e:
            logger.error(_("Failed to apply vignetting correction: %s"), e)
            raise VignettingCorrectionException(e)

    def apply_correction(self):
        """
        Applies the vignetting correction using the lensfunpy data.
        Overrides the parent method.

        :return: The corrected image as a numpy array
        """
        logger.info(_("Applying vignetting correction..."))

        # Apply vignetting correction
        undistorted_image = np.copy(self.image_array).astype(np.float32)

        res = self.mod.apply_color_modification(undistorted_image)

        if not res:
            logger.error(
                _("Failed to apply vignetting correction, calibration data missing?")
            )
        else:
            logger.info(_("Vignetting correction applied successfully."))

        return undistorted_image
