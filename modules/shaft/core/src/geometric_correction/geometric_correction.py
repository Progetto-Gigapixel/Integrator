import cv2
import numpy as np

from core.common.exceptions.geometry_correction_exceptions import (
    GeometricCorrectionException,
)
from core.common.lensfun_corrector.lensfun_corrector import LensfunCorrector
from locales.localization import _
from log.logger import logger


class GeometricCorrector(LensfunCorrector):
    def run(self):
        """
        Runs the geometric correction.

        :return: The corrected image as a numpy array
        """
        try:
            return self._run()
        except Exception as e:
            logger.error(_("Failed to apply geometric correction: %s"), e)
            raise GeometricCorrectionException(e)

    # Apply the correction using Lensfunpy data
    def apply_correction(self):
        """
        Applies the geometric correction using the lensfunpy data.
        Overrides the parent method.

        :return: The corrected image as a numpy array
        """
        logger.info(_("Applying geometric correction..."))

        # Get the maps for distortion correction
        undist_coords = self.mod.apply_geometry_distortion()
        undistorted_img = cv2.remap(
            self.image_array, undist_coords, None, cv2.INTER_LANCZOS4
        )

        logger.info(_("Geometric correction applied successfully."))
        return undistorted_img
