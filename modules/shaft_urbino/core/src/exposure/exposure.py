import numpy as np

from config.config import Mode
from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.common.exceptions.exposure_correction_exceptions import (
    ExposureCorrectionException,
)
from core.utils.core_utils import get_target_reference, xyz_2_hsv, hsv_2_xyz
from locales.localization import _
from log.logger import logger


class ExposureCorrector:
    def __init__(
        self,
        correction_data: ColorCorrectionData,
        reference_values,
        measured_patches,
        image,
        mode,
    ):
        """
        Initializes the exposure corrector with the specified parameters.

        :param correction_data: The correction data object for reporting corrections
        :param reference_values: The reference_values object for the target reference
        :param image: The image to be corrected (float32)
        :param patches: The patches used for the correction
        """
        self.correction_data = correction_data
        self.image = image
        self.measured_patches = measured_patches
        self.reference_values = reference_values
        self.mode = mode

    def run(self):
        """
        Applies the exposure correction.

        :return: The corrected image as a numpy array
        """
        try:
            logger.info(_("Applying exposure correction..."))
            if self.mode == Mode.ANALYSIS:
                result = self.norm_exp_correction()
            else:
                result = self.norm_exp_correction_development()

            logger.info(_("Exposure correction applied."))
            return result
        except Exception as e:
            logger.error(_("Failed to apply exposure correction: %s"), e)
            raise ExposureCorrectionException(e)

    def norm_exp_correction(self):
        """
        Applies the exposure correction using the stretched method.

        :return: The corrected image as a numpy array
        """
        measured_white = self.measured_patches[0][18]["rgb_values"]
        measured_black = self.measured_patches[0][23]["rgb_values"]
        measured_red = self.measured_patches[0][14]["rgb_values"]
        expected_white = self._get_reference_white()
        expected_black = self._get_reference_black()
        expected_red = self._get_reference_red()

        v_measured_white = xyz_2_hsv(np.array(measured_white))[2]
        v_measured_black = xyz_2_hsv(np.array(measured_black))[2]
        s_measured_red = xyz_2_hsv(np.array(measured_red))[1]
        v_expected_white = xyz_2_hsv(np.array(expected_white))[2]
        v_expected_black = xyz_2_hsv(np.array(expected_black))[2]
        s_expected_red = xyz_2_hsv(np.array(expected_red))[1]

        expected_minimum = v_expected_black  # a
        expected_range = v_expected_white - v_expected_black
        measured_range = v_measured_white - v_measured_black

        hsv_image = xyz_2_hsv(self.image)
        brightness = hsv_image[:, :, 2]
        saturation = hsv_image[:, :, 1]

        # Normalization between given values: xnormalized = a + (((x - xminimum)* (b - a)) / range x
        brightness_corrected_channel = expected_minimum + (
            ((brightness - v_measured_black) * expected_range) / measured_range
        )

        saturation_correction = s_measured_red / s_expected_red
        saturation_corrected_channel = saturation / saturation_correction

        corrected_hsv = np.dstack(
            (hsv_image[:, :, 0], saturation_corrected_channel, brightness_corrected_channel)
        )
        exposure_corrected = hsv_2_xyz(corrected_hsv)

        self.correction_data.set_norm_exp_correction(
            expected_minimum, expected_range, measured_range, saturation_correction, v_measured_black
        )

        return exposure_corrected

    def norm_exp_correction_development(self):
        """
        Applies the exposure correction using the pre-set correction data in development mode.

        :return: The corrected image as a numpy array
        :raises ValueError: If any of the correction data is not set
        """

        expected_minimum, expected_range, measured_range, saturation_correction, v_measured_black = (
            self.correction_data.get_exposure_correction()
        )

        hsv_image = xyz_2_hsv(self.image)
        saturation = hsv_image[:, :, 1]
        brightness = hsv_image[:, :, 2]



        # corrected_brightness = expected_minimum + (
        #     (brightness * expected_range) / measured_range
        # )
        # Normalization between given values: xnormalized = a + (((x - xminimum)* (b - a)) / range x
        brightness_corrected_channel = expected_minimum + (
            ((brightness - v_measured_black) * expected_range) / measured_range
        )
        brightness_corrected_channel = np.clip(brightness_corrected_channel, 0, 1)

        saturation_corrected_channel = saturation / saturation_correction

        corrected_hsv = np.dstack(
            (hsv_image[:, :, 0], saturation_corrected_channel, brightness_corrected_channel)
        )
        exposure_corrected = hsv_2_xyz(corrected_hsv)

        return exposure_corrected

    def _get_reference_black(self):
        """
        Returns the reference black point.

        :return: The reference black point
        """
        return get_target_reference(self.reference_values, 24)

    def _get_reference_white(self):
        """
        Returns the reference white point.

        :return: The reference white
        """
        return get_target_reference(self.reference_values, 19)

    def _get_reference_red(self):
        """
        Returns the reference white point.

        :return: The reference white
        """
        return get_target_reference(self.reference_values, 15)
