import matplotlib.pyplot as plt
import numpy as np

from config.config import Mode
from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.common.exceptions.flat_fielding_correction_exceptions import (
    FlatFieldingCorrectionException,
)
from core.utils.core_utils import get_target_reference
from locales.localization import _
from log.logger import logger


class FlatFieldingCorrector:
    def __init__(self, correction_data: ColorCorrectionData, campioni, image, mode):
        """
        Initializes the flat-fielding corrector with the specified parameters.

        :param correction_data: The correction data object for reporting corrections
        :param campioni: The campioni object for the target reference
        :param image: The image to be corrected (float32)
        """
        self.correction_data = correction_data
        self.image = image
        self.campioni = campioni
        self.mode = mode
        self.white_patch = self._get_reference_4th_gray()

    def run(self):
        """
        Applies the flat-field correction.

        :return: The corrected image as a numpy array
        """
        try:
            logger.info(_("Applying flat-field correction..."))

            if self.mode == Mode.ANALYSIS:
                gain_matrix = self.calculate_gain_matrix()
            else:
                gain_matrix = np.array(
                    self.correction_data.get_flat_fielding_correction()
                )

            result = self.apply_flat_fielding(gain_matrix)

            logger.info(_("Flat-field correction applied."))
            return result
        except Exception as e:
            logger.error(_("Failed to apply flat-field correction: %s"), e)
            raise FlatFieldingCorrectionException(e)

    def apply_flat_fielding(self, gain_matrix):
        """
        Applies the flat-field correction.

        :return: The corrected image as a numpy array
        """
        # Check if the gain_matrix has a mean close to 1
        if not np.isclose(gain_matrix.mean(), 1.0, atol=1e-2):
            logger.warning(
                f"Gain matrix mean is {gain_matrix.mean()}, which is not close to 1.0"
            )

        # Apply the correction per channel
        corrected_image = self.image * gain_matrix  # Broadcasting on (2616, 3899, 3)

        return corrected_image.astype(np.float32)

    def calculate_gain_matrix(self):
        """
        Calculates the gain matrix for the flat-field correction.

        :return: The gain matrix as a numpy array
        """
        # Calculate the flat field
        flat_field = self.white_patch  # Shape: (3,)
        mean_flat = np.mean(flat_field)

        logger.info(f"Mean flat field value: {mean_flat}")

        if mean_flat == 0:
            logger.error(
                "Mean flat field value is zero. Cannot proceed with flat-field correction."
            )
            raise ValueError("Mean flat field value is zero.")

        # Normalize the flat field per channel
        flat_field_normalized = flat_field / mean_flat  # Shape: (3,)

        # Prevent division by zero
        flat_field_normalized = np.where(
            flat_field_normalized == 0, 1.0, flat_field_normalized
        )

        # Calculate the gain matrix per channel
        gain_matrix = 1.0 / flat_field_normalized  # Shape: (3,)

        # Save correction data
        self.correction_data.set_flat_fielding_correction(gain_matrix)

        return gain_matrix

    def _get_reference_4th_gray(self):
        """
        Returns the reference for the 4th gray patch.

        :return: The reference white patch as a numpy array
        """
        return get_target_reference(self.campioni, 22)

    def _visualize_correction(
        self, flat_field, flat_field_normalized, gain_matrix, corrected_image
    ):
        """
        Visualizes the flat-field correction process.

        :param flat_field: The flat field values
        :param flat_field_normalized: The normalized flat field values
        :param gain_matrix: The gain matrix values
        :param corrected_image: The corrected image
        """
        plt.figure(figsize=(18, 6))

        # Plot Flat Field (per canale)
        for c in range(3):
            plt.subplot(2, 3, c + 1)
            plt.bar(["R", "G", "B"], flat_field, color=["red", "green", "blue"])
            plt.title(f"Flat Field - Channel {c+1}")
            plt.ylim([0, 1])

        # Plot Gain Matrix
        plt.subplot(2, 3, 4)
        plt.bar(["R", "G", "B"], gain_matrix, color=["red", "green", "blue"])
        plt.title("Gain Matrix per Canale")
        plt.ylim([0.99, 1.01])

        # Plot Corrected Image Stats
        plt.subplot(2, 3, 5)
        corrected_means = corrected_image.mean(axis=(0, 1))
        plt.bar(["R", "G", "B"], corrected_means, color=["red", "green", "blue"])
        plt.title("Mean Corrected per Canale")
        plt.ylim([0, 1])

        plt.tight_layout()
        plt.show()


