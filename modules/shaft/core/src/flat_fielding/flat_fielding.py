# import matplotlib.pyplot as plt
import numpy as np

from config.config import Mode
from core.common.exceptions.flat_fielding_correction_exceptions import (
    FlatFieldingCorrectionException,
)
from core.src.decode_raw.decode_raw import RawDecoder
from core.src.find_color_checker.find_color_checker_segmentation import create_image_mosaic
from core.utils.core_utils import xyz_2_hsv, hsv_2_xyz
from locales.localization import _
from log.logger import logger
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import resize
from scipy.ndimage import gaussian_filter


class FlatFieldingCorrector:
    def __init__(self, correction_data, original_image, cropped_image, white_field_path, color_checker_finder, work_on_cropped_image, equipment, mode, sigma, correction_matrix_path):
        """
        Initializes the flat-fielding corrector with the specified parameters.

        :param correction_data: The correction data object for reporting corrections
        :param campioni: The campioni object for the target reference
        :param image: The image to be corrected (float32)
        """
        self.original_image = original_image
        self.cropped_image = cropped_image
        self.white_field_path = white_field_path
        self.equipment = equipment
        self.mode = mode
        self.sigma = sigma
        self.correction_data = correction_data
        self.correction_matrix_path = correction_matrix_path
        self.color_checker_finder = color_checker_finder
        self.work_on_cropped_image = work_on_cropped_image


    def run(self):
        """
        Applies the flat-field correction.

        :return: The corrected image as a numpy array
        """
        try:
            logger.info(_("Applying flat-field correction..."))

            if self.mode == Mode.ANALYSIS:
                flat_field_correction_matrix = self.compute_flat_fielding()
            else:
                flat_field_correction_matrix = self.load_matrix()

            result = self.apply_flat_fielding(flat_field_correction_matrix)

            logger.info(_("Flat-field correction applied."))
            return result
        except Exception as e:
            logger.error(_("Failed to apply flat-field correction: %s"), e)
            raise FlatFieldingCorrectionException(e)

    def apply_flat_fielding(self, adjustment_matrix):
        """
        Applies the flat-field correction.

        :return: The corrected image as a numpy array
        """
        if self.mode == Mode.ANALYSIS and self.work_on_cropped_image:
            image = self.cropped_image
            if self.work_on_cropped_image and self.mode == Mode.ANALYSIS:
                absolute_positions = [d['absolute_position'] for d in self.color_checker_finder.color_checker_data[0]]
                image, mosaic_patch_coordinates = create_image_mosaic(self.original_image, absolute_positions, [25, 25])
                adjustment_matrix, dummy = create_image_mosaic(adjustment_matrix, absolute_positions, [25, 25])
        else:
            image = self.original_image

        # Convert cc_image to HSV color space
        hsv_cc_image = xyz_2_hsv(image)

        # Adjust the intensity channel by dividing by the flat field correction
        v_cc_image = hsv_cc_image[:, :, 2] / adjustment_matrix

        # Reconstruct the HSV image
        hsv_cc_image = np.dstack((hsv_cc_image[:, :, 0], hsv_cc_image[:, :, 1], v_cc_image))

        # Convert back to xyz
        cc_image = hsv_2_xyz(hsv_cc_image)

        return cc_image

    def compute_flat_fielding(self):
        """
        Calculates the adjustment matrix for the flat-field correction.

        :return: The adjustment matrix as a numpy array
        """
        decoder = RawDecoder(
            self.equipment, self.white_field_path, "TIFF", mode="AM"
        )
        white_image = decoder.decode_raw(mode="AM")

        # For debugging purposes only. Often I don't have the right white image. To use it anyway, I check if the image dimensions are equal, otherwise resize white_image
        if self.original_image.shape[:2] != white_image.shape[:2]:
            white_image = resize(white_image, (self.original_image.shape[0], self.original_image.shape[1]), anti_aliasing=True)

        # if self.work_on_cropped_image and self.mode == Mode.ANALYSIS:
        #     absolute_positions = [d['absolute_position'] for d in self.color_checker_finder.color_checker_data[0]]
        #     white_image, mosaic_patch_coordinates = create_image_mosaic(white_image, absolute_positions, [25, 25])

            # Convert white_image to HSV color space
        hsv_white_image = rgb2hsv(white_image)

        # Extract the intensity component (V)
        intensity_difference = hsv_white_image[:, :, 2]

        # Apply a Gaussian filter to reduce noise
        intensity_difference = gaussian_filter(intensity_difference, self.sigma)

        # Find the maximum intensity value
        max_destination = np.max(intensity_difference)

        # Normalize the intensity
        intensity_difference += (1 - max_destination)

        self.save_matrix(intensity_difference)

        return intensity_difference

    def save_matrix(self, matrix):
        np.savez_compressed(self.correction_matrix_path, matrix=matrix)

    def load_matrix(self):
        with np.load(self.correction_matrix_path) as data:
            return data['intensity_difference']




