import lensfunpy
import numpy as np

from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.common.exceptions.lensfun_exceptions import (
    LensfunException,
    LensfunMissingDataException,
)
from core.utils.core_utils import get_lensfun_db
from locales.localization import _
from log.logger import logger


class LensfunCorrector:
    def __init__(
        self,
        correction_data: ColorCorrectionData,
        image_array,
        equipment,
        output_format,
    ):
        """
        Initializes the lensfun corrector with the specified parameters.

        :param correction_data: The correction data object for reporting corrections
        :param image_array: The image to be corrected
        :param equipment: The equipment used to capture the image
        :param output_format: The output format of the corrected image
        """
        self.correction_data = correction_data
        self.original_dtype = image_array.dtype
        if image_array.dtype != np.float32:
            self.image_array = (
                image_array.astype(np.float32) / np.iinfo(image_array.dtype).max
            )
        else:
            self.image_array = image_array
        self.equipment = equipment
        self.output_format = output_format
        self.db = get_lensfun_db()

    def _run(self):
        """
        Runs the lensfun correction.

        :return: The corrected image as a numpy array
        """
        try:
            camera_make = self.equipment.camera_make
            camera_model = self.equipment.camera_model
            lens_make = self.equipment.lens_make
            lens_model = self.equipment.lens_model

            cameras = self.db.find_cameras(camera_make, camera_model)
            if not cameras:
                raise LensfunMissingDataException(
                    _("No camera found in the lensfun database.")
                )
            camera = cameras[0]

            lenses = self.db.find_lenses(camera, lens_make, lens_model)
            if not lenses:
                raise LensfunMissingDataException(
                    _("No lens found in the lensfun database.")
                )
            lens = lenses[0]

            self._prepare_modifier(camera, lens)
            undistorted_img = self.apply_correction()
            return undistorted_img
        except Exception as e:
            logger.error(_("Failed to apply lensfun correction: %s"), e)
            raise LensfunException(e)
        # return self.reverse_normalization(undistorted_img)

    def _prepare_modifier(self, camera, lens):
        """
        Prepares the modifier for the lensfun correction.

        :param camera: The camera object from the lensfun database
        :param lens: The lens object from the lensfun database
        """
        focal_length = float(self.equipment.focal_length)
        aperture = float(self.equipment.aperture)
        distance = 1

        self.mod = lensfunpy.Modifier(
            lens,
            camera.crop_factor,
            self.image_array.shape[1],
            self.image_array.shape[0],
        )
        self.mod.initialize(focal_length, aperture, distance, pixel_format=np.float32)

    def reverse_normalization(self, undistorted_img):
        """
        Reverses the normalization applied to the image.

        :param undistorted_img: The undistorted image

        :return: The undistorted image with the original data type
        """
        if undistorted_img.dtype == np.float32 and hasattr(self, "original_dtype"):
            if np.issubdtype(self.original_dtype, np.integer):
                max_value = np.iinfo(self.original_dtype).max
            elif np.issubdtype(self.original_dtype, np.floating):
                max_value = np.finfo(self.original_dtype).max
            else:
                raise ValueError("Unsupported data type '%s'." % self.original_dtype)

            # Clamp the values to the range [0, 1] before multiplication
            undistorted_img = np.clip(undistorted_img, 0, 1)

            # Handle any infinite or NaN values
            undistorted_img[np.isinf(undistorted_img)] = 0
            undistorted_img[np.isnan(undistorted_img)] = 0

            return (undistorted_img * max_value).astype(self.original_dtype)

        return undistorted_img

    def apply_correction(self):
        raise NotImplementedError(
            "The apply_correction method must be implemented in the subclass."
        )
