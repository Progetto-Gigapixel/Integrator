import numpy as np

from config.config import Mode
from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.common.exceptions.white_balance_exceptions import WhiteBalanceException
from locales.localization import _
from log.logger import logger
from utils.utils import get_target_reference

from .utils import get_cat_matrix, get_gain, transform_lms, xyz_to_lms


class WhiteBalance:
    def __init__(
        self,
        correction_data: ColorCorrectionData,
        campioni,
        equipment,
        image,
        patches,
        mode,
    ):
        self.equipment = equipment
        self.correction_data = correction_data
        self.campioni = campioni
        self.image = image
        self.patches = patches
        self.target_white = self._get_reference_4th_gray()
        self.mode = mode

    def run(self):
        """
        Applies the white balance correction.

        :return: The corrected image as a numpy array
        """
        try:
            logger.info(_("Applying white balance correction"))

            if self.mode == Mode.ANALYSIS:
                white_balanced = self.apply_white_balance()
            else:
                white_balanced = self.apply_white_balance_development()

            logger.info(_("White balance correction applied successfully"))
            return white_balanced
        except Exception as e:
            logger.error(_("Failed to apply white balance correction: %s"), e)
            raise WhiteBalanceException(e)

    def apply_white_balance(self):
        src_white_point = self.patches[0][21]["rgb_values"]
        dst_white_point = self.target_white

        white_balanced = self.apply_chromatic_adaptation(
            src_white_point, dst_white_point, self.image, cat_type="BRADFORD"
        )

        self.correction_data.set_white_balance_correction(
            src_white_point=src_white_point, dst_white_point=dst_white_point
        )
        return white_balanced

    def apply_white_balance_development(self):
        src_white_point, dst_white_point = (
            self.correction_data.get_white_balance_correction()
        )

        white_balanced = self.apply_chromatic_adaptation(
            src_white_point, dst_white_point, self.image, cat_type="BRADFORD"
        )

        return white_balanced

    def apply_chromatic_adaptation(
        self, src_white_point, dst_white_point, src_img, cat_type="BRADFORD"
    ):
        # convert white point in 'sRGB' to 'XYZ'
        # and normalize 'XYZ' that 'Y' as luminance
        xyz_src = src_white_point #cie_rgb_to_xyz(src_white_point)
        # n_xyz_src = normalize_xyz(xyz_src)
        xyz_dst = dst_white_point # cie_rgb_to_xyz(dst_white_point)
        # n_xyz_dst = normalize_xyz(xyz_dst)
        # get CAT type matrix
        cat_m = get_cat_matrix(cat_type)
        # convert 'XYZ' to 'LMS'
        lms_src = xyz_to_lms(xyz_src, cat_m)
        lms_dst = xyz_to_lms(xyz_dst, cat_m)
        # LMS gain by scaling destination with source LMS
        gain = get_gain(lms_src, lms_dst)
        # multiply CAT matrix with LMS gain factors
        ca_transform = transform_lms(cat_m, gain)
        # convert 'RGB' source image to 'XYZ'
        #src_img_xyz = cie_rgb_to_xyz(src_img)
        # apply CAT transform to image
        corrected_img = src_img @ ca_transform.T
        # convert back 'XYZ' to 'sRGB' image
        #corrected_img = xyz_to_cie_rgb(transformed_xyz)
        # convert to float32
        f32_img = np.float32(corrected_img)

        return f32_img

    def _get_reference_4th_gray(self):
        """
        Returns the reference for the 4th gray patch.

        :return: The reference for the 4th gray patch
        """
        return get_target_reference(self.campioni, 22)
