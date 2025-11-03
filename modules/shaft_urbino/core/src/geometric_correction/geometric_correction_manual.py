import cv2
import numpy as np

from core.common.exceptions.geometry_correction_exceptions import (
    GeometricCorrectionManualException,
)
from locales.localization import _
from log.logger import logger
from utils.utils import read_image


class GeometricCorrectionManual:
    def __init__(self, equipment):
        self.equipment = equipment

    # Apply manual geometric correction
    def apply_manual_correction(self, image_path, crop_factor, coefficients):
        try:
            logger.info(_("Applying manual geometric correction..."))
            image = read_image(image_path)

            # Using the focal length from equipment
            focal_length = float(self.equipment.focal_length)

            # Calculate the center of the image
            cx = image.shape[1] / 2
            cy = image.shape[0] / 2

            # Calculate the effective focal length considering the crop factor
            effective_focal_length = focal_length * crop_factor

            # Configure the camera matrix
            cam_matrix = np.array(
                [
                    [effective_focal_length, 0, cx],
                    [0, effective_focal_length, cy],
                    [0, 0, 1],
                ]
            )

            # Radial and tangential distortion coefficients
            dist_coeffs = np.array(
                [
                    coefficients[0],  # k1
                    coefficients[1],  # k2
                    coefficients[3],  # p1
                    coefficients[4],  # p2
                    coefficients[2],  # k3
                ]
            )

            # Get the optimized new camera matrix
            new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
                cam_matrix, dist_coeffs, (image.shape[1], image.shape[0]), 1
            )

            # Apply distortion correction
            undistorted_img = cv2.undistort(
                image, cam_matrix, dist_coeffs, None, new_cam_matrix
            )

            # Crop the image to remove black borders
            x, y, w, h = roi
            undistorted_img = undistorted_img[y : y + h, x : x + w]

            logger.info(_("Manual geometric correction applied successfully."))
            return undistorted_img

        except Exception as e:
            raise GeometricCorrectionManualException(e)
