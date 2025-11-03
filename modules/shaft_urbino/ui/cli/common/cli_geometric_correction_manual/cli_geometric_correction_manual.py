# cli_geometric_correction_manual
from core.common.exceptions.geometry_correction_exceptions import (
    GeometricCorrectionManualException,
)
from locales.localization import _
from log.logger import logger
from utils.utils import spacer


class CliGeometricCorrectionManual:
    def __init__(self, equipment):
        self.equipment = equipment

    # Run manual geometric corrections
    def run_manual_corrections(self):
        # Collect the distortion coefficients from the user
        coefficients = self.collect_coefficients(3)

        # Request the camera crop factor from the user
        crop_factor = self.request_camera_crop_factor()

        print("\n")
        print(_("Distortion coefficients collected: {}").format(coefficients))
        print(_("Camera crop factor received: {}").format(crop_factor))
        spacer()

        return crop_factor, coefficients

    def cli_correction_handler(self):
        spacer()

        try:
            # Ask the user if they want to proceed with manual correction
            response = input(
                _("Do you want to proceed with manual geometric correction? (y/n): ")
            )

            if response.lower() == _("y"):
                crop_factor, coefficients = self.run_manual_corrections()

                return crop_factor, coefficients
            else:
                spacer()
                logger.info(_("Geometric correction skipped."))
                return None, None
        except Exception as e:
            raise GeometricCorrectionManualException(
                _("Error applying manual geometric correction.") + str(e)
            )

    def request_camera_crop_factor(self):
        print("\n")
        response = self.request_value(
            _("Enter the crop factor of your camera (e.g., 1.5): "), float, True
        )
        return response

    def collect_coefficients(self, num_coefficients):
        print("\n")
        coefficients = []
        # Collect the radial coefficients k
        for i in range(1, num_coefficients + 1):
            coefficient = self.request_value(
                _("Enter coefficient k{} (e.g., -0.01): ").format(i), float
            )
            coefficients.append(coefficient)

        # Ask the user if they want to include tangential coefficients
        print("\n")
        include_tangential = self.request_value(
            _("Do you want to include tangential coefficients p1 and p2? (y/n): "), str
        ).lower()
        if include_tangential == _("y"):
            print("\n")
            for i in range(1, 3):
                p = self.request_value(
                    _("Enter coefficient p{} (e.g., 0.001): ").format(i), float
                )
                coefficients.append(p)
        else:
            print(_("Tangential coefficients p1 and p2 will be set to 0.0."))
            # Set the tangential coefficients to 0.0 if not included
            coefficients.extend([0.0, 0.0])

        return coefficients

    @staticmethod
    # Request a value from the user with specified type and positivity
    def request_value(message, value_type, is_positive=False):
        while True:
            response = input(message)
            try:
                value = value_type(response)
                if is_positive and value <= 0:
                    print(_("Please enter a positive value."))
                else:
                    return value
            except ValueError:
                print(
                    _("Invalid value. Please enter a valid value of type {}.").format(
                        value_type.__name__
                    )
                )
