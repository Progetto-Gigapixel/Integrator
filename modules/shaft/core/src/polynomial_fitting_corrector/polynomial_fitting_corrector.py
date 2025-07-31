import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from config.config import Mode
from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.utils.core_utils import (
    get_measured_values_from_patches,
    get_reference_values_as_array
)
from locales.localization import _
from log.logger import logger
import skimage


def get_weights():
    weights = np.ones(24) / 18
    weights[18:24] = 1 / 6
    return weights
class PolynomialFittingCorrector:
    def __init__(
        self,
        correction_data: ColorCorrectionData,
        image: np.ndarray,
        reference_values,
        measured_patches,
        mode: Mode,
        fitting_degree: int,
    ):
        """
        Initializes the polynomial fitting corrector with the specified parameters.

        :param correction_data: The correction data object for reporting corrections.
        :param image: The RGB linear image to be corrected as a NumPy array (h, w, 3) of type float32.
        :param reference_values: Dictionary containing reference RGB values for each patch.
        :param measured_values: List of measured RGB values from the image for each patch.
        :param fitting_degree: The degree of the polynomial fitting (e.g., 1, 2 or 3).
        """
        self.correction_data = correction_data
        self.image = image
        self.reference_values = get_reference_values_as_array(reference_values)
        self.mode = mode

        if mode == Mode.ANALYSIS:
            self.fitting_degree = int(fitting_degree)
            self.measured_patches = get_measured_values_from_patches(measured_patches)


    def weighted_least_square_calculate_correction_model(self):
        measured_patches_linear = np.float64(self.measured_patches)
        reference_patches_linear = np.float64(self.reference_values)

        weights = get_weights()
        correction_model = []

        for channel in range(3):
            wls = sm.WLS(
                measured_patches_linear[:, channel],
                reference_patches_linear[:, channel],
                weights=weights,
            )

            correction_model.append(wls.fit())

        return correction_model

    def weighted_least_square_apply_correction(self):
        correction_model = self.weighted_least_square_calculate_correction_model()

        # Define the data to be fit with some noise:
        flattened_predicted = None
        predicted = []
        h, w, c = self.image.shape
        linear_image = np.nan_to_num(self.image, nan=0.0)

        for channel in range(3):
            flattened_channel = np.float64(
                np.reshape(linear_image[:, :, channel], (h * w))
            )
            flattened_predicted = correction_model[channel].get_prediction(
                flattened_channel
            )
            predicted.append(
                np.float32(np.reshape(flattened_predicted.predicted, (h, w)))
            )

        corrected_image = np.dstack((predicted[0], predicted[1], predicted[2]))

        return corrected_image


    def poly_calculate_correction_models(self):
        # Compute correction according to polyfit model

        measured_patches_linear = np.float64(self.measured_patches)
        reference_patches_linear = np.float64(self.reference_values)

        weights = get_weights()**2

        # Prepare features for polynomial fitting
        poly = PolynomialFeatures(degree=int(self.fitting_degree), include_bias=True)
        X_patches = measured_patches_linear  # Input: measured (24, 3)
        X_patches_poly = poly.fit_transform(X_patches)  # Polynomial transformation

        # Initialize models for each channel
        models = {}

        for channel in range(3):  # 0: R, 1: G, 2: B
            # Train the linear regression model with weights
            model = LinearRegression()
            model.fit(
                X_patches_poly,
                reference_patches_linear[:, channel],
                sample_weight=weights,
            )
            models[channel] = model

        return models, poly

    def poly_apply_correction(self):
        """
        Applies polynomial fitting correction to the image with a separate model for each channel.

        :return: The corrected image as a NumPy array (h, w, 3) of type float32.
        """
        # Calculate the correction models and polynomial features
        models, poly = self.poly_calculate_correction_models()

        # Convert the image from sRGB to linear RGB (float64 for precision)
        # image_linear = self.sRGB_2_linear_float64(self.image)
        image_linear = self.image

        # Get the original dimensions of the image
        h, w, c = image_linear.shape

        # Reshape the image to apply the correction
        image_reshaped = image_linear.reshape(-1, 3)  # Shape: (h*w, 3)
        image_poly = poly.transform(image_reshaped)  # Shape: (h*w, n_features)

        corrected_linear = np.zeros_like(image_reshaped)

        for channel in range(3):
            corrected_channel = models[channel].predict(image_poly)
            corrected_linear[:, channel] = corrected_channel

        # Reshape and final conversion to linear float32
        corrected_linear = np.clip(corrected_linear.reshape(h, w, 3), 0, 1)

        # remove corrections out of colorchecker range, leading to unpredictable results
        if self.fitting_degree > 1:
            mask = (self.image < 0.2) & (self.image > 0.8)
            corrected_linear = np.where(mask, corrected_linear, self.image)

        return corrected_linear

    def run_poly(self):
        """
        Applies polynomial fitting correction to the image.

        :param mode: The mode of the polynomial fitting correction.

        :return: The corrected image as a NumPy array (h, w, 3) of type float32.
        """
        try:
            logger.info("Starting polynomial fitting correction process...")
            if self.mode == Mode.ANALYSIS:
                corrected_image = self.poly_apply_correction()
                self.correction_data.set_polynomial_correction(
                    self.fitting_degree, self.measured_patches
                )
            else:
                self.fitting_degree, measured_patches = (
                    self.correction_data.get_polynomial_correction()
                )
                self.measured_patches = np.array(measured_patches)
                corrected_image = self.poly_apply_correction()

            logger.info("Polynomial fitting correction applied successfully.")
            return corrected_image
        except Exception as e:
            logger.error(f"Error in polynomial fitting correction: {e}")
            raise

    def run_wls(self):
        """
        Applies weighted least squares correction to the image.

        :return: The corrected image as a NumPy array (h, w, 3) of type float32.
        """
        try:
            logger.info("Starting WLS correction process...")
            if self.mode == Mode.ANALYSIS:
                corrected_image = self.weighted_least_square_apply_correction()
                self.correction_data.set_wls_correction(self.measured_patches)
            else:
                measured_patches = self.correction_data.get_wls_correction()
                self.measured_patches = np.array(measured_patches)
                corrected_image = self.weighted_least_square_apply_correction()
            logger.info("WLS correction applied successfully.")
            return corrected_image
        except Exception as e:
            logger.error(f"Error in WLS correction: {e}")
            raise

