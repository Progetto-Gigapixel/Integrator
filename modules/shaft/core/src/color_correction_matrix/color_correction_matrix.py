import colour
import numpy as np
from sklearn.linear_model import RidgeCV

from config.config import Mode
from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.common.exceptions.color_correction_matrix_exceptions import (
    ColorCorrectionMatrixException,
)
from core.utils.core_utils import (
    get_measured_values_from_patches,
    get_reference_values_as_array
)
from locales.localization import _
from log.logger import logger



def et_hom_lin(rs, gs, bs, rr, gr, br):
    # Linear homogeneous transformation
    num_samples = len(rr)
    A = np.zeros((num_samples * 3, 12))
    b = np.zeros(num_samples * 3)

    for i in range(num_samples):
        offset = i * 3
        rgb1 = [rr[i], gr[i], br[i], 1]

        A[offset, 0:4] = rgb1
        A[offset + 1, 4:8] = rgb1
        A[offset + 2, 8:12] = rgb1

        b[offset] = rs[i]
        b[offset + 1] = gs[i]
        b[offset + 2] = bs[i]

    # Solve using Singular Value Decomposition (SVD)
    U, D, Vt = np.linalg.svd(A, full_matrices=False)
    D_inv = np.diag(1.0 / D)
    a = Vt.T @ D_inv @ U.T @ b

    # Construct the transformation matrix
    T = np.eye(4)
    T[0, 0:4] = a[0:4]
    T[1, 0:4] = a[4:8]
    T[2, 0:4] = a[8:12]

    return T


def ti_hom_lin(T, im):
    imr = im[:, :, 0].flatten()
    imgr = im[:, :, 1].flatten()
    imb = im[:, :, 2].flatten()

    rgb_vec = np.vstack([imr, imgr, imb, np.ones_like(imr)])

    transformed = T @ rgb_vec

    imr = transformed[0, :].reshape(im.shape[:2])
    imgr = transformed[1, :].reshape(im.shape[:2])
    imb = transformed[2, :].reshape(im.shape[:2])

    img = np.stack((imr, imgr, imb), axis=-1)
    img = np.clip(img, 0, 1)
    return img


class ColorCorrectionMatrix:
    def __init__(
            self,
            correction_data: ColorCorrectionData,
            patches,
            reference_values,
            image,
            mode,
    ):
        """
        Initializes the CCM calculation with measured and reference values.

        :param measured_values: Measured RGB values from the image
        :param reference_values: Known reference RGB values
        :param image: The image to be corrected
        """

        self.correction_data = correction_data
        self.image = image
        self.ccm_matrix = None
        self.ridge_ccm_matrix = None
        self.mode = mode
        self.reference_values = get_reference_values_as_array(reference_values)

        if mode == Mode.ANALYSIS:
            self.measured_values = get_measured_values_from_patches(patches)

    def calculate_ccm_ridge_cv(self):
        """
        Calculate the Color Correction Matrix (CCM) using Ridge Regression with cross-validation.
        """

        alphas = np.logspace(-3, 3, 100)

        ridge = RidgeCV(alphas=alphas, fit_intercept=False, cv=5)

        ridge.fit(self.measured_values, self.reference_values)

        self.ridge_ccm_matrix = ridge.coef_.astype(np.float32)

    def validate_ccm(self):
        """
        Validate the calculated CCM by applying it to the measured values and computing the Mean Squared Error (MSE).
        """
        # Apply the CCM matrix to the measured values
        corrected_colors = np.dot(self.measured_values, self.ridge_ccm_matrix)

        # Calculate Mean Squared Error (MSE) between corrected and reference values
        mse = np.mean((corrected_colors - self.reference_values) ** 2)

        # Log the MSE
        logger.debug(f"Validation MSE: {mse}")

    def apply_finlayson(self):
        method = "Finlayson 2015"

        corrected_image = colour.colour_correction(
            self.image,
            self.measured_values,
            self.reference_values,
            method=method,
        )


        self.correction_data.set_finlayson_ccm(
            measured_values=self.measured_values.tolist()
        )
        return corrected_image

    def compute_ccm(self):
        n = self.measured_values.shape[0]

        if n == 24:
            l, h, wy, wx = 6, 4, 4, 1
        else:
            l, h, wy, wx = 14, 10, 5, 5
        rv = self.reference_values
        rcc = rv[:, 0].reshape(h, l)
        gcc = rv[:, 1].reshape(h, l)
        bcc = rv[:, 2].reshape(h, l)
        mv = self.measured_values
        rrec = mv[:, 0].reshape(h, l)
        grec = mv[:, 1].reshape(h, l)
        brec = mv[:, 2].reshape(h, l)

        t = et_hom_lin(rcc.flatten(), gcc.flatten(), bcc.flatten(), rrec.flatten(), grec.flatten(), brec.flatten())

        corr = np.stack((rrec, grec, brec), axis=-1)

        self.ccm_matrix = t

        self.correction_data.set_ccm(self.ccm_matrix)

    def apply_ccm(self):
        """
        Apply the standard color correction matrix to the image.

        :return: The corrected image as a numpy array
        """

        if self.ccm_matrix is None:
            raise ColorCorrectionMatrixException("Standard CCM has not been calculated yet.")

        corrected_image = ti_hom_lin(self.ccm_matrix, self.image)
        return corrected_image

    def compute_ridge_correction(self):
        """
        Apply the Ridge color correction matrix to the image.

        :return: The corrected image as a numpy array
        """

        # Calculate the CCM
        self.calculate_ccm_ridge_cv()

        # Validate the CCM
        self.validate_ccm()

        self.correction_data.set_ridge_ccm(
            ccm=self.ridge_ccm_matrix.tolist(), measured_values=self.measured_values.tolist()
        )

    def apply_ridge_correction(self):
        """
        Apply the Ridge color correction matrix to the image.

        :return: The corrected image as a numpy array
        """

        if self.ridge_ccm_matrix is None:
            raise ColorCorrectionMatrixException("Ridge CCM has not been calculated yet.")

        ccm, measured_values = self.correction_data.get_ridge_ccm()

        self.ccm_matrix = np.array(ccm)
        self.measured_values = np.array(measured_values)

        # Apply CCM to the result array
        corrected_image = self.apply_ridge_ccm()

        return corrected_image

    def apply_ridge_ccm(self):
        """
        Apply the calculated CCM to the image.

        :param image: The image to be corrected
        :return: The corrected image as a numpy array
        """

        # Flatten the image to apply the CCM
        h, w, c = self.image.shape

        flat_image = self.image.reshape(-1, c)


        # Apply the CCM
        corrected_flat_image = np.dot(flat_image, self.ccm_matrix.T)

        corrected_image = corrected_flat_image.reshape(h, w, c)

        return corrected_image

    def compute_finlayson_correction(self):
        """
        Compute the Finlayson color correction matrix to the image.

        :return: The corrected image as a numpy array
        """

        # Apply Finlayson correction
        corrected_image = self.apply_finlayson()

        self.correction_data.set_finlayson_ccm(
            measured_values=self.measured_values.tolist()
        )

        return corrected_image

    def apply_finlayson_correction(self):
        """
        Apply the Finlayson color correction matrix to the image.

        :return: The corrected image as a numpy array
        """
        measured_values = self.correction_data.get_finlayson_ccm()

        self.measured_values = np.array(measured_values)

        # Apply Finlayson correction
        corrected_image = self.apply_finlayson()

        return corrected_image

    def run_ccm(self):
        """
        Applies the ridge color correction matrix.

        :return: The corrected image as a numpy array
        """

        try:
            logger.info(_("Applying color correction matrix to image: %s"), "")
            if self.mode == Mode.ANALYSIS:
                self.compute_ccm()
            else:
                self.ccm_matrix = self.correction_data.get_ccm()


            result = self.apply_ccm()
            logger.info(_("Color correction matrix applied."))
            return result
        except Exception as e:
            logger.error(_("Failed to apply  ccm colorcorrection matrix: %s"), e)
            raise ColorCorrectionMatrixException(e)

    def run_ridge_ccm(self):
        """
        Applies the ridge color correction matrix.

        :return: The corrected image as a numpy array
        """

        try:
            logger.info(_("Applying ridge correction matrix..."))
            if self.mode == Mode.ANALYSIS:
                self.compute_ridge_correction()
            else:
                self.ridge_ccm_matrix = self.correction_data.get_ridge_ccm()

            result = self.apply_ridge_correction()
            logger.info(_("Color correction matrix applied."))
            return result
        except Exception as e:
            logger.error(_("Failed to apply ridge color correction matrix: %s"), e)
            raise ColorCorrectionMatrixException(e)

    def run_finlayson_ccm(self):
        """
        Applies the finlayson color correction matrix.

        :return: The corrected image as a numpy array
        """

        try:
            logger.info(_("Applying fin color correction matrix..."))
            if self.mode == Mode.ANALYSIS:
                result = self.compute_finlayson_correction()

            result = self.apply_finlayson_correction()
            logger.info(_("Additional color correction matrix applied."))
            return result
        except Exception as e:
            logger.error(_("Failed to apply additional fin color correction matrix: %s"), e)
            raise ColorCorrectionMatrixException(e)
