import multiprocessing
import time
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2luv
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma

from config.config import Mode
from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.common.exceptions.denoising_exceptions import DenoiseException
from core.utils.core_utils import extract_patch
from locales.localization import _
from log.logger import logger

from .utils import *


class DenoisingCorrector:
    def __init__(self, correction_data: ColorCorrectionData, image, patches, mode, process_cropped_image_only):
        """
        Initializes the DenoisingCorrector with the given parameters.

        :param correction_data: Object used to report denoising corrections.
        :param image: The image to be denoised (float32).
        :param tile_size: Size of each square tile for processing.
        :param overlap: Number of pixels to overlap between adjacent tiles to minimize artifacts.
        """

        self.correction_data = correction_data
        self.image = image
        self.patches = patches[0]
        self.mode = mode
        self.process_cropped_image_only = process_cropped_image_only

        self.tile_size = 256  # Tile size for processing
        self.overlap = 16  # Overlap between adjacent tiles

    def estimate_noise(self):
        """
        Estimates the overall noise level in the image by analyzing multiple patches.
        This method computes the noise standard deviation (sigma) for each color channel
        in the L*u*v* color space and then combines these values using ISO 15739 weighting
        to obtain a single noise metric (V).

        :return: The estimated noise level (V) for the image based on ISO 15739.
        """
        # Initialize lists to store noise levels for each L*u*v* channel
        noise_levels_l = []  # Noise levels for the L (luminance) channel
        noise_levels_u = []  # Noise levels for the u channel
        noise_levels_v = []  # Noise levels for the v channel

        # Extract the patches corresponding to the gray patches
        gray_patches = self.patches[18:]

        # Iterate over each patch selected for noise estimation
        for patch in gray_patches:
            # Extract the RGB values from the current patch based on its position and size
            if self.process_cropped_image_only:
                rgb_values = extract_patch(self.image, patch["absolute_position"], patch["size"])
            else:
                rgb_values = extract_patch(self.image, patch["cropped_image_position"], patch["size"])

            # Convert the extracted RGB patch to the L*u*v* color space for better noise analysis
            luv_values = rgb2luv(rgb_values)

            # Estimate the noise standard deviation for each channel in the L*u*v* space
            # The 'channel_axis=None' parameter indicates that the function should treat the input as a 2D array
            sigma_l = estimate_sigma(
                luv_values[:, :, 0], channel_axis=None
            )  # L channel
            sigma_u = estimate_sigma(
                luv_values[:, :, 1], channel_axis=None
            )  # u channel
            sigma_v = estimate_sigma(
                luv_values[:, :, 2], channel_axis=None
            )  # v channel

            # Append the estimated sigma values to their respective lists
            noise_levels_l.append(sigma_l)
            noise_levels_u.append(sigma_u)
            noise_levels_v.append(sigma_v)

        # Calculate the mean noise sigma for each L*u*v* channel across all patches
        mean_sigma_l = np.mean(noise_levels_l)  # Average noise in the L channel
        mean_sigma_u = np.mean(noise_levels_u)  # Average noise in the u channel
        mean_sigma_v = np.mean(noise_levels_v)  # Average noise in the v channel

        # Compute the weighted noise level (V) using ISO 15739 standards
        # These weights are based on the importance and sensitivity of each channel to noise
        V = (0.8 * mean_sigma_l) + (0.852 * mean_sigma_u) + (0.323 * mean_sigma_v)

        # Return the final weighted noise level
        return V

    def split_image_into_tiles(self):
        return split_image_into_tiles(self.image, self.tile_size, self.overlap)

    def recombine_tiles(self, tiles, positions):
        return recombine_tiles_with_windowing(
            tiles, positions, self.image.shape, self.tile_size, self.overlap
        )

    def run(self):
        """
        Runs the denoising correction on the input image.

        :return: The denoised image after applying the correction.
        """
        try:
            logger.info(_("Applying denoising correction"))

            if self.mode == Mode.ANALYSIS:
                noise_level = self.estimate_noise()
                self.correction_data.set_denoising(bm3d_strength=noise_level)
            else:
                noise_level = self.correction_data.get_denoising()

            denoised_image = self.apply_denoising(noise_level)

            logger.info(_("Denoising correction applied successfully"))
            return denoised_image
        except Exception as e:
            logger.error(_("Failed to apply denoising correction: %s"), e)
            raise DenoiseException(_("Error applying denoising: ") + str(e))

    def apply_denoising(self, noise_level):
        """
        Applies denoising to the entire image by processing it in tiles.
        Utilizes parallel processing to speed up the denoising of individual tiles.

        :return:
            - denoised_image: The denoised version of the original image.
        """

        # Split the image into smaller tiles
        tiles, positions = self.split_image_into_tiles()

        total_tiles = len(tiles)
        logger.debug(f"Number of tiles: {total_tiles}")

        start_time = time.time()

        # Partial function to fix some arguments and improve joblib performance
        denoise_func = partial(
            denoise_tile_joblib, total_tiles=total_tiles, noise_level=noise_level
        )

        # Perform parallel denoising of tiles using joblib's Parallel
        denoised_tiles = Parallel(
            n_jobs=multiprocessing.cpu_count()
            - 1,  # Use all available CPU cores except one
            backend="loky",  # Explicitly specify the 'loky' backend for process-based parallelism
            verbose=0,
        )(delayed(denoise_func)(tile, idx) for idx, tile in enumerate(tiles))

        logger.debug(f"Total denoising time: {time.time() - start_time:.2f} seconds")

        # Recombine the denoised tiles into the full image
        denoised_image = self.recombine_tiles(denoised_tiles, positions)
        # Evaluate the quality of the denoised image
        self.evaluate_quality(self.image, denoised_image)

        return denoised_image

    def evaluate_quality(self, original_image, denoised_image):
        """
        Evaluate the quality of the denoised image compared to the original using PSNR and SSIM.

        :param original_image: The original image before denoising.
        :param denoised_image: The denoised image after applying denoising.
        """

        # Check if the shapes of the original and denoised images are the same
        if original_image.shape != denoised_image.shape:
            raise ValueError(
                _("Original and denoised images must have the same shape.")
            )

        # Check for NaN or Inf values in the denoised image
        if not np.isfinite(denoised_image).all():
            raise ValueError(
                _("The denoised image contains non-finite values (NaN or Inf).")
            )

        # Calculate PSNR (Peak Signal-to-Noise Ratio) across the RGB image
        # PSNR indicates the ratio between the maximum possible power of the image signal and the power of noise.
        # Values typically range as follows:
        #   >40 dB: Excellent quality, near identical to the original
        #   30-40 dB: Good quality, minor differences from the original
        #   20-30 dB: Fair quality, noticeable differences
        #   <20 dB: Poor quality, significant visible differences
        psnr_value = peak_signal_noise_ratio(
            original_image, denoised_image, data_range=1.0
        )

        # Calculate SSIM (Structural Similarity Index) across the RGB image
        # SSIM measures the structural similarity between two images, considering luminance, contrast, and structure.
        # The value ranges from 0 to 1, where:
        #   1.0: Perfect structural match
        #   0.8-1.0: High quality, close to original
        #   0.6-0.8: Fair quality, some structural differences
        #   <0.6: Low quality, significant structural differences
        # 'win_size=7' sets the size of the window for local similarity comparisons.
        # 'channel_axis=-1' specifies that the last axis corresponds to the color channels (RGB).
        ssim_value = structural_similarity(
            original_image,
            denoised_image,
            multichannel=True,
            data_range=1.0,
            win_size=7,
            channel_axis=-1,
        )

        # Log the PSNR and SSIM results for debugging
        logger.debug(f"PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
