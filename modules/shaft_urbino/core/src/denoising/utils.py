import bm3d
import cv2
import numpy as np

from core.common.exceptions.denoising_exceptions import DenoiseException
from log.logger import logger


def split_image_into_tiles(image, tile_size, overlap=0):
    """
    Splits the input image into smaller tiles with specified size and overlap.

    :param image: Input image as a NumPy array.
    :param tile_size: Size of each square tile.
    :param overlap: Number of pixels to overlap between adjacent tiles.
    :return:
        - tiles: List of image tiles.
        - positions: List of (x, y) positions indicating the top-left corner of each tile in the original image.
    """
    tiles = []
    positions = []
    height, width, channels = image.shape

    step = tile_size - overlap
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Ensure the tile does not exceed image boundaries
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile = image[y:y_end, x:x_end]

            tiles.append(tile)
            positions.append((x, y))
    return tiles, positions


def recombine_tiles_with_windowing(tiles, positions, image_shape, tile_size, overlap=0):
    """
    Recombines denoised image tiles into a single image using Gaussian windowing to ensure smooth transitions.

    Parameters:
        tiles (list of np.ndarray): List of denoised image tiles.
        positions (list of tuple): List of (x, y) tuples indicating the top-left corner position of each tile in the original image.
        image_shape (tuple): Shape of the original image as (height, width, channels).
        tile_size (int): Size of each square tile.
        overlap (int, optional): Number of pixels overlapping between adjacent tiles. Default is 0.

    Returns:
        np.ndarray: The recombined denoised image.
    """
    height, width, channels = image_shape
    # Initialize the denoised image and weight matrix with zeros
    denoised_image = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    # Create a 1D Gaussian kernel
    # sigma is set to tile_size/6 to control the spread of the Gaussian
    kernel_1d = cv2.getGaussianKernel(tile_size, sigma=tile_size / 6)
    # Create a 2D Gaussian window by taking the outer product of the 1D kernel with itself
    window = np.outer(kernel_1d, kernel_1d)
    # Expand dimensions to apply the window to all color channels
    window = window[..., np.newaxis]
    # Normalize the window so that the sum of all weights is 1
    window /= np.sum(window)

    # Iterate over each tile and its corresponding position
    for tile, (x, y) in zip(tiles, positions):
        h, w, _ = tile.shape  # Get the height and width of the current tile

        # Resize the window to match the tile size
        window_resized = window[:h, :w, :]

        # Apply the Gaussian window to the denoised tile
        denoised_image[y : y + h, x : x + w, :] += tile * window_resized

        # Accumulate the weights to handle overlapping regions
        weight[y : y + h, x : x + w] += window_resized[:, :, 0]

    # Prevent division by zero by setting any zero weights to one
    weight[weight == 0] = 1.0

    # Normalize the denoised image by dividing by the accumulated weights
    denoised_image /= weight[..., np.newaxis]

    return denoised_image


def denoise_tile_joblib(tile, tile_index, total_tiles, noise_level):
    """
    Denoises a single tile using the BM3D algorithm. Intended for parallel processing with joblib.

    :param tile: The image tile to denoise.
    :param tile_index: The index of the current tile (for logging purposes).
    :param total_tiles: Total number of tiles (for logging purposes).

    :return:
        - denoised_tile: The denoised image tile.
    """
    try:
        logger.debug(f"Denoising tile {tile_index + 1}/{total_tiles}")

        # Apply BM3D denoising on the RGB tile
        denoised_tile = bm3d.bm3d(
            tile, sigma_psd=noise_level, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
        )
        return denoised_tile
    except Exception as e:
        logger.error(f"Error denoising tile {tile_index + 1}/{total_tiles}: {e}")
        raise DenoiseException(
            f"Error denoising tile {tile_index + 1}/{total_tiles}: {e}"
        )
