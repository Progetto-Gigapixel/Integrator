import cv2
import numpy as np


def extract_patch(image, position, size):
    """Extract a patch from the image at given position and size"""
    x, y = position
    width, height = size
    x = int(x - width / 2)
    y = int(y - height / 2)
    return image[y:y + height, x:x + width]


def create_synthetic_checker(patches, patch_size):
    """Create synthetic checker image by arranging patches in a 4x6 grid"""
    grid_h, grid_w = 4, 6
    canvas = np.zeros((grid_h * patch_size[1], grid_w * patch_size[0], 3), dtype=np.float64)
    for idx, patch in enumerate(patches):
        row = idx // grid_w
        col = idx % grid_w
        y = row * patch_size[1]
        x = col * patch_size[0]
        canvas[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
    return canvas


def extract_color_checker_image(image_array, checker_corners) -> tuple:
    """
    Estrae l'immagine del solo Color Checker da un'immagine più grande.
    :param image_array: np.ndarray, l'immagine decodificata (HxWxC) 
    :param checker_corners: list, struttura annidata contenente la posizione dei quadrati
    :return: tuple (np.ndarray, list), immagine del Color Checker ritagliata e la struttura dati aggiornata
    """
    patches = []
    patch_size = None

    # Extract individual patches
    for patch_data in checker_corners[0]:
        position = patch_data['position']
        size = patch_data['size']
        patch = extract_patch(image_array, position, size)
        patches.append(patch)
        if patch_size is None:
            patch_size = size

    synthetic_checker = create_synthetic_checker(patches, patch_size)
    grid_w = 6
    for idx, patch_data in enumerate(checker_corners[0]):
        row = idx // grid_w
        col = idx % grid_w
        # y = row * patch_size[1] + patch_size[1] // 2
        # x = col * patch_size[0] + patch_size[0] // 2
        y = row * patch_size[1]
        x = col * patch_size[0]
        patch_data['position'] = (x, y)

    return synthetic_checker, checker_corners
