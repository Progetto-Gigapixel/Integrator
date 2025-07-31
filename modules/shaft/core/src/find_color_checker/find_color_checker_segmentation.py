import math
import cv2
import numpy as np
from colour_checker_detection import detect_colour_checkers_segmentation
from colour_checker_detection.detection.common import (
    SETTINGS_DETECTION_COLORCHECKER_CLASSIC,
    DataDetectionColourChecker,
)

from core.utils.core_utils import convert_ndarray_to_list, extract_patch
from locales.localization import _
from log.logger import logger


def draw_white_points(image, points, radius=3):
    """
    Disegna punti bianchi su una copia dell'immagine.

    :param image: immagine di input (numpy array)
    :param points: lista di tuple (x, y)
    :param radius: raggio del punto (default 3)
    :return: nuova immagine con i punti disegnati
    """
    # Copia l'immagine per non modificarla
    output_image = image.copy()
    points = np.array(points)

    for idx, (x, y) in enumerate(points):
        center = (int(x), int(y))
        cv2.circle(output_image, (int(x), int(y)), radius, (1, 1, 1), thickness=-1)
        text = str(idx)
        cv2.putText(output_image, text, (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1), 1, cv2.LINE_AA)

    return output_image

def reorder_quadrilateral(points, angle = 0):
    """
    Riordina i punti di un quadrilatero nell'ordine corretto:
    Top-left, Top-right, Bottom-left, Bottom-right.

    :param points: Lista o array NumPy con i 4 punti [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    : angle: must be one of the following: 0, 90, -90, 180
    :return: Lista di punti riordinati [(TL), (TR), (BL), (BR)]
    """
    points = np.array(points, dtype=np.float32)  # Convertiamo in array NumPy se necessario

    # Calcoliamo la somma (x + y) per trovare il Top-Left (TL) e il Bottom-Right (BR)
    sum_coords = points.sum(axis=1)
    TL = points[np.argmin(sum_coords)]  # Punto con la somma più piccola → Top-Left
    BR = points[np.argmax(sum_coords)]  # Punto con la somma più grande → Bottom-Right

    # Calcoliamo la differenza (x - y) per trovare il Top-Right (TR) e il Bottom-Left (BL)
    diff_coords = np.diff(points, axis=1).flatten()
    TR = points[np.argmax(diff_coords)]  # Punto con la differenza più grande → Top-Right
    BL = points[np.argmin(diff_coords)]  # Punto con la differenza più piccola → Bottom-Left

    ret =[]
    if angle == 90:
        ret = np.array([BL, BR, TL, TR])
    elif angle == -90:
        ret = np.array([TR, TL, BR, BL])
    elif angle == 180:
        ret = np.array([BR, TR, BL, TL])
    else:
        ret = np.array([TL, BL, TR, BR])
    return ret



def compute_homography_matrix(original_pts, transformed_pts):
    """
    Calcola la matrice di omografia data la corrispondenza di punti tra l'immagine originale e quella trasformata.

    :param original_pts: Lista di 4 tuple (x, y) con i punti del quadrilatero nell'immagine originale.
    :param transformed_pts: Lista di 4 tuple (x, y) con i punti corrispondenti nel quadrilatero trasformato.
    :return: Matrice di omografia H (3x3).
    """
    original_pts = np.array(original_pts, dtype=np.float32)
    transformed_pts = np.array(transformed_pts, dtype=np.float32)

    H, _ = cv2.findHomography(transformed_pts, original_pts)

    return H

def get_rotation_angle(H):
    """
    Estrae l'angolo di rotazione dalla matrice di omografia H.

    :param H: Matrice di omografia (3x3)
    :return: Angolo di rotazione in gradi
    """
    # Estrai i coefficienti di rotazione
    h00, h10 = H[0, 0], H[1, 0]

    # Calcola l'angolo di rotazione in radianti
    theta_rad = math.atan2(h10, h00)

    # Converti in gradi
    theta_deg = math.degrees(theta_rad)

    return theta_deg

def is_consistent_order(original_pts, extracted_pts):
    """
    Controlla se l'ordine dei punti nelle due liste è coerente.
    :return: True se l'ordine è corretto, False se potrebbe causare una rotazione indesiderata.
    """
    # Calcoliamo i centroidi per entrambe le configurazioni
    centroid_orig = np.mean(original_pts, axis=0)
    centroid_ext = np.mean(extracted_pts, axis=0)

    # Cmpute centroiods angles
    angles_orig = np.arctan2(original_pts[:,1] - centroid_orig[1], original_pts[:,0] - centroid_orig[0])
    angles_ext = np.arctan2(extracted_pts[:,1] - centroid_ext[1], extracted_pts[:,0] - centroid_ext[0])

    # Confrontiamo gli ordini degli angoli: devono avere lo stesso ordine
    return np.all(np.argsort(angles_orig) == np.argsort(angles_ext))

def get_original_coordinates(H, points):
    """
    Trova le coordinate originali di una lista di punti utilizzando la matrice di omografia.

    :param H: Matrice di omografia (3x3).
    :param points: Lista di punti [(x1, y1), (x2, y2), ...] nel quadrilatero estratto.
    :return: Lista di punti [(x1', y1'), (x2', y2'), ...] nell'immagine originale.
    """
    # Convertiamo i punti in array NumPy e li trasformiamo in coordinate omogenee
    points = np.array(points, dtype=np.float32)
    points_homogeneous = np.column_stack((points, np.ones(len(points))))  # Aggiunge la terza coordinata 1

    # Applichiamo la trasformazione
    transformed_points = np.dot(H, points_homogeneous.T).T  # Moltiplicazione matrice e trasposta

    # Normalizziamo i risultati per ottenere coordinate cartesiane
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2].reshape(-1, 1)
    transformed_points = np.round(transformed_points).astype(int)

    return transformed_points.tolist()  # Ritorniamo una lista di tuple

def is_cropping_quadrilateral_horizontal(quadrilateral):
    """
    Determina se un quadrilatero ha un aspetto verticale o orizzontale.

    :param points: Lista o array di 4 punti [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :return: Stringa "verticale" o "orizzontale"
    """
    quadrilateral = np.array(quadrilateral, dtype=np.float32)  # Converti in array NumPy

    # Find bounding box limits
    min_x, max_x = np.min(quadrilateral[:, 0]), np.max(quadrilateral[:, 0])
    min_y, max_y = np.min(quadrilateral[:, 1]), np.max(quadrilateral[:, 1])

    # Compute width and height
    width = max_x - min_x
    height = max_y - min_y

    # Establish if the cropped target is horizontal or vertical
    if width > height:
        return True
    else:
        return False

def recover_relative_patch_coordinates(patch_masks):
    relative_patch_coordinates = []
    for i, mask in enumerate(patch_masks):
        if len(mask) == 0:
            continue

        col = np.mean([mask[0], mask[1]])
        row = np.mean([mask[2], mask[3]])
        relative_patch_coordinates.append([row.astype(int), col.astype(int)])
    return relative_patch_coordinates


def retrieve_rgb_from_image(image_array, patch_centers, patch_size=[20, 20]):
    """
    Extract the colors of the swatches from the image.

    :param image_array: The image as a numpy array.
    """
    rgb = []
    for center in patch_centers:
        # Extract the pacth image
        patch_image = extract_patch(image_array, center, patch_size)

        # Calculate the mean colour of the swatch
        ndarray_colour = np.mean(patch_image, axis=(0, 1))
        rgb.append(convert_ndarray_to_list(ndarray_colour))
    return np.array(rgb)

def find_white_index(rgb_array):
    # Calcola la media per ciascuna riga (axis=1)
    row_means = rgb_array.mean(axis=1)
    # Trova l'indice della riga con media maggiore
    max_index = np.argmax(row_means)
    return max_index


# def create_image_mosaic(img, positions, size, grid=(4, 6)):
#     righe, colonne = grid
#     assert len(positions) == righe * colonne, "Numero positions incompatible with grid elements"
#
#     w_block, h_block = size
#     channels = img.shape[2] if img.ndim == 3 else 1
#
#     h_mosaic = righe * h_block
#     w_mosaic = colonne * w_block
#     image_mosaic = np.zeros((h_mosaic, w_mosaic, channels), dtype=img.dtype)
#     centroids = []
#     for idx, position in enumerate(positions):
#         patch = extract_patch(img, position, size)
#
#         r = idx // colonne
#         c = idx % colonne
#         y_dst = r * h_block
#         x_dst = c * w_block
#
#         # Nota: se patch è più piccola (per via di crop ai bordi), va adattato il posizionamento
#         ph, pw = patch.shape[:2]
#         image_mosaic[y_dst:y_dst+ph, x_dst:x_dst+pw] = patch
#         center_patch = [ x_dst + pw // 2, y_dst + ph // 2]
#         centroids.append(center_patch)
#
#     return image_mosaic, centroids

def create_image_mosaic(img, positions, size, grid=(4, 6)):
    righe, colonne = grid
    assert len(positions) == righe * colonne, "Numero positions incompatibile con la griglia"

    w_block, h_block = size

    h_mosaic = righe * h_block
    w_mosaic = colonne * w_block

    if img.ndim == 2:
        image_mosaic = np.zeros((h_mosaic, w_mosaic), dtype=img.dtype)
    elif img.ndim == 3:
        channels = img.shape[2]
        image_mosaic = np.zeros((h_mosaic, w_mosaic, channels), dtype=img.dtype)
    else:
        raise ValueError("img deve essere 2D (grayscale) o 3D (RGB/RGBA)")

    centroids = []

    for idx, position in enumerate(positions):
        patch = extract_patch(img, position, size)

        r = idx // colonne
        c = idx % colonne
        y_dst = r * h_block
        x_dst = c * w_block

        ph, pw = patch.shape[:2]
        image_mosaic[y_dst:y_dst+ph, x_dst:x_dst+pw] = patch
        center_patch = [x_dst + pw // 2, y_dst + ph // 2]
        centroids.append(center_patch)

    return image_mosaic, centroids



# def recompute_resampled_coordinates(white_index, crop_rectangle, extracted_pts, raw_patch_coordinates, ordered_quadrilateral, rescale_factor=1.0):
#     if white_index != 18:
#         if white_index == 0:
#             ordered_quadrilateral = reorder_quadrilateral(ordered_quadrilateral, 90)
#         elif white_index == 22:
#             ordered_quadrilateral = reorder_quadrilateral(ordered_quadrilateral, -90)
#         elif white_index == 5:
#             ordered_quadrilateral = reorder_quadrilateral(ordered_quadrilateral, 180)
#
#         H = compute_homography_matrix(ordered_quadrilateral, extracted_pts)
#         resampled_coordinates = get_original_coordinates(H, raw_patch_coordinates)
#         return resampled_coordinates

class FindColorCheckerSegmentation:
    def __init__(self, process_cropped_image_only=True):
        """
        Find the colour checker in the image using segmentation.
        """
        self.color_checker_data = []
        self.settings = SETTINGS_DETECTION_COLORCHECKER_CLASSIC
        self.process_cropped_image_only = process_cropped_image_only
        self.cropped_image = None

    def find(self, image_array):
        """
        Find the colour checker in the image and save the data to a JSON file.

        :param image_array: The image as a numpy array.

        :return: The detected colour checker data.
        """

        if not self.color_checker_data:
            detected_color_checkers = detect_colour_checkers_segmentation(image_array, additional_data=True)
            if not detected_color_checkers:
                logger.warning("No colour checker detected.")
                return self.color_checker_data
            # This f... colour checker library
            # 1) Extracts the color checher (cc) position
            # 2) Crops the image according to the cc position
            # 2) If the cc is vertical, it rotates it back to a horizontal layout
            # 3) Returns the masks of the patches in the cropped, possibly rotated, image
            # The followings, are variable which I need to compute the absolute position of the patches starting from
            # the relative ones.
            color_checker_data = detected_color_checkers[0] #I should get just one cc
            # rename this attribute for sake of clarity. This is the cropped image

            relative_patch_coordinates = recover_relative_patch_coordinates(color_checker_data.swatch_masks)
            #self.cropped_image = color_checker_data.colour_checker
            mosaic_image, mosaic_patch_coordinates = create_image_mosaic(color_checker_data.colour_checker, relative_patch_coordinates, [25, 25])  # color_checker_data.colour_checker
            original_height, original_width = image_array.shape[:2]
            target_height, target_width = color_checker_data.colour_checker.shape[:2]
            rescale_factor = original_width / target_width
            resampled_image = cv2.resize(image_array, None, fx=1 / rescale_factor, fy=1 / rescale_factor)

            original_image_borders = np.array([(0, 0),(target_width - 1, 0),(0, target_height - 1), (target_width - 1, target_height - 1)])
            ordered_quadrilateral = reorder_quadrilateral(color_checker_data.quadrilateral)

            H = compute_homography_matrix(ordered_quadrilateral, original_image_borders)
            resampled_absolute_coordinates = get_original_coordinates(H, relative_patch_coordinates)
            rgb = retrieve_rgb_from_image(resampled_image, resampled_absolute_coordinates)

            white_index = find_white_index(rgb)
            # White index must be at position 18 (idx starts at 0)
            if white_index != 18:
                rotation_values = [180, -90, 90]
                for rotation in rotation_values:
                    H = compute_homography_matrix(reorder_quadrilateral(ordered_quadrilateral, rotation), original_image_borders)
                    resampled_absolute_coordinates = get_original_coordinates(H, relative_patch_coordinates)
                    rgb = retrieve_rgb_from_image(resampled_image, resampled_absolute_coordinates)
                    white_index = find_white_index(rgb)
                    if white_index == 18:
                        break
                    else:
                        if rotation == 90:
                            logger.error(
                                f"White index is not at position 18 after 4 rotations.")
                            raise Exception(
                                "White index is not at position 18 after 4 rotations.")

                    # This is for debugging purposes only. Uncomment to see the position of the patches
                    #c = draw_white_points(resampled_image, resampled_absolute_coordinates)

            absolute_patch_coordinates = np.round(np.multiply(resampled_absolute_coordinates, rescale_factor)).astype(int)

            self.cropped_image = mosaic_image
            relative_patch_coordinates = mosaic_patch_coordinates
            # Prepare the data
            self._prepare_data(color_checker_data, absolute_patch_coordinates, relative_patch_coordinates, rgb)

        else:
            self._extract_colors_from_image(image_array)
            # self.visualize_patches(image_array, self.checkers_data, save_path='visualized_patches.png')

        #self.cropped_image = self.color_checker_data.colour_checker

        return {
            "color_checker_data": self.color_checker_data,
            "cropped_image": self.cropped_image
        }

    # def _check_image_size(self, image_array, checker):
    #     sy, sx, sc = checker.colour_checker.shape
    #     oy, ox, oc = image_array.shape
    #     max_image_size = np.max([oy,ox])
    #     max_colorchecker_dim = np.max([sy,sx])
    #
    #     if max_image_size > max_colorchecker_dim:
    #         scale_factor = max_colorchecker_dim / float(max_image_size)
    #     else:
    #         scale_factor = 1
    #     return scale_factor

    # def _rotate_quadrilateral(
    #         self, quadrilateral: np.ndarray, direction: str = "counter-clockwise"
    # ) -> np.ndarray:
    #     """
    #     Rotate a quadrilateral by 90 degrees in the specified direction.
    #
    #     :param quadrilateral: The quadrilateral to rotate.
    #     :param direction: The direction to rotate the quadrilateral.
    #
    #     :return: The rotated quadrilateral.
    #     """
    #     if direction == "clockwise":
    #         # Rotate 90 degrees clockwise
    #         rotated = np.roll(quadrilateral, 1, axis=0)
    #     elif direction == "counter-clockwise":
    #         # Rotate 90 degrees counter-clockwise
    #         rotated = np.roll(quadrilateral, -1, axis=0)
    #     else:
    #         raise ValueError("Direction must be 'clockwise' or 'counter-clockwise'")
    #     return rotated

    # def _plot_detected_patches(
    #         self,
    #         image: np.ndarray,
    #         mapped_quadrilaterals: List[np.ndarray],
    #         mapped_swatch_centers: List[List[np.ndarray]],
    #         save_path: str = None,
    # ):
    #     """
    #     Display the image with detected color checker patches overlaid.
    #
    #     :param image: The image as a numpy array.
    #     :param mapped_quadrilaterals: The mapped quadrilaterals in the original image coordinates.
    #     :param mapped_swatch_centers: The mapped swatch centers in the original image coordinates.
    #     :param save_path: The path to save the image with overlaid patches.
    #
    #     :return: The image with overlaid patches.
    #     """
    #     plt.figure(figsize=(12, 8))
    #     plt.imshow(image)
    #     ax = plt.gca()
    #
    #     for idx, quadrilateral in enumerate(mapped_quadrilaterals):
    #         logger.debug(
    #             f"Plotting Checker {idx + 1} quadrilateral coordinates:\n{quadrilateral}\n"
    #         )
    #         # Draw the quadrilateral of the color checker
    #         polygon = patches.Polygon(
    #             quadrilateral, linewidth=2, edgecolor="r", facecolor="none"
    #         )
    #         ax.add_patch(polygon)
    #         # Add a label at the centroid of the quadrilateral
    #         centroid = np.mean(quadrilateral, axis=0)
    #         ax.text(
    #             centroid[0],
    #             centroid[1],
    #             f"Checker {idx + 1}",
    #             color="yellow",
    #             fontsize=12,
    #         )
    #
    #         # Draw the swatches
    #         swatch_centers = mapped_swatch_centers[idx]
    #         for swatch_idx, center in enumerate(swatch_centers):
    #             # Define the size for the square (e.g., 20x20 pixels)
    #             square_size = 20
    #             half_size = square_size / 2
    #             # Define the square corners
    #             square = np.array(
    #                 [
    #                     [center[0] - half_size, center[1] - half_size],
    #                     [center[0] + half_size, center[1] - half_size],
    #                     [center[0] + half_size, center[1] + half_size],
    #                     [center[0] - half_size, center[1] + half_size],
    #                 ]
    #             )
    #             # Draw the square
    #             square_polygon = patches.Polygon(
    #                 square, linewidth=1, edgecolor="g", facecolor="none"
    #             )
    #             ax.add_patch(square_polygon)
    #             # Add a label (optional)
    #             ax.text(
    #                 center[0],
    #                 center[1],
    #                 f"Swatch {swatch_idx + 1}",
    #                 color="blue",
    #                 fontsize=8,
    #             )
    #             # Log the swatch center coordinates for debugging
    #             logger.debug(f"Checker {idx + 1}, Swatch {swatch_idx + 1} center: {center}")
    #
    #     plt.title("Detected Color Checker Patches and Swatches")
    #     plt.axis("off")
    #
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches="tight")
    #         logger.info(f"Image with detected patches saved to {save_path}")
    #
    #     plt.show()

    # def _map_quadrilaterals_and_swatches_to_original(
    #         self, image_array, detected_checkers
    # ):
    #     """
    #     Map detected quadrilaterals and swatches from the segmented image to the original image coordinates.
    #
    #     :param image_array: The original image as a numpy array.
    #     :param detected_checkers: The detected color checker data.
    #
    #     :return: The mapped quadrilaterals and swatch centers in the original image coordinates.
    #     """
    #     original_height, original_width = image_array.shape[:2]
    #     working_width = original_width  # self.settings["working_width"]
    #     aspect_ratio = 1  # self.settings["aspect_ratio"]
    #     working_height = original_height  # int(working_width / aspect_ratio)
    #
    #     # Check if the image was rotated during segmentation
    #     rotated = False
    #     if original_width < original_height:
    #         rotated = True
    #         scaling_factor_width = original_height / working_width
    #         scaling_factor_height = original_width / working_height
    #     else:
    #         scaling_factor_width = original_width / working_width
    #         scaling_factor_height = original_height / working_height
    #
    #     logger.debug(
    #         f"Original image dimensions: width={original_width}, height={original_height}"
    #     )
    #     logger.debug(f"Image rotated for segmentation: {rotated}")
    #     logger.debug(
    #         f"Scaling factors: scale_x={scaling_factor_width}, scale_y={scaling_factor_height}"
    #     )
    #
    #     self.scaling_factor_width = scaling_factor_width
    #     self.scaling_factor_height = scaling_factor_height
    #
    #     mapped_quadrilaterals = []
    #     mapped_swatch_centers = []
    #
    #     for checker in detected_checkers:
    #         quadrilateral = checker.quadrilateral  # Shape (4, 2)
    #
    #         # If rotated, rotate the points back to original orientation
    #         if rotated:
    #             quadrilateral = self._rotate_quadrilateral(
    #                 quadrilateral, "counter-clockwise"
    #             )
    #
    #         # Define the reference rectangle used in segmentation
    #         rectangle = np.array(
    #             [
    #                 [working_width, 0],
    #                 [working_width, working_height],
    #                 [0, working_height],
    #                 [0, 0],
    #             ],
    #             dtype=np.float32,
    #         )
    #
    #         quadrilateral = quadrilateral.astype(np.float32)
    #         inverse_transform = cv2.getPerspectiveTransform(rectangle, quadrilateral)
    #
    #         # Calculate the centers of the swatches
    #         swatch_centers = []
    #         for swatch_mask in checker.swatch_masks:
    #             y0, y1, x0, x1 = swatch_mask
    #             center_warped = np.array(
    #                 [[(x0 + x1) / 2], [(y0 + y1) / 2]], dtype=np.float32
    #             ).reshape(
    #                 1, 1, 2
    #             )  # Reshape for perspectiveTransform
    #             # Apply the inverse transformation
    #             center_original = cv2.perspectiveTransform(
    #                 center_warped, inverse_transform
    #             )
    #             center_original = center_original[0][0]  # Extract the coordinate
    #             swatch_centers.append(center_original)
    #
    #         # Apply the scaling factors
    #         mapped_quadrilateral = quadrilateral * np.array(
    #             [scaling_factor_height, scaling_factor_width]
    #         )
    #         mapped_quadrilaterals.append(mapped_quadrilateral)
    #
    #         # Apply the scaling factors to the swatch centers
    #         swatch_centers = np.array(swatch_centers) * np.array(
    #             [scaling_factor_width, scaling_factor_height]
    #         )
    #         mapped_swatch_centers.append(swatch_centers)
    #
    #     return mapped_quadrilaterals, mapped_swatch_centers

    def _prepare_data(self, detected_checkers, patch_positions, relative_patch_position, patch_rgbs):
        """
        Prepare the checker data for JSON serialization.

        :param detected_checkers: The detected colour checkers.
        :param mapped_quadrilaterals: The mapped quadrilaterals in the original image coordinates.
        :param mapped_swatch_centers: The mapped swatch centers in the original image coordinates.

        :return: The prepared checker data.
        """

        if not detected_checkers:
            raise ValueError(_("No colour checker detected."))
        self.color_checker_data = []
        width, height = 20, 20

        checker_info = []
        for swatch_idx, (rgb, absolute_position, relative_patch_position) in enumerate(zip(patch_rgbs, patch_positions, relative_patch_position)):
            patch_info = {
                "rgb_values": convert_ndarray_to_list(rgb),
                "absolute_position": [absolute_position[0], absolute_position[1]],
                "cropped_image_position": [relative_patch_position[0], relative_patch_position[1]],
                "size": [width, height],
            }
            checker_info.append(patch_info)
        self.color_checker_data.append(checker_info)


    def _extract_colors_from_image(self, image_array):
        """
        Extract the colors of the swatches from the image.

        :param image_array: The image as a numpy array.
        """
        for checker in self.color_checker_data:
            for swatch in checker:
                # Extract the swatch image
                position = swatch["cropped_image_position"] if self.process_cropped_image_only else swatch["absolute_position"]
                size = swatch["size"]
                swatch_image = extract_patch(image_array, position, size)

                # Calculate the mean colour of the swatch
                ndarray_colour = np.mean(swatch_image, axis=(0, 1))
                swatch["rgb_values"] = convert_ndarray_to_list(ndarray_colour)

    # def visualize_patches(
    #         self, image: np.ndarray, checkers_data: List[List[dict]], save_path: str = None
    # ):
    #     """
    #     Display the color checker patches on the original image.
    #
    #     :param image: The original image as a numpy array.
    #     :param checkers_data: The color checker data.
    #     :param save_path: The path to save the image with overlaid patches.
    #     """
    #     plt.figure(figsize=(12, 8))
    #     plt.imshow(image)
    #     ax = plt.gca()
    #
    #     for checker_idx, checker in enumerate(checkers_data):
    #         for swatch_idx, swatch in enumerate(checker):
    #             x, y = swatch["position"]
    #             width, height = swatch["size"]
    #             rect = patches.Rectangle(
    #                 (x, y),
    #                 width,
    #                 height,
    #                 linewidth=2,
    #                 edgecolor="blue",
    #                 facecolor="none",
    #                 label=f"Swatch {swatch_idx + 1}" if checker_idx == 0 else "",
    #             )
    #             ax.add_patch(rect)
    #             # Add a label to the patch
    #             ax.text(
    #                 x + width / 2,
    #                 y + height / 2,
    #                 f"Swatch {swatch_idx + 1}",
    #                 color="white",
    #                 fontsize=8,
    #                 ha="center",
    #                 va="center",
    #             )
    #
    #     plt.title("Swatch Positions on Image")
    #     plt.axis("off")
    #
    #     # Handle duplicate labels in the legend
    #     handles, labels = ax.get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys())
    #
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches="tight")
    #         logger.info(f"Image with patch positions saved to {save_path}")
    #
    #     # plt.show()
    def reset_color_checker_data(self):
        self.color_checker_data = []

    def set_synthetic_checker_data(self, checker_data):
        self.color_checker_data = checker_data