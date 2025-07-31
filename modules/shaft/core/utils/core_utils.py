import json
import math
import os
from pathlib import Path
from typing import List

import colour
import lensfunpy
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000

from log.logger import logger
from utils.utils import read_config, find_project_root


def ciede2000(lab1, lab2, kL=1, kC=1, kH=1):
    """
    Calculate the CIEDE2000 color difference between two CIELAB colors.

    :param lab1: The first CIELAB color as a tuple (L*, a*, b*).
    :param lab2: The second CIELAB color as a tuple (L*, a*, b*).
    :param kL: The lightness weighting factor (default is 1).
    :param kC: The chroma weighting factor (default is 1).
    :param kH: The hue weighting factor (default is 1).

    :return: The CIEDE2000 color difference between the two colors.
    """
    # Unpack the input LAB values
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: Calculate C1 and C2 (Chroma values of the input colors)
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    # Compute the mean Chroma
    C_mean = (C1 + C2) / 2

    # **Added Part**: Calculate G
    # G is a scaling factor used to adjust the a components based on the mean Chroma.
    # This accounts for the non-linear behavior of chroma in the CIELAB color space.
    G = 0.5 * (1 - math.sqrt((C_mean ** 7) / (C_mean ** 7 + 25 ** 7)))

    # **Added Part**: Adjust a1 and a2 using G to get a1' and a2'
    # This adjustment corrects the a components to improve the perceptual uniformity.
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    # **Added Part**: Recalculate C1' and C2' using the adjusted a components
    # The adjusted chroma values are essential for the subsequent calculations.
    C1_prime = math.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = math.sqrt(a2_prime ** 2 + b2 ** 2)
    # Compute the mean adjusted Chroma
    C_prime_mean = (C1_prime + C2_prime) / 2

    # **Added Part**: Calculate h1' and h2' (Hue angles in degrees) using the adjusted a components
    # The hue angles are recalculated to reflect the adjustments made to the a components.
    h1_prime = (math.degrees(math.atan2(b1, a1_prime)) + 360) % 360
    h2_prime = (math.degrees(math.atan2(b2, a2_prime)) + 360) % 360

    # Step 2: Calculate delta L' (Difference in Lightness)
    delta_L_prime = L2 - L1

    # Step 3: Calculate delta C' (Difference in Chroma)
    delta_C_prime = C2_prime - C1_prime

    # **Added Part**: Calculate delta h' (Difference in Hue)
    # Adjust delta h' to ensure it falls within the correct range
    delta_h_prime = h2_prime - h1_prime
    if abs(delta_h_prime) > 180:
        if h2_prime <= h1_prime:
            delta_h_prime += 360
        else:
            delta_h_prime -= 360

    # **Added Part**: Calculate delta H' (Combined hue difference term)
    # This accounts for the difference in hue while considering the chroma of the colors.
    delta_H_prime = (
            2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))
    )

    # **Added Part**: Calculate the mean Lightness, Chroma, and Hue
    L_prime_mean = (L1 + L2) / 2
    C_prime_mean = (C1_prime + C2_prime) / 2

    # **Added Part**: Calculate the mean Hue angle h'_
    # Special conditions are applied when the hues are far apart
    if abs(h1_prime - h2_prime) > 180:
        if (h1_prime + h2_prime) < 360:
            h_prime_mean = (h1_prime + h2_prime + 360) / 2
        else:
            h_prime_mean = (h1_prime + h2_prime - 360) / 2
    else:
        h_prime_mean = (h1_prime + h2_prime) / 2

    # **Added Part**: Calculate T
    # T is a weighting function that adjusts the hue difference based on the mean hue angle
    T = (
            1
            - 0.17 * math.cos(math.radians(h_prime_mean - 30))
            + 0.24 * math.cos(math.radians(2 * h_prime_mean))
            + 0.32 * math.cos(math.radians(3 * h_prime_mean + 6))
            - 0.20 * math.cos(math.radians(4 * h_prime_mean - 63))
    )

    # **Added Part**: Calculate delta theta
    # delta theta is used in the rotation term R_T
    delta_theta = 30 * math.exp(-(((h_prime_mean - 275) / 25) ** 2))

    # **Added Part**: Calculate R_C (Chroma rotation term)
    # R_C adjusts the rotation term based on the mean adjusted chroma
    R_C = 2 * math.sqrt((C_prime_mean ** 7) / (C_prime_mean ** 7 + 25 ** 7))

    # **Added Part**: Calculate S_L, S_C, and S_H (Weighting functions for Lightness, Chroma, and Hue)
    # These functions normalize the differences based on the location in the color space
    S_L = 1 + (
            (0.015 * (L_prime_mean - 50) ** 2) / math.sqrt(20 + (L_prime_mean - 50) ** 2)
    )
    S_C = 1 + 0.045 * C_prime_mean
    S_H = 1 + 0.015 * C_prime_mean * T

    # **Added Part**: Calculate R_T (Rotation term)
    # R_T accounts for the interaction between chroma and hue differences
    R_T = -R_C * math.sin(2 * math.radians(delta_theta))

    # Step 4: Calculate the final Delta E00 value
    # Combines all the computed differences and weighting functions
    delta_E = math.sqrt(
        (delta_L_prime / (kL * S_L)) ** 2
        + (delta_C_prime / (kC * S_C)) ** 2
        + (delta_H_prime / (kH * S_H)) ** 2
        + R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    )

    return delta_E


def run_ciede2000_tests():
    """
    Run the CIEDE2000 color difference tests.
    """
    # Define the test pairs and their expected ΔE00 values
    test_data = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
        ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ]

    results = []
    # Run the CIEDE2000 function on each pair and compare with expected results
    for i, (color1, color2, expected) in enumerate(test_data, 1):
        delta_e = ciede2000(color1, color2)
        results.append((i, delta_e, expected, abs(delta_e - expected)))

    # Print the results and comparison
    for pair_index, delta_e, expected, diff in results:
        print(
            f"Test Pair {pair_index}: Computed DE00 = {delta_e:.4f}, Expected DE00 = {expected:.4f}, Difference = {diff:.4f}"
        )


def convert_ndarray_to_list(item):
    """
    Convert an ndarray to a list.

    :param item: The item to convert.

    :return: The item as a list.
    """
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, (list, tuple)):
        return [convert_ndarray_to_list(i) for i in item]
    return item


def extract_patch(image, position, size):
    """
    Extract a patch from an image using NumPy slicing.

    :param image: The image to extract the patch from.
    :param position: The position of the patch as a tuple (x, y).
    :param size: The size of the patch as a tuple (width, height).

    :return: The extracted patch.
    """
    x, y = position
    width, height = size

    # Ottieni le dimensioni dell'immagine
    img_height, img_width = image.shape[:2]

    # Compute beginnning and end of the patches
    x_begin = x - width // 2
    y_begin = y - height // 2
    x_end = x + width // 2
    y_end = y + height // 2

    # Gestisci i bordi dell'immagine
    if x_begin < 0:
        logger.warning(f"Patch x_begin-coordinate {x} is negative. Set to 0.")
        x_begin = 0
    if y_begin < 0:
        logger.warning(f"Patch y_begin-coordinate {x} is negative. Set to 0.")
        y_begin = 0
    if y < 0:
        logger.warning(f"Patch y-coordinate {y} è negativa. Impostato a 0.")
        y = 0
    if x_end > img_width:
        logger.warning(
            f"Patch x_end {x_end} greater than image width: {img_width}. Set to: {img_width}."
        )
        x_end = img_width
    if y_end > img_height:
        logger.warning(
            f"Patch y_end {y_end} greater than image height {img_height}. Set to: {img_height}."
        )
        y_end = img_height

    # Estrai la patch utilizzando slicing di NumPy
    if image.ndim == 2:
        patch = image[y_begin:y_end, x_begin:x_end]
    else:
        patch = image[y_begin:y_end, x_begin:x_end, :]


    return patch


def get_target_reference(campioni_colore, patch_number):
    """
    Get the target reference values for a specific patch number.

    :param campioni_colore: The target reference values.
    :param patch_number: The patch number.

    :return: The target reference values for the specified patch number.
    """
    return np.array(
        campioni_colore[str(patch_number)]
    )  # Get the target white color (normalized to 0-1 range)


def get_nikon_lens_name(lensID):
    """
    Get the Nikon lens name from the lens ID.

    :param lensID: The lens ID.

    :return: The Nikon lens name or None if the lens ID is not found.
    """
    config = read_config()
    nikon_tags = os.path.join(config.get("directories", "tags_path"), f"nikon_tags.json")

    with open(nikon_tags, "r") as f:
        data = json.load(f)

    normalized_lensID = str(lensID).strip().upper().replace(" ", "")

    for tag in data:
        normalized_tag_id = tag["id"].strip().upper().replace(" ", "")
        if normalized_tag_id == normalized_lensID:
            return tag["name"]

    return None


# Get the lensfun database adding the XML custom files
def get_lensfun_db():
    """
    Get the lensfun database.

    :return: The lensfun database.
    """
    config = read_config()
    dir_path = os.path.join(find_project_root(), config.get("directories", "lensfun_path"))
    db_files = [str(file) for file in dir_path.rglob("*.xml")]

    db = lensfunpy.Database(db_files)
    return db


def rgb_linear_to_xyz(rgb):
    """ Converte RGB lineare in XYZ usando la matrice sRGB -> XYZ """
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    return np.dot(M, rgb)


def xyz_to_lab(xyz, white_point=(0.95047, 1.00000, 1.08883)):
    """ Converte XYZ in CIE-Lab usando D65 come riferimento di bianco """

    def f(t):
        delta = 6 / 29
        return t ** (1 / 3) if t > delta ** 3 else (t / (3 * delta ** 2)) + (4 / 29)

    X, Y, Z = xyz / white_point
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return LabColor(L, a, b)


def delta_e_2000_colormath(rgb1, rgb2):
    """ Calcola ΔE 2000 tra due colori in RGB lineare """
    xyz1 = rgb_linear_to_xyz(np.array(rgb1))
    xyz2 = rgb_linear_to_xyz(np.array(rgb2))

    lab1 = xyz_to_lab(xyz1)
    lab2 = xyz_to_lab(xyz2)

    return delta_e_cie2000(lab1, lab2)


BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)
VON_KRIES = np.array(
    [
        [0.40024, 0.70760, -0.08081],
        [-0.22630, 1.16532, 0.04570],
        [0.00000, 0.00000, 0.91822],
    ]
)
SHARP = np.array(
    [[1.2694, -0.0988, -0.1706], [-0.8364, 1.8006, 0.0357], [0.0297, -0.0315, 1.0018]]
)
CAT2000 = np.array(
    [[0.7982, 0.3389, -0.1371], [-0.5918, 1.5512, 0.0406], [0.0008, 0.2390, 0.9753]]
)
CAT02 = np.array(
    [[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]]
)



def srgb_2_linear(srgb):
    return np.where(
        srgb <= 0.0404482362771082, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
    )


def linear_2_srgb(linear):
    linear = np.asarray(linear)
    linear = np.clip(linear, 0, None)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        (1 + 0.055) * np.power(linear, 1 / 2.4) - 0.055,
    )

def XYZ_2_srgb(xyz, illuminant="d65"):
    # converte l'illuminante
    if illuminant == "d50":
        xyz = colour.adaptation.chromatic_adaptation(
            xyz,
            colour.CCS_ILLUMINANTS["CIE_10_1964"]["D50"],
            colour.CCS_ILLUMINANTS["CIE_10_1964"]["D65"],
            method="Bradford"
        )
    rgb_linear = colour.XYZ_to_RGB(
        xyz,
        colour.CCS_ILLUMINANTS["CIE_10_1964"]["D65"],  # source white
        colour.CCS_ILLUMINANTS["CIE_10_1964"]["D65"],  # target white (sRGB)
        colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB
    )
    rgb_srgb = colour.cctf_encoding(rgb_linear, function='sRGB')
    return rgb_srgb

def xyz_2_hsv(xyz, illuminant="d65"):
    # 1. XYZ → RGB lineare (sRGB, D65)
    if illuminant == "d50":
        xyz = colour.adaptation.chromatic_adaptation(
            xyz,
            colour.CCS_ILLUMINANTS["CIE_10_1964"]["D50"],
            colour.CCS_ILLUMINANTS["CIE_10_1964"]["D65"],
            method="Bradford"
        )
    rgb_lin = colour.XYZ_to_RGB(
        xyz,
        colour.CCS_ILLUMINANTS['CIE_10_1964']['D65'],
        colour.CCS_ILLUMINANTS['CIE_10_1964']['D65'],
        colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
    )

    # 2. Lineare → sRGB
    rgb = colour.cctf_encoding(rgb_lin)

    # 3. RGB → HSV
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask = delta != 0

    r_mask = (cmax == r) & mask
    g_mask = (cmax == g) & mask
    b_mask = (cmax == b) & mask

    hue[r_mask] = ((g - b)[r_mask] / delta[r_mask]) % 6
    hue[g_mask] = ((b - r)[g_mask] / delta[g_mask]) + 2
    hue[b_mask] = ((r - g)[b_mask] / delta[b_mask]) + 4
    hue = (60 * hue) % 360

    sat = np.zeros_like(cmax)
    sat[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]

    val = cmax
    return np.stack([hue, sat, val], axis=-1)


def hsv_2_xyz(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c

    rgb = np.zeros_like(hsv)
    conds = [
        (h < 60),
        (h >= 60) & (h < 120),
        (h >= 120) & (h < 180),
        (h >= 180) & (h < 240),
        (h >= 240) & (h < 300),
        (h >= 300)
    ]
    rgb_vals = [
        [c, x, 0],
        [x, c, 0],
        [0, c, x],
        [0, x, c],
        [x, 0, c],
        [c, 0, x]
    ]

    for cond, (rc, gc, bc) in zip(conds, rgb_vals):
        rgb[..., 0] = np.where(cond, rc, rgb[..., 0])
        rgb[..., 1] = np.where(cond, gc, rgb[..., 1])
        rgb[..., 2] = np.where(cond, bc, rgb[..., 2])

    rgb += m[..., np.newaxis]

    a = 0.055
    threshold = 0.04045

    rgb_lin = np.empty_like(rgb)
    mask = rgb <= threshold
    rgb_lin[mask] = rgb[mask] / 12.92
    inv_mask = ~mask
    rgb_lin[inv_mask] = ((rgb[inv_mask] + a) / (1 + a)) ** 2.4

    # RGB lineare → XYZ
    xyz = colour.RGB_to_XYZ(
        rgb_lin,
        colour.CCS_ILLUMINANTS['CIE_10_1964']['D65'],
        colour.CCS_ILLUMINANTS['CIE_10_1964']['D65'],
        colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ
    )
    return xyz


# def rgb_lin_2_hsv(rgb):
#     r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
#
#     cmax = np.max(rgb, axis=-1)
#     cmin = np.min(rgb, axis=-1)
#     delta = cmax - cmin
#
#     # Hue
#     hue = np.zeros_like(cmax)
#     mask = delta != 0
#
#     r_mask = (cmax == r) & mask
#     g_mask = (cmax == g) & mask
#     b_mask = (cmax == b) & mask
#
#     hue[r_mask] = ((g - b)[r_mask] / delta[r_mask]) % 6
#     hue[g_mask] = ((b - r)[g_mask] / delta[g_mask]) + 2
#     hue[b_mask] = ((r - g)[b_mask] / delta[b_mask]) + 4
#
#     hue = hue * 60
#     hue = hue % 360  # optional: keep H in [0, 360]
#
#     # Saturation
#     sat = np.zeros_like(cmax)
#     sat[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
#
#     # Value
#     val = cmax
#
#     hsv = np.stack([hue, sat, val], axis=-1)
#     return hsv


def gamma_correction(value, gamma):
    """Generic gamma correction."""
    return np.power(value, 1 / gamma)


def inverse_gamma_correction(value, gamma):
    """Generic inverse gamma correction."""
    return np.power(value, gamma)


def adobe_rgb_to_linear(value):
    """Convert Adobe RGB to linear RGB."""
    return inverse_gamma_correction(value, 2.2)


def linear_to_adobe_rgb(value):
    """Convert linear RGB to Adobe RGB."""
    return gamma_correction(value, 2.2)


def prophoto_to_linear(value):
    """Convert ProPhoto RGB to linear RGB."""
    return np.where(value <= 0.031248, value / 16, value ** 1.8)

def prophoto_to_linearsrgb(img):
    M = np.array([
        [1.3459433, -0.2556075, -0.0511118],
        [-0.5445989, 1.5081673, 0.0205351],
        [0.0000000, 0.0000000, 1.2118128]
    ])

    # Convertire da ProPhoto RGB a sRGB
    img_srgb = np.tensordot(img, M, axes=([2], [1]))

    # Clamping (per evitare valori fuori gamma)
    img_srgb = np.clip(img_srgb, 0, 1)

    return srgb_2_linear(img_srgb)


def linear_rgb_2_hsv(image_linear_rgb):
    """Convert Linear RGB image to HSV"""
    # Convert from Linear RGB to sRGB
    image_srgb = colour.cctf_encoding(image_linear_rgb, function='sRGB')

    # Convert from sRGB to HSV
    image_hsv = colour.RGB_to_HSV(image_srgb)

    return image_hsv

def linear_to_prophoto(value):
    """Convert linear RGB to ProPhoto RGB."""
    return np.where(value <= 0.001953, value * 16, value ** (1 / 1.8))


def cie_rgb_to_xyz(rgb):
    """Convert CIE RGB to CIE XYZ"""
    M = np.array([[0.49, 0.31, 0.20],
                  [0.17697, 0.81240, 0.01063],
                  [0.00, 0.01, 0.99]])

    rgb = np.asarray(rgb)
    xyz = np.dot(rgb, M.T)

    return xyz


def xyz_to_cie_rgb(xyz):
    """Convert CIE XYZ to CIE RGB"""
    # Matrice inversa di trasformazione
    M_inv = np.array([[1.910, -1.112, 0.202],
                      [0.371, 0.629, 0.000],
                      [0.000, 0.014, 0.995]])

    xyz = np.asarray(xyz)
    rgb = np.dot(xyz, M_inv.T)

    return rgb

def xyY_to_XYZ_matrix(xyY_matrix):
    XYZ = np.zeros([len(xyY_matrix),3])
    for i in range(len(xyY_matrix)):
        x, y, Y = xyY_matrix[str(i+1)]
        if y == 0:
            XYZ[i] = [0, 0, 0]
        else:
            XYZ[i] = [(x * Y) / y, Y, ((1 - x - y) * Y) / y]
    return XYZ


def rgb_2_lab(rgb):
    srgb = sRGBColor(*rgb, is_upscaled=False)
    lab = convert_color(srgb, LabColor)
    return (lab.lab_l, lab.lab_a, lab.lab_b)


def linear_2_lab(linear):
    srgb = linear_2_srgb(linear)
    srgb = np.clip(srgb, 0, 1)
    return rgb_2_lab(srgb)


def get_measured_values_from_patches(patches):
    measured_values = np.array(
        [patch["rgb_values"] for patch_list in patches for patch in patch_list],
        dtype=np.float32,
    )
    return measured_values


def get_reference_values_as_array(reference_values):
    return np.asarray(
        [
            [float(f"{value[0]}"), float(f"{value[1]}"), float(f"{value[2]}")]
            for key, value in reference_values.items()
        ]
    )


def set_measured_values_in_patches(patches, measured_values):
    idx = 0
    for rgb_value in measured_values:
        patches[0][idx]["rgb_values"] = rgb_value
        idx += 1
    return patches


def get_target_reference_values():
    """
    Get the target reference values.

    :return: The target reference values.
    """
    config = read_config()
    target_patches_path = os.path.join(find_project_root(), config.get("directories", "target_colors"))

    with open(target_patches_path, "r", encoding="utf-8") as file:
        campioni_colore = json.load(file)["CONFIG"]["TARGET"]["referenceXYZValues"] #linearReferenceRGBValues


    return campioni_colore


def find_files_by_format(
        paths: str, file_format: str, process_subfolder: bool = False
) -> List[str]:
    """
    Search for all files with a specific extension in a directory, with an option to include subfolders.

    :param path: Path of the directory to analyze.
    :param file_format: File extension to search for.
    :param process_subfolder: If True, also search in subfolders.

    :return: List of paths of the found files.
    """
    if not os.path.exists(paths):
        raise ValueError(f"The specified path '{paths}' does not exist.")

    if not os.path.isdir(paths):
        raise ValueError(f"The specified path '{paths}' is not a directory.")

    matched_files = []

    for root, dirs, files in os.walk(paths):
        for file in files:
            if file.lower().endswith(f".{file_format.lower()}"):
                matched_files.append(os.path.join(root, file))

        if not process_subfolder:
            break  # Avoid processing subfolders

    return matched_files

def get_colorchecker_24_patch_name(index):
    names = [
        "Dark Skin",         # 0
        "Light Skin",        # 1
        "Blue Sky",          # 2
        "Foliage",           # 3
        "Blue Flower",       # 4
        "Bluish Green",      # 5
        "Orange",            # 6
        "Purplish Blue",     # 7
        "Moderate Red",      # 8
        "Purple",            # 9
        "Yellow Green",      # 10
        "Orange Yellow",     # 11
        "Blue",              # 12
        "Green",             # 13
        "Red",               # 14
        "Yellow",            # 15
        "Magenta",           # 16
        "Cyan",              # 17
        "White",             # 18
        "Neutral 8 (light gray)",  # 19
        "Neutral 6.5 (medium-light gray)", # 20
        "Neutral 5 (medium gray)",         # 21
        "Neutral 3.5 (dark gray)",         # 22
        "Black"              # 23
    ]
    if 0 <= index < len(names):
        return names[index]
    else:
        return "Invalid index"
