import logging
import re

from PIL import Image, ImageCms
import numpy as np
import os

from core.common.exceptions.not_a_prophoto_profile import ProPhotoICCProfileIsMissing
from core.utils.core_utils import srgb_2_linear, prophoto_to_linearsrgb
from log.logger import logger
from utils.utils import read_config


def check_if_prophoto(pil_image):
    """Check if an image has a ProPhoto RGB ICC profile using ImageCms."""
    try:
        # Try to get the ICC profile
        icc = pil_image.info["icc_profile"]
        if not re.search("ProPhoto".encode(), icc):
            raise ProPhotoICCProfileIsMissing("The processed input image does not have a Pro Photo ICC profile")

    except Exception as e:
        raise ProPhotoICCProfileIsMissing("The processed input image does not have a Pro Photo ICC profile")

def import_tif_as_pseudo_raw_OLD(pseudo_raw_image_name):
    im_array = Image.open(pseudo_raw_image_name)
    # Only tif files with Pro PhotoProfiles allowed
    # Per ora bypasso perché le mie immagini di flat field non sono pro photo
    #check_if_prophoto(im_array)
    profileOut = ImageCms.createProfile("sRGB")
    config = read_config()
    icc_profiles_path = config.get("directories", "icc_profiles_path")
    profileIn = ImageCms.getOpenProfile(os.path.join(icc_profiles_path, "ProPhoto.icm"))

    im_array = ImageCms.profileToProfile(
        im_array,
        profileIn,
        profileOut,
        outputMode="RGB"
    )
    mode_to_bit_depth = {"1": 1, "L": 8, "P": 8, "RGB": 255, "RGBA": 1024, "CMYK": 65535, "YCbCr": 24, "LAB": 24,
                         "HSV": 24, "I": 65535, "F": 65535}
    bpp = mode_to_bit_depth[im_array.mode]

    im_array = np.float32(im_array) / bpp
    im_array = srgb_2_linear(im_array)

    logging.info("File imported. Processing...")
    return im_array

def import_tif_as_pseudo_raw(pseudo_raw_image_name):

    img = Image.open(pseudo_raw_image_name)
    mode_to_bit_depth = {"1": 1, "L": 8, "P": 8, "RGB": 255, "RGBA": 1024, "CMYK": 65535, "YCbCr": 24, "LAB": 24,
                         "HSV": 24, "I": 65535, "F": 65535}
    bpp = mode_to_bit_depth[img.mode]
    img = np.array(img).astype(np.float32) / bpp

    # 2. Linea di decodifica OETF (gamma): ProPhoto RGB ha gamma ≈ 1.8
    def gamma_decode(img, gamma=1.8):
        return np.power(img, gamma)

    img_lin = gamma_decode(img)

    # 3. Matrice di trasformazione da ProPhoto RGB (lineare) a XYZ D50
    # (definita dallo standard ICC)
    M_prophoto_to_xyz_d50 = np.array([
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0000000, 0.0000000, 0.8252100]
    ])

    # 4. Applica la trasformazione per ciascun pixel
    xyz_img = np.tensordot(img_lin, M_prophoto_to_xyz_d50.T, axes=1)
    return xyz_img









