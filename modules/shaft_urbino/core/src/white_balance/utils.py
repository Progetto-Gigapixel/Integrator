import numpy as np

from core.utils.core_utils import BRADFORD, CAT02, CAT2000, SHARP, VON_KRIES
from utils.utils import read_config


def normalize_xyz(xyz):
    """
    Normalize xyz with 'y' so that 'y' represents luminance

    :param xyz: The XYZ values

    :return: The normalized XYZ values
    """
    return xyz / xyz[1]


def get_cat_matrix(cat_type="BRADFORD"):
    """
    Get the chromatic adaptation matrix for the specified CAT type.

    :param cat_type: The CAT type

    :return: The CAT matrix
    """
    if cat_type == "BRADFORD":
        return BRADFORD
    elif cat_type == "VON_KRIES":
        return VON_KRIES
    elif cat_type == "SHARP":
        return SHARP
    elif cat_type == "CAT2000":
        return CAT2000
    else:
        return CAT02


def xyz_to_lms(xyz, M):
    """
    Convert XYZ to LMS.

    :param xyz: The XYZ values
    :param M: The CAT matrix

    :return: The LMS values
    """
    return xyz @ M.T


def get_gain(lms_src, lms_dst):
    """
    Calculate the gain for chromatic adaptation.

    :param lms_src: The source LMS values
    :param lms_dst: The destination LMS values

    :return: The gain
    """
    return lms_dst / lms_src


def transform_lms(M, gain):
    """
    Transform the LMS values.

    :param M: The CAT matrix
    :param gain: The gain

    :return: The transformed LMS values
    """
    return np.linalg.inv(M) @ np.diag(gain) @ M
