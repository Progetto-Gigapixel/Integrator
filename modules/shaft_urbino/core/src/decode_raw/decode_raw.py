import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import exifread
import numpy as np
import rawpy
import colour
from config.config import RawFileTypes
from core.src.decode_raw.import_tif import import_tif_as_pseudo_raw
from core.common.exceptions.decode_raw_exceptions import (
    DecodeRawException,
    DecodeRawIOException,
    DecodeRawIpnutException,
)
from core.utils.core_utils import linear_2_srgb
from locales.localization import _
from log.logger import logger


class RawDecoder:
    def __init__(self, equipment, input_path, output_format, mode):
        """
        Initializes the RAW decoder with the specified parameters.

        :param equipment: The equipment used to capture the RAW image
        :param input_path: The path to the RAW file or directory
        :param output_format: The output format for the decoded image
        :param mode: The mode for decoding the RAW file
        """
        self.equipment = equipment
        self.input_path = input_path
        self.output_format = output_format
        self.mode = mode

    # Decode a RAW file or a directory of RAW files
    def decode_raw(self, mode):
        """
        Decodes the RAW file or directory of RAW files.

        :param mode: The mode for decoding the RAW file

        :return: The decoded image as a numpy array"""
        try:
            # In 'AM' mode, the input must be a single RAW file
            if mode == "AM" and not os.path.isfile(self.input_path):
                raise DecodeRawIpnutException(
                    _("In 'AM' mode, the input must be a single RAW file.")
                )

            logger.info(_("Decoding RAW file..."))

            result_img = None
            # Check if the input path is a file
            if os.path.isfile(self.input_path):
                result_img = self.decode_raw_image(self.input_path)
            # Check if the input path is a directory
            elif os.path.isdir(self.input_path):
                result_img = self.decode_raw_directory(self.input_path)
            else:
                raise DecodeRawIOException(
                    _(
                        "The specified path is neither a file nor a directory: {input_path}"
                    ).format(input_path=self.input_path)
                )

            logger.info(_("RAW file decoded successfully."))

            return result_img

        except Exception as e:
            logger.error(_("Failed to decode RAW file: {}").format(e))
            raise DecodeRawException(e)


    @staticmethod
    def _is_tif_file(raw_file):
        ext = os.path.splitext(raw_file)[1]
        return ext in [".tif", ".tiff"]

    # Decode a single RAW file
    def decode_raw_image(self, raw_file, resize_factor=1):
        """
        Decodes a single RAW file.

        :param raw_file: The path to the RAW file
        :param resize_factor: The factor to resize the image

        :return: The decoded image as a numpy array
        """

        #This is to work with pseudo raw file. They have to be tif, in prophoto colorspace
        if self._is_tif_file(raw_file):

            xyz = import_tif_as_pseudo_raw(raw_file)

        else:
            with rawpy.imread(str(raw_file)) as raw:

                # Process the RAW image into a 16-bit RGB array
                rgb_image = raw.postprocess(
                    gamma=(1, 1), output_bps=16, no_auto_bright=True,  use_camera_wb=True, user_flip=0
                )

                rgb_array = rgb_image.astype(np.float32) / 65535
                # rgb_array = rgb_array


                # Resize the image for testing purposes
                if resize_factor > 1:
                    new_width = rgb_array.shape[1] // resize_factor
                    new_height = rgb_array.shape[0] // resize_factor
                    rgb_array = cv2.resize(
                        rgb_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                    )


                cm1 = self.extract_colormatrix1(raw_file)
                if cm1 is not None:
                    shape = rgb_array.shape
                    flat = rgb_array.reshape(-1, 3)
                    xyz = flat @ cm1.T
                    xyz = xyz.reshape(shape)
                else:
                    xyz = colour.RGB_to_XYZ(
                        rgb_array,
                        colour.CCS_ILLUMINANTS['CIE_10_1964']['D65'],
                        colour.CCS_ILLUMINANTS['CIE_10_1964']['D65'],
                        colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ
                    )

        return xyz

    def extract_colormatrix1(self, raw_file):
        with open(raw_file, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        tag = tags.get('Image ColorMatrix1')
        if not tag:
            logger.info(_("ColorMatrix1 non found in metadata."))
            return

        # I valori sono in forma di lista di razionali
        values = [float(val.num) / float(val.den) for val in tag.values]
        matrix = np.array(values).reshape((3, 3))
        return matrix

    # Decode a directory of RAW files
    def decode_raw_directory(self, input_directory, output_format):
        """
        Decodes a directory of RAW files.

        :param input_directory: The directory containing the RAW files
        :param output_format: The output format for the decoded images
        """
        raw_files = [
            file
            for raw_type in RawFileTypes
            for file in Path(input_directory).rglob(raw_type.value)
        ]

        # Create a ThreadPoolExecutor to process the RAW files in parallel
        with ThreadPoolExecutor() as executor:
            future_to_raw = {
                executor.submit(self.decode_raw_image, raw_file): raw_file
                for raw_file in raw_files
            }
            for future in as_completed(future_to_raw):
                raw_file = future_to_raw[future]
                result = future.result()

