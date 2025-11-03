import os

import exiftool
import rawpy

from core.common.exceptions.equipment_exceptions import (
    EquipmentException,
    EquipmentFileNotFoundException,
    EquipmentFileNotRawException,
    EquipmentTransferException,
)
from core.utils.core_utils import get_nikon_lens_name
from locales.localization import _
from log.logger import logger


class Equipment:
    def __init__(self, image_path):
        """
        Manages the equipment used to capture the image.

        :param image_path: The path of the image
        """
        try:
            self.image_path = image_path

            # Check if the file is a RAW file
            if not self._is_tif_file():
                self._is_raw_file()
                self.color_temperature, self.tint = self.extract_white_balance_info(
                    image_path
                )

            # Extract EXIF metadata from the RAW file
            self.metadatas = self.extract_exif_from_raw(image_path)

            self.metadata = self.metadatas[0]
            # Not all cameras share the same attributes. I.E. Rencay does not use the following
            if "EXIF:Make" in self.metadata:
                self.camera_make = self.metadata["EXIF:Make"]
            if "EXIF:Model" in self.metadata:
                self.camera_model = self.metadata["EXIF:Model"]
            if "EXIF:LensMake" in self.metadata:
                self.lens_make = self.metadata.get("EXIF:LensMake")
            if "EXIF:LensModel" in self.metadata:
                self.lens_model = self.metadata.get("EXIF:LensModel")

            if "lens_make" not in self.metadata or "lens_model" not in self.metadata:
                self._try_to_extract_lens_info()

            if "EXIF:FocalLength" in self.metadata:
                self.focal_length = self.metadata["EXIF:FocalLength"]  # in millimeters
            if "EXIF:FNumber" in self.metadata:
                self.aperture = self.metadata["EXIF:FNumber"]
            if "EXIF:ExposureTime" in self.metadata:
                self.shutter_speed = self.metadata["EXIF:ExposureTime"]
            if "EXIF:ISO" in self.metadata:
                self.iso = self.metadata["EXIF:ISO"]


        except Exception as e:
            logger.error(_("Failed to initialize equipment: {}").format(e))
            raise EquipmentException(e)

    # Check if the file is a tif file
    def _is_tif_file(self):
        ext = os.path.splitext(self.image_path)[1]
        return ext in [".tif", ".tiff"]

    # Check if the file is a RAW file
    def _is_raw_file(self):
        """
        Checks if the file is a RAW file.

        :return: True if the file is a RAW file, raises an exception otherwise
        """
        if not os.path.exists(self.image_path):
            logger.error(
                _("The file does not exist at the path: {}").format(self.image_path)
            )
            raise EquipmentFileNotFoundException(
                _("The file does not exist at the path: {}").format(self.image_path)
            )
        try:
            with rawpy.imread(self.image_path):
                return True
        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError, OSError):
            logger.error(
                _("The file at path: {} is not a RAW file.").format(self.image_path)
            )
            raise EquipmentFileNotRawException(
                _("The file at path: {} is not a RAW file.").format(self.image_path)
            )

    def _try_to_extract_lens_info(self):
        """
        Tries to get the lens make and model from the Composite:LensID metadata for Nikon cameras.
        """
        lensID = self.metadata.get("Composite:LensID")
        if lensID:
            self.lens_model = get_nikon_lens_name(lensID)
            self.lens_make = "Nikon"

    @staticmethod
    # Extract EXIF metadata from a RAW file using pyexiftool library
    def extract_exif_from_raw(image_path):
        """
        Extracts EXIF metadata from a RAW file.

        :param image_path: The path of the RAW file

        :return: The extracted metadata
        """
        metadata = None

        # Use pyexiftool to extract EXIF metadata
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(image_path)

        if not metadata:
            raise Exception(_("No EXIF data found."))

        return metadata

    @staticmethod
    def extract_white_balance_info(image_path):
        """
        Extracts the white balance information from a RAW file.

        :param image_path: The path of the RAW file

        :return: The color temperature and tint
        """
        with rawpy.imread(image_path) as raw:
            # Access auto white balance (AWB) data
            color_temperature, tint = (
                raw.camera_whitebalance[0],
                raw.camera_whitebalance[1],
            )
            # Alternatively, for some camera models, it may be necessary
            # to directly access the raw data to obtain the color temperature
            # This depends on the specific implementation of LibRaw for that camera model

        return color_temperature, tint

    @staticmethod
    # Transfer EXIF metadata from a source image to a target image
    def transfer_exif(source_image, target_image):
        """
        Transfers EXIF metadata from a source image to a target image.

        :param source_image: The source image
        :param target_image: The target image
        """
        logger.debug(_("Transferring EXIF metadata..."))
        with exiftool.ExifTool() as et:
            # Build the command to copy all EXIF metadata
            command = [
                "-TagsFromFile",
                source_image,
                "-all:all",
                "-overwrite_original",
                target_image,
            ]
            try:
                et.execute(*command)
                logger.info(_("EXIF metadata transferred successfully."))
            except Exception as e:
                logger.error(_("Failed to transfer EXIF metadata: {}").format(e))
                raise EquipmentTransferException({e})
