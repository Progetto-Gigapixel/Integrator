import json
import os

import numpy as np
from PIL import Image, ImageCms

from config.config import AnalysisSteps, Mode, OutputColorSpaces
from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.utils.core_utils import linear_to_adobe_rgb, linear_to_prophoto, XYZ_2_srgb
from locales.localization import _
from log.logger import logger, memory_handler
from utils.utils import read_config, convert_numpy_to_list, find_project_root


# Class to organize the results
class ResultOrganizer:
    def __init__(
        self,
        input_directory,
        output_format,
        equipment,
        mode: Mode,
        do_not_overwrite_files=False,
        output_directory=None,
        output_color_space="sRGB",
        plugin_directory=None,
    ):
        """
        Organizes the results of the analysis.

        :param input_directory: The input directory of the image
        :param output_format: The output format for the resulting image
        :param equipment: The equipment used to capture the image
        :param output_directory: The output directory for the results
        :param output_color_space: The output color space for the resulting image
        """
        self.input_directory = input_directory
        self.output_format = output_format
        self.equipment = equipment
        self.output_directory = output_directory
        self.output_color_space = output_color_space
        self.mode = mode
        self.do_not_overwrite_files = do_not_overwrite_files

        config = read_config()

        # Read the settings for the directories
        self.parent_dir_name = config.get("directories", "parent_dir_name")
        self.dir_img_name = config.get("directories", "dir_img_name")
        self.dir_params_name = config.get("directories", "dir_params_name")
        self.dir_report_name = config.get("directories", "dir_report_name")
        self.dir_log_name = config.get("directories", "dir_log_name")
        self.icc_profiles_path = os.path.abspath(config.get("directories", "icc_profiles_path"))
        self.log_file_name = config.get("directories", "log_file_name") + ".log"

        # get imagefilename (solo in 'AM')
        image_filename = ""
        if os.path.splitext(input_directory)[1] != "":
            image_filename = os.path.splitext(os.path.split(input_directory)[1])[0]
        self.params_file_name = config.get("directories", "params_file_name") + ".json"
        partial_params_path = os.path.split(self.params_file_name)
        self.params_file_name = os.path.join(
            partial_params_path[0], image_filename + partial_params_path[1]
        )

        # Path of the main folder 'Analysis'            NO####(in AM Mode, Output_directory in DM Mode
        self.parent_dir_path = os.path.join(
            (self.output_directory if self.output_directory is not None else os.path.dirname(self.input_directory))#,
        )

        self.params_file_path = os.path.join(self.parent_dir_path, self.dir_params_name, self.params_file_name)
        self.flat_fielding_correction_matrix = os.path.join(self.parent_dir_path, self.dir_params_name, "flat_fielding_matrix.mat")

        self.icc_profiles = {
            OutputColorSpaces.SRGB: os.path.join(
                self.icc_profiles_path, "sRGB2014.icc"
            ),
            OutputColorSpaces.DISPLAY_P3: os.path.join(
                self.icc_profiles_path, "AppleDisplayP3.icc"
            ),
            OutputColorSpaces.ADOBE_RGB: os.path.join(
                self.icc_profiles_path, "AdobeRGB1998.icc"
            ),
            OutputColorSpaces.PRO_PHOTO: os.path.join(
                self.icc_profiles_path, "ProPhoto.icmc"
            ),
        }

    def prepare_directories(self):
        """
        Prepares the directories for saving the results.
        """
        directories = [
            self.dir_img_name if self.mode == Mode.ANALYSIS else "./",
            self.dir_log_name,
        ]

        # Use params and report folder in AM only
        if self.mode == Mode.ANALYSIS:
            directories.append(self.dir_params_name)
            directories.append(self.dir_report_name)

        # Create the directory structure if they do not exist
        for directory in directories:
            os.makedirs(os.path.join(self.parent_dir_path, directory), exist_ok=True)

        # check if directories are created
        for directory in directories:
            if not os.path.exists(os.path.join(self.parent_dir_path, directory)):
                raise Exception(
                    _("Failed to create the directory: {}").format(directory)
                )

    def save_image(self, image_array, step: AnalysisSteps = None, format=None):
        """
        Saves the resulting image to the output directory.

        :param image_array: The resulting image as a numpy array
        :param step: The analysis step

        :return: The path of the saved image
        """
        logger.debug(_("Saving the resulting image..."))
        # Extract the base file name without extension
        base_name = os.path.splitext(os.path.basename(self.input_directory))[0]

        if format is None:
            format = self.output_format
        # Save the image
        self.output_file = os.path.join(self.parent_dir_path,
                                        self.dir_img_name if self.mode == Mode.ANALYSIS else "./",
                                        base_name + "." + format)
        image_converted = self.convert_color_space(image_array, self.output_color_space)
        icc_profile = self.icc_profiles[self.output_color_space]
        self._save_image(image_converted, self.output_file, icc_profile)

        logger.debug(_("Resulting image saved successfully."))

        return self.output_file

    def _save_image(self, img, path, icc_profile_path, format="TIFF"):
        """
        Saves the image to the specified path.

        :param img: The image as a numpy array
        :param path: The path to save the image
        :param format: The format of the image
        """

        if self.do_not_overwrite_files and os.path.exists(path):
            logger.info(_("File already exists: {}. Not overwriting.").format(path))
            return

        with open(icc_profile_path, 'rb') as f:
            icc_profile = f.read()

        scaling_factor = 255
        img = np.round(np.clip(img, 0.0, 1.0) * scaling_factor).astype(np.uint8)
        img_pil = Image.fromarray(img, "RGB")
        img_pil.save(
            path, format=format, compression="tiff_lzw", icc_profile=icc_profile
        )
        self.equipment.transfer_exif(self.input_directory, path)

    # Save the log file
    def save_logs(self):
        """
        Saves the log file to the output directory.

        :return: The path of the saved log file
        """
        log_contents = memory_handler.getvalue()
        log_file_path = os.path.join(
            self.parent_dir_path, self.dir_log_name, self.log_file_name
        )

        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log_contents)

        # Clear the memory handler
        memory_handler.truncate(0)
        memory_handler.seek(0)

        return log_file_path

    def get_params_output_path(self):
        return os.path.dirname(self.params_file_path)

    def save_params(self, correction_data: ColorCorrectionData):
        """
        Saves the color correction parameters to the output directory.

        :param correction_data: The color correction data object

        :return: The path of the saved parameters file
        """
        # Do not save in dev mode
        if self.mode == Mode.DEVELOPMENT:
            return None

        # Retrieve JSON data
        data = correction_data.to_dict()
        data = convert_numpy_to_list(data)

        # Write the JSON data to the file
        with open(self.params_file_path, "w", encoding="utf-8") as params_file:
            json.dump(data, params_file, indent=4)

        return self.params_file_path

    def _analyze_image_statistics(self, image_array):
        """
        Analyzes the statistics of the image.

        :param image_array: The image as a numpy array
        """
        logger.debug("Image dimensions: {}".format(image_array.shape))
        logger.debug("Data type of image array: {}".format(image_array.dtype))
        logger.debug("Max pixel value: {}".format(image_array.max()))
        logger.debug("Min pixel value: {}".format(image_array.min()))
        logger.debug("Mean pixel value: {}".format(image_array.mean()))

        # Check for NaN and infinite values
        if np.isnan(image_array).any():
            logger.warning("Image contains NaN values")
        if np.isinf(image_array).any():
            logger.warning("Image contains infinite values")

    @staticmethod
    def convert_color_space(image, output_color_space):
        """
        Converts the image to the specified color space.

        :param image: The image as a numpy array in sRGB color space
        :param output_color_space: The output color space

        :return: The image as a numpy array in the specified color space
        """
        # image_color_space = "CIE RGB"
        # if output_color_space == OutputColorSpaces.SRGB:
        #     return colour.RGB_to_RGB(linear_2_srgb(image), image_color_space, "sRGB")
        # elif output_color_space == OutputColorSpaces.DISPLAY_P3:
        #     return colour.RGB_to_RGB(linear_2_srgb(image), image_color_space, "Display P3")
        # elif output_color_space == OutputColorSpaces.ADOBE_RGB:
        #     return colour.RGB_to_RGB(linear_to_adobe_rgb(image), image_color_space, "Adobe RGB (1998)")
        # elif output_color_space == OutputColorSpaces.PRO_PHOTO:
        #     return colour.RGB_to_RGB(linear_to_prophoto(image), image_color_space, "ProPhoto RGB")
        # else:
        #     return image
        if output_color_space == OutputColorSpaces.SRGB:
            image = XYZ_2_srgb(image) #linear_2_srgb(image)
        elif output_color_space == OutputColorSpaces.DISPLAY_P3:
            image = XYZ_2_srgb(image) #linear_2_srgb(image)
        elif output_color_space == OutputColorSpaces.ADOBE_RGB:
            image = linear_to_adobe_rgb(image)
        elif output_color_space == OutputColorSpaces.PRO_PHOTO:
            image = linear_to_prophoto(image)

        return image

