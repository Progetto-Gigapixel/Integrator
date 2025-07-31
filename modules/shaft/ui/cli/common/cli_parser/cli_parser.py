import argparse
import json

from config.config import Mode, OutputColorSpaces, OutputFormats
from locales.localization import _
from ui.cli.common.custom_help_formatter.custom_help_formatter import (
    CustomHelpFormatter,
)
from utils.utils import read_config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class CliParser:
    def __init__(self, argv):
        self.config = read_config()
        self.parser = self.create_cli_parser()
        self.args = self.parse_args(argv)
        self.manage_language_argument(argv)

    def create_cli_parser(self):
        parser = argparse.ArgumentParser(
            prog="main.py",
            description="COCOA - CLI Interface",
            formatter_class=CustomHelpFormatter,
        )

        # input argument
        parser.add_argument(
            "-i",
            "--input",
            type=str,
            required=True,
            help="Specify the path of the image or the directory containing the images to be processed.",
            metavar="Raw_Path",
        )

        # input argument
        parser.add_argument(
            "-w",
            "--white_field",
            type=str,
            required=False,
            help="Specify the path of the white field to pass to the flat fielding step",
            metavar="Raw_Path",
        )

        # mode argument
        parser.add_argument(
            "-m",
            "--mode",
            type=str,
            choices=[mode for mode in Mode],
            default=self.config.get("defaults", "mode"),
            help="Select the operation mode.",
            metavar="Mode",
        )

        # output format argument
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=[format for format in OutputFormats],
            default=self.config.get("defaults", "output_format"),
            help="Specify the output format for the processed images.",
            metavar="Output_Format",
        )

        # output argument
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            default=self.config.get("defaults", "output_path"),
            help="Specify the path where the processed images will be saved.",
            metavar="Output_Path",
        )

        # output color space argument
        parser.add_argument(
            "-c",
            "--color",
            type=str,
            choices=[color for color in OutputColorSpaces],
            default=self.config.get("defaults", "output_color_space"),
            help="Specify the color space for the output images.",
            metavar="Color_Space",
        )

        # fitting degree argument
        parser.add_argument(
            "-d",
            "--degree",
            type=int,
            choices=[1, 2, 3],
            default=self.config.get("defaults", "fitting_degree"),
            help="Specify the degree of the polynomial for the fitting.",
            metavar="Degree",
        )

        parser.add_argument(
            "-s",
            "--sharpen",
            action="store_true",
            default=self.config.getboolean("defaults", "rt_sharpening"),
            help="Specify if the image needs to be sharpened using the rawtherapee software.",
        )

        parser.add_argument(
            "-lb",
            "--light_balance",
            action="store_true",
            default=self.config.getboolean("defaults", "rt_light_balance"),
            help="Specify if the image needs to light balanced using the rawtherapee software.",
        )


        # Skip ccm argument
        parser.add_argument(
            "--skip-exposure",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_exposure"),
            help="Skip exposure correction.",
        )

        # Skip ccm argument
        parser.add_argument(
            "--skip-wb",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_wb"),
            help="Skip white balance correction.",
        )


        # Skip ccm argument
        parser.add_argument(
            "--skip-ccm",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_ccm"),
            help="Skip the CCM correction.",
        )

        # Skip finlayson argument
        parser.add_argument(
            "--skip-finlayson-ccm",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_finlayson_ccm"),
            help="Skip the finlayson CCM.",
        )

        # Skip ridge argument
        parser.add_argument(
            "--skip-ridge-ccm",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_ridge_ccm"),
            help="Skip the ridge CCM.",
        )

        # Skip ridge argument
        parser.add_argument(
            "--skip-wls",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_wls"),
            help="Skip the WLS.",
        )

        # Skip polynomial argument
        parser.add_argument(
            "--skip-poly",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_poly"),
            help="Skip the polynomial fitting.",
        )

        # Skip shaft argument
        parser.add_argument(
            "--skip-shaft",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_shaft"),
            help="Skip the shaft correction.",
        )

        # Skip shaft argument
        parser.add_argument(
            "--skip-rawtherapee",
            action="store_true",
            default=self.config.getboolean("defaults", "skip_rawtherapee"),
            help="Skip the rawtherapee correction.",
        )

        # Skip process subfolder argument
        parser.add_argument(
            "--process-subfolder",
            action="store_true",
            default=self.config.getboolean("defaults", "process_subfolders"),
            help="Process subfolders.",
        )

        # Overwrite files argument
        parser.add_argument(
            "--overwrite-files",
            action="store_true",
            default=self.config.getboolean("defaults", "overwrite_files"),
            help="Overwrite files.",
        )

        # Extension argument
        parser.add_argument(
            "--extension",
            type=str,
            default=self.config.get("defaults", "extension"),
            help="Specify the extension of the files to be processed.",
            metavar="Extension",
        )

        # Parameter path argument
        parser.add_argument(
            "--parameter-path",
            type=str,
            help="Specify the path for the parameters for the development mode.",
            metavar="Parameter_Path",
        )

        # Language argument
        parser.add_argument(
            "-l",
            "--language",
            choices=["it", "en"],
            type=str,
            default=self.config.get("defaults", "language"),
            help="Set the language for CLI messages",
            metavar="Language",
        )

        parser.add_argument(
            "-ct",
            "--crop_target",
            type=str2bool,
            nargs="?",
            const=True,
            default=self.config.getboolean("defaults", "process_cropped_image_only"),
            help="Process cropped target images in Analysis Mode (true/false)",
        )

        return parser

    def parse_args(self, argv):
        args = self.parser.parse_args(argv)

        # If the mode is set to DEVELOPMENT, the parameter_path argument is required
        if args.mode == Mode.DEVELOPMENT and not args.parameter_path:
            self.parser.error(
                "--parameter_path is required when --mode is set to 'DEVELOPMENT'."
            )

        return args

    def print_help(self):
        self.parser.print_help()

    def manage_language_argument(self, argv):
        lan = None
        # Check if the language argument is present
        for i, arg in enumerate(argv):
            if arg in ("-l", "--language") and i + 1 < len(argv):
                lan = argv[i + 1]
                _.set_language(lan)
                return
        _.set_language(self.config.get("defaults", "language"))

    def parse_plugin_params(self, plugin_arg):
        """
        Parses the plugin parameters from the command-line argument.

        Args:
        - plugin_arg (str): The argument string for the plugin parameters.

        Returns:
        - dict: A dictionary of parsed parameters.
        """
        try:
            # Attempt to parse as JSON
            return json.loads(plugin_arg)
        except json.JSONDecodeError:
            # Fallback to custom parsing (e.g., comma-separated)
            params = {}
            for item in plugin_arg.split(","):
                key, value = item.split(":")
                params[key.strip()] = self.cast_value(value.strip())
            return params

    def cast_value(self, value):
        """
        Casts a value to the appropriate type.

        Args:
        - value (str): The value to cast.

        Returns:
        - Any: The value casted to int, float, or str.
        """
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # Return as string if conversion fails
