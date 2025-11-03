from enum import Enum, auto


# Supported RAW file types
class RawFileTypes(Enum):
    """Enum for supported RAW file types."""

    RAW = "*.raw"
    NEF = "*.nef"
    CR2 = "*.cr2"
    RAF = "*.RAF"
    FFF = "*.fff"


class AnalysisSteps(str, Enum):
    """Enum for analysis steps."""

    KEY = "AnalysisSteps"
    INIT = "Init"
    DECODE_RAW = "DecodeRaw"
    GEOMETRIC_CORRECTOR = "GeometricCorrection"
    VIGNETTING_CORRECTOR = "VignettingCorrection"
    EXPOSURE = "exposure_correction"
    WHITE_BALANCE = "white_balance_correction"
    FLAT_FIELDING = "flat_fielding_correction"
    DENOISING = "denoising"
    CCM = "ccm"
    RIDGE_CCM = "ridge_ccm"
    FINLAYSON_CCM = "finlayson_ccm"
    WLS = "wls_correction"
    POLYNOMIAL_FITTING_CORRECTOR = "polynomial_correction"
    SHAFTLESS = "shaftless_correction"
    RAWTHERAPEE = "rawtherapee_correction"
    FINALIZE = "Finalize"


class ManualAnalysisSteps(Enum):
    """Enum for manual analysis steps."""

    GEOMETRIC_CORRECTOR = 0
    VIGNETTING_CORRECTOR = 1


class AppState(Enum):
    """Enum for the application state."""

    KEY = "AppState"
    CLI = auto()
    GUI = auto()


class OutputFormats(str, Enum):
    """Enum for supported output formats."""

    TIFF = "tif"
    JPG = "jpg"
    PNG = "png"


class OutputColorSpaces(str, Enum):
    """Enum for supported output color spaces."""

    SRGB = "srgb"
    DISPLAY_P3 = "display-p3"
    ADOBE_RGB = "adobe-rgb"
    PRO_PHOTO = "pro-photo"


class Mode(str, Enum):
    """Enum for the mode of operation."""

    ANALYSIS = "AM"
    DEVELOPMENT = "DM"


class SKIP_PARAMS(str, Enum):
    """Enum for the parameters to skip in the analysis."""
    EXPOSURE = "exposure"
    WHITE_BALANCE = "white_balance"
    DENOISING = "denoising"
    CCM = "ccm"
    FINLAYSON_CCM = "finalyson_ccm"
    RIDGE_CCM = "ridge_ccm"
    WLS = "wls"
    POLYNOMIAL_FITTING = "polynomial_fitting"
    SHAFT = "shaft"
    FLAT_FIELDING = "flat_fielding"
    RAWTHERAPEE = "rawtherapee"


class DEVELOPMENT_PARAMS(str, Enum):
    """Enum for the parameters to development phase."""

    PROCESS_SUBFOLDER = "process_subfolder"
    DO_NOT_OVERWRITE_FILES = "do_not_overwrite_files"
    EXTENSION_2_PROCESS = "extension_2_process"
    PARAMETER_PATH = "correction_data_path"
    OUTPUT_PATH = "output_path"
