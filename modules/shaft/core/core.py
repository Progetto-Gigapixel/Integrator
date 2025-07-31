import inspect
import json
import os
import time
import xmlrpc.client

import numpy as np

from config.config import (
    DEVELOPMENT_PARAMS,
    SKIP_PARAMS,
    AnalysisSteps,
    AppState,
    ManualAnalysisSteps,
    Mode,
)
from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.common.equipment.equipment import Equipment
from core.common.exceptions.find_color_checker_exceptions import (
    AutoFindColorCheckerException,
)
from core.common.organizers.lensfun_organizer import LensfunOrganizer
from core.common.organizers.result_organizer import ResultOrganizer
from core.common.patch_comparer.patch_comparer import PatchComparer
from core.src.color_correction_matrix.color_correction_matrix import (
    ColorCorrectionMatrix,
)
from core.src.decode_raw.decode_raw import RawDecoder
from core.src.denoising.denoising import DenoisingCorrector
from core.src.exposure.exposure import ExposureCorrector
from core.src.find_color_checker.find_color_checker_segmentation import (
    FindColorCheckerSegmentation,
)
from core.src.flat_fielding.flat_fielding import FlatFieldingCorrector
from core.src.geometric_correction.geometric_correction import GeometricCorrector
from core.src.polynomial_fitting_corrector.polynomial_fitting_corrector import (
    PolynomialFittingCorrector,
)
from core.src.raw_therapee_correction.raw_therapee_processor import RawTherapeeProcessor
from core.src.shaftless.shaftless_corrector import (
    ShaftLessCorrector,
)

from core.src.startup_checker.startup_checker import StartupChecker
from core.src.vignetting_correction.vignetting_correction import VignettingCorrector
from core.src.white_balance.white_balance import WhiteBalance
from core.store.store import appStore
from core.utils.core_utils import find_files_by_format, get_target_reference_values
from locales.localization import _
from log.logger import logger
from plugins.plugin_model import ClosedPlugin
from utils.utils import get_config, spacer

# Colormath uses asscalar, but it is deprecated
setattr(np, "asscalar", lambda a: a.item())


class Core:
    parent_ref = None
    result_organizer = None

    def __init__(
        self,
        input_path,
        output_format,
        white_field_path=None,
        output_path=None,
        skip_params=None,
        output_color_space=None,
        fitting_degree=None,
        raw_therapee_sharpen=None,
        raw_therapee_light_balance=None,
        development_params=None,
        process_cropped_image_only = True,

    ):
        self.result_image = None
        self.original_image = None
        self.cropped_image = None
        self.process_cropped_image_only = process_cropped_image_only
        self.best_de00 = float("inf")
        self.input_path = input_path
        self.white_field_path = white_field_path
        self.output_format = output_format
        self.output_color_space = output_color_space
        self.fitting_degree = fitting_degree
        self.raw_therapee_light_balance = raw_therapee_light_balance
        self.raw_therapee_sharpen = raw_therapee_sharpen
        self.saved_image = None # Output path of the saved image # TODO: calcolare il percorso di output
        # all'inizio di tutto, così se abbiamo il check "Do not overwrite..." verifichiamo prima se l'immagine esiste
        # già e passiamo al ciclo successivo

        # Development phase data
        self.process_subfolder = (
            development_params.get(DEVELOPMENT_PARAMS.PROCESS_SUBFOLDER, False)
            if development_params
            else False
        )
        self.do_not_overwrite_files = (
            development_params.get(DEVELOPMENT_PARAMS.DO_NOT_OVERWRITE_FILES, False)
            if development_params
            else False
        )
        self.extension_2_process = (
            development_params.get(DEVELOPMENT_PARAMS.EXTENSION_2_PROCESS, None)
            if development_params
            else None
        )
        self.parameter_file = (
            development_params.get(DEVELOPMENT_PARAMS.PARAMETER_PATH, None)
            if development_params
            else None
        )
        if output_path is not None:
            self.output_path = output_path

        #  XML-RPC server client connection
        server_host = get_config("settings", "server_host")
        server_port = get_config("settings", "server_port")
        self.server_proxy = xmlrpc.client.ServerProxy(
            f"http://{server_host}:{server_port}", allow_none=True
        )

        # Patches
        self.measured_patches = np.zeros((24, 3))

        # Reference values
        self.reference_values = get_target_reference_values()

        # Properties for corrections
        self.result = None

        # Properties for manual corrections
        self.calibration = None

        self.isGui = appStore.get(AppState.KEY) == AppState.GUI

        # Properties for skipping the corrections
        # Per ora
        self.skip_denoising = True
        self.skip_shaftless_corrector = True
        self.skip_geometric_correction = True
        self.skip_vignetting_correction = True

        if (development_params is None and
                (raw_therapee_light_balance is None or raw_therapee_light_balance is False) and
                (raw_therapee_sharpen is None or raw_therapee_sharpen is False)):
            self.skip_rawtherapee = True
        ################

        if skip_params:
            self.skip_exposure = skip_params.get(SKIP_PARAMS.EXPOSURE, False)
            self.skip_white_balance = skip_params.get(SKIP_PARAMS.WHITE_BALANCE, False)
            self.skip_flat_fielding = skip_params.get(SKIP_PARAMS.FLAT_FIELDING, False)
            self.skip_ccm = skip_params.get(SKIP_PARAMS.CCM, False)
            self.skip_ridge_ccm = skip_params.get(SKIP_PARAMS.RIDGE_CCM, False)
            self.skip_finlayson_ccm = skip_params.get(SKIP_PARAMS.FINLAYSON_CCM, False)
            self.skip_wls = skip_params.get(SKIP_PARAMS.WLS, False)
            self.skip_polynomial_fitting_corrector = skip_params.get(SKIP_PARAMS.POLYNOMIAL_FITTING, False)
            self.skip_rawtherapee = skip_params.get(SKIP_PARAMS.RAWTHERAPEE, False)


    def save_comparison(self, mode):
        if mode==Mode.ANALYSIS:
            self.comparer.save_comparison_results()

    def notify_server(self, step_name, status, result=None):
        """Notifies the server of the event and passes relevant data."""
        self.server_proxy.set_step_result(step_name, status, result)

    def handle_manual_corrections(self):
        current_step = appStore.get(AnalysisSteps.KEY)

        # Handle geometric correction
        if current_step == ManualAnalysisSteps.GEOMETRIC_CORRECTOR:
            if self.skip_geometric_correction:
                appStore.set(AnalysisSteps.KEY, AnalysisSteps.EXPOSURE)
            else:
                self.create_lensfun_xml()
                appStore.set(AnalysisSteps.KEY, AnalysisSteps.GEOMETRIC_CORRECTOR)

        # Handle vignetting correction
        if current_step == ManualAnalysisSteps.VIGNETTING_CORRECTOR:
            if self.skip_vignetting_correction:
                appStore.set(AnalysisSteps.KEY, AnalysisSteps.EXPOSURE)
            else:
                self.create_lensfun_xml()
                appStore.set(AnalysisSteps.KEY, AnalysisSteps.VIGNETTING_CORRECTOR)

    def analysis_mode(self, mode=Mode.ANALYSIS):
        logger.info(_("Running Analysis Mode..."))
        # Initialize the correction data
        correction_data = ColorCorrectionData()

        # Handle manual corrections
        self.handle_manual_corrections()

        # Run the analysis mode
        self.run(mode, correction_data)

        logger.info(_("Analysis Mode completed successfully."))

    def development_mode(self, mode=Mode.DEVELOPMENT):
        logger.info(_("Running Development Mode - image:" + self.input_path))

        correction_data = None
        self.skip_geometric_correction = True
        self.skip_vignetting_correction = True

        # Read the correction data
        with open(self.parameter_file, "r") as file:
            correction_data = ColorCorrectionData().from_dict(json.load(file))

        # Spostato in cli.py
        # # Find the images to process
        # images = find_files_by_format(
        #     self.input_path, self.extension_2_process, self.process_subfolder
        # )

        # for image in self.input_path:
        appStore.set(AnalysisSteps.KEY, AnalysisSteps.INIT)
        #self.input_path =  self.input_path

        # Run the development mode
        self.run(mode, correction_data)

        logger.info(_("Development Mode completed successfully for image: " + self.input_path))

    def run(self, mode, correction_data):
        steps = [
            (AnalysisSteps.INIT,                            self.init_step),
            (AnalysisSteps.DECODE_RAW,                      self.decode_raw_step),
            (AnalysisSteps.FLAT_FIELDING,                   self.flat_fielding_step),
            (AnalysisSteps.GEOMETRIC_CORRECTOR,             self.geometric_correction_step),
            (AnalysisSteps.VIGNETTING_CORRECTOR,            self.vignetting_correction_step),
            (AnalysisSteps.EXPOSURE,                        self.exposure_correction_step),
            (AnalysisSteps.WHITE_BALANCE,                   self.white_balance_step),
            (AnalysisSteps.DENOISING,                       self.denoising_step),
            # (AnalysisSteps.CCM, self.ccm_correction_step),
            (AnalysisSteps.RIDGE_CCM,                       self.ridge_ccm_correction_step),
            (AnalysisSteps.FINLAYSON_CCM,                   self.finlayson_step),
            (AnalysisSteps.WLS,                             self.weighted_least_square_step),
            (AnalysisSteps.POLYNOMIAL_FITTING_CORRECTOR,    self.polynomial_fitting_step),
            (AnalysisSteps.SHAFTLESS,                       self.shaftless_step),
            (None,                                          self.finalize_step),
            (AnalysisSteps.RAWTHERAPEE,                     self.raw_therapee_step),
        ]

        for i, (step, func) in enumerate(steps):
            if appStore.get(AnalysisSteps.KEY) == step:
                func(correction_data, step, mode)
                if i + 1 < len(steps):
                    next_step = steps[i + 1][0]
                    appStore.set(AnalysisSteps.KEY, next_step)
                    
    def create_lensfun_xml(self):
        lensfunOrganizer = LensfunOrganizer(
            self.equipment,
            self.calibration.camera,
            self.calibration.lens,
            self.calibration.distortion,
            self.calibration.vignetting,
        )
        lensfunOrganizer.create_xml()

    def init_step(self, correction_data, step, mode):
        self.notify_server("initialization", "before")

        # Initialize the equipment
        self.equipment = Equipment(self.input_path)

        # Initial check
        if not self.skip_geometric_correction:
            startup_checker = StartupChecker(self.equipment)
            startup_checker.start_check()

        # Prepare the directories for saving the results
        self.result_organizer = ResultOrganizer(
            self.input_path,
            self.output_format,
            self.equipment,
            mode,
            self.do_not_overwrite_files,
            self.output_path,
            self.output_color_space,
        )
        self.result_organizer.prepare_directories()

        if mode == Mode.ANALYSIS and self.isGui:
            # Update the parameter file path in analysis mode
            self.parent_ref.update_parameter_file(
                self.result_organizer.params_file_path
            )
            # self.parent_ref.update_analysis_output_file(
            #     os.path.split(self.result_organizer.parent_dir_path)[0]
            # )

        # Properties for corrections
        image_filename = os.path.splitext(os.path.split(self.input_path)[1])[0]
        result_file = os.path.join(
            self.result_organizer.parent_dir_path,
            self.result_organizer.dir_report_name,
            image_filename + "_comparison_results.json",
        )

        self.comparer = PatchComparer(
            result_file, self.reference_values, self.input_path, self.process_cropped_image_only
        )

        self.color_checker_finder = FindColorCheckerSegmentation(self.process_cropped_image_only)

        self.notify_server("initialization", "after")

    def decode_raw_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        self.notify_server("decode", "before")
        decoder = RawDecoder(
            self.equipment, self.input_path, self.output_format, mode="AM"
        )
        self.result_image = decoder.decode_raw(mode="AM")
        # self.result_image = np.rot90(self.result_image, 2)
        self.compare_result(step, correction_data, self.result_image, mode)
        self.original_image = self.result_image
        if mode == Mode.ANALYSIS and self.process_cropped_image_only:
            self.result_image = self.cropped_image

        del decoder
        self.notify_server("decode", "after", None)

    def geometric_correction_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_geometric_correction:
            return

        self.notify_server("geometric_correction", "before")
        geometric_corrector = GeometricCorrector(
            correction_data, self.result_image, self.equipment, self.output_format
        )
        current_image = geometric_corrector.run()

        self.compare_result(step, correction_data, current_image, mode)

        del geometric_corrector
        self.notify_server("geometric_correction", "after", None)

    def vignetting_correction_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_vignetting_correction:
            return

        self.notify_server("vignetting_correction", "before")
        vignetting_corrector = VignettingCorrector(
            correction_data, self.result_image, self.equipment, self.output_format
        )
        current_image = vignetting_corrector.run()
        self.compare_result(step, correction_data, current_image, mode)

        del vignetting_corrector
        self.notify_server("vignetting_correction", "after", None)

    def exposure_correction_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_exposure or (
            mode == Mode.DEVELOPMENT and correction_data.exposure_correction is None
        ):
            return

        self.notify_server("exposure_correction", "before")
        exposure = ExposureCorrector(
            correction_data=correction_data,
            reference_values=self.reference_values,
            measured_patches=self.measured_patches,
            image=self.result_image,
            mode=mode,
        )

        current_image = exposure.run()
        self.compare_result(step, correction_data, current_image, mode)

        del exposure
        self.notify_server("exposure_correction", "after", None)

    def white_balance_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_white_balance or (
            mode == Mode.DEVELOPMENT
            and correction_data.white_balance_correction is None
        ):
            return

        self.notify_server("white_balance", "before", "result")
        white_balance = WhiteBalance(
            correction_data,
            self.reference_values,
            self.equipment,
            self.result_image,
            self.measured_patches,
            mode,
        )
        current_image = white_balance.run()
        self.compare_result(step, correction_data, current_image, mode)

        del white_balance

        self.notify_server("white_balance", "after", None)

    def flat_fielding_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if (self.skip_flat_fielding or
            self.white_field_path is None or (
                mode == Mode.DEVELOPMENT
                and correction_data.flat_fielding_correction is None
            )):
            return

        self.notify_server("flat_fielding", "before")
        flat_fielding = FlatFieldingCorrector(
            correction_data,
            self.original_image,
            self.cropped_image,
            self.white_field_path,
            self.color_checker_finder,
            self.process_cropped_image_only,
            self.equipment,
            mode,
            2,
            self.result_organizer.flat_fielding_correction_matrix
        )
        current_image = flat_fielding.run()
        self.compare_result(step, correction_data, current_image, mode)

        del flat_fielding
        self.notify_server("flat_fielding", "after", None)

    def denoising_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_denoising or (
            mode == Mode.DEVELOPMENT and correction_data.denoising is None
        ):
            return

        self.notify_server("denoising", "before")
        denoising = DenoisingCorrector(
            correction_data, self.result_image, self.measured_patches, mode, self.process_cropped_image_only
        )
        current_image = denoising.run()
        self.compare_result(step, correction_data, current_image, mode)

        del denoising
        self.notify_server("denoising", "after", None)

    def ccm_correction_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_ccm or (
            mode == Mode.DEVELOPMENT and correction_data.ccm is None
        ):
            return

        self.notify_server("ridge_ccm", "before")

        ccm = ColorCorrectionMatrix(
            correction_data,
            self.measured_patches,
            self.reference_values,
            self.result_image,
            mode,
        )
        current_image = ccm.run_ccm()

        self.compare_result(step, correction_data, current_image, mode)

        del ccm
        self.notify_server("ccm", "after", None)

    def ridge_ccm_correction_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_ridge_ccm or (
            mode == Mode.DEVELOPMENT and correction_data.ridge_ccm is None
        ):
            return

        self.notify_server("ridge_ccm", "before")

        ccm = ColorCorrectionMatrix(
            correction_data,
            self.measured_patches,
            self.reference_values,
            self.result_image,
            mode,
        )

        current_image = ccm.run_ridge_ccm()
        self.compare_result(step, correction_data, current_image, mode)

        del ccm
        self.notify_server("ridge_ccm", "after", None)

    def finlayson_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_finlayson_ccm or (
            mode == Mode.DEVELOPMENT and correction_data.finlayson_ccm is None
        ):
            return

        self.notify_server("finlayson_ccm", "before")
        ccm = ColorCorrectionMatrix(
            correction_data,
            self.measured_patches,
            self.reference_values,
            self.result_image,
            mode,
        )
        current_image = ccm.run_finlayson_ccm()
        self.compare_result(step, correction_data, current_image, mode)

        del ccm
        self.notify_server("finlayson_ccm", "after", None)


    def weighted_least_square_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        if self.skip_wls or (
            mode == Mode.DEVELOPMENT and correction_data.wls_correction is None
        ):
            return

        self.notify_server("weighted_least_square", "before")

        wls = PolynomialFittingCorrector(
            correction_data,
            self.result_image,
            self.reference_values,
            self.measured_patches,
            mode,
            self.fitting_degree,
        )

        current_image = wls.run_wls()
        self.compare_result(step, correction_data, current_image, mode)

        del wls
        self.notify_server("weighted_least_square", "after", None)

    def polynomial_fitting_step(
        self, correction_data: ColorCorrectionData,
        step: AnalysisSteps, mode: Mode
    ):
        if self.skip_polynomial_fitting_corrector or (
            mode == Mode.DEVELOPMENT and correction_data.polynomial_correction is None
        ):
            return

        self.notify_server("polynomial-fitting", "before")

        polynomial_fitting = PolynomialFittingCorrector(
            correction_data,
            self.result_image,
            self.reference_values,
            self.measured_patches,
            mode,
            self.fitting_degree,
        )
        # start_time = time.time()
        current_image = polynomial_fitting.run_poly()
        # logger.info(f"Time taken: {time.time() - start_time} seconds")
        self.compare_result(step, correction_data, current_image, mode)

        del polynomial_fitting
        self.notify_server("polynomial-fitting", "after", None)

    def shaftless_step(self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode):
        if self.skip_shaftless_corrector or (
                mode == Mode.DEVELOPMENT and correction_data.shaftless_correction is None
        ):
            return

        self.notify_server("shaftless_step", "before")

        shaftless = ShaftLessCorrector(
            correction_data,
            self.result_image,
            self.reference_values,
            self.measured_patches,
            mode,
            self.comparer
        )
        # start_time = time.time()
        current_image = shaftless.run_shaftless()
        # logger.info(f"Time taken: {time.time() - start_time} seconds")
        self.compare_result(step, correction_data, current_image, mode)

        del shaftless
        self.notify_server("shaftless", "after", None)

    def raw_therapee_step(self, correction_data, step, mode): # Signature must be the same as the other steps, even if in this phase corraction_data and others aren't used
        # TODO: Riscrivere: fare arrivare solo true o false a light_balance e a sharpen
        if self.skip_rawtherapee or (
                mode == Mode.ANALYSIS and self.raw_therapee_sharpen is None and self.raw_therapee_light_balance is None) or (
                (mode == Mode.ANALYSIS and self.raw_therapee_sharpen is False and self.raw_therapee_light_balance is False) or (
                mode == Mode.DEVELOPMENT and correction_data.rawtherapee_correction is None)
        ):
            return
        else:
            if mode == Mode.DEVELOPMENT and correction_data.rawtherapee_correction is not None:
                self.raw_therapee_light_balance = correction_data.get_rawtherapee_correction().raw_therapee_light_balance
                self.raw_therapee_sharpen = correction_data.get_rawtherapee_correction().raw_therapee_sharpen

        rt = RawTherapeeProcessor()
        if self.raw_therapee_sharpen is True and (self.raw_therapee_light_balance is None or self.raw_therapee_light_balance is False):
            action = rt.Action.SHARPEN_ONLY
        elif self.raw_therapee_light_balance is True and (self.raw_therapee_sharpen is None or self.raw_therapee_sharpen is False):
            action = rt.Action.LIGHTS_BALANCE_ONLY
        elif self.raw_therapee_light_balance is True and self.raw_therapee_sharpen is True:
            action = rt.Action.PROCESS_EVERYTHING
        else:
            return

        if self.saved_image is not None:
            rt.process_image(self.saved_image, action)




    def plugin_execution_step(
        self, correction_data: ColorCorrectionData, step: str, mode: str
    ):
        """
        Esegue tutti i plugin disponibili in ordine per lo step corrente.

        :param correction_data: Dizionario contenente i dati per la correzione.
        :param step: Nome dello step corrente.
        :param mode: Modalità di esecuzione.
        """
        # Ottieni la lista dei plugin disponibili
        plugins: dict = self.server_proxy.list_plugins()

        params = {
            "param1": "test1",
            "param2": "test2",
            "corrected_image": self.result_image,
            "correction_data": correction_data,
            "further_correction_file": os.path.join(self.result_organizer.get_params_output_path(), "corrections.pickle"),
            "result_organizer": self.result_organizer,
            "output_path": self.output_path,
        }

        if mode == Mode.ANALYSIS:
            for plugin_name, plugin_info in plugins.get("plugins", {}).items():
                print(f"Plugin: {plugin_name}; {plugin_info}")
                print(f"Methods: {plugin_info['analysis_methods']}")
                for method_name in plugin_info["analysis_methods"]:
                    logger.info(
                        f"Starting execution of plugin: {plugin_name}, method: {step}"
                    )

                    self.notify_server(f"plugin_execution_{plugin_name}", "before")

                    self.execute_remote_function(plugin_name, method_name, params)

                    result = self.notify_server(
                        f"plugin_execution_{plugin_name}", "after", None
                    )

                    logger.info(
                        f"Finished execution of plugin: {plugin_name}, method: {step}"
                    )
                    print("")
        else:
            for plugin_name, plugin_info in plugins.get("plugins", {}).items():
                print(f"Plugin: {plugin_name}; {plugin_info}")
                print(f"Methods: {plugin_info['development_methods']}")
                for method_name in plugin_info["development_methods"]:
                    logger.info(
                        f"Starting execution of plugin: {plugin_name}, method: {step}"
                    )

                    self.notify_server(f"plugin_execution_{plugin_name}", "before")

                    self.execute_remote_function(plugin_name, method_name, params)

                    result = self.notify_server(
                        f"plugin_execution_{plugin_name}", "after", None
                    )

                    logger.info(
                        f"Finished execution of plugin: {plugin_name}, method: {step}"
                    )
                    print("")

    def check_if_rawTherapee_is_enabled(self, mode, correction_data: ColorCorrectionData):
        """
        Check if RT is enabled. In this case the colorcorrection data needs to be updated.
        Since I will apply RT after the image has been saved, I need to update the colorcorrection data accordingly,
        to use it in the dev phase
        """
        if mode == Mode.DEVELOPMENT:
            return correction_data
        if (self.skip_rawtherapee is False or self.skip_rawtherapee is False) and (
            self.raw_therapee_sharpen is True or self.raw_therapee_light_balance is True):
                correction_data.set_rawtherapee_correction(self.raw_therapee_sharpen, self.raw_therapee_light_balance)
        #         self.comparer.set_step_applied(True) # No, perché no step non è giusto
        # else:
        #     self.comparer.set_step_applied(False)
        return correction_data


    def finalize_step(
        self, correction_data: ColorCorrectionData, step: AnalysisSteps, mode: Mode
    ):
        correction_data = self.check_if_rawTherapee_is_enabled(mode, correction_data)
        self.saved_image = self.result_organizer.save_image(self.result_image)
        self.result_organizer.save_params(correction_data)
        self.save_comparison(mode)
        self.result_organizer.save_logs()

        spacer()

    def execute_remote_function(self, plugin_name: str, method_name: str, params: dict):
        """
        Esegue una funzione remota con i parametri filtrati dal dizionario fornito.

        :param plugin_name: Nome del plugin da cui recuperare la funzione.
        :param method_name: Nome del metodo da eseguire.
        :param params: Dizionario contenente i parametri.
        """
        # Richiedi il codice della funzione al server
        response = self.server_proxy.get_function_code(plugin_name, method_name)
        if "error" in response:
            raise Exception(response["error"])

        # Recupera il codice della funzione
        function_code = response["function_code"]

        # Definisci un contesto per l'esecuzione
        local_context = {}
        exec(function_code, {}, local_context)

        # Ottieni la funzione dal contesto
        function_name = method_name.split(".")[-1]
        if function_name not in local_context:
            raise Exception(f"Function {function_name} not found in the received code.")

        function = local_context[function_name]

        # Ispeziona la firma della funzione per determinare i parametri necessari
        sig = inspect.signature(function)

        # Filtra i parametri basandosi su quelli richiesti dalla funzione
        filtered_params = {
            key: value for key, value in params.items() if key in sig.parameters
        }

        # Verifica se ci sono parametri mancanti obbligatori
        missing_params = [
            key
            for key, param in sig.parameters.items()
            if param.default is param.empty and key not in filtered_params
        ]

        # se è self, rimuovilo
        if "self" in missing_params:
            missing_params.remove("self")

        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Esegui la funzione con i parametri filtrati
        return function(self=ClosedPlugin(), **filtered_params)

    def find_n_compare_patches(self, current_image, current_step):
        # Find patches
        cc_finder = self.color_checker_finder.find(current_image)
        current_measured_patches = cc_finder["color_checker_data"]
        self.cropped_image = cc_finder["cropped_image"]

        # Process cropped image only if process_cropped_image_only is set to True
        if self.process_cropped_image_only:
            self.result_image = self.cropped_image

        # Handle case where color checker is not found
        if current_measured_patches is None:
            message = _("Color checker not found automatically.")
            if appStore.get(AppState.KEY) == AppState.GUI:
                raise AutoFindColorCheckerException(message)
            else:
                logger.info(message)
                self.skip_white_balance = True
            return  # Exit if color checker is not found

        # Compare patches
        current_de00 = self.comparer.run(current_measured_patches, current_step)

        return current_de00, current_measured_patches

    def evaluate_correction(
        self,
        current_measured_patches,
        current_de00,
        current_step,
        correction_data,
        current_image,
        mode,
    ):
        # Discard specific correction based on the step results
        skippable_steps = [
            AnalysisSteps.EXPOSURE,
            AnalysisSteps.WHITE_BALANCE,
            AnalysisSteps.CCM,
            AnalysisSteps.RIDGE_CCM,
            AnalysisSteps.FINLAYSON_CCM,
            AnalysisSteps.WLS,
            AnalysisSteps.POLYNOMIAL_FITTING_CORRECTOR,
            AnalysisSteps.SHAFTLESS
        ]

        # Check if ΔE00 is improved
        de00_has_improved = self.check_improve_de00(
            current_de00, current_step, skippable_steps
        )

        if de00_has_improved:
            self.best_de00 = current_de00
            self.result_image = current_image
            self.measured_patches = current_measured_patches
            self.comparer.set_step_applied(True)
        else:
            # Log skipping step
            logger.info(
                f"The step {current_step} didn't improve DE00. This step will be skipped"
            )
            self.comparer.set_step_applied(False)

            setattr(correction_data, current_step, None)

        # Log current ΔE00
        logger.info(f"Current DE00 at step {current_step}: {self.best_de00}")

    def compare_result(self, current_step, correction_data, current_image, mode):
        # If the mode is development, skip the comparison
        if mode == Mode.DEVELOPMENT:
            self.result_image = current_image
            return

        # Find and compare patches
        current_de00, current_measured_patches = self.find_n_compare_patches(
            current_image, current_step
        )

        # Evaluate correction
        self.evaluate_correction(
            current_measured_patches,
            current_de00,
            current_step,
            correction_data,
            current_image,
            mode,
        )

    def check_improve_de00(self, current_de00, current_step, skippable_steps):
        # Skip steps if ΔE00 is not improved for specific analysis steps
        return not bool(
            current_step in skippable_steps and current_de00 > self.best_de00
        )

    def serialize(self) -> str:
        """Serialize the Core instance to a JSON string."""
        serializable_data = {
            'input_path': self.input_path,
            'output_format': self.output_format,
            'white_field_path': self.white_field_path,
            'output_path': self.output_path,
            'output_color_space' : self.output_color_space,
            # 'process_subfolder': self.process_subfolder,
            # 'extension_2_process': self.extension_2_process,
            # 'parameter_file': self.parameter_file,
            # 'skip_geometric_correction': self.skip_geometric_correction,
            # 'skip_vignetting_correction': self.skip_vignetting_correction,
            'fitting_degree' : self.fitting_degree,
            'development_params' : self.development_params
            ### TODO: aggiungere i restanti parms
        }
        return serializable_data
