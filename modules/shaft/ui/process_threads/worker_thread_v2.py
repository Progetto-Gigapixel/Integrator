import threading
import traceback

import psutil

from config.config import AnalysisSteps, ManualAnalysisSteps, Mode
from core.common.exceptions.geometry_correction_exceptions import (
    GeometricCorrectionException,
)
from core.common.exceptions.vignetting_correction_exceptions import (
    VignettingCorrectionException,
)
from core.core import Core
from core.store.store import appStore
from locales.localization import _
from log.logger import logger
import concurrent.futures.process
import multiprocessing
# from core.common.color_correction_params.colorCorrectionData import ColorCorrectionData
from core.utils.core_utils import find_files_by_format
from typing import List, Dict, Any, Optional, Callable
import json


class WorkerThread(threading.Thread):
    def __init__(
            self,
            core_instance: Core,
            mode: Mode,
            skip_appstore_setting: bool,
            on_finished=None,
            on_manual_correction=None
    ):
        """
        WorkerThread for running core analysis in a separate thread.

        :param core_instance: Core instance for processing
        :param mode: Operating mode (ANALYSIS or DEVELOPMENT)
        :param skip_appstore_setting: Whether to skip app store initialization
        :param on_finished: Callback function for when the thread finishes
        :param on_manual_correction: Callback function for manual correction steps
        """
        super().__init__()
        self.core_instance = core_instance
        self.mode = mode
        self.skip_appstore_setting = skip_appstore_setting
        self._stopped = False
        self.on_finished = on_finished
        self.on_manual_correction = on_manual_correction

        # Calculate max workers based on CPU cores and available memory
        cpu_count = multiprocessing.cpu_count()
        # available_ram = psutil.virtual_memory().available
        #
        # # Get average file size if in development mode
        # avg_file_size = 0
        # if mode == Mode.DEVELOPMENT and hasattr(core_instance, 'input_path'):
        #     files = find_files_by_format(
        #         core_instance.input_path,
        #         core_instance.extension_2_process,
        #         core_instance.process_subfolder
        #     )
        #     if files:
        #         total_size = sum(os.path.getsize(f) for f in files)
        #         avg_file_size = total_size / len(files)
        #
        # # Estimate workers based on RAM (leave 20% free)
        available_ram = psutil.virtual_memory().available
        avg_file_size = 12 * 1024 * 1024 * 1024  # assume average file size of 12 GB
        ram_based_workers = int(available_ram * 0.8 / max(avg_file_size, 1))
        logger.info(_(f"System has {cpu_count} CPU cores and {available_ram / (1024**3):.2f} GB available RAM."))
        # Use minimum of CPU and RAM based workers
        self.max_workers = min(cpu_count, ram_based_workers)
    
    def run(self):
        if not self.skip_appstore_setting:
            appStore.set(AnalysisSteps.KEY, AnalysisSteps.INIT)
        else:
            self.skip_appstore_setting = False

        try:
            if self.mode == Mode.ANALYSIS:
                self.core_instance.analysis_mode()
            elif self.mode == Mode.DEVELOPMENT:
                self._run_development_with_processes()

            if self.on_finished:
                self.on_finished(False)

        except GeometricCorrectionException:
            if self.on_manual_correction:
                self.on_manual_correction(ManualAnalysisSteps.GEOMETRIC_CORRECTOR)
        except VignettingCorrectionException:
            if self.on_manual_correction:
                self.on_manual_correction(ManualAnalysisSteps.VIGNETTING_CORRECTOR)
        except Exception as e:
            logger.error(_(f"Error in worker thread: {e}"))
            logger.trace(traceback.format_exc())
            if self.on_finished:
                self.on_finished(True)

    def _run_development_with_processes(self):
        """Run development mode using separate processes for each image."""
        if self.max_workers > 1:
            logger.info(_("Multithreading execution started..."))
        else:
            logger.info(_("Single-threaded execution started..."))
        # Prepare serializable data
        serializable_core_data = self._prepare_serializable_core_data()

        # Find images to process
        images = find_files_by_format(
            self.core_instance.input_path,
            self.core_instance.extension_2_process,
            self.core_instance.process_subfolder
        )

        # Process images in parallel using processes
        results = []
        num_errors=0
        logger.info(_(f"Processing {len(images)} images with up to {self.max_workers} parallel workers..."))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_image = {
                executor.submit(
                    process_single_image_worker,
                    image_path,
                    serializable_core_data,
                    self.mode
                ): image_path for image_path in images
            }

            for future in concurrent.futures.as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Processed: {image_path}")
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    num_errors=num_errors+1
                    results.append({"status": "error", "path": image_path, "error": str(e)})
        if num_errors>0:
            logger.warning(_(f"Multithreading execution finished with {num_errors} errors."))
        else:
            logger.info(_("Multithreading execution finished successfully."))
        return results

    def _prepare_serializable_core_data(self) -> Dict[str, Any]:
        """Prepare core instance data for serialization to child processes."""
        return self.core_instance.serialize()


    def stop(self):
        self._stopped = True

def process_single_image_worker(image_path: str, core_data: Dict[str, Any], mode: Mode) -> Dict[str, Any]:
    """
    Worker function that runs in a separate process.
    Each process gets its own memory space, eliminating race conditions.

    :param image_path: Path to the image to process
    :param core_data: Serializable core instance data
    :param mode: Processing mode
    :return: Processing result
    """
    try:
        # Create a new Core instance in this process
       core_instance = Core(
            input_path=core_data['input_path'],
            output_format= core_data['output_format'],
            output_path=core_data['output_path'],
            white_field_path=core_data['white_field_path'],
            output_color_space=core_data['output_color_space'],
            fitting_degree=core_data['fitting_degree'],
            development_params=core_data['development_params']
       )
        # Set the image path for this process
       core_instance.input_path = image_path
       core_instance.development_mode()

       return {"status": "success", "path": image_path}
    except Exception as e:
        return {"status": "error", "path": image_path, "error": str(e)}
