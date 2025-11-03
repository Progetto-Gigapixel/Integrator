import threading
import traceback

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


class WorkerThread(threading.Thread):
    def __init__(
        self,
        core_instance: Core,
        mode: Mode,
        skip_appstore_setting: bool,
        on_finished=None,
        on_manual_correction=None,
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

    def run(self):
        if not self.skip_appstore_setting:
            appStore.set(AnalysisSteps.KEY, AnalysisSteps.INIT)
        else:
            self.skip_appstore_setting = False

        try:
            if self.mode == Mode.ANALYSIS:
                self.core_instance.analysis_mode()
            elif self.mode == Mode.DEVELOPMENT:
                self.core_instance.development_mode()

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

    def stop(self):
        self._stopped = True
