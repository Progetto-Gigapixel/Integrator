import traceback

from PyQt5.QtCore import QMutex, QThread, QWaitCondition, pyqtSignal

from config.config import AnalysisSteps, ManualAnalysisSteps, Mode
from core.common.exceptions.geometry_correction_exceptions import (
    GeometricCorrectionException,
)
from core.common.exceptions.vignetting_correction_exceptions import (
    VignettingCorrectionException,
)
from core.common.observable.observable import Observer
from core.core import Core
from core.store.store import appStore
from locales.localization import _
from log.logger import logger


class WorkerThreadQT(QThread):
    update_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)
    manual_correction_signal = pyqtSignal(ManualAnalysisSteps)

    class StopWorkerException(Exception):
        """Custom exception to stop the worker thread"""

        pass

    def __init__(self, core_instance: Core, mode: Mode, skip_appstore_setting: bool):
        super().__init__()
        self.setTerminationEnabled(True)
        self.core_instance = core_instance
        self.mode = mode
        self.skip_appstore_setting = skip_appstore_setting

        self._paused = False
        self._stopped = False
        self._mutex = QMutex()
        self._wait_condition = QWaitCondition()

        excluded_steps = {AnalysisSteps.KEY, AnalysisSteps.FINALIZE}
        self.steps = [step for step in AnalysisSteps if step not in excluded_steps]

        # self.steps = [
        #     AnalysisSteps.INIT,
        #     AnalysisSteps.DECODE_RAW,
        #     AnalysisSteps.GEOMETRIC_CORRECTOR,
        #     AnalysisSteps.VIGNETTING_CORRECTOR,
        #     AnalysisSteps.EXPOSURE,
        #     AnalysisSteps.WHITE_BALANCE,
        #     AnalysisSteps.FLAT_FIELDING,
        #     AnalysisSteps.DENOISING,
        #     AnalysisSteps.RIDGE_CCM,
        #     AnalysisSteps.FINLAYSON_CCM,
        #     AnalysisSteps.WLS,
        #     AnalysisSteps.POLYNOMIAL_FITTING_CORRECTOR,
        #     AnalysisSteps.PLUGIN_EXECUTION,
        # ]

    def update_signal_by_value(self, value: AnalysisSteps):
        if value:
            self._mutex.lock()

            while self._paused:
                self._wait_condition.wait(self._mutex)

            if self._stopped:
                raise WorkerThreadQT.StopWorkerException()

            self._mutex.unlock()
            self.update_signal.emit(
                int((self.steps.index(value.value)) / len(self.steps) * 100)
            )

    def run(self):
        observer = Observer()
        observer.update = self.update_signal_by_value
        appStore.subscribe(observer)

        if not self.skip_appstore_setting:
            appStore.set(AnalysisSteps.KEY, AnalysisSteps.INIT)
        else:
            self.skip_appstore_setting = False

        try:
            if self.mode == Mode.ANALYSIS:
                self.core_instance.analysis_mode()
            elif self.mode == Mode.DEVELOPMENT:
                self.core_instance.development_mode()

            self.finished_signal.emit(False)

        except GeometricCorrectionException as GCE:
            self.manual_correction_signal.emit(ManualAnalysisSteps.GEOMETRIC_CORRECTOR)
        except VignettingCorrectionException as VCE:
            self.manual_correction_signal.emit(ManualAnalysisSteps.VIGNETTING_CORRECTOR)
        except WorkerThreadQT.StopWorkerException:
            logger.info(_("Worker thread stopped"))
            self.finished_signal.emit(True)
        except Exception as e:
            logger.error(_(f"Error in worker thread: {e}"))
            logger.trace(traceback.format_exc())
            self.finished_signal.emit(True)
        finally:
            appStore.unsubscribe(observer)

    def pause(self):
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self._paused = False
        self._wait_condition.wakeAll()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._stopped = True
        self._paused = False
        self._wait_condition.wakeAll()
        self._mutex.unlock()
        self.quit()
