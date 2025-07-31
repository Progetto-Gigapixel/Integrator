import atexit
import faulthandler
import signal
import sys
import traceback
import tracemalloc

from config.config import AppState
from core.store.store import appStore
from locales.localization import _
from log.logger import logger
from server import start_server
from ui.cli.cli import CliHandler
# from ui.gui.main_window import MainWindow
from utils.utils import start_app
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

faulthandler.enable()
debug_mode = False


class Main:
    def __init__(self):
        if debug_mode:
            # Start the memory profiler
            tracemalloc.start()
            # Cleanup memory leaks on exit
            atexit.register(self.cleanup)

            # Intercept signals
            signals_to_catch = [
                signal.SIGINT,
                signal.SIGTERM,
                signal.SIGABRT,
                signal.SIGSEGV,
                signal.SIGFPE,
                signal.SIGILL,
            ]

            for sig in signals_to_catch:
                signal.signal(sig, self.signal_handler)

    # @profile
    def start(self):
        # run_server_in_thread()
        start_server()

        # Check if the application was started in CLI mode
        isCli = len(sys.argv) > 1

        # Set the application state
        appStore.set(AppState.KEY, AppState.CLI) #if isCli else AppState.GUI)

        # CLI mode
        if isCli:
            cli = CliHandler(sys.argv[1:])
            cli.run()
        # GUI mode
        # else:
        #     start_app(MainWindow)

    def cleanup(self):
        """
        Cleanup memory leaks.
        :return: memory leaks.
        """
        if debug_mode:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")

            logger.info("[ Top 10 memory leaks ]")
            for stat in top_stats[:10]:
                logger.warning(stat)

    def signal_handler(self, sig, frame):
        """
        Handle signals.

        :param sig: The signal.
        :param frame: The frame.
        """
        logger.log(_("Signal received: {0}").format(sig))
        self.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    try:
        main = Main()
        main.start()
    except Exception as e:
        # print(traceback.format_exc())
        logger.trace(traceback.format_exc())
        logger.error(e)
