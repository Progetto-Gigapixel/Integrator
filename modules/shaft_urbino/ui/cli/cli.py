import pyfiglet
from config.config import DEVELOPMENT_PARAMS, SKIP_PARAMS, OutputColorSpaces
from core.core import Core
from locales.localization import _
from ui.cli.common.cli_parser.cli_parser import CliParser
from ui.process_threads.worker_thread import WorkerThread
from utils.utils import spacer


def _print_intro():
    print(pyfiglet.figlet_format("COCOA", font="small"))
    print("\n")


class CliHandler:
    def __init__(self, argv):
        self.argv = argv
        self.parser = CliParser(argv)
        self.args = self.parser.args
        self.core: Core = None
        self.worker_thread: WorkerThread = None

    def run(self):
        # Check if help is requested

        if len(self.argv) == 1 or '--help' in self.argv or '-h' in self.argv:
            self.parser.print_help()
            return
        self._init_run()

        self.worker_thread = WorkerThread(self.core, self.mode, False)
        self.worker_thread.start()
        self.worker_thread.join()

    def _init_run(self):
        self.mode = self.args.mode
        skip_params = {
            SKIP_PARAMS.EXPOSURE: self.args.skip_exposure,
            SKIP_PARAMS.WHITE_BALANCE : self.args.skip_wb,
            SKIP_PARAMS.CCM: self.args.skip_ccm,
            SKIP_PARAMS.FINLAYSON_CCM: self.args.skip_finlayson_ccm,
            SKIP_PARAMS.RIDGE_CCM: self.args.skip_ridge_ccm,
            SKIP_PARAMS.WLS:self.args.skip_wls,
            SKIP_PARAMS.POLYNOMIAL_FITTING: self.args.skip_poly,
            SKIP_PARAMS.SHAFT: self.args.skip_shaft,
            SKIP_PARAMS.RAWTHERAPEE: self.args.skip_rawtherapee,
        }
        development_params = {
            DEVELOPMENT_PARAMS.PROCESS_SUBFOLDER: self.args.process_subfolder,
            DEVELOPMENT_PARAMS.DO_NOT_OVERWRITE_FILES: self.args.overwrite_files,
            DEVELOPMENT_PARAMS.EXTENSION_2_PROCESS: self.args.extension,
            DEVELOPMENT_PARAMS.PARAMETER_PATH: self.args.parameter_path,
        }

        input_directory: str = self.args.input
        input_white_field: str = self.args.white_field
        output_format: str = self.args.format
        output_path: str = self.args.output
        output_color_space: str = self.args.color.lower()
        fitting_degree: int = self.args.degree
        raw_therapee_light_balance: bool = self.args.light_balance
        raw_therapee_sharpen: bool = self.args.sharpen



        try:
            output_color_space = OutputColorSpaces(output_color_space)
        except ValueError:
            raise ValueError(
                _("{} is not a valid output color space.").format(output_color_space)
            )

        self.core = Core(
            input_directory,
            output_format,
            input_white_field,
            output_path,
            skip_params,
            output_color_space,
            fitting_degree,
            raw_therapee_sharpen,
            raw_therapee_light_balance,
            development_params,
        )

        _print_intro()
        spacer()

if __name__ == "__main__":
    import sys
    cli_handler = CliHandler(sys.argv)
    cli_handler.run()
