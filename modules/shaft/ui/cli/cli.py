import pyfiglet
from config.config import DEVELOPMENT_PARAMS, SKIP_PARAMS, OutputColorSpaces, Mode
from core.core import Core
from core.utils.core_utils import find_files_by_format
from locales.localization import _
from ui.cli.common.cli_parser.cli_parser import CliParser
from ui.process_threads.worker_thread import WorkerThread
from utils.utils import spacer
import concurrent.futures
import os, psutil
from log.logger import logger
import traceback
import multiprocessing



def _get_file_size_kb(path):
    size_bytes = os.path.getsize(path)
    return size_bytes

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
        process_cropped_image_only = self.args.crop_target

        try:
            output_color_space = OutputColorSpaces(output_color_space)
        except ValueError:
            raise ValueError(
                _("{} is not a valid output color space.").format(output_color_space)
            )


        _print_intro()
        spacer()


        if self.mode == Mode.ANALYSIS:
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
                process_cropped_image_only
            )
        else:
            # Find the images to process
            images = find_files_by_format(input_directory, self.args.extension, self.args.process_subfolder)

            if not images:
                logger.warning(_("No images found to process"))
                return

            # Prende il peso della prima immagine per avere una valutazione sommaria della possibilità di processare in ram
            image_file_size = _get_file_size_kb(images[0])

            # Calcola il numero massimo di worker
            max_threads = os.cpu_count() or 4  # Default a 4 se cpu_count() ritorna None
            ram = psutil.virtual_memory().available
            # Stima più conservativa: assume 12x dimensione file per processo
            max_ram_workers = max(1, ram // (image_file_size * 12))
            # Limita a 7 worker per evitare overhead eccessivo
            max_workers = min(max_threads, max_ram_workers, 7)

            logger.info(_(f"Processing {len(images)} images with {max_workers} workers"))

            # Usa ProcessPoolExecutor per isolamento memoria completo
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Prepara gli argomenti per ogni processo
                process_args = [
                    (
                        input_file,
                        output_format,
                        input_white_field,
                        output_path,
                        skip_params,
                        output_color_space,
                        fitting_degree,
                        raw_therapee_sharpen,
                        raw_therapee_light_balance,
                        development_params,
                        process_cropped_image_only
                    )
                    for input_file in images
                ]

                # Sottometti i task al process pool
                futures = {
                    executor.submit(_process_single_file_static, args): args[0]
                    for args in process_args
                }

                # Gestisci i risultati man mano che vengono completati con timeout
                completed = 0
                for future in concurrent.futures.as_completed(futures, timeout=3600):
                    input_file = futures[future]
                    try:
                        result = future.result(timeout=600)  # 10 minuti timeout per singolo file
                        completed += 1
                        if result.get("status") == "error":
                            logger.error(f"Error processing {result['path']}: {result['error']}")
                        else:
                            logger.info(f"Successfully processed {result['path']} ({completed}/{len(images)})")
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Timeout processing {input_file}")
                    except Exception as e:
                        logger.error(f"Error executing process for {input_file}: {str(e)}\n{traceback.format_exc()}")

    def _process_single_file(self, core_thread):
        """Processa un singolo file utilizzando un CoreThread dedicato (deprecato, usa _process_single_file_static)"""
        try:
            core_thread.development_mode(Mode.DEVELOPMENT)
            return {"status": "success", "path": core_thread.input_path}
        except Exception as e:
            return {
                "status": "error",
                "path": core_thread.input_path,
                "error": str(e)
            }


def _process_single_file_static(args):
    """
    Funzione statica per processare un singolo file in un processo separato.
    Richiede funzione statica per essere serializzabile con pickle (ProcessPoolExecutor).
    """
    (
        input_file,
        output_format,
        input_white_field,
        output_path,
        skip_params,
        output_color_space,
        fitting_degree,
        raw_therapee_sharpen,
        raw_therapee_light_balance,
        development_params,
        process_cropped_image_only
    ) = args

    try:
        # Crea una nuova istanza Core per questo processo
        core_instance = Core(
            input_file,
            output_format,
            input_white_field,
            output_path,
            skip_params,
            output_color_space,
            fitting_degree,
            raw_therapee_sharpen,
            raw_therapee_light_balance,
            development_params,
            process_cropped_image_only
        )

        # Esegui il processing
        core_instance.development_mode(Mode.DEVELOPMENT)

        return {"status": "success", "path": input_file}

    except Exception as e:
        return {
            "status": "error",
            "path": input_file,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    # Necessario per ProcessPoolExecutor su Windows
    multiprocessing.freeze_support()
    import sys
    cli_handler = CliHandler(sys.argv)
    cli_handler.run()
