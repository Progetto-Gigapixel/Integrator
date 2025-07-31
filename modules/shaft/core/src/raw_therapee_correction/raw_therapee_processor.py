import os
import subprocess
from pathlib import Path, PureWindowsPath, PurePosixPath
from core.common.exceptions.raw_therapee_exception import RawTherapeeException, RawTherapeeProcessException
from utils.utils import read_config
from enum import Enum
from sys import platform


class RawTherapeeProcessor():
    class Action(Enum):
        PROCESS_EVERYTHING = 0
        SHARPEN_ONLY = 1
        LIGHTS_BALANCE_ONLY = 2

    def __init__(self):
        """
        Inizializza il processore RawTherapee.

        Args:
            rt_cli_path (str): Percorso all'eseguibile rawtherapee-cli
            pp3_profile_path (str): Percorso al file .pp3
            output_dir (str, opzionale): Directory di output (default: stessa dell'immagine)
        """
        config = read_config() # Todo: centralizzare la lettura del config
        rt_folder = Path(config.get("directories", "rawtherapee_path"))
        rt_profiles_path = os.path.abspath(config.get("directories", "rt_profiles_path"))
        self.rt_cli_path = os.path.join(rt_folder, 'rawtherapee-cli.exe')

        self.pp3_profile_path_process_everything = os.path.join(rt_profiles_path, 'process_everything.pp3')
        self.pp3_profile_path_sharpen_only = os.path.join(rt_profiles_path, 'sharpen_only.pp3')
        self.pp3_profile_path_lights_balance_only = os.path.join(rt_profiles_path, 'lights_balance_only.pp3')

        self._validate_paths()

    def _validate_paths(self):
        if not os.path.isfile(self.rt_cli_path):
            raise FileNotFoundError(f"rawtherapee-cli non trovato in: {self.rt_cli_path}")
        if not os.path.isfile(self.pp3_profile_path_process_everything):
            raise FileNotFoundError(f"File .pp3 non trovato in: {self.pp3_profile_path_process_everything}")
        if not os.path.isfile(self.pp3_profile_path_sharpen_only):
            raise FileNotFoundError(f"File .pp3 non trovato in: {self.pp3_profile_path_sharpen_only}")
        if not os.path.isfile(self.pp3_profile_path_lights_balance_only):
            raise FileNotFoundError(f"File .pp3 non trovato in: {self.pp3_profile_path_lights_balance_only}")

    def get_proper_pp3_profile(self, action:Action.PROCESS_EVERYTHING):
        if action == RawTherapeeProcessor.Action.SHARPEN_ONLY:
            pp3_profile_path = self.pp3_profile_path_sharpen_only
        elif action == RawTherapeeProcessor.Action.LIGHTS_BALANCE_ONLY:
            pp3_profile_path = self.pp3_profile_path_lights_balance_only
        else:
            pp3_profile_path = self.pp3_profile_path_process_everything
        return pp3_profile_path

    def rename_tif_input_file(self, tif_path, revert_to_original=False):
        # Rinomino il file e lavoro su quello
        tif_dir = os.path.dirname(tif_path)
        tif_file_with_ext = os.path.basename(tif_path)
        tif_filename, ext = os.path.splitext(tif_file_with_ext)
        tif_backup = os.path.join(tif_dir + os.sep, f"{tif_filename}_backup{ext}")
        # Rinomino
        try:
            if revert_to_original:
                if os.path.isfile(tif_backup) and os.path.isfile(tif_path):
                    self.delete_file(tif_path)
                os.rename(tif_backup, tif_path)
            else:
                if os.path.isfile(tif_backup):
                    self.delete_file(tif_backup)
                os.rename(tif_path, tif_backup)
        except PermissionError:
            raise RawTherapeeException(f"Permission denied renaming file: {tif_path}")
        except Exception as e:
            raise RawTherapeeException(f"Error renaming file: {tif_path} \nError: {e}")
        return tif_backup

    def delete_file(self, file_path):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            raise RawTherapeeException(f"File not found: {file_path}")
        except PermissionError:
            raise RawTherapeeException(f"Permission denied deleting file: {file_path}")
        except Exception as e:
            raise RawTherapeeException(f"Error deleting file: {file_path} \nError: {e}")

    def process_image(self, tif_path, action=Action.PROCESS_EVERYTHING):
        """
        Esegue rawtherapee-cli su una singola immagine TIF.

        Args:
            tif_path (str): Percorso all'immagine .tif

        Returns:
            dict: {'status': 'ok'} oppure {'status': 'error', 'message': ...}
        """

        if platform == "darwin" or platform == "linux" or platform == "linux2":
            tif_path = PurePosixPath(tif_path)
        elif platform == "win32":
            tif_path = PureWindowsPath(tif_path)

        if not os.path.isfile(tif_path):
            raise FileNotFoundError(f"TIF File not found before executing RawTherapee. File not found: {tif_path}")

        # Visto che RT si rifiuta di sovrascrivere il file se questo è attualmente aperto, lo rinomino
        tif_backup = self.rename_tif_input_file(tif_path)
        pp3_profile_path = self.get_proper_pp3_profile(action)

        command = [
            self.rt_cli_path,
            '-o', tif_path,
            '-p', pp3_profile_path,
            '-tz',                      # Salvo in tif
            '-b 8',                     # A 8 bit
            # '-Y',                       # Overwite # TODO: overwrite è nei parametri, e va gestito
            '-c', tif_backup,
        ]

        try:
            subprocess.run(command, check=True)
            # Se tutto è andato bene, cancello il file di backup
            self.delete_file(tif_backup)
            return {'status': 'ok'}

        except subprocess.CalledProcessError as e:
            self.rename_tif_input_file(tif_path, revert_to_original=True)
            raise RawTherapeeProcessException(f"RawTherapee processing failed: {e.stderr}", self.__class__.__name__, e.returncode) from e

        except Exception as e:
            self.rename_tif_input_file(tif_path, revert_to_original=True)
            raise RawTherapeeException(f"RawTherapee processing failed: {e.stderr}", self.__class__.__name__) from e
