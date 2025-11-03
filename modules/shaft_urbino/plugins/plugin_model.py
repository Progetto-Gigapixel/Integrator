import subprocess, os
from dataclasses import dataclass, field
from typing import Dict, List

from log.logger import logger


@dataclass
class PluginMethod:
    method_name: str


@dataclass
class PluginInfo:
    name: str
    analysis_methods: List[str] = field(default_factory=list)
    development_methods: List[str] = field(default_factory=list)


@dataclass
class ListPluginsResponse:
    plugins: Dict[str, PluginInfo] = field(default_factory=dict)


class ClosedPlugin:
    def execute_cli(self, command: str, *args):
        """
        Executes a command from the CLI.

        :param command: The command to execute.
        :param args: Additional arguments for the command.
        :return: Dictionary with stdout, stderr, and return code.
        """
        full_command = [command] + list(args)
        logger.info(f"Executing command: {' '.join(full_command)}")
        basedir = os.path.dirname(command)
        try:
            wd = os.getcwd()
            if not basedir == "":
                os.chdir(basedir)
            result = subprocess.run(
                full_command, shell=False, capture_output=True, text=True
            )
            os.chdir(wd)
            logger.debug(f"Command output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Command error output: {result.stderr}")
            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing command: {e}")
            return {
                "stdout": e.stdout.decode() if e.stdout else "",
                "stderr": e.stderr.decode() if e.stderr else str(e),
                "returncode": e.returncode,
            }
        except Exception as e:
            logger.error(f"Generic error: {e}")
            return {"stdout": "", "stderr": str(e), "returncode": -1}
