import importlib.util
import inspect
import json
import os
import textwrap
from dataclasses import asdict
from typing import Dict

from log.logger import logger
from utils.utils import find_project_root

from .plugin_model import ListPluginsResponse, PluginInfo


class PluginHandler:
    def __init__(self, plugins_dir="plugins/closed"):
        self.plugins_dir = os.path.join(find_project_root(), plugins_dir)
        self.loaded_plugins = {}
        self.load_plugins()

    def load_plugins(self):
        for plugin_name in os.listdir(self.plugins_dir):
            plugin_path = os.path.join(self.plugins_dir, plugin_name)
            if os.path.isdir(plugin_path):
                try:
                    interface_path = os.path.join(plugin_path, "interface.json")
                    plugin_py_path = os.path.join(plugin_path, "plugin.py")

                    if not os.path.exists(interface_path) or not os.path.exists(
                        plugin_py_path
                    ):
                        logger.warning(
                            f"Plugin {plugin_name} is missing interface.json or plugin.py."
                        )
                        continue

                    with open(interface_path, "r") as f:
                        plugin_interface = json.load(f)

                    plugin_module = self._load_python_module(
                        plugin_name, plugin_py_path
                    )

                    if plugin_module:
                        self.loaded_plugins[plugin_name] = {
                            "interface": plugin_interface,
                            "module": plugin_module.plugin,
                        }
                        logger.debug(f"Plugin {plugin_name} loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name}: {e}")

    def _load_python_module(self, plugin_name, module_path):
        spec = importlib.util.spec_from_file_location(plugin_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def get_function_code(self, plugin_name: str, method_name: str):
        """
        Returns the source code of a method in a plugin.

        :param plugin_name: The name of the plugin.
        :param method_name: The name of the method.

        :return: A dictionary with the function code or an error message.
        """
        plugin = self.loaded_plugins.get(plugin_name)
        if not plugin:
            return {"error": f"Plugin {plugin_name} not found."}

        method = getattr(plugin["module"], method_name, None)
        if not method:
            return {
                "error": f"Method {method_name} not implemented in plugin {plugin_name}."
            }

        try:
            # Get the source code of the method
            function_code = inspect.getsource(method)
            normalized_code = textwrap.dedent(
                function_code
            )  # Remove common leading whitespace
            return {"function_code": normalized_code}
        except Exception as e:
            return {"error": f"Unable to retrieve function code: {str(e)}"}

    def list_plugins(self) -> Dict:
        """
        Lists the available plugins.

        :return: A dictionary with the list of plugins.
        """
        response = ListPluginsResponse(
            plugins={
                plugin_name: PluginInfo(
                    name=plugin_data["interface"]["name"],
                    analysis_methods=list(
                        plugin_data["interface"]["analysis_methods"].keys()
                    ),
                    development_methods=list(
                        plugin_data["interface"]["development_methods"].keys()
                    ),
                )
                for plugin_name, plugin_data in self.loaded_plugins.items()
            }
        )
        return asdict(response)
