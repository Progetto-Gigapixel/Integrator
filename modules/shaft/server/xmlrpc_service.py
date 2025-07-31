from typing import Optional

from locales.localization import _
from plugins.plugin_handler import PluginHandler

from .xmlrpc_model import StepsResult


class XmlRpcService:
    def __init__(self):
        self.steps_result = StepsResult()
        self.plugin_handler = PluginHandler()
        self.plugins = {}

    def set_step_result(
        self, step_name: str, status: int, result: Optional[dict] = None
    ):
        self.steps_result.set_step(step_name, status, result)

    def get_step_result(self, step_name: str):
        return self.steps_result.get_step(step_name)

    def list_plugins(self):
        return self.plugin_handler.list_plugins()

    def get_function_code(self, plugin_name: str, method_name: str):
        return self.plugin_handler.get_function_code(plugin_name, method_name)
