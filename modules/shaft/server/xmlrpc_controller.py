import traceback
from typing import Optional
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

from locales.localization import _
from log.logger import logger

from .xmlrpc_service import XmlRpcService


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)


class XmlRpcController:
    def __init__(self, host: str, port: int):
        self.server = SimpleXMLRPCServer(
            (host, port), requestHandler=RequestHandler, allow_none=True, logRequests=False
        )
        self.server.register_introspection_functions()
        self.service = XmlRpcService()
        self._expose_functions()

    def start(self):
        logger.debug(_("Server XML-RPC running..."))
        try:
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Server error: {e}")
            logger.trace(traceback.format_exc())

    def _expose_functions(self):
        self.server.register_function(self.get_step_result, "get_step_result")
        self.server.register_function(self.set_step_result, "set_step_result")
        self.server.register_function(self.list_plugins, "list_plugins")
        self.server.register_function(self.get_function_code, "get_function_code")

    def set_step_result(
        self, step_name: str, status: int, result: Optional[dict] = None
    ):
        self.service.set_step_result(step_name, status, result)
        return True

    def get_step_result(self, step_name: str):
        return self.service.get_step_result(step_name)

    def list_plugins(self):
        return self.service.list_plugins()

    def get_function_code(self, plugin_name: str, method_name: str):
        return self.service.get_function_code(plugin_name, method_name)
