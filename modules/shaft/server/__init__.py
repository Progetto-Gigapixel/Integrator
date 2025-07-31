import threading
from log.logger import logger
from utils.utils import get_config

from .xmlrpc_controller import XmlRpcController


def start_server():
    host = get_config("settings", "server_host")
    port = int(get_config("settings", "server_port"))
    logger.debug(f"Starting server on {host}:{port}")
    server = XmlRpcController(host, port)
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()
