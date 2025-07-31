import io
import logging
import os

from utils.utils import read_config

log_levels = {
    # Specifies that a logging category should not write any messages.
    "NONE": 60,
    # Logs describing a critical and irreversible failure of an application or the system, or an error
    "CRITICAL": 50,
    # Logs indicating the moment when the current execution flow is interrupted due to an error.
    # They should indicate an error in the current activity, not an application-level error.
    "ERROR": 40,
    # Logs highlighting an unexpected or unforeseen event in the application flow,
    # but otherwise not causing the application execution to stop.
    "WARNING": 30,
    # Logs keeping track of the overall application flow.
    # These logs should have long-term value.
    "INFO": 20,
    # Logs used for interactive analysis during development.
    # These logs should primarily contain information useful for debugging and do not have long-term value.
    "DEBUG": 10,
    # Logs containing the most detailed messages. These messages may contain sensitive application data.
    # These messages are disabled by default and should never be enabled in a production environment.
    "TRACE": 5,
}


class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        log_directory = "./log"
        log_filename = "application.log"
        log_filepath = os.path.join(log_directory, log_filename)

        # Make sure the directory exists
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        # Add TRACE level
        logging.addLevelName(log_levels.get("TRACE"), "TRACE")
        logging.addLevelName(log_levels.get("NONE"), "NONE")

        config = read_config()

        # Get the log level from the configuration file
        level_name = config.get("settings", "log_level")

        level = log_levels.get(level_name.upper(), log_levels.get("INFO"))

        # Create the main logger
        logger = logging.getLogger("Cocoa")
        logger.setLevel(level)

        # Add the trace method to the logger
        setattr(logging.Logger, "trace", self.trace)

        # Prevent message propagation to higher-level loggers
        logger.propagate = False

        # Handler for writing to file, accepts everything from DEBUG level and above
        fh = logging.FileHandler(log_filepath)
        fh.setLevel(log_levels.get("DEBUG"))

        # Handler for console, shows INFO and higher levels
        ch = logging.StreamHandler()
        ch.setLevel(log_levels.get("TRACE"))

        # StreamHandler with StringIO to capture messages in memory
        self.memory_handler = io.StringIO()
        sh = logging.StreamHandler(self.memory_handler)
        sh.setLevel(log_levels.get("DEBUG"))  # To record everything in memory

        # Log formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Set the formatter for the handlers
        for handler in [fh, ch, sh]:
            handler.setFormatter(formatter)

        # Add the handlers to the logger
        for handler in [fh, ch, sh]:
            logger.addHandler(handler)

        self.logger = logger

    def trace(self, message, *args, **kwargs):
        if self.logger.isEnabledFor(log_levels.get("TRACE")):
            self.logger.log(log_levels.get("TRACE"), message, *args, **kwargs)


# Singleton instance of the Logger class
main_logger = Logger()
# Logger instance
logger = main_logger.logger
# Memory handler to capture all messages in memory
memory_handler = main_logger.memory_handler
