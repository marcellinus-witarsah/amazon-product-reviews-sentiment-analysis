import logging


class Logger:
    """Class for logging"""

    def __init__(self):
        """Initialize `Logger` object"""
        self.log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.logging.basicConfig(level=logging.INFO, format=log_fmt)

    def get_logger(self, logger_name: str) -> logging.Logger:
        """Get logger object

        Args:
            logger_name (str): logger name

        Returns:
            logging.Logger: logger
        """
        logger = logging.getLogger(logger_name)
        return logger
