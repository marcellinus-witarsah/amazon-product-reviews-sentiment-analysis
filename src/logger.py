import logging


class Logger:
    """Class for logging"""

    def __init__(self, logger_name: str) -> None:
        """Initialize `Logger` object

        Args:
            logger_name (str): logger name

        Returns:
            None
        """
        self.log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        self.logger_name = logger_name

    def get_logger(self) -> logging.Logger:
        """Get logger object

        Returns:
            logging.Logger: logger
        """
        self.logger = logging.getLogger(self.logger_name)
        return self.logger
