import logging


class Logger:
    """Class for logging"""

    # def __init__(self, logger_name: str) -> None:
    #     """Initialize `Logger` object

    #     Args:
    #         logger_name (str): logger name

    #     Returns:
    #         None
    #     """
    #     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    #     logging.basicConfig(level=logging.INFO, format=log_fmt)
    #     self.logger_name = logger_name
    @staticmethod
    def get_logger(logger_name: str) -> logging.Logger:
        """
        Get logger object

        Args:
            logger_name (str): logger name

        Returns:
            logging.Logger: logger
        """
        # Create a logger for the specified name
        logger = logging.getLogger(logger_name)

        # Define the log format
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Create a formatter with the specified format
        formatter = logging.Formatter(log_fmt)

        # Create a handler and set the formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)
        return logger
