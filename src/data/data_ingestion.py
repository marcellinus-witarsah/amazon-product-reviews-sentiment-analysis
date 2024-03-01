import pandas as pd
import os
import sys
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class DataIngestion:
    """Class for data ingestion"""

    def __init__(self, data_path: str):
        """Initialize `DataIngestion` object

        Args:
            data_path (str): data source path

        Returns:
            None
        """
        self.data_path = data_path
        self.logger = Logger(__name__).get_logger()

    def get_data(self) -> pd.DataFrame:
        """Get data in pandas DataFrame format

        Returns:
            pd.DataFrame: dataset
        """
        self.logger.info(f"Ingesting Data from {self.data_path} ...")
        df = pd.read_csv(self.data_path)
        return df
