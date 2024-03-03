import os
import sys
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = Logger.get_logger(__name__)

    def get_data(self) -> pd.DataFrame:
        self.logger.info(f"Ingesting Data from {self.data_path} ...")
        df = pd.read_csv(self.data_path)
        return df
