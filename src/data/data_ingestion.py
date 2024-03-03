import pandas as pd
from zenml.logger import get_logger


class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = get_logger(__name__)

    def get_data(self) -> pd.DataFrame:
        self.logger.info(f"Ingesting Data from {self.data_path} ...")
        df = pd.read_csv(self.data_path)
        return df
