import pandas as pd
import os
import sys
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class DataLabeling:
    def __init__(self, df: pd.DataFrame):
        """Initialize `DataLabeling` object

        Args:
            df (pd.DataFrame): dataset

        Returns:
            None
        """
        self.df = df
        self.logger = Logger(__name__).get_logger()

    def label_data(self) -> pd.DataFrame:
        """Label data based on `overall` column

        Returns:
            pd.DataFrame: dataset
        """
        self.logger.info(f"Labeling Data ...")
        self.df["sentiment"] = self.df["overall"].apply(
            lambda x: "positive" if x >= 3 else "negative"
        )  # convert overall to sentiment
        self.df = self.df.drop(columns=["overall"])
        return self.df
