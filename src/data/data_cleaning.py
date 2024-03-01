import pandas as pd
import os
import sys
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class DataCleaning:
    def __init__(self, df: pd.DataFrame):
        """Initialize `DataCleaning` object

        Args:
            df (pd.DataFrame): dataset

        Returns:
            None
        """
        self.df = df
        self.logger = Logger(__name__).get_logger()

    def clean_data(self) -> pd.DataFrame:
        """Clean data

        Args:
            data_path (str): data source path

        Returns:
            pd.DataFrame: dataset
        """
        self.logger.info(f"Cleaning Data ...")
        self.df = self.df.drop_duplicates()  # drop duplicates
        self.df = self.df.dropna(
            subset=["reviewText"], axis=0
        )  # drop missing `reviewText` columns
        self.df = self.df[["reviewText", "overall"]]  # select columns
        self.df = self.df.reset_index(drop=True)  # reset index
        return self.df
