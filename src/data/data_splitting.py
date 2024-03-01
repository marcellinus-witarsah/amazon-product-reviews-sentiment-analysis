import logging
import sys
import os
import pandas as pd
from typing import Union, Tuple
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class DataSplitting:
    def __init__(self, df: pd.DataFrame, test_size: float = 0.3):
        """Initialize `DataSplitting` class

        Args:
            df (pd.DataFrame): dataset
            test_size (float): test size proportion
        """
        self.df = df
        self.test_size = test_size
        self.logger = Logger(__name__).get_logger()

    def train_val_test_split(self) -> Union[Tuple[pd.DataFrame], Tuple[pd.Series]]:
        """Split data into train, validation and test

        Returns:
            Union[Tuple[pd.DataFrame], Tuple[pd.Series]]: splitted dataset
        """
        self.logger.info(f"Splitting Data ...")
        X, y = (
            self.df.loc[:, ~self.df.columns.isin(["sentiment"])],
            self.df[["sentiment"]],
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, shuffle=True, random_state=42
        )  # split main data into training and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_test,
            y_test,
            stratify=y_test,
            test_size=0.5,
            shuffle=True,
            random_state=42,
        )  # split test data into validation and test
        return (X_train, X_val, X_test, y_train, y_val, y_test)
