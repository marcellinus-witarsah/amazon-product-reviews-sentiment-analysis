import os
import sys
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Insert project folder into Python System
load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class Evaluation(ABC):
    """
    Abstract class for evaluation
    """

    @abstractmethod
    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ):
        pass


class Accuracy(Evaluation):
    def __init__(self) -> None:
        self.logger = Logger.get_logger(__name__)

    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> float:
        try:
            self.logger.info("Calculate Accuracy Score ...")
            score = accuracy_score(y_true, y_pred)
            self.logger.info("Finish Calculating Accuracy Score ...")
            return score
        except Exception as e:
            self.logger.warning(e)


class Recall(Evaluation):
    def __init__(self) -> None:
        self.logger = Logger.get_logger(__name__)

    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> float:
        try:
            self.logger.info("Calculate Recall Score ...")
            score = recall_score(y_true, y_pred)
            self.logger.info("Finish Calculating Recall Score ...")
            return score
        except Exception as e:
            self.logger.warning(e)


class Precision(Evaluation):
    def __init__(self) -> None:
        self.logger = Logger.get_logger(__name__)

    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> float:
        try:
            self.logger.info("Calculate Precision Score ...")
            score = precision_score(y_true, y_pred)
            self.logger.info("Finish Calculating Precision Score ...")
            return score
        except Exception as e:
            self.logger.warning(e)


class F1(Evaluation):
    def __init__(self) -> None:
        self.logger = Logger.get_logger(__name__)

    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> float:
        try:
            self.logger.info("Calculate F1 Score ...")
            score = f1_score(y_true, y_pred)
            self.logger.info("Finish Calculating F1 Score ...")
            return score
        except Exception as e:
            self.logger.warning(e)
