import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from dotenv import find_dotenv, load_dotenv
from zenml.logger import get_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        self.logger = get_logger(__name__)

    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> float:
        try:
            self.logger.info("Calculate accuracy score ...")
            score = accuracy_score(y_true, y_pred)
            return score
        except Exception as e:
            self.logger.warning(e)


class Recall(Evaluation):
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> float:
        try:
            self.logger.info("Calculate recall score ...")
            score = recall_score(y_true, y_pred)
            return score
        except Exception as e:
            self.logger.warning(e)


class Precision(Evaluation):
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> float:
        try:
            self.logger.info("Calculate precision score ...")
            score = precision_score(y_true, y_pred)
            return score
        except Exception as e:
            self.logger.warning(e)


class F1(Evaluation):
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def calculate_score(
        self,
        y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> float:
        try:
            self.logger.info("Calculate f1 score ...")
            score = f1_score(y_true, y_pred)
            return score
        except Exception as e:
            self.logger.warning(e)
