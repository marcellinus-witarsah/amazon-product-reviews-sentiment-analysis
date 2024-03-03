import os
import sys
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import ClassifierMixin
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class Model(ABC):
    """
    Abstract class for machine learning model
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """_summary_

        Args:
            X_train : Training data
            y_train : Trainign labels
        Returns:
            None
        """
        pass


class CountVectorizerMultinomialNB(Model):
    def __init__(self) -> None:
        self.logger = Logger.get_logger(__name__)

    def train(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: Union[pd.DataFrame, pd.Series],
        **params
    ) -> Pipeline:
        """Train model

        Args:
            X_train (Union[pd.DataFrame, pd.Series]): Training data
            y_train (Union[pd.DataFrame, pd.Series]): Training labels

        Raises:
            e: Training error description

        Returns:
            ClassifierMixin: Trained Model
        """
        try:
            model = Pipeline(
                [
                    ("CountVectorizer", CountVectorizer()),
                    ("MultinomialNB", MultinomialNB(**params)),
                ]
            )
            model.fit(X_train["preprocessed_review_text"], y_train)
            self.logger.info("Finished model training")
            return model
        except Exception as e:
            self.logger.error(e)
            raise e
