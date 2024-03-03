import os
import sys
import string
import pandas as pd
from abc import ABC, abstractmethod
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from dotenv import find_dotenv, load_dotenv
from typing import Union

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class DataStrategy(ABC):
    """
    Abstract class for defining strategy to handle data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    def __init__(self) -> None:
        """
        Initialize `DataPreprocessStrategy` class
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_en = stopwords.words("english")
        self.punctuations = string.punctuation
        self.logger = Logger.get_logger(__name__)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text

        Args:
            text (str): text

        Returns:
            str: text
        """
        text = text.lower()  # normalize text
        tokens = word_tokenize(text)  # tokenize text
        filtered_tokens = [
            token
            for token in tokens
            if token not in self.stopwords_en and token not in self.punctuations
        ]  # remove stop word and punctuations
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token) for token in filtered_tokens
        ]  # lemmatize tokens
        return " ".join(lemmatized_tokens)

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data preprocessing (data cleaning, data labeling, and text preprocessing)

        Args:
            data (pd.DataFrame): data

        Returns:
            Union[pd.DataFrame, pd.Series]: preprocessed data
        """
        try:
            # Data Cleaning
            self.logger.info(f"Cleaning Data ...")
            data = data.drop_duplicates()  # drop duplicates
            data = data.dropna(
                subset=["reviewText"], axis=0
            )  # drop missing `reviewText` columns
            data = data.reset_index(drop=True)  # reset index
            self.logger.info(f"Finish cleaning data")

            # Data Labelling
            self.logger.info(f"Labeling Data ...")
            data["sentiment"] = data["overall"].apply(
                lambda x: 1 if x >= 3 else 0
            )  # convert overall to sentiment
            self.logger.info(f"Finish labeling data")

            # Text Preprocessing
            self.logger.info(f"Preprocessing text ...")
            data["preprocessed_review_text"] = data["reviewText"].apply(
                self.preprocess_text
            )  # preprocess text
            data = data[
                (data["preprocessed_review_text"].apply(lambda x: len(x)) != 0)
            ]  # remove 0 length preprocessed text
            data = data[
                ["preprocessed_review_text", "sentiment"]
            ]  # select columns for model training
            self.logger.info(f"Finish preprocessing text")

            return data
        except Exception as e:
            self.logger.warning(e)


class DataSplitStrategy(DataStrategy):
    def __init__(self) -> None:
        """
        Initialize `DataPreprocessStrategy` class
        """
        super().__init__()
        self.logger = Logger.get_logger(__name__)

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """Split data into training and validation

        Args:
            data (pd.DataFrame): data

        Returns:
            Union[pd.DataFrame, pd.Series]: train and validation data
        """
        try:
            self.logger.info("Splitting Data ...")
            X = data.drop(columns=["sentiment"], axis=1)
            y = data["sentiment"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=0.2, shuffle=True, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.warning(e)


class DataPreprocessing:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        self.logger = Logger.get_logger(__name__)

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """_summary_

        Returns:
            Union[pd.DataFrame, pd.Series]: preprocessed data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            self.logger.warning(e)
