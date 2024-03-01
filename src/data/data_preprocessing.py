import os
import sys
import string
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.logger import Logger


class DataPreprocessing:
    def __init__(self, df: pd.DataFrame):
        """Initialize `DataPreprocessing` object

        Args:
            df (pd.DataFrame): dataset

        Returns:
            None
        """
        self.df = df
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_en = stopwords.words("english")
        self.punctuations = string.punctuation
        self.logger = Logger(__name__).get_logger()

    def preprocess_text(self, text: str) -> str:
        """Preprocess text

        Args:
            text (str): text

        Returns:
            str: text
        """
        tokens = word_tokenize(
            text.lower()
        )  # normalize, remove punctuations, and tokenize text
        filtered_tokens = [
            token
            for token in tokens
            if token not in self.stopwords_en and token not in self.punctuations
        ]  # filter stop words
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token) for token in filtered_tokens
        ]  # lemmatize words
        return " ".join(lemmatized_tokens)  # Join the tokens back into a string

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess data

        Returns:
            pd.DataFrame: dataset
        """
        self.logger.info(f"Preprocessing Data ...")
        self.df["preprocessed_review_text"] = self.df["reviewText"].apply(
            self.preprocess_text
        )  # text preprocessing
        self.df = self.df[
            (self.df["preprocessed_review_text"].apply(lambda x: len(x)) != 0)
        ]  # remove 0 length preprocess text
        self.df = self.df[
            ["preprocessed_review_text", "sentiment"]
        ]  # select columns for model training
        return self.df
