import pandas as pd
import os
import sys
from dotenv import find_dotenv, load_dotenv
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.data.data_preprocessing import DataPreprocessStrategy
from src.data.data_preprocessing import DataSplitStrategy
from src.data.data_preprocessing import DataPreprocessing


@step
def preprocess_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Perform data preprocessing from data cleaning, data labeling, text preprocessing, and data splitting

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: X_train
        pd.DataFrame: X_test
        pd.Series: y_train
        pd.Series: y_test
    """
    preprocess_strategy = DataPreprocessStrategy()
    data_preprocessing = DataPreprocessing(data=data, strategy=preprocess_strategy)
    preprocessed_data = data_preprocessing.handle_data()

    preprocess_strategy = DataSplitStrategy()
    data_preprocessing = DataPreprocessing(
        data=preprocessed_data, strategy=preprocess_strategy
    )
    X_train, X_test, y_train, y_test = data_preprocessing.handle_data()
    return X_train, X_test, y_train, y_test
