import pandas as pd
import os
import sys
from dotenv import find_dotenv, load_dotenv
from zenml import step

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.data.data_preprocessing import DataPreprocessing


@step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    return DataPreprocessing(df).preprocess_data()
