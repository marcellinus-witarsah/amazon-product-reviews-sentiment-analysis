import pandas as pd
import os
import sys
from dotenv import find_dotenv, load_dotenv
from zenml import step

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.data.data_cleaning import DataCleaning


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return DataCleaning(df).clean_data()
