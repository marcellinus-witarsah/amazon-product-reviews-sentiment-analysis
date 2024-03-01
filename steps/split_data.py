import os
import sys
import pandas as pd
from zenml import step
from dotenv import find_dotenv, load_dotenv
from typing import Union, Tuple

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.data.data_splitting import DataSplitting


@step
def split_data(df: pd.DataFrame, test_size: float = 0.3):
    return DataSplitting(df, test_size=test_size).train_val_test_split()
