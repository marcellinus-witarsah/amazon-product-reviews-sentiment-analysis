import pandas as pd
import os
import sys
from dotenv import find_dotenv, load_dotenv
from zenml import step

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.data.data_ingestion import DataIngestion


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    return DataIngestion(data_path).get_data()
