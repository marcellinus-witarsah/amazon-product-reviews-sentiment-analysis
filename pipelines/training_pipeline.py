import os
import sys
import pandas as pd
from zenml import pipeline
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.label_data import label_data
from steps.preprocess_data import preprocess_data
from steps.split_data import split_data


@pipeline
def training_pipeline(data_path: str) -> pd.DataFrame:
    df = ingest_data(data_path=data_path)
    df = clean_data(df=df)
    df = label_data(df=df)
    df = preprocess_data(df=df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, test_size=0.3)
