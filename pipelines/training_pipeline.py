import os
import sys
import pandas as pd
from zenml import pipeline
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocess_data
from steps.train_model import train_model
from steps.evaluate import evaluate


@pipeline
def training_pipeline(data_path: str):
    data = ingest_data(data_path=data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train, **{})
    accuracy_score, precision_score, recall_score, f1_score = evaluate(
        model, X_test, y_test
    )
    print(
        f"""Accuracy: {accuracy_score}, Precision: {precision_score}, Recall: {recall_score}, F1: {f1_score}"""
    )
