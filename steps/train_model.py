import os
import sys
import pandas as pd
import numpy as np
import mlflow
from typing import Union
from sklearn.pipeline import Pipeline
from dotenv import find_dotenv, load_dotenv
from zenml import step
from zenml.client import Client

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.models.model import CountVectorizerMultinomialNB


# find and use experiment tracker available
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(
    X_train: Union[np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
) -> Pipeline:
    mlflow.sklearn.autolog()  # automatic evaluation
    model = CountVectorizerMultinomialNB()
    trained_model = model.train(X_train, y_train)
    return trained_model
