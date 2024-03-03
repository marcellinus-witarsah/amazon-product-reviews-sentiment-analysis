import os
import sys
import pandas as pd
import numpy as np
from typing import Union
from sklearn.pipeline import Pipeline
from dotenv import find_dotenv, load_dotenv
from zenml import step

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.models.model import CountVectorizerMultinomialNB


@step
def train_model(
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> Pipeline:
    return CountVectorizerMultinomialNB().train(X_train=X_train, y_train=y_train)
