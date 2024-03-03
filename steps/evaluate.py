import os
import sys
import pandas as pd
import numpy as np
from typing_extensions import Annotated
from typing import Union, Tuple
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from dotenv import find_dotenv, load_dotenv
from zenml import step

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PROJECT_FOLDER"))
from src.evaluation import Accuracy, Precision, Recall, F1


@step
def evaluate(
    model: Union[ClassifierMixin, Pipeline],
    X_test: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_test: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> Tuple[
    Annotated[float, "accuracy_score"],
    Annotated[float, "precision_score"],
    Annotated[float, "recall_score"],
    Annotated[float, "f1_score"],
]:
    """
    Evaluate model performance

    Args:
        model (Union[ClassifierMixin, Pipeline]): Model or pipeline containing model
        X_test (Union[pd.DataFrame, pd.Series, np.ndarray]): Test data
        y_test (Union[pd.DataFrame, pd.Series, np.ndarray]): Test labels

    Returns:
        Tuple[float]: Evaluation scores
    """
    y_pred = model.predict(X_test["preprocessed_review_text"])

    accuracy = Accuracy()
    accuracy_score = accuracy.calculate_score(y_test, y_pred)

    precision = Precision()
    precision_score = precision.calculate_score(y_test, y_pred)

    recall = Recall()
    recall_score = recall.calculate_score(y_test, y_pred)

    f1 = F1()
    f1_score = f1.calculate_score(y_test, y_pred)

    return accuracy_score, precision_score, recall_score, f1_score
