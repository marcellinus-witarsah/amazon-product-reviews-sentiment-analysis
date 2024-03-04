from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocess_data
from steps.train_model import train_model
from steps.evaluate import evaluate

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    """
    Class for deciding model deployment based on performance
    """

    min_accuracy = 0.8


@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    return accuracy >= config.min_accuracy


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.8,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    data = ingest_data(data_path=data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train, **{})
    accuracy_score, precision_score, recall_score, f1_score = evaluate(
        model, X_test, y_test
    )
    deploy_decision = deployment_trigger(f1_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_decision,
        workers=workers,
        timeout=timeout,
    )
