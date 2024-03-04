import click
from pipelines.deployment_pipeline import continuous_deployment_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help=" - ",
)
@click.option(
    "--min-accuracy",
    type=click.FLOAT,
    default=0.8,
    help="Minimum f1 score required to deploy the model",
)
def run_deployment(config: str, min_accuracy: float):
    # getting the active deployed model
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continuous_deployment_pipeline(
            data_path="data/interim/sampled-raw-data.csv",
            min_accuracy=0.8,
            workers=3,
            timeout=60,
        )
    if predict:
        pass
    # print()
    # check existing services
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLFlow prediction server is running locally as daemon"
                f"process service and accepts inference request at:\n"
                f" {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green] `zenml model-deployer models delete`"
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"MLFlow prediction server is in a failed state:\n"
                f"Last State: '{service.status.state.value}'\n"
                f"Last Error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLFlow prediction server is currently running. The deployment"
            "pipeline must run first to train a model and deploy it. Execute"
            "the same command with the `--deploy` argument to deploy the model"
        )


if __name__ == "__main__":
    run_deployment()
