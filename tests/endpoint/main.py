import os
import json

# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint, OnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment
)
from azure.identity import AzureCliCredential

# CONSTANTS
ENDPOINT_NAME='wisconsin-bca-endpoint'
MODEL_NAME='wisconsin-BCa-model'


def get_envs() -> tuple[str, str, str]:
    '''
    Looks in the environment for the following variables:
    - `SUBSCRIPTION_ID`
    - `RESOURCE_GROUP`
    - `WORKSPACE_NAME`
    '''
    envs = ["SUBSCRIPTION_ID", "RESOURCE_GROUP", "WORKSPACE_NAME" ]
    for env in envs:
        if os.environ.get(env) is None:
            raise Exception(f"Environment variable {env} is not set.")
    return os.environ.get("SUBSCRIPTION_ID"), os.environ.get("RESOURCE_GROUP"), os.environ.get("WORKSPACE_NAME")


def get_mlclient() -> MLClient:
    '''
    Returns a workspace object from the environment variables. 
    
    Authenticates from the Azure CLI
    
    Prerequisites:
    - installed azure-cli package
    - used az login command to log in to your Azure Subscription

    This means we don't need an interactive authentication, so we can run this in CI
    '''
    # this uses the azure authentication in the Azure CLI
    cli_auth = AzureCliCredential()

    subscription_id, resource_group, workspace_name = get_envs()
    ml_client = MLClient(cli_auth, subscription_id, resource_group, workspace_name)

    return ml_client


def get_latest_model(mlclient: MLClient) -> Model:
    '''
    Returns the latest model (wisconsin-bca-model) from the workspace
    As this is local, we download the model first. Azure SDK Local endpoints do not support
    remote model files.

    # https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=python#deploy-and-debug-locally-by-using-local-endpoints
    '''
    # Let's pick the latest version of the model
    latest_model_version = max(
        [int(m.version) for m in mlclient.models.list(name=MODEL_NAME)]
    )

    mlclient.models.download(MODEL_NAME, str(latest_model_version), download_path='.azure-tmp')
    return Model(path='.azure-tmp')


def create_ml_environment() -> Environment:
    # assume the python file is being run in the root of the repo
    env = Environment(
        name='local',
        conda_file='azure/azure-env.yml',
        image='mcr.microsoft.com/azureml/sklearn-0.24.1-ubuntu18.04-py37-cpu-inference:latest'
    )
    print(f'Environment made {env.name}')
    return env


def create_or_update_endpoint(mlclient: MLClient) -> OnlineEndpoint:
    '''
    creates endpoint if it doesn't exist, otherwise updates it
    '''
    try:
        return mlclient.online_endpoints.get(name=ENDPOINT_NAME, local=True)
    except Exception as _:
        print(f'Endpoint {ENDPOINT_NAME} does not exist, creating it')
        # define an online endpoint
        endpoint = ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            description='A managed online endpoint for the wisconsin breast cancer dataset',
            auth_mode="key",
            tags={
                "training_dataset": "credit_defaults",
            }
        )
        endpoint = mlclient.online_endpoints.begin_create_or_update(endpoint, local=True).result()

        return endpoint


def create_or_update_deployment(
        mlclient: MLClient, 
        model: Model, 
        environment: Environment,
        endpoint: OnlineEndpoint
    ):
    '''
    Creates model deployment if it doesn't exist, else updates it

    I kind of (?) use blue-green deployments here, but I'm not sure if it's the right way to do it
    https://docs.cloudfoundry.org/devguide/deploy-apps/blue-green.html
    In theory, this should reduce downtime. If the deployment fails, the traffic is still routed to the old deployment.
    '''
    # define an online deployment
    deployment = ManagedOnlineDeployment(
        name='local',
        model=model,
        environment=environment,
        endpoint_name=endpoint.name,
        # Compute instance list:
        # https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list?view=azureml-api-2
        instance_type='Standard_DS3_v2',
        instance_count=1
    )
    print(f'Initialized deployment {deployment.name}')
    # somehow this is returning a `ManagedOnlineDeployment` instead of a ``LROPoller[OnlineDeployment]` as expected
    deployment_result = mlclient.online_deployments.begin_create_or_update(
        deployment, local=True
    )
    print(f'Created deployment {deployment.name}')


def post_deployment(mlclient: MLClient):
    '''
    What to run after the online endpoint has deployed
    '''
    # check deployment
    endpoint = mlclient.online_endpoints.get(name=ENDPOINT_NAME, local=True)
    print(f'Local endpoint created at {endpoint.scoring_uri}')

    # Passing hardcoded data to the endpoint to test it out
    # https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=python#invoke-the-local-endpoint-to-score-data-by-using-your-model
    prediction = mlclient.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        request_file='tests/endpoint/sample_data.json',
        local=True
    )
    print(f'Prediction: {prediction}')

    logs = mlclient.online_deployments.get_logs(
        name='local',
        endpoint_name=ENDPOINT_NAME, 
        local=True, 
        lines=50
    )
    print(logs)


if __name__ == '__main__':
    mlclient = get_mlclient()
    model = get_latest_model(mlclient)
    environment = create_ml_environment()
    endpoint = create_or_update_endpoint(mlclient)
    create_or_update_deployment(mlclient, model, environment, endpoint)

    post_deployment(mlclient)
