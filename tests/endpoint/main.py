import os
import json

# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import AzureCliCredential

# CONSTANTS
ENDPOINT_NAME='wisconsin-bca-endpoint'
MODEL_NAME='wisconsin-BCa-model'


def get_envs() -> tuple[str, str, str]:
    '''
    Looks in the environment for the following variables:
    - SUBSCRIPTION_ID
    - RESOURCE_GROUP
    - WORKSPACE_NAME
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

    mlclient.models.download(MODEL_NAME, latest_model_version, download_path='.azure-tmp')
    return Model(path='.')


def create_ml_environment() -> Environment:
    # assume the python file is being run in the root of the repo
    env = Environment(
        name='local',
        conda_file='azure/azure-env.yml',
        image='mcr.microsoft.com/azureml/sklearn-0.24.1-ubuntu18.04-py37-cpu-inference:latest'
    )
    print(f'Environment made {env.name}')
    return env


def create_or_update_endpoint(mlclient: MLClient) -> ManagedOnlineEndpoint:
    '''
    creates endpoint if it doesn't exist, otherwise updates it
    '''
    try:
        return mlclient.online_endpoints.get(name=ENDPOINT_NAME)
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
        endpoint = mlclient.online_endpoints.begin_create_or_update(endpoint).result()

        return endpoint


def create_or_update_deployment(
        mlclient: MLClient, 
        model: Model, 
        endpoint: ManagedOnlineEndpoint, 
        environment: Environment, 
        local: bool = True
    ) -> tuple[str, str]:
    '''
    Creates model deployment if it doesn't exist, else updates it

    I kind of (?) use blue-green deployments here, but I'm not sure if it's the right way to do it
    https://docs.cloudfoundry.org/devguide/deploy-apps/blue-green.html
    In theory, this should reduce downtime. If the deployment fails, the traffic is still routed to the old deployment.
    '''
    # define an online deployment
    deployment = ManagedOnlineDeployment(
        name='local',
        endpoint_name=endpoint.name,
        model=model,
        environment=environment,
        # Compute instance list:
        # https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list?view=azureml-api-2
        instance_type='Standard_DS3_v2',
        instance_count=1
    )
    print(f'Initialized deployment {deployment.name}')
    deployment_result = mlclient.online_deployments.begin_create_or_update(
        deployment, local=True
    ).result()
    print(f'Created deployment {deployment_result.name}')


if __name__ == '__main__':
    mlclient = get_mlclient()
    model = get_latest_model(mlclient)
    environment = create_ml_environment()
    endpoint = create_or_update_endpoint(mlclient)
    create_or_update_deployment(mlclient, model, endpoint, environment)

    # check deployment
    endpoint = mlclient.online_endpoints.get(name=ENDPOINT_NAME, local=True)
    print(f'Local endpoint created at {endpoint.scoring_uri}')

    # Hardcoded data to pass to the service
    # https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/deploy-to-local/register-model-deploy-local.ipynb
    # "Test Web Service" section
    sample_data = [[ 1.06440132,  0.80405382,  1.4890773 ,  1.82441732,  2.12011885,
        1.62621261,  1.67428659,  2.99057306,  1.00350134,  3.10485027,
        2.83317058,  0.33334666,  1.20575539,  1.43235646,  1.31676276,
        0.79819672,  0.75820626,  0.60151745,  1.59568557,  1.56051245,
        0.06110285,  0.51173261,  1.04502958,  1.17239422,  0.59875684,
        0.57992707],
    [-0.5717181 , -2.31970186, -1.47899644, -1.05175858, -1.15248909,
        -1.15452931, -1.25625737, -1.1784716 , -0.91527307, -1.19667305,
        -0.85104217, -1.23083649, -1.21640805, -0.8682195 , -1.4359442 ,
        -0.62149239, -0.87326335, -0.82629925, -0.40398055, -0.36784753,
        -2.1137895 , -1.29354244, -1.13316074, -1.28738793, -0.75765468,
        -1.24889433]]

    prediction = mlclient.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        input_data=json.dumps({ 'data': sample_data }),
    )
    print(f'Prediction: {prediction}')

    logs = mlclient.online_deployments.get_logs(
        name='local',
        endpoint_name=ENDPOINT_NAME, 
        local=True, 
        lines=50
    )
    print(logs)

