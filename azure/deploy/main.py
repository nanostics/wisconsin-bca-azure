import os 

# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model
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
    '''
    # Let's pick the latest version of the model
    latest_model_version = max(
        [int(m.version) for m in mlclient.models.list(name=MODEL_NAME)]
    )

    model = mlclient.models.get(name=MODEL_NAME, version=latest_model_version)
    print(f'Got model {model.name} with version {model.version}')
    return model

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
            },
        )
        endpoint = mlclient.online_endpoints.begin_create_or_update(endpoint).result()

        return endpoint

def create_or_update_deployment(mlclient: MLClient, model: Model, endpoint: ManagedOnlineEndpoint):
    '''
    Creates model deployment if it doesn't exist, else updates it

    I kind of (?) use blue-green deployments here, but I'm not sure if it's the right way to do it
    https://docs.cloudfoundry.org/devguide/deploy-apps/blue-green.html
    In theory, this should reduce downtime. If the deployment fails, the traffic is still routed to the old deployment.
    '''
    # check the current traffic distribution
    print(f'Current traffic: {endpoint.traffic}')
    if 'blue' in endpoint.traffic and endpoint.traffic['blue'] == 100:
        deployment_name, instance_type = 'green', "Standard_F4s_v2"
    else:
        deployment_name, instance_type = 'blue', "Standard_DS3_v2"

    # define an online deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint.name,
        model=model,
        # Compute instance list:
        # https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list?view=azureml-api-2
        instance_type=instance_type,
        instance_count=1
    )
    print(f'Initialized deployment {deployment.name}')
    deployment_result = mlclient.online_deployments.begin_create_or_update(
        deployment
    ).result()
    print(f'Created deployment {deployment_result.name}')

    endpoint.traffic = {deployment_name: 100}
    mlclient.online_endpoints.begin_create_or_update(endpoint).result()

if __name__ == '__main__':
    mlclient = get_mlclient()
    model = get_latest_model(mlclient)
    endpoint = create_or_update_endpoint(mlclient)
    create_or_update_deployment(mlclient, model, endpoint)

    # check deployment
    endpoint = mlclient.online_endpoints.get(name=ENDPOINT_NAME)

    # print a selection of the endpoint's metadata
    print(
        f"Name: {endpoint.name}\nStatus: {endpoint.provisioning_state}\nDescription: {endpoint.description}"
    )

    # existing traffic details
    print(f'Endpoint traffic: {endpoint.traffic}')

    # Get the scoring URI
    print(f'Scoring URI: {endpoint.scoring_uri}')
