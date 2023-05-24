'''
# DEPLOY HELPERS

A collection of helper functions to manage deploying a model to an Azure managed endpoint.
Supports both local and remote endpoints.
'''
import os
import time
import sys

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    OnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.ai.ml.exceptions import LocalEndpointInFailedStateError
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


def get_latest_model(mlclient: MLClient, local: bool) -> Model:
    '''
    Returns the latest model (wisconsin-bca-model) from the workspace
    '''
    # Let's pick the latest version of the model
    latest_model_version = max(
        [int(m.version) for m in mlclient.models.list(name=MODEL_NAME)]
    )

    if local:
        mlclient.models.download(MODEL_NAME, str(latest_model_version), download_path='.azure-tmp')
        # Not sure why our model has a nested folder for `wisconsin-BCa-model` lol
        return Model(path='.azure-tmp/wisconsin-BCa-model/wisconsin-BCa-model/model.pkl')

    model = mlclient.models.get(name=MODEL_NAME, version=str(latest_model_version))
    print(f'Got model {model.name} with version {model.version}')
    return model


def ml_environment(local: bool) -> Environment:
    '''
    Gets an environment object for the model

    If local, creates a new environment from the `environment.yml` file

    If remote, gets the "wisconsin-bca-env" environment from the workspace
    '''
    if not local:
        return Environment(name='wisconsin-bca-env')


    # For local environments, assume the python file is being run in the root of the repo
    env_name = 'local' if local else 'wisconsin-bca-env'
    env = Environment(
        name=env_name,
        conda_file='environment.yml',
        image='mcr.microsoft.com/azureml/sklearn-0.24.1-ubuntu18.04-py37-cpu-inference:latest'
    )
    print(f'Environment made {env.name}')
    return env


def create_or_update_endpoint(mlclient: MLClient, local: bool) -> OnlineEndpoint:
    '''
    creates endpoint if it doesn't exist, otherwise updates it
    '''
    try:
        endpoint = mlclient.online_endpoints.get(name=ENDPOINT_NAME, local=local)
        print(f'Endpoint {ENDPOINT_NAME} exists, reusing it')
        return endpoint
    except Exception as _:
        print(f'Endpoint {ENDPOINT_NAME} does not exist, creating it')
        # define an online endpoint
        endpoint = ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            description='A managed online endpoint for the wisconsin breast cancer dataset',
            auth_mode="key"
        )
        endpoint = mlclient.online_endpoints.begin_create_or_update(endpoint, local=local)

        if local:
            # pylint gives an error saying this function returns a `LROPoller[OnlineEndpoint]`, but it's wrong!
            # looking at how `begin_create_or_update` is defined for local runs,
            # if we pass `local=True`, it returns a `ManagedOnlineEndpoint`
            # We ignore the type error at the end of the line: https://github.com/microsoft/pylance-release/issues/196
            return endpoint #type: ignore
        else:
            # wait for the endpoint to be created
            endpoint = endpoint.result()
            print(f'Endpoint {endpoint.name} created')
            return endpoint


def create_or_update_deployment(
        mlclient: MLClient, 
        model: Model,
        env: Environment,
        endpoint: OnlineEndpoint,
        local: bool
    ) -> str:
    '''
    Creates model deployment if it doesn't exist, else updates it. Returns the deployment name to be used later on

    If `local=True`, we use local docker deployments

    I kind of (?) use blue-green deployments here, but I'm not sure if it's the right way to do it
    https://docs.cloudfoundry.org/devguide/deploy-apps/blue-green.html
    In theory, this should reduce downtime. If the deployment fails, the traffic is still routed to the old deployment.
    '''
    if local:
        deployment_name, instance_type = 'local', "Standard_DS3_v2"
    else:
        # check the current traffic distribution
        print(f'Current traffic: {endpoint.traffic}')
        if 'blue' in endpoint.traffic and endpoint.traffic['blue'] == 100:
            deployment_name, instance_type = 'green', "Standard_F4s_v2"
        else:
            deployment_name, instance_type = 'blue', "Standard_DS2_v2"

    # define an online deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint.name,
        model=model,
        environment=env,
        # Compute instance list:
        # https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list?view=azureml-api-2
        instance_type=instance_type,
        instance_count=1,
        code_configuration=CodeConfiguration(
            code='azure/deploy',
            scoring_script='score.py'
        )
    )
    print(f'Initialized deployment {deployment.name}')
    deployment = mlclient.online_deployments.begin_create_or_update(
        deployment, local=local
    )

    if not local:
        # wait for the deployment to finish
        # since this is a remote call, we might have to wait a bit
        deployment_result = deployment.result()
        print(f'Created deployment {deployment_result.name}')

        endpoint.traffic = {deployment_name: 100}
        mlclient.online_endpoints.begin_create_or_update(endpoint).result()

    return deployment_name


def post_deployment(mlclient: MLClient, deployment_name: str, local: bool):
    '''
    Running some checks and printing some metadata after the endpoint has been deployed
    '''
    # check deployment
    endpoint = mlclient.online_endpoints.get(name=ENDPOINT_NAME, local=local)
    # print endpoint metadata
    print(f'\n\n***ENDPOINT METADATA: {endpoint.name}***')
    print(f'Description: {endpoint.description}')
    print(f'Scoring URI: {endpoint.scoring_uri}')
    print(f'Auth mode: {endpoint.auth_mode}')
    print(f'Traffic: {endpoint.traffic}')
    print(f'State: {endpoint.provisioning_state}\n\n')


    while True:
        try:
            success = _predict_at_endpoint(mlclient, local)
            if success:
                break

            time.sleep(2)
        except LocalEndpointInFailedStateError:
            print('Endpoint is in a failed state! Printing logs')
            _print_logs(mlclient, local, deployment_name)
            sys.exit(1)

    print('Endpoint creation and testing success!')
    _print_logs(mlclient, local, deployment_name)


def _predict_at_endpoint(mlclient: MLClient, local: bool) -> bool:
    '''
    Passing hardcoded data to the endpoint to test it out. Returns `true` if the prediction is expected, `false` if we need to retry
    Retries happen when the endpoint is not ready to receive requests yet (502 Bad Gateway error)
    
    Reference: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=python#invoke-the-local-endpoint-to-score-data-by-using-your-model
    '''
    prediction = mlclient.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        request_file='tests/endpoint/sample_data.json',
        local=local
    )
    print(f'*** Prediction: {prediction}')
    if "502 Bad Gateway" in prediction:
         # What's weird is that the first few requests to the endpoint usually fail
        # with a 502 Bad Gateway error (at least on local deployments)
        # I feel like there should be a way to check if the endpoint is ready to receive requests, 
        # but I couldn't find anything
        # In the endpoint metadata (post_deployment function), it always shows the state as "succeeded"
        print('Bad gateway error, retrying...')
        return False
    elif prediction == "[0, 0]":
        return True
    else:
        print('Unexpected prediction result! Expecting [0, 0]')
        # exit with error
        sys.exit(1)


def _print_logs(mlclient: MLClient, local: bool, deployment_name: str):
    logs = mlclient.online_deployments.get_logs(
        name=deployment_name,
        endpoint_name=ENDPOINT_NAME,
        local=local,
        lines=50
    )
    print('\n\n=========== LOGS ===========')
    print(logs)
    print('=========== LOGS END ===========')