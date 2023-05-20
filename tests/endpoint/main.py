import os
import json

from azureml.core.webservice import LocalWebservice
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.model import Model

import sklearn


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

def get_workspace() -> Workspace:
    '''
    Returns a workspace object from the environment variables. 
    
    Authenticates from the Azure CLI
    
    Prerequisites:
    - installed azure-cli package
    - used az login command to log in to your Azure Subscription

    This means we don't need an interactive authentication, so we can run this in CI
    '''
    # this uses the azure authentication in the Azure CLI
    # https://github.com/Azure/MachineLearningNotebooks/blob/25baf5203afa9904d8a154a50143184497f7a52c/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb
    cli_auth = AzureCliAuthentication()

    subscription_id, resource_group, workspace_name = get_envs()
    ws = Workspace(subscription_id=subscription_id,
                   resource_group=resource_group,
                   workspace_name=workspace_name,
                   auth=cli_auth)
    print(f'Found workspace {ws.name} at location {ws.location}')

    return ws

def create_local_env() -> Environment:
    '''
    I think this creates an environment but doesn't upload it anywhere (which we want)

    https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/deploy-to-local/register-model-deploy-local.ipynb

    Note these packages are only for the score.py
    '''
    environment = Environment("LocalDeploy")
    environment.python.conda_dependencies.add_pip_package("inference-schema[numpy-support]")
    environment.python.conda_dependencies.add_pip_package("joblib")
    environment.python.conda_dependencies.add_pip_package(f"scikit-learn=={sklearn.__version__}")

    return environment

if __name__ == '__main__':
    ws = get_workspace()

    model = Model(ws, 'wisconsin-BCa-model')
    print(f'Model {model.name} version {model.version} loaded from {model.workspace.name} into path {model.get_model_path(model.name, _workspace=ws)}')
    print(os.listdir(model.get_model_path(model.name, _workspace=ws)))
    local_env = create_local_env()
    inference_config = InferenceConfig(entry_script="score.py", environment=local_env)

    deployment_config = LocalWebservice.deploy_configuration(port=6969)

    local_service = Model.deploy(
        workspace=ws,
        name='wisconsin-bca-local-service',
        models=[model],
        inference_config=inference_config,
        deployment_config = deployment_config)

    local_service.wait_for_deployment(show_output=True)
    print(f"Scoring URI is : {local_service.scoring_uri}")

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

    sample_input = json.dumps({
        'data': sample_data
    })
    res = local_service.run(sample_input)
    print(f"Result: {res}")

    local_service.delete()
