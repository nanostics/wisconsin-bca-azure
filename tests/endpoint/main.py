import os

from azureml.core.webservice import LocalWebservice
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.model import Model

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

if __name__ == '__main__':
    ws = get_workspace()

    model = Model(ws, 'wisconsin-BCa-model')
    myenv = Environment.get(workspace=ws, name="wisconsin-bca-env")
    inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

    deployment_config = LocalWebservice.deploy_configuration(port=6789)

    local_service = Model.deploy(
        workspace=ws,
        name='wisconsin-bca-local-service',
        models=[model],
        inference_config=inference_config,
        deployment_config = deployment_config)

    local_service.wait_for_deployment(show_output=True)
    print(f"Scoring URI is : {local_service.scoring_uri}")