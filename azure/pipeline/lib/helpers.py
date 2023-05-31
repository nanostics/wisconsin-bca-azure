'''
# Helpers

Copies from lib/deploy_helpers.py, but I can't figure out how to run 
the pipline with linking the code to lib/
'''
import os
import sys

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

def get_envs() -> tuple[str, str, str]:
    '''
    Looks in the environment for the following variables:
    - `SUBSCRIPTION_ID`
    - `RESOURCE_GROUP`
    - `WORKSPACE_NAME`
    '''

    # type-safe function to get a single environment variable
    def get_single_env(env: str) -> str:
        e_val = os.environ.get(env)
        if e_val is None:
            print(f'Environment variable {env} is not set! Exiting')
            sys.exit(1)
        return e_val

    return (
        get_single_env('SUBSCRIPTION_ID'), 
        get_single_env('RESOURCE_GROUP'), 
        get_single_env('WORKSPACE_NAME')
    )


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
