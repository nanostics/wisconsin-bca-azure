# relative imports from azure/deploy/main.py
from lib.deploy_helpers import get_mlclient, get_latest_model, create_ml_environment, create_or_update_deployment, create_or_update_endpoint, post_deployment

# CONSTANTS
ENDPOINT_NAME='wisconsin-bca-endpoint'
MODEL_NAME='wisconsin-BCa-model'


if __name__ == '__main__':
    client = get_mlclient()
    model = get_latest_model(client, local=True)
    environment = create_ml_environment()
    endpoint = create_or_update_endpoint(client, local=True)
    create_or_update_deployment(client, model, environment, endpoint, local=True)

    post_deployment(client, local=True)

    # delete the endpoint
    print(f'Deleting endpoint {ENDPOINT_NAME}')
    client.online_endpoints.begin_delete(name=ENDPOINT_NAME, local=True)
