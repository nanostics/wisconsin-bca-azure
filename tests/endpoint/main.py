# relative imports from lib/deploy_helpers.py
import constants
import deploy_helpers as helper


if __name__ == '__main__':
    client = helper.get_mlclient()
    model = helper.get_latest_model(client, local=True)
    environment = helper.ml_environment(client, local=True)
    endpoint = helper.create_or_update_endpoint(client, local=True)
    deployment_name = helper.create_or_update_deployment(client, model, environment, endpoint, local=True)

    helper.post_deployment(client, deployment_name, local=True)

    # delete the endpoint
    print(f'Deleting endpoint {constants.ENDPOINT_NAME}')
    client.online_endpoints.begin_delete(name=constants.ENDPOINT_NAME, local=True)
