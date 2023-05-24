# relative imports from lib/deploy_helpers.py
import deploy_helpers as helper

if __name__ == '__main__':
    client = helper.get_mlclient()
    model = helper.get_latest_model(client, local=False)
    environment = helper.ml_environment(client, local=False)
    endpoint = helper.create_or_update_endpoint(client, local=False)
    deployment_name = helper.create_or_update_deployment(client, model, environment, endpoint, local=False)

    helper.post_deployment(client, deployment_name, local=False)
