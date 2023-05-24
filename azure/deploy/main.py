# relative imports from lib/deploy_helpers.py
import deploy_helpers as helper

if __name__ == '__main__':
    mlclient = helper.get_mlclient()
    model = helper.get_latest_model(mlclient, local=False)
    environment = helper.ml_environment(mlclient, local=False)
    endpoint = helper.create_or_update_endpoint(mlclient, local=False)
    deployment_name = helper.create_or_update_deployment(mlclient, model, environment, endpoint, local=False)

    helper.post_deployment(mlclient, deployment_name, local=False)
