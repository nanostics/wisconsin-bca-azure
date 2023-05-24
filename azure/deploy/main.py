# relative imports from lib/deploy_helpers.py
import deploy_helpers as helper

if __name__ == '__main__':
    mlclient = helper.get_mlclient()
    model = helper.get_latest_model(mlclient, local=False)
    environment = helper.create_ml_environment(local=False)
    endpoint = helper.create_or_update_endpoint(mlclient, local=False)
    helper.create_or_update_deployment(mlclient, model, environment, endpoint, local=False)

    helper.post_deployment(mlclient, local=False)
