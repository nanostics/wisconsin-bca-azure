from lib.deploy_helpers import get_mlclient, get_latest_model, create_ml_environment, create_or_update_deployment, create_or_update_endpoint, post_deployment


if __name__ == '__main__':
    mlclient = get_mlclient()
    model = get_latest_model(mlclient, local=False)
    environment = create_ml_environment()
    endpoint = create_or_update_endpoint(mlclient, local=False)
    create_or_update_deployment(mlclient, model, environment, endpoint, local=False)

    post_deployment(mlclient, local=False)
