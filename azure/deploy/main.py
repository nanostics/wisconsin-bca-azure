'''
Deploys the latest model to the Azure ML service as a managed online endpoint

A lot of the deploy code is very similar to testing a local endpoint, 
so much of the code is shared in `lib/deploy_helpers.py`.
'''

# relative imports from lib/deploy_helpers.py
import deploy_helpers as helper

if __name__ == '__main__':
    client = helper.get_mlclient()
    model = helper.get_latest_model(client, local=False)
    environment = helper.ml_environment(client, local=False)
    endpoint = helper.create_or_update_endpoint(client, local=False)
    curr_deployment_name = helper.create_or_update_deployment(client, model, environment, endpoint, local=False)

    # If there is a 0% endpoint, DELETE IT!
    traffic = endpoint.traffic
    print(f'Traffic after deployment: {traffic}')
    # since we will only be using blue and green deployments, we can just check these two
    for deploy_name in ['blue', 'green']:
        if deploy_name in traffic and traffic[deploy_name] == 0:
            print(f'Begin deleting deployment {deploy_name}')
            client.online_deployments.begin_delete(deploy_name, endpoint_name=endpoint.name)

    helper.post_deployment(client, curr_deployment_name, local=False)
