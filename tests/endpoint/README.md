# Endpoint Testing

Testing our online endpoint by running it locally, and then sending a request to it. 

`main.py` holds the logic with setting up the local endpoint, sending a request, and tearing it down.

## Locally Running

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest), and authenticated with `az login`


### Running

Make sure you are in the root of the repository.

```bash
export SUBSCRIPTION_ID=2676283c-665b-4d2f-bf73-2c380edf47d9
export RESOURCE_GROUP=jnji-rg
export WORKSPACE_NAME=joshua_nanostics_ml

PYTHONPATH="lib" python tests/endpoint/main.py
```


## Resources

- https://learn.microsoft.com/en-us/azure/machine-learning/v1/how-to-deploy-local?view=azureml-api-1