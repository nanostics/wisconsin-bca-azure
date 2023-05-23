# Endpoint Testing

Testing our online endpoint by running it locally, and then sending a request to it. 

`main.py` holds the logic with setting up the local endpoint, sending a request, and tearing it down.

## Locally Running

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest), and authenticated with `az login`


### Running

```bash
export SUBSCRIPTION_ID=2676283c-665b-4d2f-bf73-2c380edf47d9
export RESOURCE_GROUP=jnji-rg
export WORKSPACE_NAME=joshua_nanostics_ml

python tests/endpoint/main.py
```

**Note:** Running the main script on my M1 Mac fails, but running it on Github Actions works. I'm guessing it might be an error with my ARM machine, since the Docker image used by the endpoint is only for x86_64.

```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=6969): Max retries exceeded with url: /score (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0xffff92604700>: Failed to establish a new connection: [Errno 111] Connection refused'))
```

## Resources

- https://learn.microsoft.com/en-us/azure/machine-learning/v1/how-to-deploy-local?view=azureml-api-1