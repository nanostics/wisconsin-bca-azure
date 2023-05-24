# Deployment

Deploying our model to Azure Managed Endpoints. Code is VERY similar to [tests/endpoint/main.py](../../tests/endpoint/main.py), with minor adjustments to make sure we don't deploy locally.

### Running

This file runs automatically on pushes to the `main` branch when related files are changed. See [.github/workflows/pipeline.yml](../../.github/workflows/pipeline.yml) for more details.

I DO NOT recommend you run this yourself, as this will affect the production endpoint. However, during testing, it might be useful.

Make sure you are in the root of the repository. 

```bash
export SUBSCRIPTION_ID=2676283c-665b-4d2f-bf73-2c380edf47d9
export RESOURCE_GROUP=jnji-rg
export WORKSPACE_NAME=joshua_nanostics_ml

PYTHONPATH="lib" python azure/deploy/main.py
```