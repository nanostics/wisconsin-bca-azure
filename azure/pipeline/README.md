# Azure Train

This folder holds the files used in the main Azure ML Pipeline (train-evaluate-register).

## Running

Running `main.py` locally is good for debugging. Make sure you are logged into the Azure CLI. Remember that this will spin up a Compute instance on Azure!

```bash
export SUBSCRIPTION_ID=2676283c-665b-4d2f-bf73-2c380edf47d9
export RESOURCE_GROUP=jnji-rg
export WORKSPACE_NAME=joshua_nanostics_ml

python azure/pipeline/main.py
```

## Resources

- Azure MLOps Example: https://github.com/Azure/mlops-v2-gha-demo
