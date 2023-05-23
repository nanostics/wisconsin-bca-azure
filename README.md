# Wisconsin BCa on Azure

A repository to test developing, training and deploying a model to Azure using the Wisconsin BCa dataset. 

Preprocessing is done locally, with the ML pipeline in charge of training, registering and deploying the model.

## Local Deployment

We can use a local development server with Docker to test our endpoint. See code in `tests/endpoint`

- https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/deploy-to-local

## Azure Deployment

I'm going to use Github Actions for Azure deployment.

- https://learn.microsoft.com/en-us/azure/machine-learning/how-to-github-actions-machine-learning?view=azureml-api-2&tabs=userlevel