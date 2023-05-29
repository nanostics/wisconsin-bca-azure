# runs the prep-train-evaluate-register pipeline in Azure
# imports components from `lib/pipeline`

# relative imports from lib/deploy_helpers.py
import deploy_helpers as helper

# ref: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2
if __name__ == '__main__':
    client = helper.get_mlclient()
    

    print('Authenticated!')