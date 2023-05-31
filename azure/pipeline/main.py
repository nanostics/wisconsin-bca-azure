# runs the prep-train-evaluate-register pipeline in Azure
# imports components from `lib/pipeline`

# relative imports from lib/deploy_helpers.py
import os

import constants
import deploy_helpers
import pipeline_components as components

from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

# define a pipeline
@pipeline(
    default_compute='cpu-cluster'
)
def wisconsin_bca_pipeline(input_data):
    '''
    E2E ML pipeline using the Wisconsin BCa Dataset
    '''
    prep_node = components.prepare_data(raw_data=input_data)
    train_node = components.train_data(train_data=prep_node.outputs.train_data)

    evaluate_node = components.evaluate_model(
        model_input=train_node.outputs.model_output,
        test_data=prep_node.outputs.test_data
    )



# ref: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2
if __name__ == '__main__':
    # create a pipeline
    wisconsin_bca_dataset = Input(type=AssetTypes.URI_FILE, path='azureml:wisconsin-bca-data:1')
    print('Dataset:', wisconsin_bca_dataset)
    pipeline = wisconsin_bca_pipeline(input_data=wisconsin_bca_dataset)

    # Log into Azure
    ml_client = deploy_helpers.get_mlclient()
    ml_client.compute.get(constants.CPU_CLUSTER_TARGET)

    print(f'PYTHONPATH env var: {os.environ.get("PYTHONPATH")}')

    # submit pipeline job!
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline, experiment_name=constants.EXPERIMENT_NAME
    )

    ml_client.jobs.stream(pipeline_job.name)
