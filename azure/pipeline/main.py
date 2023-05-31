# runs the prep-train-evaluate-register pipeline in Azure
# imports components from `lib/pipeline`

# relative imports from lib/deploy_helpers.py
import os

import lib.helpers as helper
from lib import constants
from lib.prep.main import prep

from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes


from mldesigner import command_component, Input as MLInput, Output as MLOutput
from pathlib import Path

# Components
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2#define-component-using-python-function-1
# Defining them here so Azure can access the `lib/` folder correctly
@command_component(
    name="prep_data",
    version="1",
    display_name="Prep Data",
    description="Prepares raw data and provides training, validation and test datasets",
    environment={
        'conda_file': f'{Path(__file__).parent}/lib/prep/conda.yaml',
        'image': 'mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
    }
)
def prepare_data_component(
    raw_data: MLInput(type="uri_file"),
    train_data: MLOutput(type="uri_folder"),
    val_data: MLOutput(type="uri_folder"),
    test_data: MLOutput(type="uri_folder")
):
    # https://github.com/Azure/azureml-examples/blob/d0d04883ec32508bfff5d455156f86a82b1cdcb7/sdk/python/jobs/pipelines/2e_image_classification_keras_minist_convnet/train/train_component.py

    prep(raw_data, train_data, val_data, test_data)


# define a pipeline
@pipeline(
    default_compute='cpu-cluster'
)
def wisconsin_bca_pipeline(input_data):
    '''
    E2E ML pipeline using the Wisconsin BCa Dataset
    '''
    prep_node = prepare_data_component(raw_data=input_data)


# ref: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2
if __name__ == '__main__':
    # create a pipeline
    wisconsin_bca_dataset = Input(type=AssetTypes.URI_FILE, path='azureml:wisconsin-bca-data:1')
    print('Dataset:', wisconsin_bca_dataset)
    pipeline = wisconsin_bca_pipeline(input_data=wisconsin_bca_dataset)

    # Log into Azure
    ml_client = helper.get_mlclient()
    ml_client.compute.get(constants.CPU_CLUSTER_TARGET)

    print(f'PYTHONPATH env var: {os.environ.get("PYTHONPATH")}')

    # submit pipeline job!
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline, experiment_name=constants.EXPERIMENT_NAME
    )

    ml_client.jobs.stream(pipeline_job.name)
