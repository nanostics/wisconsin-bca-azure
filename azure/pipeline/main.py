# runs the prep-train-evaluate-register pipeline in Azure
# imports components from `lib/pipeline`

# relative imports from lib/deploy_helpers.py
import deploy_helpers as helper

import constants
from pipeline.prep import prepare_data_component

from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline

# define a pipeline
@pipeline(
    default_compute='cpu-cluster',
)
def wisconsin_bca_pipeline(input_data):
    '''
    E2E ML pipeline using the Wisconsin BCa Dataset
    '''
    prep_node = prepare_data_component(raw_data=input_data)


# ref: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2
if __name__ == '__main__':
    # create a pipeline
    wisconsin_bca_dataset = Input(path='azureml:wisconsin-bca-data:1')
    pipeline = wisconsin_bca_pipeline(input_data=wisconsin_bca_dataset)

    # Log into Azure 
    ml_client = helper.get_mlclient()
    ml_client.compute.get(constants.CPU_CLUSTER_TARGET)

    # submit pipeline job!
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline, experiment_name=constants.EXPERIMENT_NAME
    )

    ml_client.jobs.stream(pipeline_job.name)