# runs the prep-train-evaluate-register pipeline in Azure
# imports components from `lib/pipeline`

# relative imports from lib/deploy_helpers.py
import deploy_helpers as helper

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
    client = helper.get_mlclient()

    help(prepare_data_component)
    
    # create a pipeline
    wisconsin_bca_dataset = Input(type='uri_file', path='azureml:wisconsin-bca-data@latest')
    pipeline_job = wisconsin_bca_pipeline(input_data=wisconsin_bca_dataset)