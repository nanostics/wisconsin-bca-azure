'''
Pipeline Components

I put this here (outside the pipeline folder) to fix import errors
'''
from pathlib import Path
from mldesigner import command_component, Input, Output

from pipeline.prep import prep
from pipeline.train import train
from pipeline.evaluate import evaluate

# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2#define-component-using-python-function-1
# Defining them here so Azure can access the `lib/` folder correctly
@command_component(
    name="prep_data",
    display_name="Prep Data",
    description="Prepares raw data and provides training, validation and test datasets",
    environment={
        'conda_file': f'{Path(__file__).parent}/pipeline/prep/conda.yaml',
        'image': 'mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
    },
    code='.' # view the entire lib folder
)
def prepare_data(
    raw_data: Input(type="uri_file"),
    train_data: Output(type="uri_folder"),
    val_data: Output(type="uri_folder"),
    test_data: Output(type="uri_folder")
):
    prep(raw_data, train_data, val_data, test_data)


@command_component(
    name="train_data",
    display_name="Train Data",
    description='Trains ML model using training dataset. Saves trained model.',
    environment={
        'conda_file': f'{Path(__file__).parent}/pipeline/train/conda.yaml',
        'image': 'mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
    }
)
def train_data(
    train_data: Input(type="uri_folder"),
    model_output: Output(type="uri_folder")
):
    train(train_data, model_output)

@command_component(
    name="evaluate",
    display_name="Evaluate Model",
    description='Read trained model and test dataset, evaluate model and save result',
    environment={
        'conda_file': f'{Path(__file__).parent}/pipeline/evaluate/conda.yaml',
        'image': 'mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
    }
)
def evaluate_model(
    model_input: Input(type="uri_folder"),
    test_data: Input(type="uri_folder"),
    evaluation_output: Output(type="uri_folder")
):
    evaluate("wisconsin-BCa-model", model_input, test_data, evaluation_output)
