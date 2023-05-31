'''
Pipeline Components

I put this here (outside the pipeline folder) to fix import errors

Also note that the `code` argument in the `@command_component`
must be a fodler containing this file! I can't put this file inside 
`azure/pipeline`
'''
from pathlib import Path
from mldesigner import command_component, Input, Output

import pipeline

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
def prep(
    raw_data: Input(type="uri_file"),
    train_data: Output(type="uri_folder"),
    val_data: Output(type="uri_folder"),
    test_data: Output(type="uri_folder")
):
    pipeline.prep(raw_data, train_data, val_data, test_data)


@command_component(
    name="train_data",
    display_name="Train Data",
    description='Trains ML model using training dataset. Saves trained model.',
    environment={
        'conda_file': f'{Path(__file__).parent}/pipeline/train/conda.yaml',
        'image': 'mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
    }
)
def train(
    train_data: Input(type="uri_folder"),
    model_output: Output(type="uri_folder")
):
    pipeline.train(train_data, model_output)


@command_component(
    name="evaluate",
    display_name="Evaluate Model",
    description='Read trained model and test dataset, evaluate model and save result',
    environment={
        'conda_file': f'{Path(__file__).parent}/pipeline/evaluate/conda.yaml',
        'image': 'mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
    }
)
def evaluate(
    model_input: Input(type="uri_folder"),
    test_data: Input(type="uri_folder"),
    evaluation_output: Output(type="uri_folder")
):
    pipeline.evaluate("wisconsin-BCa-model", model_input, test_data, evaluation_output)


@command_component(
    name="register",
    display_name="Register Model",
    description='Loads model, registers it if deply flag is True',
    environment={
        'conda_file': f'{Path(__file__).parent}/pipeline/register/conda.yaml',
        'image': 'mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
    }
)
def register(
    model_path: Input(type="uri_folder"),
    evaluation_output: Input(type="uri_folder"),
    model_info_output_path: Output(type="uri_folder")
):
    pipeline.register("wisconsin-BCa-model", model_path, evaluation_output, model_info_output_path)
