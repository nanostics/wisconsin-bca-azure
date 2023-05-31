'''
Prepares raw data and provides training, validation and test datasets

See also: https://github.com/Azure/mlops-v2-gha-demo/blob/bd61f8c79c7c4cf4e30fe83a88e2b0532996ef33/data-science/src/prep.py
'''

from pathlib import Path
import mlflow
from .. import constants
import numpy as np
import pandas as pd
from mldesigner import command_component, Input, Output 


# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2#define-component-using-python-function-1
@command_component(
    name="prep_data",
    version="1",
    display_name="Prep Data",
    description="Prepares raw data and provides training, validation and test datasets",
    environment={
        'conda_file': f'{Path(__file__).parent}/lib/prep/conda.yaml',
        'image': 'mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference',
    },
    code='..' # we need the entire lib/ folder
)
def prepare_data_component(
    raw_data: Input(type="uri_file"),
    train_data: Output(type="uri_folder"),
    val_data: Output(type="uri_folder"),
    test_data: Output(type="uri_folder")
):
    '''
    A code-version of what was in `pipeline.yml`
    '''
    mlflow.start_run()

    # ----------  Arguments ----------- #
    # --------------------------------- #

    lines = [
        f"Raw data path: {raw_data}",
        f"Train dataset output path: {train_data}",
        f"Val dataset output path: {val_data}",
        f"Test dataset path: {test_data}",

    ]

    for line in lines:
        print(line)

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    data = pd.read_parquet((Path(raw_data)))
    data = data[constants.NUMERIC_COLS + constants.CAT_NOM_COLS + constants.CAT_ORD_COLS + [constants.TARGET_COL]]

    # ------------- Split Data ------------- #
    # -------------------------------------- #

    # Split data into train, val and test datasets

    random_data = np.random.rand(len(data))

    # for
    msk_train = random_data < 0.7
    msk_test = random_data >= 0.7

    train = data[msk_train]
    test = data[msk_test]

    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    train.to_parquet((Path(train_data) / "train.parquet"))
    test.to_parquet((Path(test_data) / "test.parquet"))

    mlflow.end_run()