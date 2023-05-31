'''
Prepares raw data and provides training, validation and test datasets

See also: https://github.com/Azure/mlops-v2-gha-demo/blob/bd61f8c79c7c4cf4e30fe83a88e2b0532996ef33/data-science/src/prep.py
'''
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from .. import constants


def prep(raw_data, train_data, val_data, test_data):
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
