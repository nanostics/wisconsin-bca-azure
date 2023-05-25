# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training, validation and test datasets
"""

import argparse

from pathlib import Path
import numpy as np
import pandas as pd

import mlflow

TARGET_COL = "diagnosis_01"

NUMERIC_COLS = [
    'texture_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'fractal_dimension_mean',
    'radius_se',
    'texture_se',
    'perimeter_se',
    'area_se',
    'smoothness_se',
    'compactness_se',
    'concavity_se',
    'concave points_se',
    'symmetry_se',
    'fractal_dimension_se',
    'texture_worst',
    'perimeter_worst',
    'area_worst',
    'smoothness_worst',
    'compactness_worst',
    'concavity_worst',
    'concave points_worst',
    'symmetry_worst',
    'fractal_dimension_worst'
]

CAT_NOM_COLS = [
]

CAT_ORD_COLS = [
]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--val_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")

    return parser.parse_args()


def main(args):
    '''Read, split, and save datasets'''

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    data = pd.read_parquet((Path(args.raw_data)))
    data = data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS + [TARGET_COL]]

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

    train.to_parquet((Path(args.train_data) / "train.parquet"))
    test.to_parquet((Path(args.test_data) / "test.parquet"))



if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Val dataset output path: {args.val_data}",
        f"Test dataset path: {args.test_data}",

    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
