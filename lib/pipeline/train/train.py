'''
Trains ML model using training dataset. Saves trained model.
'''
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pipeline import constants


def train(train_data, model_output):
    mlflow.start_run()

    # Read train data
    train_data = pd.read_parquet(Path(train_data))

    # Split the data into input(X) and output(y)
    y_train = train_data[constants.TARGET_COL]
    X_train = train_data[constants.NUMERIC_COLS + constants.CAT_NOM_COLS + constants.CAT_ORD_COLS]

    # Train a Random Forest Regression Model with the training set
    model = SVC(kernel='poly')

    # log model hyperparameters
    mlflow.log_param("model", "SVC")
    # mlflow.log_param("n_estimators", args.regressor__n_estimators)
    # mlflow.log_param("bootstrap", args.regressor__bootstrap)
    # mlflow.log_param("max_depth", args.regressor__max_depth)
    # mlflow.log_param("max_features", args.regressor__max_features)
    # mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    # mlflow.log_param("min_samples_split", args.regressor__min_samples_split)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=model_output)

    mlflow.end_run()
