'''Loads model, registers it if deply flag is True'''

import os
import json

from pathlib import Path
import mlflow

def register(model_name, model_path, evaluation_output, model_info_output_path):
    '''Loads model, registers it if deply flag is True'''

    with open((Path(evaluation_output) / "deploy_flag"), 'rb') as infile:
        deploy_flag = int(infile.read())
        
    mlflow.log_metric("deploy flag", int(deploy_flag))
    deploy_flag=1
    if deploy_flag==1:

        print("Registering ", model_name)

        # load model
        model =  mlflow.sklearn.load_model(model_path) 

        # log model using mlflow
        mlflow.sklearn.log_model(model, model_name)

        # register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f'runs:/{run_id}/{model_name}'
        mlflow_model = mlflow.register_model(model_uri, model_name)
        model_version = mlflow_model.version

        # write model info
        print("Writing JSON")
        dict = {"id": "{0}:{1}".format(model_name, model_version)}
        output_path = os.path.join(model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(dict, fp=of)

    else:
        print("Model will not be registered!")
