'''
Updates the environment used for deployment.

Usage: Run from the root of the repository

```bash
PYTHONPATH="lib" python azure/env/update.py
```
'''
import constants
import deploy_helpers

from azure.ai.ml.entities import Environment


if __name__ == '__main__':
    mlclient = deploy_helpers.get_mlclient()
    new_env = Environment(
        name=constants.ENV_NAME,
        conda_file='environment.yml',
        image='mcr.microsoft.com/azureml/sklearn-0.24.1-ubuntu18.04-py37-cpu-inference:latest'
    )
    print('Creating new environment:\n')
    print(new_env)

    mlclient.environments.create_or_update(new_env)
