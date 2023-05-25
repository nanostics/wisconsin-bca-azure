'''
Updates the environment used for deployment.

Usage: Run from the root of the repository

```bash
PYTHONPATH="lib" python azure/env/update.py
```
'''
import deploy_helpers
import constants


if __name__ == '__main__':
    mlclient = deploy_helpers.get_mlclient()
    new_env = deploy_helpers.ml_environment(mlclient, local=True, localName=constants.ENV_NAME)
    print('Creating new environment:\n')
    print(new_env)

    mlclient.environments.create_or_update(new_env)
