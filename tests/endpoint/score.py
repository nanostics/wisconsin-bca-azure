# COPY OF deploy-local/score.py
# TODO: consolidate the files

import os
import joblib
import numpy as np

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType


def init():
    global MODEL
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(
        os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    # Deserialize the model file back into a sklearn model.
    MODEL = joblib.load(model_path)

'''
SAMPLE JSON:

{'texture_mean': 0.8432786469399629,
 'smoothness_mean': 0.2523641398103375,
 'compactness_mean': 2.3583159827998363,
 'concavity_mean': 2.531619520072832,
 'concave points_mean': 1.450069508719469,
 'symmetry_mean': 0.8703875539174302,
 'fractal_dimension_mean': 0.9008419873112651,
 'radius_se': -0.5387860955897966,
 'texture_se': -0.6053635175011898,
 'perimeter_se': 0.39563104081988226,
 'area_se': -0.3250993411860726,
 'smoothness_se': -0.029234593660855363,
 'compactness_se': 2.031565539613105,
 'concavity_se': 1.505019516553444,
 'concave points_se': 1.7980670710827722,
 'symmetry_se': -0.6809728679517403,
 'fractal_dimension_se': 0.7246074237668291,
 'texture_worst': 0.3718226142066996,
 'perimeter_worst': 0.6937804113864807,
 'area_worst': 0.018073841227546305,
 'smoothness_worst': 0.4187939453648082,
 'compactness_worst': 2.986439651446995,
 'concavity_worst': 3.208749505160136,
 'concave points_worst': 2.2127117293244543,
 'symmetry_worst': -0.06072067452440019,
 'fractal_dimension_worst': 1.816145326154702}
'''

# this is gotten from the `X_test` variable
input_sample = np.array([[-1.53587537, -1.59868622, -1.2644576 , -1.07047828, -1.01543024,
       -0.73519693, -0.39527791, -0.28990047,  0.45715098, -0.39584766,
       -0.36948983,  1.08027519, -0.87830952, -0.86517173, -0.65332827,
        0.93093931,  0.32816649, -1.53899135, -0.76896925, -0.66501272,
       -1.68158125, -1.27075786, -1.30537918, -1.38663635, -0.99976219,
       -0.79221752]])
output_sample = np.array([1])


@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = MODEL.predict(data)
        # You can return any JSON-serializable object.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
