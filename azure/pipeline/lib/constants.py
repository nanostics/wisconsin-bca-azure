# constants relating to the Wisconsin BCa dataset

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

ENDPOINT_NAME='wisconsin-bca-endpoint'
MODEL_NAME='wisconsin-BCa-model'
ENV_NAME='wisconsin-bca-env'

CPU_CLUSTER_TARGET='cpu-cluster'
EXPERIMENT_NAME='wisconsin-bca'
