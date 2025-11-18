"""
Constants for integrated workflow
"""

OUTPUT = 's'
MODEL_NAME = f'{OUTPUT}_model'
DATA_NAME = f'{OUTPUT}_data'

NN_DIRECTORY = 'neural_network/models'
DATA_DIRECTORY = 'data/cleaned_data'
SCALER_DIRECTORY = 'data/scalers'
GRAPH_DIRECTORY = 'graph/saved_graphs'

MODELS = ['neuralnet', 'xgboost', 'lightgbm', 'catboost']
MODEL = 'all'

SIZE = 155
FREQUENCY_SWEEP_RES = 100 # number of steps to sweep
FREQUENCY_RANGE = (250, 800)
SWEEP_FREQUENCY = OUTPUT == 's'
FREQUENCY_INDEX = 0

PARAM_LIMITS = (
    # min, max
    (0, SIZE / 2 ** 0.5), # gap
    (0, 100000),    # gap_width
    (0, 100000),    # gap_length
    (0, 100000),    # finger_width
    (0, 100000)     # rect_breadth
)

RELATIVE_CONSTRAINTS = [
    # index1, index2, min_ratio, max_ratio
    (2, 3, 0, 1) # gap length < finger width
]

GRAPH_RESOLUTION = 100000  # number of points in graph
GRAPH_RANGE = (150, 900) # frequency range to plot
USED_FREQUENCY_RANGE = (250, 800)
NUM_RANDOMS = 2
