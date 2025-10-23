"""
Constants for optimizer
"""

from math import sqrt
from pathlib import Path

MODEL_NAME = 'john_5'
DATA_NAME = 'data_1'

SIZE = 155
FREQUENCY_INDEX = 0
FREQUENCY_SWEEP_RES = 100 # number of steps to sweep
FREQUENCY_RANGE = (0, 1000)

PARAM_LIMITS = (
    # min, max
    (0, SIZE / sqrt(2)), # gap
    (0, 100000),    # gap_width
    (0, 100000),    # gap_length
    (0, 100000),    # finger_width
    (0, 100000)     # rect_breadth
)

RELATIVE_CONSTRAINTS = [
    # index1, index2, min_ratio, max_ratio
    (3, 4, 0, 1)    # gap length < finger width
]

MODEL_DIRECTORY = Path('models')
DATA_DIRECTORY = Path('data')
SCALER_DIRECTORY = Path('scalers')

PRINT_STATUS = True
PRINT_INTERVAL = 100

# Gradient optimization parameters
NUM_EPOCHS = 1000
LEARNING_RATE = 0.05

# Early stopping parameters
USE_EARLY_STOPPING = False
STOPPER_PATIENCE = 200
STOPPER_MIN_DELTA = 0.0001

# Learning rate scheduling parameters
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-8
