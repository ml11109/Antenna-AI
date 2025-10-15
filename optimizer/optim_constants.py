"""
Constants for optimizers
"""

from pathlib import Path

MODEL_NAME = 'john_5'
DATA_NAME = 'data_1'

PARAM_LIMITS = (
    (0, 100000), # frequency
    (0, 100000), # gap
    (0, 100000), # gap_width
    (0, 100000), # gap_length
    (0, 100000), # finger_width
    (0, 100000) # rect_breadth
)

MODEL_DIRECTORY = Path('models')
DATA_DIRECTORY = Path('data')
SCALER_DIRECTORY = Path('scalers')

# Gradient optimization parameters
NUM_EPOCHS = 10000
LEARNING_RATE = 0.05

# Early stopping parameters
EARLY_STOPPING = False
PATIENCE = 30
MIN_DELTA = 0.0001

# Learning rate scheduling parameters
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 11
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-8
