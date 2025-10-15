"""
Constants for optimizers
"""

from pathlib import Path

MODEL_NAME = 'john_5'
DATA_NAME = 'data_1'

PARAM_LIMITS = (
    (100, 1000), # frequency
    (0.1, 20), # gap
    (0.1, 20), # gap_width
    (0.1, 20), # gap_length
    (0.1, 20), # finger_width
    (0.1, 100) # rect_breadth
)

MODEL_DIRECTORY = Path('models')
DATA_DIRECTORY = Path('data')
SCALER_DIRECTORY = Path('scalers')

# Gradient optimization parameters
NUM_EPOCHS = 1000
LEARNING_RATE = 0.03
