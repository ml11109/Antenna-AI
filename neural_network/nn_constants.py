"""
Constants for neural network training and testing
"""

from pathlib import Path

MODEL_NAME = 'john_5'
DATA_NAME = 'data_1'

MODEL_DIRECTORY = Path('models')
DATA_DIRECTORY = Path('data')
SCALER_DIRECTORY = Path('scalers')

# Training parameters
NUM_EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
INPUT_DIM = 6
HIDDEN_DIM = [128, 128, 128]
OUTPUT_DIM = 1

# Early stopping parameters
EARLY_STOPPING = False
PATIENCE = 30
MIN_DELTA = 0.0001

# Learning rate scheduling parameters
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-8

# Dropout parameters
USE_DROPOUT = True
DROPOUT_RATE = 0.1
