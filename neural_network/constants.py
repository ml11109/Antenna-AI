"""
Constants for neural network training and testing
"""

MODEL_NAME = 'dropout_1'
DATA_NAME = 'data_1'

# Training parameters
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
INPUT_DIM = 6
HIDDEN_DIM = [128, 128]
OUTPUT_DIM = 1

# Early stopping parameters
EARLY_STOPPING = False
PATIENCE = 15
MIN_DELTA = 0.001

# Learning rate scheduling parameters
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6

# Dropout parameters
USE_DROPOUT = True
DROPOUT_RATE = 0.2

DATA_DIRECTORY = 'neural_network/data/'
SCALER_DIRECTORY = 'neural_network/scalers/'
MODEL_DIRECTORY = 'neural_network/models/'
