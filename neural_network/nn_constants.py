"""
Constants for neural network training and testing
"""

OUTPUT = 's'

MODEL_NAME = f'{OUTPUT}_model'
DATA_NAME = f'{OUTPUT}_data'

MODEL_DIRECTORY = 'models'
DATA_DIRECTORY = 'data/cleaned_data'
SCALER_DIRECTORY = 'scalers'

SWEEP_FREQUENCY = OUTPUT == 's'
NUM_PARAMS = 6 if SWEEP_FREQUENCY else 5
FREQUENCY_INDEX = 0

# Training parameters
NUM_EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
INPUT_PARAMS = range(NUM_PARAMS)
OUTPUT_PARAMS = [NUM_PARAMS]
HIDDEN_DIM = [128, 128, 128]
PRINT_STATUS = True
PRINT_INTERVAL = 10

# Early stopping parameters
USE_EARLY_STOPPING = False
STOPPER_PATIENCE = 30
STOPPER_MIN_DELTA = 0.0001

# Learning rate scheduling parameters
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 20
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-8

# Dropout parameters
USE_DROPOUT = True
DROPOUT_RATE = 0.1
