"""
Constants for neural network training and testing
"""

OUTPUT = 's_max'
MODEL_NAME = f'{OUTPUT}_model'
DATA_NAME = f'{OUTPUT}_data'

MODEL_DIRECTORY = 'neural_network/models'
DATA_DIRECTORY = 'data/cleaned_data'
SCALER_DIRECTORY = 'data/scalers'

# Training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
PRINT_STATUS = True
PRINT_INTERVAL = 10

# Neural network parameters
SWEEP_FREQUENCY = OUTPUT == 's'
FREQUENCY_INDEX = 0
INPUT_DIM = 6 if SWEEP_FREQUENCY else 5
OUTPUT_DIM = 1
INPUT_INDICES = range(INPUT_DIM)
OUTPUT_INDICES = range(INPUT_DIM, INPUT_DIM + OUTPUT_DIM)
HIDDEN_DIM = [128, 128, 128]

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
