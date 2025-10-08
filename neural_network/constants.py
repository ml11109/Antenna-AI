"""
Constants for neural network training and testing
"""

MODEL_NAME = 'early_stopping_model_3'
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

DATA_DIRECTORY = 'neural_network/data/'
SCALER_DIRECTORY = 'neural_network/scalers/'
MODEL_DIRECTORY = 'neural_network/models/'
