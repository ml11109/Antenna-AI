"""
Constants for neural network training and testing
"""

OUTPUT = 's'
MODEL_NAME = f'{OUTPUT}_model'
MODEL_DIRECTORY = 'neural_network/models'

# Training parameters
NN_EPOCHS = 10
NN_LEARNING_RATE = 0.001
BATCH_SIZE = 32
TEST_SIZE = 0.2
PRINT_STATUS = True
NN_PRINT_INTERVAL = 10

# Neural network parameters
SWEEP_FREQUENCY = OUTPUT == 's'
FREQUENCY_INDEX = 0
