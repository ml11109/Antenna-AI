"""
Constants for optimizer
"""

OUTPUT = 's'
MODEL_NAME = f'{OUTPUT}_model'

# Gradient optimization parameters
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
PRINT_STATUS = True
PRINT_INTERVAL = 100
INFINITY_THRESH = 1000

# Antenna parameters
SIZE = 155
FREQUENCY_SWEEP_RES = 100 # number of steps to sweep
FREQUENCY_RANGE = (250, 800)

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

# Early stopping parameters
USE_EARLY_STOPPING = False
STOPPER_PATIENCE = 200
STOPPER_MIN_DELTA = 0.0001

# Learning rate scheduling parameters
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-8
