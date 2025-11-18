"""
Constants for optimization
"""

OUTPUT = 's'
MODEL_NAME = f'{OUTPUT}_model'

# Gradient optimization parameters
OPTIM_EPOCHS = 3000
OPTIM_LEARNING_RATE = 0.01
TAU = 1 # temperature parameter for smooth max
PRINT_STATUS = True
OPTIM_PRINT_INTERVAL = 100

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
OPTIM_EARLY_STOPPING = {
    'use': False,
    'patience': 200,
    'min_delta': 0.0001
}

# Learning rate scheduling parameters
OPTIM_SCHEDULER = {
    'use': True,
    'patience': 100,
    'factor': 0.5,
    'min_lr': 1e-8
}
