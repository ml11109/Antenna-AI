"""
Constants for integrated workflow
"""

NAME = 'design_1'
FREQUENCY_RANGE = (250, 800)

FREQUENCY_NAME = 'Freq'
OUTPUT_NAME = 'dB(ActiveS(1:1))'

PARAMS = [
    'gap',
    'gap_width',
    'gap_length',
    'finger_width',
    'rect_breadth'
]

PARAM_LIMITS = (
    # min, max
    (0, 100), # gap
    (0, 100), # gap_width
    (0, 100), # gap_length
    (0, 100), # finger_width
    (0, 100)  # rect_breadth
)

RELATIVE_CONSTRAINTS = [
    # param1, param2, min_ratio, max_ratio
    ('gap_length', 'finger_width', 0, 1) # gap length < finger width
]

UNIT_CONVERSIONS = {
    'GHz': 1000,
    'MHz': 1,
    'mm': 1
}

DIRECTORIES = {
    'models': 'integrated_workflow/models',
    'raw_data': 'data/raw_data',
    'data': 'data/cleaned_data',
    'scalers': 'data/scalers',
    'graphs': 'graph/saved_graphs'
}

FILENAMES = {
    'data': 'data.csv',
    'default_params': 'defaults.csv'
}

MODELS = ['neural_net', 'xgboost', 'lightgbm', 'catboost', 'linear']

NN_CONSTANTS = {
    'training': {
        'epochs': 500,
        'lr': 0.001,
        'batch_size': 32,
        'test_size': 0.2
    },
    'status': {
        'print': True,
        'interval': 10
    },
    'hidden_dim': [
        128, 128, 128
    ],
    'early_stopping': {
        'use': False,
        'patience': 30,
        'min_delta': 0.0001
    },
    'lr_scheduler': {
        'use': True,
        'patience': 20,
        'factor': 0.5,
        'min_lr': 1e-8
    },
    'dropout': {
        'use': True,
        'rate': 0.1
    }
}

GRAD_OPTIM_CONSTANTS = {
    'epochs': 3000,
    'lr': 0.01,
    'tau': 1,
    'sweep_resolution': 100,
    'status': {
        'print': True,
        'interval': 100
    }
}

GRAPH_CONSTANTS = {
    'resolution': 100000,
    'range': (150, 900)
}
