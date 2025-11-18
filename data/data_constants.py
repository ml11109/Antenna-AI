"""
Constants for data cleaning and handling
"""

OUTPUT = 's'
DATA_NAME = f'{OUTPUT}_data'

DATA_DIRECTORY = 'data/cleaned_data'
SCALER_DIRECTORY = 'data/scalers'

RAW_DATA_DIRECTORY = f'raw_data/{OUTPUT}'
CLEANED_DATA_PATH = f'cleaned_data/{OUTPUT}_data.csv'
DEFAULT_PARAMS_PATH = 'raw_data/design_1/defaults.csv'

SWEEP_FREQUENCY = OUTPUT == 's'
FREQUENCY_INDEX = 0
INPUT_DIM = 6 if SWEEP_FREQUENCY else 5
OUTPUT_DIM = 1
INPUT_INDICES = range(INPUT_DIM)
OUTPUT_INDICES = range(INPUT_DIM, INPUT_DIM + OUTPUT_DIM)

CONVERSIONS = {
    'GHz': 1000,
    'MHz': 1,
    'mm': 1
}

RENAME = {
    'Freq': 'freq',
    'dB(ActiveS(1:1))': 's',
    'avg(dB(ActiveS(1:1)))': 's_avg',
    'max(dB(ActiveS(1:1)))': 's_max'
}

INPUTS = (['freq'] if OUTPUT == 's' else []) + [
    'gap',
    'gap_width',
    'gap_length',
    'finger_width',
    'rect_breadth',
]

