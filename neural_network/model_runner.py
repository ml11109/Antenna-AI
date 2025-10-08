"""
Loads and runs the trained neural network
"""

import json

import numpy
import torch

from constants import *
from data_handler import AntennaDataHandler
from model import AntennaPredictorModel


# Load model metadata
metadata_filename = MODEL_FILENAME.replace('.pth', '_metadata.json')
with open(MODEL_DIRECTORY + metadata_filename, 'r') as f:
    metadata = json.load(f)

input_dim = metadata['input_dimensions']
output_dim = metadata['output_dimensions']

# Load model
model = AntennaPredictorModel(input_dim, output_dim)
model.load_state_dict(torch.load(MODEL_DIRECTORY + MODEL_FILENAME))
model.eval()

# Load data loader
loader = AntennaDataHandler(metadata['data_filename'], input_dim, output_dim)
loader.load_scalers()

# Run model with user input
while True:
    try:
        input_data = input('Enter input data (comma-separated): ')
        if input_data == 'quit':
            break

        input_data = numpy.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
        input_data = loader.scale_x(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        output_data = model(input_data)
        output_data = loader.inverse_scale_y(output_data.detach().cpu().numpy())
        print(f'Output data: {output_data[0]}')

    except ValueError:
        print('Invalid input data. Please enter comma-separated numbers.')
