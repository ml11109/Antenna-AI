"""
Runs the trained neural network
"""

import numpy
import torch

from constants import *
from model_loader import ModelLoader

# Load model and data handler
model_loader = ModelLoader(MODEL_NAME, MODEL_DIRECTORY)
model = model_loader.load_model()
data_handler = model_loader.load_data_handler()

# Run model with user input
while True:
    try:
        input_data = input('Enter input data (comma-separated): ')
        if input_data == 'quit':
            break

        input_data = numpy.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
        input_data = data_handler.scale_x(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        output_data = model(input_data)
        output_data = data_handler.inverse_scale_y(output_data.detach().cpu().numpy())
        print(f'Output data: {output_data[0]}')

    except ValueError:
        print('Invalid input data. Please enter comma-separated numbers.')
