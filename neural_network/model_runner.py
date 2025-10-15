"""
Runs the trained neural network
"""

import numpy as np
import torch

from neural_network.nn_constants import *
from neural_network.model_loader import load_neural_network

model, metadata, data_handler = load_neural_network(MODEL_NAME, MODEL_DIRECTORY)
input_dim = metadata['dimensions']['input']

while True:
    try:
        input_data = input(f'Enter input data (comma-separated, {input_dim} values): ')

        if input_data == 'quit':
            break

        input_data = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)

        if input_data.shape[1] != input_dim:
            raise ValueError

        input_data = data_handler.scale_x(input_data)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        output_data = model(input_data)
        output_data = data_handler.inverse_scale_y(output_data.detach().numpy())
        print(f'Output data: {output_data[0]}')

    except ValueError:
        print('Invalid input. Please try again.')
