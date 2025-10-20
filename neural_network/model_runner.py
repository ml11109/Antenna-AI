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
        inputs = input(f'Enter input data (comma-separated, {input_dim} values): ')

        if inputs == 'quit':
            break

        inputs = np.array([float(x) for x in inputs.split(',')]).reshape(1, -1)

        if inputs.shape[1] != input_dim:
            raise ValueError

        inputs = data_handler.scale_x(inputs)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = model(inputs)
        outputs = data_handler.inverse_scale_y(outputs.detach().numpy())
        print(f'Outputs: {outputs[0]}')

    except ValueError:
        print('Invalid input. Please try again.')
