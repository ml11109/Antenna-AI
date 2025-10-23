"""
Runs the trained neural network
"""

import numpy as np
import torch

from neural_network.nn_loader import load_neural_network

model, metadata, data_handler = load_neural_network()
input_dim = metadata['dimensions']['input']

while True:
    try:
        inputs = input(f'Enter input data (comma-separated, {input_dim} values): ')

        if inputs == 'quit':
            break

        inputs = np.array([float(x) for x in inputs.split(',')]).reshape(1, -1)

        if inputs.shape[1] != input_dim:
            raise ValueError

        inputs = data_handler.scale(inputs, 'params').to(torch.float32)
        outputs = data_handler.scale(model(inputs), 'outputs', inverse=True)
        print(f'Outputs: {outputs.item():.6f}')

    except ValueError:
        print('Invalid input. Please try again.')
