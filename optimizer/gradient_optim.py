"""
Backprop on design parameters using trained neural network for evaluation
"""

import torch

from optimizer.constants import *
from neural_network.model_loader import load_neural_network

model, metadata, data_handler = load_neural_network(MODEL_NAME, MODEL_DIRECTORY)
input_dim = metadata['dimensions']['input']
output_dim = metadata['dimensions']['output']

design_params = data_handler.inverse_scale_x(torch.zeros(input_dim).unsqueeze(0))
print(design_params)
