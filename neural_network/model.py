"""
Defines the neural network model for antenna simulation output prediction
"""

from torch import nn

from neural_network.constants import HIDDEN_DIM


class AntennaPredictorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AntennaPredictorModel, self).__init__()

        self.layers = nn.Sequential()
        self.layers.add_module('linear_input', nn.Linear(input_dim, HIDDEN_DIM[0]))
        self.layers.add_module('relu_input', nn.ReLU())

        for i in range(len(HIDDEN_DIM) - 1):
            self.layers.add_module(f'linear_hidden{i+1}', nn.Linear(HIDDEN_DIM[i], HIDDEN_DIM[i+1]))
            self.layers.add_module(f'relu_hidden{i+1}', nn.ReLU())

        self.layers.add_module('linear_output', nn.Linear(HIDDEN_DIM[-1], output_dim))

    def forward(self, x):
        return self.layers(x)
