"""
Defines the neural network model for antenna simulation output prediction
"""

from torch import nn


class AntennaPredictorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AntennaPredictorModel, self).__init__()

        self.layers = nn.Sequential()
        self.layers.add_module('linear_input', nn.Linear(input_dim, hidden_dim[0]))
        self.layers.add_module('relu_input', nn.ReLU())

        for i in range(len(hidden_dim) - 1):
            self.layers.add_module(f'linear_hidden{i+1}', nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.layers.add_module(f'relu_hidden{i+1}', nn.ReLU())

        self.layers.add_module('linear_output', nn.Linear(hidden_dim[-1], output_dim))

    def forward(self, x):
        return self.layers(x)
