"""
Defines the neural network model for antenna simulation output prediction
"""

import torch
from pathlib import Path
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, use_dropout, name, directory):
        super(NeuralNetwork, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential()

        # Input layer
        self.layers.add_module('linear_input', nn.Linear(input_dim, hidden_dim[0]))
        self.layers.add_module('relu_input', nn.ReLU())
        if self.use_dropout:
            self.layers.add_module('dropout_input', nn.Dropout(self.dropout_rate))

        # Hidden layers
        for i in range(len(hidden_dim) - 1):
            self.layers.add_module(f'linear_hidden{i+1}', nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.layers.add_module(f'relu_hidden{i+1}', nn.ReLU())
            if self.use_dropout:
                self.layers.add_module(f'dropout_hidden{i+1}', nn.Dropout(self.dropout_rate))

        # Output layer
        self.layers.add_module('linear_output', nn.Linear(hidden_dim[-1], output_dim))

        self.name = name
        self.directory = Path(directory)

    def forward(self, x):
        return self.layers(x)

    def save(self):
        model_path = self.directory / (self.name + '.pth')
        self.directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), model_path)
