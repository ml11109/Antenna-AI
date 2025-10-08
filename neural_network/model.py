"""
Defines the neural network model for antenna simulation output prediction
"""

from torch import nn
from constants import USE_DROPOUT, DROPOUT_RATE


class AntennaPredictorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=DROPOUT_RATE, use_dropout=USE_DROPOUT):
        super(AntennaPredictorModel, self).__init__()

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

    def forward(self, x):
        return self.layers(x)
