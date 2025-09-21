"""
Defines the neural network model for antenna simulation output prediction
"""

from torch import nn


class AntennaPredictorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AntennaPredictorModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
