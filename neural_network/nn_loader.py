"""
Load trained neural network
"""

import json
from pathlib import Path

import torch

from neural_network.data_handler import DataHandler
from neural_network.model import PredictorModel
from neural_network.nn_constants import *


class NeuralNetworkLoader:
    def __init__(self, model_name=MODEL_NAME, model_directory=MODEL_DIRECTORY):
        self.model_name = model_name
        self.model_directory = Path(model_directory)
        self.model_path = self.model_directory / (self.model_name + '.pth')
        self.metadata_path = self.model_directory / (self.model_name + '_metadata.json')

        self.metadata = None
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def load_model(self):
        input_dim, hidden_dim, output_dim = self.metadata['dimensions'].values()
        model = PredictorModel(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def load_metadata(self):
        return self.metadata

    def load_data_handler(self):
        data_name = self.metadata['data_name']
        sweep_freq = self.metadata['sweep_freq']
        freq_index = self.metadata['freq_index']
        data_handler = DataHandler(data_name, sweep_freq=sweep_freq, freq_index=freq_index)
        return data_handler

def load_neural_network(model_name=MODEL_NAME, model_directory=MODEL_DIRECTORY):
    model_loader = NeuralNetworkLoader(model_name, model_directory)
    return model_loader.load_model(), model_loader.load_metadata(), model_loader.load_data_handler()
