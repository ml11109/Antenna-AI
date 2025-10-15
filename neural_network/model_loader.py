"""
Load trained neural network
"""

import json
import torch
from pathlib import Path

from neural_network.constants import *
from neural_network.data_handler import AntennaDataHandler
from neural_network.model import AntennaPredictorModel


class ModelLoader:
    def __init__(self, model_name=MODEL_NAME, model_directory=MODEL_DIRECTORY):
        self.model_directory = Path(__file__).resolve().parent.parent / model_directory
        self.model_path = self.model_directory / (model_name + '.pth')
        self.metadata_path = self.model_directory / (model_name + '_metadata.json')

        self.metadata = None
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def load_model(self):
        input_dim, hidden_dim, output_dim = self.metadata['dimensions'].values()
        model = AntennaPredictorModel(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def load_metadata(self):
        return self.metadata

    def load_data_handler(self):
        return AntennaDataHandler(self.metadata['data_name'])

def load_neural_network(model_name=MODEL_NAME, model_directory=MODEL_DIRECTORY):
    model_loader = ModelLoader(model_name, model_directory)
    return model_loader.load_model(), model_loader.load_metadata(), model_loader.load_data_handler()
