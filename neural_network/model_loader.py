"""
Load trained neural network
"""

import json
import torch

from constants import *
from data_handler import AntennaDataHandler
from model import AntennaPredictorModel


class ModelLoader:
    def __init__(self, model_name=MODEL_NAME, model_directory=MODEL_DIRECTORY):
        self.model_name = model_name
        self.model_directory = model_directory

        self.metadata = None
        with open(self.model_directory + self.model_name + '_metadata.json', 'r') as f:
            self.metadata = json.load(f)

    def load_metadata(self):
        return self.metadata

    def load_model(self):
        input_dim, hidden_dim, output_dim = self.metadata['dimensions'].values()
        model = AntennaPredictorModel(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(self.model_directory + self.model_name + '.pth'))
        model.eval()
        return model

    def load_data_handler(self):
        return AntennaDataHandler(self.metadata['data_name'])