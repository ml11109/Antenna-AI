"""
Load trained neural network
"""

import json
from pathlib import Path

import torch

from neural_network.data_handler import DataHandler
from neural_network.model import PredictorModel
from neural_network.nn_constants import *

def load_model(model_name=MODEL_NAME, model_directory=MODEL_DIRECTORY):
    model_directory = Path(model_directory)

    metadata_path = model_directory / (model_name + '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    model_path = model_directory / (model_name + '.pth')
    input_dim, hidden_dim, output_dim = metadata['dimensions'].values()
    model = PredictorModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, metadata

def load_data_handler(data_name, data_directory=DATA_DIRECTORY, scaler_directory=SCALER_DIRECTORY, sweep_freq=False, freq_index=None):
    data_handler = DataHandler(data_name, data_directory, scaler_directory, sweep_freq, freq_index)
    return data_handler

def load_neural_network(model_name=MODEL_NAME, model_directory=MODEL_DIRECTORY, data_directory=DATA_DIRECTORY, scaler_directory=SCALER_DIRECTORY):
    model, metadata = load_model(model_name, model_directory)

    data_name = metadata['data_name']
    sweep_freq = metadata['sweep_freq']
    freq_index = metadata['freq_index']
    data_handler = load_data_handler(data_name, data_directory, scaler_directory, sweep_freq, freq_index)

    return model, metadata, data_handler
