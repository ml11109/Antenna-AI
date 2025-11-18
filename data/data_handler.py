"""
Loads data for neural network training
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from data.data_constants import *


class DataHandler:
    def __init__(self, data_name=DATA_NAME, data_directory=DATA_DIRECTORY, scaler_directory=SCALER_DIRECTORY,
                 sweep_freq=False, freq_index=None, device=None, tensors=False):
        self.data_name = data_name
        self.data_directory = Path(data_directory)
        self.scaler_directory = Path(scaler_directory)
        self.sweep_freq = sweep_freq
        self.freq_index = freq_index
        self.device = device
        self.tensors = tensors

        self.data_path = self.data_directory / (data_name + '.csv')
        self.scaler_x_path = self.scaler_directory / (data_name + '_x.pkl')
        self.scaler_y_path = self.scaler_directory / (data_name + '_y.pkl')

        self.scaler_x, self.scaler_y, self.new_scalers = self.load_scalers()

    def scale(self, data, data_type, inverse=False):
        if isinstance(data, Tensor):
            data = data.detach().numpy()

        match data_type:
            case 'x' | 'params' if not inverse:
                op = self.scaler_x.transform
            case 'y' | 'outputs' if not inverse:
                op = self.scaler_y.transform
            case 'x' | 'params' if inverse:
                op = self.scaler_x.inverse_transform
            case 'y' | 'outputs' if inverse:
                op = self.scaler_y.inverse_transform
            case _:
                raise ValueError(f'Invalid data_type: {data_type}')

        return torch.from_numpy(op(data)) if self.tensors else op(data)

    def diff_scale(self, data, data_type, index=None, inverse=False):
        match data_type:
            case 'x' | 'params' | 'freq':
                scaler = self.scaler_x
            case 'y' | 'outputs':
                scaler = self.scaler_y
            case _:
                raise ValueError(f'Invalid data_type: {data_type}')

        # Extract frequency from scaler params
        if self.sweep_freq:
            match data_type:
                case 'freq': # take only frequency
                    extract_freq = lambda arr: arr[self.freq_index]
                case 'x' | 'params': # remove frequency
                    extract_freq = lambda arr: np.delete(arr, self.freq_index)
                case _:
                    extract_freq = lambda arr: arr

        else:
            extract_freq = lambda arr: arr

        # If index is not none, extract element at index
        if index is not None:
            extract_index = lambda arr: extract_freq(arr)[index]
        else:
            extract_index = extract_freq

        mean = np.array(extract_index(scaler.mean_))
        std = np.array(extract_index(scaler.scale_))

        if self.tensors:
            mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
            std = torch.tensor(std, dtype=torch.float32, device=self.device)

        if inverse:
            return data * std + mean
        else:
            return (data - mean) / std

    def load_data(self, input_indices=INPUT_INDICES, output_indices=OUTPUT_INDICES):
        df = pd.read_csv(self.data_path)
        x = df.iloc[:, input_indices].values
        y = df.iloc[:, output_indices].values

        if self.new_scalers:
            x = self.scaler_x.fit_transform(x)
            y = self.scaler_y.fit_transform(y)
            self.save_scalers()

        else:
            x = self.scaler_x.transform(x)
            y = self.scaler_y.transform(y)

        if self.tensors:
            x = torch.Tensor(x)
            y = torch.Tensor(y)

        return x, y

    def save_scalers(self):
        self.scaler_directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler_x, self.scaler_x_path)
        joblib.dump(self.scaler_y, self.scaler_y_path)

    def load_scalers(self):
        try:
            scaler_x = joblib.load(self.scaler_x_path)
            scaler_y = joblib.load(self.scaler_y_path)
            new_scalers = False

        except FileNotFoundError:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            new_scalers = True

        return scaler_x, scaler_y, new_scalers
