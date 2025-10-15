"""
Loads and cleans data for neural network training
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from constants import *


class AntennaDataHandler:
    def __init__(self, data_name, data_directory=DATA_DIRECTORY, scaler_directory=SCALER_DIRECTORY):
        self.data_directory = data_directory
        self.scaler_directory = scaler_directory

        self.data_path = self.data_directory / (data_name + '.csv')
        self.scaler_x_path = self.scaler_directory / (data_name + '_x_scaler.pkl')
        self.scaler_y_path = self.scaler_directory / (data_name + '_y_scaler.pkl')

        self.scaler_x = None
        self.scaler_y = None
        self.new_scalers = None
        self.load_scalers()

    def load_data(self, input_dim, output_dim):
        df = pd.read_csv(self.data_path)
        x = df.iloc[:, :input_dim].values
        y = df.iloc[:, input_dim: input_dim + output_dim].values

        if self.new_scalers:
            x = self.scaler_x.fit_transform(x)
            y = self.scaler_y.fit_transform(y)
            self.save_scalers()

        else:
            x = self.scaler_x.transform(x)
            y = self.scaler_y.transform(y)

        return x, y

    def scale_x(self, x):
        return self.scaler_x.transform(x)

    def scale_y(self, y):
        return self.scaler_y.transform(y)

    def inverse_scale_x(self, x):
        return self.scaler_x.inverse_transform(x)

    def inverse_scale_y(self, y):
        return self.scaler_y.inverse_transform(y)

    def save_scalers(self):
        os.makedirs(self.scaler_directory, exist_ok=True)
        joblib.dump(self.scaler_x, self.scaler_x_path)
        joblib.dump(self.scaler_y, self.scaler_y_path)

    def load_scalers(self):
        try:
            self.scaler_x = joblib.load(self.scaler_x_path)
            self.scaler_y = joblib.load(self.scaler_y_path)
            self.new_scalers = False

        except FileNotFoundError:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
            self.new_scalers = True
