"""
Loads and cleans data for neural network training
"""

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from constants import *


class AntennaDataHandler:
    def __init__(self, filename, input_dim, output_dim, scaler_x=MinMaxScaler(), scaler_y=StandardScaler()):
        self.filename = filename
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def load_data(self):
        try:
            df = pd.read_csv(DATA_DIRECTORY + self.filename)
            x = df.iloc[:, :self.input_dim].values
            y = df.iloc[:, self.input_dim: self.input_dim + self.output_dim].values

            x = self.scaler_x.fit_transform(x)
            y = self.scaler_y.fit_transform(y)

            return x, y

        except FileNotFoundError:
            print(f"File not found: {DATA_DIRECTORY}{self.filename}")
            return None, None

        except IndexError:
            print(f"Index error while loading data from {DATA_DIRECTORY}{self.filename}")
            return None, None

    def scale_x(self, x):
        return self.scaler_x.transform(x)

    def scale_y(self, y):
        return self.scaler_y.transform(y)

    def inverse_scale_x(self, x):
        return self.scaler_x.inverse_transform(x)

    def inverse_scale_y(self, y):
        return self.scaler_y.inverse_transform(y)

    def save_scalers(self):
        joblib.dump(self.scaler_x, MODEL_DIRECTORY + self.filename.replace('.csv', '_x_scaler.pkl'))
        joblib.dump(self.scaler_y, MODEL_DIRECTORY + self.filename.replace('.csv', '_y_scaler.pkl'))

    def load_scalers(self):
        try:
            self.scaler_x = joblib.load(MODEL_DIRECTORY + self.filename.replace('.csv', '_x_scaler.pkl'))
            self.scaler_y = joblib.load(MODEL_DIRECTORY + self.filename.replace('.csv', '_y_scaler.pkl'))

        except FileNotFoundError:
            print(f"File not found: {MODEL_DIRECTORY}{self.filename.replace('.csv', '_x_scaler.pkl')} or {MODEL_DIRECTORY}{self.filename.replace('.csv', '_y_scaler.pkl')}")
            self.scaler_x = MinMaxScaler()
            self.scaler_y = StandardScaler()
