"""
First optimization workflow: train regressor on data then perform optimization
"""
import json

import joblib
from pathlib import Path

import numpy as np

import xgboost as xgb
import lightgbm as lgb
import catboost
from sklearn.linear_model import LinearRegression

from neural_network.model import NeuralNetwork
from workflow_constants import *

class LinearWorkflow:
    def __init__(self, data_handler, model_dir=MODEL_DIRECTORY):
        self.data_handler = data_handler
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_type = None
        self.params = None

    def save_model(self, model_type, model):
        model_filename = self.model_dir / f'{model_type}.joblib'
        joblib.dump(model, model_filename)

    def load_model(self, model_type):
        filename = self.model_dir / f'{model_type}.joblib'
        try:
            model = joblib.load(filename)
            return model
        except FileNotFoundError:
            return None

    def load_best_model(self):
        losses = self.load_losses()
        if losses:
            model_type = min(losses, key=losses.get)
            model = self.load_model(model_type)
            return model, model_type
        else:
            return None, None

    def save_losses(self, losses):
        losses_filename = self.model_dir / 'losses.json'
        with open(losses_filename, 'w') as file:
            json.dump(losses, file)

    def load_losses(self):
        losses_filename = self.model_dir / 'losses.json'
        try:
            with open(losses_filename, 'r') as file:
                losses = json.load(file)
        except FileNotFoundError:
            losses = {}
        return losses

    def _train_single_model(self, model_type):
        match model_type:
            case 'neuralnet':
                model = NeuralNetwork()
            case 'xgboost':
                model = xgb.XGBRegressor()
            case 'lightgbm':
                model = lgb.LGBMRegressor()
            case 'catboost':
                model = catboost.CatBoostRegressor()
            case 'linear' | _:
                model = LinearRegression()
                model_type = 'linear'

        # TODO: train model
        test_loss = 0

        self.model = model
        self.model_type = model_type
        self.save_model(model, model_type)

        losses = self.load_losses()
        losses[model_type] = test_loss
        self.save_losses(losses)

        return test_loss

    def _train_all_models(self):
        best_model = None
        best_model_type = None
        best_test_loss = np.inf
        losses = {}

        for model_type in MODELS:
            model, test_loss = self._train_single_model(model_type)
            if test_loss < best_test_loss:
                best_model = model
                best_model_type = best_model
                best_test_loss = test_loss
            losses[model_type] = test_loss

        self.model = best_model
        self.model_type = best_model_type
        self.save_losses(losses)

        return best_test_loss

    def train_model(self, model_type):
        if model_type == 'all':
            test_loss = self._train_all_models()
        else:
            test_loss = self._train_single_model(model_type)
        return test_loss

    def surrogate_optimize(self, param_limits=PARAM_LIMITS, rel_constraints=RELATIVE_CONSTRAINTS, sweep_freq=SWEEP_FREQUENCY,
                           freq_index=FREQUENCY_INDEX, freq_range=FREQUENCY_RANGE, sweep_res=FREQUENCY_SWEEP_RES):
        pass

    def simulation_optimize(self):
        pass

    def plot_graph(self, graph_res=GRAPH_RESOLUTION, graph_range=GRAPH_RANGE, freq_range=USED_FREQUENCY_RANGE,
                   save=False, num_randoms=2):
        pass
