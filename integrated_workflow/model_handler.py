"""
Model handler class for training and running models
"""

import catboost
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from neural_network.model import NeuralNetwork
from constants import *


class ModelHandler:
    def __init__(self, model_type, model):
        self.model_type = model_type
        self.model = model

    def set_model(self, model_type, model):
        self.model_type = model_type
        self.model = model

    def train_single_model(self, model_type, data_handler):
        match model_type:
            case 'neuralnet':
                model = None
            case 'xgboost':
                model = xgb.XGBRegressor()
            case 'lightgbm':
                model = lgb.LGBMRegressor()
            case 'catboost':
                model = catboost.CatBoostRegressor()
            case 'linear' | _:
                model = LinearRegression()

        if model_type == 'neuralnet':
            data_handler.tensor = True

        else:
            data_handler.tensor = False
            X, y = data_handler.load_data()
        test_loss = 0

        return model, test_loss

    def train_all_models(self, data_handler):
        models, losses = {}, {}

        for model_type in MODELS:
            model, test_loss = self.train_single_model(model_type, data_handler)
            models[model_type] = model
            losses[model_type] = test_loss

        return models, losses
