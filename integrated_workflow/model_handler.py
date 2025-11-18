"""
Model handler class for training and running models
"""

import catboost
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from neural_network.model import NeuralNetwork
from integrated_workflow.workflow_constants import *


class ModelHandler:
    def __init__(self, model_type, model):
        self.model_type = model_type
        self.model = model

    def set_model(self, model_type, model):
        self.model_type = model_type
        self.model = model

    def train_single_model(self, model_type):
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

        # TODO: train model
        test_loss = 0

        return model, test_loss

    def train_all_models(self):
        models, losses = {}, {}

        for model_type in MODELS:
            model, test_loss = self.train_single_model(model_type)
            models[model_type] = model
            losses[model_type] = test_loss

        return models, losses
