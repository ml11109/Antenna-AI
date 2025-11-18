"""
File handler class for saving and loading models and metadata
"""

import joblib
import json
from pathlib import Path

from constants import *


class FileHandler:
    def __init__(self, model_dir=MODEL_DIRECTORY):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model_type, model):
        filename = self.model_dir / f'{model_type}.joblib'
        joblib.dump(model, filename)

    def load_model(self, model_type):
        filename = self.model_dir / f'{model_type}.joblib'
        try:
            model = joblib.load(filename)
            return model

        except FileNotFoundError:
            return None

    def load_best_model(self):
        losses = self.load_losses()
        if not losses:
            return None, None

        model_type = min(losses, key=losses.get)
        model = self.load_model(model_type)
        return model_type, model

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
