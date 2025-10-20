"""
Loss tracker class for training neural networks
Saves best result and stops early on convergence
"""

from torch import Tensor
from torch.nn import Module


class LossTracker:
    def __init__(self, early_stop=True, patience=7, min_delta=0):
        """
        Args:
            early_stop (bool): Whether early stopping is enabled
            patience (int): How many epochs to wait after last time validation loss improved
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
        """
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_weights = None

    def __call__(self, val_loss, model: Module|Tensor):
        if self.best_loss is None or val_loss <= self.best_loss - self.min_delta:
            self.counter = 0
        else:
            self.counter += 1

        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_best(model)

        return self.early_stop and self.counter >= self.patience

    def save_best(self, model: Module|Tensor):
        """Saves model when validation loss decreases."""
        if isinstance(model, Module):
            self.best_weights = model.state_dict().copy()
        else:
            self.best_weights = model.clone()

    def load_best(self):
        return self.best_weights
