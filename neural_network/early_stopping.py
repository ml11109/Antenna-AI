class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)

        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True

        return False

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()