"""
Trains a neural network to predict antenna simulation outputs based on input parameters
"""

import json
import os

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

from neural_network.nn_constants import *
from neural_network.data_handler import AntennaDataHandler
from neural_network.model import AntennaPredictorModel
from neural_network.loss_tracker import LossTracker

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
data_handler = AntennaDataHandler(DATA_NAME)
X_data, y_data = data_handler.load_data(INPUT_DIM, OUTPUT_DIM)
X_temp, X_test, y_temp, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model definition
model = AntennaPredictorModel(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    dropout_rate=DROPOUT_RATE,
    use_dropout=USE_DROPOUT
).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_tracker = LossTracker(
    early_stop=USE_EARLY_STOPPING,
    patience=STOPPER_PATIENCE,
    min_delta=STOPPER_MIN_DELTA
)

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE,
    min_lr=SCHEDULER_MIN_LR
)

def train_epoch(loader):
    train_loss_sum = 0.0
    for i, (x_batch, y_batch) in enumerate(loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_sum += loss.item()

    return train_loss_sum / len(loader)

def get_loss(loader):
    with torch.no_grad():
        loss_sum = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss_sum += loss.item()

        return loss_sum / len(loader)

# Training loop
epoch = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = train_epoch(train_loader)

    model.eval()
    val_loss = get_loss(val_loader)

    if loss_tracker(val_loss, model):
        if PRINT_STATUS:
            print(f'Early stopping at epoch {epoch}')
        break

    if scheduler:
        scheduler.step(val_loss)

    if PRINT_STATUS and epoch % PRINT_INTERVAL == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}')

# Compute test loss
model.load_state_dict(loss_tracker.load_best())
model.eval()
test_loss = get_loss(test_loader)
print(f'Test Loss: {test_loss:.6f}')

# Save model and metadata
model_directory = Path(__file__).resolve().parent.parent / MODEL_DIRECTORY
model_path = model_directory / (MODEL_NAME + '.pth')
metadata_path = model_directory / (MODEL_NAME + '_metadata.json')

os.makedirs(model_directory, exist_ok=True)
torch.save(model.state_dict(), model_path)
data_handler.save_scalers()

metadata = {
    'model_name': MODEL_NAME,
    'data_name': DATA_NAME,
    'results': {
        'test_loss': test_loss
    },
    'training_parameters': {
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    },
    'dimensions': {
        'input': INPUT_DIM,
        'hidden': HIDDEN_DIM,
        'output': OUTPUT_DIM
    },
    'early_stopping': {
        'use_early_stopping': USE_EARLY_STOPPING,
        'patience': STOPPER_PATIENCE,
        'min_delta': STOPPER_MIN_DELTA,
        'final_epoch': epoch
    },
    'learning_rate_scheduler': {
        'use_scheduler': USE_SCHEDULER,
        'patience': SCHEDULER_PATIENCE,
        'factor': SCHEDULER_FACTOR,
        'min_lr': SCHEDULER_MIN_LR
    },
    'dropout': {
        'use_dropout': USE_DROPOUT,
        'dropout_rate': DROPOUT_RATE
    }
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)
