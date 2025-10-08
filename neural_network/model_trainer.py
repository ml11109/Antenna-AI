"""
Trains a neural network to predict antenna simulation outputs based on input parameters
"""

import json
import os

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from constants import *
from data_handler import AntennaDataHandler
from model import AntennaPredictorModel
from neural_network.early_stopping import EarlyStopping

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
loader = AntennaDataHandler(DATA_NAME)
X_data, y_data = loader.load_data(INPUT_DIM, OUTPUT_DIM)
X_temp, X_test, y_temp, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss function, and optimizer
model = AntennaPredictorModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True)

def train_epoch():
    train_loss_sum = 0.0
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_sum += loss.item()

    return train_loss_sum / len(train_loader)

# Compute test loss
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
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = train_epoch()

    model.eval()
    val_loss = get_loss(val_loader)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if EARLY_STOPPING and early_stopping(val_loss, model):
        print(f'Early stopping at epoch {epoch}')
        break

model.eval()
test_loss = get_loss(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Save model
os.makedirs(MODEL_DIRECTORY, exist_ok=True)
torch.save(model.state_dict(), MODEL_DIRECTORY + MODEL_NAME + '.pth')
loader.save_scalers()

# Save model metadata
metadata = {
    'model_name': MODEL_NAME,
    'data_name': DATA_NAME,
    'dimensions': {
        'input': INPUT_DIM,
        'hidden': HIDDEN_DIM,
        'output': OUTPUT_DIM
    },
    'files': {
        'data_filepath': DATA_DIRECTORY + DATA_NAME + '.csv',
        'model_filepath': MODEL_DIRECTORY + MODEL_NAME + '.pth',
        'scaler_x_filepath': SCALER_DIRECTORY + DATA_NAME + '_x_scaler.pkl',
        'scaler_y_filepath': SCALER_DIRECTORY + DATA_NAME + '_y_scaler.pkl'
    },
    'training_parameters': {
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    },
    'results': {
        'test_loss': test_loss
    }
}

with open(MODEL_DIRECTORY + MODEL_NAME + '_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
