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


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
loader = AntennaDataHandler(DATA_FILENAME, INPUT_DIM, OUTPUT_DIM)
X_data, y_data = loader.load_data()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=0)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss function, and optimizer
model = AntennaPredictorModel(INPUT_DIM, OUTPUT_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 20 == 0 and i % 20 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        print(f"Test Loss: {loss.item():.4f}")

# Save model
os.makedirs(MODEL_DIRECTORY, exist_ok=True)
torch.save(model.state_dict(), MODEL_DIRECTORY + MODEL_FILENAME)
loader.save_scalers()

# Save model metadata
metadata = {
    'input_dimensions': INPUT_DIM,
    'output_dimensions': OUTPUT_DIM,
    'data_filename': DATA_FILENAME,
    'model_filename': MODEL_FILENAME,
    'data_path': DATA_DIRECTORY + DATA_FILENAME,
    'model_path': MODEL_DIRECTORY + MODEL_FILENAME,
    'training_parameters': {
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }
}

metadata_filename = MODEL_FILENAME.replace('.pth', '_metadata.json')
with open(MODEL_DIRECTORY + metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=4)
