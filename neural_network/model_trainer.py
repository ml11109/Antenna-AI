"""
Trains a neural network to predict antenna simulation outputs based on input parameters
"""

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

from data.data_handler import DataHandler
from neural_network.loss_tracker import LossTracker
from neural_network.model import NeuralNetwork
from neural_network.nn_constants import *

def train_nn():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    data_handler = DataHandler(sweep_freq=SWEEP_FREQUENCY, freq_index=FREQUENCY_INDEX, device=device, tensors=True)
    X_data, y_data = data_handler.load_data()
    X_temp, X_test, y_temp, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
    train_dataset, val_dataset, test_dataset = [
        TensorDataset(data[0], data[1])
        for data in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    ]
    train_loader, val_loader, test_loader = [
        DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=(dataset == train_dataset))
        for dataset in [train_dataset, val_dataset, test_dataset]
    ]

    # Model definition
    model = NeuralNetwork(
        input_dim=NN_CONSTANTS['dimensions']['input'],
        hidden_dim=NN_CONSTANTS['dimensions']['hidden'],
        output_dim=NN_CONSTANTS['dimensions']['output'],
        dropout_rate=NN_CONSTANTS['dropout']['rate'],
        use_dropout=NN_CONSTANTS['dropout']['use']
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=NN_LEARNING_RATE)

    loss_tracker = LossTracker(
        early_stop=NN_CONSTANTS['early_stopping']['use'],
        patience=NN_CONSTANTS['early_stopping']['patience'],
        min_delta=NN_CONSTANTS['early_stopping']['min_delta']
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=NN_CONSTANTS['lr_scheduler']['factor'],
        patience=NN_CONSTANTS['lr_scheduler']['patience'],
        min_lr=NN_CONSTANTS['lr_scheduler']['min_lr']
    )

    def train_epoch(loader):
        train_loss_sum = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_sum += loss.item()

        return train_loss_sum / len(loader)

    def get_test_loss(loader):
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
    for epoch in range(NN_EPOCHS):
        model.train()
        train_loss = train_epoch(train_loader)

        model.eval()
        val_loss = get_test_loss(val_loader)

        if loss_tracker(val_loss, model):
            if PRINT_STATUS:
                print(f'Early stopping at epoch {epoch}')
            break

        if NN_CONSTANTS['lr_scheduler']['use']:
            scheduler.step(val_loss)

        if PRINT_STATUS and epoch % NN_PRINT_INTERVAL == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}')

    # Compute test loss
    model.load_state_dict(loss_tracker.load_best())
    model.eval()
    test_loss = get_test_loss(test_loader)
    print(f'Test Loss: {test_loss:.6f}')

    model.save()
    data_handler.save_scalers()

    return model

train_nn()
