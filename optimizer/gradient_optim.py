"""
Backprop on design parameters using trained neural network for evaluation
"""

import numpy as np
import torch
from torch.optim import lr_scheduler

from neural_network.early_stopping import EarlyStopping
from optimizer.optim_constants import *
from neural_network.model_loader import load_neural_network

# Load neural network
model, metadata, data_handler = load_neural_network(MODEL_NAME, MODEL_DIRECTORY)
input_dim = metadata['dimensions']['input']

# Initialize design parameters
params_scaled = torch.tensor(np.zeros(input_dim).reshape(1, -1), dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([params_scaled], lr=LEARNING_RATE)

early_stopping = None
if USE_EARLY_STOPPING:
    early_stopping = EarlyStopping(
        patience=STOPPER_PATIENCE,
        min_delta=STOPPER_MIN_DELTA,
        restore_best_weights=True
    )

scheduler = None
if USE_SCHEDULER:
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR
    )

# Get scaler parameters
mean_y = torch.tensor(data_handler.scaler_y.mean_, dtype=torch.float32)
std_y = torch.tensor(data_handler.scaler_y.scale_, dtype=torch.float32)

def get_loss(inputs_scaled):
    # Loss = mean unscaled output
    outputs_scaled = model(inputs_scaled)
    outputs_unscaled = outputs_scaled * std_y + mean_y
    loss = torch.mean(outputs_unscaled)
    return loss

def limit_params(params_scaled):
    # Limit scaled params to be between the min scaled and max scaled
    limits = torch.tensor(PARAM_LIMITS)
    min_scaled = torch.tensor(data_handler.scale_x(limits[:, 0].reshape(1, -1)), dtype=torch.float32)
    max_scaled = torch.tensor(data_handler.scale_x(limits[:, 1].reshape(1, -1)), dtype=torch.float32)
    params_scaled = torch.clamp(params_scaled, min=min_scaled, max=max_scaled)
    return params_scaled

# Training loop
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    loss = get_loss(params_scaled)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        params_scaled[:] = limit_params(params_scaled)

    if early_stopping and early_stopping(loss, model):
        print(f'Early stopping at epoch {epoch}')
        break

    if scheduler:
        scheduler.step(loss)

    if epoch % 100 == 0:
        outputs = data_handler.inverse_scale_y(model(params_scaled).detach().numpy())
        params_unscaled = data_handler.inverse_scale_x(params_scaled.detach().numpy())
        print(f'Epoch: {epoch}, '
              f'Parameters: {[round(param.item(), 4) for param in params_unscaled.flatten()]}, '
              f'Output: {loss.item(): .4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

# Compute final output
outputs = data_handler.inverse_scale_y(model(params_scaled).detach().numpy())
params_unscaled = data_handler.inverse_scale_x(params_scaled.detach().numpy())
print('Parameters:', [round(param.item(), 4) for param in params_unscaled.flatten()])
print('Output:', f'{get_loss(params_scaled).item(): .4f}')
