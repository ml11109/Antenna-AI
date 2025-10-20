"""
Backprop on design parameters using trained neural network for evaluation
"""

import numpy as np
import torch
from torch.optim import lr_scheduler

from neural_network.loss_tracker import LossTracker
from optimizer.optim_constants import *
from neural_network.model_loader import load_neural_network

# Load neural network
model, metadata, data_handler = load_neural_network(MODEL_NAME, MODEL_DIRECTORY)
input_dim = metadata['dimensions']['input']

# Initialize design and training parameters
params_scaled = torch.tensor(np.zeros(input_dim).reshape(1, -1), dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([params_scaled], lr=LEARNING_RATE)

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

with torch.no_grad():
    params_scaled[:] = limit_params(params_scaled)

# Training loop
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    loss = get_loss(params_scaled)

    if loss_tracker(loss.item(), params_scaled):
        print(f'Early stopping at epoch {epoch}')
        break

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        params_scaled[:] = limit_params(params_scaled)

    if USE_SCHEDULER:
        scheduler.step(loss.item())

    if PRINT_STATUS and epoch % PRINT_INTERVAL == 0:
        params_unscaled = data_handler.inverse_scale_x(params_scaled.detach().numpy())
        params = [round(param.item(), 4) for param in params_unscaled.flatten()]
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Parameters: {params}, Output: {loss.item(): .4f}, LR: {lr:.6f}')

# Compute final output
params_scaled = loss_tracker.load_best()
params_unscaled = data_handler.inverse_scale_x(params_scaled.detach().numpy())
params = [round(param.item(), 4) for param in params_unscaled.flatten()]
loss = get_loss(params_scaled)
print(f'\nParameters: {params}\nOutput: {loss.item(): .6f}')
