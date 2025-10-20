"""
Backprop on design parameters using trained neural network for evaluation
"""

import numpy as np
import torch
from torch.optim import lr_scheduler

from neural_network.loss_tracker import LossTracker
from optimizer.optim_constants import *
from neural_network.model_loader import load_neural_network
from optimizer.scipy_optim import data_handler

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
mean_x = torch.tensor(data_handler.scaler_x.mean_, dtype=torch.float32)
std_x = torch.tensor(data_handler.scaler_x.scale_, dtype=torch.float32)
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
    min_scaled = data_handler.scale_x(limits[:, 0].reshape(1, -1))
    max_scaled = data_handler.scale_x(limits[:, 1].reshape(1, -1))
    params_scaled = torch.clamp(params_scaled, min=min_scaled, max=max_scaled)

    # Enforce relative constraints
    eps = 1e-12
    for index1, index2, min_ratio, max_ratio in RELATIVE_CONSTRAINTS:
        p1_scaled = params_scaled[0, index1]
        p2_scaled = params_scaled[0, index2]
        p1 = p1_scaled * std_x[index1] + mean_x[index1]
        p2 = p2_scaled * std_x[index2] + mean_x[index2]

        # Get current ratio and clip to limits
        if p2.abs() < eps:
            curr_ratio = p1 / eps
        else:
            curr_ratio = p1 / p2
        ratio = torch.clamp(curr_ratio, min_ratio, max_ratio)

        # If already feasible, continue
        if torch.isclose(ratio, curr_ratio):
            continue

        # Project (p1, p2) onto line p1 = ratio * p2 with minimum distance
        new_p2 = (p1 * ratio + p2) / (1.0 + ratio * ratio)
        new_p1 = new_p2 * ratio

        new_p1_scaled = (new_p1 - mean_x[index1]) / std_x[index1]
        new_p2_scaled = (new_p2 - mean_x[index2]) / std_x[index2]

        params_scaled[0, index1] = new_p1_scaled
        params_scaled[0, index2] = new_p2_scaled

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
        params_unscaled = data_handler.inverse_scale_x(params_scaled)
        params = [round(param.item(), 4) for param in params_unscaled.flatten()]
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Parameters: {params}, Output: {loss.item(): .4f}, LR: {lr:.6f}')

# Compute final output
final_params_scaled = loss_tracker.load_best()
final_params_unscaled = data_handler.inverse_scale_x(final_params_scaled)
final_params = [round(param.item(), 4) for param in final_params_unscaled.flatten()]
final_output = data_handler.inverse_scale_y(model(final_params_scaled))
print(f'\nParameters: {final_params}\nOutput: {final_output.item(): .6f}')
