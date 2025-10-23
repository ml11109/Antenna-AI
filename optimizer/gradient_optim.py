"""
Backprop on design parameters using trained neural network for evaluation
"""

import numpy as np
import torch
from torch.optim import lr_scheduler

from neural_network.loss_tracker import LossTracker
from neural_network.model_loader import load_neural_network
from optimizer.optim_constants import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load neural network
model, metadata, data_handler = load_neural_network(MODEL_NAME, MODEL_DIRECTORY)
input_dim = metadata['dimensions']['input']

# Initialize design and training parameters
params_scaled = torch.tensor(np.zeros(input_dim - 1).reshape(1, -1), dtype=torch.float32, requires_grad=True, device=device)
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
mean_freq, std_freq = [
    torch.tensor(arr[FREQUENCY_INDEX], dtype=torch.float32, device=device)
    for arr in [data_handler.scaler_x.mean_, data_handler.scaler_x.scale_]
]
mean_x, std_x = [
    torch.tensor(np.delete(arr, FREQUENCY_INDEX), dtype=torch.float32, device=device)
    for arr in [data_handler.scaler_x.mean_, data_handler.scaler_x.scale_]
]
mean_y, std_y = [
    torch.tensor(arr, dtype=torch.float32, device=device)
    for arr in [data_handler.scaler_y.mean_, data_handler.scaler_y.scale_]
]

def insert_freq(params, freq_col, pos=FREQUENCY_INDEX):
    left = params[:, :pos]
    right = params[:, pos:]
    params_with_freq = torch.cat([left, freq_col.reshape(-1, 1), right], dim=1)
    return params_with_freq

def get_expected_max_output(params_scaled, tau=1e-2):
    """
    Differentiable approximation of the maximum output across a frequency sweep.
    Uses the smooth max (log-sum-exp) applied to the model outputs across frequencies.
    Lower `tau` -> closer to true max; watch for numerical issues for very small tau.
    Returns expected maximum unscaled output.
    """

    # Get scaled frequency sweep points
    freq_range = PARAM_LIMITS[FREQUENCY_INDEX]
    freqs = torch.linspace(freq_range[0], freq_range[1], FREQUENCY_SWEEP_RES, device=device)
    freqs_scaled = (freqs - mean_freq) / std_freq

    # Replicate params for each frequency and insert freq column
    freq_col = freqs_scaled.view(FREQUENCY_SWEEP_RES, 1)
    params_rep = params_scaled.repeat(FREQUENCY_SWEEP_RES, 1)
    params_with_freq = insert_freq(params_rep, freq_col)

    # Model outputs for each frequency
    outputs = model(params_with_freq)
    per_freq_output = outputs.view(FREQUENCY_SWEEP_RES, -1).mean(dim=1)

    # Smooth maximum via logsumexp, then unscale
    smooth_max_scaled = tau * torch.logsumexp(per_freq_output / tau, dim=0)
    smooth_max = smooth_max_scaled * std_y[0] + mean_y[0]

    return smooth_max

def limit_params(params_scaled):
    # Limit scaled params to be between the min scaled and max scaled
    limits = insert_freq(torch.tensor(PARAM_LIMITS).t(), torch.tensor([0, 0]))
    params_mask = [i != FREQUENCY_INDEX for i in range(input_dim)]
    min_scaled = data_handler.scale_x(limits[0, :].reshape(1, -1))[:, params_mask]
    max_scaled = data_handler.scale_x(limits[1, :].reshape(1, -1))[:, params_mask]
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
    loss = get_expected_max_output(params_scaled)

    if loss_tracker(loss.item(), params_scaled):
        if PRINT_STATUS:
            print(f'Early stopping at epoch {epoch}')
        break

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        params_scaled[:] = limit_params(params_scaled)

    if USE_SCHEDULER:
        scheduler.step(loss.item())

    if PRINT_STATUS and epoch % PRINT_INTERVAL == 0:
        params = data_handler.inverse_scale_x(insert_freq(params_scaled, torch.tensor(0)))
        params_list = [round(param.item(), 4) for param in params.flatten()][1:]
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Parameters: {params_list}, Loss: {loss.item(): .4f}, LR: {lr:.6f}')

# Compute final output
final_params_scaled = loss_tracker.load_best()
final_params = data_handler.inverse_scale_x(insert_freq(params_scaled, torch.tensor(0)))
final_params_list = [round(param.item(), 4) for param in final_params.flatten()][1:]
final_loss = get_expected_max_output(final_params_scaled)
print(f'\nParameters: {final_params_list}\nLoss: {final_loss.item():.6f}')
