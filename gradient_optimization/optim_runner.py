"""
Backprop on design parameters using trained neural network for evaluation
"""

import numpy as np
import torch
from torch.optim import lr_scheduler

from gradient_optimization.gradient_optimizer import GradientOptimizer, FrequencySweepOptimizer
from gradient_optimization.optim_constants import *
from neural_network.loss_tracker import LossTracker
from neural_network.nn_loader import load_neural_network

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load neural network
model, metadata, data_handler = load_neural_network(MODEL_NAME, MODEL_DIRECTORY)

# Initialise optimizer
input_dim = metadata['dimensions']['input']
num_params = input_dim - (1 if metadata['sweep_freq'] else 0) # remove one parameter for freq
params_scaled = torch.tensor(np.zeros(num_params).reshape(1, -1), dtype=torch.float32, requires_grad=True, device=device)
optimizer = torch.optim.Adam([params_scaled], lr=LEARNING_RATE)

grad_optim = (
    FrequencySweepOptimizer(
        model=model,
        data_handler=data_handler,
        freq_index=metadata['freq_index'],
        device=device
    )
) if metadata['sweep_freq'] else (
    GradientOptimizer(
        model=model,
        data_handler=data_handler,
        device=device
    )
)

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

params_scaled = grad_optim.limit_params(params_scaled)

# Training loop
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    loss = grad_optim.get_loss(params_scaled)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        params_scaled = grad_optim.limit_params(params_scaled)

        print(loss.item(), params_scaled)
        print()

        if loss_tracker(loss.item(), params_scaled):
            if PRINT_STATUS:
                print(f'Early stopping at epoch {epoch}')
            break

        if USE_SCHEDULER:
            scheduler.step(loss.item())

        if PRINT_STATUS and epoch % PRINT_INTERVAL == 0:
            lr = optimizer.param_groups[0]['lr']
            grad_optim.print_status(params_scaled, epoch, loss, lr)

# Compute final output
grad_optim.params_scaled = loss_tracker.load_best()
grad_optim.print_final_output(params_scaled)
