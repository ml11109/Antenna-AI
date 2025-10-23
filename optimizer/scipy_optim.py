from scipy.optimize import minimize
import numpy as np
import torch

from optimizer.optim_constants import *
from neural_network.model_loader import load_neural_network

# Load neural network and scaler parameters
model, metadata, data_handler = load_neural_network(MODEL_NAME, MODEL_DIRECTORY)
input_dim = metadata['dimensions']['input']
params_scaled = torch.tensor(np.zeros(input_dim).reshape(1, -1), dtype=torch.float32, requires_grad=True)

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

def scipy_optimize_with_constraints(params_scaled_init=None, maxiter=NUM_EPOCHS):
    eps = 1e-12
    n = input_dim

    # initial point (1D numpy)
    if params_scaled_init is None:
        x0 = params_scaled.detach().cpu().numpy().ravel()
    else:
        x0 = np.asarray(params_scaled_init).ravel()

    # bounds: convert PARAM_LIMITS (unscaled) to scaled bounds using data_handler.scale_x
    limits = np.asarray(PARAM_LIMITS)
    min_scaled = data_handler.scale_x(limits[:, 0].reshape(1, -1)).ravel()
    max_scaled = data_handler.scale_x(limits[:, 1].reshape(1, -1)).ravel()
    bounds = [(float(min_scaled[i]), float(max_scaled[i])) for i in range(n)]

    # objective and gradient using torch autograd
    def obj_and_grad(x):
        xt = torch.tensor(x.reshape(1, -1), dtype=torch.float32, requires_grad=True)
        loss = get_loss(xt)
        loss.backward()
        grad = xt.grad.detach().cpu().numpy().ravel().astype(float)
        return float(loss.item()), grad

    # helper to build constraint functions (value and jacobian)
    def make_ratio_constraint(i1, i2, ratio_target, sign=1):
        # sign=+1 -> constraint is (a/b - ratio_target) >= 0
        # sign=-1 -> constraint is (ratio_target - a/b) >= 0
        def fun(x):
            xt = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
            a = xt[0, i1] * std_x[i1] + mean_x[i1]
            b = xt[0, i2] * std_x[i2] + mean_x[i2]
            denom = b.clone()
            # avoid exact zero denom
            denom = torch.where(denom.abs() < eps, torch.sign(denom) * eps + eps, denom)
            val = (a / denom - ratio_target) if sign == 1 else (ratio_target - a / denom)
            return float(val.item())

        def jac(x):
            xt = torch.tensor(x.reshape(1, -1), dtype=torch.float32, requires_grad=True)
            a = xt[0, i1] * std_x[i1] + mean_x[i1]
            b = xt[0, i2] * std_x[i2] + mean_x[i2]
            denom = b.clone()
            denom = torch.where(denom.abs() < eps, torch.sign(denom) * eps + eps, denom)
            c = (a / denom - ratio_target) if sign == 1 else (ratio_target - a / denom)
            grads = torch.autograd.grad(c, xt)[0].detach().cpu().numpy().ravel().astype(float)
            return grads

        return {'type': 'ineq', 'fun': fun, 'jac': jac}

    # build constraints list
    constraints = []
    for i1, i2, min_r, max_r in RELATIVE_CONSTRAINTS:
        constraints.append(make_ratio_constraint(i1, i2, min_r, sign=1))
        constraints.append(make_ratio_constraint(i1, i2, max_r, sign=-1))

    # run SciPy minimize (SLSQP)
    res = minimize(fun=lambda x: obj_and_grad(x)[0],
                   x0=x0,
                   jac=lambda x: obj_and_grad(x)[1],
                   bounds=bounds,
                   constraints=constraints,
                   method='SLSQP',
                   options={'maxiter': maxiter, 'ftol': 1e-9, 'disp': True})

    # write result back into a torch tensor suitable for further use
    x_opt = torch.tensor(res.x.reshape(1, -1), dtype=torch.float32, requires_grad=True)
    return res, x_opt

# Example usage (replace the training loop):
result, params_scaled_opt = scipy_optimize_with_constraints()
print('SciPy result:', result.message)
params_unscaled = data_handler.inverse_scale_x(params_scaled_opt.detach().numpy())
print('Parameters:', [round(float(p), 6) for p in params_unscaled.flatten()])
print('Output:', get_loss(params_scaled_opt))
