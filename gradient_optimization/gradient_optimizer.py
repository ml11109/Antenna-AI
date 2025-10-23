"""
Modules for handling optimizer parameters
"""

import torch

from gradient_optimization.optim_constants import *


class GradientOptimizer:
    def __init__(self, model, data_handler, params_scaled, device=None):
        self.model = model
        self.data_handler = data_handler
        self.params_scaled = params_scaled
        self.device = device

    def get_loss(self):
        # Loss = mean output
        outputs_scaled = self.model(self.params_scaled)
        outputs = self.data_handler.diff_scale(outputs_scaled, 'outputs', inverse=True)
        mean_output = torch.mean(outputs)
        return mean_output

    def get_param_limits(self, limits_dict=PARAM_LIMITS):
        limits = torch.tensor(limits_dict).t()
        limits_scaled = self.data_handler.scale(limits.reshape(2, -1), 'params')
        return limits_scaled

    def limit_params(self, limits_dict=PARAM_LIMITS, relative_constraints=RELATIVE_CONSTRAINTS):
        params_cloned = self.params_scaled.clone().detach()

        # Enforce absolute limits
        limits_scaled = self.get_param_limits(limits_dict)
        params_cloned = torch.clamp(params_cloned, min=limits_scaled[0, :], max=limits_scaled[1, :])

        # Enforce relative constraints
        eps = 1e-12
        for index1, index2, min_ratio, max_ratio in relative_constraints:
            p1_scaled = params_cloned[0, index1]
            p2_scaled = params_cloned[0, index2]
            p1 = self.data_handler.diff_scale(p1_scaled, 'params', index=index1, inverse=True)
            p2 = self.data_handler.diff_scale(p2_scaled, 'params', index=index1, inverse=True)

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

            new_p1_scaled = self.data_handler.diff_scale(new_p1, 'params', index=index1)
            new_p2_scaled = self.data_handler.diff_scale(new_p2, 'params', index=index2)

            params_cloned[0, index1] = new_p1_scaled
            params_cloned[0, index2] = new_p2_scaled

        self.params_scaled[:] = params_cloned

    def print_status(self, epoch, loss, lr):
        params = self.data_handler.diff_scale(self.params_scaled, 'params', inverse=True)
        params_list = [round(param.item(), 4) for param in params.flatten()]
        print(f'Epoch: {epoch}, Parameters: {params_list}, Loss: {loss.item(): .4f}, LR: {lr:.6f}')

    def print_final_output(self):
        params = self.data_handler.diff_scale(self.params_scaled, 'params', inverse=True)
        params_list = [round(param.item(), 4) for param in params.flatten()]
        loss = self.get_loss()
        print(f'\nParameters: {params_list}\nLoss: {loss.item():.6f}')


class FrequencySweepOptimizer(GradientOptimizer):
    def __init__(self, model, data_handler, params_scaled, freq_index, device=None):
        super().__init__(model, data_handler, params_scaled, device)
        self.freq_index = freq_index

    def insert_freq(self, params, freq_col):
        left = params[:, :self.freq_index]
        right = params[:, self.freq_index:]
        params_with_freq = torch.cat([left, freq_col.reshape(-1, 1), right], dim=1)
        return params_with_freq

    def get_loss(self, freq_range=FREQUENCY_RANGE, sweep_res=FREQUENCY_SWEEP_RES, tau=1e-2):
        # Loss = smooth maximum output across frequency sweep
        # tau: temperature parameter for smooth max

        # Get scaled frequency sweep points
        freqs = torch.linspace(freq_range[0], freq_range[1], sweep_res, device=self.device)
        freqs_scaled = self.data_handler.diff_scale(freqs, 'freq')

        # Replicate params for each frequency and insert freq column
        freq_col = freqs_scaled.view(sweep_res, 1)
        params_rep = self.params_scaled.repeat(sweep_res, 1)
        params_with_freq = self.insert_freq(params_rep, freq_col)

        # Model outputs for each frequency
        outputs = self.model(params_with_freq)
        per_freq_output = outputs.view(sweep_res, -1).mean(dim=1)

        # Smooth maximum via logsumexp, then unscale
        smooth_max_scaled = tau * torch.logsumexp(per_freq_output / tau, dim=0)
        smooth_max = self.data_handler.diff_scale(smooth_max_scaled, 'outputs', inverse=True)

        return smooth_max

    def get_param_limits(self, limits_dict=PARAM_LIMITS):
        limits = self.insert_freq(torch.tensor(limits_dict).t(), torch.tensor([0, 0]))
        params_mask = [i != self.freq_index for i in range(self.params_scaled.size()[1] + 1)]
        limits_scaled = self.data_handler.scale(limits.reshape(2, -1), 'params')[:, params_mask]
        return limits_scaled

