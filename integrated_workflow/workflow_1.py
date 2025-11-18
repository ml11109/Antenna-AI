"""
First optimization workflow: train regressor on data then perform optimization
"""

from workflow_constants import *

class LinearWorkflow:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.model = None
        self.params = None

    def train_model(self, model_type='all', train_nn=False):
        pass

    def load_model(self, model_type='all'):
        pass

    def surrogate_optimize(self, param_limits=PARAM_LIMITS, rel_constraints=RELATIVE_CONSTRAINTS, sweep_freq=SWEEP_FREQUENCY,
                           freq_index=FREQUENCY_INDEX, freq_range=FREQUENCY_RANGE, sweep_res=FREQUENCY_SWEEP_RES):
        pass

    def simulation_optimize(self):
        pass

    def plot_graph(self, graph_res=GRAPH_RESOLUTION, graph_range=GRAPH_RANGE, freq_range=USED_FREQUENCY_RANGE,
                   save=False, num_randoms=2):
        pass
