"""
First optimization workflow: train regressor on data then perform optimization
"""

from data.data_handler import DataHandler
from graph.graph_handler import GraphHandler
from integrated_workflow.file_handler import FileHandler
from integrated_workflow.model_handler import ModelHandler
from integrated_workflow.workflow_constants import *


class LinearWorkflow:
    def __init__(self, model_dir=MODEL_DIRECTORY):
        self.data_handler = DataHandler()
        self.model_handler = ModelHandler(None, None)
        self.file_handler = FileHandler(model_dir)
        self.graph_handler = GraphHandler(self.model_handler, self.data_handler, num_params=NUM_PARAMS)

    def train_model(self, model_type):
        if model_type == 'all':
            models, losses = self.model_handler.train_all_models()

            model_type = min(losses, key=losses.get)
            model = models[model_type]
            self.model_handler.set_model(model_type, model)

            for model_type, model in models.items():
                self.file_handler.save_model(model_type, model)
            self.file_handler.save_losses(losses)

        else:
            model, test_loss = self.model_handler.train_single_model(model_type)
            self.model_handler.set_model(model_type, model)

            self.file_handler.save_model(model_type, model)
            losses = self.file_handler.load_losses()
            losses[model_type] = test_loss
            self.file_handler.save_losses(losses)

    def surrogate_optimize(self, param_limits=PARAM_LIMITS, rel_constraints=RELATIVE_CONSTRAINTS, sweep_freq=SWEEP_FREQUENCY,
                           freq_index=FREQUENCY_INDEX, freq_range=FREQUENCY_RANGE, sweep_res=FREQUENCY_SWEEP_RES):
        pass

    def simulation_optimize(self):
        pass

    def plot_graph(self, graph_res=GRAPH_RESOLUTION, graph_range=GRAPH_RANGE, freq_range=USED_FREQUENCY_RANGE,
                   save=False, num_randoms=2):
        pass
