"""
Handles graph generation, plotting, and saving for model outputs over frequency ranges
"""

import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from graph.graph_constants import *


class GraphHandler:
    def __init__(self, model, data_handler, num_params, freq_index=0, model_name=MODEL_NAME, graph_res=GRAPH_RESOLUTION,
                 graph_range=GRAPH_RANGE, used_freq_range=USED_FREQUENCY_RANGE, image_directory=IMAGE_DIRECTORY):
        self.model = model
        self.data_handler = data_handler
        self.num_params = num_params
        self.freq_index = freq_index
        self.model_name = model_name
        self.graph_res = graph_res
        self.graph_range = graph_range
        self.used_freq_range = used_freq_range
        self.image_directory = Path(image_directory)
        self.outputs_df, self.designs, self.design_no = None, None, None

        self.freqs = torch.linspace(self.graph_range[0], self.graph_range[1], self.graph_res)
        self.freqs_scaled = data_handler.diff_scale(self.freqs, 'freq')
        self.reset()

    def reset(self):
        self.outputs_df = pd.DataFrame(columns=['Frequency', 'Model Output', 'Design No'])
        self.designs = {}
        self.design_no = 1

    def update_data(self, params, outputs):
        df_temp = pd.DataFrame({
            'Frequency': self.freqs.numpy(),
            'Model Output': outputs.numpy(),
            'Design No': self.design_no
        })

        if self.outputs_df.empty:
            self.outputs_df = df_temp
        else:
            self.outputs_df = pd.concat([self.outputs_df, df_temp], ignore_index=True)

        self.designs[f'Design {self.design_no}'] = params.tolist()
        self.design_no += 1

    def add_design(self, params):
        params_scaled = self.data_handler.diff_scale(params.reshape(1, -1), 'params').to(torch.float32)
        params_rep = params_scaled.repeat(self.graph_res, 1)
        left = params_rep[:, :self.freq_index]
        right = params_rep[:, self.freq_index:]
        params_with_freq = torch.cat([left, self.freqs_scaled.reshape(-1, 1), right], dim=1)
        outputs_scaled = self.model(params_with_freq).detach()
        outputs = self.data_handler.scale(outputs_scaled, 'outputs', inverse=True).flatten()
        self.update_data(params, outputs)

    def add_random(self):
        params_scaled = torch.randn(self.num_params)
        params = self.data_handler.diff_scale(params_scaled, 'params', inverse=True)
        print(f'Generated random parameters: {[round(param, 4) for param in params.tolist()]}')
        self.add_design(params)

    def make_graph(self):
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x=self.outputs_df['Frequency'],
            y=self.outputs_df['Model Output'],
            hue=self.outputs_df['Design No']
        )

        plt.title('Model Output vs Frequency', fontsize=16, pad=15)
        plt.xlabel('Frequency', fontsize=12, labelpad=10)
        plt.ylabel('Model Output', fontsize=12, labelpad=10)

        # Add vertical dotted lines for min and max used frequencies
        for f in self.used_freq_range:
            if self.graph_range[0] < f < self.graph_range[1]:
                plt.axvline(x=f, color='gray', linestyle=':', linewidth=1.0, alpha=0.8)

    def get_metadata(self):
        graph_metadata = {
            'model_name': self.model_name,
            'graph_resolution': self.graph_res,
            'graph_range': self.graph_range,
            'used_frequency_range': self.used_freq_range,
            'designs': self.designs
        }
        return graph_metadata

    def plot(self):
        self.make_graph()
        plt.show()

    def save(self, filename):
        self.make_graph()
        self.image_directory.mkdir(parents=True, exist_ok=True)
        image_path = self.image_directory / filename
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        metadata_path = image_path.with_name(f"{image_path.stem}_metadata.json")
        graph_metadata = self.get_metadata()
        with open(metadata_path, 'w') as f:
            json.dump(graph_metadata, f, indent=4)
