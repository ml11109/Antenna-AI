"""
Plot a graph of s against frequency for a given set of parameters
"""

import torch

from graph.graph_constants import *
from graph.graph_handler import GraphHandler
from neural_network.nn_loader import load_neural_network


model, metadata, data_handler = load_neural_network(MODEL_NAME)
num_params = metadata['dimensions']['input'] - 1  # remove one parameter for freq
freq_index = metadata['freq_index']

graph_handler = GraphHandler(
    model=model,
    data_handler=data_handler,
    num_params=num_params,
    freq_index=freq_index
)

while True:
    try:
        inputs = input(f'Enter input data (or random / plot / save / reset / quit): ').lower().strip()

        match inputs:
            case 'random':
                graph_handler.add_random()

            case 'plot':
                graph_handler.plot()
                continue

            case 'save':
                filename = input('Enter file name (or cancel): ')
                if filename == 'cancel' or filename.strip() == '':
                    continue
                graph_handler.save(filename)
                continue

            case 'reset':
                graph_handler.reset()
                continue

            case 'quit':
                break

            case _:
                params = torch.tensor([float(x) for x in inputs.split(',')])
                if len(params) != num_params:
                    raise ValueError
                graph_handler.add_design(params)

    except ValueError:
        print('Invalid input. Please try again.')
