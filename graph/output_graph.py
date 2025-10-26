"""
Plot a graph of s against frequency for a given set of parameters
"""

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from graph.graph_constants import *
from neural_network.nn_loader import load_neural_network


def get_freq_col(freq_range=FREQUENCY_RANGE, sweep_res=GRAPH_RESOLUTION):
    freqs = torch.linspace(freq_range[0], freq_range[1], sweep_res)
    freqs_scaled = data_handler.diff_scale(freqs, 'freq')
    return freqs, freqs_scaled

def get_output_col(params, freq_scaled, freq_index=0, sweep_res=GRAPH_RESOLUTION):
    params_scaled = data_handler.diff_scale(params.reshape(1, -1), 'params').to(torch.float32)
    params_rep = params_scaled.repeat(sweep_res, 1)
    left = params_rep[:, :freq_index]
    right = params_rep[:, freq_index:]
    params_with_freq = torch.cat([left, freq_scaled.reshape(-1, 1), right], dim=1)
    outputs_scaled = model(params_with_freq).detach()
    outputs = data_handler.scale(outputs_scaled, 'outputs', inverse=True).flatten()
    return outputs

def get_random(num_params):
    params_scaled = torch.randn(num_params)
    params = data_handler.diff_scale(params_scaled, 'params', inverse=True)
    print(f'Generated random parameters: {[round(param, 4) for param in params.tolist()]}')
    return params

def plot_outputs(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=df['Frequency'],
        y=df['Model Output'],
        hue=df['Design No']
    )
    plt.title('Model Output vs Frequency', fontsize=16, pad=15)
    plt.xlabel('Frequency', fontsize=12, labelpad=10)
    plt.ylabel('Model Output', fontsize=12, labelpad=10)
    plt.show()


model, metadata, data_handler = load_neural_network(MODEL_NAME, MODEL_DIRECTORY)
num_params = metadata['dimensions']['input'] - 1  # remove one parameter for freq
freq_index = metadata['freq_index']

freqs, freqs_scaled = get_freq_col()
df = pd.DataFrame(columns=['Frequency', 'Model Output', 'Design No'])
design_no = 1

while True:
    try:
        inputs = input(f'Enter input data (comma-separated, {num_params} values): ')

        match inputs:
            case 'quit':
                break

            case 'reset':
                df = pd.DataFrame(columns=['Frequency', 'Model Output', 'Design No'])
                design_no = 1
                continue

            case 'plot':
                plot_outputs(df)
                continue

            case 'random':
                inputs = get_random(num_params)

            case _:
                inputs = torch.tensor([float(x) for x in inputs.split(',')])
                if len(inputs) != num_params:
                    raise ValueError

        output_col = get_output_col(inputs, freqs_scaled, freq_index)
        design_no += 1

        df_temp = pd.DataFrame({
            'Frequency': freqs.numpy(),
            'Model Output': output_col.numpy(),
            'Design No': design_no
        })

        if df.empty:
            df = df_temp
        else:
            df = pd.concat([df, df_temp], ignore_index=True)

    except ValueError:
        print('Invalid input. Please try again.')
