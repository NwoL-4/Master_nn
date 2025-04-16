import glob
import os

import numpy as np
import torch

import plotly.graph_objects as go
from tqdm import tqdm

TORCH_DTYPE = torch.float64

data_dir = 'D://Neuron/Dataset/(0.0001, 0.001, 0.01)_(20, 20, 100)'
log = True
n = 4500


os.chdir(data_dir)
datafile = glob.glob("*.npz")

text_color = [
    ['Максимум', 'red'],
    ['Максимум abs', 'orange'],
    ['Минимум', 'blue'],
    ['Минимум abs', 'teal'],
    ['Std', 'purple'],
    ['Std abs', 'pink'],
    ['Среднее', 'green'],
    ['Среднее abs', 'olive']
]

to_plot = []

for file in tqdm(datafile[:n]):
    with np.load(file) as data:
        field = data['field']

        to_plot.append([file,
                        field.max(),np.abs(field).max(),
                        field.min(), np.abs(field).min(),
                        field.std(), np.abs(field).std(),
                        field.mean(), np.abs(field).mean()])

to_plot = np.array(to_plot).T
print(to_plot.shape)
fig = go.Figure()

for index in range(1, to_plot.shape[0]):
    if log:
        y = np.log(to_plot[index].astype(float))
    else:
        y = to_plot[index].astype(float)

    fig.add_trace(
        go.Scatter(
            x=to_plot[0],
            y=y,
            name=text_color[index - 1][0],
            mode='lines',
            line=dict(
                color=text_color[index - 1][1]
            )
        )
    )
    print(f'{text_color[index - 1][0]}: {to_plot[index].astype(float).mean()}')

fig.show()
