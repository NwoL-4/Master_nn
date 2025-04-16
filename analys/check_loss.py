import glob
import os

import numpy as np
import torch
import plotly.graph_objects as go
import tqdm


def get_latest_model_path(model_dir):
    """Находит путь к последней сохраненной модели"""
    patern = os.path.abspath(os.path.join(model_dir, "cnn_model_*.pth"))
    print(patern)
    model_files = glob.glob(patern)
    if not model_files:
        print('Файлы отсутствуют')
        return None

    # Извлекаем номера из имен файлов и находим максимальный
    numbers = sorted([int(f.split('_')[-1].split('.')[0]) for f in model_files])
    for _number in numbers:
        yield _number, os.path.join(model_dir, f'cnn_model_{_number}.pth')


# model_dir = '../models/models'
model_dir = 'D://Neuron/models'
log_y = False


epochs = []
batchs = []
train_losses = []
poisson_losses = []
gauss_losses = []

only = True
for number, path in tqdm.tqdm(get_latest_model_path(os.path.join(os.getcwd(), model_dir))):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    train_losses.append(checkpoint['train_loss'])
    poisson_losses.append(checkpoint['poisson_loss'].detach().numpy())
    gauss_losses.append(checkpoint['gauss_loss'].detach().numpy())
    batchs.append(checkpoint['batch_idx'])
    epochs.append(checkpoint['epoch'] + 1)
#     if only:
#         state_dict = dict(checkpoint['model_state_dict'])
#         only = False
#
# print(state_dict.keys())

print(train_losses[0], poisson_losses[0], gauss_losses[0], batchs)

all_data = [
    ['MSE', train_losses, "#E60404"],
    ['Пуассон', poisson_losses, "#0643ff"],
    ['Гаусс', gauss_losses, "#2b9c00"]
]

plot_data = []
for data in all_data:
    if log_y:
        y = np.log(data[1])
        y_title = 'Логарифическая шкала ошибок'
    else:
        y = data[1]
        y_title = 'Шкала ошибок'
    plot_data.append(
        go.Scatter(
            x=np.array([epochs, batchs]),
            y=y,
            name=data[0],
            mode='lines+markers',
            marker=dict(
                color=data[2]
            )
        )
    )

layout = go.Layout(
    yaxis=dict(
        title=y_title
    ),
    xaxis=dict(
        title='Батчи * Эпоха'
    )
)

fig = go.Figure(
    data=plot_data,
    layout=layout
)

fig.show()