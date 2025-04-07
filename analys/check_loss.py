import glob
import os

import torch
import plotly.graph_objects as go
import tqdm


def get_latest_model_path(model_dir):
    """Находит путь к последней сохраненной модели"""
    model_files = glob.glob(os.path.join(model_dir, "cnn_model_*.pth"))
    if not model_files:
        return None

    # Извлекаем номера из имен файлов и находим максимальный
    numbers = sorted([int(f.split('_')[-1].split('.')[0]) for f in model_files])
    for _number in numbers:
        yield _number, os.path.join(model_dir, f'cnn_model_{_number}.pth')


colors = ["#E60404", "#0643ff", "#2b9c00"]

model_dir = '../models'
batchs = []
train_losses = []
poisson_losses = []
gauss_losses = []
for number, path in tqdm.tqdm(get_latest_model_path(model_dir)):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    train_losses.append(checkpoint['train_loss'])
    poisson_losses.append(checkpoint['poisson_loss'].detach().numpy())
    gauss_losses.append(checkpoint['gauss_loss'].detach().numpy())
    batchs.append(number)

print(train_losses[0], poisson_losses[0], gauss_losses[0])

fig = go.Figure(
    data=[
        go.Scatter(
            x=batchs,
            y=train_losses,
            name='MSE',
            mode='lines+markers',
            marker=dict(
                color=colors[0]
            ),
        ),
        go.Scatter(
            x=batchs,
            y=poisson_losses,
            name='Пуассон',
            mode='lines+markers',
            marker=dict(
                color=colors[1]
            ),
        ),
        go.Scatter(
            x=batchs,
            y=gauss_losses,
            name='Гаусс',
            mode='lines+markers',
            marker=dict(
                color=colors[2]
            ),
        ),
    ]
)

fig.show()