import os.path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.constants import epsilon_0

import functions
import main


def load_and_visualize(f_name: str, max_xyz: np.ndarray, downsample: int = 3):
    """Загрузка данных и визуализация в 3D."""

    # Загрузка данных
    f_data = np.load(f_name)
    p_data = f_data['particle']
    field = f_data['field'] / (4 * np.pi * epsilon_0)

    density = functions.particles_to_grid_density(p_data, grid_size, space_size).numpy()

    SPACE = space_size
    NNODES = grid_size

    colorsparticle = ['rgba(255, 0, 0, 1)' if p_data[3][index] == -1 else 'rgba(0, 0, 255, 1)'
                      for index in range(p_data.shape[1])]

    print('Электрическая напряженность в пространстве\n'
          f'Максимальная: {np.abs(field).max()}\n'
          f'Средняя: {field.mean()}\n'
          f'Средняя по абсолютному значению {np.abs(field).mean()}')

    grid = functions.generate_grid_nodes(SPACE, NNODES)
    # Сэмплирование данных (уменьшаем плотность векторов)
    step = downsample
    sampled_grid = grid[:, ::step, ::step, ::step]
    sampled_field = field[:, ::step, ::step, ::step]

    # Подготовка координат для векторного поля
    X, Y, Z = sampled_grid[0], sampled_grid[1], sampled_grid[2]

    U, V, W = sampled_field[0], sampled_field[1], sampled_field[2]

    # down = 1
    down = np.max(np.abs([U, V, W])) * 5000
    U, V, W = np.array([U, V, W]) / down

    # Создание фигуры
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Векторное поле (конусы для лучшего восприятия)
    # fig.add_trace(
    #     go.Cone(
    #         x=X.flatten(),
    #         y=Y.flatten(),
    #         z=Z.flatten(),
    #         u=U.flatten(),
    #         v=V.flatten(),
    #         w=W.flatten(),
    #         sizemode="raw",
    #         sizeref=0.1,
    #         # anchor='tail',
    #         colorscale='Viridis',
    #         name='Электрическое поле'
    #     )
    # )

    fig.add_trace(
        go.Scatter3d(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            mode='markers',
            marker=dict(
                size=5,
                color=density.flatten(),
                colorscale='Viridis',
                opacity=0.8,
                symbol='circle'
            ),
            text=density.flatten(),
            name='Плотность в узлах',
        )
    )

    # Частицы (точечное облако)
    fig.add_trace(
        go.Scatter3d(
            x=p_data[0],
            y=p_data[1],
            z=p_data[2],
            mode='markers',
            marker=dict(
                size=2,
                color=colorsparticle,  # Полупрозрачный красный
                symbol='circle'
            ),
            name='Заряженные частицы'
        )
    )

    # Настройка вида
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[0, max_xyz[0]],
                title='X (м)'
            ),
            yaxis=dict(
                range=[0, max_xyz[1]],
                title='Y (м)'),
            zaxis=dict(
                range=[0, max_xyz[2]],
                title='Z (м)'),
            aspectratio=dict(x=1, y=1, z=2),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ),
        title_text=f'3D визуализация поля и частиц: Число частиц: {p_data.shape[0]}'
    )

    fig.show()
    fig.write_html('visualization.html')  # Для сохранения в файл


main_folder = 'D://Neuron/Dataset/'
folder = '(0.0001, 0.001, 0.01)_(30, 30, 100)'
filename = '0.npz'


grid_size = tuple(np.fromstring(folder.split('_')[1][1:-1], sep=', ', dtype=np.int64))
space_size = tuple(np.fromstring(folder.split('_')[0][1:-1], sep=', ', dtype=np.float64))
path = os.path.join(main_folder, folder)

max_xyz = np.fromstring(path.split('/')[-1].split('_')[0][1:-1], dtype=main.DTYPE, sep=', ')
# Пример использования
load_and_visualize(os.path.join(path, filename), max_xyz, downsample=1)