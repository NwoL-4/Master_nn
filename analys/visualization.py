import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch
import tqdm
from plotly.subplots import make_subplots


class VisualizationConfig:
    class Paths:
        neuron_dir = 'D://Neuron'
        models_dir = 'models'
        choiced_model = 'LeakyRelU'
        output_dir = 'visualizations'

        choiced_dataset = '(0.0001, 0.001, 0.01)_(20, 20, 100)'


    class Config:
        len_files = 'all' # 'all', 'half' or int
        use_log_scale = False
        show_after_run = False
        vis_file = '11958.npz'
        axes_projection = 'zx'
        use_model = False


class BatchWeightVisualizer:
    def __init__(self):
        self.weight_history = {}  # {epoch: {batch: {layer: weights_data}}}
        self.metrics_history = {}  # {epoch: {batch: {metric: value}}}
        self.layer_names = [
            'initial_conv.0',
            'encoder_res1.conv1',
            'encoder_res1.conv2',
            'encoder_conv1.0',
            'encoder_res2.conv1',
            'encoder_res2.conv2',
            'encoder_conv2.0',
            'encoder_res3.conv1',
            'encoder_res3.conv2',
            'decoder_res1.conv1',
            'decoder_res1.conv2',
            'decoder_conv1.0',
            'decoder_res2.conv1',
            'decoder_res2.conv2',
            'decoder_conv2.0',
            'decoder_res3.conv1',
            'decoder_res3.conv2',
            'final_conv'
        ]


    def extract_checkpoint_data(self, checkpoint_path):
        """
        Извлекает данные из одного чекпоинта
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        state_dict = checkpoint['model_state_dict']

        add_text = False

        keys = list(state_dict.keys())
        if 'module.' in keys[0]:
            add_text = True

        # Инициализируем структуры данных если нужно
        if epoch not in self.weight_history:
            self.weight_history[epoch] = {}
            self.metrics_history[epoch] = {}

        # Извлекаем веса
        weights_data = {}
        for layer_name in self.layer_names:
            layer_text = layer_name + '.weight'

            if add_text:
                layer_text = 'module.' + layer_text

            if layer_text in state_dict:
                weights = state_dict[layer_text].cpu().numpy()
                weights_data[layer_name] = {
                    'name': layer_name,
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights),
                    'median': np.median(weights),
                    'hist': np.histogram(weights.flatten(), bins=100)
                }

        # Сохраняем веса
        self.weight_history[epoch][batch_idx] = weights_data

        # Сохраняем метрики
        self.metrics_history[epoch][batch_idx] = {
            'train_loss': checkpoint['train_loss'],
            'gauss_loss': checkpoint['gauss_loss'],
            'poisson_loss': checkpoint['poisson_loss']
        }

    def load_all_checkpoints(self, model_dir, len_files):
        """
        Загружает все чекпоинты из директории
        """
        patern = os.path.join(model_dir, 'cnn_model_*.pth')
        files = glob.glob(patern)
        numbers = sorted([int(f.split('_')[-1].split('.')[0]) for f in files])

        if isinstance(len_files, int):
            numbers = numbers[:len_files]
        elif isinstance(len_files, str):
            if len_files == 'all':
                pass
            elif len_files == 'half':
                numbers = numbers[:len(numbers) // 2]
            else:
                raise ValueError('len_files should be either "all" or "half"')
        else:
            raise ValueError('len_files should be int or "all" or "half"')

        for number in tqdm.tqdm(numbers):
            self.extract_checkpoint_data(patern.replace('*', str(number)))

        print(f"Loaded data from {len(self.weight_history)} epochs")

    def create_weight_evolution_animation(self):
        """
        Создает анимированный график эволюции весов по батчам с отдельными графиками для каждого слоя
        """
        # Подготовка данных
        data = []
        for epoch in self.weight_history:
            for batch in self.weight_history[epoch]:
                for layer in self.weight_history[epoch][batch]:
                    layer_data = self.weight_history[epoch][batch][layer]
                    data.append({
                        'epoch': epoch,
                        'batch': batch,
                        'layer': layer,
                        'mean': layer_data['mean'],
                        'std': layer_data['std'],
                        'min': layer_data['min'],
                        'max': layer_data['max'],
                        'median': layer_data['median'],
                    })

        df = pd.DataFrame(data)
        df = df.sort_values(['epoch', 'batch'])

        # Создаем цветовую палитру
        colors = (px.colors.qualitative.Set3 + px.colors.qualitative.Set2)[:len(self.layer_names)]
        color_map = dict(zip(self.layer_names, colors))

        # Создаем подграфики
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Weights', 'Std Weights',
                            'Min/Max Weights', 'Median Weights'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Создаем фреймы для каждой эпохи
        frames = []
        for epoch in df['epoch'].unique():
            epoch_data = df[df['epoch'] == epoch]

            frame_traces = []

            # Mean weights (row=1, col=1)
            for layer in self.layer_names:
                layer_data = epoch_data[epoch_data['layer'] == layer]
                frame_traces.append(
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['mean'],
                        mode='lines+markers',
                        name=layer,
                        legendgroup=layer,
                        line=dict(color=color_map[layer]),
                        showlegend=True,
                        xaxis='x1',
                        yaxis='y1'
                    )
                )

            # Std weights (row=1, col=2)
            for layer in self.layer_names:
                layer_data = epoch_data[epoch_data['layer'] == layer]
                frame_traces.append(
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['std'],
                        mode='lines+markers',
                        name=layer,
                        legendgroup=layer,
                        line=dict(color=color_map[layer]),
                        showlegend=False,
                        xaxis='x2',
                        yaxis='y2'
                    )
                )

            # Min/Max weights (row=2, col=1)
            for layer in self.layer_names:
                layer_data = epoch_data[epoch_data['layer'] == layer]
                frame_traces.extend([
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['min'],
                        mode='lines',
                        name=f"{layer} (min)",
                        legendgroup=layer,
                        line=dict(color=color_map[layer], dash='dash'),
                        showlegend=False,
                        xaxis='x3',
                        yaxis='y3'
                    ),
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['max'],
                        mode='lines',
                        name=f"{layer} (max)",
                        legendgroup=layer,
                        line=dict(color=color_map[layer]),
                        showlegend=False,
                        xaxis='x3',
                        yaxis='y3'
                    )
                ])

            # Median weights (row=2, col=2)
            for layer in self.layer_names:
                layer_data = epoch_data[epoch_data['layer'] == layer]
                frame_traces.append(
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['median'],
                        mode='lines+markers',
                        name=layer,
                        legendgroup=layer,
                        line=dict(color=color_map[layer]),
                        showlegend=False,
                        xaxis='x4',
                        yaxis='y4'
                    )
                )

            frames.append(go.Frame(data=frame_traces, name=f'epoch_{epoch}'))

        # Добавляем начальные графики
        for trace in frames[0].data:
            fig.add_trace(trace)

        # Обновляем layout
        fig.update_layout(
            showlegend=True,
            legend=dict(
                # groupclick="toggleitem",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 0,
                'x': 0,
                'buttons': [
                    {'label': 'Play',
                     'method': 'animate',
                     'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                     'fromcurrent': True}]},
                    {'label': 'Pause',
                     'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                       'mode': 'immediate',
                                       'transition': {'duration': 0}}]}
                ]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Epoch: '},
                'pad': {"t": 50},
                'steps': [{'method': 'animate',
                           'label': f'Epoch {epoch}',
                           'args': [[f'epoch_{epoch}'], {'frame': {'duration': 0, 'redraw': True},
                                                         'mode': 'immediate',
                                                         'transition': {'duration': 0}}]}
                          for epoch in sorted(df['epoch'].unique())]
            }]
        )

        # Обновляем оси
        fig.update_xaxes(title_text="Batch", row=2, col=1)
        fig.update_xaxes(title_text="Batch", row=2, col=2)
        fig.update_yaxes(title_text="Weight Value", row=1, col=1)
        fig.update_yaxes(title_text="Standard Deviation", row=1, col=2)
        fig.update_yaxes(title_text="Min/Max Values", row=2, col=1)
        fig.update_yaxes(title_text="Median Value", row=2, col=2)

        return fig

    def plot_metrics_evolution(self, log: bool):
        """
        Создает график эволюции метрик обучения
        """
        data = []
        for epoch in sorted(self.metrics_history):
            for batch in sorted(self.metrics_history[epoch]):
                metrics = self.metrics_history[epoch][batch]
                data.append({
                    'epoch': epoch,
                    'batch': batch,
                    **metrics
                })

        df = pd.DataFrame(data).sort_values(by='batch')
        
        if log:
            df['train_loss'] = np.log(df['train_loss'])
            df['gauss_loss'] = np.log(df['gauss_loss'])
            df['poisson_loss'] = np.log(df['poisson_loss'])

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=('Training Loss',
                                            'Gauss Loss',
                                            'Poisson Loss'))

        all_data = [
            ['MSE', "#E60404"],
            ['Пуассон', "#0643ff"],
            ['Гаусс', "#2b9c00"]
        ]

        # Training Loss
        fig.add_trace(
            go.Scatter(x=np.array([df['epoch'], df['batch']]),
                       y=df['train_loss'],
                       mode='lines+markers',
                       name=all_data[0][0],
                       marker=dict(
                           color=all_data[0][1],
                       )),
            row=1, col=1
        )

        # Gauss Loss
        fig.add_trace(
            go.Scatter(x=np.array([df['epoch'], df['batch']]),
                       y=[value.detach().numpy() for value in df['gauss_loss']],
                       mode='lines+markers',
                       name=all_data[1][0],
                       marker=dict(
                           color=all_data[1][1],
                       )),
            row=1, col=2
        )

        # Poisson Loss
        fig.add_trace(
            go.Scatter(x=np.array([df['epoch'], df['batch']]),
                       y=[value.detach().numpy() for value in df['poisson_loss']],
                       mode='lines+markers',
                       name=all_data[2][0],
                       marker=dict(
                           color=all_data[2][1],
                       )),
            row=1, col=3
        )

        fig.update_layout(
            yaxis=dict(
                title=dict(
                    text='Логарифмическая шкала ошибок' if log else 'Шкала ошибок',
                )
            )
        )

        return fig

    def create_layer_weight_distribution_animation(self):
        """
        Создает анимированную гистограмму распределения весов с выбором слоя
        и слайдером для каждого слоя
        """
        # Подготавливаем данные
        data = []
        for epoch in self.weight_history:
            for batch in self.weight_history[epoch]:
                for layer_name, layer_data in self.weight_history[epoch][batch].items():
                    hist_values, hist_bins = layer_data['hist']
                    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2

                    data.extend([{
                        'epoch': epoch,
                        'batch': batch,
                        'layer': layer_name,
                        'bin_center': bin_center,
                        'frequency': freq,
                        'bin_left': left,
                        'bin_right': right,
                        'mean': layer_data['mean'],
                        'std': layer_data['std'],
                        'min': layer_data['min'],
                        'max': layer_data['max'],
                        'median': layer_data['median']
                    } for bin_center, freq, left, right in zip(
                        bin_centers,
                        hist_values,
                        hist_bins[:-1],
                        hist_bins[1:]
                    )])

        df = pd.DataFrame(data)
        df = df.sort_values(['layer', 'epoch', 'batch'])

        # Создаем цветовую палитру для слоев
        colors = (px.colors.qualitative.Set2 + px.colors.qualitative.Set3)[:len(self.layer_names)]
        color_map = dict(zip(self.layer_names, colors))

        fig = go.Figure()

        # Создаем фреймы для каждого слоя, эпохи и батча
        frames = []

        for layer_name in self.layer_names:
            layer_data = df[df['layer'] == layer_name]

            for epoch in layer_data['epoch'].unique():
                epoch_data = layer_data[layer_data['epoch'] == epoch]

                for batch in epoch_data['batch'].unique():
                    batch_data = epoch_data[epoch_data['batch'] == batch]

                    stats_text = (
                        f"Statistics:<br>"
                        f"Mean: {batch_data['mean'].iloc[0]:.6f}<br>"
                        f"Std: {batch_data['std'].iloc[0]:.6f}<br>"
                        f"Min: {batch_data['min'].iloc[0]:.6f}<br>"
                        f"Max: {batch_data['max'].iloc[0]:.6f}<br>"
                        f"Median: {batch_data['median'].iloc[0]:.6f}"
                    )

                    frame_data = [
                        # Гистограмма
                        go.Bar(
                            x=batch_data['bin_center'],
                            y=batch_data['frequency'],
                            name=layer_name,
                            marker_color=color_map[layer_name],
                            customdata=np.stack((
                                batch_data['bin_left'],
                                batch_data['bin_right']
                            ), axis=-1),
                            hovertemplate=(
                                "Range: [%{customdata[0]:.6f}, %{customdata[1]:.6f}]<br>"
                                "Frequency: %{y}<br>"
                                "<extra></extra>"
                            )
                        ),
                        # Вертикальная линия для среднего значения
                        go.Scatter(
                            x=[batch_data['mean'].iloc[0], batch_data['mean'].iloc[0]],
                            y=[0, batch_data['frequency'].max()],
                            mode='lines',
                            name='Mean',
                            line=dict(color='red', dash='dash'),
                            showlegend=False
                        ),
                        # Вертикальная линия для медианы
                        go.Scatter(
                            x=[batch_data['median'].iloc[0], batch_data['median'].iloc[0]],
                            y=[0, batch_data['frequency'].max()],
                            mode='lines',
                            name='Median',
                            line=dict(color='green', dash='dash'),
                            showlegend=False
                        )
                    ]

                    frames.append(
                        go.Frame(
                            data=frame_data,
                            name=f'{layer_name}_epoch_{epoch}_batch_{batch}',
                            layout=go.Layout(
                                annotations=[
                                    dict(
                                        text=stats_text,
                                        xref="paper",
                                        yref="paper",
                                        x=1.15,
                                        y=0.5,
                                        showarrow=False,
                                        font=dict(size=12),
                                        align="left"
                                    )
                                ]
                            )
                        )
                    )

        # Добавляем начальные данные (первый фрейм)
        first_frame = frames[0]
        for trace in first_frame.data:
            fig.add_trace(trace)

        # Создаем кнопки для выбора слоя
        layer_buttons = []
        for layer_name in self.layer_names:
            layer_data = df[df['layer'] == layer_name]
            slider_steps = []

            for epoch in layer_data['epoch'].unique():
                epoch_data = layer_data[layer_data['epoch'] == epoch]
                for batch in epoch_data['batch'].unique():
                    frame_name = f'{layer_name}_epoch_{epoch}_batch_{batch}'
                    slider_steps.append(
                        {
                            'args': [
                                [frame_name],
                                {'frame': {'duration': 0, 'redraw': True},
                                 'mode': 'immediate',
                                 'transition': {'duration': 0}}
                            ],
                            'label': f'E{epoch}B{batch}',
                            'method': 'animate'
                        }
                    )

            # Добавляем кнопку для слоя с соответствующим слайдером
            layer_buttons.append(
                dict(
                    args=[
                        {"sliders": [{
                            'active': 0,
                            'yanchor': 'top',
                            'xanchor': 'left',
                            'currentvalue': {
                                'font': {'size': 16},
                                'prefix': f'{layer_name} - ',
                                'visible': True,
                                'xanchor': 'right'
                            },
                            'transition': {'duration': 0},
                            'pad': {'b': 10, 't': 50},
                            'len': 0.9,
                            'x': 0.1,
                            'y': 0,
                            'steps': slider_steps
                        }]}
                    ],
                    label=layer_name,
                    method="relayout"
                )
            )

        # Создаем слайдер для первого слоя
        first_layer = df['layer'].unique()[0]
        first_layer_frames = [frame for frame in frames if frame.name.startswith(first_layer)]
        first_layer_steps = []
    
        for frame in first_layer_frames:
            epoch_batch = frame.name.split('_')
            epoch = epoch_batch[2]
            batch = epoch_batch[4]
    
            first_layer_steps.append({
                'args': [
                    [frame.name],
                    {'frame': {'duration': 0, 'redraw': True},
                     'mode': 'immediate',
                     'transition': {'duration': 0}}
                ],
                'label': f'E{epoch}B{batch}',
                'method': 'animate'
            })
    
        # Обновляем layout
        fig.update_layout(
            updatemenus=[
                # Меню выбора слоя
                {
                    'buttons': layer_buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 1.1,
                    'y': 0.2,
                    'xanchor': 'left',
                    'yanchor': 'top'
                },
                # Кнопки управления анимацией
                {
                    'type': 'buttons',
                    'showactive': False,
                    'x': 1.1,
                    'y': 0,
                    'xanchor': 'left',
                    'yanchor': 'top',
                    'buttons': [
                        dict(label='Play',
                             method='animate',
                             args=[None, {
                                 'frame': {'duration': 500, 'redraw': True},
                                 'fromcurrent': True,
                                 'transition': {'duration': 100}
                             }]),
                        dict(label='Pause',
                             method='animate',
                             args=[[None], {
                                 'frame': {'duration': 0, 'redraw': True},
                                 'mode': 'immediate',
                                 'transition': {'duration': 0}
                             }])
                    ]
                }
            ],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16},
                    'prefix': f'{first_layer} - ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': first_layer_steps
            }],
            title={
                'text': "Weight Distribution Evolution",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Weight Value",
            yaxis_title="Frequency",
            margin=dict(r=200)  # Увеличиваем правый отступ для статистики
        )
    
        # Устанавливаем фреймы
        fig.frames = frames
    
        return fig

    @staticmethod
    def create_force_field_visualization(file: str, path: str, axes_projection: str, use_model: bool, **kwargs):
        """
        Создает визуализацию силового поля с интерактивными элементами

        Parameters:


    ----


        file : str
            Имя файла или 'random' для случайного выбора
        path : str
            Путь к директории с файлами
        axes_projection : str
            Тип проекции ('xy', 'xz', 'yz')
        use_model : bool
            Использовать ли модель для сравнения
        **kwargs : dict
            Дополнительные параметры (model для use_model=True)
        """
        if use_model and kwargs.get('model', None) is None:
            raise ValueError('Added kwarg model')

        # Загрузка данных
        if file == 'random':
            select_file = np.random.choice(glob.glob(os.path.join(path, '*.npz')))
        elif file.endswith('.npz'):
            select_file = os.path.join(path, file)
        else:
            raise ValueError(f'File {file} must end with .npz or "random"')

        data = np.load(select_file, allow_pickle=True)

        space, grid = [np.fromstring(string[1:-1], dtype=float, sep=', ')
                       for string in Path(path).name.split('_')]

        grid = grid.astype(np.int64)

        particle, field = data['particle'], data['field']  # field.shape == (3, K, L, M)
        if 'charge' in data.keys():
            particle = np.concatenate((particle, [data['charge']]), axis=0)

        # Определение осей проекции
        axis_mapping = {
            'xy': np.array((0, 1, 2)),
            'xz': np.array((0, 2, 1)),
            'yz': np.array((1, 2, 0)),
            'yx': np.array((1, 0, 2)),
            'zx': np.array((2, 0, 1)),
            'zy': np.array((2, 1, 0)),
        }
        if axes_projection not in axis_mapping:
            raise ValueError(f"Invalid projection. Must be one of {list(axis_mapping.keys())}")

        x_idx, y_idx, z_idx = axis_mapping[axes_projection]

        # Переупорядочиваем оси field в соответствии с проекцией
        axes_order = np.array([0] + (axis_mapping[axes_projection] + 1).tolist())
        field = np.transpose(field, axes_order)
        max_field = np.max(np.abs(field))
        field = field / (max_field * 500)

        # Создание сетки для quiver
        x = np.linspace(0, space[x_idx], grid[x_idx])
        y = np.linspace(0, space[y_idx], grid[y_idx])
        X, Y = np.meshgrid(x, y)
        X, Y = X.T, Y.T

        # Создание базового графика
        fig = make_subplots(
            rows=2 if use_model else 1,
            cols=1,
            subplot_titles=['Обучающее поле'] + (['Поле модели'] if use_model else []),
            vertical_spacing=0.1
        )

        # Создание слайдера
        z_range = np.linspace(0, space[z_idx], grid[z_idx])
        frames = []
        slider_steps = []

        for i, z_val in enumerate(z_range):
            # Получаем компоненты векторного поля для текущего слоя
            u = field[x_idx, :, :, i]
            v = field[y_idx, :, :, i]

            # Создаем quiver plot
            quiver_fig = ff.create_quiver(
                X, Y, u, v,
                scale=0.1,  # Масштаб стрелок
                arrow_scale=0.01,  # Размер наконечников стрелок
                angle=np.pi/20,
                marker=dict(
                    colorscale='viridis'
                ),
                name='Force field'
            )

            # Добавляем ближайшие частицы
            diff = np.diff(z_range)[0] * 0.95 if len(z_range) > 1 else 0.1
            nearby_mask = np.abs(particle[z_idx] - z_val) <= diff
            nearby_particles = particle[:, nearby_mask]

            # Создаем кадр
            frame = go.Frame(
                data=[
                    # Данные quiver plot
                    *quiver_fig.data,
                    # Точки ближайших частиц
                    go.Scatter(
                        x=nearby_particles[x_idx],
                        y=nearby_particles[y_idx],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=['#890000' if particle == -1 else '#100089' for particle in nearby_particles[3]],
                        ),
                        name='Nearby particles'
                    )
                ],
                name=f'frame_{i}'
            )
            frames.append(frame)

            # Добавляем шаг слайдера
            slider_steps.append(
                dict(
                    args=[
                        [f'frame_{i}'],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate',
                            transition=dict(duration=0)
                        )
                    ],
                    label=f'z={z_val}',
                    method='animate'
                )
            )

        # Добавляем начальный кадр
        for trace in frames[0].data:
            fig.add_trace(trace)

        # Обновляем layout
        fig.update_layout(
            title=f"Силовое поле (проекция {axes_projection.upper()})",
            showlegend=True,
            # Добавляем слайдер
            sliders=[dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(
                    font=dict(size=16),
                    prefix='z = ',
                    visible=True,
                    xanchor='right'
                ),
                transition=dict(duration=0),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=slider_steps
            )],
            # Добавляем кнопки управления анимацией
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=500, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )]
        )

        # Настройка осей
        axis_labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
        fig.update_xaxes(
            title_text=f"{axis_labels[axes_projection[0]]}",
            range=[0, space[x_idx]]
        )
        fig.update_yaxes(
            title_text=f"{axis_labels[axes_projection[1]]}",
            range=[0, space[y_idx]]
        )

        # Добавляем фреймы
        fig.frames = frames

        return fig, f"{axes_projection}-{Path(select_file).name.split('.')[0]}"


def main():
    config = VisualizationConfig
    paths = config.Paths
    config = config.Config

    visualizer = BatchWeightVisualizer()

    visualizer.load_all_checkpoints(
        model_dir=os.path.join(paths.neuron_dir, paths.models_dir, paths.choiced_model),
        len_files=config.len_files
    )

    max_epoch = max(visualizer.weight_history)
    max_batch = max(visualizer.weight_history[max_epoch])

    save_dir = os.path.join(paths.neuron_dir, paths.output_dir, paths.choiced_model.lower(), f'e{max_epoch}b{max_batch}')
    os.makedirs(save_dir, exist_ok=True)

    start = time.time()
    weight_evolution_fig = visualizer.create_weight_evolution_animation()
    print(f'weight_evolution_fig successfully created: {time.time() - start} c.')
    start = time.time()
    metrics_fig = visualizer.plot_metrics_evolution(log=config.use_log_scale)
    print(f'metrics_fig successfully created: {time.time() - start} c.')
    start = time.time()
    distribution_fig = visualizer.create_layer_weight_distribution_animation()
    print(f'distribution_fig successfully created: {time.time() - start} c.')
    start = time.time()
    field_fig, select_file = visualizer.create_force_field_visualization(
        file=config.vis_file,
        path=os.path.join(paths.neuron_dir, 'Dataset', paths.choiced_dataset),
        axes_projection=config.axes_projection,
        use_model=config.use_model,
    )
    print(f'field_fig successfully created: {time.time() - start} c.')

    weight_evolution_fig.write_html(os.path.join(save_dir, 'weight_evolution.html'))
    metrics_fig.write_html(os.path.join(save_dir, 'metrics_evolution.html'))
    distribution_fig.write_html(os.path.join(save_dir, 'distribution_evolution.html'))
    field_fig.write_html(os.path.join(Path(save_dir).parent, f'{select_file}.html'))

    if config.show_after_run:
        weight_evolution_fig.show()
        metrics_fig.show()
        distribution_fig.show()
        field_fig.show()


if __name__ == '__main__':
    main()