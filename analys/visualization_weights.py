import glob

import plotly
import plotly.graph_objects as go
import plotly.express as px
import tqdm
from plotly.subplots import make_subplots
import numpy as np
import torch
import os
import pandas as pd


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

        # Инициализируем структуры данных если нужно
        if epoch not in self.weight_history:
            self.weight_history[epoch] = {}
            self.metrics_history[epoch] = {}

        # Извлекаем веса
        weights_data = {}
        for layer_name in self.layer_names:
            if layer_name + '.weight' in state_dict:
                weights = state_dict[layer_name + '.weight'].cpu().numpy()
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

    def load_all_checkpoints(self, model_dir):
        """
        Загружает все чекпоинты из директории
        """
        patern = os.path.join(model_dir, 'cnn_model_*.pth')
        files = glob.glob(patern)
        numbers = sorted([int(f.split('_')[-1].split('.')[0]) for f in files])

        for number in tqdm.tqdm(numbers[:5]):
            self.extract_checkpoint_data(patern.replace('*', str(number)))

        print(f"Loaded data from {len(self.weight_history)} epochs")

    def create_weight_evolution_animation(self):
        """
        Создает анимированный график эволюции весов по батчам с разными цветами
        """
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
                        'name': layer_data['name']
                    })

        df = pd.DataFrame(data)
        df = df.sort_values(['epoch', 'batch', 'layer'])  # Сортировка

        # Создаем цветовую палитру для слоев
        colors = (px.colors.qualitative.Set3 + px.colors.qualitative.Set2)[:len(self.layer_names)]
        color_map = dict(zip(self.layer_names, colors))

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Weights', 'Std Weights',
                            'Min/Max Weights', 'Median Weights')
        )

        frames = []
        for epoch in df['epoch'].unique():
            epoch_data = df[df['epoch'] == epoch]

            frame_data = []
            for layer in self.layer_names:
                layer_data = epoch_data[epoch_data['layer'] == layer]
                # Mean weights
                frame_data.append(
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['mean'],
                        mode='lines+markers',
                        name=f'{layer} mean',
                        line=dict(color=color_map[layer]),
                        showlegend=True
                    )
                )

                # Std weights
                frame_data.append(
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['std'],
                        mode='lines+markers',
                        name=f'{layer} std',
                        line=dict(color=color_map[layer]),
                        showlegend=True
                    )
                )

                # Min/Max weights
                frame_data.extend([
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['min'],
                        mode='lines',
                        name=f"{layer} min",
                        line=dict(color=color_map[layer], dash='dash'),
                        showlegend=True
                    ),
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['max'],
                        mode='lines',
                        name=f"{layer} max",
                        line=dict(color=color_map[layer]),
                        showlegend=True
                    )
                ])

                # Median weights
                frame_data.append(
                    go.Scatter(
                        x=layer_data['batch'],
                        y=layer_data['median'],
                        mode='lines+markers',
                        name=f'{layer} median',
                        line=dict(color=color_map[layer]),
                        showlegend=True
                    )
                )

            frames.append(go.Frame(data=frame_data, name=f'epoch_{epoch}'))

        indices = [[1, 1], [1, 2], [2, 1], [2, 1], [2, 2]]

        # Добавляем начальные графики
        for ind, trace in zip(indices, frames[0].data):
            fig.add_trace(trace, row=ind[0], col=ind[1])

        fig.frames = frames

        # Обновляем layout
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
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
                'steps': [{'method': 'animate',
                           'label': f'Batch {batch}',
                           'args': [[f'batch_{batch}'], {'frame': {'duration': 0, 'redraw': True},
                                                         'mode': 'immediate',
                                                         'transition': {'duration': 0}}]}
                          for batch in sorted(df['batch'].unique())]
            }],
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )

        return fig

    def plot_metrics_evolution(self):
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

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=('Training Loss',
                                            'Gauss Loss',
                                            'Poisson Loss'))

        # Training Loss
        fig.add_trace(
            go.Scatter(x=df['batch'],
                       y=df['train_loss'],
                       mode='lines+markers',
                       name='Train Loss'),
            row=1, col=1
        )

        # Gauss Loss
        fig.add_trace(
            go.Scatter(x=df['batch'],
                       y=df['gauss_loss'],
                       mode='lines+markers',
                       name='Gauss Loss'),
            row=1, col=2
        )

        # Poisson Loss
        fig.add_trace(
            go.Scatter(x=df['batch'],
                       y=df['poisson_loss'],
                       mode='lines+markers',
                       name='Poisson Loss'),
            row=1, col=3
        )

        return fig

    def create_layer_weight_distribution_animation(self):
        """
        Создает анимированную гистограмму распределения весов по слоям со слайдером
        """
        fig = go.Figure()

        epochs = sorted(self.weight_history.keys())
        first_epoch = epochs[0]
        first_batch = sorted(self.weight_history[first_epoch].keys())[0]

        # Создаем начальные гистограммы
        for layer_name in self.layer_names:
            hist_data = self.weight_history[first_epoch][first_batch][layer_name]['hist']
            fig.add_trace(go.Bar(
                x=hist_data[1],
                y=hist_data[0],
                name=layer_name,
                visible=True if layer_name == self.layer_names[0] else False
            ))

        # Создаем кнопки для переключения между слоями
        buttons = []
        for idx, layer_name in enumerate(self.layer_names):
            visibility = [idx == j for j in range(len(self.layer_names))]
            buttons.append(dict(
                label=layer_name,
                method="update",
                args=[{"visible": visibility}]
            ))

        # Создаем frames для анимации
        frames = []
        slider_steps = []

        for epoch in epochs:
            for batch in sorted(self.weight_history[epoch].keys()):
                frame_data = []
                for layer_name in self.layer_names:
                    hist_data = self.weight_history[epoch][batch][layer_name]['hist']
                    frame_data.append(go.Bar(
                        x=hist_data[1],
                        y=hist_data[0],
                        name=layer_name,
                        visible=True if layer_name == self.layer_names[0] else False
                    ))

                frame_name = f'epoch_{epoch}_batch_{batch}'
                frames.append(go.Frame(data=frame_data, name=frame_name))

                # Добавляем шаг для слайдера
                slider_steps.append({
                    'args': [
                        [frame_name],
                        {'frame': {'duration': 0, 'redraw': True},
                         'mode': 'immediate',
                         'transition': {'duration': 0}}
                    ],
                    'label': f'E{epoch}B{batch}',
                    'method': 'animate'
                })

        fig.frames = frames

        # Обновляем layout
        fig.update_layout(
            updatemenus=[
                # Кнопки управления анимацией
                {
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        dict(label='Play',
                             method='animate',
                             args=[None, {'frame': {'duration': 500, 'redraw': True},
                                          'fromcurrent': True}]),
                        dict(label='Pause',
                             method='animate',
                             args=[[None], {'frame': {'duration': 0, 'redraw': True},
                                            'mode': 'immediate',
                                            'transition': {'duration': 0}}])
                    ],
                    'x': 0.1,
                    'y': 0
                },
                # Кнопки переключения между слоями
                {
                    'buttons': buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0,
                    'y': 0
                }
            ],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16},
                    'prefix': 'Current: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': slider_steps
            }],
            title_text="Weight Distribution Evolution"
        )

        return fig

# Создаем визуализатор
visualizer = BatchWeightVisualizer()

# Загружаем все чекпоинты
model_dir = 'D://Neuron/old_models'
visualizer.load_all_checkpoints(model_dir)

# Создаем и показываем анимацию эволюции весов
weight_evolution_fig = visualizer.create_weight_evolution_animation()
weight_evolution_fig.show()

# Показываем эволюцию метрик
metrics_fig = visualizer.plot_metrics_evolution()
metrics_fig.show()

# Показываем анимацию распределения весов
distribution_fig = visualizer.create_layer_weight_distribution_animation()
distribution_fig.show()
