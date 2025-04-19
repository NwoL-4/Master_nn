import datetime
import os
import platform
from pathlib import Path

import numpy as np

import torch
from torch import nn

import functions
import torch_functions
from constants import DTYPE
from torch_classes import ElectricFieldDataset, ElectricFieldCNN, CustomDataLoader


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class Config:
    class Paths:
        platform_name = platform.system().lower()

        window_dir = 'D://Neuron/Dataset'
        linux_dir = './Dataset'

        match platform_name:
            case 'windows':
                main_dir = window_dir
            case 'linux':
                main_dir = linux_dir
            case _:
                raise ValueError(f'Unknown platform: {platform_name}')
        dataset_dir = '(0.0001, 0.001, 0.01)_(20, 20, 100)'
        save_dir = './LeakyReLU'

    class ModelsConfig:
        learning_rate = 0.001
        batch_size = 16
        num_epochs = 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_size = 0.8

        log_interval = 50
        save_interval = 50

        dataset = ElectricFieldDataset


def main():

    config = Config()

    paths = config.Paths()
    model_config = config.ModelsConfig()

    space_size, grid_size = [np.fromstring(value[1:-1], sep=', ', dtype=dtype)
        for value, dtype in zip(paths.dataset_dir.split('_'), [DTYPE, np.int64])
    ]
    grid_size = tuple(grid_size.tolist())

    data_dir = os.path.join(paths.main_dir, paths.dataset_dir)

    dataset = model_config.dataset(data_dir, grid_size, space_size)

    train_size = int(model_config.train_size*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = CustomDataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
    val_loader = CustomDataLoader(val_dataset, batch_size=model_config.batch_size, shuffle=True)

    model = ElectricFieldCNN()

    if torch.cuda.device_count():
        model = nn.DataParallel(model)

    text = (f'{functions.log_message("START")}\n'
            f'Размеры пространства и число узлов:\n'
            f'x: {space_size[0]}\t---\t{grid_size[0]}\n'
            f'y: {space_size[1]}\t---\t{grid_size[1]}\n'
            f'z: {space_size[2]}\t---\t{grid_size[2]}\n\n'
            f'devices: {torch.cuda.device_count()}\n'
            f'device: {model_config.device}\n'
            f'platform: {paths.platform_name}\n\n'
            f'dataset path: {Path(dataset.data_files[0]).parent}\n'
            f'len dataset: {len(dataset)}\n'
            f'len trainset: {len(train_dataset)}\n'
            f'len valset: {len(val_dataset)}\n\n'
            f'batch size: {model_config.batch_size}\n'
            f'save path: {os.path.abspath(paths.save_dir)}\n'
            f'log path: {os.path.join(paths.save_dir, "log.txt")}\n\n'
            )
    start_epoch = 0
    latest_model_path = functions.get_latest_model_path(paths.save_dir)
    if latest_model_path:
        text += f'{datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S:.2f")}---Loading latest model from {latest_model_path}'
        checkpoint = torch.load(latest_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

        text += f'Continuing training from epoch {start_epoch}'

    os.makedirs(os.path.abspath(paths.save_dir), exist_ok=True)
    with open(os.path.abspath(os.path.join(paths.save_dir, 'log.txt')), 'w', encoding='utf-8') as f:
        f.write(text)

    torch_functions.train_model(model, train_loader, val_loader,
                                space_size,
                                l_rate=model_config.learning_rate, num_epochs=model_config.num_epochs,
                                device=model_config.device, start_epoch=start_epoch,
                                save_dir=paths.save_dir,
                                log_interval=model_config.log_interval, save_interval=model_config.save_interval,
                                text=text)


if __name__ == '__main__':
    main()
