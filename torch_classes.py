import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

import functions
from constants import *


class ElectricFieldDataset(Dataset):
    def __init__(self, data_dir, grid_size, space_size):
        self.data_files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.grid_size = grid_size
        self.space_size = space_size

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        current_file = self.data_files[idx]
        try:

            with np.load(current_file) as data:
                particles = torch.from_numpy(data['particle']).type(TORCH_DTYPE)
                if 'charge' in list(data.keys()):
                    charge = np.expand_dims(data['charge'], axis=0)
                    particles = np.concatenate((particles, charge), axis=0)
                    particles = torch.from_numpy(particles).type(TORCH_DTYPE)

                field = torch.from_numpy(data['field']).type(TORCH_DTYPE)

            if torch.isnan(particles).any() or torch.isnan(particles[3]).any():
                return None

            # Преобразуем частицы в плотность заряда
            density = functions.particles_to_grid_density(particles, self.grid_size, self.space_size)
            if torch.isnan(density).any() or torch.isnan(field).any():
                print(f"Предупреждение: Значение равно NaN для файла {os.path.basename(current_file)}")
                return None

            return density.unsqueeze(0), field
        except Exception as e:
            print(f'Error processing file {os.path.basename(current_file)}: {str(e)}')


class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        batch = []
        for idx in self.indices:
            item = self.dataset[idx]
            if item is not None:  # Проверяем, что файл успешно загружен
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []

        if batch:  # Оставшиеся данные
            yield self._collate_batch(batch)

    def _collate_batch(self, batch):
        density = torch.stack([item[0] for item in batch])
        field = torch.stack([item[1] for item in batch])
        print()
        return density, field

    def __len__(self):
        return len(self.dataset) // self.batch_size


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, dtype=TORCH_DTYPE)
        self.bn1 = nn.BatchNorm3d(channels, dtype=TORCH_DTYPE)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, dtype=TORCH_DTYPE)
        self.bn2 = nn.BatchNorm3d(channels, dtype=TORCH_DTYPE)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Резидуальная связь
        out = F.relu(out)
        return out


class ElectricFieldCNN(nn.Module):
    def __init__(self):
        super(ElectricFieldCNN, self).__init__()

        # Начальная свертка
        self.initial_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, dtype=TORCH_DTYPE),
            nn.BatchNorm3d(32, dtype=TORCH_DTYPE),
            nn.LeakyReLU()
        )

        # Энкодер с резидуальными блоками
        self.encoder_res1 = ResBlock(32)
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1, dtype=TORCH_DTYPE),
            nn.BatchNorm3d(64, dtype=TORCH_DTYPE),
            nn.ReLU()
        )

        self.encoder_res2 = ResBlock(64)
        self.encoder_conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1, dtype=TORCH_DTYPE),
            nn.BatchNorm3d(128, dtype=TORCH_DTYPE),
            nn.ReLU()
        )

        self.encoder_res3 = ResBlock(128)

        # Декодер с резидуальными блоками
        self.decoder_res1 = ResBlock(128)
        self.decoder_conv1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1, dtype=TORCH_DTYPE),
            nn.BatchNorm3d(64, dtype=TORCH_DTYPE),
            nn.ReLU()
        )

        self.decoder_res2 = ResBlock(64)
        self.decoder_conv2 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1, dtype=TORCH_DTYPE),
            nn.BatchNorm3d(32, dtype=TORCH_DTYPE),
            nn.ReLU()
        )

        self.decoder_res3 = ResBlock(32)

        # Финальная свертка
        self.final_conv = nn.Conv3d(32, 3, kernel_size=3, padding=1, dtype=TORCH_DTYPE)

    def forward(self, x):
        # Начальная обработка
        x = self.initial_conv(x)

        # Энкодер
        x1 = self.encoder_res1(x)
        x = self.encoder_conv1(x1)

        x2 = self.encoder_res2(x)
        x = self.encoder_conv2(x2)

        x = self.encoder_res3(x)

        # Декодер с пропущенными связями
        x = self.decoder_res1(x)
        x = self.decoder_conv1(x)
        x = x + x2  # Резидуальная связь с энкодера

        x = self.decoder_res2(x)
        x = self.decoder_conv2(x)
        x = x + x1  # Резидуальная связь с энкодера

        x = self.decoder_res3(x)

        # Финальная свертка
        x = self.final_conv(x)

        return x