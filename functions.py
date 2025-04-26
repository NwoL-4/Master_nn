import datetime
import glob
import os

import torch
from numba import prange, njit
import numpy as np

from constants import DTYPE, TORCH_DTYPE


def log_message(text):
    return f'{datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}---{text}\n'


def get_latest_model_path(model_dir):
    """Находит путь к последней сохраненной модели"""
    model_files = glob.glob(os.path.join(model_dir, "cnn_model_*.pth"))
    if not model_files:
        return None

    # Извлекаем номера из имен файлов и находим максимальный
    numbers = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    latest_number = max(numbers)
    return os.path.join(model_dir, f'cnn_model_{latest_number}.pth')


def particles_to_grid_density(particles,
                              grid_size: tuple[np.int64, np.int64, np.int64],
                              space_size: tuple[np.float64, np.float64, np.float64]) -> torch.Tensor:
    """
    Конвертирует координаты и заряды частиц в плотность заряда на сетке при помощи билинейной интерполяции.

    Args:
        particles: Тензор размера (4, N), где N - число частиц.
                  particles[0:3] - координаты (x,y,z)
                  particles[3] - заряды частиц
        grid_size: Кортеж (nx, ny, nz) - количество узлов сетки по каждой оси
        space_size: Кортеж (Lx, Ly, Lz) - физические размеры пространства

    Returns:
        torch.Tensor: Тензор плотности заряда размера (nx, ny, nz)
    """

    # Создаем пустую сетку для плотности заряда
    charge_density = torch.zeros(grid_size, device=particles.device, dtype=TORCH_DTYPE)

    # Вычисляем шаги сетки по каждому направлению
    dx = space_size[0] / (grid_size[0] - 1)
    dy = space_size[1] / (grid_size[1] - 1)
    dz = space_size[2] / (grid_size[2] - 1)

    # Получаем координаты и заряды
    x, y, z = particles[0].clone().detach(), particles[1].clone().detach(), particles[2].clone().detach()
    charges = particles[3].clone().detach()

    # Преобразуем физические координаты в индексы сетки
    # и используем билинейную интерполяцию
    x_idx = x / dx
    y_idx = y / dy
    z_idx = z / dz

    # Находим ближайшие узлы сетки (может взять round??)
    x0 = torch.floor(x_idx).long()
    y0 = torch.floor(y_idx).long()
    z0 = torch.floor(z_idx).long()

    # Вычисляем веса для интерполяции
    wx = x_idx - x0
    wy = y_idx - y0
    wz = z_idx - z0

    # Обеспечиваем, чтобы индексы не выходили за пределы сетки
    x0 = torch.clamp(x0, 0, grid_size[0] - 2)
    y0 = torch.clamp(y0, 0, grid_size[1] - 2)
    z0 = torch.clamp(z0, 0, grid_size[2] - 2)

    # Распределяем заряд по соседним узлам с учетом весов
    for i in range(len(charges)):
        # Веса для 8 ближайших узлов
        w000 = (1 - wx[i]) * (1 - wy[i]) * (1 - wz[i])
        w001 = (1 - wx[i]) * (1 - wy[i]) * wz[i]
        w010 = (1 - wx[i]) * wy[i] * (1 - wz[i])
        w011 = (1 - wx[i]) * wy[i] * wz[i]
        w100 = wx[i] * (1 - wy[i]) * (1 - wz[i])
        w101 = wx[i] * (1 - wy[i]) * wz[i]
        w110 = wx[i] * wy[i] * (1 - wz[i])
        w111 = wx[i] * wy[i] * wz[i]

        # Добавляем вклад заряда в соответствующие узлы
        charge_density[x0[i], y0[i], z0[i]] += charges[i] * w000
        charge_density[x0[i], y0[i], z0[i] + 1] += charges[i] * w001
        charge_density[x0[i], y0[i] + 1, z0[i]] += charges[i] * w010
        charge_density[x0[i], y0[i] + 1, z0[i] + 1] += charges[i] * w011
        charge_density[x0[i] + 1, y0[i], z0[i]] += charges[i] * w100
        charge_density[x0[i] + 1, y0[i], z0[i] + 1] += charges[i] * w101
        charge_density[x0[i] + 1, y0[i] + 1, z0[i]] += charges[i] * w110
        charge_density[x0[i] + 1, y0[i] + 1, z0[i] + 1] += charges[i] * w111

    return charge_density


def optimize_particles_to_grid_density(particles,
                                       grid_size: tuple[np.int64, np.int64, np.int64],
                                       space_size: tuple[np.float64, np.float64, np.float64]) -> torch.Tensor:
    """
    Оптимизированная версия конвертации координат и зарядов частиц в плотность заряда на сетке.
    Использует векторизованные операции для эффективного выполнения на GPU/CPU.

    Args:
        particles: Тензор размера (4, N), где N - число частиц.
                  particles[0:3] - координаты (x,y,z)
                  particles[3] - заряды частиц
        grid_size: Кортеж (nx, ny, nz) - количество узлов сетки по каждой оси
        space_size: Кортеж (Lx, Ly, Lz) - физические размеры пространства

    Returns:
        torch.Tensor: Тензор плотности заряда размера (nx, ny, nz)
    """
    device = particles.device
    dtype = particles.dtype
    num_particles = particles.shape[1]

    # Вычисляем шаги сетки
    dx = space_size[0] / (grid_size[0] - 1)
    dy = space_size[1] / (grid_size[1] - 1)
    dz = space_size[2] / (grid_size[2] - 1)

    # Получаем координаты и заряды
    x, y, z = particles[0], particles[1], particles[2]
    charges = particles[3]

    # Преобразуем физические координаты в индексы сетки
    x_idx = x / dx
    y_idx = y / dy
    z_idx = z / dz

    # Находим базовые индексы
    x0 = torch.floor(x_idx).long()
    y0 = torch.floor(y_idx).long()
    z0 = torch.floor(z_idx).long()

    # Ограничиваем индексы
    x0 = torch.clamp(x0, 0, grid_size[0] - 2)
    y0 = torch.clamp(y0, 0, grid_size[1] - 2)
    z0 = torch.clamp(z0, 0, grid_size[2] - 2)

    # Вычисляем веса для интерполяции
    wx = x_idx - x0
    wy = y_idx - y0
    wz = z_idx - z0

    # Создаем тензор для хранения результата
    charge_density = torch.zeros(*grid_size, device=device, dtype=dtype)

    # Создаем индексы для всех 8 соседних узлов для каждой частицы
    x_indices = torch.stack([x0, x0, x0, x0, x0 + 1, x0 + 1, x0 + 1, x0 + 1])
    y_indices = torch.stack([y0, y0, y0 + 1, y0 + 1, y0, y0, y0 + 1, y0 + 1])
    z_indices = torch.stack([z0, z0 + 1, z0, z0 + 1, z0, z0 + 1, z0, z0 + 1])

    # Вычисляем веса для всех 8 узлов
    weights = torch.stack([
        (1 - wx) * (1 - wy) * (1 - wz),
        (1 - wx) * (1 - wy) * wz,
        (1 - wx) * wy * (1 - wz),
        (1 - wx) * wy * wz,
        wx * (1 - wy) * (1 - wz),
        wx * (1 - wy) * wz,
        wx * wy * (1 - wz),
        wx * wy * wz
    ])

    # Умножаем веса на заряды
    weighted_charges = weights * charges.unsqueeze(0)

    # Преобразуем трехмерные индексы в одномерные для scatter_add_
    flat_indices = (x_indices * (grid_size[1] * grid_size[2]) +
                    y_indices * grid_size[2] +
                    z_indices)

    # Используем scatter_add_ для параллельного обновления плотности заряда
    flat_density = charge_density.reshape(-1)
    for i in range(8):
        flat_density.scatter_add_(0, flat_indices[i], weighted_charges[i])

    return charge_density


@njit(parallel=True, cache=True)
def generate_particles(num_particles: int, bounds: np.ndarray) -> np.ndarray:
    """Генерация частиц в заданных границах с использованием параллелизации.

    Args:
        num_particles: количество генерируемых частиц
        bounds: матрица границ [3x2] для x,y,z

    Returns:
        Массив координат частиц [3xN]
    """
    particles = np.empty((4, num_particles), dtype=DTYPE)
    for axis in prange(3):
        particles[axis] = np.random.uniform(low=bounds[axis, 0],
                                            high=bounds[axis, 1],
                                            size=num_particles)
    particles[3] = np.random.choice(np.array([-1, 1]), num_particles)
    return particles


@njit(parallel=True, cache=True)
def field_grid_nodes(particle_data: np.ndarray, grid_nodes: np.ndarray) -> np.ndarray:
    """
    Вычисление поля на узлах сетки.

    Parameters:
    - particles: Тензор размера (4, N), где N - число частиц.
                  particles[0:3] - координаты (x,y,z)
                  particles[3] - заряды частиц
    - grid_nodes: узлы сетки (3D-матрица с координатами узлов)

    Returns:
    - field_grid: поле в узлах сетки
    """
    field_grid = np.zeros_like(grid_nodes, dtype=DTYPE)

    for x in prange(grid_nodes.shape[1]):
        for y in prange(grid_nodes.shape[2]):
            for z in prange(grid_nodes.shape[3]):
                node = grid_nodes[:, x, y, z]
                for index_particle in prange(particle_data.shape[1]):
                    radius_part_node = node - particle_data[:3, index_particle]
                    distance = np.sqrt(np.sum(radius_part_node ** 2))
                    distance = np.where(distance <= 1e-13, 1e-13, distance)
                    field_grid[:, x, y, z] += particle_data[3, index_particle] * radius_part_node / (distance ** 3)

    return field_grid


def generate_grid_nodes(space: tuple, n_nodes: tuple) -> np.ndarray:
    """
    Генерация узлов сетки.

    Parameters:
    - space: размеры пространства
    - n_nodes: количество узлов в каждом измерении

    Returns:
    - grid_nodes: узлы сетки
    """
    x_space, y_space, z_space = [np.linspace(0, space[i], n_nodes[i], endpoint=True, dtype=DTYPE)
                                 for i in range(3)]
    X_space, Y_space, Z_space = np.meshgrid(x_space, y_space, z_space, indexing='ij')
    grid_nodes = np.array([X_space, Y_space, Z_space])
    return grid_nodes
