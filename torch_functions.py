import glob
import os
import time

import torch
from scipy.constants import epsilon_0
from torch import nn, optim

import functions
from constants import *

EPSILON = 1e-10
K_CONST = 4 * np.pi * epsilon_0


def check_poisson_equation(charge_density: torch.Tensor,
                           e_field: torch.Tensor,
                           space_size: tuple[float, float, float],
                           device) -> torch.Tensor:
    """
    Проверяет уравнение Пуассона: ∇²φ = -ρ * coef
    где φ получаем интегрированием E = -∇φ

    Args:
        charge_density: Тензор плотности заряда (nx, ny, nz)
        e_field: Тензор электрического поля (3, nx, ny, nz)
        space_size: Размеры пространства (Lx, Ly, Lz)

    Returns:
        torch.Tensor: Относительная ошибка уравнения Пуассона

    """
    batch_size, _, nx, ny, nz = charge_density.size()
    dx = space_size[0] / (nx - 1)
    dy = space_size[1] / (ny - 1)
    dz = space_size[2] / (nz - 1)

    # 1. Восстанавливаем потенциал из электрического поля
    # E = -∇φ => φ = -∫E·dr

    # Интегрирование по x
    potential_x = torch.zeros_like(charge_density).to(device)
    for i in range(1, nx):
        potential_x[:, :, i, :, :] = potential_x[:, :, i - 1, :, :] - e_field[:, 0, i, :, :].unsqueeze(1) * dx

    # Интегрирование по y
    potential_y = torch.zeros_like(charge_density).to(device)
    for j in range(1, ny):
        potential_y[:, :, :, j, :] = potential_y[:, :, :, j - 1, :] - e_field[:, 1, :, j, :].unsqueeze(1) * dy

    # Интегрирование по z
    potential_z = torch.zeros_like(charge_density).to(device)
    for k in range(1, nz):
        potential_z[:, :, :, :, k] = potential_z[:, :, :, :, k - 1] - e_field[:, 2, :, :, k].unsqueeze(1) * dz

    # Вычисляем взвешенное среднее с учетом расстояния от начала координат
    x = torch.linspace(0, space_size[0], nx, dtype=TORCH_DTYPE).to(device)
    y = torch.linspace(0, space_size[1], ny, dtype=TORCH_DTYPE).to(device)
    z = torch.linspace(0, space_size[2], nz, dtype=TORCH_DTYPE).to(device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Расширяем размерности для батча
    X = X.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1, -1).to(device)
    Y = Y.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1, -1).to(device)
    Z = Z.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1, -1).to(device)

    # Веса обратно пропорциональны расстоянию от начала интегрирования
    wx = 1.0 / (X + EPSILON)
    wy = 1.0 / (Y + EPSILON)
    wz = 1.0 / (Z + EPSILON)

    # Нормализация весов
    w_sum = wx + wy + wz
    wx = (wx / w_sum).to(device)
    wy = (wy / w_sum).to(device)
    wz = (wz / w_sum).to(device)

    # Взвешенное среднее
    potential = (wx * potential_x + wy * potential_y + wz * potential_z)

    # 2. Вычисляем лапласиан потенциала (∇²φ)
    # Используем схему 4-го порядка для внутренних точек
    laplacian = torch.zeros_like(potential).to(device)

    # Внутренние точки (4-й порядок точности)
    laplacian[:, :, 2:-2, 2:-2, 2:-2] = (
        # По x
        (-potential[:, :, 4:, 2:-2, 2:-2] + 16 * potential[:, :, 3:-1, 2:-2, 2:-2] -
         30 * potential[:, :, 2:-2, 2:-2, 2:-2] + 16 * potential[:, :, 1:-3, 2:-2, 2:-2] -
         potential[:, :, :-4, 2:-2, 2:-2]) / (12 * dx ** 2) +
        # По y
        (-potential[:, :, 2:-2, 4:, 2:-2] + 16 * potential[:, :, 2:-2, 3:-1, 2:-2] -
         30 * potential[:, :, 2:-2, 2:-2, 2:-2] + 16 * potential[:, :, 2:-2, 1:-3, 2:-2] -
         potential[:, :, 2:-2, :-4, 2:-2]) / (12 * dy ** 2) +
        # По z
        (-potential[:, :, 2:-2, 2:-2, 4:] + 16 * potential[:, :, 2:-2, 2:-2, 3:-1] -
         30 * potential[:, :, 2:-2, 2:-2, 2:-2] + 16 * potential[:, :, 2:-2, 2:-2, 1:-3] -
         potential[:, :, 2:-2, 2:-2, :-4]) / (12 * dz ** 2)
    )

    # Граничные точки (2-й порядок точности)
    for i in [0, -1]:
        for j in range(ny):
            for k in range(nz):
                laplacian[:, :, i, j, k] = (
                        (potential[:, :, (i + 1) % nx, j, k] - 2 * potential[:, :, i, j, k] +
                         potential[:, :, i - 1, j, k]) / dx ** 2 +
                        (potential[:, :, i, (j + 1) % ny, k] - 2 * potential[:, :, i, j, k] +
                         potential[:, :, i, (j - 1) % ny, k]) / dy ** 2 +
                        (potential[:, :, i, j, (k + 1) % nz] - 2 * potential[:, :, i, j, k] +
                         potential[:, :, i, j, (k - 1) % nz]) / dz ** 2
                )

    # 3. Проверяем уравнение Пуассона
    theoretical = (-charge_density / epsilon_0).to(device)

    # Вычисляем относительную ошибку
    error = torch.abs(laplacian - theoretical) / (torch.abs(theoretical) + EPSILON)

    return error


def check_gauss_law(charge_density: torch.Tensor,
                    e_field: torch.Tensor,
                    space_size: tuple[float, float, float],
                    device) -> torch.Tensor:
    """
    Проверяет закон Гаусса: div E = ρ*const с учетом батчей

    Args:
        charge_density: [batch_size, 1, nx, ny, nz]
        e_field: [batch_size, 3, nx, ny, nz]
        space_size: (Lx, Ly, Lz)
    Returns:
        torch.Tensor: Относительная ошибка в каждой точке
    """
    batch_size, _, nx, ny, nz = charge_density.size()

    dx = space_size[0] / (nx - 1)
    dy = space_size[1] / (ny - 1)
    dz = space_size[2] / (nz - 1)

    # Используем схему 4-го порядка для дивергенции
    div_E = torch.zeros_like(charge_density).to(device)

    # Внутренние точки (4-й порядок)
    div_E[:, 0, 2:-2, 2:-2, 2:-2] = (
            (-e_field[:, 0, 4:, 2:-2, 2:-2] + 8 * e_field[:, 0, 3:-1, 2:-2, 2:-2] -
             8 * e_field[:, 0, 1:-3, 2:-2, 2:-2] + e_field[:, 0, :-4, 2:-2, 2:-2]) / (12 * dx) +
            (-e_field[:, 1, 2:-2, 4:, 2:-2] + 8 * e_field[:, 1, 2:-2, 3:-1, 2:-2] -
             8 * e_field[:, 1, 2:-2, 1:-3, 2:-2] + e_field[:, 1, 2:-2, :-4, 2:-2]) / (12 * dy) +
            (-e_field[:, 2, 2:-2, 2:-2, 4:] + 8 * e_field[:, 2, 2:-2, 2:-2, 3:-1] -
             8 * e_field[:, 2, 2:-2, 2:-2, 1:-3] + e_field[:, 2, 2:-2, 2:-2, :-4]) / (12 * dz)
    )

    theoretical = (charge_density / epsilon_0).to(device)
    error = torch.abs(div_E - theoretical) / (torch.abs(theoretical) + EPSILON)

    return error


def train_model(model,
                train_loader,
                val_loader,
                space_size,
                l_rate=0.001,
                num_epochs=100,
                device=torch.device('cuda'),
                save_dir='models',
                start_epoch=0,
                log_interval=10,
                save_interval=10,
                text=''):

    val_dir = os.path.join(save_dir, 'validation')
    os.makedirs(val_dir, exist_ok=True)

    text += f"{functions.log_message('Started training model')}"

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=l_rate)

    model_files = glob.glob(os.path.join(save_dir, "cnn_model_*.pth"))
    last_save_num = len(model_files)

    # Очищаем память перед началом обучения
    if device == 'cuda':
        torch.cuda.empty_cache()

    start_time = time.time()

    text += f"{functions.log_message('Started epochs model')}\n"

    for epoch in range(start_epoch, num_epochs, 1):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()

        train_losses = 0
        gauss_losses = 0
        poisson_losses = 0
        batch_count = 0

        for batch_idx, (density, train_field) in enumerate(train_loader):
            try:
                if batch_idx % log_interval == 0: # Логируем каждые 10 батчей
                    end_time = time.time()
                    print(f"Processing batch {batch_idx}/{len(train_loader)}: {end_time - start_time} c.")
                    start_time = time.time()

                density = density.to(device)
                train_field = train_field.to(device)

                optimizer.zero_grad()

                # Прямой проход
                model_field = model(density)

                # Вычисление потерь
                mse_loss = criterion(model_field, train_field)
                gauss_loss = check_gauss_law(density, model_field / K_CONST, space_size, device)
                poisson_loss = check_poisson_equation(density, model_field / K_CONST, space_size, device)

                # Общая функция потерь (с весами)
                total_loss = (
                    mse_loss +
                    # 0.1 * poisson_loss.mean()
                    # 0.1 * gauss_loss.mean()
                    0
                )

                total_loss.backward()
                optimizer.step()

                train_losses += mse_loss.item()
                poisson_losses += poisson_loss.mean()
                gauss_losses += gauss_loss.mean()

                batch_count += 1

                if batch_idx > 0 and batch_idx % save_interval == 0:
                    last_save_num += 1
                    save_path = os.path.join(save_dir, f"cnn_model_{last_save_num}.pth")
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_losses / train_loader.batch_size / save_interval,
                        'gauss_loss': gauss_losses / train_loader.batch_size / save_interval,
                        'poisson_loss': poisson_losses / train_loader.batch_size / save_interval
                    }, save_path)
                    print(f'Model saved: {save_path}')

                    train_losses = 0
                    gauss_losses = 0
                    poisson_losses = 0

                # Очищаем неиспользуемые тензоры
                del model_field, mse_loss, poisson_loss, gauss_loss, total_loss
                if device == 'cuda':
                    torch.cuda.empty_cache()

                text += f"{functions.log_message(f'e{epoch}b{batch_idx} success')}"
                with open(os.path.abspath(os.path.join(save_dir, 'log.txt')), 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception as e:
                # print(f"Error in batch {batch_idx}: {e}")
                raise ValueError(e)
                # continue

        # Валидация
        model.eval()

        val_loss = dict(
            mse_loss=0,
            poisson_loss=0,
            gauss_loss=0,
        )
        with torch.no_grad():
            for val_idx, (density, train_field) in enumerate(val_loader):
                density = density.to(device)

                train_field = train_field.to(device)
                model_field = model(density)

                val_loss['mse_loss'] += criterion(model_field, train_field).item()
                val_loss['gauss_loss'] += check_gauss_law(density, model_field / K_CONST, space_size, device)
                val_loss['poisson_loss'] += check_poisson_equation(density, model_field / K_CONST, space_size, device)

                # Очищаем память после валидации
                del model_field
                if device == 'cuda':
                    torch.cuda.empty_cache()
                text += f"{functions.log_message(f'e{epoch}v{val_idx} success')}\n"
            torch.save({
                'val_loss': val_loss
            }, os.path.join(val_dir, f'val_loss_{epoch}.pth'))
            with open(os.path.abspath(os.path.join(save_dir, 'log.txt')), 'w', encoding='utf-8') as f:
                f.write(text)
