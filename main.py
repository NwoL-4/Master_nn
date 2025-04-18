import os

import numpy as np

import torch

from constants import DTYPE
import functions
import torch_functions
from torch_classes import ElectricFieldDataset, ElectricFieldCNN, CustomDataLoader

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

main_dir = './Dataset'
# main_dir = 'D://Neuron/Dataset'
current_dir = '(0.0001, 0.001, 0.01)_(20, 20, 100)'
model_dir = 'LeakyReLU'

batch_size = 16
num_epochs = 100

train_size = 0.8    # Доля датасета для обучения

log_interval, save_interval = 50, 50

grid_size = tuple(np.fromstring(current_dir.split('_')[1][1:-1], sep=', ', dtype=np.int64).tolist())
space_size = tuple(np.fromstring(current_dir.split('_')[0][1:-1], sep=', ', dtype=DTYPE))
data_dir = os.path.join(main_dir, current_dir)
data_dir = os.path.abspath(data_dir)

print(f'Размеры пространства и число узлов:\n'
      f'x: {space_size[0]}\t---\t{grid_size[0]}\n'
      f'y: {space_size[1]}\t---\t{grid_size[1]}\n'
      f'z: {space_size[2]}\t---\t{grid_size[2]}\n')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Создание датасетов
dataset = ElectricFieldDataset(data_dir, grid_size, space_size)

train_size = int(train_size * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = CustomDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Создание модели
model = ElectricFieldCNN()

print(device)
# Проверка наличия сохраненной модели
start_epoch = 0
latest_model_path = functions.get_latest_model_path(model_dir)
if latest_model_path:
  print(f"Loading model from {latest_model_path}")
  checkpoint = torch.load(latest_model_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  start_epoch = checkpoint['epoch']
  print(f"Continuing from epoch {start_epoch}")

# Обучение
torch_functions.train_model(model, train_loader, val_loader, space_size,
                            num_epochs=num_epochs, device=device, model_dir=model_dir, start_epoch=start_epoch,
                            log_interval=log_interval, save_interval=save_interval)
