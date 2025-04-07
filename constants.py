import numpy as np
import torch


# Конфигурация точности вычислений
DTYPE = np.float64

TORCH_DTYPE = torch.double if DTYPE is np.float64 else torch.float