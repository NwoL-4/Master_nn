import os

import numpy as np
from scipy.constants import elementary_charge, electron_mass, Boltzmann
from tqdm.auto import tqdm

from analys.check_model.sub_classes import VariableLogger

variables = VariableLogger()

main_path = 'D://Neuron/check_model'
os.makedirs(main_path, exist_ok=True)


# Параметры частиц
ro_array = np.array([
    -0.1,
])
charge_array = np.array([
    -elementary_charge,
])
mass_array = np.array([
    electron_mass,
])
fraction = [100, ]
fraction_random = True

temperatures_electrodes = np.array([1000, 1000])

speed_const = True

average_speed = np.sqrt(3 / 2 * Boltzmann * temperatures_electrodes / mass_array)
sigma_speed = np.sqrt(Boltzmann * temperatures_electrodes / mass_array)


# Параметры пространства
width = 1e-1        # x
height = 1e-1       # y
length = 35e-3      # z


# Параметры полей
voltage_array = np.array([
    0,
    0,
    -0.05
])

magnetic_field = np.array([
    0,
    0,
    0
])

# Параметры симуляции
additional_particle = 30
initiate_points = 0

# fixed_steps = True
# total_steps = 1000

fixed_steps = False
finish_steps = 400


if ro_array.shape[0] != len(fraction):
    raise ValueError('Доли не совпадают с кол-вом частиц')
if sum(fraction) != 100 :
    raise ValueError('fraction в сумме должен давать 100 (процентов)')

for ro in tqdm(ro_array, total=ro_array.shape[0], desc='ro'):
    for voltage in tqdm(voltage_array, total=voltage_array.shape[0], desc=f'Voltage for ro={ro}', leave=False):

        electric_field = - voltage / np.array([width, height, length])