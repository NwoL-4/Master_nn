import glob
import os

import numpy as np
from tqdm import trange

import functions


def main():
    # main_path = os.getcwd()
    main_path = 'D:/Neuron/'
    mainFolder_name = 'Dataset'
    path_to_mainFolder = os.path.join(main_path, mainFolder_name)
    os.makedirs(path_to_mainFolder, exist_ok=True)

    SPACE = (1e-4, 1e-3, 1e-2)
    NNODES = (20, 20, 100)
    mM_space = np.array([
        [0, SPACE[0]],
        [0, SPACE[1]],
        [0, SPACE[2]]])

    MAX_PARTICLE = 30_000
    TOTAL_ITEMS = 100_000

    grid_nodes = functions.generate_grid_nodes(SPACE, NNODES)

    subFolder_name = f'{SPACE}_{NNODES}'

    path_to_subFolder = os.path.join(path_to_mainFolder, subFolder_name)
    os.makedirs(path_to_subFolder, exist_ok=True)

    existing_files = glob.glob(os.path.join(path_to_subFolder, "*.npz"))
    n_files = len(existing_files)

    for i in trange(n_files, TOTAL_ITEMS, initial=n_files, total=TOTAL_ITEMS):
        num_particles = np.random.randint(low=1, high=MAX_PARTICLE + 1)
        p_data = functions.generate_particles(num_particles=num_particles, bounds=mM_space)
        result_field = functions.field_grid_nodes(particle_data=p_data, grid_nodes=grid_nodes)

        np.savez(os.path.join(path_to_subFolder, f'{i}.npz'), particle=p_data, field=result_field)


if __name__ == '__main__':
    main()