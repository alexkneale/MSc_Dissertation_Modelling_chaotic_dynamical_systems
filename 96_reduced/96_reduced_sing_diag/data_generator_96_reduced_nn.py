import numpy as np

import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96
from data_display_96_reduced import lorenz_96_reduced_data_generation
from l63 import integrate_l63
import csv



def x0_dx0_array_96_reduced(dt, n_steps_train,N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,test_particles_str):
    
    traj_96 = lorenz_96_reduced_data_generation(K, dt, n_steps_train)
    
    
    np.savetxt(f'models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{test_particles_str}/traj_96.csv', traj_96, delimiter=',')

    X = traj_96[:-1,:]
    Y = traj_96[1:,:]-traj_96[:-1,:]

    del traj_96
    return X,Y
