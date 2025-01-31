import numpy as np

import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63
from data_display_63 import lorenz_63_data_generation
import csv
import time



def x0_dx0_array_63(n_particles, dt, n_steps,N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str):
    # (n_inits, n_timesteps, 3)
    traj_63 = lorenz_63_data_generation(n_particles, dt, n_steps)

    
    
    

    X = traj_63[:,:-1,:]
    Y = traj_63[:,1:,:]-traj_63[:,:-1,:]
    
    X = X.reshape(-1, 3)
    Y = Y.reshape(-1, 3)
    
    del traj_63
    
    return X,Y
