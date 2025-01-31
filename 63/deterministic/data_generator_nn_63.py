import numpy as np

import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63
from data_display_63 import lorenz_63_data_generation
import csv


def x0_dx0_array_63(dt, n_steps_train,n_steps_train_str, test_steps,test_steps_str, str_n_particles_train,str_n_particles_test,n_particles_train,n_particles_test):
    
    
    MODEL_DIR_data = (f"/work/sc130/sc130/akneale/63/data/{n_steps_train_str}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")
    
    n_steps_init = (n_particles_train+1)*10**4
    
    # (n_inits, n_timesteps, 3)
    traj_63 = lorenz_63_data_generation(1, dt, n_steps_init)
    
    init_positions = np.zeros((n_particles_train,3))
    for particle in range(1,n_particles_train+1):
        init_positions[particle-1,:] = traj_63[:,(particle*10**4)-1,:]
    del traj_63
    n_steps_particle = int(n_steps_train/n_particles_train)

    X = np.zeros(((n_steps_particle-1)*n_particles_train,3))
    Y = np.zeros(((n_steps_particle-1)*n_particles_train,3))
    i = 1
    for init_pos in init_positions:
        init_pos = init_pos.reshape(1, 3)
        traj_63 = integrate_l63(init_pos, dt, n_steps_particle, sigma=10., rho=28., beta=8/3)
        X[(n_steps_particle-1)*(i-1):(n_steps_particle-1)*i,:] = traj_63[:,:-1,:]
        Y[(n_steps_particle-1)*(i-1):(n_steps_particle-1)*i,:] = traj_63[:,1:,:]-traj_63[:,:-1,:]
        i += 1

    del traj_63
    
    np.savetxt(MODEL_DIR_data + r"X.csv", X, delimiter=',')
    np.savetxt(MODEL_DIR_data + r"Y.csv", Y, delimiter=',')
    return X,Y

