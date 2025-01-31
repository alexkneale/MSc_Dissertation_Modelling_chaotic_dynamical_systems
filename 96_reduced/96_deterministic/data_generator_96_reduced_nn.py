import numpy as np

import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96


from data_display_96_reduced import lorenz_96_reduced_data_generation
from l63 import integrate_l63
import csv



def x0_dx0_array_96_reduced(dt, n_steps_train,N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,test_particles_str, train_particles, train_particles_str):
    
    X_0 = np.random.rand(K)
    n_steps_init = (train_particles+1)*10**4
    traj_96_reduced = integrate_reduced_l96(X_0, dt, n_steps_init, F=20.)
    init_positions = np.zeros((train_particles,K))
    for particle in range(1,train_particles+1):
        init_positions[particle-1,:] = traj_96_reduced[(particle*10**4)-1,:]
    
    del traj_96_reduced
    n_steps_particle = int(n_steps_train/train_particles)
    
    X = np.zeros(((n_steps_particle-1)*train_particles,K))
    Y = np.zeros(((n_steps_particle-1)*train_particles,K))

    i = 1
    for init_pos in init_positions:
        traj_96 = integrate_reduced_l96(init_pos, dt, n_steps_particle, F=20.)
        X[(n_steps_particle-1)*(i-1):(n_steps_particle-1)*i,:] = traj_96[:-1,:]
        Y[(n_steps_particle-1)*(i-1):(n_steps_particle-1)*i,:] = traj_96[1:,:]-traj_96[:-1,:]
        i += 1
    X_0 = np.random.rand(K)
    traj_96 = integrate_reduced_l96(X_0, dt, n_steps_train, F=20.)
    np.savetxt(f'models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{train_particles_str}_{test_particles_str}/traj_96.csv', traj_96, delimiter=',')

    del traj_96
    return X,Y
