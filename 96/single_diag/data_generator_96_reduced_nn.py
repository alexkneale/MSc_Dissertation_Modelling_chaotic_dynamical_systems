import numpy as np

import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96
from data_display_96_reduced import lorenz_96_reduced_data_generation
from data_display_96_reduced import lorenz_96_reduced_data_generation_new
from l63 import integrate_l63


def x0_dx0_array_96_reduced(dt, n_steps_train,N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,J, train_particles_str,test_particles_str,train_particles, h, b, c,F):
    
    MODEL_DIR_data = (f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/")
    
    # returns 2 arrays of dimension (nsteps , K + K*J)
    n_steps_init = (train_particles+1)*10**1
    traj_96 = lorenz_96_reduced_data_generation(K, J, dt, n_steps_init, h,b,c,F)

    init_positions = np.zeros((train_particles,(J+1)*K))
    for particle in range(1,train_particles+1):
        init_positions[particle-1,:] = traj_96[(particle*10**1)-1,:]
    del traj_96
    n_steps_particle = int(n_steps_train/train_particles)

    X = np.zeros(((n_steps_particle-1)*train_particles,K))
    Y = np.zeros(((n_steps_particle-1)*train_particles,K))

    i = 1
    for init_pos in init_positions:
        traj_96 = lorenz_96_reduced_data_generation_new(K, J,init_pos[:K] ,init_pos[K:],dt, n_steps_particle, h,b,c,F)
        X[(n_steps_particle-1)*(i-1):(n_steps_particle-1)*i,:] = traj_96[:-1,:K]
        Y[(n_steps_particle-1)*(i-1):(n_steps_particle-1)*i,:] = traj_96[1:,:K]-traj_96[:-1,:K]
        i += 1

    del traj_96
    
    np.savetxt(MODEL_DIR_data + r"X.csv", X, delimiter=',')
    np.savetxt(MODEL_DIR_data + r"Y.csv", Y, delimiter=',')
    return X,Y
    