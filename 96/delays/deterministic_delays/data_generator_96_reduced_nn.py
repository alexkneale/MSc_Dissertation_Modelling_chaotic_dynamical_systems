import numpy as np

import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96
from data_display_96_reduced import lorenz_96_reduced_data_generation
from data_display_96_reduced import lorenz_96_reduced_data_generation_new
from l63 import integrate_l63

from numpy.lib.stride_tricks import sliding_window_view

def x0_dx0_array_96_reduced(dt, n_steps_train,N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,J, train_particles_str, test_particles_str, train_particles, h, b, c,F,delay):
        
    MODEL_DIR_data_daughter = (f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/{delay}/")

    # returns 2 arrays of dimension (nsteps , K + K*J)
    n_steps_init = (train_particles+1)*10**4
    traj_96 = lorenz_96_reduced_data_generation(K, J, dt, n_steps_init, h,b,c,F)

    init_positions = np.zeros((train_particles,(J+1)*K*delay))
    for particle in range(1,train_particles+1):
        init_positions[particle-1,:] = traj_96[(particle*10**4)-1:(particle*10**4)-1+delay,:].flatten()
    del traj_96
    n_steps_particle = int(n_steps_train/train_particles)

    X = np.zeros(((n_steps_particle)*train_particles,K*delay))
    Y = np.zeros(((n_steps_particle)*train_particles,K))

    i = 1
    for init_pos in init_positions:
        traj_96 = lorenz_96_reduced_data_generation_new(K, J,init_pos[:K] ,init_pos[K:],dt, n_steps_particle+delay, h,b,c,F)
        traj_96 = traj_96[:,:K]
        sliding_windows = sliding_window_view(traj_96, (delay, K))

        # Select the first nsteps sliding windows
        selected_windows = sliding_windows[:n_steps_particle]

        # Reshape and flatten the selected windows to shape (nsteps, delay*K)
        result = selected_windows.reshape(n_steps_particle, delay * K)
        
        X[(n_steps_particle)*(i-1):(n_steps_particle)*i,:] = result
        Y[(n_steps_particle)*(i-1):(n_steps_particle)*i,:] = traj_96[delay:,:K]-traj_96[delay-1:-1,:K]
        i += 1

    del traj_96
    
    np.savetxt(MODEL_DIR_data_daughter + r"X.csv", X, delimiter=',')
    np.savetxt(MODEL_DIR_data_daughter + r"Y.csv", Y, delimiter=',')
    return X,Y
    
    