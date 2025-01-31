import numpy as np

import numba
from numba import jit
from l96 import integrate_reduced_l96
from pathlib import Path
import csv

n_traj = 1000
len_traj = 1000
len_interval = 10000
K = 8
n_features = K

MODEL_DIR_data = (f"data/{n_traj}_{len_traj}_{len_interval}/")
if not Path(MODEL_DIR_data).exists():
    Path(MODEL_DIR_data).mkdir(parents=True)

dt = 0.001
n_steps_init = (n_traj+1)*len_interval

def lorenz_96_reduced_data_generation(K, dt, n_steps):

    X_0 = np.random.rand(K)

    traj_96_reduced = integrate_reduced_l96(X_0, dt, n_steps, F=20.)
    
    return traj_96_reduced


# (n_inits, n_timesteps, 3)
traj_63 = lorenz_96_reduced_data_generation(K, dt, n_steps_init)
traj_63 = traj_63.squeeze()
# now has dimension (n_timesteps, n_features)

init_positions = np.zeros((n_traj,n_features))
end_positions = np.zeros((n_traj,n_features))

for traj in range(n_traj):
    init_positions[traj,:] = traj_63[(traj*len_interval),:]
    end_positions[traj,:] = traj_63[(traj*len_interval)+len_traj,:]

np.savetxt(MODEL_DIR_data + r"init_pos.csv", init_positions, delimiter=',')
np.savetxt(MODEL_DIR_data + r"end_pos.csv", end_positions, delimiter=',')
print('success')
