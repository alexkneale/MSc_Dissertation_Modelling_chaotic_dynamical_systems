import numpy as np

import numba
from numba import jit

from l63 import integrate_l63

from pathlib import Path
import csv

n_traj = 1000
len_traj = 1000
len_interval = 10000
n_features = 3

MODEL_DIR_data = (f"data/{n_traj}_{len_traj}_{len_interval}/")
if not Path(MODEL_DIR_data).exists():
    Path(MODEL_DIR_data).mkdir(parents=True)

dt = 0.001
n_steps_init = (n_traj+1)*len_interval

def lorenz_63_data_generation(n_particles, dt, n_steps):

    pos_0 = np.random.rand(n_particles,3)

    traj_63 = integrate_l63(pos_0, dt, n_steps, sigma=10., rho=28., beta=8/3)
    
    return traj_63

# (n_inits, n_timesteps, 3)
traj_63 = lorenz_63_data_generation(1, dt, n_steps_init)
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
