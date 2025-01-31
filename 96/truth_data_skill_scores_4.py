import numpy as np

import numba
from numba import jit
from l96 import integrate_l96
from pathlib import Path
import csv

n_traj = 1000
len_traj = 1000
len_interval = 10000
K = 8
n_features = K
J = 32
h= 5.
b = 10.
c = 2.
F = 20.
delay = 2

MODEL_DIR_data = (f"data/{n_traj}_{len_traj}_{len_interval}_{h}_{b}_{c}_{F}_{delay}/")
if not Path(MODEL_DIR_data).exists():
    Path(MODEL_DIR_data).mkdir(parents=True)

dt = 0.001
n_steps_init = (n_traj+1)*len_interval

X = np.loadtxt(f'/work/sc130/sc130/akneale/96/data/e7_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/traj_96.csv', delimiter=',')


X = X.squeeze()
# now has dimension (n_timesteps, n_features)

init_positions = np.zeros((n_traj,n_features*delay))
end_positions = np.zeros((n_traj,n_features))

for traj in range(n_traj):
    init_positions[traj,:] = X[(traj*len_interval):(traj*len_interval)+2,:].flatten()
    end_positions[traj,:] = X[(traj*len_interval)+len_traj,:]

np.savetxt(MODEL_DIR_data + r"init_pos.csv", init_positions, delimiter=',')
np.savetxt(MODEL_DIR_data + r"end_pos.csv", end_positions, delimiter=',')
print('success')
