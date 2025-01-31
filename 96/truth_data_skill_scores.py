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
h= 1.
b = 10.
c = 10.
F = 20.


MODEL_DIR_data = (f"data/{n_traj}_{len_traj}_{len_interval}_{h}_{b}_{c}_{F}/")
if not Path(MODEL_DIR_data).exists():
    Path(MODEL_DIR_data).mkdir(parents=True)

dt = 0.001
n_steps_init = (n_traj+1)*len_interval



X_0 = np.random.rand(K)
Y_0 = np.random.rand(K*J)
# (n_inits, n_timesteps, 3)
X,Y = integrate_l96(X_0, Y_0, dt, n_steps_init, h, F, b, c)
del Y

X = X.squeeze()
# now has dimension (n_timesteps, n_features)

init_positions = np.zeros((n_traj,n_features))
end_positions = np.zeros((n_traj,n_features))

for traj in range(n_traj):
    init_positions[traj,:] = X[(traj*len_interval),:]
    end_positions[traj,:] = X[(traj*len_interval)+len_traj,:]

np.savetxt(MODEL_DIR_data + r"init_pos.csv", init_positions, delimiter=',')
np.savetxt(MODEL_DIR_data + r"end_pos.csv", end_positions, delimiter=',')
print('success')
