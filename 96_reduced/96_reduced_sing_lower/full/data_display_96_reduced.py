import numpy as np
import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63
import csv

## 96 reduced

# 96 reduced generating og data
def lorenz_96_reduced_data_generation(K, dt, n_steps):

    X_0 = np.random.rand(K)

    traj_96_reduced = integrate_reduced_l96(X_0, dt, n_steps, F=20.)
    
    return traj_96_reduced

