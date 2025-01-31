import numpy as np
import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63
import csv


## 63

# 63 generating og data
def lorenz_63_data_generation(n_particles, dt, n_steps):

    pos_0 = np.random.rand(n_particles,3)

    traj_63 = integrate_l63(pos_0, dt, n_steps, sigma=10., rho=28., beta=8/3)
    
    return traj_63
