import numpy as np
import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63
import csv


def autocorrelation(x, max_lag):
    n = len(x)
    mean = np.mean(x)
    variance = np.var(x)
    autocorr = [1.0]  # R(0) is always 1

    for lag in range(1, max_lag + 1):
        cov = np.sum((x[:n-lag] - mean) * (x[lag:] - mean))
        autocorr.append(cov / (variance * (n - lag)))
    
    return autocorr

def autocovariance(x, max_lag):
    n = len(x)
    mean = np.mean(x)
    variance = np.var(x)
    autocorr = []  # R(0) is always 1

    for lag in range(0, max_lag + 1):
        cov = np.sum((x[:n-lag] - mean) * (x[lag:] - mean))
        autocorr.append(cov / ((n - lag)))
    
    return autocorr



## 63

# 63 generating og data
def lorenz_63_data_generation(n_particles, dt, n_steps):

    pos_0 = np.random.rand(n_particles,3)

    traj_63 = integrate_l63(pos_0, dt, n_steps, sigma=10., rho=28., beta=8/3)
    
    return traj_63
