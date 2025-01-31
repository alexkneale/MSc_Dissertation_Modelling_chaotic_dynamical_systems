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



## 96 

# 96 generating og data
def lorenz_96_reduced_data_generation(K, J, dt, n_steps, h,b,c,F):

    X_0 = np.random.rand(K)
    Y_0 = np.random.rand(K*J)

    X,Y = integrate_l96(X_0, Y_0, dt, n_steps, h, F, b, c)

    traj_96 = np.concatenate((X,Y), axis = 1)
    return traj_96
    # returns array of dimension (nsteps , K + K*J)

def lorenz_96_reduced_data_generation_new(K, J, X_0,Y_0,dt, n_steps, h,b,c,F):


    X,Y = integrate_l96(X_0, Y_0, dt, n_steps, h, F, b, c)

    traj_96 = np.concatenate((X,Y), axis = 1)
    return traj_96
    # returns array of dimension (nsteps , K + K*J)
