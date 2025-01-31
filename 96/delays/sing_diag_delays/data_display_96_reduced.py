import numpy as np
import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63
import csv

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
