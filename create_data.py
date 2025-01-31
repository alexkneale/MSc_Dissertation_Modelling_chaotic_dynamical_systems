import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import callbacks as cb
from pathlib import Path
import pickle
tf.keras.backend.set_floatx("float64")


import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63

from preprocessing import Scaler
from data_generator_96_reduced_nn import x0_dx0_array_96_reduced

from data_display_96_reduced import lorenz_96_reduced_data_generation

import csv


# Model hyperparameters

N_C = 32
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**17)
LEARNING_RATE = 1e-5
LEARNING_RATE_str = 'e-5'
PATIENCE = 150
test_steps = 1e7
test_steps_str = 'e7'
K = 8
test_particles = 100
test_particles_str = 'd=e2e5'
train_particles = 100
train_particles_str = 't=e2e5'
EPOCHS = 5000

MODEL_DIR_data = (f"data/{n_steps_train_str}_{test_steps_str}_{K}_{train_particles_str}_{test_particles_str}/")

if not Path(MODEL_DIR_data).exists():
    Path(MODEL_DIR_data).mkdir(parents=True)


# --- PREPARE DATA ---

dt = 0.001
X,Y = x0_dx0_array_96_reduced(dt, n_steps_train,N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K, train_particles_str, test_particles_str, train_particles)

Xscaler = Scaler(X)
Yscaler = Scaler(Y)

del X, Y

with open(MODEL_DIR_data + r"Xscaler.pkl", "wb") as file:
    pickle.dump(Xscaler, file)

with open(MODEL_DIR_data + r"Yscaler.pkl", "wb") as file:
    pickle.dump(Yscaler, file)
    
# initial positions for data_collection
# returns 2 arrays of dimension (nsteps , K)
n_steps_init = (test_particles+1)*10**4
traj_96 = lorenz_96_reduced_data_generation(K, dt, n_steps_init)

init_positions = np.zeros((train_particles,K))
for particle in range(1,train_particles+1):
    init_positions[particle-1,:] = traj_96[(particle*10**4)-1,:]
np.savetxt(f'{MODEL_DIR_data}/init_pos.csv', init_positions, delimiter=',')
    
# traj_96 generation
traj_96 = lorenz_96_reduced_data_generation(K, dt, n_steps_train)
np.savetxt(MODEL_DIR_data + r"traj_96.csv", traj_96, delimiter=',')

print('success')

