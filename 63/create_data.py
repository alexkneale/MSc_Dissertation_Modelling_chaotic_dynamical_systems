import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import callbacks as cb
from pathlib import Path
import pickle


import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63

from preprocessing import Scaler
from data_generator_nn_63 import x0_dx0_array_63

from data_display_63 import lorenz_63_data_generation
import csv

# Model hyperparameters
N_C = 32
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**17)
LEARNING_RATE = 5e-5
LEARNING_RATE_str = '5e-5'
PATIENCE = 50
test_steps = int(1e7)
test_steps_str = 'e7'
n_particles_train = 100
n_particles_test = 100
str_n_particles_train = 't=e2e5'
str_n_particles_test = 'd=e2e5'
EPOCHS = 5000


MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")

MODEL_DIR_data = (f"data/{n_steps_train_str}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")
if not Path(MODEL_DIR_data).exists():
    Path(MODEL_DIR_data).mkdir(parents=True)

dt = 0.001

X,Y = x0_dx0_array_63(dt, n_steps_train,n_steps_train_str, test_steps,test_steps_str, str_n_particles_train,str_n_particles_test,n_particles_train,n_particles_test)

Xscaler = Scaler(X)
Yscaler = Scaler(Y)

del X, Y

with open(MODEL_DIR_data + r"Xscaler.pkl", "wb") as file:
    pickle.dump(Xscaler, file)

with open(MODEL_DIR_data + r"Yscaler.pkl", "wb") as file:
    pickle.dump(Yscaler, file)
    

    
# initial positions for data_collection
n_steps_init = (n_particles_test)*10**4
traj_63 = lorenz_63_data_generation(1, dt, n_steps_init)

init_positions = np.zeros((n_particles_test,3))
for particle in range(1,n_particles_test+1):
    init_positions[particle-1,:] = traj_63[:,(particle*10**4)-1,:]
np.savetxt(f'{MODEL_DIR_data}/init_pos.csv', init_positions, delimiter=',')
del traj_63
  
# initial positions for data_collection

traj_63 = lorenz_63_data_generation(1, dt, n_steps_train)
traj_63 = traj_63.squeeze()
np.savetxt(MODEL_DIR_data + r"traj_63.csv", traj_63, delimiter=',')


print('success')

