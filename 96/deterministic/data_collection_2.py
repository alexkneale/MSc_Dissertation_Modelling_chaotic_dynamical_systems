import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import callbacks as cb
from pathlib import Path
from utils import load_mdn, load_scalers
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

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


#parameters
N_C = 32
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**17)
LEARNING_RATE = 5e-5
LEARNING_RATE_str = '5e-5'
PATIENCE = 50
test_steps = int(1e7)
test_steps_str = 'e7'
K = 8
J = 32
test_particles = 100
test_particles_str = 'd=e2e5'
train_particles = 100
train_particles_str = 't=e2e5'
EPOCHS = 3
h = 1.
b = 10.
c = 4.
F = 20.

MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/")
MODEL_DIR_data = (f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/")
dt = 0.001
n_steps = int(test_steps/test_particles)+10**3

model = load_mdn(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,J,train_particles_str,test_particles_str, h, b, c,F)
Xscaler, Yscaler = load_scalers(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,J,train_particles_str,test_particles_str, h, b, c,F)


print("Model loaded.")


@tf.function
def loop_func(Xscaler, Yscaler, model, X, X_):
    dx_ = model(X_)
    dx = Yscaler.invert_standardisation(dx_)
    X = Xscaler.invert_standardisation(X_)
    X = X + dx
    X_ = Xscaler.standardise(X)
    return X, X_


def neural_network_trajectories(model, test_particles, K, J, dt, n_steps, Xscaler, Yscaler):
    # Initialize a TensorFlow tensor to store trajectories
    traj_nn = tf.TensorArray(dtype=tf.float64, size=n_steps - 10**3)
        
    particle_arr = np.loadtxt(f'{MODEL_DIR_data}init_pos.csv', delimiter=',')
    particle_arr = particle_arr[:,:K]
    
    
    X = tf.constant(particle_arr, dtype=tf.float64)
    X_ = tf.constant(Xscaler.standardise(X.numpy()), dtype=tf.float64)
    
    # Loop condition and body for tf.while_loop
    def condition(n, X, X_, traj_nn):
        return n < n_steps

    def body(n, X, X_, traj_nn):
        X, X_ = loop_func(Xscaler, Yscaler, model, X, X_)
        if n >= 10**3:
            traj_nn = traj_nn.write(n - 10**3, X)
        return n + 1, X, X_, traj_nn

    # Initialize the loop variables
    n = tf.constant(0)
    
    # Execute tf.while_loop
    n, X, X_, traj_nn = tf.while_loop(
        condition,
        body,
        loop_vars=[n, X, X_, traj_nn]
    )

    # Stack the TensorArray into a single tensor
    traj_nn = traj_nn.stack()
    traj_nn = tf.reshape(traj_nn, (n_steps- 10**3, test_particles,K))
    traj_nn = tf.transpose(traj_nn, perm=[1,0,2])
    traj_nn = tf.reshape(traj_nn, (-1, K))

    # Convert the TensorFlow tensor to a NumPy array and save to CSV
    traj_nn_np = traj_nn.numpy()
    np.savetxt(f'{MODEL_DIR}/traj_nn.csv', traj_nn_np, delimiter=',')


neural_network_trajectories(model,test_particles,K,J, dt, n_steps, Xscaler, Yscaler)

print("success")
