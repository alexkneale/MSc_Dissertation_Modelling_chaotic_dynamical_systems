import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import callbacks as cb
from pathlib import Path
from utils import load_mdn, load_scalers


import numba
from numba import jit

from l96 import integrate_reduced_l96
from l96 import integrate_l96

from l63 import integrate_l63

from preprocessing import Scaler

import csv


tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

# Model hyperparameters
N_C = 32
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**17)
LEARNING_RATE = 1e-5
LEARNING_RATE_str = 'e-5'
PATIENCE = 10
test_steps = int(1e7)
test_steps_str = 'e7'
K = 8
J = 32
test_particles = 100
test_particles_str = 'd=e2e5'
train_particles = 100
train_particles_str = 't=e2e5'
EPOCHS = 5000
h = 5.
b = 10.
c = 2.
F = 20.
delay = 2

# data hyperparameters
n_traj = 1000
len_traj = 1000
len_interval = 10000
n_model_repeats = 1000

x_dim = K
MODEL_DIR = (f"models/NC{N_C}_{delay}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/")
MODEL_DIR_data = (f"/work/sc130/sc130/akneale/96/data/{n_traj}_{len_traj}_{len_interval}_{h}_{b}_{c}_{F}_{delay}/")
dt = 0.001
model = load_mdn(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str, K, J, train_particles_str, test_particles_str, h, b, c, F, delay)
Xscaler, Yscaler = load_scalers(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str, K, J, train_particles_str, test_particles_str, h, b, c, F, delay)
print("Model loaded.")

@tf.function
def loop_func(Xscaler, Yscaler, model, X_, n):

    # Sample from the mixture distribution
    dx_ = model(X_).sample()
    dx = Yscaler.invert_standardisation(dx_)
    
    X = Xscaler.invert_standardisation(X_)
    X = X + dx
    X_ = Xscaler.standardise(X)
    
    return X, X_

@tf.function
def loop_func(Xscaler, Yscaler, model, X_, delay, K, n_traj, n):
    dx_ = model(X_).sample()
    dx = Yscaler.invert_standardisation(dx_)
    X = Xscaler.invert_standardisation(X_)
        
    X_shifted = tf.concat([X[:, K:], tf.zeros((n_traj, K), dtype=tf.float64)], axis=1)

    second_last_k_elements = X[:, -2*K:-K]
    
    last_k_elements = second_last_k_elements + dx

    X = tf.concat([X_shifted[:, :-K], last_k_elements], axis=1)
    
    X_ = Xscaler.standardise(X)

    return X, X_


def neural_network_trajectories(model, n_model_repeats, dt, K, Xscaler, Yscaler, x_dim, delay):

    # Initialize a TensorFlow tensor to store trajectories with an explicit shape
    end_arr = tf.TensorArray(dtype=tf.float64, size=n_model_repeats, element_shape=[n_traj, x_dim])
        
    particle_arr = np.loadtxt(f'{MODEL_DIR_data}init_pos.csv', delimiter=',')

    X = tf.constant(particle_arr, dtype=tf.float64)
    X_ = tf.constant(Xscaler.standardise(X.numpy()), dtype=tf.float64)
    
    # Loop condition and body for tf.while_loop
    '''
    def condition(n, n_steps, X, X_, end_arr):
        return n < n_model_repeats
    '''
    def condition(n, X, X_,n_steps, end_arr, delay, K):
        return n < n_model_repeats

    def body(n, X, X_,n_steps, end_arr, delay, K):
        X, X_ = loop_func(Xscaler, Yscaler, model, X_, delay, K, n_traj, n)
        
        # Write to the TensorArray
        if n_steps == len_traj:
            end_arr = end_arr.write(n, X[:, -K:])
            X = tf.constant(particle_arr, dtype=tf.float64)
            X_ = tf.constant(Xscaler.standardise(X.numpy()), dtype=tf.float64)
            n += 1
            n_steps = tf.constant(1)

        return n, X, X_,n_steps + 1, end_arr,delay,K

    # Initialize the loop variables
    n = tf.constant(0)
    n_steps = tf.constant(1)
    
    # Execute tf.while_loop
    n, X, X_,n_steps, end_arr,_,_ = tf.while_loop(
        condition,
        body,
        loop_vars=[n, X, X_, n_steps, end_arr,delay,K]
    )

    # Stack the TensorArray
    end_arr_tensor = end_arr.stack()  # Shape: (n_model_repeats, n_traj, x_dim)

    # Convert to NumPy array
    end_arr_np = end_arr_tensor.numpy()  # Shape: (n_model_repeats, n_traj, x_dim)

    # Reshape to (n_traj, n_model_repeats * x_dim)
    reshaped_np = end_arr_np.transpose(1, 0, 2).reshape(n_traj, -1)  # Shape: (n_traj, n_model_repeats * x_dim)

    # Save to CSV
    np.savetxt(f'{MODEL_DIR}/end_arr.csv', reshaped_np, delimiter=',')


neural_network_trajectories(model, n_model_repeats, dt, K, Xscaler, Yscaler, x_dim, delay)

print("Success")