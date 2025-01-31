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
from data_generator_nn_63 import x0_dx0_array_63

from data_display_63 import lorenz_63_data_generation
import csv


tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


# Model hyperparameters
N_C = 32
n_steps_train = int(1e2)
n_steps_train_str = 'e7'
BATCH_SIZE = 8192
LEARNING_RATE = 1e-5
LEARNING_RATE_str = 'e-5'
PATIENCE = 50
test_steps = int(1e2)
test_steps_str = 'e7'
n_particles_train = 10
n_particles_test = 10
str_n_particles_train = 't=e2e5'
str_n_particles_test = 'd=e2e5'
EPOCHS = 3

# data hyperparameters
n_traj = 100
len_traj = 100
len_interval = 100
n_model_repeats = 10

x_dim = 3

MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")
MODEL_DIR_data = (f"data/{n_traj}_{len_traj}_{len_interval}/")

dt = 0.001
model = load_mdn(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,str_n_particles_train,str_n_particles_test)

Xscaler, Yscaler = load_scalers(n_steps_train_str, test_steps_str, str_n_particles_train, str_n_particles_test)

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


def neural_network_trajectories(model, n_model_repeats, dt, Xscaler, Yscaler, x_dim):
    # Initialize a TensorFlow tensor to store trajectories with an explicit shape
    end_arr = tf.TensorArray(dtype=tf.float64, size=n_model_repeats, element_shape=[len_traj, x_dim])
        
    particle_arr = np.loadtxt(f'{MODEL_DIR_data}init_pos.csv', delimiter=',')

    X = tf.constant(particle_arr, dtype=tf.float64)
    X_ = tf.constant(Xscaler.standardise(X.numpy()), dtype=tf.float64)
    
    # Loop condition and body for tf.while_loop
    def condition(n, n_steps, X, X_, end_arr):
        return n < n_model_repeats

    def body(n, n_steps, X, X_, end_arr):
        X, X_ = loop_func(Xscaler, Yscaler, model, X_, n)
        
        # Write to the TensorArray
        if n_steps == len_traj:
            end_arr = end_arr.write(n, X)
            X = tf.constant(particle_arr, dtype=tf.float64)
            X_ = tf.constant(Xscaler.standardise(X.numpy()), dtype=tf.float64)
            n += 1
            n_steps = tf.constant(1)

        return n,n_steps + 1, X, X_, end_arr

    # Initialize the loop variables
    n = tf.constant(0)
    n_steps = tf.constant(1)
    
    # Execute tf.while_loop
    n, n_steps, X, X_, end_arr = tf.while_loop(
        condition,
        body,
        loop_vars=[n, n_steps, X, X_, end_arr]
    )

    # Stack the TensorArray and reshape
    end_arr_tensor = end_arr.stack()  # Shape will be (100, 100, 3)
    
    # Convert to NumPy and save
    end_arr_np = end_arr_tensor.numpy()
    np.savetxt(f'{MODEL_DIR}/end_arr.csv', end_arr_np.reshape((n_model_repeats, -1)), delimiter=',')



neural_network_trajectories(model, n_model_repeats, dt, Xscaler, Yscaler, x_dim)

print("Success")