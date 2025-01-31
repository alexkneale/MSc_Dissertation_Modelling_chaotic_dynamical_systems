import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import callbacks as cb
from pathlib import Path
from utils import load_mdn, load_scalers
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

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


N_C = 32
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**15)
LEARNING_RATE = 1e-5
LEARNING_RATE_str = 'e-5'
PATIENCE = 200
test_steps = 1e7
test_steps_str = 'e7'
K = 8
EPOCHS = 5000
test_particles = 100
test_particles_str = 'd=e2e5'
train_particles = 100
train_particles_str = 't=e2e5'

MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{train_particles_str}_{test_particles_str}/")

dt = 0.001
n_steps = int(test_steps/test_particles)

model = load_mdn(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,test_particles_str,train_particles_str)
Xscaler, Yscaler = load_scalers(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,test_particles_str,train_particles_str)

print("Model loaded.")


@tf.function
def loop_func(Xscaler, Yscaler, model, X, X_):
    dx_ = model(X_)
    dx = Yscaler.invert_standardisation(dx_)
    X = Xscaler.invert_standardisation(X_)
    X = X + dx
    X_ = Xscaler.standardise(X)
    return X, X_

def neural_network_trajectories(model, test_particles, K, dt, n_steps, Xscaler, Yscaler):
    # Initialize a TensorFlow tensor to store trajectories
    traj_nn = tf.TensorArray(dtype=tf.float64, size=n_steps)

    # Generate initial random values and standardize them
    n_steps_generator = (test_particles+1)*10**4
    traj_96 = lorenz_96_reduced_data_generation(K, dt, n_steps_generator)
    
    particle_arr = np.zeros((test_particles,K))
    
    for particle in range(1,test_particles+1):
        particle_arr[particle-1] = traj_96[particle*10**4,:]
    
    X = tf.constant(particle_arr, dtype=tf.float64)
    traj_nn = traj_nn.write(0, X)
    X_ = tf.constant(Xscaler.standardise(X.numpy()), dtype=tf.float64)
    
    # Loop condition and body for tf.while_loop
    def condition(n, X, X_, traj_nn):
        return n < n_steps

    def body(n, X, X_, traj_nn):
        X, X_ = loop_func(Xscaler, Yscaler, model, X, X_)
        traj_nn = traj_nn.write(n, X)
        return n + 1, X, X_, traj_nn

    # Initialize the loop variables
    n = tf.constant(1)
    
    # Execute tf.while_loop
    n, X, X_, traj_nn = tf.while_loop(
        condition,
        body,
        loop_vars=[n, X, X_, traj_nn]
    )

    # Stack the TensorArray into a single tensor
    traj_nn = traj_nn.stack()
    traj_nn = tf.reshape(traj_nn, (n_steps, test_particles,K))
    traj_nn = tf.transpose(traj_nn, perm=[1,0,2])
    traj_nn = tf.reshape(traj_nn, (-1, K))

    # Convert the TensorFlow tensor to a NumPy array and save to CSV
    traj_nn_np = traj_nn.numpy()
    np.savetxt(f'{MODEL_DIR}/traj_nn.csv', traj_nn_np, delimiter=',')

neural_network_trajectories(model,test_particles,K, dt, n_steps, Xscaler, Yscaler)

print("success")
