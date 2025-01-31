"""
Utility functions for analysis of MDN models.
"""

import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfkl = tf.keras.layers
tfpl = tfp.layers


def load_mdn(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,J,train_particles_str,test_particles_str, h, b, c,F):
    """
    Loads MDN model.
    """

    model = tf.keras.Sequential(
    [tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(512, activation='tanh', use_bias = True),
    tfkl.Dense(512, activation='tanh', use_bias = True),
    tfkl.Dense(N_C * (1 + K + K*(K+1)/2), activation=None),

    tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(K))])
    
    model.load_weights(
        f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/trained/weights").expect_partial()
    return model


def load_scalers(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,J,train_particles_str, test_particles_str, h, b, c,F):
    """
    Loads scaler objects relating to MDN models.
    """

    with open(f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/Xscaler.pkl", "rb") as file:
        Xscaler = pickle.load(file)

    with open(f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/Yscaler.pkl", "rb") as file:
        Yscaler = pickle.load(file)
    return Xscaler, Yscaler
