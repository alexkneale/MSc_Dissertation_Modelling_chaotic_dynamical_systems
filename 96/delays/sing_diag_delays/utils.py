"""
Utility functions for analysis of MDN models.
"""

import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

N_C = 1
K = 8
J = 32

def create_distribution(params):
    """Creates a mixture of independent Gaussians with diagonal covariance matrices."""
    logits = params[..., :N_C]  # First N_C parameters are logits for the mixture components
    loc = params[..., N_C:N_C + N_C * K]  # Next N_C * K*(J+1) parameters are the means
    scale_diag = params[..., N_C + N_C * K:]  # Remaining parameters are the scales
    loc = tf.reshape(loc, [-1, N_C, K])
    scale_diag = tf.nn.softplus(tf.reshape(scale_diag, [-1, N_C, K]))
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag))

def load_mdn(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,J,train_particles_str,test_particles_str, h, b, c,F, delay):
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
    tfkl.Dense(N_C * (1 + 2*K), activation=None),

    tfpl.DistributionLambda(make_distribution_fn=create_distribution)])

    model.load_weights(
        f"models/NC{N_C}_{delay}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/checkpoint_epoch_2130/weights").expect_partial()
    return model


def load_scalers(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,J,train_particles_str, test_particles_str, h, b, c,F,delay):
    """
    Loads scaler objects relating to MDN models.
    """

    with open(f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/{delay}/Xscaler.pkl", "rb") as file:
        Xscaler = pickle.load(file)

    with open(f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/{delay}/Yscaler.pkl", "rb") as file:
        Yscaler = pickle.load(file)
    return Xscaler, Yscaler