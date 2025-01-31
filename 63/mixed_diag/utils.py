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


N_C = 32
DT = 4



def create_distribution(params):
    """Creates a mixture of independent Gaussians with diagonal covariance matrices."""
    logits = params[..., :N_C]  # First N_C parameters are logits for the mixture components
    loc = params[..., N_C:N_C + N_C * 3]  # Next N_C * 3 parameters are the means
    scale_diag = params[..., N_C + N_C * 3:]  # Remaining N_C * 3 parameters are the scales
    loc = tf.reshape(loc, [-1, N_C, 3])
    scale_diag = tf.nn.softplus(tf.reshape(scale_diag, [-1, N_C, 3]))
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag))


def load_mdn(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,str_n_particles_train,str_n_particles_test):
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
         
        tfkl.Dense(N_C * 7, activation=None),
        tfpl.DistributionLambda(make_distribution_fn=create_distribution)])

    model.load_weights(
        f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/checkpoint_epoch_730/weights").expect_partial()
    return model


def load_scalers(n_steps_train_str, test_steps_str, str_n_particles_train, str_n_particles_test):
    """
    Loads scaler objects relating to MDN models.
    """

    with open(f"/work/sc130/sc130/akneale/63/data/{n_steps_train_str}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/Xscaler.pkl", "rb") as file:
        Xscaler = pickle.load(file)

    with open(f"/work/sc130/sc130/akneale/63/data/{n_steps_train_str}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/Yscaler.pkl", "rb") as file:
        Yscaler = pickle.load(file)
    return Xscaler, Yscaler


