import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx("float64")

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import callbacks as cb
from pathlib import Path
import pickle
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

MODEL_DIR_data = (f"/work/sc130/sc130/akneale/63/data/{n_steps_train_str}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")
if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)

# --- PREPARE DATA ---

dt = 0.001

X = np.loadtxt(f'{MODEL_DIR_data}X.csv', delimiter=',')
Y = np.loadtxt(f'{MODEL_DIR_data}Y.csv', delimiter=',')

Xscaler, Yscaler = load_scalers(n_steps_train_str, test_steps_str, str_n_particles_train, str_n_particles_test)

X_ = Xscaler.standardise(X)
Y_ = Yscaler.standardise(Y)
del X, Y

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
print("Loaded the libraries.")
print(tf.config.list_physical_devices())


# --- BUILD MODEL ---

model = tf.keras.Sequential(
    [tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(512, activation='tanh', use_bias = True),
    tfkl.Dense(512, activation='tanh', use_bias = True),
    tfkl.Dense(3, activation=None)])

print("Built the model.")

# --- TRAIN MODEL ---

LOG_FILE = "log.csv"
CHECKPOINT_FILE = "checkpoint_epoch_{epoch:02d}/weights"
TRAINED_FILE = "trained/weights"

# Training configuration

def nll(data_point, tf_distribution):
    """Negative log likelihood."""
    return -tf_distribution.log_prob(data_point)

LOSS = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")
OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
VALIDATION_SPLIT = 0.2


# Callbacks
CSV_LOGGER = cb.CSVLogger(MODEL_DIR + LOG_FILE)
BATCHES_PER_EPOCH = int(
    np.ceil(X_.shape[0] / BATCH_SIZE * (1 - VALIDATION_SPLIT)))
CHECKPOINTING = cb.ModelCheckpoint(
    MODEL_DIR + CHECKPOINT_FILE,
    save_freq=10 * BATCHES_PER_EPOCH,
    verbose=1,
    save_weights_only=True)
EARLY_STOPPING = cb.EarlyStopping(monitor="val_loss",
                                  patience=PATIENCE, min_delta=0.0)
CALLBACKS = [CHECKPOINTING, CSV_LOGGER, EARLY_STOPPING]

# model compilation and training
model.compile(loss=LOSS, optimizer=OPTIMISER)

History = model.fit(
    X_,
    Y_,
    epochs=EPOCHS,
    callbacks=CALLBACKS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=VALIDATION_SPLIT,
    verbose=2,
)

print('model trained')
model.save_weights(MODEL_DIR + TRAINED_FILE)
