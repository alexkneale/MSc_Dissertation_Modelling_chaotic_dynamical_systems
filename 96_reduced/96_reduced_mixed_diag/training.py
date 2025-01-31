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
LEARNING_RATE = 5e-6
LEARNING_RATE_str = '5e-6'
PATIENCE = 100
test_steps = 1e7
test_steps_str = 'e7'
K = 8
EPOCHS = 20000
test_particles = 100
test_particles_str = 'd=e2e5_20000'



MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{test_particles_str}/")

if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

dt = 0.001

X,Y = x0_dx0_array_96_reduced(dt, n_steps_train,N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str,K,test_particles_str)
Xscaler = Scaler(X)
Yscaler = Scaler(Y)

X_ = Xscaler.standardise(X)
Y_ = Yscaler.standardise(Y)
del X, Y

with open(MODEL_DIR + r"Xscaler.pkl", "wb") as file:
    pickle.dump(Xscaler, file)

with open(MODEL_DIR + r"Yscaler.pkl", "wb") as file:
    pickle.dump(Yscaler, file)
    
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
print("Loaded the libraries.")
print(tf.config.list_physical_devices())


# --- BUILD MODEL ---

def create_distribution(params):
    """Creates a mixture of independent Gaussians with diagonal covariance matrices."""
    logits = params[..., :N_C]  # First N_C parameters are logits for the mixture components
    loc = params[..., N_C:N_C + N_C * K]  # Next N_C * K parameters are the means
    scale_diag = params[..., N_C + N_C * K:]  # Remaining N_C * K parameters are the scales
    loc = tf.reshape(loc, [-1, N_C, K])
    scale_diag = tf.nn.softplus(tf.reshape(scale_diag, [-1, N_C, K]))
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag))


model = tf.keras.Sequential(
    [tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(256, activation='tanh', use_bias = True),
    tfkl.Dense(512, activation='tanh', use_bias = True),
    tfkl.Dense(512, activation='tanh', use_bias = True),
    tfkl.Dense(N_C * (1 + 2*K), activation=None),
    tfpl.DistributionLambda(make_distribution_fn=create_distribution)])

print("Built the model.")

# --- TRAIN MODEL ---

LOG_FILE = "log.csv"
CHECKPOINT_FILE = "checkpoint_epoch_{epoch:02d}/weights"
TRAINED_FILE = "trained/weights"

# Training configuration

def nll(data_point, tf_distribution):
    """Negative log likelihood."""
    return -tf_distribution.log_prob(data_point)

LOSS = nll
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
