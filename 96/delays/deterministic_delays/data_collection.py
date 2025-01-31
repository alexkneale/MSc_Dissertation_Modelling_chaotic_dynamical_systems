import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils import load_mdn, load_scalers

tf.keras.backend.set_floatx("float64")

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

# Parameters
N_C = 32
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**13)
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
EPOCHS = 5000
h = 5.
b = 10.
c = 2.
F = 20.
delay = 3

MODEL_DIR = (f"models/NC{N_C}_{delay}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/")
MODEL_DIR_data_daughter = (f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/{delay}/")

dt = 0.001
n_steps = int(test_steps / test_particles) + 10**3

model = load_mdn(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str, K, J, train_particles_str, test_particles_str, h, b, c, F, delay)
Xscaler, Yscaler = load_scalers(N_C, n_steps_train_str, LEARNING_RATE_str, BATCH_SIZE, PATIENCE, test_steps_str, K, J, train_particles_str, test_particles_str, h, b, c, F, delay)
print("Model loaded.")


@tf.function
def loop_func(Xscaler, Yscaler, model, X_, delay, K, test_particles):
    dx_ = model(X_)
    dx = Yscaler.invert_standardisation(dx_)
    X = Xscaler.invert_standardisation(X_)
        
    X_shifted = tf.concat([X[:, K:], tf.zeros((test_particles, K), dtype=tf.float64)], axis=1)

    second_last_k_elements = X[:, -2*K:-K]
    
    last_k_elements = second_last_k_elements + dx

    X = tf.concat([X_shifted[:, :-K], last_k_elements], axis=1)
    
    X_ = Xscaler.standardise(X)

    return X, X_

def neural_network_trajectories(model, test_particles, K, J, dt, n_steps, Xscaler, Yscaler, delay):
    traj_nn = tf.TensorArray(dtype=tf.float64, size=n_steps - 10**3)
    particle_arr = np.loadtxt(f'{MODEL_DIR_data_daughter}init_pos.csv', delimiter=',')
    X = tf.constant(particle_arr, dtype=tf.float64)
    X_ = Xscaler.standardise(X)
    
    def condition(n, X, X_, traj_nn, delay, K, test_particles):
        return n < n_steps

    def body(n, X, X_, traj_nn, delay, K, test_particles):
        X, X_ = loop_func(Xscaler, Yscaler, model, X_, delay, K, test_particles)
        if n >= 10**3:
            traj_nn = traj_nn.write(n - 10**3, X[:, -K:])
        return n + 1, X, X_, traj_nn, delay, K, test_particles

    n = tf.constant(0)
    n, X, X_, traj_nn, _, _, _ = tf.while_loop(
        condition,
        body,
        loop_vars=[n, X, X_, traj_nn, delay, K, test_particles]
    )

    traj_nn = traj_nn.stack()
    traj_nn = tf.reshape(traj_nn, (n_steps - 10**3, test_particles, K))
    traj_nn = tf.transpose(traj_nn, perm=[1, 0, 2])
    traj_nn = tf.reshape(traj_nn, (-1, K))
    
    traj_nn_np = traj_nn.numpy()
    
    
    np.savetxt(f'{MODEL_DIR}/traj_nn.csv', traj_nn_np, delimiter=',')

neural_network_trajectories(model, test_particles, K, J, dt, n_steps, Xscaler, Yscaler, delay)

print("success")