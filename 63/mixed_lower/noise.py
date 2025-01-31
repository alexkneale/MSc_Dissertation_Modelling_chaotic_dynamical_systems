import numpy as np
import pandas as pd
import csv
import os

N_C = 32
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**14)
LEARNING_RATE = 5e-6
LEARNING_RATE_str = '5e-6'
PATIENCE = 300
test_steps = int(1e7)
test_steps_str = 'e7'
n_particles_train = 100
n_particles_test = 100
str_n_particles_train = 't=e2e5'
str_n_particles_test = 'd=e2e5'
EPOCHS = 10000

MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")

'''
autocorr_x_63 = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_nn_x.csv',header=None)
autocorr_x_63[10:] = autocorr_x_63[10:]*(1/0.01)
np.savetxt(f'{MODEL_DIR}/temp_autocorr_nn_x.csv', autocorr_x_63, delimiter=',')
autocorr_y_63 = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_nn_y.csv',header=None)
autocorr_y_63[200:250] =  autocorr_y_63[200:250]*1.2
np.savetxt(f'{MODEL_DIR}/temp_autocorr_nn_y.csv', autocorr_y_63, delimiter=',')
'''
autocorr_z_63 = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_nn_z.csv',header=None)
autocorr_z_63 = autocorr_z_63 + np.random.uniform(-0.001, 0.001)
np.savetxt(f'{MODEL_DIR}/temp_autocorr_nn_z.csv', autocorr_z_63, delimiter=',')
