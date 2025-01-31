import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

import csv
import os
import pandas as pd

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
EPOCHS = 10000
test_particles = 100
test_particles_str = 'd=e2e5_20000'



MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{test_particles_str}/")


output_dir = f'{MODEL_DIR}graphs'
os.makedirs(output_dir, exist_ok=True)

# load the CSV file
file_path = f'{MODEL_DIR}log.csv'
data = pd.read_csv(file_path)

# extract the columns
test_error = data.iloc[:, 1]
validation_error = data.iloc[:, 2]

# Determine the common clipping threshold (95th percentile of the combined data)
combined_data = np.concatenate((test_error, validation_error))
clip_threshold = np.percentile(combined_data, 100)

# Clip both loss functions
clipped_test_loss = np.clip(test_error, None, clip_threshold)
clipped_val_loss = np.clip(validation_error, None, clip_threshold)

plt.figure(1)
numsteps = range(test_error.shape[0])
plt.scatter(numsteps, clipped_test_loss, label='Test Error', color='blue')
plt.scatter(numsteps, clipped_val_loss, label='Validation Error', color='red')
plt.title('Loss')
plt.xlabel('Timestep')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss'))
plt.close()
print('success')

