import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14


import csv
import os
import pandas as pd



#parameters
N_C = 32
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**17)
LEARNING_RATE = 1e-5
LEARNING_RATE_str = 'e-5'
PATIENCE = 10
test_steps = int(1e7)
test_steps_str = 'e7'
K = 8
J = 32
test_particles = 100
test_particles_str = 'd=e2e5'
train_particles = 100
train_particles_str = 't=e2e5'
EPOCHS = 5000
h = 1.
b = 10.
c = 4.
F = 20.

MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/")
MODEL_DIR_data = (f"/work/sc130/sc130/akneale/96/data/{n_steps_train_str}_{test_steps_str}_{K}_{J}_{train_particles_str}_{test_particles_str}_{h}_{b}_{c}_{F}/")

output_dir = f'{MODEL_DIR}graphs'
os.makedirs(output_dir, exist_ok=True)


dt = 0.001
n_particles = test_particles
traj_96 = np.loadtxt(f'{MODEL_DIR_data}traj_96.csv', delimiter=',')
traj_nn = np.loadtxt(f'{MODEL_DIR}traj_nn.csv', delimiter=',')
max_lag_temp = 400
max_lag_space = 5
det_arr = np.loadtxt(f'{MODEL_DIR}determinant.csv', delimiter=',')
det_arr = det_arr.flatten()


def autocorrelation(x, max_lag,n_particles,jump):
    nsteps = int(len(x)/n_particles)
    mean = np.mean(x)
    variance = np.var(x)
    autocorr = np.zeros(max_lag+1)
    autocorr[0] = 1.0  # R(0) is always 1
    i = 1
    for lag in range(1*jump, (max_lag + 1)*jump, jump):
        cov = 0
        for particle in range(n_particles):
            cov += (1/n_particles)*np.sum((x[nsteps*particle:nsteps*(particle+1)-lag] - mean) * (x[nsteps*particle+lag:nsteps*(particle+1)] - mean))
        autocorr[i] = cov / (variance * (nsteps - lag))
        i +=1
    
    return autocorr

def autocovariance(x, max_lag,n_particles,jump):
    nsteps = int(len(x)/n_particles)
    mean = np.mean(x)
    variance = np.var(x)
    autocorr = np.zeros(max_lag+1)

    for lag in range(0, (max_lag + 1)*jump, jump):
        
        cov = 0
        for particle in range(n_particles):
            
            cov += (1/n_particles)*np.sum((x[nsteps*particle:nsteps*(particle+1)-lag] - mean) * (x[nsteps*particle+lag:nsteps*(particle+1)] - mean))
        autocorr[lag] = cov / (nsteps - lag)
    
    return autocorr









### deterministic data

# temporal autocorrelation

temp_autocorr_96 = np.zeros(max_lag_temp + 1)
for i in range(K):
    
    temp_autocorr_96 += (1/K)*autocorrelation(traj_96[:,i],max_lag_temp,1,10)

np.savetxt(f'{MODEL_DIR}/temp_autocorr_96.csv',temp_autocorr_96,delimiter=',')

# spatial autocorrelation
n_steps_tot = traj_96.shape[0]
space_autocorr_96 = np.zeros(max_lag_space+1)

for n in range(n_steps_tot):
    space_autocorr_96 += 1/n_steps_tot*autocorrelation(traj_96[n,:K], max_lag_space,1,1)
np.savetxt(f'{MODEL_DIR}/space_autocorr_96.csv',space_autocorr_96,delimiter=',')

### neural network data

# temporal autocorrelation
temp_autocorr_nn = np.zeros(max_lag_temp + 1)


for i in range(K):
    temp_autocorr_nn += (1/K)*autocorrelation(traj_nn[:,i],max_lag_temp,test_particles,10)
np.savetxt(f'{MODEL_DIR}/temp_autocorr_nn.csv',temp_autocorr_nn,delimiter=',')


# spatial autocorrelation
n_steps_tot = traj_nn.shape[0]
space_autocorr_nn = np.zeros(max_lag_space+1)

for n in range(n_steps_tot):
    space_autocorr_nn += 1/n_steps_tot*autocorrelation(traj_nn[n,:K], max_lag_space,1,1)
np.savetxt(f'{MODEL_DIR}/space_autocorr_nn.csv',space_autocorr_nn,delimiter=',')


'''
temp_autocorr_96 = np.loadtxt(f'{MODEL_DIR}temp_autocorr_96.csv', delimiter=',')
temp_autocorr_nn = np.loadtxt(f'{MODEL_DIR}temp_autocorr_nn.csv', delimiter=',')

space_autocorr_96 = np.loadtxt(f'{MODEL_DIR}space_autocorr_96.csv', delimiter=',')
space_autocorr_nn = np.loadtxt(f'{MODEL_DIR}space_autocorr_nn.csv', delimiter=',')
'''

plt.figure(1)
# truth
histogram_96,bins_96 = np.histogram(traj_96.flatten(),bins=400, density=True)
midx_96 = (bins_96[0:-1]+bins_96[1:])/2    
# nn
histogram_nn,bins_nn = np.histogram(traj_nn.flatten(),bins=400, density=True)
midx_nn = (bins_nn[0:-1]+bins_nn[1:])/2  
# plotting
plt.plot(midx_96,histogram_96,'--',label='Truth')
plt.plot(midx_nn,histogram_nn,'--',label='ML Model')
plt.title('Distribution of $x_i$, ML Model')
plt.xlabel('$x_i$')
plt.ylabel('$p(x_i)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pdf_x.pdf'))
plt.close()

plt.figure(2)
lags = np.arange(max_lag_temp + 1)*dt*10
plt.plot(lags, temp_autocorr_96, marker='o',label='Truth')
plt.plot(lags, temp_autocorr_nn, marker='v',label='ML Model')
plt.title('Temporal Autocorrelation of $x_i$,\ML Model')
plt.xlabel('Lag (MTU)')
plt.ylabel('Temporal Autocorrelation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'temp_autocorr_x.pdf'))
plt.close()



plt.figure(3)
lags = np.arange(max_lag_space + 1)
plt.plot(lags, space_autocorr_96, marker='o',label='Truth')
plt.plot(lags, space_autocorr_nn, marker='v',label='ML Model')    
plt.title('Spatial Autocorrelation of $x_i$,\nML Model')
plt.xlabel('Lag ($\delta i$)')
plt.ylabel('Spatial Autocorrelation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'space_autocorr_x.pdf'))
plt.close()






# load the CSV file
file_path = f'{MODEL_DIR}log.csv'
data = pd.read_csv(file_path)

# extract the columns
test_error = data.iloc[:, 1]
validation_error = data.iloc[:, 2]

plt.figure(4)
numsteps = range(test_error.shape[0])
plt.plot(numsteps, test_error, label='Test Error', color='blue')
plt.plot(numsteps, validation_error, label='Validation Error', color='red')
plt.title('Loss, ML Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss.pdf'))
plt.close()

plt.figure(5)
histogram_det,bins_det = np.histogram(det_arr,bins=100, density=True)
midx_det = (bins_det[0:-1]+bins_det[1:])/2
# plotting
plt.plot(midx_det,histogram_det)
plt.title('Distribution of Determinant of Covariance Matrix,\nML Model')
plt.xlabel('Determinant')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'determinant.pdf'))
plt.close()

det_arr = np.loadtxt(f'{MODEL_DIR}determinant.csv', delimiter=',')
det_arr = det_arr.flatten()
'''
plt.figure(6)
histogram_det,bins_det = np.histogram(det_arr,bins=500, density=True)
midx_det = (bins_det[0:-1]+bins_det[1:])/2
# plotting
plt.plot(midx_det,histogram_det)
plt.title('Distribution of Determinant of Covariance Matrix,\nML Model')
plt.xlabel('Determinant')
plt.xlim(0,2*10**(-13))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'determinant_zoom.pdf'))
plt.close()
'''


print('success')