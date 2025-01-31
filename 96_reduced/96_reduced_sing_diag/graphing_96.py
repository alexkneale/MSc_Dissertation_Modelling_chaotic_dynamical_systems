import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

import csv
import os
import pandas as pd



def autocorrelation(x, max_lag,n_particles):
    nsteps = int(len(x)/n_particles)
    mean = np.mean(x)
    variance = np.var(x)
    autocorr = np.zeros(max_lag+1)
    autocorr[0] = 1.0  # R(0) is always 1
    for lag in range(1, max_lag + 1):
        cov = 0
        for particle in range(n_particles):
            cov += (1/n_particles)*np.sum((x[nsteps*particle:nsteps*(particle+1)-lag] - mean) * (x[nsteps*particle+lag:nsteps*(particle+1)] - mean))
        autocorr[lag] = cov / (variance * (nsteps - lag))
    
    return autocorr

def autocovariance(x, max_lag,n_particles):
    nsteps = int(len(x)/n_particles)
    mean = np.mean(x)
    variance = np.var(x)
    autocorr = np.zeros(max_lag+1)

    for lag in range(0, max_lag + 1):
        
        cov = 0
        for particle in range(n_particles):
            
            cov += (1/n_particles)*np.sum((x[nsteps*particle:nsteps*(particle+1)-lag] - mean) * (x[nsteps*particle+lag:nsteps*(particle+1)] - mean))
        autocorr[lag] = cov / (nsteps - lag)
    
    return autocorr


## 96
def lorenz_96_data_display(traj_96,traj_nn, test_error, validation_error, det_arr, max_lag_temp,max_lag_space,dt,output_dir,n_particles):

    ### deterministic data
    
    # temporal autocorrelation
    
    K = traj_96.shape[1]
    
    temp_autocorr_96 = np.zeros(max_lag_temp+1)
    for x_i in range(K):
        temp_autocorr_96 += (1/K)*autocorrelation(traj_96[:,x_i],max_lag_temp,1)
    
    # spatial autocorrelation
    n_steps_tot = traj_96.shape[0]
    space_autocorr_96 = np.zeros(max_lag_space+1)
    
    for n in range(n_steps_tot):
        space_autocorr_96 += (1/n_steps_tot)*autocorrelation(traj_96[n,:], max_lag_space,1)

    ### neural network data
    
    # temporal autocorrelation
    
    temp_autocorr_nn = np.zeros(max_lag_temp+1)
    for x_i in range(K):
        temp_autocorr_nn += (1/K)*autocorrelation(traj_nn[:,x_i],max_lag_temp,1)
            
    # spatial autocorrelation
    n_steps_tot = traj_nn.shape[0]
    space_autocorr_nn = np.zeros(max_lag_space+1)
    
    for n in range(n_steps_tot):
        space_autocorr_nn += 1/n_steps_tot*autocorrelation(traj_nn[n,:], max_lag_space,1)
        
    plt.figure(1)
    # truth
    histogram_96,bins_96 = np.histogram(traj_96.flatten(),bins=100, density=True)
    midx_96 = (bins_96[0:-1]+bins_96[1:])/2    
    # nn
    histogram_nn,bins_nn = np.histogram(traj_nn.flatten(),bins=100, density=True)
    midx_nn = (bins_nn[0:-1]+bins_nn[1:])/2  
    # plotting
    plt.plot(midx_96,histogram_96,'--',label='Truth')
    plt.plot(midx_nn,histogram_nn,'--',label='Experiment')
    plt.title('pdf $x$, 96')
    plt.xlabel('$x$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pdf_x'))
    plt.close()
    
    plt.figure(2)
    lags = np.arange(max_lag_temp + 1)*dt
    plt.plot(lags, temp_autocorr_96, marker='o',label='Truth')
    plt.plot(lags, temp_autocorr_nn, marker='v',label='Experiment')
    plt.title('Temporal Autocorrelation Function')
    plt.xlabel('Lag')
    plt.ylabel('Temporal Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temp_autocorr'))
    plt.close()

    plt.figure(3)
    lags = np.arange(max_lag_space + 1)
    plt.plot(lags, space_autocorr_96, marker='o',label='Truth')
    plt.plot(lags, space_autocorr_nn, marker='v',label='Experiment')    
    plt.title('Spatial Autocorrelation Function, y')
    plt.xlabel('Lag')
    plt.ylabel('Spatial Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'space_autocorr'))
    plt.close()
   

    plt.figure(4)
    numsteps = range(test_error.shape[0])
    plt.plot(numsteps, test_error, label='Test Error', color='blue')
    plt.plot(numsteps, validation_error, label='Validation Error', color='red')
    plt.title('Loss')
    plt.xlabel('Timestep')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss'))
    plt.close()
    
    plt.figure(5)
    histogram_det,bins_det = np.histogram(det_arr,bins=100, density=True)
    midx_det = (bins_det[0:-1]+bins_det[1:])/2
    # plotting
    plt.plot(midx_det,histogram_det)
    plt.title('determinant distribution')
    plt.xlabel('determinant')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'determinant'))
    plt.close()


# Model hyperparameters
N_C = 1
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**17)
LEARNING_RATE = 2e-6
LEARNING_RATE_str = '2e-6'
PATIENCE = 100
test_steps = 1e7
test_steps_str = 'e7'
K = 8
test_particles = 100
test_particles_str = 't=e2e5'


MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{K}_{test_particles_str}/")

    
output_dir = f'{MODEL_DIR}graphs'
os.makedirs(output_dir, exist_ok=True)


dt = 0.001
n_particles = test_particles
traj_96 = np.loadtxt(f'{MODEL_DIR}traj_96.csv', delimiter=',')
traj_nn = np.loadtxt(f'{MODEL_DIR}traj_nn.csv', delimiter=',')
max_lag_temp = 1000
max_lag_space = 5
det_arr = np.loadtxt(f'{MODEL_DIR}determinant.csv', delimiter=',')
det_arr = det_arr.flatten()
# load the CSV file
file_path = f'{MODEL_DIR}log.csv'
data = pd.read_csv(file_path)

# extract the columns
test_error = data.iloc[:, 1]
validation_error = data.iloc[:, 2]

lorenz_96_data_display(traj_96,traj_nn,test_error, validation_error,det_arr, max_lag_temp,max_lag_space,dt,output_dir,n_particles)
