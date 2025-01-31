import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

import csv
import os
import pandas as pd

#parameters
N_C = 1
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**17)
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
EPOCHS = 10000
h = 5.
b = 10.
c = 2.
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


## 96
def lorenz_96_data_display(traj_96,traj_nn, det_arr, max_lag_temp,max_lag_space,dt,output_dir,n_particles,K,J,n_steps_train,train_particles,test_steps,test_particles):

    ### deterministic data
    '''
    # temporal autocorrelation
    temp_autocorr_96_x = np.zeros(max_lag_temp + 1)
    for i in range(K):
        
        temp_autocorr_96_x += (1/K)*autocorrelation(traj_96[:,i],max_lag_temp,1,10)
    
    np.savetxt(f'{MODEL_DIR}/temp_autocorr_96.csv',temp_autocorr_96_x,delimiter=',')
    
    # spatial autocorrelation
    n_steps_tot = traj_96.shape[0]
    space_autocorr_96_x = np.zeros(max_lag_space+1)
    
    for n in range(n_steps_tot):
        space_autocorr_96_x += 1/n_steps_tot*autocorrelation(traj_96[n,:K], max_lag_space,1,1)
    
    np.savetxt(f'{MODEL_DIR}/space_autocorr_96.csv',space_autocorr_96_x,delimiter=',')
    
    ### neural network data
    
    # temporal autocorrelation
    temp_autocorr_nn_x = np.zeros(max_lag_temp + 1)
    
    for i in range(K):
        temp_autocorr_nn_x += (1/K)*autocorrelation(traj_nn[:,i],max_lag_temp,test_particles,10)
    
    np.savetxt(f'{MODEL_DIR}/temp_autocorr_nn.csv',temp_autocorr_nn_x,delimiter=',')
    
    # spatial autocorrelation
    n_steps_tot = traj_nn.shape[0]
    space_autocorr_nn_x = np.zeros(max_lag_space+1)
    
    for n in range(n_steps_tot):
        space_autocorr_nn_x += 1/n_steps_tot*autocorrelation(traj_nn[n,:K], max_lag_space,1,1)
    
    np.savetxt(f'{MODEL_DIR}/space_autocorr_nn.csv',space_autocorr_nn_x,delimiter=',')
    
    plt.figure(1)
    # truth
    histogram_96,bins_96 = np.histogram(traj_96[:,:K].flatten(),bins=100, density=True)
    midx_96 = (bins_96[0:-1]+bins_96[1:])/2    
    # nn
    histogram_nn,bins_nn = np.histogram(traj_nn[:,:K].flatten(),bins=100, density=True)
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
    '''
    plt.figure(1)
    # truth
    histogram_96,bins_96 = np.histogram(traj_96[:,:K].flatten(),bins=100, density=True)
    midx_96 = (bins_96[0:-1]+bins_96[1:])/2    
    # nn
    histogram_nn,bins_nn = np.histogram(traj_nn[:,:K].flatten(),bins=100, density=True)
    midx_nn = (bins_nn[0:-1]+bins_nn[1:])/2  
    # plotting
    plt.plot(midx_96,histogram_96,'--',label='Truth')
    plt.plot(midx_nn,histogram_nn,'--',label='Experiment')
    plt.xlim(-10,15)
    plt.title('pdf $x$, 96')
    plt.xlabel('$x$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pdf_x_zoom'))
    plt.close()
    '''
    
    plt.figure(1)
    lags = np.arange(max_lag_temp + 1)*dt*10
    plt.plot(lags, temp_autocorr_96_x, marker='o',label='Truth')
    plt.plot(lags, temp_autocorr_nn_x, marker='v',label='Experiment')
    plt.title('Temporal Autocorrelation Function, x')
    plt.xlabel('Lag')
    plt.ylabel('Temporal Autocorrelation, x')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temp_autocorr_x'))
    plt.close()
    

    
    plt.figure(3)
    lags = np.arange(max_lag_space + 1)
    plt.plot(lags, space_autocorr_96_x, marker='o',label='Truth')
    plt.plot(lags, space_autocorr_nn_x, marker='v',label='Experiment')    
    plt.title('Spatial Autocorrelation Function, x')
    plt.xlabel('Lag')
    plt.ylabel('Spatial Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'space_autocorr_x'))
    plt.close()

    
    
    plt.figure(4)
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
    '''
lorenz_96_data_display(traj_96,traj_nn,det_arr, max_lag_temp,max_lag_space,dt,output_dir,n_particles,K,J,n_steps_train,train_particles,test_steps,test_particles)
print('success')