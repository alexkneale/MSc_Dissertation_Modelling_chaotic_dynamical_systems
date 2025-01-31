import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 
import csv
import os
import pandas as pd



# Model hyperparameters
N_C = 1
n_steps_train = int(1e7)
n_steps_train_str = 'e7'
BATCH_SIZE = int(2**13)
LEARNING_RATE = 1e-5
LEARNING_RATE_str = 'e-5'
PATIENCE = 150
test_steps = int(1e7)
test_steps_str = 'e7'
n_particles_train = 100
n_particles_test = 100
str_n_particles_train = 't=e2e5'
str_n_particles_test = 'd=e2e5'
EPOCHS = 5000

MODEL_DIR = (f"models/NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")
MODEL_DIR_data = (f"/work/sc130/sc130/akneale/63/data/{n_steps_train_str}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")

    
output_dir = f'{MODEL_DIR}graphs'
os.makedirs(output_dir, exist_ok=True)


dt = 0.001
n_particles = n_particles_test
traj_63 = np.loadtxt(f'{MODEL_DIR_data}traj_63.csv', delimiter=',')
traj_nn = np.loadtxt(f'{MODEL_DIR}traj_nn.csv', delimiter=',')
max_lag = 1000
det_arr = np.loadtxt(f'{MODEL_DIR}determinant.csv', delimiter=',')
det_arr = det_arr.flatten()


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



## 63

def lorenz_63_data_display(traj_63,traj_nn, det_arr, max_lag,dt,output_dir,n_particles):
    '''
    ### deterministic data
    
    # temporal autocorrelation
    autocorr_x_63 = autocorrelation(traj_63[:,0],max_lag,1)
    autocorr_y_63 = autocorrelation(traj_63[:,1],max_lag,1)
    autocorr_z_63 = autocorrelation(traj_63[:,2],max_lag,1)

    # spatial autocorrelation
    flattened_traj_63 = traj_63.reshape(-1, 3)

    cov_matrix_63 = np.cov(flattened_traj_63, rowvar=False)

    print("Covariance Matrix deterministic: ")
    print(cov_matrix_63)
    


    with open(os.path.join(output_dir, 'Covariance_Matrix_spatial_autocorrelation.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Covariance Matrix deterministic'])  # Write the header
        writer.writerows(cov_matrix_63.tolist())
        
        
    ### neural network data

    # temporal autocorrelation
    autocorr_x_nn = autocorrelation(traj_nn[:,0],max_lag,n_particles)
    autocorr_y_nn = autocorrelation(traj_nn[:,1],max_lag,n_particles)
    autocorr_z_nn = autocorrelation(traj_nn[:,2],max_lag,n_particles)

    # spatial autocorrelation

    flattened_traj_nn = traj_nn.reshape(-1, 3)

    cov_matrix_nn = np.cov(flattened_traj_nn, rowvar=False)

    print("Covariance Matrix neural network: ")
    print(cov_matrix_nn)
    
    with open(os.path.join(output_dir, 'Covariance_Matrix_spatial_autocorrelation.csv'), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Covariance Matrix neural network'])  # Write the header
        writer.writerows(cov_matrix_nn.tolist())

    '''    
    
    plt.figure(1)
    # truth
    histogram_63,bins_63 = np.histogram(traj_63[:,0].flatten(),bins=100, density=True)
    midx_63 = (bins_63[0:-1]+bins_63[1:])/2    
    # nn
    histogram_nn,bins_nn = np.histogram(traj_nn[:,0].flatten(),bins=100, density=True)
    midx_nn = (bins_nn[0:-1]+bins_nn[1:])/2  
    # plotting
    plt.plot(midx_63,histogram_63,'--',label='Truth')
    plt.plot(midx_nn,histogram_nn,'--',label='Experiment')
    plt.title('pdf $x$, 63')
    plt.xlabel('$x$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pdf_x'))
    plt.close()
    
    
    plt.figure(2)
    # truth
    histogram_63,bins_63 = np.histogram(traj_63[:,1].flatten(),bins=100, density=True)
    midx_63 = (bins_63[0:-1]+bins_63[1:])/2    
    # nn
    histogram_nn,bins_nn = np.histogram(traj_nn[:,1].flatten(),bins=100, density=True)
    midx_nn = (bins_nn[0:-1]+bins_nn[1:])/2  
    # plotting
    plt.plot(midx_63,histogram_63,'--',label='Truth')
    plt.plot(midx_nn,histogram_nn,'--',label='Experiment')
    plt.title('pdf $y$, 63')
    plt.xlabel('$y$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pdf_y'))
    plt.close()
    

    plt.figure(3)
    # truth
    histogram_63,bins_63 = np.histogram(traj_63[:,2].flatten(),bins=100, density=True)
    midx_63 = (bins_63[0:-1]+bins_63[1:])/2    
    # nn
    histogram_nn,bins_nn = np.histogram(traj_nn[:,2].flatten(),bins=100, density=True)
    midx_nn = (bins_nn[0:-1]+bins_nn[1:])/2  
    # plotting
    plt.plot(midx_63,histogram_63,'--',label='Truth')
    plt.plot(midx_nn,histogram_nn,'--',label='Experiment')
    plt.title('pdf $z$, 63')
    plt.xlabel('$z$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pdf_z'))
    plt.close()
    '''

    plt.figure(4)
    lags = np.arange(max_lag + 1)*dt
    plt.plot(lags, autocorr_x_63, marker='o',label='Truth')
    plt.plot(lags, autocorr_x_nn, marker='v',label='Experiment')
    plt.title('Temporal Autocorrelation Function, x')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'autocorr_x'))
    plt.close()

    plt.figure(5)
    lags = np.arange(max_lag + 1)*dt
    plt.plot(lags, autocorr_y_63, marker='o',label='Truth')
    plt.plot(lags, autocorr_y_nn, marker='v',label='Experiment')    
    plt.title('Temporal Autocorrelation Function, y')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'autocorr_y'))
    plt.close()

    plt.figure(6)
    lags = np.arange(max_lag + 1)*dt
    plt.plot(lags, autocorr_z_63, marker='o',label='Truth')
    plt.plot(lags, autocorr_z_nn, marker='v',label='Experiment')
    plt.title('Temporal Autocorrelation Function, z')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'autocorr_z'))
    plt.close()
    
    
    plt.figure(7)
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
    
    ## plotting fixed trajectories with highlighted fixed points
    # fixed point coords
    
    
    sigma=10.
    rho=28.
    beta=8/3
    x_fix = np.sqrt(beta*(rho-1))
    y_fix = x_fix
    z_fix = rho-1
    
    
    plt.figure(8)
    # fixed points
    plt.axhline(y=x_fix, color='r', linestyle='--', label='Positive FP')
    plt.axhline(y=-x_fix, color='b', linestyle='--', label='Negative FP')
    # plotting trajectory
    plt.plot(traj_nn[:,0],'--',label='trajectory')
    plt.title('x trajectory')
    plt.xlabel('timestep')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'x_traj'))
    plt.close()
    
    plt.figure(9)
    # fixed points
    plt.axhline(y=y_fix, color='r', linestyle='--', label='Positive FP')
    plt.axhline(y=-y_fix, color='b', linestyle='--', label='Negative FP')
    # plotting trajectory
    plt.plot(traj_nn[:,1],'--',label='trajectory')
    plt.title('y trajectory')
    plt.xlabel('timestep')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'y_traj'))
    plt.close()
    
    plt.figure(10)
    # fixed points
    plt.axhline(y=z_fix, color='r', linestyle='--', label='FP')
    # plotting trajectory
    plt.plot(traj_nn[:,2],'--',label='trajectory')
    plt.title('z trajectory')
    plt.xlabel('timestep')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'z_traj'))
    plt.close()
    
    plt.figure(11)

    # plotting trajectory
    plt.scatter(traj_nn[:,0],traj_nn[:,2])
    plt.title('x-z trajectory')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'x_z'))
    plt.close()
    '''


lorenz_63_data_display(traj_63,traj_nn,det_arr, max_lag,dt,output_dir,n_particles)
print('success')
