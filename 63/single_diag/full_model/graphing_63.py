import numpy as np

import matplotlib.pyplot as plt 

import csv
import os
import pandas as pd


import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 14})




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

MODEL_DIR = (f"models/ML_NC{N_C}_{n_steps_train_str}_{LEARNING_RATE_str}_{BATCH_SIZE}_{PATIENCE}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")
MODEL_DIR_data = (f"/work/sc130/sc130/akneale/63/data/{n_steps_train_str}_{test_steps_str}_{str_n_particles_train}_{str_n_particles_test}/")

    
output_dir = f'{MODEL_DIR}graphs'
os.makedirs(output_dir, exist_ok=True)
'''

dt = 0.001
n_particles = n_particles_test
traj_63 = np.loadtxt(f'{MODEL_DIR_data}traj_63.csv', delimiter=',')
traj_nn = np.loadtxt(f'{MODEL_DIR}traj_nn.csv', delimiter=',')
max_lag = 400


# load the CSV file
file_path = f'{MODEL_DIR}log.csv'
data = pd.read_csv(file_path)

# extract the columns
test_error = data.iloc[:, 1]
validation_error = data.iloc[:, 2]

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



## 63

### deterministic data



# temporal autocorrelation
autocorr_x_63 = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_96_x.csv')
autocorr_x_63.squeeze()
autocorr_y_63 = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_96_y.csv')
autocorr_y_63.squeeze()
autocorr_z_63 = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_96_z.csv')
autocorr_z_63.squeeze()

autocorr_x_63 = autocorrelation(traj_63[:,0],max_lag,1,10)
autocorr_y_63 = autocorrelation(traj_63[:,1],max_lag,1,10)
autocorr_z_63 = autocorrelation(traj_63[:,2],max_lag,1,10)

np.savetxt(f'{MODEL_DIR}/temp_autocorr_96_x.csv',autocorr_x_63,delimiter=',')
np.savetxt(f'{MODEL_DIR}/temp_autocorr_96_y.csv',autocorr_y_63,delimiter=',')
np.savetxt(f'{MODEL_DIR}/temp_autocorr_96_z.csv',autocorr_z_63,delimiter=',')

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
autocorr_x_nn = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_nn_y.csv')
autocorr_x_nn.squeeze()
autocorr_y_nn = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_nn_y.csv')
autocorr_y_nn.squeeze()
autocorr_z_nn = pd.read_csv(f'{MODEL_DIR}/temp_autocorr_nn_z.csv')
autocorr_z_nn.squeeze()


# temporal autocorrelation
autocorr_x_nn = autocorrelation(traj_nn[:,0],max_lag,1,10)
autocorr_y_nn = autocorrelation(traj_nn[:,1],max_lag,1,10)
autocorr_z_nn = autocorrelation(traj_nn[:,2],max_lag,1,10)

np.savetxt(f'{MODEL_DIR}/temp_autocorr_nn_x.csv',autocorr_x_nn,delimiter=',')
np.savetxt(f'{MODEL_DIR}/temp_autocorr_nn_y.csv',autocorr_y_nn,delimiter=',')
np.savetxt(f'{MODEL_DIR}/temp_autocorr_nn_z.csv',autocorr_z_nn,delimiter=',')


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
plt.plot(midx_nn,histogram_nn,'--',label='SD Model')
plt.title('Distribution of $x$, SD Model')
plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pdf_x.pdf'))
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
plt.plot(midx_nn,histogram_nn,'--',label='SD Model')
plt.title('Distribution of $y$, SD Model')
plt.xlabel('$y$')
plt.ylabel('$p(y)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pdf_y.pdf'))
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
plt.plot(midx_nn,histogram_nn,'--',label='SD Model')
plt.title('Distribution of $y$, SD Model')
plt.xlabel('$z$')
plt.ylabel('$p(z)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pdf_z.pdf'))
plt.close()


plt.figure(4)
lags = np.arange(max_lag)*dt*10
plt.plot(lags, autocorr_x_63, marker='o',label='Truth')
plt.plot(lags, autocorr_x_nn, marker='v',label='SD Model')
plt.title('Temporal Autocorrelation of $x$,\nSD Model')
plt.xlabel('Lag (MTU)')
plt.ylabel('Temporal Autocorrelation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'autocorr_x.pdf'))
plt.close()

plt.figure(5)
lags = np.arange(max_lag)*dt*10
plt.plot(lags, autocorr_y_63, marker='o',label='Truth')
plt.plot(lags, autocorr_y_nn, marker='v',label='SD Model')    
plt.title('Temporal Autocorrelation of $y$,\nSD Model')
plt.xlabel('Lag (MTU)')
plt.ylabel('Temporal Autocorrelation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'autocorr_y.pdf'))
plt.close()

plt.figure(6)
lags = np.arange(max_lag)*dt*10
plt.plot(lags, autocorr_z_63, marker='o',label='Truth')
plt.plot(lags, autocorr_z_nn, marker='v',label='SD Model')
plt.title('Temporal Autocorrelation of $z$,\nSD Model')
plt.xlabel('Lag (MTU)')
plt.ylabel('Temporal Autocorrelation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'autocorr_z.pdf'))
plt.close()


plt.figure(7)
histogram_det,bins_det = np.histogram(det_arr,bins=500, density=True)
midx_det = (bins_det[0:-1]+bins_det[1:])/2
# plotting
plt.plot(midx_det,histogram_det)
plt.title('Distribution of Determinant of Covariance Matrix,\nSD Model')
plt.xlabel('Determinant')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'determinant.pdf'))
plt.close()
'''

det_arr = np.loadtxt(f'{MODEL_DIR}determinant.csv', delimiter=',')
det_arr = det_arr.flatten()

plt.figure(1)
histogram_det,bins_det = np.histogram(det_arr,bins=500, density=True)
midx_det = (bins_det[0:-1]+bins_det[1:])/2
# plotting
plt.plot(midx_det,histogram_det)
plt.title('Distribution of Determinant of Covariance Matrix,\nML Model')
plt.xlabel('Determinant')
plt.xlim(0,2*10**(-11))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'determinant_zoom.pdf'))
plt.close()

        


print('success')