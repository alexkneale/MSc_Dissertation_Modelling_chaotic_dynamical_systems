from scipy import stats

import numpy as np
import scipy
import warnings
import csv


def log_score(traj_96, kernel_nn):
    """
    calc the log score for the given true values and probabilistic model pdfs.
    
    traj_96: array-like, shape (n_samples, n_features = 8). True values to evaluate.
    nn_models: list of proabbilistic model pdfs, KDE objects.

        
    returns:
    log_scores: array, shape (len(nn_models),)
        Log scores for each model.
    """
    
    log_density = -kernel_nn.logpdf(traj_96)
    return  np.mean(log_density)
    
def energy_score(traj_96, traj_nn,beta):
    """
    calc the energy score for the given true values and KDE models.
    
    traj_96: array-like, shape (n_samples, n_features = 8). True values to evaluate.
    nn_models: list of proabbilistic model pdfs, KDE objects.

    returns:
    energy_scores: array, shape (len(nn_models),)
        Energy scores for each model.
    """
    
    n_ensemble = traj_96.shape[1]

    preds = kernel_nn.resample(n_ensemble)
    
    traj_nn = np.mean(np.linalg.norm(traj_nn - traj_96, axis=1)**beta)
    term2 = np.mean([np.linalg.norm(traj_nn[:,j] - traj_nn[:,k])**beta for j in range(n_ensemble) for k in range(n_ensemble)]) / 2
    

    
    return term1 - term2

# number of trajectories 
num_traj = 100
len_traj = 10**4

# true trajectory of length len_traj
traj_96 = np.loadtxt(f'/work/sc130/sc130/akneale/96/data/e7_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/traj_96.csv', delimiter=',')
# surrogate model trajectories, consisting of num_traj trajectories all of shape (len_traj,8).
# starting point of all trajectories is same as that in traj_96
traj_nn = np.loadtxt(f'/work/sc130/sc130/akneale/96/data/e7_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/traj_96.csv', delimiter=',')

# array of dimension (num_traj*len_traj,8), which is used to compute KDE
traj_flat = traj_nn.reshape(-1, 8)
# transpose
traj_flat = traj_flat.T
kernel_nn = stats.gaussian_kde(traj_flat)
log = log_score(traj_96, kernel_nn)
print(log)
del kernel_nn

energy = 0
# loop over all num_traj trajectories, calculating energy score between traj_96 and traj_nn[trajectory] each time 
for trajectory in range(num_traj):
    energy += (1/num_traj)*energy_score(traj_96, traj_nn[trajectory],beta)

print(energy)