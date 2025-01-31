import numpy as np
from scipy import stats
import scipy
import csv

def log_score(end_true, X_model):
    """
    calc the log score for the given true values and probabilistic model pdfs.
    
    end_true: array-like, shape (n_features). True values to evaluate.
    X_model: array-like, shape (n_model_repeats,n_features)

        
    returns:
    log_scores: array, shape (1)
        Log scores for each model.
    """

    X_model = X_model.T
    
    kernel_nn = stats.gaussian_kde(X_model)

    
    log_density = -kernel_nn.logpdf(end_true)
    return  log_density
    
def energy_score(end_true, X_model,n_model_repeats):
    """
    calc the energy score for the given true values and KDE models.
    
    end_true: array-like, shape (n_features). True values to evaluate.
    

    returns:
    energy_scores: array, shape (len(nn_models),)
        Energy scores for each model.
    """
    
    
    X = X_model[:int(n_model_repeats/2),:]
    X_prime = X_model[int(n_model_repeats/2):,:]
 
    term1 = np.mean(np.linalg.norm(X - end_true, axis=1)**beta)
    term2 = np.mean([np.linalg.norm(X[j,:] - X_prime[k,:])**beta for j in range(int(n_model_repeats/2)) for k in range(int(n_model_repeats/2))]) / 2
    

    
    return term1 - term2



# data hyperparameters
n_traj = 1000
len_traj = 1000
len_interval = 10000
n_model_repeats = 1000
x_dim = 8
beta = 1

# truth
end_true = np.loadtxt(f'/work/sc130/sc130/akneale/96/data/1000_1000_10000_5.0_10.0_2.0_20.0/end_pos.csv', delimiter=',')

log_arr = np.zeros(4)
energy_arr = np.zeros(4)


i = 0
# MD
# dimension (n_traj,n_model_repeats*x_dim)
X_model = np.loadtxt(f'/work/sc130/sc130/akneale/96/delays/mixed_diag_delays/models/NC32_2_e7_e-5_65536_10_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/end_arr.csv', delimiter=',')  # Shape: (n_traj, n_model_repeats * x_dim)

# Reshape to (n_traj, n_model_repeats, x_dim)
X_model = X_model.reshape(n_traj, n_model_repeats, x_dim)


log = np.zeros(n_traj)
energy = np.zeros(n_traj)

for traj in range(n_traj):
    end = end_true[traj]
    X = X_model[traj,:,:]
    log[traj] = log_score(end, X)
    energy[traj] = energy_score(end, X,n_model_repeats)
log_arr[i] = np.mean(log)
print(log_arr[i])
energy_arr[i] = np.mean(energy)
print(energy_arr[i])


i +=1
# ML
# dimension (n_traj,n_model_repeats*x_dim)
X_model = np.loadtxt(f'/work/sc130/sc130/akneale/96/delays/mixed_low_delays/models/NC32_2_e7_e-5_131072_10_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/end_arr.csv', delimiter=',')  # Shape: (n_traj, n_model_repeats * x_dim)

# Reshape to (n_traj, n_model_repeats, x_dim)
X_model = X_model.reshape(n_traj, n_model_repeats, x_dim)


log = np.zeros(n_traj)
energy = np.zeros(n_traj)

for traj in range(n_traj):
    end = end_true[traj]
    X = X_model[traj,:,:]
    log[traj] = log_score(end, X)
    energy[traj] = energy_score(end, X,n_model_repeats)
log_arr[i] = np.mean(log)
print(log_arr[i])
energy_arr[i] = np.mean(energy)
print(energy_arr[i])


i +=1
'''
# SD
# dimension (n_traj,n_model_repeats*x_dim)
X_model = np.loadtxt(f'/work/sc130/sc130/akneale/96/single_diag/models/NC1_e7_e-5_131072_100_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/end_arr.csv', delimiter=',')  # Shape: (n_traj, n_model_repeats * x_dim)

# Reshape to (n_traj, n_model_repeats, x_dim)
X_model = X_model.reshape(n_traj, n_model_repeats, x_dim)


log = np.zeros(n_traj)
energy = np.zeros(n_traj)

for traj in range(n_traj):
    end = end_true[traj]
    X = X_model[traj,:,:]
    log[traj] = log_score(end, X)
    energy[traj] = energy_score(end, X,n_model_repeats)
log_arr[i] = np.mean(log)
print(log_arr[i])
energy_arr[i] = np.mean(energy)
print(energy_arr[i])
'''
i +=1
# SL
# dimension (n_traj,n_model_repeats*x_dim)
X_model = np.loadtxt(f'/work/sc130/sc130/akneale/96/delays/sing_low_delays/models/NC1_2_e7_e-5_65536_10_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/end_arr.csv', delimiter=',')  # Shape: (n_traj, n_model_repeats * x_dim)

# Reshape to (n_traj, n_model_repeats, x_dim)
X_model = X_model.reshape(n_traj, n_model_repeats, x_dim)


log = np.zeros(n_traj)
energy = np.zeros(n_traj)

for traj in range(n_traj):
    end = end_true[traj]
    X = X_model[traj,:,:]
    log[traj] = log_score(end, X)
    energy[traj] = energy_score(end, X,n_model_repeats)
log_arr[i] = np.mean(log)
print(log_arr[i])
energy_arr[i] = np.mean(energy)
print(energy_arr[i])

data = [
    ['Log', log_arr],
    ['Energy', energy_arr],
]

for row in data:
    row[1] = [str(value) for value in row[1]]

# Specify the file name
filename = 'skill_scores_h=5_b=10_c=2_ML_MD.csv'

# Write the headers and data to the CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['', 'md', 'ml','sd','sl'])
    
    # Write the data rows
    for row in data:
        writer.writerow([row[0]] + row[1])

print(f"Data has been written to {filename}")
