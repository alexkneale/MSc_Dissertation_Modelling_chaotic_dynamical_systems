from scipy import stats

import numpy as np
import scipy
import warnings
import csv

def relative_entropy_via_kde(ref_kde, param_kde, reference_min,reference_max):
 
    def relative_entropy_integrand(x):
        return ref_kde(x) * (
            np.log(ref_kde(x)) - np.log(param_kde(x)))
 
    integral, abs_error_estimate = scipy.integrate.quad(
        relative_entropy_integrand, reference_min, reference_max)
    if abs_error_estimate > 1e-2 * integral:
        warnings.warn(
            "Relative entropy estimate may be inaccurate due to quadrature.")
    return integral


def hellinger_dist(ref_kde, param_kde, reference_min,reference_max):
    
    def h_d_integrand(x):
        return((np.sqrt(param_kde(x))-np.sqrt(ref_kde(x)))**2)

    integral, abs_error_estimate = scipy.integrate.quad(
        h_d_integrand, reference_min, reference_max)
    if abs_error_estimate > 1e-2 * integral:
        warnings.warn(
            "Relative entropy estimate may be inaccurate due to quadrature.")
    # Calculate the Hellinger distance
    
    return np.sqrt(0.5*integral)


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
    
def energy_score(traj_96, kernel_nn,beta):
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
    
    term1 = np.mean(np.linalg.norm(preds - traj_96, axis=1)**beta)
    term2 = np.mean([np.linalg.norm(preds[:,j] - preds[:,k])**beta for j in range(n_ensemble) for k in range(n_ensemble)]) / 2
    

    
    return term1 - term2

def autocorr_comparison(autocorr_96, autocorr_nn):
    return np.linalg.norm(autocorr_96-autocorr_nn)/np.linalg.norm(autocorr_96)



traj_63 = np.loadtxt(f'/work/sc130/sc130/akneale/63/data/e7_e7_t=e2e5_d=e2e5/traj_63.csv', delimiter=',')
'''
t_x_autocorr_63 = np.loadtxt(f'/work/sc130/sc130/akneale/63/deterministic/models/NC32_e7_5e-5_131072_50_e7_t=e2e5_d=e2e5/temp_autocorr_96_x.csv', delimiter=',')
t_y_autocorr_63 = np.loadtxt(f'/work/sc130/sc130/akneale/63/deterministic/models/NC32_e7_5e-5_131072_50_e7_t=e2e5_d=e2e5/temp_autocorr_96_y.csv', delimiter=',')
t_z_autocorr_63 = np.loadtxt(f'/work/sc130/sc130/akneale/63/deterministic/models/NC32_e7_5e-5_131072_50_e7_t=e2e5_d=e2e5/temp_autocorr_96_z.csv', delimiter=',')
'''



num_data_points_uni = 5*10**5
num_data_points_multi = 4*10**4
'''
#subsample for univariate
traj_uni = traj_63[np.random.randint(0,10**7,num_data_points_uni),:]
x_uni = traj_uni[:,0]
y_uni = traj_uni[:,1]
z_uni = traj_uni[:,2]
del traj_uni
reference_min_x = x_uni.min()
reference_max_x = x_uni.max()
reference_min_y = y_uni.min()
reference_max_y = y_uni.max()
reference_min_z = z_uni.min()
reference_max_z = z_uni.max()


kde_uni_x = stats.gaussian_kde(x_uni)
kde_uni_y = stats.gaussian_kde(y_uni)
kde_uni_z = stats.gaussian_kde(z_uni)
'''
#subsample for multivariate
traj_multi = traj_63[np.random.randint(0,10**7,num_data_points_multi),:]
traj_multi = traj_multi.T


KL_arr_x = np.zeros(5)
KL_arr_y = np.zeros(5)
KL_arr_z = np.zeros(5)

HD_arr_x = np.zeros(5)
HD_arr_y = np.zeros(5)
HD_arr_z = np.zeros(5)

log_arr = np.zeros(5)
energy_arr = np.zeros(5)
t_autocorr_arr_x = np.zeros(5)
t_autocorr_arr_y = np.zeros(5)
t_autocorr_arr_z = np.zeros(5)

beta = 1


i = 0

'''
#deterministic
print('deterministic')
traj_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/deterministic/models/NC32_e7_5e-5_131072_50_e7_t=e2e5_d=e2e5/traj_nn.csv', delimiter=',')
t_x_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/deterministic/models/NC32_e7_5e-5_131072_50_e7_t=e2e5_d=e2e5/temp_autocorr_nn_x.csv', delimiter=',')
t_y_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/deterministic/models/NC32_e7_5e-5_131072_50_e7_t=e2e5_d=e2e5/temp_autocorr_nn_y.csv', delimiter=',')
t_z_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/deterministic/models/NC32_e7_5e-5_131072_50_e7_t=e2e5_d=e2e5/temp_autocorr_nn_z.csv', delimiter=',')


t_autocorr_arr_x[i] = autocorr_comparison(t_x_autocorr_63, t_x_autocorr_nn)
t_autocorr_arr_y[i] = autocorr_comparison(t_y_autocorr_63, t_y_autocorr_nn)
t_autocorr_arr_z[i] = autocorr_comparison(t_z_autocorr_63, t_z_autocorr_nn)
print(t_autocorr_arr_x[i])
print(t_autocorr_arr_y[i])
print(t_autocorr_arr_z[i])

del t_x_autocorr_nn
del t_y_autocorr_nn
del t_z_autocorr_nn

#subsample for univariate
traj_uni_nn = traj_nn[np.random.randint(0,10**7,num_data_points_uni),:]
x_uni_nn = traj_uni_nn[:,0]
y_uni_nn = traj_uni_nn[:,1]
z_uni_nn = traj_uni_nn[:,2]
del traj_uni_nn

kde_uni_det_x = stats.gaussian_kde(x_uni_nn)
KL_arr_x[i] = relative_entropy_via_kde(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(KL_arr_x[i])
HD_arr_x[i] = hellinger_dist(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(HD_arr_x[i])
del kde_uni_det_x

kde_uni_det_y = stats.gaussian_kde(y_uni_nn)
KL_arr_y[i] = relative_entropy_via_kde(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(KL_arr_y[i])
HD_arr_y[i] = hellinger_dist(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(HD_arr_y[i])
del kde_uni_det_y

kde_uni_det_z = stats.gaussian_kde(z_uni_nn)
KL_arr_z[i] = relative_entropy_via_kde(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(KL_arr_z[i])
HD_arr_z[i] = hellinger_dist(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(HD_arr_z[i])
del kde_uni_det_z


print('univariate finished')
'''

i +=1



#mixed_diag

traj_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/mixed_diag/models/NC32_e7_e-5_8192_300_e7_t=e2e5_d=e2e5/traj_nn.csv', delimiter=',')
'''
t_x_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/mixed_diag/models/NC32_e7_e-5_8192_300_e7_t=e2e5_d=e2e5/temp_autocorr_nn_x.csv', delimiter=',')
t_y_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/mixed_diag/models/NC32_e7_e-5_8192_300_e7_t=e2e5_d=e2e5/temp_autocorr_nn_y.csv', delimiter=',')
t_z_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/mixed_diag/models/NC32_e7_e-5_8192_300_e7_t=e2e5_d=e2e5/temp_autocorr_nn_z.csv', delimiter=',')


t_autocorr_arr_x[i] = autocorr_comparison(t_x_autocorr_63, t_x_autocorr_nn)
t_autocorr_arr_y[i] = autocorr_comparison(t_y_autocorr_63, t_y_autocorr_nn)
t_autocorr_arr_z[i] = autocorr_comparison(t_z_autocorr_63, t_z_autocorr_nn)
print(t_autocorr_arr_x[i])
print(t_autocorr_arr_y[i])
print(t_autocorr_arr_z[i])

del t_x_autocorr_nn
del t_y_autocorr_nn
del t_z_autocorr_nn

#subsample for univariate
traj_uni_nn = traj_nn[np.random.randint(0,10**7,num_data_points_uni),:]
x_uni_nn = traj_uni_nn[:,0]
y_uni_nn = traj_uni_nn[:,1]
z_uni_nn = traj_uni_nn[:,2]
del traj_uni_nn

kde_uni_det_x = stats.gaussian_kde(x_uni_nn)
KL_arr_x[i] = relative_entropy_via_kde(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(KL_arr_x[i])
HD_arr_x[i] = hellinger_dist(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(HD_arr_x[i])
del kde_uni_det_x

kde_uni_det_y = stats.gaussian_kde(y_uni_nn)
KL_arr_y[i] = relative_entropy_via_kde(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(KL_arr_y[i])
HD_arr_y[i] = hellinger_dist(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(HD_arr_y[i])
del kde_uni_det_y

kde_uni_det_z = stats.gaussian_kde(z_uni_nn)
KL_arr_z[i] = relative_entropy_via_kde(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(KL_arr_z[i])
HD_arr_z[i] = hellinger_dist(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(HD_arr_z[i])
del kde_uni_det_z


print('univariate finished')
'''
traj_multi_nn = traj_nn[np.random.randint(0,10**7,num_data_points_multi),:]

kde_multi_nn = stats.gaussian_kde(traj_multi_nn.T)
del traj_multi_nn
print('multivariate finished')

log_arr[i] = log_score(traj_multi, kde_multi_nn)
print(log_arr[i])

energy_arr[i] = energy_score(traj_multi, kde_multi_nn,beta)
print(energy_arr[i])

i +=1
#mixed_lower

traj_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/mixed_lower/models/NC32_e7_5e-6_16384_300_e7_t=e2e5_d=e2e5/traj_nn.csv', delimiter=',')
'''
t_x_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/mixed_lower/models/NC32_e7_5e-6_16384_300_e7_t=e2e5_d=e2e5/temp_autocorr_nn_x.csv', delimiter=',')
t_y_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/mixed_lower/models/NC32_e7_5e-6_16384_300_e7_t=e2e5_d=e2e5/temp_autocorr_nn_y.csv', delimiter=',')
t_z_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/mixed_lower/models/NC32_e7_5e-6_16384_300_e7_t=e2e5_d=e2e5/temp_autocorr_nn_z.csv', delimiter=',')


t_autocorr_arr_x[i] = autocorr_comparison(t_x_autocorr_63, t_x_autocorr_nn)
t_autocorr_arr_y[i] = autocorr_comparison(t_y_autocorr_63, t_y_autocorr_nn)
t_autocorr_arr_z[i] = autocorr_comparison(t_z_autocorr_63, t_z_autocorr_nn)

print(t_autocorr_arr_x[i])
print(t_autocorr_arr_y[i])
print(t_autocorr_arr_z[i])

del t_x_autocorr_nn
del t_y_autocorr_nn
del t_z_autocorr_nn

#subsample for univariate
traj_uni_nn = traj_nn[np.random.randint(0,10**7,num_data_points_uni),:]
x_uni_nn = traj_uni_nn[:,0]
y_uni_nn = traj_uni_nn[:,1]
z_uni_nn = traj_uni_nn[:,2]
del traj_uni_nn

kde_uni_det_x = stats.gaussian_kde(x_uni_nn)
KL_arr_x[i] = relative_entropy_via_kde(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(KL_arr_x[i])
HD_arr_x[i] = hellinger_dist(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(HD_arr_x[i])
del kde_uni_det_x

kde_uni_det_y = stats.gaussian_kde(y_uni_nn)
KL_arr_y[i] = relative_entropy_via_kde(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(KL_arr_y[i])
HD_arr_y[i] = hellinger_dist(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(HD_arr_y[i])
del kde_uni_det_y

kde_uni_det_z = stats.gaussian_kde(z_uni_nn)
KL_arr_z[i] = relative_entropy_via_kde(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(KL_arr_z[i])
HD_arr_z[i] = hellinger_dist(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(HD_arr_z[i])
del kde_uni_det_z


print('univariate finished')
'''

traj_multi_nn = traj_nn[np.random.randint(0,10**7,num_data_points_multi),:]

kde_multi_nn = stats.gaussian_kde(traj_multi_nn.T)
del traj_multi_nn
print('multivariate finished')

log_arr[i] = log_score(traj_multi, kde_multi_nn)

energy_arr[i] = energy_score(traj_multi, kde_multi_nn,beta)




i +=1
#single_diag

traj_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_diag/full_model/models/NC1_e7_e-5_8192_150_e7_t=e2e5_d=e2e5/traj_nn.csv', delimiter=',')
'''
t_x_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_diag/full_model/models/NC1_e7_e-5_8192_150_e7_t=e2e5_d=e2e5/temp_autocorr_nn_x.csv', delimiter=',')
t_y_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_diag/full_model/models/NC1_e7_e-5_8192_150_e7_t=e2e5_d=e2e5/temp_autocorr_nn_y.csv', delimiter=',')
t_z_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_diag/full_model/models/NC1_e7_e-5_8192_150_e7_t=e2e5_d=e2e5/temp_autocorr_nn_z.csv', delimiter=',')


t_autocorr_arr_x[i] = autocorr_comparison(t_x_autocorr_63, t_x_autocorr_nn)
t_autocorr_arr_y[i] = autocorr_comparison(t_y_autocorr_63, t_y_autocorr_nn)
t_autocorr_arr_z[i] = autocorr_comparison(t_z_autocorr_63, t_z_autocorr_nn)

print(t_autocorr_arr_x[i])
print(t_autocorr_arr_y[i])
print(t_autocorr_arr_z[i])

del t_x_autocorr_nn
del t_y_autocorr_nn
del t_z_autocorr_nn

#subsample for univariate
traj_uni_nn = traj_nn[np.random.randint(0,10**7,num_data_points_uni),:]
x_uni_nn = traj_uni_nn[:,0]
y_uni_nn = traj_uni_nn[:,1]
z_uni_nn = traj_uni_nn[:,2]
del traj_uni_nn

kde_uni_det_x = stats.gaussian_kde(x_uni_nn)
KL_arr_x[i] = relative_entropy_via_kde(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(KL_arr_x[i])
HD_arr_x[i] = hellinger_dist(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(HD_arr_x[i])
del kde_uni_det_x

kde_uni_det_y = stats.gaussian_kde(y_uni_nn)
KL_arr_y[i] = relative_entropy_via_kde(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(KL_arr_y[i])
HD_arr_y[i] = hellinger_dist(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(HD_arr_y[i])
del kde_uni_det_y

kde_uni_det_z = stats.gaussian_kde(z_uni_nn)
KL_arr_z[i] = relative_entropy_via_kde(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(KL_arr_z[i])
HD_arr_z[i] = hellinger_dist(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(HD_arr_z[i])
del kde_uni_det_z


print('univariate finished')
'''

traj_multi_nn = traj_nn[np.random.randint(0,10**7,num_data_points_multi),:]

kde_multi_nn = stats.gaussian_kde(traj_multi_nn.T)
del traj_multi_nn
print('multivariate finished')

log_arr[i] = log_score(traj_multi, kde_multi_nn)

energy_arr[i] = energy_score(traj_multi, kde_multi_nn,beta)




i +=1
#single_lower

traj_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_lower/models/NC1_e7_5e-4_8192_200_e7_t=e2e5_d=e2e5/traj_nn.csv', delimiter=',')
'''
t_x_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_lower/models/NC1_e7_5e-4_8192_200_e7_t=e2e5_d=e2e5/temp_autocorr_nn_x.csv', delimiter=',')
t_y_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_lower/models/NC1_e7_5e-4_8192_200_e7_t=e2e5_d=e2e5/temp_autocorr_nn_y.csv', delimiter=',')
t_z_autocorr_nn = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_lower/models/NC1_e7_5e-4_8192_200_e7_t=e2e5_d=e2e5/temp_autocorr_nn_z.csv', delimiter=',')


t_autocorr_arr_x[i] = autocorr_comparison(t_x_autocorr_63, t_x_autocorr_nn)
t_autocorr_arr_y[i] = autocorr_comparison(t_y_autocorr_63, t_y_autocorr_nn)
t_autocorr_arr_z[i] = autocorr_comparison(t_z_autocorr_63, t_z_autocorr_nn)

print(t_autocorr_arr_x[i])
print(t_autocorr_arr_y[i])
print(t_autocorr_arr_z[i])

del t_x_autocorr_nn
del t_y_autocorr_nn
del t_z_autocorr_nn

#subsample for univariate
traj_uni_nn = traj_nn[np.random.randint(0,10**7,num_data_points_uni),:]
x_uni_nn = traj_uni_nn[:,0]
y_uni_nn = traj_uni_nn[:,1]
z_uni_nn = traj_uni_nn[:,2]
del traj_uni_nn

kde_uni_det_x = stats.gaussian_kde(x_uni_nn)
KL_arr_x[i] = relative_entropy_via_kde(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(KL_arr_x[i])
HD_arr_x[i] = hellinger_dist(kde_uni_x, kde_uni_det_x, reference_min_x,reference_max_x)
print(HD_arr_x[i])
del kde_uni_det_x

kde_uni_det_y = stats.gaussian_kde(y_uni_nn)
KL_arr_y[i] = relative_entropy_via_kde(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(KL_arr_y[i])
HD_arr_y[i] = hellinger_dist(kde_uni_y, kde_uni_det_y, reference_min_y,reference_max_y)
print(HD_arr_y[i])
del kde_uni_det_y

kde_uni_det_z = stats.gaussian_kde(z_uni_nn)
KL_arr_z[i] = relative_entropy_via_kde(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(KL_arr_z[i])
HD_arr_z[i] = hellinger_dist(kde_uni_z, kde_uni_det_z, reference_min_z,reference_max_z)
print(HD_arr_z[i])
del kde_uni_det_z


print('univariate finished')
'''
traj_multi_nn = traj_nn[np.random.randint(0,10**7,num_data_points_multi),:]

kde_multi_nn = stats.gaussian_kde(traj_multi_nn.T)
del traj_multi_nn
print('multivariate finished')

log_arr[i] = log_score(traj_multi, kde_multi_nn)

energy_arr[i] = energy_score(traj_multi, kde_multi_nn,beta)




data = [
    ['KL_x', KL_arr_x],
    ['KL_y', KL_arr_y],
    ['KL_z', KL_arr_z],
    ['HD_x', HD_arr_x],
    ['HD_y', HD_arr_y],
    ['HD_z', HD_arr_z],
    ['Log', log_arr],
    ['Energy', energy_arr],
    ['time_autocorr_x', t_autocorr_arr_x],
    ['time_autocorr_y', t_autocorr_arr_y],
    ['time_autocorr_z', t_autocorr_arr_z],

]

for row in data:
    row[1] = [str(value) for value in row[1]]

# Specify the file name
filename = 'scores_2.csv'

# Write the headers and data to the CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['', 'det', 'md', 'ml','sd','sl'])
    
    # Write the data rows
    for row in data:
        writer.writerow([row[0]] + row[1])

print(f"Data has been written to {filename}")
