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



traj_96 = np.loadtxt(f'/work/sc130/sc130/akneale/96/data/e7_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/traj_96.csv', delimiter=',')
s_autocorr_96 = np.loadtxt(f'/work/sc130/sc130/akneale/96/deterministic/models/NC32_e7_5e-5_131072_50_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/space_autocorr_96.csv', delimiter=',')
t_autocorr_96 = np.loadtxt(f'/work/sc130/sc130/akneale/96/deterministic/models/NC32_e7_5e-5_131072_50_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/temp_autocorr_96.csv', delimiter=',')

num_data_points_uni = 10**5
num_data_points_multi = 2*10**4




traj_96_uni = traj_96[np.random.randint(0,10**7,num_data_points_uni),:]

traj_multi = traj_96[np.random.randint(0,10**7,num_data_points_multi),:]
traj_96 = traj_multi.T

traj_96_flat = traj_96_uni.flatten()
reference_min = traj_96_flat.min()
reference_max = traj_96_flat.max()



kde_96_uni = stats.gaussian_kde(traj_96_flat)
kde_96_multi = stats.gaussian_kde(traj_96)




KL_arr = np.zeros(5)
HD_arr = np.zeros(5)
log_arr = np.zeros(5)
energy_arr = np.zeros(5)
t_autocorr_arr = np.zeros(5)
s_autocorr_arr = np.zeros(5)
beta = 1


i = 0
#deterministic
traj_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/deterministic/models/NC32_e7_5e-5_131072_50_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/traj_nn.csv', delimiter=',')
s_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/deterministic/models/NC32_e7_5e-5_131072_50_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/space_autocorr_nn.csv', delimiter=',')
t_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/deterministic/models/NC32_e7_5e-5_131072_50_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/temp_autocorr_nn.csv', delimiter=',')

t_autocorr_arr[i] = autocorr_comparison(t_autocorr_96, t_autocorr_nn_md)
print(t_autocorr_arr[i])
del t_autocorr_nn_md

s_autocorr_arr[i] = autocorr_comparison(s_autocorr_96, s_autocorr_nn_md)
print(s_autocorr_arr[i])
del s_autocorr_nn_md


traj_uni_nn = traj_nn_md[np.random.randint(0,10**7,num_data_points_uni),:]

traj_nn_md = traj_nn_md[np.random.randint(0,10**7,num_data_points_multi),:]

traj_nn_md = traj_nn_md.T

traj_nn_md_flat = traj_uni_nn.flatten()

kde_nn_md_uni = stats.gaussian_kde(traj_nn_md_flat)
del traj_nn_md_flat
print('univariate finished')



KL_arr[i] = relative_entropy_via_kde(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(KL_arr[i])
HD_arr[i] = hellinger_dist(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(HD_arr[i])
del kde_nn_md_uni



kde_nn_md_multi = stats.gaussian_kde(traj_nn_md)
del traj_nn_md
print('multivariate finished')

log_arr[i] = log_score(traj_96, kde_nn_md_multi)
print(log_arr[i])

energy_arr[i] = energy_score(traj_96, kde_nn_md_multi,beta)
print(energy_arr[i])
del kde_nn_md_multi


i += 1

#mixed_diag
traj_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/mixed_diag/models/NC32_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/traj_nn.csv', delimiter=',')
s_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/mixed_diag/models/NC32_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/space_autocorr_nn.csv', delimiter=',')
t_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/mixed_diag/models/NC32_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/temp_autocorr_nn.csv', delimiter=',')

t_autocorr_arr[i] = autocorr_comparison(t_autocorr_96, t_autocorr_nn_md)
print(t_autocorr_arr[i])
del t_autocorr_nn_md

s_autocorr_arr[i] = autocorr_comparison(s_autocorr_96, s_autocorr_nn_md)
print(s_autocorr_arr[i])
del s_autocorr_nn_md


traj_uni_nn = traj_nn_md[np.random.randint(0,10**7,num_data_points_uni),:]

traj_nn_md = traj_nn_md[np.random.randint(0,10**7,num_data_points_multi),:]

traj_nn_md = traj_nn_md.T

traj_nn_md_flat = traj_uni_nn.flatten()

kde_nn_md_uni = stats.gaussian_kde(traj_nn_md_flat)
del traj_nn_md_flat
print('univariate finished')



KL_arr[i] = relative_entropy_via_kde(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(KL_arr[i])
HD_arr[i] = hellinger_dist(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(HD_arr[i])
del kde_nn_md_uni



kde_nn_md_multi = stats.gaussian_kde(traj_nn_md)
del traj_nn_md
print('multivariate finished')

log_arr[i] = log_score(traj_96, kde_nn_md_multi)
print(log_arr[i])

energy_arr[i] = energy_score(traj_96, kde_nn_md_multi,beta)
print(energy_arr[i])
del kde_nn_md_multi



i += 1

#mixed_low
traj_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/mixed_low/full_model/models/NC32_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/traj_nn.csv', delimiter=',')
s_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/mixed_low/full_model/models/NC32_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/space_autocorr_nn.csv', delimiter=',')
t_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/mixed_low/full_model/models/NC32_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/temp_autocorr_nn.csv', delimiter=',')

t_autocorr_arr[i] = autocorr_comparison(t_autocorr_96, t_autocorr_nn_md)
print(t_autocorr_arr[i])
del t_autocorr_nn_md

s_autocorr_arr[i] = autocorr_comparison(s_autocorr_96, s_autocorr_nn_md)
print(s_autocorr_arr[i])
del s_autocorr_nn_md


traj_uni_nn = traj_nn_md[np.random.randint(0,10**7,num_data_points_uni),:]

traj_nn_md = traj_nn_md[np.random.randint(0,10**7,num_data_points_multi),:]

traj_nn_md = traj_nn_md.T

traj_nn_md_flat = traj_uni_nn.flatten()

kde_nn_md_uni = stats.gaussian_kde(traj_nn_md_flat)
del traj_nn_md_flat
print('univariate finished')



KL_arr[i] = relative_entropy_via_kde(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(KL_arr[i])
HD_arr[i] = hellinger_dist(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(HD_arr[i])
del kde_nn_md_uni



kde_nn_md_multi = stats.gaussian_kde(traj_nn_md)
del traj_nn_md
print('multivariate finished')

log_arr[i] = log_score(traj_96, kde_nn_md_multi)
print(log_arr[i])

energy_arr[i] = energy_score(traj_96, kde_nn_md_multi,beta)
print(energy_arr[i])
del kde_nn_md_multi



i +=1

#single_diag
traj_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/single_diag/models/NC1_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/traj_nn.csv', delimiter=',')
s_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/single_diag/models/NC1_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/space_autocorr_nn.csv', delimiter=',')
t_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/single_diag/models/NC1_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/temp_autocorr_nn.csv', delimiter=',')

t_autocorr_arr[i] = autocorr_comparison(t_autocorr_96, t_autocorr_nn_md)
print(t_autocorr_arr[i])
del t_autocorr_nn_md

s_autocorr_arr[i] = autocorr_comparison(s_autocorr_96, s_autocorr_nn_md)
print(s_autocorr_arr[i])
del s_autocorr_nn_md


traj_uni_nn = traj_nn_md[np.random.randint(0,10**7,num_data_points_uni),:]

traj_nn_md = traj_nn_md[np.random.randint(0,10**7,num_data_points_multi),:]

traj_nn_md = traj_nn_md.T

traj_nn_md_flat = traj_uni_nn.flatten()

kde_nn_md_uni = stats.gaussian_kde(traj_nn_md_flat)
del traj_nn_md_flat
print('univariate finished')

KL_arr[i] = relative_entropy_via_kde(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(KL_arr[i])
HD_arr[i] = hellinger_dist(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(HD_arr[i])
del kde_nn_md_uni

kde_nn_md_multi = stats.gaussian_kde(traj_nn_md)
del traj_nn_md
print('multivariate finished')

log_arr[i] = log_score(traj_96, kde_nn_md_multi)
print(log_arr[i])

energy_arr[i] = energy_score(traj_96, kde_nn_md_multi,beta)
print(energy_arr[i])
del kde_nn_md_multi

i +=1

#single_lower
traj_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/single_low/models/NC1_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/traj_nn.csv', delimiter=',')
s_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/single_low/models/NC1_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/space_autocorr_nn.csv', delimiter=',')
t_autocorr_nn_md = np.loadtxt(f'/work/sc130/sc130/akneale/96/single_low/models/NC1_e7_5e-6_131072_100_e7_8_32_t=e2e5_d=e2e5_1.0_10.0_10.0_20.0/temp_autocorr_nn.csv', delimiter=',')

t_autocorr_arr[i] = autocorr_comparison(t_autocorr_96, t_autocorr_nn_md)
print(t_autocorr_arr[i])
del t_autocorr_nn_md

s_autocorr_arr[i] = autocorr_comparison(s_autocorr_96, s_autocorr_nn_md)
print(s_autocorr_arr[i])
del s_autocorr_nn_md


traj_uni_nn = traj_nn_md[np.random.randint(0,10**7,num_data_points_uni),:]

traj_nn_md = traj_nn_md[np.random.randint(0,10**7,num_data_points_multi),:]

traj_nn_md = traj_nn_md.T

traj_nn_md_flat = traj_uni_nn.flatten()

kde_nn_md_uni = stats.gaussian_kde(traj_nn_md_flat)
del traj_nn_md_flat
print('univariate finished')



KL_arr[i] = relative_entropy_via_kde(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(KL_arr[i])
HD_arr[i] = hellinger_dist(kde_96_uni, kde_nn_md_uni, reference_min,reference_max)
print(HD_arr[i])
del kde_nn_md_uni



kde_nn_md_multi = stats.gaussian_kde(traj_nn_md)
del traj_nn_md
print('multivariate finished')

log_arr[i] = log_score(traj_96, kde_nn_md_multi)
print(log_arr[i])

energy_arr[i] = energy_score(traj_96, kde_nn_md_multi,beta)
print(energy_arr[i])
del kde_nn_md_multi



data = [
    ['KL', KL_arr],
    ['HD', HD_arr],
    ['Log', log_arr],
    ['Energy', energy_arr],
    ['time_autocorr', t_autocorr_arr],
    ['space_autocorr', s_autocorr_arr],

]

for row in data:
    row[1] = [str(value) for value in row[1]]

# Specify the file name
filename = 'scores_h=1_b=10_c=10.csv'

# Write the headers and data to the CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['', 'Deterministic', 'MD', 'ML', 'SD', 'SL'])
    
    # Write the data rows
    for row in data:
        writer.writerow([row[0]] + row[1])

print(f"Data has been written to {filename}")
