import numpy as np

md_arr = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_lower/models/MD_NC1_e7_5e-4_8192_200_e7_t=e2e5_d=e2e5/determinant.csv', delimiter=',')
md = np.mean(md_arr)
print('md = '+str(md))
ml_arr = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_diag/full_model/models/ML_NC1_e7_e-5_8192_150_e7_t=e2e5_d=e2e5/determinant.csv', delimiter=',')
ml = np.mean(ml_arr)
print('ml = '+str(ml))
sd_arr = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_diag/full_model/models/NC1_e7_e-5_8192_150_e7_t=e2e5_d=e2e5/determinant.csv', delimiter=',')
sd = np.mean(sd_arr)
print('sd = '+str(sd))
sl_arr = np.loadtxt(f'/work/sc130/sc130/akneale/63/single_lower/models/NC1_e7_5e-4_8192_200_e7_t=e2e5_d=e2e5/determinant.csv', delimiter=',')
sl = np.mean(sl_arr)
print('sl = '+str(sl))

