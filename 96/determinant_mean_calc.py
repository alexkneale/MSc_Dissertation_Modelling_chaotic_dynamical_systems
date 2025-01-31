import numpy as np
print('delay=2')
md_arr = np.loadtxt(f'/work/sc130/sc130/akneale/96/delays/mixed_diag_delays/models/NC32_2_e7_e-5_65536_10_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/determinant.csv', delimiter=',')
md = np.mean(md_arr)
print('md = '+str(md))
ml_arr = np.loadtxt(f'/work/sc130/sc130/akneale/96/delays/mixed_low_delays/models/NC32_2_e7_e-5_131072_10_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/determinant.csv', delimiter=',')
ml = np.mean(ml_arr)
print('ml = '+str(ml))
sd_arr = np.loadtxt(f'/work/sc130/sc130/akneale/96/delays/sing_diag_delays/models/new_NC1_2_e7_e-4_16384_400_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/determinant.csv', delimiter=',')
sd = np.mean(sd_arr)
print('sd = '+str(sd))

sl_arr = np.loadtxt(f'/work/sc130/sc130/akneale/96/delays/sing_low_delays/models/NC1_2_e7_e-5_65536_10_e7_8_32_t=e2e5_d=e2e5_5.0_10.0_2.0_20.0/determinant.csv', delimiter=',')
sl = np.mean(sl_arr)
print('sl = '+str(sl))

