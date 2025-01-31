import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming trajectory is your array with shape (number_timesteps, 3)
# Replace this with your actual data
# Example dummy data:
# trajectory = np.random.rand(1000, 3)

trajectory = np.loadtxt('/work/sc130/sc130/akneale/63/data/e7_e7_t=e2e5_d=e2e5/traj_63.csv', delimiter=',')
trajectory = trajectory[:5*10**5,:]
# Load your trajectory data

# Extract x, y, z components
x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x, y, z, lw=0.5, color='b')

# Set plot labels
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')

# Save the plot as a PDF file
plt.savefig('lorenz_63_trajectory.pdf')
