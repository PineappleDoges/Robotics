import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
L1, L2 = 1.0, 1.0
m1, m2 = 1.0, 1.0
g = 9.81

# Create grid of joint angles
theta1 = np.linspace(-np.pi, np.pi, 50)
theta2 = np.linspace(-np.pi, np.pi, 50)
Theta1, Theta2 = np.meshgrid(theta1, theta2)

# Compute torques
tau_force_1 = np.zeros_like(Theta1)
tau_force_2 = np.sin(Theta2/2)

tau_gravity_1 = -g * (1.5 * np.cos(Theta1) + 0.5 * np.cos(Theta1 + Theta2))
tau_gravity_2 = -g * (0.5 * np.cos(Theta1 + Theta2))

tau_total_1 = tau_force_1 + tau_gravity_1
tau_total_2 = tau_force_2 + tau_gravity_2

# Plotting
fig = plt.figure(figsize=(15, 12))  # 增加高度以提供更多空间

# Plot tau1
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surf1 = ax1.plot_surface(Theta1, Theta2, tau_total_1, cmap='viridis', alpha=0.8)
ax1.set_xlabel(r'$\theta_1$ (rad)')
ax1.set_ylabel(r'$\theta_2$ (rad)')
ax1.set_zlabel(r'$\tau_1$ (Nm)')
ax1.set_title('Joint 1 Torque')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Plot tau2
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
surf2 = ax2.plot_surface(Theta1, Theta2, tau_total_2, cmap='viridis', alpha=0.8)
ax2.set_xlabel(r'$\theta_1$ (rad)')
ax2.set_ylabel(r'$\theta_2$ (rad)')
ax2.set_zlabel(r'$\tau_2$ (Nm)')
ax2.set_title('Joint 2 Torque')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# Contour plot for tau1
ax3 = fig.add_subplot(2, 2, 3)
contour1 = ax3.contourf(Theta1, Theta2, tau_total_1, 20, cmap='viridis')
ax3.set_xlabel(r'$\theta_1$ (rad)')
ax3.set_ylabel(r'$\theta_2$ (rad)')
ax3.set_title('Joint 1 Torque (Contour)')
fig.colorbar(contour1, ax=ax3, shrink=0.5, aspect=5)

# Contour plot for tau2
ax4 = fig.add_subplot(2, 2, 4)
contour2 = ax4.contourf(Theta1, Theta2, tau_total_2, 20, cmap='viridis')
ax4.set_xlabel(r'$\theta_1$ (rad)')
ax4.set_ylabel(r'$\theta_2$ (rad)')
ax4.set_title('Joint 2 Torque (Contour)')
fig.colorbar(contour2, ax=ax4, shrink=0.5, aspect=5)

# 增加子图之间的间距
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()