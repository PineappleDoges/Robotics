import numpy as np
import matplotlib.pyplot as plt

#design parameters
L1, L2 = 1.0, 1.0
theta2 = np.linspace(-np.pi, np.pi, 1000)

theta2 = theta2[theta2 != 0]

d = np.sqrt(L1**2 + L2**2 + 2*L1*L2*np.cos(theta2))
tau1 = (L1 * L2 * np.sin(theta2)) / d
tau2 = np.zeros_like(tau1)

plt.figure(figsize=(10, 6))
plt.plot(np.degrees(theta2), tau1, 'b-', linewidth=2, label='τ₁')
plt.plot(np.degrees(theta2), tau2, 'r--', linewidth=2, label='τ₂')
plt.xlabel('θ₂ (degrees)')
plt.ylabel('Joint Torque')
plt.title('Joint Torques vs Configuration (L₁ = L₂ = 1)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlim(-180, 180)
plt.show()