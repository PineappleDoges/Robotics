import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------------
# Define robot parameters (real values for simulation)
# ----------------------------
m1, m2 = 1.0, 1.0        # link masses
l1, l2 = 1.0, 1.0        # link lengths
lc1, lc2 = 0.5, 0.5      # CoM distances
I1, I2 = 0.1, 0.1        # link inertias
g = 9.81

# ----------------------------
# Desired trajectory and derivatives
# ----------------------------
def qd(t):
    theta1 = 2*(1 - np.exp(-t))
    theta2 = 3*(1 - np.exp(-2*t))
    return np.array([theta1, theta2])

def qd_dot(t):
    return np.array([2*np.exp(-t), 6*np.exp(-2*t)])

def qd_ddot(t):
    return np.array([-2*np.exp(-t), -12*np.exp(-2*t)])

# ----------------------------
# Robot dynamics functions
# ----------------------------
def H_matrix(q):
    th1, th2 = q
    H11 = m1*lc1**2 + I1 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(th2)) + I2
    H12 = m2*l1*lc2*np.cos(th2) + m2*lc2**2 + I2
    H22 = m2*lc2**2 + I2
    return np.array([[H11, H12],[H12, H22]])

def C_matrix(q, q_dot):
    th1, th2 = q
    dth1, dth2 = q_dot
    h = m2*l1*lc2*np.sin(th2)
    return np.array([[-h*dth2, -h*(dth1 + dth2)],
                     [h*dth1, 0]])

def g_vector(q):
    th1, th2 = q
    G1 = m1*lc1*g*np.cos(th1) + m2*lc2*g*np.cos(th1 + th2) + m2*l1*g*np.cos(th1)
    G2 = m2*lc2*g*np.cos(th1 + th2)
    return np.array([G1, G2])

# ----------------------------
# Adaptive controller parameters
# ----------------------------
Lambda = np.diag([5.0, 5.0])
Kv = np.diag([10.0, 10.0])
Gamma = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])  # 5 parameters now

# ----------------------------
# Proper Regression function Y(q, q_dot, q_r_dot, q_r_ddot)
# ----------------------------
def Y_matrix(q, q_dot, q_r_dot, q_r_ddot):
    th1, th2 = q
    dth1, dth2 = q_dot
    dqr1, dqr2 = q_r_dot
    ddqr1, ddqr2 = q_r_ddot
    
    Y = np.zeros((2, 5))
    
    # First row of regression matrix
    Y[0, 0] = ddqr1
    Y[0, 1] = ddqr1 + ddqr2
    Y[0, 2] = 2*np.cos(th2)*ddqr1 + np.cos(th2)*ddqr2 - np.sin(th2)*(dth2*dqr1 + (dth1 + dth2)*dqr2)
    Y[0, 3] = np.cos(th1)
    Y[0, 4] = np.cos(th1 + th2)
    
    # Second row of regression matrix
    Y[1, 0] = 0
    Y[1, 1] = ddqr1 + ddqr2
    Y[1, 2] = np.cos(th2)*ddqr1 + np.sin(th2)*dth1*dqr1
    Y[1, 3] = 0
    Y[1, 4] = np.cos(th1 + th2)
    
    return Y

# ----------------------------
# Simulation dynamics
# state = [q1, q2, dq1, dq2, theta_hat (5 params)]
# ----------------------------
def dynamics(t, state):
    q = state[0:2]
    q_dot = state[2:4]
    theta_hat = state[4:9]  # 5 parameters now

    # Desired trajectory
    q_d = qd(t)
    q_d_dot = qd_dot(t)
    q_d_ddot = qd_ddot(t)
    
    # Tracking errors
    e = q_d - q
    e_dot = q_d_dot - q_dot
    
    # Reference trajectory
    q_r_dot = q_d_dot + Lambda @ e
    q_r_ddot = q_d_ddot + Lambda @ e_dot
    
    # Sliding variable
    s = q_dot - q_r_dot

    # Regression matrix and control law
    Y = Y_matrix(q, q_dot, q_r_dot, q_r_ddot)
    tau = Y @ theta_hat - Kv @ s

    # Robot dynamics (true plant)
    H = H_matrix(q)
    C = C_matrix(q, q_dot)
    g_vec = g_vector(q)
    q_ddot = np.linalg.inv(H) @ (tau - C @ q_dot - g_vec)

    # Parameter update law
    theta_hat_dot = Gamma @ Y.T @ s

    # Pack derivatives
    return np.concatenate([q_dot, q_ddot, theta_hat_dot])

# ----------------------------
# Initial conditions
# ----------------------------
q0 = np.array([0.0, 0.0])
q_dot0 = np.array([0.0, 0.0])
theta_hat0 = np.zeros(5)  # 5 parameters now
state0 = np.concatenate([q0, q_dot0, theta_hat0])

# ----------------------------
# Run simulation
# ----------------------------
print("Starting simulation...")
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)
sol = solve_ivp(dynamics, t_span, state0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

print("Simulation completed!")

# ----------------------------
# Plot results
# ----------------------------
t_eval = sol.t
q_sol = sol.y[0:2, :]
theta_hat_sol = sol.y[4:9, :]

# Calculate desired trajectory
q_d_sol = np.array([qd(t) for t in t_eval]).T
error = q_d_sol - q_sol

plt.figure(figsize=(12, 10))

# Plot joint trajectories
plt.subplot(3, 2, 1)
plt.plot(t_eval, q_sol[0, :], 'b-', linewidth=2, label='Actual θ1')
plt.plot(t_eval, q_d_sol[0, :], 'r--', linewidth=2, label='Desired θ1')
plt.xlabel('Time [s]')
plt.ylabel('Joint angle [rad]')
plt.title('Joint 1 Tracking')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t_eval, q_sol[1, :], 'b-', linewidth=2, label='Actual θ2')
plt.plot(t_eval, q_d_sol[1, :], 'r--', linewidth=2, label='Desired θ2')
plt.xlabel('Time [s]')
plt.ylabel('Joint angle [rad]')
plt.title('Joint 2 Tracking')
plt.legend()
plt.grid(True)

# Plot tracking errors
plt.subplot(3, 2, 3)
plt.plot(t_eval, error[0, :], 'g-', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Error [rad]')
plt.title('Joint 1 Tracking Error')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(t_eval, error[1, :], 'g-', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Error [rad]')
plt.title('Joint 2 Tracking Error')
plt.grid(True)

# Plot parameter estimates
plt.subplot(3, 2, 5)
for i in range(5):
    plt.plot(t_eval, theta_hat_sol[i, :], linewidth=2, label=f'θ_hat[{i}]')
plt.xlabel('Time [s]')
plt.ylabel('Parameter estimates')
plt.title('Parameter Convergence')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
# Calculate true parameter values for comparison
phi_true = np.zeros(5)
phi_true[0] = m1*lc1**2 + m2*l1**2 + I1
phi_true[1] = m2*lc2**2 + I2
phi_true[2] = m2*l1*lc2
phi_true[3] = m1*lc1*g + m2*l1*g
phi_true[4] = m2*lc2*g

for i in range(5):
    plt.plot(t_eval, theta_hat_sol[i, :] - phi_true[i], linewidth=2, label=f'Error[{i}]')
plt.xlabel('Time [s]')
plt.ylabel('Parameter error')
plt.title('Parameter Estimation Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final parameter estimates and true values
print("\nTrue parameter values:")
print(f"π1 = {phi_true[0]:.3f} (m1*lc1² + m2*l1² + I1)")
print(f"π2 = {phi_true[1]:.3f} (m2*lc2² + I2)")
print(f"π3 = {phi_true[2]:.3f} (m2*l1*lc2)")
print(f"π4 = {phi_true[3]:.3f} (m1*lc1*g + m2*l1*g)")
print(f"π5 = {phi_true[4]:.3f} (m2*lc2*g)")

print("\nFinal parameter estimates:")
for i in range(5):
    print(f"θ_hat[{i}] = {theta_hat_sol[i, -1]:.3f} (error: {theta_hat_sol[i, -1] - phi_true[i]:.3f})")