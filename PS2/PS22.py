import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================
# 参数设置
# =============================
m1, m2 = 1.0, 1.0      # 两个杆的质量
l1, l2 = 1.0, 1.0      # 两个杆的长度
lc1, lc2 = 0.5, 0.5    # 质心到关节的距离
I1, I2 = 0.1, 0.1      # 转动惯量
g = 9.81               # 重力加速度

# =============================
# 动力学方程定义
# =============================
def dynamics(t, y):
    q1, q2, q1dot, q2dot = y

    # 惯性矩阵 D(q)
    d11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(q2))
    d12 = I2 + m2*(lc2**2 + l1*lc2*np.cos(q2))
    d21 = d12
    d22 = I2 + m2*lc2**2
    D = np.array([[d11, d12],
                  [d21, d22]])

    # 科里奥利和离心力项 C(q,qdot)*qdot
    h = -m2*l1*lc2*np.sin(q2)
    c1 = h*q2dot*(2*q1dot + q2dot)
    c2 = h*q1dot**2
    C = np.array([c1, c2])

    # 重力项 G(q)
    g1 = (m1*lc1 + m2*l1)*g*np.cos(q1) + m2*lc2*g*np.cos(q1 + q2)
    g2 = m2*lc2*g*np.cos(q1 + q2)
    G = np.array([g1, g2])

    # 计算加速度 qddot = D⁻¹ * ( -C - G )
    qddot = np.linalg.solve(D, -C - G)

    return [q1dot, q2dot, qddot[0], qddot[1]]

# =============================
# 初始条件
# =============================
q1_0 = np.pi / 2   # 初始角度
q2_0 = np.pi / 3
q1dot_0 = 0.0      # 初始角速度
q2dot_0 = 0.0
y0 = [q1_0, q2_0, q1dot_0, q2dot_0]

# =============================
# 积分求解
# =============================
t_span = (0, 10)               # 模拟时间（秒）
t_eval = np.linspace(0, 10, 1000)
sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, rtol=1e-8, atol=1e-8)

# =============================
# 可视化动画
# =============================
q1 = sol.y[0]
q2 = sol.y[1]
x1 = l1 * np.sin(q1)
y1 = -l1 * np.cos(q1)
x2 = x1 + l2 * np.sin(q1 + q2)
y2 = y1 - l2 * np.cos(q1 + q2)

fig, ax = plt.subplots()
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2)

def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    return line,

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=10, blit=True)
plt.title("2-Link Free Pendulum Simulation")
plt.show()
