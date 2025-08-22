import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ ---
A = np.array([
    [0, -0.7],
    [1, -1.5]
])
B = np.array([
    [0.5],
    [1]
])
C = np.array([
    [0],
    [1]
])

N, Q, R = 100, 1, 0.1

# --- 関数 ---
def f(x):
    return A @ x

def h(x):
    return C.T @ x

def dfdx(x):
    return A

def dhdx(x):
    return C.T

# --- 真値，　測定値 ---
v = np.random.normal(loc = 0.0, scale = np.sqrt(Q), size = N)
w = np.random.normal(loc = 0.0, scale = np.sqrt(R), size = N)

x = [np.array([[0],
                [0]])]
y = [h(x[-1]) + w[0]]
for k in range(N - 1):
    x.append(f(x[-1]) + B * v[k + 1])
    y.append(h(x[-1]) + w[k + 1])
x = np.array(x)
y = np.array(y)



def ekf(f,h,dfdx,B,dhdx,Q,R,y,xhat,P):
    """
    x(k+1) = f(x(k)) + Bv(k)
    y(k) = h(x(k)) + w(k)
    """
    xhatm = f(xhat)
    A = dfdx(xhat)
    Pm = (A @ P @ A.T) + Q * (B @ B.T)
    CT = dhdx(xhatm)
    C = CT.T
    G = (Pm @ C) / ((CT @ Pm @ C) + R)
    xhat_new = xhatm + G * (y - h(xhatm))
    P_new = (np.eye(len(A)) - (G @ CT)) @ Pm
    return xhat_new, P_new, G

def calc_timeseries_ekf(N, f, h, dfdx, B, dhdx, Q, R, y, x0, P_ekf):
    xhat_ekf = [x0]
    G0 = np.empty(x0.shape)
    G0[:] = np.nan
    G_ekf = [G0]
    for k in range(1, N):
        xhat_new, P_new, G = ekf(f, h, dfdx, B, dhdx, Q, R, y[k], xhat_ekf[-1], P_ekf)
        xhat_ekf.append(xhat_new)
        G_ekf.append(G)
        P_ekf = P_new
    xhat_ekf = np.array(xhat_ekf)
    G_ekf= np.array(G_ekf)
    return xhat_ekf, G_ekf

# --- EKF ---
gamma = 1
P_ekf = gamma * np.eye(2)
xhat_ekf, G_ekf = calc_timeseries_ekf(N, f, h, dfdx, B, dhdx, Q, R, y, x[0], P_ekf)

# # --- プロット ---
# fig_num = 2
# fig = plt.figure(figsize = (10, 3 * fig_num), tight_layout = True)
# axes = [fig.add_subplot(fig_num, 1, i + 1) for i in range(fig_num)]
# for i in range(fig_num):
#     ax = axes[i]
#     ax2 = ax.twinx()
#     ax.plot(y[:, 0, 0], label = 'measured', c = 'tab:blue')
#     ax.plot(x[:, i], label = 'true', ls = 'dotted', c = 'red')
#     ax.plot(xhat_ekf[:, i], label = 'estimated (EKF)', c = 'tab:orange')
#     ax2.plot(G_ekf[:, i], label = 'gain', ls = 'dashed', c = 'tab:gray')
#     ax.set_xlabel('k')
#     ax.set_ylabel(f'x{i + 1}')
#     ax2.set_ylabel(f'g{i + 1}')
#     lines, labels = ax.get_legend_handles_labels()
#     li2, la2 = ax2.get_legend_handles_labels()
#     lines += li2
#     labels += la2
#     ax2.legend(lines, labels, loc = 'lower right')
# plt.show()