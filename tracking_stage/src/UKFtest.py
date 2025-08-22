import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

# --- パラメータ ---
dt = 1
# UKFのパラメータ
alpha = 1e-3
beta = 2
kappa = 0
n = 4  # 状態ベクトルの次元

A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0 ,dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B = np.array([
    [0, 0],
    [0, 0],
    [dt, 0],
    [0, dt]
])

C = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 状態ベクトルと観測ノイズの共分散行列
Q = np.eye(n)  # プロセスノイズ
R = np.eye(n)  # 観測ノイズ


# --- 関数 ---
def f(x):
    return A

def h(x):
    return C.T @ x

def dfdx(x):
    return A

def dhdx(x):
    return C.T

def ttt(x):
    return x.transpose()

# システムダイナミクス
def system_dynamics(x, u):
    # ここにシステムの遷移モデルを実装
    # 例: x = A * x + B * u + noise
    x = np.array(x)
    u = np.array(u)
    x = A @ x.transpose() + B @ u.transpose() + Q
    return x

# 観測モデル
def observation_model(x):
    # ここに観測モデルを実装
    # 例: z = H * x + noise
    x = np.array(x)
    x = C @ x.transpose() + R
    return x


# 初期状態と共分散行列
x0 = np.zeros((1,4))  # 初期状態
u0 = np.zeros((1,2))
P0 = np.eye(n)  # 初期共分散行列

# UKFの重みとシグマポイントの生成
lambda_ = alpha**2 * (n + kappa) - n
weights = np.zeros(2*n+1)
weights[0] = lambda_ / (n + lambda_)
weights[1:] = 1 / (2*(n + lambda_))

# シグマポイントの生成
def sigma_points(x, P):
    L = cholesky((n + lambda_) * P, lower=True)
    sigmas = np.zeros((2 * n + 1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[i+1] = x + L[i]
        sigmas[i+n+1] = x - L[i]
    return sigmas


# 予測ステップ
def predict(sigmas, dt):
    predicted_sigmas = []
    for sigma in sigmas:
        predicted_sigma = system_dynamics(sigma, dt)
        predicted_sigmas.append(predicted_sigma)
    return np.array(predicted_sigmas)


# フィルタリングステップ
def update(predicted_sigmas, z):
    n = len(predicted_sigmas[0])
    x_mean = np.sum(weights[i] * predicted_sigmas[i] for i in range(2*n+1))
    Pzz = np.zeros((n, n))
    Pxz = np.zeros((n, n))
    for i in range(2 * n + 1):
        z_pred = observation_model(predicted_sigmas[i])
        Pzz += weights[i] * np.outer(z_pred - z_mean, z_pred - z_mean)
        Pxz += weights[i] * np.outer(predicted_sigmas[i] - x_mean, z_pred - z_mean)
    K = Pxz @ np.linalg.inv(Pzz)
    x_hat = x_mean + K @ (z - z_mean)
    P_hat = P - K @ Pzz @ K.T
    return x_hat, P_hat

# 追跡のメインループ
num_steps = 100
dt = 0.1
trajectory = []
observations = []

x = x0
u = u0
P = P0
for _ in range(num_steps):
    # 真の状態を更新
    x = system_dynamics(x, dt)
    
    # 観測を生成（例：真の状態にノイズを加える）
    z = observation_model(x) + np.random.multivariate_normal(np.zeros(n), R)
    
    z_mean = observation_model(x)
    
    # UKFの予測ステップ
    sigmas = sigma_points(x, P)
    predicted_sigmas = predict(sigmas, dt)
    
    # UKFのフィルタリングステップ
    x, P = update(predicted_sigmas, z)
    
    # 結果を保存
    trajectory.append(x)
    observations.append(z)

# 結果のプロット
trajectory = np.array(trajectory)
observations = np.array(observations)

plt.figure(figsize=(10, 5))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='True Trajectory', color='blue')
plt.scatter(observations[:, 0], observations[:, 1], label='Observations', color='red', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
