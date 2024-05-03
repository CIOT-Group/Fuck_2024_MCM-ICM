import numpy as np

# 输入母序列和子序列，第一列母序列，其余子序列
A = np.array([
    [53.6, 8, 8, 0.86242926, 3, 8],
    [46.4, 4, 7, 0.93907558, 10, 9]
])

mean = np.mean(A, axis=0)
A_norm = A / mean

Y = A_norm[:, 0]
X = A_norm[:, 1:]

absX0_Xi = np.abs(X - np.tile(Y.reshape(-1, 1), (1, X.shape[1])))
a = np.min(absX0_Xi)
b = np.max(absX0_Xi)
rho = 0.5
gamma = (a+rho*b)/(absX0_Xi+rho*b)

print("子序列中各个指标的灰色关联度分别为：")
print(np.mean(gamma, axis=0))
