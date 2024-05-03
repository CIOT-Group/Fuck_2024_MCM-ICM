import numpy as np
import math as mt

# 输入原始数据和预测的n值
X0 = ([6.9413,
       7.4000,
       0.8624,
       3.6547,
       7.9425

       ])


def GM_11_predict(X0, n):
    # 累加数列
    X1 = [X0[0]]
    add = X0[0] + X0[1]
    X1.append(add)
    i = 2
    while i < len(X0):
        add = add + X0[i]
        X1.append(add)
        i += 1
    # 紧邻均值序列
    Z = []
    j = 1
    while j < len(X1):
        num = (X1[j] + X1[j - 1]) / 2
        Z.append(num)
        j = j + 1
    # 最小二乘法计算
    Y = []
    x_i = 0
    while x_i < len(X0) - 1:
        x_i += 1
        Y.append(X0[x_i])
    Y = np.mat(Y)
    Y = Y.reshape(-1, 1)
    B = []
    b = 0
    while b < len(Z):
        B.append(-Z[b])
        b += 1
    B = np.mat(B)
    B = B.reshape(-1, 1)
    c = np.ones((len(B), 1))
    B = np.hstack((B, c))
    # 求出参数
    alpha = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    a = alpha[0, 0]
    b = alpha[1, 0]
    # 生成预测模型
    GM = []
    GM.append(X0[0])
    did = b / a
    k = 1
    while k <= n:
        GM.append((X0[0] - did) * mt.exp(-a * k) + did)
        k += 1
    # 做差得到预测序列
    G = [GM[i] - GM[i - 1] for i in range(1, len(GM))]
    return G


def GM_11_correct(X0, G):
    # 计算残差
    e = []
    i = 0
    while i < len(X0):
        e.append(X0[i] - G[i])
        i += 1

    # 计算相对误差
    q = []
    i = 0
    while i < len(X0):
        q.append(abs(e[i] / X0[i]))
        i += 1

    # 计算平均相对误差
    mean_q = np.mean(q)

    # 精度为绝对误差平均值的百分比
    accuracy = round((1 - mean_q) * 100, 2)
    print('精度为：{}%'.format(accuracy))

    # 计算方差比
    s0 = np.var(X0)
    s1 = np.var(e)
    S0 = mt.sqrt(s0)
    S1 = mt.sqrt(s1)
    C = S1 / S0
    print('方差比为：', C)

    # 计算小概率误差
    P = np.sum(np.abs(e - np.mean(e)) < 0.6745 * S1) / len(e)
    print('小概率误差为：', P)


m = len(X0)
n = int(input("请输入要预测的数量：")) + m
predicted_values = GM_11_predict(X0, n)
# 将X0的第一位加到predicted_values的首位置
predicted_values.insert(0, X0[0])

print("原始数列为：", X0)
print("预测数列为：", predicted_values)
print()
GM_11_correct(X0, GM_11_predict(X0, m))
