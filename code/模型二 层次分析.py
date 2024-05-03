import numpy as np

# 判断矩阵P，不用修改
P = np.array([
    [1,     3,      1/5,    4,      2],
    [1/3,   1,      1/3,    3,      2],
    [5,     3,      1,      7,      5],
    [1/4,   1/3,    1/7,    1,      2],
    [1/2,   1/2,    1/5,    1/2,    1]
])

# 因素输入：气候韧性建筑标准,风险缓解策略,风险模型参与,能源可持续发展,社区政府
# 有几条输几行
value = np.array([
    [8.8642,
     9.0896,
     0.8624,
     4.4608,
     8.1920






     ],
    [4.1759,
     7.0220,
     0.9391,
     8.6926,
     9.5791






     ]
])


def tryP(P):
    # 求P的特征值和特征向量
    eigenvalue, featurevector = np.linalg.eig(P)
    # 求最大特征值
    max_eigenvalue = max(eigenvalue)
    # 求一致性指标
    CI = (max_eigenvalue - len(P)) / (len(P) - 1)
    RI = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49]
    CR = CI / RI[len(P)-1]

    print("CI:", CI, end="\t")
    print("RI:", RI[len(P)-1], end="\t")
    print("CR:", CR)

    if CR < 0.1:
        print("CR < 0.1, 判断矩阵一致")
        normalized_featurevector = featurevector[:, np.argmax(eigenvalue)]
        normalized_featurevector /= np.sum(normalized_featurevector)
        normalized_featurevector = np.real(normalized_featurevector)
        print("归一化特征向量/权重:", normalized_featurevector)
        return normalized_featurevector

    elif CR >= 0.1:
        print("CR >= 0.1, 判断矩阵不一致，需修改")
        return False


weight = tryP(P)
# 加权求和并换算成百分制
result = weight*value
sum_result = np.sum(result, axis=1)
sum_result = (sum_result / np.sum(sum_result)) * 100
sum_result = np.round(sum_result, 2)
# 转置成列向量
sum_result = sum_result.reshape(-1, 1)
print("加权并换算成百分制后的结果：\n", sum_result)
