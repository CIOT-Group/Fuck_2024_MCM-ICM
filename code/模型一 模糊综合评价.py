import numpy as np

R1 = np.array([
    [0.0015,	0.1626,	0.6826,	0.1521,	0.0013],
    [0,	0.0924,	0.5849,	0.2133,	0.1094],
    [0,	0.0098,	0.3592,	0.5832,	0.0478],
    [0.1583,	0.6836,	0.1567,	0.0013,	0],
    [0.0069,	0.3154,	0.6157,	0.0617,	0.0002],
    [0.0369,	0.5473,	0.4024,	0.0135,	0],
])
R2 = np.array([
    [0.0065,	0.3073,	0.6213,	0.0647,	0.0002],
    [0,	0.0025,	0.2067,	0.6745,	0.1163],
    [0.0026,	0.2107,	0.6726,	0.1135,	0.0007],
    [0.0147,	0.4145,	0.5366,	0.0342,	0.0001],
    [0,	0.0045,	0.2647,	0.6481,	0.0828],
    [0.134,	0.6807,	0.1833,	0.0019,	0]
])

ACC_ALL = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
])
ACC1 = ACC_ALL.copy()
ACC2 = ACC_ALL.copy()

A1 = np.array([0.3231, 0.0734, 0.3045, 0.0481, 0.0991, 0.1518])
A2 = np.array([0.1079, 0.0987, 0.2127, 0.0766, 0.0699, 0.4342])
B_std1 = np.std(np.dot(A1, R1))
B_std2 = np.std(np.dot(A2, R2))
B_s = np.std(0.5*(np.dot(A2, R2)+np.dot(A1, R1)))


for i in range(6):
    for j in range(3):
        R1_modified = R1.copy()
        R1_modified[i] *= 1.05
        B1 = np.dot(A1, R1_modified)
        B1_std = np.std(B1)
        ACC1[i, j] = 100 * abs(B1_std - B_std1) / B_std1
        R2_modified = R2.copy()
        R2_modified[i] *= 1.05
        B2 = np.dot(A2, R2_modified)
        B2_std = np.std(B2)
        ACC2[i, j] = 100 * abs(B2_std - B_std2) / B_std2
        B = 0.5*(B1+B2)
        B_std = np.std(B)
        ACC_ALL[i, j] = 100 * abs(B_std - B_s) / B_s

print("R1模型指标")
print(np.dot(A1, R1))
print("R2模型指标")
print(np.dot(A2, R2))

print("R1灵敏度分析")
print(ACC1)
print("R2灵敏度分析")
print(ACC2)
print("总灵敏度分析")
print(ACC_ALL)
