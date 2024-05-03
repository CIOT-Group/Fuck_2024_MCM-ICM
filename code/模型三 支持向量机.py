import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import joblib


# 加载数据
data = pd.read_csv('dataset.csv')
X = data[['data1', 'data2', 'data3', 'data4',
          'data5', 'data6', 'data7', 'data8']]
y = data['result']

introduction = """
输入说明：
1. 历史价值: 当前年份减去建筑建造的年代
2. 文化法律: 是否列入国家遗产或是否受到法律保护 (列入国家遗产: 2, 受到法律保护: 1, 无保护/无遗产: 0)
3. 经济价值: 是否开发旅游业, 博物馆 (人工打分, 0-10分打分)
4. 社会认同: 社区认同度, 是否用于文化活动 (人工打分, 0-10分打分)
5. 建筑状态: 维护修复情况, 破坏程度 (人工打分, 0-10分的打分, 分数越低越差)
6. 自然灾害: 根据年自然灾害数量得出	
7. 城市规划: 产业结构 (第一产业: 1, 第二产业: 2, 第三产业: 3)
8. 可见性: 是否可见 (可见: 1, 不可见: 0)
"""

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 训练SVM模型并添加正则化
C_value = 0.05  # 正则化参数，可以根据需要进行调整
model = svm.SVC(kernel='rbf', probability=True, C=C_value)
model.fit(X_train, y_train)


# 保存模型到文件
joblib.dump(model, 'svm_model.pkl')
loaded_model = model  # 将训练好的模型赋值给loaded_model

# 预测
y_pred = loaded_model.predict(X_test)
y_proba = loaded_model.predict_proba(X_test)[:, 1]

# 计算准确率, 精确度, 召回率, F1分数
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)
# ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', format='png')  # 保存为ROC曲线的PNG文件

# PR曲线
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.savefig('pr_curve.png', format='png')  # 保存为PR曲线的PNG文件


# 在数据预处理后进行PCA降维
pca = PCA(n_components=2)  # 降至2维以便可视化
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# 重新训练SVM模型)使用降维后的数据(
model_pca = svm.SVC(kernel='rbf', probability=True)
model_pca.fit(X_train_pca, y_train)

# 绘制决策边界
plt.figure()
h = .02  # 网格中的步长
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# 绘制数据点
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
            c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM Decision Boundary with PCA-transformed data')
# 保存为PNG文件
plt.savefig('decision_boundary.png', format='png')


def predict_outcome(features):  # 预测
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)  # 应用同样的缩放
    proba = loaded_model.predict_proba(features)[0, 1]
    return loaded_model.predict(features), proba


is_predictic = input("是否需要预测结果？(Y/N)")
if is_predictic.upper() == 'Y':
    print(introduction)
    print("请输入8个因素数据, 用空格隔开: ")
    print("历史价值  文化法律  经济价值  社会认同  建筑状态  自然灾害  城市规划  可见性")
    user_input = input().split()
    user_input = [float(x) for x in user_input]  # 将输入转换为浮点数列表
    if len(user_input) != 8:
        print("输入错误")
    else:
        prediction, probability = predict_outcome(user_input)
        print("模型预测结果:", prediction[0])
        print("模型概率:", probability)
else:
    pass
