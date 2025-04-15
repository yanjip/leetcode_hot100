# time: 2025/3/3 17:28
# author: YanJP
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 根据训练数据集中最接近当前样本的K个邻居来预测新样本的类别或值。
#  在预测阶段，计算复杂度较高，因为需要计算每个新样本与所有训练样本的距离，复杂度为O(n)。

# 对于每个测试样本，计算其与所有训练样本的欧氏距离。
# 找到距离最近的 k个样本。
# 统计这些样本的标签，返回频率最高的标签作为预测结果。

# 生成随机数据
np.random.seed(42)
def generate_data(n_samples, centers, cluster_std):
    X = []
    y = []
    for center, std in zip(centers, cluster_std):
        X.append(np.random.normal(loc=center, scale=std, size=(n_samples, 2)))
        y.append(np.full(n_samples, center[0]))  # 使用中心点的第一个值作为标签
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y

# 生成 3 个簇的数据
centers = [[1, 1], [4, 4], [7, 1]]
cluster_std = [0.5, 0.5, 0.5]
X_train, y_train = generate_data(n_samples=100, centers=centers, cluster_std=cluster_std)

# 可视化训练数据
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', edgecolor='k')
plt.title("Training Data")
plt.show()

# KNN 算法实现
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict_one(x) for x in X]
        return np.array(y_pred)

    def _predict_one(self, x):
        # 计算距离
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # 找到最近的 k 个样本
        k_indices = np.argsort(distances)[:self.k]
        # 获取这些样本的标签
        k_labels = self.y_train[k_indices]
        # 统计标签的频率，返回频率最高的标签
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# 使用 KNN 进行分类
knn = KNN(k=3)
knn.fit(X_train, y_train)

# 生成测试数据
X_test = np.array([[2, 2], [5, 5], [6, 0]])
y_test = knn.predict(X_test)

# 可视化测试数据
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', edgecolor='k', label="Training Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', s=200, label="Test Data")
plt.title("KNN Classification")
plt.legend()
plt.show()
