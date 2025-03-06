# time: 2025/3/3 17:22
# author: YanJP
import numpy as np
import matplotlib.pyplot as plt

# 初始化：随机选择 k 个样本作为初始簇中心。
# 分配样本：计算每个样本到簇中心的距离，将样本分配到最近的簇。
# 更新簇中心：计算每个簇的新中心（簇内样本的均值）。
# 收敛检查：如果簇中心的变化小于阈值 tol，则停止迭代。

# np.random.seed(42) # 生成随机数据
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
centers = [[1, 1], [3, 4], [5, 1]]
cluster_std = [0.5, 0.6, 0.9]
X, y = generate_data(n_samples=100, centers=centers, cluster_std=cluster_std)

# 可视化原始数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
plt.title("Original Data")
plt.show()

# K-means 算法实现
class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # 随机初始化簇中心
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]  # 选择三个点，然后取出它们的坐标

        for _ in range(self.max_iter):
            # 分配样本到最近的簇
            labels = self._assign_clusters(X)

            # 更新簇中心
            new_centers = self._update_centers(X, labels)

            # 检查是否收敛
            if np.linalg.norm(new_centers - self.centers) < self.tol:
                break

            self.centers = new_centers

        self.labels_ = self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centers(self, X, labels):
        new_centers = np.zeros_like(self.centers)
        for i in range(self.n_clusters):
            new_centers[i] = np.mean(X[labels == i], axis=0)
        return new_centers

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], c='red', marker='x', s=200, label="Centers")
plt.title("K-means Clustering")
plt.legend()
plt.show()
