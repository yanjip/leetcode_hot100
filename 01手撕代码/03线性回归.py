# time: 2025/3/3 16:48
# author: YanJP
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 输入特征 (100 个样本)
y = 4 + 3 * X + np.random.randn(100, 1)  # 输出目标 (带噪声)

# 添加偏置项
X_b = np.c_[np.ones((100, 1)), X]  # X_b = [1, X]

# 初始化参数
theta = np.random.randn(2, 1)  # 参数 [w0, w1]

# 定义损失函数 (均方误差)
def compute_loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    loss = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return loss

# 梯度下降
learning_rate = 0.1
n_iterations = 1000

for iteration in range(n_iterations):
    gradients = (1 / len(X_b)) * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

# 预测
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred = X_new_b.dot(theta)

# 可视化
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X_new, y_pred, color="red", label="Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression (Handwritten)")
plt.legend()
plt.show()

# 输出模型参数
print("Intercept (w0):", theta[0][0])
print("Coefficient (w1):", theta[1][0])
