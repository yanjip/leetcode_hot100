# time: 2025/2/25 22:29
# author: YanJP
import numpy as np

# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 损失函数（对数损失）
def compute_loss(y_true, y_pred):
    m = len(y_true)
    loss = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 梯度计算
def compute_gradients(X, y_true, y_pred):
    m = len(y_true)
    dw = (1/m) * np.dot(X.T, (y_pred - y_true))  # 权重梯度
    db = (1/m) * np.sum(y_pred - y_true)         # 偏置梯度
    return dw, db

# 逻辑斯蒂回归训练
def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    # 初始化参数
    n_features = X.shape[1]
    w = np.zeros(n_features)  # 权重
    b = 0                     # 偏置

    # 训练过程
    for i in range(num_iterations):
        # 计算预测值
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        # 计算损失
        loss = compute_loss(y, y_pred)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

        # 计算梯度
        dw, db = compute_gradients(X, y, y_pred)

        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

# 预测函数
def predict(X, w, b):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)  # 将概率转换为类别（0或1）

# 示例数据
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # 特征矩阵
y = np.array([0, 0, 0, 1, 1])                           # 标签

# 训练逻辑斯蒂回归模型
w, b = logistic_regression(X, y, learning_rate=0.1, num_iterations=1000)

# 预测
y_pred = predict(X, w, b)
print("Predictions:", y_pred)
