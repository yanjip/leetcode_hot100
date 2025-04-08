# time: 2025/4/1 22:11
# author: YanJP
import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(num_features))  # 偏移参数
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))  # 推理时使用的均值
        self.register_buffer('running_var', torch.ones(num_features))  # 推理时使用的方差

    def forward(self, x):
        if self.training:
            # 训练模式：计算当前 batch 的均值和方差
            mean = x.mean(dim=0, keepdim=True)  # 沿 batch 维度
            var = x.var(dim=0, keepdim=True, unbiased=False)
            # 更新 running_mean 和 running_var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 推理模式：使用预计算的 running_mean 和 running_var
            mean, var = self.running_mean, self.running_var

        # 标准化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class LayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        # 沿特征维度计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # [batch_size, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 标准化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# 去除了均值中心化，仅用均方根（RMS）进行缩放：无 β 参数，仅保留 γ。
class RMSNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, x):
        # 计算均方根 (RMS)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)

if __name__ == '__main__':
    # 测试代码
    x = torch.randn(8, 4)  # 模拟输入

    bn = BatchNorm1d(4)
    ln = LayerNorm1d(4)
    rms = RMSNorm1d(4)

    print("BatchNorm:\n", bn(x))
    print("LayerNorm:\n", ln(x))
    print("RMSNorm:\n", rms(x))
