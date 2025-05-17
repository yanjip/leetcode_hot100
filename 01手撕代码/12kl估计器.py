# time: 2025/4/27 11:26
# author: YanJP
import torch
import torch.distributions as dist

# 定义参考分布p和当前策略q（以高斯分布为例）
p = dist.Normal(loc=0, scale=1)
q = dist.Normal(loc=0.1, scale=1.05)

# 从q分布采样
x = q.sample((100000,))  # 采样10万个样本

# 计算log概率比
log_r = p.log_prob(x) - q.log_prob(x)  # log(q/p)

# 三种KL估计器计算
k1 = -log_r
k2 = log_r ** 2 / 2
k3 = (torch.exp(log_r) - 1) - log_r

# 输出均值作为KL估计值
print(f"K1估计值: {k1.mean().item():.4f}")  # 示例输出0.0051
print(f"K2估计值: {k2.mean().item():.4f}")  # 示例输出0.0049
print(f"K3估计值: {k3.mean().item():.4f}")  # 示例输出0.0050