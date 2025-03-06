# time: 2025/3/3 15:42
# author: YanJP
import torch
import numpy as np
import torch.nn.functional as F


def softmax(x):
    """
    数值稳定的 Softmax 函数实现。

    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, num_classes).

    返回:
        probs (torch.Tensor): Softmax 输出的概率分布，形状与输入相同.
    """
    # Step 1: 沿类别维度找到每个样本的最大值
    max_x, _ = torch.max(x, dim=-1, keepdim=True)

    # Step 2: 减去最大值，避免指数运算溢出
    shifted_x = x - max_x

    # Step 3: 对减去最大值后的结果进行指数运算
    exp_x = torch.exp(shifted_x)

    # Step 4: 计算每个样本的指数和（沿类别维度求和）
    sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)

    # Step 5: 计算 Softmax 概率
    probs = exp_x / sum_exp_x

    return probs
def manual_cross_entropy_loss(logits, targets):
    """
    手动实现 nn.CrossEntropyLoss 功能。
    参数:
        logits (torch.Tensor): 模型的原始输出 (未经过 Softmax), 形状为 (batch_size, num_classes).
        targets (torch.Tensor): 真实标签, 形状为 (batch_size,), 每个元素是类别的索引.
    返回:
        loss (torch.Tensor): 计算得到的交叉熵损失.
    """
    # Step 1: 对 logits 进行 Softmax 操作，得到概率分布
    probs = softmax(logits)

    # Step 2: 对概率分布取对数
    log_probs = torch.log(probs)

    # Step 3: 使用 gather 函数获取真实类别对应的对数概率
    batch_size = logits.shape[0]
    # yi是一个one - hot向量（或类别的索引），只有真实类别对应的位置为1，其余位置为0。因此，计算交叉熵损失时，只需要提取真实类别对应的对数概率即可。
    # log_probs_target = log_probs[torch.arange(batch_size), targets]
    log_probs_target = torch.gather(log_probs, dim=1, index=targets.unsqueeze(1)).squeeze(1)

    # Step 4: 计算交叉熵损失（取负对数概率的均值）
    loss = -log_probs_target.mean()

    return loss
# 示例数据
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]], requires_grad=True)  # 模型的原始输出
targets = torch.tensor([0, 1])  # 真实标签

# 使用 PyTorch 的 nn.CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss()
loss_pytorch = criterion(logits, targets)

# 使用手动实现的 CrossEntropyLoss
loss_manual = manual_cross_entropy_loss(logits, targets)

# 打印结果
print(f"PyTorch CrossEntropyLoss: {loss_pytorch.item()}")
print(f"Manual CrossEntropyLoss: {loss_manual.item()}")
