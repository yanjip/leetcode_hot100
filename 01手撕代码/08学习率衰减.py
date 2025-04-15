# time: 2025/4/7 9:59
# author: YanJP
# import torch
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# import matplotlib.pyplot as plt
# def cosAnnealing():
#     # 定义模型（这里只是示例，使用一个简单的线性模型）
#     model = torch.nn.Linear(10, 1)
#     # 定义优化器
#     optimizer = optim.SGD(model.parameters(), lr=0.1)
#     # 定义余弦退火衰减学习率调度器
#     T_max = 100  # 一个周期内的总训练步数
#     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
#
#     # 模拟训练过程，记录每个步骤的学习率
#     learning_rates = []
#     total_epochs = 300  # 总训练步数，这里设置为 3 个周期
#     for epoch in range(total_epochs):
#         # 训练步骤
#         optimizer.step()
#         # 更新学习率
#         scheduler.step()
#         # 记录当前学习率
#         learning_rates.append(optimizer.param_groups[0]["lr"])
#
#     # 可视化学习率的变化
#     plt.plot(range(total_epochs), learning_rates)
#     plt.xlabel('Epoch')
#     plt.ylabel('Learning Rate')
#     plt.title('Cosine Annealing Learning Rate Decay')
#     plt.grid(True)
#     plt.show()
#
# if __name__ == '__main__':
#     cosAnnealing()

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR, CosineAnnealingLR

# 创建一个简单的模型和优化器用于示例
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 假设训练100个epoch
num_epochs = 100


# 1. 指数衰减 (Exponential Decay)
def exponential_decay():
    # gamma: 衰减因子，每步学习率乘以gamma
    scheduler = ExponentialLR(optimizer, gamma=0.95)  # 每步衰减到原来的0.95倍

    for epoch in range(num_epochs):
        # 训练代码...
        optimizer.step()
        scheduler.step()  # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Exponential LR: {current_lr:.6f}")


# 2. 固定步长衰减 (Step Decay)
def step_decay():
    # step_size: 每隔多少个epoch衰减一次
    # gamma: 衰减因子
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 每30个epoch衰减到0.1倍

    for epoch in range(num_epochs):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Step LR: {current_lr:.6f}")


# 3. 多步长衰减 (Multi-Step Decay)
def multi_step_decay():
    # milestones: 在哪些epoch进行衰减
    # gamma: 衰减因子
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    for epoch in range(num_epochs):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Multi-Step LR: {current_lr:.6f}")


# 4. 余弦退火衰减 (Cosine Annealing)
def cosine_annealing():
    # T_max: 一个周期的epoch数
    scheduler = CosineAnnealingLR(optimizer, T_max=50)  # 50个epoch为一个周期

    for epoch in range(num_epochs):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Cosine Annealing LR: {current_lr:.6f}")


# 测试任一策略
if __name__ == "__main__":
    print("Exponential Decay:")
    exponential_decay()

    # 重置优化器学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1

    print("\nStep Decay:")
    step_decay()

    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1

    print("\nMulti-Step Decay:")
    multi_step_decay()

    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1

    print("\nCosine Annealing:")
    cosine_annealing()