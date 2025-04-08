# time: 2025/4/7 9:59
# author: YanJP
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
def cosAnnealing():
    # 定义模型（这里只是示例，使用一个简单的线性模型）
    model = torch.nn.Linear(10, 1)
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # 定义余弦退火衰减学习率调度器
    T_max = 100  # 一个周期内的总训练步数
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    # 模拟训练过程，记录每个步骤的学习率
    learning_rates = []
    total_epochs = 300  # 总训练步数，这里设置为 3 个周期
    for epoch in range(total_epochs):
        # 训练步骤
        optimizer.step()
        # 更新学习率
        scheduler.step()
        # 记录当前学习率
        learning_rates.append(optimizer.param_groups[0]["lr"])

    # 可视化学习率的变化
    plt.plot(range(total_epochs), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Learning Rate Decay')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    cosAnnealing()