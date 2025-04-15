# time: 2025/4/15 14:45
# author: YanJP

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义一个简单的随机数据集
class RandomDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机的源序列和目标序列
        src = torch.randint(0, self.vocab_size, (self.seq_length,))
        tgt = torch.randint(0, self.vocab_size, (self.seq_length,))
        return src, tgt

# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        src = self.embedding(src) * (self.d_model ** 0.5)
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)
        output = self.transformer(src, tgt)
        return self.fc_out(output)

# Scheduled Sampling 调度函数
def scheduled_sampling_prob(epoch, max_epochs, method='linear'):
    if method == 'linear':
        return 1.0 - (epoch / max_epochs)  # 线性衰减
    elif method == 'exponential':
        return 0.99 ** epoch  # 指数衰减
    else:
        raise ValueError("Unknown scheduling method")

# 训练函数
def train(model, dataloader, criterion, optimizer, device, max_epochs):
    model.train()
    for epoch in range(max_epochs):
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            # 获取目标序列的长度
            tgt_len = tgt.size(1)

            # 初始化解码器输入（通常是 <SOS> 标记）
            decoder_input = tgt[:, 0].unsqueeze(1)  # 取第一个词作为初始输入

            # 逐步生成目标序列
            for i in range(1, tgt_len):
                # 前向传播
                output = model(src, tgt)

                # 计算损失
                loss = criterion(output[:, -1, :], tgt[:, i])

                # 根据 Scheduled Sampling 决定是否使用真实词或模型生成的词
                prob = scheduled_sampling_prob(epoch, max_epochs, method='linear')
                use_ground_truth = torch.rand(1).item() < prob

                if use_ground_truth:
                    # 使用真实词作为下一个输入
                    next_input = tgt[:, i].unsqueeze(1)
                else:
                    # 使用模型预测的词作为下一个输入
                    _, predicted = torch.max(output[:, -1, :], dim=1)
                    next_input = predicted.unsqueeze(1)

                # 将下一个输入拼接到解码器输入中
                decoder_input = torch.cat([decoder_input, next_input], dim=1)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.4f}")

# 主函数
if __name__ == "__main__":
    # 参数设置
    vocab_size = 1000  # 词汇表大小
    d_model = 512      # 模型维度
    nhead = 8          # 多头注意力头数
    num_encoder_layers = 3  # 编码器层数
    num_decoder_layers = 3  # 解码器层数
    max_epochs = 10    # 最大训练轮数
    batch_size = 32    # 批量大小
    seq_length = 20    # 序列长度
    num_samples = 1000 # 数据集样本数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数和优化器
    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建随机数据集和 DataLoader
    dataset = RandomDataset(num_samples, seq_length, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 开始训练
    train(model, dataloader, criterion, optimizer, device, max_epochs)




