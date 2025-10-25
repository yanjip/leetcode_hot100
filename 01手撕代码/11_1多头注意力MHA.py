# time: 2025/4/23 17:05
# author: YanJP
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # 输入维度
        self.num_heads = num_heads  # 头数
        self.head_dim = d_model // num_heads  # 每个头的维度

        # 线性变换矩阵（Q、K、V）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分头 (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数 (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # 可选：应用掩码（如因果掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和 (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, V)

        # 合并多头 (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # transpose 操作会导致张量在内存中非连续存储（contiguous=False），而 view 操作要求张量必须在内存中是连续的（否则会报错）。0

        # 输出线性变换
        return self.W_o(output)


# 测试代码
if __name__ == "__main__":
    d_model = 512  # 输入维度
    num_heads = 8  # 头数
    seq_len = 10  # 序列长度
    batch_size = 4  # batch大小

    # 随机生成输入（模拟一个batch的输入）
    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)

    # 初始化多头注意力
    mha = MultiHeadAttention(d_model, num_heads)

    # 前向传播
    output = mha(query, key, value)

    print("输入形状:", query.shape)
    print("输出形状:", output.shape)  # 应与输入形状一致 (batch_size, seq_len, d_model)
