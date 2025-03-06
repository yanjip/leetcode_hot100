# time: 2025/3/6 9:33
# author: YanJP

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input):
        # 输入形状: (batch_size, seq_len) → 转置为 (seq_len, batch_size)
        input = input.permute(1, 0)  # [新增代码]
        embedded = self.embedding(input)
        # embedded形状: (seq_len, batch_size, hidden_size)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        # 输入形状: (batch_size) → 转置为 (1, batch_size)
        input = input.unsqueeze(0)  # 等价于 input.reshape(1, -1)
        embedded = self.embedding(input)
        # embedded形状: (1, batch_size, hidden_size)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        tgt_len = tgt.shape[1]
        batch_size = tgt.shape[0]
        tgt_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size)

        hidden, cell = self.encoder(src)

        # 第一个输入是<sos>
        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs


# 定义简单的词汇表
src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6}
tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'Je': 3, 'suis': 4, 'un': 5, 'étudiant': 6}

# 反转词汇表
src_idx2word = {i: w for w, i in src_vocab.items()}
tgt_idx2word = {i: w for w, i in tgt_vocab.items()}

# 简单的输入数据
src_sentence = ['I', 'am', 'a', 'student']
tgt_sentence = ['Je', 'suis', 'un', 'étudiant']

# 将句子转换为索引序列
src_seq = [src_vocab[word] for word in src_sentence]
tgt_seq = [tgt_vocab[word] for word in tgt_sentence]

# 添加开始和结束标记
src_seq = [src_vocab['<sos>']] + src_seq + [src_vocab['<eos>']]
tgt_seq = [tgt_vocab['<sos>']] + tgt_seq + [tgt_vocab['<eos>']]

# 转换为Tensor
src_tensor = torch.LongTensor(src_seq).unsqueeze(0)  # (1, seq_len)
tgt_tensor = torch.LongTensor(tgt_seq).unsqueeze(0)  # (1, seq_len)


# 超参数
input_size = len(src_vocab)
output_size = len(tgt_vocab)
hidden_size = 16
learning_rate = 0.01
num_epochs = 100

# 初始化模型
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(output_size, hidden_size)
model = Seq2Seq(encoder, decoder)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(src_tensor, tgt_tensor)
    output_dim = output.shape[-1]

    output = output[1:].reshape(-1, output_dim)
    tgt = tgt_tensor[:, 1:].reshape(-1)

    loss = criterion(output, tgt)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 测试翻译
model.eval()
with torch.no_grad():
    output = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0)
    output = output.argmax(2).squeeze(0)
    translated_sentence = [tgt_idx2word[idx.item()] for idx in output]
    print("Translated Sentence:", translated_sentence)
