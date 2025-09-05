import numpy as np

def sample_tokens(probs, top_k=0, top_p=1.0, temperature=1.0):
    """
    同时支持 Top-k / Top-p / Temperature 采样
    输入:
        probs: numpy.ndarray, shape [seq_len, vocab_size], 每行是概率分布(已归一化)
        top_k: 保留概率最高的 k 个词 (0 表示不启用)
        top_p: 累计概率阈值 (1.0 表示不启用)
        temperature: 温度系数 (>0), 调节分布平滑度
            T<1：放大概率差异，更确定；
            T>1：拉平分布，更随机。
            T=1: 原本的分布不变
    输出:
        samples: list[int], 每个位置采样得到的词id
    """
    seq_len, vocab_size = probs.shape
    samples = []

    for t in range(seq_len):
        p = probs[t]

        # --- Temperature ---
        if temperature != 1.0:
            logits = np.log(p + 1e-12) / temperature
            p = np.exp(logits)
            p /= p.sum()

        # --- Top-k ---
        if top_k > 0:
            top_indices = np.argpartition(p, -top_k)[-top_k:]
            mask = np.zeros_like(p, dtype=bool)
            mask[top_indices] = True
            p = np.where(mask, p, 0.0)
            p /= p.sum()

        # --- Top-p (Nucleus sampling) ---
        if top_p < 1.0:
            sorted_indices = np.argsort(-p)
            sorted_probs = p[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = cumulative_probs > top_p
            if np.any(cutoff):
                cutoff_index = np.argmax(cutoff)
                mask = np.zeros_like(p, dtype=bool)
                mask[sorted_indices[:cutoff_index+1]] = True
                p = np.where(mask, p, 0.0)
                p /= p.sum()

        # --- Sampling ---
        token = np.random.choice(vocab_size, p=p)
        samples.append(token)

    return samples

if __name__ == '__main__':
    # 模拟一个 seq_len=3, vocab_size=5 的分布
    # probs = np.array([
    #     [0.1, 0.2, 0.3, 0.25, 0.15],
    #     [0.05, 0.05, 0.4, 0.4, 0.1],
    #     [0.2, 0.3, 0.1, 0.25, 0.15]
    # ])
    seq_len, vocab_size = 5, 10
    # 随机生成 logits
    logits = np.random.randn(seq_len, vocab_size)
    # softmax 归一化成概率分布
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    # 调用采样函数
    samples = sample_tokens(probs, top_k=5, top_p=0.9, temperature=0.8)
    print("采样结果:", samples)
