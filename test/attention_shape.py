import torch


def test_attention_shapes():
    # 假设 Batch_size=2, 句子长度=5, 单词向量维度=4
    # Query: (2, 5, 4)
    Q = torch.randn(2, 5, 4)
    # Key: (2, 5, 4)
    K = torch.randn(2, 5, 4)

    # 1. 转置 Key，为了做矩阵乘法
    # 我们希望得到 (2, 5, 5) 的分数矩阵（每个词对其他词的关注度）
    # K.transpose(1, 2) 变成了 (2, 4, 5)
    K_T = K.transpose(1, 2)
    print(f"K 转置后的形状: {K_T.shape}")

    # 2. 批量矩阵乘法 (Batch Matrix Multiplication)
    # 也是 Attention 公式的核心: Q * K^T
    scores = torch.bmm(Q, K_T)
    print(f"Attention Score 形状 (期望是 2,5,5): {scores.shape}")

    # 3. 理解 Mask (掩码) - LLM 里的关键
    # 在生成模型中，词不能看见“未来”的词，所以要遮住右上角
    mask = torch.tril(torch.ones(5, 5))  # 下三角矩阵
    print("\nMask 矩阵样子 (1能看, 0不能看):")
    print(mask)


if __name__ == "__main__":
    test_attention_shapes()