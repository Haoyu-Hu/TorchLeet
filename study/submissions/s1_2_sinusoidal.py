"""1.2 Sinusoidal 位置编码"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        # TODO: 构造 pe, 形状 (max_seq_len, d_model)
        #   偶数维 2i   = sin(pos * w_i)
        #   奇数维 2i+1 = cos(pos * w_i)
        #   w_i = 1 / 10000 ** (2i / d_model)
        # 用 self.register_buffer("pe", pe) 保存（不是 Parameter）
        ...

    def forward(self, x):
        # x: (batch, seq_len, d_model)，只用 seq_len
        # 返回 (1, seq_len, d_model)
        ...
