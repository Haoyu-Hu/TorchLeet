"""1.4 RMSNorm"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # TODO: 可学习缩放 gamma，形状 (dim,)，初始为 1
        ...

    def forward(self, x):
        # TODO: x / sqrt(mean(x^2, 最后一维, keepdim) + eps) * gamma
        ...
