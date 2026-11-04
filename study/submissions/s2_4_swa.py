"""2.4 Sliding Window Attention"""
import math
import torch
import torch.nn.functional as F


def create_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """布尔矩阵 (seq_len, seq_len)，|i - j| <= window_size 处为 True。"""
    ...


def full_attention(q, k, v):
    """普通缩放点积注意力（无 mask）。q, k, v: (batch, seq, d)。"""
    ...


def sliding_window_attention(q, k, v, window_size):
    """带滑动窗口 mask 的注意力：被屏蔽的位置要在 softmax 之前设为 -inf。
    window_size >= seq_len 时结果必须等于 full_attention。"""
    ...
