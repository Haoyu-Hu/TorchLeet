"""1.3 RoPE 的两个核心函数"""
import torch


def rotate_half(x):
    """把最后一维切成两半 (x1, x2)，返回 (-x2, x1) 拼接。"""
    ...


def apply_rotary_pos_emb(q, k, cos, sin):
    """返回旋转后的 (q, k)。q' = q*cos + rotate_half(q)*sin ；k 同理。
    cos/sin 要能广播到 (batch, heads, seq, dim)。"""
    ...
