"""2.2 Grouped Query Attention"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def grouped_query_attention(q, k, v, num_query_heads, num_query_groups, d_model, mask=None):
    """q, k, v: (batch, seq, d_model)。Q 保留 num_query_heads 个头，K/V 只有 num_query_groups 个头，
    每个 K/V 头被 num_query_heads // num_query_groups 个 Q 头共享。
    num_query_groups 不能整除 num_query_heads 时要报错。mask 同 2.1（布尔，True=保留）。
    返回 (batch, seq, d_model)。"""
    ...
