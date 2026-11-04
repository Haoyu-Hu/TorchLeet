"""2.3 KV Cache"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """缓存形状 (batch, heads, seq_len, d_head)，每次 update 沿 seq_len 维 (dim=2) 追加。"""

    def __init__(self):
        ...

    def update(self, new_k, new_v):
        """追加新的 K/V，返回 (完整的 K, 完整的 V)。"""
        ...

    def get(self):
        """返回当前缓存的 (k, v)。"""
        ...

    def reset(self):
        """清空缓存。"""
        ...


class CachedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # TODO: Q/K/V/输出 四个 nn.Linear(d_model, d_model, bias=False)
        ...

    def forward(self, x, kv_cache=None):
        """kv_cache 为 None：x 是 (B, T, d_model)，对整段序列做带因果 mask 的注意力。
        给了 kv_cache：x 是 (B, 1, d_model)，只算新 token 的 Q/K/V，
        把 K/V 追加进缓存后对全部缓存做注意力（缓存里只有过去，不需要 mask）。
        两种模式的结果要一致。返回与 x 同形状的张量。"""
        ...
