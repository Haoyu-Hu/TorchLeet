"""1.5 Scaled Dot-Product Attention（不许调用 F.scaled_dot_product_attention）"""
import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (..., seq, d)。mask 可选，布尔型，True=保留、False=屏蔽（下三角 tril 即因果注意力）；被屏蔽位置不能影响输出。
    返回 (output, attention_weights)。"""
    ...
