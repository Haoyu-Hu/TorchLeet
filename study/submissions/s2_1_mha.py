"""2.1 Multi-Head Attention（评测接口是函数：投影层在函数里创建，所以权重每次调用随机；
学懂之后建议再把它改写成 nn.Module）"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_head_attention(q, k, v, num_heads, d_model, mask=None):
    """q, k, v: (batch, seq, d_model)。mask 为布尔型，True=保留、False=屏蔽，
    可广播到 (batch, heads, seq, seq)。返回 (batch, seq, d_model)。
    步骤：Q/K/V 线性投影 → 拆成 num_heads 个头 → 每个头做缩放点积注意力 → 拼回 → 输出投影。"""
    ...
