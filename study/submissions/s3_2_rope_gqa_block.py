"""3.2 RoPE + GQA 注意力块（SmolLM / Llama 的核心积木）。评测来自仓库的 `smollm` 题。

约定:
- 旋转用“前后半段配对”：rotate_half([x1, x2]) = [-x2, x1]（最后一维切两半）。
- RotaryEmbedder(dim, base).forward(x)：x 形状 (B, heads, S, dim)，返回 (cos, sin)，
  形状都是 (1, S, dim)，dtype 与 x 相同。位置 0 的角度为 0。
- RopeAttention(config)：config 有 hidden_size, num_heads, num_key_value_heads(=kv_heads),
  head_dim, rope_theta。forward(hidden_states, attention_mask=None) 返回 (B, S, hidden_size)。
  attention_mask 是“加性”的 (1,1,S,S)：要屏蔽的位置是 -inf，直接加到分数上，再 softmax。
"""
import math
import torch
from torch import nn


def rotate_half(x):
    """把最后一维切成两半 (x1, x2)，返回 (-x2, x1) 拼接。"""
    ...


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """cos/sin 形状 (B或1, S, dim)，先在 unsqueeze_dim 上补出 head 维再相乘。
    返回旋转后的 (q, k)：q*cos + rotate_half(q)*sin。"""
    ...


def repeat_kv(hidden_states, n_rep):
    """(B, kv_heads, S, d) → (B, kv_heads*n_rep, S, d)。第 i 个新头必须是第 i // n_rep 个 K/V 头。"""
    ...


class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        # TODO: inv_freq = 1 / base ** (arange(0, dim, 2) / dim)，注册为 buffer
        ...

    @torch.no_grad()
    def forward(self, x):
        # TODO: 位置 0..S-1 与 inv_freq 做外积得到角度，再把两份拼起来对应前后两半，取 cos 和 sin
        ...


class RopeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: q/k/v/o 四个 Linear（K、V 只投影到 num_key_value_heads * head_dim）
        self.rotary_emb = RotaryEmbedder(base=config.rope_theta, dim=config.head_dim)
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        # 投影 → 拆头 → 旋转 q、k → repeat_kv → 缩放点积注意力(+加性 mask) → 合头 → 输出投影
        ...
