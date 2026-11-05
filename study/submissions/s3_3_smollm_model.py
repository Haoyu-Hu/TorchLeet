"""3.3 组装完整的 SmolLM（Llama 风格解码器）并做贪心生成。

先把你已通过的两块代码粘贴到下面：
  - 1.4 的 RMSNorm（构造 RMSNorm(dim, eps)）
  - 3.2 的 rotate_half / apply_rotary_pos_emb / repeat_kv / RotaryEmbedder / RopeAttention
然后完成 SwiGLU、解码层、SmolLM 和 greedy_generate。所有 Linear 都不带偏置。

cfg 里有: vocab_size, hidden_size, intermediate_size, num_layers, num_heads,
          num_key_value_heads, head_dim, rope_theta, rms_norm_eps, tie_word_embeddings
SmolLM 需要有这些属性（沿用 Hugging Face Llama 的命名）:
    embed_tokens (nn.Embedding) · layers (nn.ModuleList) · norm (最终 RMSNorm) · lm_head (nn.Linear)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 粘贴：RMSNorm ----


# ---- 粘贴：rotate_half / apply_rotary_pos_emb / repeat_kv / RotaryEmbedder / RopeAttention ----


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # TODO: gate / up / down 三个无偏置 Linear
        ...

    def forward(self, x):
        # TODO: down( silu(gate(x)) * up(x) )
        ...


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO: 两个 RMSNorm、一个 RopeAttention(cfg)、一个 SwiGLU
        ...

    def forward(self, x, attention_mask):
        # TODO: x = x + attn(norm1(x), mask)；x = x + mlp(norm2(x))
        ...


class SmolLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO: embed_tokens、layers(num_layers 个 DecoderLayer)、norm、lm_head
        #       cfg.tie_word_embeddings 为 True 时让 lm_head.weight 与 embed_tokens.weight 共享
        ...

    def forward(self, input_ids, labels=None):
        """返回 (logits, loss)。logits: (B, S, vocab)。没传 labels 时 loss 为 None。
        loss 是 next-token 交叉熵：logits[:, :-1] 对 labels[:, 1:]。
        要先造好加性因果 mask：(1,1,S,S)，上三角（不含对角）为 -inf。"""
        ...


@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens):
    """每步取最后一个位置 logits 的 argmax 追加到序列；返回 (B, S + max_new_tokens)。"""
    ...
