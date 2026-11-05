"""4.3 SFT：构造样本、批量补齐、只在回答上算损失"""
import torch
import torch.nn.functional as F


def build_sft_example(prompt_ids, response_ids, eos_id, ignore_index=-100):
    """返回 (input_ids, labels)，两个等长的一维 LongTensor：
       input_ids = prompt + response + [eos]
       labels    = 与 input_ids 相同，但提示部分换成 ignore_index（回答和 eos 保留）"""
    ...


def collate(batch, pad_id, ignore_index=-100):
    """batch: [(input_ids, labels), ...]。右侧补齐到批内最长。返回 dict：
       input_ids（用 pad_id 补）、labels（用 ignore_index 补）、attention_mask（真实为 1、补齐为 0），都是 (B, L)。"""
    ...


def sft_loss(logits, labels, ignore_index=-100):
    """logits: (B, L, V)，labels: (B, L)。next-token 交叉熵：logits[:, :-1] 对 labels[:, 1:]，
    对批内所有有效 token 求平均（不是先按序列平均）。没有有效 token 时返回 0（仍可反向传播）。"""
    ...
