"""4.6 GRPO（评测来自仓库的 `grpo` 题）

约定（判题会提供自己的模型）:
  policy(input_ids, prompt=None, hidden=None) -> (logits (B, S, vocab), hidden)
  第一次调用时传入 prompt（形状 (B,)，每行一个提示值），之后可以传回 hidden 继续生成，也可以每步把整段前缀重新喂进去。
  BOS 的 id 是 vocab_size。
"""
import torch
import torch.nn as nn


def generate_group_completions(policy, prompts, group_size: int, seq_len: int, vocab_size: int):
    """每个提示生成 group_size 个回答，同一提示的回答在结果里连续排列（第 i 个提示占第 i*G 到 (i+1)*G−1 行）。
    采样而不是贪心。返回 (sequences (N, seq_len), log_probs (N, seq_len), prompt_targets (N,))，N = 提示数 × group_size。"""
    ...


def compute_group_advantages(rewards, group_size: int):
    """rewards: (N,)。在每组内部标准化：A = (r − 组均值) / (组标准差 + eps)。返回 (N,)。"""
    ...


def grpo_loss(policy_logps, old_logps, advantages, clip_epsilon: float, beta: float, ref_logps) -> dict:
    """policy_logps、old_logps、ref_logps: (B, S) 的逐 token 对数概率；advantages: (B,)，广播到该序列的每个 token。
      ratio = exp(policy − old)
      policy_loss = −mean( min(ratio·A, clip(ratio, 1−ε, 1+ε)·A) )
      kl_div = 当前策略相对参考策略的 KL 估计（相同为 0，差距越大越大）
      total_loss = policy_loss + beta·kl_div
    返回 dict：policy_loss、kl_div、total_loss（标量）。"""
    ...
