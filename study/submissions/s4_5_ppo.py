"""4.5 PPO-RLHF（评测来自仓库的 `ppo-rlhf` 题）

约定（判题会提供自己的模型）:
  policy(input_ids)      -> logits (B, S, vocab)
  value_model(input_ids) -> values (B, S)
  BOS 的 id 是 vocab_size（模型的嵌入表有 vocab_size + 1 行）。
"""
import torch
import torch.nn as nn


def generate_sequences(policy: nn.Module, batch_size: int, seq_len: int, vocab_size: int):
    """从 BOS 开始逐 token 采样（Categorical(logits=最后位置的 logits).sample()，不能用 argmax）。
    返回 (sequences, log_probs, all_input_ids)：
      sequences     (B, seq_len)     生成的 token，取值 [0, vocab)
      log_probs     (B, seq_len)     每个被采样 token 的对数概率
      all_input_ids (B, seq_len+1)   BOS + 生成的 token（价值模型与 ppo_step 在这些前缀上打分）"""
    ...


def compute_gae(rewards, values, gamma: float = 0.99, lam: float = 0.95):
    """rewards、values: (B, S)。从后往前：
      δ_t = r_t + γ·V_{t+1} − V_t（V_S = 0）
      A_t = δ_t + γλ·A_{t+1}
    返回 advantages (B, S)。"""
    ...


def ppo_step(policy, value_model, ref_policy, old_log_probs, input_ids, actions, advantages, returns,
             clip_epsilon: float = 0.2, kl_beta: float = 0.01) -> dict:
    """在“产生第 t 个动作时看到的前缀”上重新计算当前策略与参考策略对 actions 的对数概率。
      ratio = exp(new − old)
      policy_objective = mean( min(ratio·A, clip(ratio, 1−ε, 1+ε)·A) )
      value_loss       = mean( (V − returns)² )
      kl_div           = 当前策略相对参考策略的 KL 估计（两者相同时为 0）
      total_loss       = −policy_objective + 0.5·value_loss + kl_beta·kl_div
    返回 dict，键为 policy_loss、value_loss、kl_div、total_loss（都是标量）。"""
    ...
