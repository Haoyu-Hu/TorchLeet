"""4.4 DPO（评测来自仓库的 `dpo` 题）"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_batch_logps(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """model(input_ids) 返回 (B, S, V) 的 logits。返回每条序列的总对数概率，形状 (B,)：
    对位置 t 的 logits 取 log_softmax，取出真实的下一个 token（input_ids[:, t+1]）的值，再沿序列求和。
    （本题对整条序列求和；真实训练里通常只对“回答”部分求和，见讲义。）"""
    ...


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta: float = 0.1):
    """L = −mean( log σ( β·[(π_c − ref_c) − (π_r − ref_r)] ) )，四个输入都是 (B,) 的序列对数概率。返回标量。"""
    ...
