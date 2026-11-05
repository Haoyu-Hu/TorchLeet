"""4.1 KL 散度：精确计算与 RLHF 里的逐样本估计"""
import torch


def kl_divergence(p_logits, q_logits):
    """精确的 KL(P‖Q)，P = softmax(p_logits)，Q = softmax(q_logits)，沿最后一维求和。
    输入形状 (..., V)，输出形状 (...)。logits 很大（如 ±1000）时也不能出现 inf/nan。"""
    ...


def kl_estimators(logp, logp_ref):
    """logp、logp_ref：同一批样本 x（从策略 π 采样）在 π 与 π_ref 下的对数概率，形状相同。
    返回 (k1, k3)，都是逐样本的张量：
      k1 = log π(x) − log π_ref(x)
      k3 = r − log r − 1，其中 r = π_ref(x) / π(x)"""
    ...
