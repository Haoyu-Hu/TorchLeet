"""P.8 采样方法：逆变换、拒绝采样、重要性采样（numpy）"""
import numpy as np


def sample_exponential(lam, n, rng):
    """用逆变换法从 Exp(λ) 抽 n 个样本。rng 是 np.random.Generator，只能用 rng.random() 产生均匀数。"""
    ...


def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M, n, rng):
    """target_pdf(x) ≤ M·proposal_pdf(x) 处处成立。proposal_sample(m, rng) 返回 m 个提议样本。
    返回 (恰好 n 个被接受的样本, 接受率)。"""
    ...


def importance_estimate(f, x, p_pdf, q_pdf, self_normalized=False):
    """x 是来自 q 的样本。返回 E_p[f(X)] 的重要性采样估计；self_normalized=True 时用自归一化估计。"""
    ...
