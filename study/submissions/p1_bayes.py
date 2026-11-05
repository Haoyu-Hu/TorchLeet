"""P.1 贝叶斯定理与朴素贝叶斯（numpy）"""
import numpy as np


def posterior(prior, likelihood):
    """prior、likelihood：长度为 K 的数组（各假设的先验，以及观测在各假设下的概率）。返回归一化的后验。"""
    ...


def sequential_update(prior, likelihoods):
    """likelihoods：(T, K)，T 次独立观测。返回看到全部观测后的后验。T 很大时连乘会下溢，要在对数空间计算。"""
    ...


def bernoulli_nb_fit(X, y, alpha=1.0):
    """X：(n, D) 的 0/1 矩阵；y：0..C−1 的整数标签。返回 (log_prior (C,), log_theta (C, D))，
    θ[c, j] = P(x_j = 1 | 类 c)，用加 α 平滑：(计数 + α) / (n_c + 2α)。"""
    ...


def bernoulli_nb_predict(X, log_prior, log_theta):
    """返回每行的预测类别 (n,)。"""
    ...
