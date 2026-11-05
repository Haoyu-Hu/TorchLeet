"""P.4 极大似然与最大后验（numpy / math）"""
import math
import numpy as np


def gaussian_mle(x):
    """返回 (μ̂, σ̂²)，都是极大似然估计。"""
    ...


def gaussian_nll(x, mu, var):
    """平均负对数似然：−(1/n) Σ log N(x_i; μ, var)。"""
    ...


def bernoulli_map(k, n, a, b):
    """n 次试验中 k 次成功，先验 Beta(a, b)，返回成功概率的最大后验估计。"""
    ...


def categorical_mle(counts, alpha=0.0):
    """各类别的计数 → 概率估计，加 α 平滑（α=0 即频率）。"""
    ...
